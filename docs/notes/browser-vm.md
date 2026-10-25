# The browser VM

The VM compiled to WebAssembly (`dream/browser/`), which runs a std.graphics
frontend -- or any Dream program -- in a Web Worker, and the desktop host
(`dream/src/graphics.cpp`) that shows such a page in a GTK window. What it is,
for a person using it, is `mind/std/docs/graphics.md`. This is what was
learned building it, as of 2026-10-23.

## One thread, pumped

The browser VM is the native scheduler with no workers: `Scheduler::pump`
runs a slice at a time on the embedding's thread, and `host.cpp` exports
`load`, `tick`, `delay`, `event` and `dispose` for the worker to drive it.
`tick` answers 1 (runnable), 2 (waiting: call again after `delay`
milliseconds, or when an event comes), 0 (finished) or -1 (failed), and the
worker runs slices for about 4 ms before giving the thread back, so the page
can post an event in.

The program is finished when the root is. A process parked on the bridge (the
event reader std.graphics starts), or a ticker the program started, does not
keep a page's program alive after `main!` returned -- which is the command
line's rule, said of the root.

### Failures are reported at the end, as natively

The first version took `Scheduler::take_failures` after every slice and failed
the run on any. That is not the command line's rule: a child's failure is
*uncaught* only if nothing joins it before the run ends, and `proc.try_map!`
joins its failures after they happen. Polling per slice failed std.proc's
tests that rely on it. Now the root's failure ends the run at once, and the
others are asked for once the run is over, as `cli.cpp` does.

### What cannot be reached

A slice ends when its reductions run out -- except inside a nested force,
which re-arms its budget and carries on (interp.cpp, "The slice ran out
somewhere this loop cannot stop"). Natively that is fine: another worker runs
the timer that times the force out, and the kill it sends is raised inside the
force. With one thread, nothing else runs until the force returns, so
`time.within! 20ms (len (list.from 1))` grew the process heap to its limit
instead of timing out. This is the one behaviour known to differ, and it is
written down in the user's document.

Yielding the force was tried: in the browser build only, a force whose slice
ran out set `park_requested` and `wake_pending`, which the parking handshake
turns into an immediate requeue, and the force is suspended as it is for a
blocking native inside it. It was wrong, not merely slow:
`len (list.from 1)` *answered* 1998. `len`'s native, re-entered, does not
resume the suspended force but starts again over what was already forced.
Reverted. Whether a park inside `len`'s walk has the same problem natively is
an open question this did not settle.

## Dream's library, in a page

`std.io` and `std.os` are reached by almost everything -- `console`, `time`,
`log`, `test` -- so a VM without them runs nothing ordinary. `services.cpp`
gives both modules every member, by the native VM's names and arities: the
standard streams (to Emscripten's `print`/`printErr`, line by line), the
clocks, no arguments and no environment, `platform` `:browser`; and
`:unsupported`, naming the member, from everything that touches files or
processes, raised when called. `std.crypto` is the native module unchanged,
with `getentropy` (at most 256 bytes a call) where Linux has `getrandom`.
`std.net`, `std.tls` and `std.ffi` are not registered, and the VM's own
"module ... is not provided by this runtime" is said at the first call.

The first version refused, at load, any image importing a module other than
`std.math`, `std.vm`, `std.tensor` and the bridge. An image imports what its
modules import, used or not, so that refused every program that printed.
Resolution is lazy (interp.cpp resolves a native import at its first call),
so nothing needed refusing.

With that, every example in `examples/` that does not open a socket prints
exactly what the native VM prints (`compat.py`), and std's test suite passes
but for files, sockets and two tests that race a 20 ms timer.

Memory: `-sMAXIMUM_MEMORY` was 256 MiB, and std.proc's kill test, which grows
an endless list for 20 ms, threw `std::bad_alloc` out of the Wasm call. It is
2 GiB now (memory grows on demand), and `tick` catches what C++ throws, so it
reaches the page as a failure in words rather than as an opaque pointer.

## The bridge is a native module

`std.browser` (`send!`, `next!`) is in the compiler's list of native modules,
not a `--host-module`. As a host module, std.graphics.browser compiled only
with the flag, and anything that swept the whole library -- example 07,
`import std;` -- failed to compile. Only the browser VM provides it; elsewhere
it is a missing native like any other, at its first call.

## WebKitGTK, 2.54

Five things stopped the desktop host's page from running, each silently:

- **A JSCContext in the UI process.** The host made one to parse and quote
  JSON. Made before the web view, it stopped the view's first load from ever
  starting: no navigation policy decision, no request, an idle main loop. The
  host now runs no JavaScript: the page posts *objects* to the script message
  handler, which arrive as WebKit's own JSCValues, and replies are JSON quoted
  by `quote`.
- **`set_http_headers` takes the headers** (transfer full), and the host also
  unreferenced them: a use-after-free in `soup_message_headers_iter_next`
  once a response was read.
- **Those headers replace the response's**, Content-Type included, and under
  `X-Content-Type-Options: nosniff` a module that has lost its type is not
  run. The type goes in the headers too.
- **A `charset` on `text/javascript`** -- `text/javascript; charset=utf-8` --
  and the custom-scheme module load never finishes. Scripts and stylesheets go
  without one.
- **A scheme registered as local** is file://-like, and under it WebKitGTK
  refuses a module's imports without `Access-Control-Allow-Origin`: main.mjs
  loaded, dom.mjs was never asked for. The scheme is secure and CORS-enabled
  only, which is also less than the page needs.

How these were found, since the symptom of each is "nothing happens": a
standalone C++ program making a view the way the host does, with each piece
switchable by an environment variable, under the same Xvfb; then the real
assets, served by it with the host's headers one at a time. MiniBrowser
(`libexec/webkit2gtk-4.1/MiniBrowser`) loading a `data:` URL is the check that
WebKit itself works there. `ptrace` attach is not permitted on this machine;
running the host under `gdb -batch -ex run` and interrupting it is.

## Order

The worker collapses the renders of one slice to the last, to send the page
less. A message after a render must not overtake it: `focus!` on a field the
update creates was posted before the render that made the field, and focused
nothing. The worker now flushes a held render before any other message, the
page flushes a render still waiting for its animation frame before a service
that acts on an element, and std.graphics holds such effects
(`after_render!`) until the render is sent.

## The window and the program

The driver closing is not the program ending: a loop that fails closes its
driver on the way out (`defer!`), and the failure comes after. When the
driver's close told the host to close the window, a failing frontend exited 0.
The page now tells the host only when the worker reports the run finished,
with its exit code. On close the host writes out what is queued for the
backend, closes its input so `serve!` returns, and gives it two seconds --
the `closing!` that saves state on the way out depends on it.

## Renders are patches

Each render used to send the whole tree. Measured on a 2,000-row table
(14,000 nodes), natively: 1.9 MB of JSON and 380 ms of Dream a render --
building `json.Value` maps for every node, then encoding them -- before the
page had parsed a byte, and the page refused it anyway, past its limits of
10,000 nodes and 1 MB. Every keystroke paid it.

Now the loop keeps the tree it rendered and `std.graphics.diff` compares the
next with it. The cheap part is that `==` on two subtrees is the VM's walk,
not Dream's: an equal row is one native comparison and becomes part of a run,
"old children 6 to 1999, kept". Closures compare by identity, so a lambda
handler written in the view makes its node unequal every render; the diff
then looks at the node's fields and children itself and finds nothing, which
costs that node a Dream walk and no bytes. Handlers that are messages or
constructors (`on_click (Msg.pick i)`, `on_input Msg.name`) compare equal.
The same keystroke is now 104 bytes and 16 ms of diff, plus the view's own
25 ms of building the tree.

The handler table went with it. It was built by a walk of the whole tree a
render, for an event that might come; made lazy it was still 29 ms on the
first event of each render. An id is now the way to a node from the root --
its key where it has one, its position where not -- and an event is routed by
following it, which reads the siblings along the way and nothing else. The
key steps are also what makes an id stable: a row that moves keeps its id,
and so does everything inside it, so a patch never has to re-id what it kept.

`ui.lazy f value` is the rest: a node not made while `f` and `value` equal
last render's. It holds the node it makes behind a closure, `fn u -> made`,
which is what keeps a parent's `==` from forcing it (the closure is unequal
by identity, so the parent's comparison stops there and the diff asks the
lazy node). With lazy rows the keystroke is 14 ms of view and diff, and on
the browser VM about 80 ms from key to screen, against 130 ms without.

Two consequences. Patches cannot be coalesced the way trees were -- the
worker kept only a slice's last render, the page only a frame's -- since each
is of what the one before made: both now queue them in order. And identity
is decided among siblings, so a keyed node moved to another parent is a new
element there; the page used to match keys across the whole tree.

## A stream of output

sleep's terminal was the first program to send its window a lot, fast:
`seq 1 200000` is 200,000 lines in half a second. Measured in the window
(Xvfb, 2026-10-24), the window had drawn 111,000 of them after 25 s, while
the backend had finished the whole flood in 0.8 s. Four things were in the
way, and only one of them was the emulator.

- **The frame was the history, not the screen.** Each frame carried every
  line changed within the 2,000 the window keeps, so a flood sent 2,000 lines
  a frame and the page drew each, though only the screen's 40 could be seen.
  A frame sent while output still arrives now carries the screen; the first
  after it stops carries everything kept ("While output streams, a frame is
  the screen" in sleep/src/session.dr). The tab draws from where its
  unbroken run of lines begins, or it drew 1,460 empty lines, numbered
  afresh each frame.
- **JSON was Dream.** `json.encode` walked `std.codec` and `json.parse` a
  list of characters; on the browser VM, with no JIT, decoding a frame cost
  more than drawing it. Both are the VM's now (`vm.json_encode`,
  `vm.json_decode` in dream/src/builtins.cpp), written to give the bytes and
  values the Dream walk gives -- `std.json`'s tests hold the two together --
  and the reader leaves anything it is not sure of (text that is not JSON,
  not UTF-8, or nests too deep) to the Dream parser, so errors read as
  before. 532 ms of decoding the coloured flood natively became 7.
- **A long keyed list re-diffed every frame.** The terminal widget's lines
  were siblings, 1,500 of them; a render compared and re-keyed them all.
  They are drawn in lazy blocks of 64 now, keyed by place, so a frame
  compares a couple of dozen.
- **`str.join_str` built a list of two per part** through `flat_map`, which
  every JSON array and rendered page paid; it conses now.

After: the flood is 0.54 s in the window, as fast as `seq` writes to a
terminal, and a coloured one (100,000 lines, four SGRs each) 1.6 s where it
was 10.
