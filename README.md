# dream

**The virtual machine for [Dream](https://github.com/dream-mindv2/dreams)**, a
lazily evaluated functional language. `dream` runs `.dream` images. It knows
nothing about source code: the compiler,
[`dreams`](https://github.com/dream-mindv2/dreams), is itself an image this VM
runs, and so is the build tool, [`mind`](https://github.com/dream-mindv2/mind).

- a **lazy graph-reduction interpreter**, written as an explicit state machine;
- **green processes** with isolated heaps and copied messages, in the style of
  the BEAM, scheduled preemptively across cores;
- a **generational collector** per process, parallel and partly concurrent;
- an **LLVM JIT** that compiles hot functions, numeric and list code alike;
- **tensors** with fused, vectorised CPU kernels and an OpenCL backend;
- the **native modules** -- `std.io`, `std.os`, `std.net`, `std.tls`,
  `std.ffi`, `std.math`, `std.crypto`, `std.tensor`, `std.vm` -- that the rest
  of the standard library is written on;
- a **C API** for embedding the VM in another program.

One image runs on Linux, macOS, Windows and Android, 64-bit little-endian.

## Building

You need a C++20 compiler, CMake 3.20, **libffi** and **OpenSSL 3** (SChannel
on Windows). Both are required: `std.ffi` and `std.tls` are part of the
interface every VM provides, and a VM without them is not a valid one. LLVM
is optional. Without it the VM is the interpreter alone, still correct, only
slower.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/dream program.dream [args]
```

To get the compiler, the build tool and the standard library too, build the
whole toolchain from the [Dream workspace](https://github.com/Custards1/dream),
where this repository is checked out at `dream/`:

```sh
git clone --recursive https://github.com/Custards1/dream
cd dream && ./build.sh && just install
```

The workspace's recipes build this VM in `build-dream/`:

```
just vm                      # build-dream/bin/dream, with the JIT if LLVM is found
just vm-no-jit               # build-nojit/bin/dream, the interpreter alone
just vm-pgo                  # build-pgo/bin/dream, trained on a self-compile (~12% faster)
just mind-build vm           # the same VM, built by std.build rather than CMake
```

## Security

Dream executes programs with host permissions; it is not a sandbox for hostile
code. See [security boundaries and hardening](docs/security.md) before
compiling or running untrusted projects.

## Documents

| | |
|---|---|
| [docs/builtins.md](docs/builtins.md) | every builtin and every native module's members, with their types |
| [docs/bytecode-format.md](docs/bytecode-format.md) | the `.dream` image as it actually is |
| [docs/gc.md](docs/gc.md) | the collector, and the rules a native must follow. Read it before touching `heap.cpp` |
| [docs/ffi.md](docs/ffi.md), [docs/images.md](docs/images.md) | calling C, and calling another image |
| [docs/platforms.md](docs/platforms.md), [docs/security.md](docs/security.md) | what an image may assume about where it runs, and what the VM does not protect against |
| [docs/dynamic-linking.md](docs/dynamic-linking.md) | the plan for one runtime holding many images |
| [docs/notes/](docs/notes) | the performance log: the interpreter, collector, scheduler and JIT ([vm-performance.md](docs/notes/vm-performance.md)), tensors, bignums, large data |

### The command line

```
dream <image> [options] [program args]
```

An image is found as the name written, that name with `.dream` added, or
either under `$MINDV2_PATH` -- so `dream mind` runs `./mind.dream` if there is
one and the installed one otherwise. `dream -x NAME` looks only in the
installation.

VM options go **before** the image; everything after it belongs to the program.

| | |
|---|---|
| `-x, --exec NAME` | run `NAME.dream` from `$MINDV2_PATH`, ignoring the working directory |
| `-e, --entry NAME` | run this global instead of `main!` |
| `-j, --workers N` | scheduler threads (default: the cores, at most 8) |
| `--stats` | reductions, collections, bytes allocated and promoted, the live set by kind, what the JIT took |
| `--profile [N]` | the N hottest functions, by reductions |
| `--dump` | disassemble the image and exit |
| `--no-jit` | stay in the interpreter |
| `--jit-threshold N` | calls before a function is compiled (default 32) |
| `--jit-policy NAME` | choose `density` (default), `calls`, `calls_times_size`, `work`, `calls_per_size_squared`, or `adaptive` |
| `--dump-jit FN` | print the LLVM IR generated for one function |
| `--any-target` | run an image built for another platform anyway |
| `--mindv2-path` | print the effective `$MINDV2_PATH` and exit |

### Environment

| | |
|---|---|
| `MINDV2_PATH` | where installed images are found |
| `DREAM_MAX_HEAP`, `DREAM_MAX_DEPTH`, `DREAM_MAX_STACK` | per-process limits: heap bytes (1 GB), pending continuations, value-stack depth |
| `DREAM_NURSERY_MAX` | the cap a process's nursery may grow to (32 MiB) |
| `DREAM_GC_THREADS`, `DREAM_GC_PAR_MIN` | the collector's helper threads, and the size of heap worth dividing among them |
| `DREAM_GC_CONCURRENT` | `1` overlaps every major's marking with the program, `0` none; unset lets size decide |
| `DREAM_GC_EVACUATE` | `all` empties every old block at every major (a test setting) |
| `DREAM_GC_TRACE` | print every collection |
| `DREAM_VERIFY_HEAP` | walk and check the heap after every collection |
| `DREAM_STUCK_SECONDS` | report what every process is doing if nothing has progressed for this long |
| `DREAM_JIT_SYNC` | compile on the calling thread rather than in the background |
| `DREAM_JIT_RANK` | default selection policy; `--jit-policy` overrides it; `calls_per_size` and `work_per_size` are aliases for `density` and `work` |
| `DREAM_JIT_TRACE` | print every compile, whether it was taken, and what it cost |
| `DREAM_JIT_DENY` | keep these functions out of the JIT: names, or `#` and an index from the trace, comma-separated -- bisect a fault the interpreter does not have |
| `DREAM_JIT_CHECK_EFFECTS` | abort, naming the function and the native, when a retried compiled call performs an effect an earlier attempt already performed -- a gap in the rule that compiled bodies are pure |
| `DREAM_DEBUG_INHERIT` | pass the debugging variables above (`DREAM_JIT_*`, `DREAM_VERIFY_HEAP`, `DREAM_GC_TRACE`, `DREAM_IO_TRACE`, `DREAM_STUCK_SECONDS`, `DREAM_PROBE_THUNK`) to children `exec!` starts; by default they are dropped, as they describe this VM only |
| `DREAM_TENSOR_THREADS`, `DREAM_OPENCL_LIB` | tensor parallelism, and which OpenCL to `dlopen` |

## Why the interpreter is a state machine

The obvious way to write an evaluator is a recursive `eval`. That would have
cost most of what follows.

A process's entire state -- where it is in the program, what it will do with
the result, and every value it can still reach -- lives in two vectors it owns:
a value stack and a continuation stack. Nothing sits in a C++ local across a
step. That one decision buys:

- **Preemption.** A process can be stopped between any two reductions and
  resumed later on another OS thread, because there is no native stack to
  save. No process can starve the others.
- **Precise GC roots.** The collector enumerates a list, not a stack. There is
  no conservative scanning and no pinning.
- **Proper tail calls.** A tail call pops its continuation before entering the
  callee, so a tail-recursive loop runs in constant space.
- **Cheap processes.** A process is a heap and two small vectors, so hundreds
  of thousands of them fit in memory.

## Values

One 64-bit word, tagged in the low bits:

| Pattern | Meaning |
|---|---|
| `....1` | fixnum -- a 63-bit signed integer, `(int64)v >> 1` |
| `...000` | pointer to a heap object (8-byte aligned); `0` means "no value" |
| `...010` | immediate -- unit, bool, char, atom, the empty list, a builtin |

Integers get the one-bit tag because arithmetic is the hot path: testing and
untagging cost a shift each, and the JIT can compare two tagged fixnums
directly, since the tagging preserves order. An integer that does not fit in
63 bits is a bignum, whose limbs live outside the heap, so fixnum code pays
nothing for integers being unbounded
([docs/notes/bignums.md](docs/notes/bignums.md)).

## Laziness

Every argument, list element, map value and `let` binding starts as a
**thunk**: a node of the program plus the frame to evaluate it in. Forcing a
thunk overwrites it in place with an indirection to its result, so every holder
of that pointer sees the value and the work happens once.

The compiler decides where evaluation order is observable and marks it in the
bytecode. The VM forces a node only when it is marked strict -- the statements
of an impure block, an `if` condition, a `try!` body, a strict parameter --
and otherwise leaves it suspended. A discarded pure statement is never
evaluated, so it cannot raise an error the program never asked for.

A node that is already a value -- a constant, a variable, a closure -- is
returned as itself rather than wrapped. Returning the *binding's* thunk rather
than a fresh one is also what preserves sharing.

## Processes

Processes share no memory. Each owns a heap that nothing else can reach, and a
message is deep-copied on the way out. That is the BEAM bargain, and it buys
the same things here:

- one process's collection never stops another and never takes a lock;
- updating a thunk needs no atomics, because only one process can force it;
- one process failing cannot corrupt another.

The cost is that a thunk shared between two processes is evaluated twice. For a
language with both concurrency and laziness, isolation is the better trade.

| | |
|---|---|
| `spawn! $( .. )` | run a suspended computation in a new process; answers the process |
| `send! p v` | copy `v` into `p`'s mailbox |
| `recv! ()` | take the next message, parking until one arrives |
| `self! ()` | the current process |
| `join! p` | wait for `p` and take its result; a failure arrives as an error |

A process that fails and that nobody joins is reported at shutdown.

**Running out is a process's failure, not the runtime's.** A process that
recurses or allocates without bound is stopped with `:stack_overflow` or
`:out_of_memory` when it passes `DREAM_MAX_DEPTH`, `DREAM_MAX_STACK` or
`DREAM_MAX_HEAP`, and the rest of the program carries on. The checks sit at
safepoints, where the process is consistent; safepoints are one reduction
apart, so a limit is overshot by a bounded amount at most.

**Scheduling** is per-worker run queues with work stealing. A process runs for
4000 reductions and goes back on a queue, whatever it is in the middle of.
Blocking IO parks the process, not the thread. When every worker is idle and
processes remain, they are all waiting on messages that cannot arrive, and the
runtime says so rather than hanging.

**Parking is a handshake.** A blocking builtin asks to be parked, and the
worker parks it once it has stopped touching the machine state. A waker that
finds the process still running leaves a note the worker checks before it
commits; otherwise a message arriving between "the mailbox is empty" and "mark
me waiting" would be lost.

## Garbage collection

Per process, and generational: a **nursery** that new objects bump-allocate
into, and an **old space** of size-classed blocks.

- A **minor** collection promotes everything reachable in the nursery to old
  space, tracing from the roots and the remembered set the write barrier
  keeps. A nursery starts at 64 KiB and doubles when a minor promotes more
  than a quarter of it, up to `DREAM_NURSERY_MAX`, so a process that allocates
  little stays small.
- A **major** is mark-sweep over the whole heap onto segregated free lists. It
  collapses the indirection chains thunk updates leave, evacuates the old
  blocks its last sweep found sparse, and hands empty blocks back.
- A large heap's collection is divided across helper threads, and a major's
  marking can run alongside the process's own reductions.

Collection happens only at interpreter safepoints, where the roots are exactly
the two stacks, the frame, the result and the globals. Allocation never
collects: its caller usually holds raw pointers in C++ locals, and a
collection would move them. A native that forces a value -- and so may run a
collection underneath itself -- has to vouch for its locals first.

[docs/gc.md](docs/gc.md) is the design, the rules a native must follow, and
the measurements.

## The JIT

A function is compiled after it has been entered `--jit-threshold` times (32
by default), on a background thread, within a budget of the program's run
time. Compiled code covers arithmetic and comparisons, branches and `match`
dispatch, self tail calls, calls to other compiled functions and to
natives, and the list, array and map operations a loop makes -- the empty
test, head and tail, cons, `.[k]` reads and writes. With deforestation in the
compiler, a pipeline of `std.list` combinators becomes one compiled loop.

The default `density` policy ranks waiting functions by calls divided by body
size; it keeps counting while LLVM works and caches each size. This was the
best general choice in interleaved self-build and kernel measurements. `work`
samples one interpreted reduction in 64 and ranks sampled work per body size.
`adaptive` uses the same samples but learns a compilation-cost model from
completed compiles, including local-slot width, and ranks work per predicted
CPU millisecond. These two policies can also make an expensive, rarely called
function hot. They are opt-in because their sampling and extra compilations
can cost more than they save on short programs.

```
dream --jit-policy adaptive program.dream
DREAM_JIT_TRACE=1 dream --jit-policy adaptive program.dream
python3 dream/tests/jit_selection_bench.py --policies density adaptive
```

`--stats` reports the chosen policy; the trace reports each queue choice,
its score and estimated cost. `--jit-policy` takes precedence over
`DREAM_JIT_RANK`. Tail recursion, additive accumulator forms and tail calls
through bindings compile to loops. Branching recursion retains its evaluation
order and falls back to the interpreter's heap continuations when the actual
machine-stack budget is spent. The budget covers nested interpreter forces
and follows the current scheduler worker's stack, including after migration.

**What it will not compile, and why.** Compiled code evaluates a self tail
call's arguments eagerly. Doing that to an argument the callee would never
have forced turns a program that terminates quietly into one that raises, so
a strictness analysis runs first and an argument the function might not force
is left lazy or the function stays interpreted. A strict parameter (`!acc`)
is the way to tell it what the analysis cannot prove.

Every fast path is inline and every slow path calls the interpreter's own
helper, so the two tiers cannot disagree about what an operation means. The
end-to-end tests run every program under both and require identical output.

**Compiled loops stay preemptible.** A back edge spends a reduction; when the
budget runs out the loop writes its values back to the frame and yields, and
the interpreter resumes it at the top of the body.

Type signatures reach the JIT as hints: an integer signature favors the
fixnum fast paths, and a float one can select guarded unboxed doubles.
`--dump-jit FN` prints the IR. Against CPython, on the workloads in
[`benchmark/`](../benchmark/README.md), compiled numeric and list loops are
several times faster; [docs/notes/vm-performance.md](docs/notes/vm-performance.md)
is the record of how, and of what was tried and not kept.

## Native modules

The parts of the standard library that must touch the machine are here, in
C++, and everything else in [`mind/std`](../mind/std/README.md) is Dream
written on top of them. [docs/builtins.md](docs/builtins.md) is the
reference.

| | |
|---|---|
| `std.io` | handles, bytes, files, pipes; reads that park the process, not the thread |
| `std.os` | arguments, environment, directories, child processes (`exec!`, `replace!`), clocks, the platform |
| `std.net` | TCP sockets and listeners, through epoll on Linux and a poller elsewhere |
| `std.tls` | TLS on a socket, upgraded in place: OpenSSL on Linux, macOS and Android, SChannel on Windows ([src/tls.hpp](src/tls.hpp)) |
| `std.ffi` | calling C through libffi; C pointers as owned handles ([docs/ffi.md](docs/ffi.md)) |
| `std.math` | the floating-point functions |
| `std.crypto` | hashes, HMAC, PBKDF2, HKDF, constant-time comparison, random bytes |
| `std.tensor` | packed numeric arrays with fused, vectorized kernels and an OpenCL backend ([docs/notes/tensors.md](docs/notes/tensors.md)) |
| `std.vm` | the compiler's hook into a VM of its own, for compile-time evaluation and macros |

## Embedding

[`include/dream/dream.h`](include/dream/dream.h) is a C API: create a VM, load
an image, register host modules, run an entry point, read the result.
[`examples/embed.c`](examples/embed.c) is a complete program.

```c
dream_vm* vm = dream_vm_new();
dream_vm_load_file(vm, "program.dream", err, sizeof err);
dream_vm_run(vm, NULL);           /* runs main! */
puts(dream_vm_result_text(vm));
dream_vm_free(vm);
```

A host function receives its arguments forced and returns a value or an error.
One registered with arity `DREAM_VARIADIC` takes however many arguments its
call site passed. Because functions are curried, such a member can never be
partially applied, so it suits a logging call and not anything a caller might
want to pre-fill.

## Layout

| | |
|---|---|
| `src/value.hpp` | value tagging and the heap objects' layouts |
| `src/heap.cpp`, `src/gc_pool.cpp` | the per-process heap, the collector, and its helper threads |
| `src/image.cpp` | loading, mapping and validating a `.dream` image ([docs/bytecode-format.md](docs/bytecode-format.md)) |
| `src/interp.cpp` | the reduction machine |
| `src/process.cpp`, `src/scheduler.cpp` | processes, mailboxes, workers, run queues, parking |
| `src/runtime.cpp` | a loaded program and the processes running it |
| `src/builtins.cpp` | the builtins and the primitives `std` wraps |
| `src/io.cpp`, `src/os.cpp` | files, sockets, processes, the event loop |
| `src/tls*.cpp` | TLS, one file per backend |
| `src/ffi.cpp` | `std.ffi` |
| `src/bigint.cpp`, `src/crypto.cpp`, `src/digest.cpp` | bignums and cryptography |
| `src/tensor*.cpp`, `src/gpu.cpp` | tensors: the CPU kernels (baseline and AVX2) and OpenCL |
| `src/jit.cpp`, `src/jit_rt.cpp` | strictness analysis, LLVM code generation, and the runtime compiled code calls |
| `src/capi.cpp` | the C embedding API |
| `src/cli.cpp` | the `dream` command |

## Tests

```
just test-vm           # the C++ unit tests: values, heap, images, atoms
just test-e2e          # real programs under the interpreter and the JIT, which must agree
just test-heap         # the same programs, the heap verified after every collection
just test-races        # the same under ThreadSanitizer
just fuzz              # malformed images must be refused, never crashed on
just test-ffi          # std.ffi against a C library built from tests/ffi
just test-tls          # std.tls against itself and against OpenSSL's tools
```

`tests/programs/` holds the end-to-end programs and the output each must
produce.
