# Dynamic linking

A `.dream` image is a whole program, and a VM runs one of them. This document
is the plan for taking that down: one `Runtime` holding many images, loaded at
startup or while the program runs, with true cross-image references — a library
image's functions and values callable from the program that loaded it.

"Dynamic linking" here does **not** mean either of the things that already
exist:

- `std.ffi` loads a **C** shared library and binds native functions. This is
  about `.dream` images.
- `vm.eval_image!` runs a whole second program in a fresh child `Runtime` and
  copies the *result* back as data — it refuses closures, because a compile-time
  value has to be something the compiler can write into an image
  (builtins.cpp:1358-1359, `import_across`). A library is code, not data, so
  that path cannot deliver it.

What exists today is the session half: `std.image` and the `image`
declaration ([images.md](images.md)) open an image in a runtime of its own and
call its functions with data, on top of `open_image!`'s sessions. Everything
below is what it would take for a closure to cross as well, and is still a plan.

The end state: `dream --library X.dream main.dream` loads `X` beside `main`, and
a `vm.load_image!` builtin does the same from inside a running program. The
loaded library's globals are ordinary values to the caller.

## The single-image runtime today

Every part of the runtime is sized and resolved against one image, and the
compiler is what makes that image self-contained:

- `Runtime` owns exactly one `std::unique_ptr<Image> image_`
  (runtime.hpp:175). `load_image_*` *replaces* it, and "an image replaced at run
  time is not something the runtime supports" (runtime.cpp:113-116).
- Every reduction re-resolves its indices against that one image. The wrapper
  `img_of(p)` (interp.cpp:17) is called per step — for the node being executed
  (interp.cpp:770-771), for a closure's `FuncRec` on apply (interp.cpp:217), for
  children and constants throughout `step_eval`. None of it is cached across
  steps.
- A closure knows only a `func` index, a thunk only a `node` index, a module
  object only an `import_index` (value.hpp:232-271). These are deliberately raw
  integers invisible to the GC (heap.cpp:421-431, 455-457) — they point into
  Runtime-owned, immutable image memory, not the heap.
- The field cache, the import cache, the profile counters, and the JIT's
  per-function cache are all sized from the one image's counts
  (runtime.cpp:117-121, jit.cpp:778-779).
- The compiler fuses cross-module calls into a single in-image `:global`
  (lower.dr:292): "a cross-module call is one index by the time the image is
  written, which is the whole point of compiling the program as a whole"
  (scope.dr:68-72). The only imports left in an image are *host* modules
  (`std.io`, `std.os`, …) provided from C++, resolved by name at runtime
  (interp.cpp:721-763).

So an image is not a library: its edges point into itself, and nothing in the
VM can say "global 300 belongs to that other image."

## The architecture: a composite merged image

The runtime will keep the loaded images (their sections stay mmap'd, untouched)
and materialize one owning, *merged* `Image` whose index space is shared. All
references — a library global, a library function, a literal it embeds — are
ordinary indices into the merged tables. Everything downstream then works
without knowing there was ever more than one image: the interpreter, the JIT
(which reads image tables only at compile time and holds no image pointers in
generated code, jit.cpp:822), the GC, the field cache. That single property is
why merging wins over the alternative:

**Why not per-object image identity.** Giving every closure, thunk, and module a
home-image id keeps the images mmap'd and unrenumbered, but the cost is that
*nothing* is shared anymore: the hot loop must resolve each object's indices
through an indirection, cross-image edges need a new (image, index) edge kind
— a format change — and the JIT's cache, the GC, and `copy_value` all grow a
field. Merge confines the change to one place: the loader. It is the difference
between a new thing and a new warm.

**Incremental append keeps old indices stable.** Each image's edges are
self-contained — its node, func, global, string, and atom indices all point
within itself, a property the loader already proves (image.cpp:326-544). So
merging is *append*: a newly loaded image's local indices map to a fresh range
just past the merged tables, and the existing tables do not move. What that buys
is that references and caches stay valid across a load:

- a node index already in a process's stack is still the same node;
- the field cache is keyed by node index (runtime.cpp:117-121), so a cached
  field hit for a node that predates the load remains correct;
- only the new image's children require remapping, and only the table *sizes*
  grow.

**The remap.** Appending is not concatenation: every edge that points *into* the
new tables must be renumbered. The pass is opcode-aware — `apply`'s `a` is the
callee node, `int`'s `a` is a `KINT` index, `global`'s `a` is a `GLBL` index, a
function's `body` is a node, a global's `target` is a func or import, a module
record's `derives` is a module index — which is precisely the knowledge
`Image::validate` already encodes per opcode (image.cpp:392-440). The merge
reuses that structure; it is a renumber of one side of every edge, and it must
also unify atoms:

- **Atoms** are compared by identity, and two images may intern `:ok` at
  different indices. Merging interns by name into the merged `KATM` through the
  runtime's atom registry, so `:ok` is one index everywhere.
- **Strings** concatenate their `SBLB` blobs and rebase every `KSTR` offset.
- **`MODS`** keeps one record per source module with its global range, so module
  identity — which globals belong to which library — survives the merge, exactly
  as it survives a whole-program compile.

**The cost.** The merged image owns its sections, so a loaded library's bytes
are copied out of the mmap into RAM. That is the format's zero-copy promise
(bytecode-format.md:4-11), given up for the second image onward. Images are
megabytes; the trade is worth it. If it ever is not, per-object identity is the
documented alternative.

## The cross-image surface

A library is presented as a *module value*: generalize `ModuleObj` and
`resolve_field` (interp.cpp:721-763), which today answer `.field` on a module
from the C++ host registry. Add a foreign registry of loaded libraries; a field
on a library's module value looks the name up in the merged `GLBL` and returns
what it finds — a function global becomes a plain closure
(`make_closure(merged_func, 0)`, valid precisely because the index space is
shared), and a pure value global resolves lazily through the ordinary
`global_value` path.

Purity is the one guarantee this gives up deliberately. The compiler enforces
"a pure function cannot call an impure one" across the whole program at compile
time; a library loaded at runtime was never part of that compilation. The `!`
flag survives on the `GLBL` record, so the information is there; enforcing it at
the boundary is a stretch goal, not part of this change. The document records
that the boundary is a documented, accepted hole.

## Concurrency and safe points

The merged image is immutable between loads, so workers can read it in
parallel today. A load at runtime is the one mutation, and worker threads hold a
local `const Image&` across several derefs within a single `step_eval`
(interp.cpp:770-771+) — swapping the image structure under them is use-after-free.

Two answers, one easy:

- **At startup**, before `sched.start()`, there are no workers. This is where
  `--library` and `dream_vm_load_library` load.
- **At runtime**, `vm.load_image!` stops the world. The scheduler already has
  the machinery: a worker that asks to park stops touching machine state and
  says so (scheduler.cpp:153-169). The builtin parks every worker, merges the
  image, grows `field_cache_`, `import_defs_`, and the profile counters, tears
  down the JIT's compiled cache (fresh functions are cold anyway), and resumes.
  The operation is rare and the park handshake is a settled mechanism, not new
  latch code.

## The plan

Each phase is shippable and leaves the single-image behavior bit-identical when
nothing is loaded:

1. **`Image::build_merged` / `append`** in C++: the opcode-aware remap, atom
   interning, incremental append. Merge-invariant tests join the image checks in
   the `test-vm` group.
2. **Startup multi-load**: `dream --library IMG …` (repeatable) and
   `dream_vm_load_library` in capi.cpp, merged before caches and the JIT are
   sized.
3. **The foreign-module surface**: field dispatch onto merged `GLBL`, a `std.vm`
   builtin that hands a loaded library to the program as a module value. An
   e2e test — a library exposing functions and the program calling them — must
   agree under the interpreter and the JIT.
4. **Runtime `vm.load_image!`**: stop-the-world, cache growth, error tables for
   malformed or unloadable images, atoms unified on the fly.

## What this does not change

- The bytecode format: no new opcode, no section, no wider index. Merging is a
  loader concern. (The `MODS` section, which a merge leans on for module
  identity, is documented in `docs/bytecode-format.md`.)
- `mind`'s whole-program build, the bootstrap seed, and the byte-equality
  guarantee: a single compiled program still produces one image. Linking is an
  explicit, separate act.
- The `comp!`/`vm.eval_image!` path: compile-time values are still copied out
  of a child VM as data, and must remain so.