# Tensors: fast numeric arrays, on the CPU and the GPU

The design behind `std.tensor`, `@`, and arithmetic on tensors. The code is
[dream/src/tensor.cpp](../../src/tensor.cpp) (checking and dispatch),
[tensor_kernels.inc](../../src/tensor_kernels.inc) (the CPU kernels) and
[gpu.cpp](../../src/gpu.cpp) (the GPU). Numbers are from a 4-core Xeon at
2.1 GHz (AVX-512 capable, kernels built for AVX2) and are as of when this was
written.

## Why a new kind of value

An array of floats in Dream is an array of pointers to float boxes. Every
element is its own sixteen-byte object, every product is another allocation,
and every read is a pointer chase. None of what makes numeric code fast --
contiguous loads, a vector unit, a blocked loop that keeps its working set in
cache -- can be done to that. Writing the matrix product as well as Dream
allows (arrays of arrays, a strict accumulator, indexing rather than walking)
measures **6.1 s for 256 x 256**. The same product as `a @ b` on tensors takes
**3 ms**.

So a tensor is a new heap object, `TensorObj`: a header holding a shape (up to
six axes, held inline so the data starts at a fixed offset) and then the
doubles themselves, row-major. It holds no references, so the collector treats
it as a leaf the way it treats a string. Anything past the largest size class
is born in old space and never copied by a minor collection, so a large tensor
is not moved by the collector at all.

It is a value like every other. Nothing mutates one; every operation answers
a new tensor. Sending one to another process copies its numbers in one
`memcpy`.

## Operators, and why `@` is a builtin

`+ - * / %` on tensors are elementwise. They work on a tensor and a tensor, or
a tensor and a number on either side. The shapes must be equal, or one must be
the trailing part of the other, which repeats it (a row added to every row).
That is numpy's broadcasting without stretching length-one axes; the other
cases raise `:shape_error` naming both shapes. They go through `arith`'s slow
path, reached only once the fixnum fast paths have declined, so ordinary
arithmetic pays nothing for them.

The operators do **not** take lists. `+` on two lists already concatenates
them. Every `std.tensor` function and `@` *do* take nested lists and arrays,
converting them on the way in, so `[1, 2, 3] @ [4, 5, 6]` is `32.0` with no
ceremony.

`a @ b` is the matrix product: a matrix and a matrix, a matrix and a vector in
either order, or two vectors (their dot product, a number). It binds like `*`.
It is not an opcode. The parser writes it as the builtin `tensor_matmul`
applied to both sides (`ast.matmul_e`), so nothing after the parser has a new
operator to learn. The arithmetic operators are opcodes because a loop of them
has to be a loop of machine instructions; this one does a matrix of work per
call, and the call costs nothing beside that.

One trap, found the first time it ran: the synthesized name must carry the
span of the `@` token, not of the whole expression. Names are known by where
they are written, the whole expression starts where its left operand does,
and `a @ b` was resolved as `a b` -- "a tensor is not a function".

## The JIT's float invariant, and the bail that keeps it

The JIT keeps floats in registers on one promise from `arith`: an operation
with a float on one side answers a float or raises (the "`Ty`" note at the
head of jit.cpp). Tensors break that promise, because `2.0 * t` is a tensor.

The JIT's type rule stays as it was. `dream_rt_arith_f`, the slow path that
unboxes an answer, now answers 2 when `arith` succeeded with something that is
not a number, and the emitted code turns that into `JIT_BAIL`. A compiled
body is pure and writes nothing but registers; the interpreter re-runs the
call from the frame it was entered with, and `note_bail` gives the function up
after a run of bails. This is the same soundness argument the summation
loop's bail already relied on. A function that scales tensors by floats is
therefore interpreted, which costs nothing that matters: its time is in the
kernels. `jit_tensor_bail.dr` in the e2e suite is the regression test, and
both tiers print the same thing.

A negated tensor is computed as `0 - t` in both tiers, because that is what
compiled code makes of a negation it cannot do in registers. The two tiers
must print the same numbers, `-0.0` included.

## The matrix product

The CPU kernel has the shape every fast GEMM has (Goto and van de Geijn):

- Pack a 256-deep panel of B into 8-column strips and a 96-row block of A
  into 6-row slivers.
- Compute a 6 x 8 tile of C with twelve vector accumulators held in registers
  while the loop walks k.
- Treat one-row and one-column products separately; packing to multiply by
  one column is all overhead.

| | 256 | 1024 | 2048 |
|---|---|---|---|
| Dream arrays (best a program can do) | 6088 ms | -- | -- |
| `@`, one thread | -- | -- | 880 ms (19.5 GFLOP/s) |
| `@`, four threads | 3 ms | 40 ms (54 GFLOP/s) | 246 ms (70 GFLOP/s) |

Above about 3e7 operations the rows of C are split into bands, one thread
each. Each band runs the whole blocked kernel and packs B for itself. The
threads are made for the call and joined before it returns, so the scheduler
never sees them. `DREAM_TENSOR_THREADS` caps them, and 1 turns this off.

**Why two translation units and not `target_clones`.** The first version put
`target_clones("arch=haswell", "default")` on the kernel. The cloned AVX2
kernel ran at **1 GFLOP/s** where the same source built with `-march=haswell`
ran at **33**. GCC will not inline a function compiled for one instruction set
into a clone compiled for another, so the micro-kernel was called out of line
and its accumulators spilled. Now `tensor_kernels.inc` is compiled twice --
`tensor_kernels.cpp` at the baseline and `tensor_kernels_avx2.cpp` with
`-mavx2 -mfma` -- each copy in its own namespace, and `tensor_kernels()` asks
the CPU once which table to hand out. The vector type is as wide as the
target's registers (two doubles at the baseline, four with AVX) to keep the
build warning-free: a 32-byte vector without AVX draws `-Wpsabi`.

`dot` and `sum` keep eight running sums rather than one. The compiler may not
split a floating-point sum, because that changes the rounding; writing the
split out is what lets it vectorize. Every BLAS makes the same trade. The VM
unit test checks both tables against the triple loop at sizes chosen to hit
every edge tile, using small integers so the answer is exact whatever the
summation order.

## The GPU

**What it looks like to a program.** `tensor.gpu t` moves a tensor to the
GPU. Everything done to it then -- `+`, `@`, `tensor.sum`, `tensor.relu` --
runs there, and `tensor.host t` brings an answer back. Which backend runs is
decided by where the operands are, never by the operation, so a program is
written once and moved with one call:

```dream
let w = tensor.gpu (tensor.random 1 [4096, 4096]);
let x = tensor.gpu (tensor.random 2 [4096, 64]);
let y = tensor.relu (w @ x + 1.0);          // all on the GPU
console.print! (tensor.sum y);              // one number comes back
```

An operation the GPU has no kernel for (`sum_axis`, today) still works on a
GPU tensor: it reads the numbers back, runs on the host, and puts the answer
on the GPU. A program is never wrong for having moved, only sometimes slower
than it could be. Mixing a host tensor and a GPU tensor in one operation
raises `:device_error`, which names the two ways out. Silently copying would
hide the one cost a GPU program has to manage.

**Float32 by default.** `tensor.gpu` makes float32 because that is what a GPU
is fast at: most consumer cards run doubles at a thirty-second of the rate.
`tensor.gpu64` asks for float64 where the device has it. A GPU tensor prints
as `gpu tensor[..]`, at float32's precision.

**OpenCL, loaded at run time.** It supports compatible NVIDIA, AMD, Intel,
and Apple OpenCL runtimes, and has a CPU implementation (POCL), which is
how the tests exercise this code on machines without a GPU. It is loaded with
`dlopen`, and the twenty-odd types and constants used are declared in gpu.cpp
rather than taken from `<CL/cl.h>`. Building the VM therefore needs nothing
new, a machine with no OpenCL runs every program that does not ask for a GPU,
and one that asks gets `:no_gpu` naming what is missing. `DREAM_OPENCL_LIB`
points at a specific loader or vendor library and disables automatic library
selection. Dream first tries the system loader (`OpenCL.dll` on Windows,
Apple's OpenCL framework on macOS, `libOpenCL.so.1`/`libOpenCL.so` on Linux).
On Linux, if no GPU is found, it also reads vendor `.icd` registrations in
`/etc/OpenCL/vendors` and NixOS's `/run/opengl-driver/etc/OpenCL/vendors`.
`OCL_ICD_VENDORS` overrides these directories. Vendor libraries are accessed
through the standard `cl_khr_icd` dispatch table, so a separate loader package
is not required for this fallback. No vendor SDK is needed to build Dream.

The first GPU discovered wins, falling back to a CPU OpenCL device only after
the candidates have been checked; it is found on first use and kept. The
runtime must support OpenCL 1.2 and compile Dream's kernels. A graphics-only,
CUDA-only, or Metal-only installation is not sufficient: these are different
APIs and would require additional backends. Missing drivers and kernel build
failures still make `gpu_available()` return false; attempting `tensor.gpu`
reports the reason as `:no_gpu`.

The kernels are one OpenCL source, built once per element type. They are:

- elementwise, where `x[i % nx]` covers equal shapes and every broadcast in
  one kernel;
- scalar, unary and transpose;
- a 16 x 16 tiled matrix product through local memory, plus a one-row-per-item
  `matvec` for the product with one column;
- a two-stage reduction, finished on the host in double precision.

The operation codes are shared with the CPU kernels (`tensor_kernels.hpp`).

**Asynchronous.** Every operation is enqueued on one in-order queue and
returns at once. Only reading an answer back (`tensor.host`, `tensor.sum`,
printing) waits. A chain of products runs on the device back to back without
the host waiting on any of it.

### Releasing device memory

A GPU tensor is a 56-byte heap object holding a reference-counted handle to a
device buffer (`gpu::Buffer`). Freeing the buffer when the tensor dies is the
first thing in this language that needs a finalizer, and two separate things
were needed to make the loop below run in bounded memory:

```dream
let rec churn n !acc = if n == 0 { acc } else { churn (n - 1) (acc * 0.5 + 1.0) };
churn 1000 (tensor.gpu (tensor.zeros [1000, 1000]))    // 4 MB per step, 8 GB in all
```

1. **The heap keeps a list of objects that hold external memory**
   (`Heap::external_`). After a collection's trace, and before the nursery is
   reset or old space swept, `reap_external` reads the headers. A promoted
   object is followed to its copy; a young object that was not copied, or an
   old one left unmarked by a major, is dead, and its reference is released.
   Copying a GPU tensor to another heap retains the buffer, so both processes
   share one immutable buffer. The shared area refuses one, having no
   collector to release it. A heap's destructor releases what it still holds.

2. **External bytes count as collection pressure.** A GPU tensor is a small
   heap object and megabytes of device memory, so a loop fills the device long
   before it fills the nursery, and nothing ever collected. Measured before
   this: an 8 GB peak for an 8 MB live set. `minor_due` now also fires when the
   external bytes double (with a 64 MB floor). A minor that frees less than
   half of them found old objects holding them, so the next collection is a
   major.

That still measured 8 GB, for a reason outside the collector. OpenCL frees a
released buffer only once every queued command that reads it has run. The
host enqueues far faster than any device computes, so the whole loop was
queued before the first buffer could be freed. `Launch::throttle` now drops a
marker every 32 launches and waits for the one before it. The device always
has a full window queued, but what is in flight is bounded. **8 GB to
669 MB**, most of which is POCL itself and the deliberately bounded window.

## Fusion

`relu (w @ x + b) * 0.5` used to run four elementwise passes after the
product. Each pass read and wrote the whole tensor, and each but the last made
a temporary as large as the answer. For a tensor larger than cache that is
four trips through memory where one would do; on a GPU it is four kernel
launches where one would do. Fusion makes it one.

### Deferring

The elementwise operators and functions do not compute any more. They record
what they would have done and answer a **deferred tensor**: a `TensorObj` with
`TENSOR_DEFERRED` set, whose payload is a `TensorExpr` (value.hpp) rather than
numbers. That is a tiny stack program (load an input, push a constant, a binary
operator, a function), the inputs it loads, and a slot for the answer.

- **Same type, same shape.** `type_of`, `len`, `tensor.shape` and the type
  checker cannot tell a deferred tensor from a computed one, and none of them
  computes anything.
- **Programs are flat.** Building `x op y` from a deferred `x` copies `x`'s
  instructions into the new program instead of pointing at `x`. A chain is
  always one program over computed tensors, never a tree of deferred objects.
- **Errors stay where they were.** Shapes and devices are checked while the
  program is built, so `:shape_error` and `:device_error` are raised at the
  same point as before. The arithmetic is the only thing deferred, and it
  cannot fail. A GPU running out of memory is the one error that now arrives
  where the numbers are read.
- **Reading computes, once.** A product, an index, printing, `==`, a move to
  or from the GPU, or `strict!` computes the program. The answer goes into
  the `result` slot (with the write barrier, since the object may be old by
  then) and the inputs are dropped, so they can die. Every later read finds
  the answer.
- **Reductions never store.** `tensor.sum`, `mean`, `minimum`, `maximum` and
  `norm` evaluate a deferred program and fold it as they go.
  `tensor.norm t` is `t * t`, deferred, summed.
- **Long chains split.** A program has at most 64 instructions, 8 inputs, 16
  constants and a stack of 12. A chain that would pass one computes its
  longer operand and starts again from the answer. A loop such as
  `acc = acc * 0.5 + 1.0` therefore runs eight iterations per pass.
- **Small tensors don't defer.** Below 4096 numbers on the host the answer is
  computed at once: those are already in cache, and recording the program
  would cost more than the pass it saves. On the GPU everything defers,
  because there the saving is launches, and a launch costs the same however
  small the tensor.

### Running a program

**CPU.** The program runs over 256-number blocks, so every stack slot stays in
L1. The stack holds pointers, not copies:

- an input as long as the answer is read where it lies;
- a constant is carried as a scalar and meets a vector through the scalar
  kernel;
- only a broadcast input or an operation's result fills a buffer;
- the last instruction writes the answer's memory directly.

Each stack position has two buffers, so an operation never writes the buffer
it reads. The kernels take `__restrict` pointers, and an aliased one would be
undefined. Every block is the existing vectorized kernels (AVX2 where the CPU
has it). A range past 256K numbers is split across threads, which elementwise
work did not have before.

**GPU.** The program becomes OpenCL source: a `value(i)` function whose body is
the program as one expression, with a `map` kernel and a `reduce` kernel around
it. Constants are kernel arguments rather than literals, so `x * 0.5` and
`x * 0.25` share a kernel, and a loop that changes a scalar every iteration
builds nothing new. Built kernels are cached by their source. `relu` and
`sigmoid` are helper functions, so nesting them does not write the operand out
twice at every level. This replaced the separate elementwise, scalar, unary
and reduction kernels.

### Measured

A five-step chain over 32 MB tensors, with `strict!` after every step as the
unfused baseline in the same binary (`benchmark/tensor/fusion.dr`):

| | stepped | fused | |
|---|---|---|---|
| `relu (a * 2 + b - 1) * 0.5`, 4 threads | 32 ms | 10 ms | 3.2x |
| the same, 1 thread | 81 ms | 22 ms | 3.7x |
| `sum (a * b)`, 4 threads | 5 ms | 1 ms | 5x |
| the chain on the GPU (POCL), with the copy back | 14 ms | 10 ms | 1.4x |

The thousand-step GPU loop under "Releasing device memory" now runs as about
125 kernels instead of 2000. Its peak fell from 669 MB to 510 MB.

`tensor_fusion.dr` in the e2e suite checks fused answers exactly against the
same arithmetic done one float at a time. It covers broadcasting through a
deferred operand, chains past the program limits, a deferred tensor sent to
another process, and every reduction. It also passes under the heap verifier
with parallel and concurrent collection.

### Fusing into the product

`relu (w @ x + b)` was a product that wrote its answer, then one fused pass
over that answer for the `+ b` and `relu`. When the product is cheap beside
its answer -- the wide, shallow layers of a network -- that second pass is a
large share of the cost. Now the chain runs inside the product:

- **`a @ b` defers.** A large product (4096 numbers or more, or any on the
  GPU, but never the dot product of two vectors) answers a deferred tensor
  whose program is one new instruction, `FUSE_PRODUCT`. The two operands are
  kept in `TensorExpr::product_a` and `product_b`, which the collector traces
  and a send copies. Building `w @ x + b` from it appends to that program like
  any other.
- **At most one product per program, as long as the answer.** The chain is
  run on each finished piece of the product, which only works if the answer
  and the product are the same numbers in the same places. A second, different
  product in one program, or a product shorter than the answer (`v + w @ x`
  with `v` a matrix and the product a row repeated down it), makes `build`
  compute that operand first. The same product twice (`h * h + h`) is one
  product.
- **CPU: as each tile is finished.** The blocked kernel calls a
  `GemmEpilogue` when a `96 x 2048` block of C has had its last panel of K
  added in. That is the moment the block will not be touched again and is
  hottest in cache, and `epilogue_tile` runs the program over it in place,
  one row at a time. Rows are contiguous, and the program indexes by position
  in the whole answer, so a row is just a range. Bands on separate threads
  report their own tiles; tiles are disjoint, so epilogues run in parallel.
  Calling the kernel once per small band of rows instead looked simpler and
  was rejected on arithmetic: every call repacks B, which at 2048 x 2048 is
  gigabytes of copying to save a 64 MB pass.
- **GPU: in the store.** A program with a product generates the tiled product
  kernel (and the one-column `matvec`) with the store
  `C[i] = value(i, n, acc, ..)`. That makes `relu (w @ x + b)` one kernel,
  cached by its source like every fused kernel.
- **In-place care.** The epilogue rewrites C where it lies, so the last
  instruction may not write its block directly while its operands still point
  into it. `run_block` writes to scratch and the caller copies over.
- **Reductions compute products first.** `tensor.sum (w @ x)` computes the
  product with its epilogue, then folds. The fold has no tile to ride along
  with.

Measured on a 4096 x 64 by 64 x 4096 layer, `relu (w @ x + bias)`, against
the product computed first (`strict!`) and the chain fused after it:

| | product, then chain | fused into product |
|---|---|---|
| 4 threads | 93 ms | 76 ms |
| 1 thread | 279 ms | 212 ms |
| GPU (POCL), 1024 x 64 by 64 x 1024 | 28 ms | 27 ms |

POCL runs the GPU kernels on the same four cores, so its number says the path
works, not what a GPU would gain; there the saving is a kernel launch and a
full read and write of C.

**Reshape stays deferred** when no input is broadcast. Each input is then
read at exactly the position the program writes, so the program means the
same under any shape of the same size, product included.

`tensor_fusion.dr` checks every fused product exactly against the product
computed first. It covers matrix by matrix (one thread and banded across
threads), the one-row and one-column shapes, a product used twice, two
products, a product too short for the answer, reshape and reduction.
`tensor_gpu.dr` checks the generated product kernels against the host.

### Measuring a fresh tensor, and the pool for large blocks

A large object has a block of its own. When it died, the sweep handed the
block straight back with `free`, and glibc serves anything this large with
`mmap` and returns it with `munmap`. Every new 32 MB tensor was therefore
fresh pages, and its first write paid 8192 page faults: about 40 ms against
8 ms of arithmetic. Whether a run landed on fresh or reused memory decided
which number it reported. The same benchmark gave 8 ms in one process and
50 ms in the next, every run of a process alike.

**The pool** (`Heap::big_pool_`) keeps a dead big block's memory, already
faulted in, for the next big object:

- Blocks are made in size classes, eight per doubling, so a block is never
  more than 12.5% larger than it was asked to be.
- A new big object takes the *best fit* among pooled blocks no more than half
  as large again as its class.
- What one sweep retires and nothing reuses before the next sweep is handed
  back then. The pool never holds more than the larger of 64 MB and the live
  heap.
- Pooled bytes stay in the heap's held total, since they are held from the
  system.

Measured, interleaved, the same binary with and without the pool:

| | without | with |
|---|---|---|
| a 32 MB chain, six times in a process (ms) | 55 11 10 11 10 18 | 51 9 9 11 9 14 |
| 300 tensors of five sizes, 8–40 MB, none kept | 2.77–2.99 s | 2.23–2.90 s |
| the same loop's peak memory | 382 MB | 358 MB |

Two versions were measured and not kept:

- **Exact size classes.** A block went only to an object of its own class.
  On the loop of five sizes this was 1–15% *slower* than no pool at all, run
  after run. glibc's own heap already reuses freed memory for any size up to
  its mmap threshold (32 MB once it has adapted), by splitting and
  coalescing. An exact-class pool missed whenever sizes varied, and took
  fresh memory while it held the old.
- **Huge pages.** Big blocks of 2 MB or more were allocated 2 MB-aligned and
  marked `MADV_HUGEPAGE`, to make a first touch one fault per 2 MB instead of
  one per 4 KB. Most runs improved slightly (5–7 ms steady). One stalled
  instead: 376 ms for the first write and about 65 ms for every one after,
  which is the kernel compacting memory synchronously on fault under the
  `madvise` defrag setting. A tenfold worst case is not worth a few
  milliseconds.

The first touch of memory the process has never had is still paid once: the
first 32 MB tensor costs about 45 ms. Warm up before measuring.

### Transposes, and chains into a product

Two passes remained that nothing needed. A transpose was copied before
anything read it, and a chain feeding a product (`(x - mean) / sd @ w`) was
computed into memory before the product packed it into panels, which is a
copy anyway. One mechanism removes both: the product reads its operands
through a descriptor rather than a pointer.

- **A transpose defers.** `tensor.transpose t` on a large tensor answers a
  deferred `[LOADT k]`, an instruction that reads an R x C input as its
  C x R transpose at any position. A chain reads it in place, gathering with
  counters rather than a division per element. A transpose of it is `t`
  itself. Computing it alone is the blocked transpose kernel, which moves a
  cache line at a time both ways.
- **A `GemmOperand`** is either a pointer with row and column strides, or a
  callback that writes a row segment. The packers read through it: row by
  row, or down columns when the columns are contiguous.
  - A computed matrix is strides `(width, 1)`.
  - A transpose is strides `(1, width of its source)`, read with nothing
    copied, which is how a BLAS takes `'T'`.
  - Any other deferred tensor on the host is a callback, `operand_segment`,
    which runs its program for exactly the row segment the packer asks for,
    straight into the buffer it packs from. A chain feeding a product is
    computed once per packing of its panel and never stored. The thin
    products (one row, one column) read through the same descriptor.
- **The GPU's product kernels take strides**, fixed and generated alike, so
  a transpose on the GPU is read in place too. A chain feeding a GPU product
  is still computed first: generated packing is the next step there.
- **Operands that cannot be read in place are computed first.** A product's
  own operand that is itself a pending product, and on the GPU anything
  deferred but a transpose.

The VM unit test checks the product with B read by strides and A by callback
against the triple loop, exactly. `tensor_fusion.dr` checks transposes read by
a chain, by a product on either side, through the thin paths, and inside a
chain feeding a product with an epilogue, all exactly against the operands
computed first. `tensor_gpu.dr` checks the strided device kernels against the
host.

Measured against `strict!` on the operand, which computes it first:

| | 4 threads | 1 thread |
|---|---|---|
| `tensor.transpose m + n`, 2000 x 2000 | 26 → 9 ms | 25 → 21 ms |
| `(x - 0.5) * 2.0 @ w`, 4096 x 1024 by 1024 x 64 | 17 → 10 ms | 47 → 39 ms |
| `a @ tensor.transpose b`, 1024 x 1024 | 48 → 35 ms | 107 → 104 ms |
| `tensor.transpose a @ b`, 1024 x 1024 | 40 → 32 ms | 110 → 96 ms |

The gather was the risk: reading a transpose element by element walks a
column of its source, a cache line per element. It still beats the blocked
kernel plus a pass, even single-threaded, because the chain's pass and the
temporary go with it. On four threads it also runs in parallel, where the
blocked kernel does not.

### Finishing fusion

The last operations that computed their operand first, and what each does
now:

- **Reshape copies nothing.** A large computed tensor on the host reshapes to
  a deferred `[LOAD k]`, which a chain reads in place and a product reads by
  strides. A deferred tensor stays deferred, as before; on the GPU the buffer
  is shared, as before. Only a small host tensor is copied.
- **An outer product is a product.** `tensor.outer a b` is the column `a` times
  the row `b`, two free reshapes and an ordinary deferred `@`. A chain after
  it fuses as after any product, and a chain before it is computed as it
  packs.
- **`dot` of anything deferred is a fused multiply and sum.** The elementwise
  product is deferred and folded as it is computed, on either device. Two
  computed host vectors keep the dot kernel, which allocates nothing.
- **`sum_axis` runs the operand's program as it sums.** On the host there are
  two traversals, by the width of what follows the axis. A wide one adds a
  segment of a row per step along the axis. A narrow one (the last axis)
  scans each output's contiguous range once; adding segments of one number
  would be a block evaluation per number. Either adds along the axis in
  order, so the answer is the plain sum written out one float at a time, the
  same at any thread count. On the GPU it is a generated `axis` kernel, one
  work-item per answer. Before, a GPU `sum_axis` read the operand back, summed
  on the host and uploaded the answer.
- **A chain feeding a GPU product is computed as the kernel loads it.** An
  operand that is a deferred program gets its own value function (`va`, `vb`)
  in the generated product kernel, prefixed so that three programs share one
  parameter list. Its elements are computed straight into the tile of local
  memory the kernel was copying into.

With `sum_axis` fused on the GPU, nothing falls back to the host any more.
The only reads back are of things that answer in host terms: a number, a
list, a printed string, `==`.

Measured against `strict!` on the operand (4 threads; 1 thread in brackets):

| | stepped | fused |
|---|---|---|
| `sum_axis 0 (m * 2 + n)`, 2000 x 2000 | 6 ms (14) | 5 ms (8) |
| `sum_axis 1 (m * 2 + n)` | 8 ms (24) | 6 ms (17) |
| `dot (flat * 2) flat`, 4M | 4 ms (12) | 1 ms (5) |
| GPU `sum_axis 0`, against the old round trip | 39 ms | 13 ms |
| GPU `(x - 0.5) * 2 @ w` (POCL) | 25 ms | 25 ms |

POCL computes the "GPU" on these same cores, so a chain computed into the
tile shows nothing there; on a GPU it saves a kernel launch and a write and
read of the operand.

**Measured and not kept: folding a reduction into the product.** `tensor.sum
(w @ x)` folded each tile as the product finished it, with the partials
combined in tile order so the answer stayed the same at any thread count. On
one thread it was 4–20% *slower* in every run (184 → 198 ms, 210 → 257,
187 → 195, 189 → 200 for a 4096 x 64 by 64 x 4096 product). On four threads it
was noise both ways (84 → 67, 67 → 80, 98 → 63). What it saved was one read
pass over C, which costs almost nothing beside the product. What it cost was
reading each finished tile while the packed panel of B is still needed for
the next block of rows, evicting it. An elementwise epilogue pays the same
eviction but saves a whole write pass and a read, which is why that one is
kept and this one is not. A reduction of a product computes the product,
keeps it, and folds it.

`tensor_fusion.dr` checks each of these exactly where the arithmetic is the
same: free reshapes in chains and products, outer products with an epilogue,
`sum_axis` along every kind of axis against the sum written out in Dream,
and `sum_axis` of a product. Dot products and sums are checked to a
tolerance, since their summation order changes. `tensor_gpu.dr` checks chains
into GPU products on either side and in the one-column kernel, the GPU
`sum_axis` on both axes, outer products and fused dots.

### What is left

- **A product as an operand of another product.** `(a @ b) @ c` computes the
  inner product, as it must: the outer product reads each of its elements K
  times.
- **Elementwise work across the two devices.** A host tensor and a GPU tensor
  in one operation is still an error (`:device_error`), by design.

## What machine learning needed

`std.ml` (see [ml.md](https://github.com/dream-mindv2/std/blob/main/docs/notes/ml.md)) is written on `std.tensor` and nothing else,
and it found the operators and the handful of functions above short of
what a network's forward and backward passes are made of. What was added, and
the shape each took:

- **Comparisons and the larger of two.** `lt le gt ge eq ne`, `max`, `min`
  and `pow` are binary `KernelOp`s like `+`, so they broadcast, defer and
  fuse exactly as the operators do (`tensor_binary`, which `tensor_arith`
  now calls). A comparison answers 1 or 0, so a relu's gradient is `g *
  tensor.gt x 0.0` -- one more instruction in the chain, not a pass. `max`
  and `min` of two are named apart from the reductions only by arity.
  `floor ceil round sign erf` joined the functions of one argument; `erf`
  is what an exact GELU needs.
- **A value per row, across the row.** Broadcasting is trailing only, and a
  softmax subtracts each row's maximum from the row: the row's value is
  *leading*. `tensor.repeat n t` gives `t` a new last axis along which each
  number repeats, and it is not stretched in memory: it is a program of one
  instruction, `FUSE_LOADR`, which reads input `k` at `(i / r) % count`. The
  instruction's argument packs the input (3 bits) and the constant holding
  `r` (4 bits), so no field was added to `TensorExpr`; on the GPU the
  repeat is a kernel argument like every constant. `softmax` of a matrix is
  therefore two folds and one fused pass, with nothing the size of the input
  stored but the answer.
- **Folds along an axis other than the sum.** `sum_axis` became
  `reduce_axis` with an op: `max_axis`, `min_axis`, and the `ARG` forms,
  which answer positions as numbers so that they can stay on the device.
  The host's fold starts from each output's first value rather than an
  identity (there is no honest identity for a maximum of floats), in order
  along the axis, so answers are the same at any thread count. The GPU's
  generated `axis` kernel takes the op; `argaxis` is a sibling kernel in
  the same generated program.
- **Rows: `slice`, `take`, `concat`, `one_hot`.** A batch is a slice; a
  shuffled batch, or an embedding lookup, is a `take`. On the GPU a run of
  consecutive rows is one `clEnqueueCopyBuffer` and anything else is one
  `gather` kernel, whose row numbers go up as a small buffer of their own.
  `concat` is copies at offsets (`gpu::copy` gained a destination offset).
- **Windows: `im2col` and `col2im`.** A convolution is every window of the
  image as a row, times the kernel as a matrix -- the arrangement every fast
  convolution reduces to, which makes its speed the product's. Images are
  channels last, so that a window's channels are contiguous and each
  window's row is `kh * kw` memcpys. `col2im` is the gradient: each pixel the
  sum of the places it was copied to. On the GPU it is written as a gather,
  one work-item per pixel adding up the windows that cover it, so nothing
  is written twice and no atomics are needed; on the host the threads divide
  the images, so the sums are in the same order at any thread count.
- **Batched products and `permute`.** `@` takes `[b, m, k]` by `[b, k, n]`
  -- attention is many small products, so the host runs one per thread
  rather than splitting each, and the GPU runs them as the third dimension
  of one tiled launch (`bmm`) -- and `[b, m, k]` by `[k, n]`, which is an
  ordinary product of the rows and so still defers and fuses. `permute`
  reorders any axes, which is how heads move between the sequence and the
  batch; the host walks the answer with a counter per axis, and the GPU
  kernel takes the strides as arguments.
- **`normal`, `to_bytes`, `of_bytes`.** Initial weights, and saving them
  exactly.

`tensor_functions.dr` checks every one of these exactly on the host, and
`tensor_gpu.dr` checks each against the host on the GPU.

### A GPU program could crash as it exited

About one exit in thirty, under POCL, a program that had used the GPU died
with SIGSEGV *after* printing everything right. The core showed the main
thread in `exit()` running library destructors (`_dl_fini`) while one of
POCL's driver threads was still compiling a kernel that had been queued and
not waited for -- its LLVM state destroyed underneath it. Nothing in the VM
waited for the queue at exit, because nothing had to for the answer: work
queued and never read is work nobody needs. The fix is an `atexit` handler,
registered when the device is set up, that waits for the queue
(`drain_at_exit` in gpu.cpp); `atexit` handlers run before the libraries are
torn down. It was older than the functions above, which only made it more
likely by queueing more. 0 crashes in 100 runs after, against about 1 in 30
before.

## On a real GPU

Everything above about the GPU was measured on POCL, which runs OpenCL on the
host's own cores and so could only say that a path worked. Run on an RTX
3070 (NVIDIA's OpenCL, 2026-10-04), against the 12-core Ryzen 5900X host,
`std.ml` training showed four things the CPU implementation hid.

**A long K with a small answer.** The tiled kernel gives each tile of C one
work-group, which walks the whole of K. A convolution's kernel gradient is
`[kh * kw * c, rows] x [rows, filters]` with `rows` a hundred thousand: two
work-groups each doing a hundred thousand steps, on a device built to run
thousands at once. **Split-K**: when C has too few tiles, K is cut into
slices (`matmul_part`, the slice the launch's third dimension) whose partial
products a second kernel adds in slice order. A product with a chain after
it is then the product and the chain as a separate fused pass
(`run_split_product`), since no slice holds a finished element to apply the
chain to.

**A long axis with few answers.** The same shape in `sum_axis`: a bias's
gradient sums a hundred thousand rows into thirty-two numbers, and the axis
kernel ran one work-item per answer -- 32 ms of a 34 ms training step. Such
an axis is folded in slices side by side, and the partials folded again by
the same generated kernel over a program that only loads them. A conv
training step went from **34 ms to 3.6 ms**. The `ARG` folds stay single
pass.

**The product kernel itself.** The textbook tiled kernel reads two values
from local memory per multiply-add, and that bound it: **1.2 TFLOP/s** for a
4096-square product on a card that does twenty. Now every product -- plain,
split slice, batched, and the generated one with a chain in its store -- is
one generator (`product_kernel`) in which a work-item computes a `w x w`
block of C held in registers, reading `2w` values per `w * w`
multiply-adds. 8 x 8 runs the 4096 product at **5.9 TFLOP/s** (23 ms) and
4 x 4 at 4.1, but a large block makes few work-groups: a 256 x 64 product
is one work-group at 8 x 8, and small networks got slower. So the products
are built at 2, 4 and 8, and each launch takes the largest that still makes
96 work-groups (`pick_wpt`).

| | host | RTX 3070 | |
|---|---|---|---|
| `@`, 512 | 4.5 ms | 0.25 ms | 18x |
| `@`, 2048 | 95 ms | 3.3 ms | 29x |
| `@`, 4096 | 545 ms | 23 ms | 23x |
| 64 products of 128 x 64 by 64 x 128 | 10.2 ms | 0.48 ms | 21x |
| 3 x 3 convolution, 64 images of 32 x 32 x 16 | 22 ms | 1.9 ms | 11x |
| a five-step fused chain over 16M | 3.1 ms | 0.47 ms | 6x |

Each is the time per call with the answer read back, after a warm-up call.

**Device memory between collections.** A training loop under a caller that
could not collect (see "Why a process" at `ml.fit!`) filled the 8 GB card.
Looking for that found two weaker bounds, both tightened though neither was
the cause: the launch throttle now also closes a window once it has
allocated an eighth of the device's memory, since two windows of
convolution launches each touching a hundred megabytes pinned gigabytes of
released buffers; and the collector's trigger for external memory, which
doubles past what survived the last collection, is capped at a quarter of
the device (`external_budget`). `vm.stats!` now reports `external_bytes`,
which is how the cause was told apart from these.

`tensor_gpu.dr` checks split products (plain, with a chain, through a
transpose, with a computed operand), split folds along each kind of axis,
and float64 products, on both POCL and the 3070.

## Considered: a server process that owns the tensors and mutates them

The question was whether tensor operations should run in a VM process (or
several) that owns the data and mutates it in place, since the data never
leaves C++ anyway.

**On the CPU, no.** Messages are copied, so every operation would copy its
operands in and its answer out -- more work than the one allocation per
operation it would save. Allocating the answer in the caller's own heap is
already the cheaper side of that trade.

**On the GPU, yes, and the backend already has that shape.** One owner (the
runtime's device and queue) holds the memory, mutation-free buffers are
shared by handle rather than copied, and work is queued to it rather than run
by the caller. It is part of the runtime rather than a Dream process, so
handles never go through message copying.

What mutation would actually buy is fewer allocations in a chain like
`a * 2.0 + b`. Fusion (above) gives that, without exposing mutation.

## Transport and compile-time values

The wire format now carries tensors as a rank, dimensions and little-endian
double values. Both the native codec and the Dream reference codec preserve
their shape and numbers, including tensors nested in containers; device
values arrive on the host. Malformed shapes and truncated payloads are rejected.

`comp` can return tensors too. Values copied out of the compile-time runtime
are materialized on the host, and the compiler emits shape and byte constants
reconstructed with `_tensor_of_bytes`. `comp_tensor.dr` covers deferred and
nested tensors.

`sum_tensor` retains a total as a one-element tensor on the input's device;
`random_like` generates uniforms with its shape, device and dtype. These
support device-local loss accumulation and dropout in `std.ml`.

## What is not done yet

- **Float32 on the host.** Twice the vector throughput, and the type a model
  is stored in. `dtype` is already in the header.
- **A faster GPU product.** Register tiling is done ("On a real GPU"); at
  5.9 TFLOP/s it is still a third or less of what cuBLAS gets from the same
  card. Vector loads, double-buffered tiles, and a CUDA or Metal backend
  behind the same `gpu::` functions are what is left.
- **Sending a host tensor without copying it.** A large one could be
  reference-counted the way BEAM shares large binaries.
