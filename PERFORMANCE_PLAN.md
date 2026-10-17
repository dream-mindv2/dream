# Self-compilation below 1.25 seconds

Current compiler work: [macro expansion performance TODO](https://github.com/dream-mindv2/dreams/blob/main/TODO.md).
It records the 2026-09-20 baseline and the plan to remove whole-program macro
snapshots, including an incremental compile-time VM option. The measurements
below are historical and predate that baseline.

## Where this actually stands (2026-09-18, second pass)

The target is not met and is not close. The best configuration measured here is
**2.32 s** against 1.25 s, a factor of 1.86, and the two changes made since the
round below are worth about 2% of that between them. What that round itself
bought, separated by building the same source twice, is less than it reads as:

| | `--no-jit -j 1` |
| --- | ---: |
| this source, no PGO | 3.055 s |
| this source, GCC PGO=USE | 2.572 s |

and against the commit before it, both built without PGO, in the same
configuration (`--no-jit -j 4`, the one the older VM survives):

| | median | min |
| --- | ---: | ---: |
| before | 2.732 s | 2.646 s |
| after | 2.715 s | 2.673 s |

So the VM changes below are worth **0.6%**, inside the ±3% placement-noise
floor this repository documents for any edit to `step_eval`. The 16% that is
real is `-fprofile-use`, a build flag. The section titled "Implementation
results" should not be read as a record of interpreter improvements.

**What the round did buy is a correctness fix, and it is worth more than the
timing work.** The VM at the previous commit does not merely fail occasionally
at `-j 1` -- it segfaults every time, 3 runs of 3, on a stage-2 self-compile.
The concurrent-mark launch fix is real and necessary.

**The build tree was left configured `DREAM_PGO=USE`.** `just vm` does not pass
`-DDREAM_PGO`, so the cached setting survived it, and `-Werror=missing-profile`
plus GCC's coverage-mismatch error meant that *any* edit to a VM source failed
to build in `build-dream` -- demonstrated with a one-line change to
`interp.cpp`. Reconfigured to `OFF`, which is the default and what the training
instructions below already said to do. Note also that those instructions build
with `-S dream` where `just vm` uses `-S .`; the two are not interchangeable
build directories.

PGO is now `just vm-pgo`, which builds into `build-pgo` and never into
`build-dream`, retrains from scratch every time, and is the only recipe that
reads that directory. `just bench-self-compile <vm>` judges it. The manual
instructions below are kept because they document what the recipe does and what
the knobs are, not because anyone should have to run them.

### Scheduler workers: a free 9%, taken

Not in the plan below and not a consequence of any of it. The default worker
count was one per core, and a self-compile is a single green process, so on a
24-core machine 23 workers polled every 500us and walked every worker's queue
twice per poll. The default was the worst setting on the curve (2.67 s at
`-j 1`, 2.46 s at `-j 4`, 2.70 s at `-j 24`). The default is now cores capped
at 8 and `steal` short-circuits on a counter: **3.097 s -> 2.805 s**
interpreted, **3.215 s -> 2.949 s** by default, non-PGO, `just test` and
`just test-races` clean. See "An idle worker is not free" in
`docs/notes/vm-performance.md` for the measurement and for the one edit next to it that is
unsound.

### Where the two changes leave it, measured together

Both built from this source, five warm runs each, medians:

| configuration | ordinary | trained (`just vm-pgo`) |
| --- | ---: | ---: |
| default flags | 3.001 s | **2.620 s** |
| `--no-jit` | 2.847 s | **2.523 s** |
| `--no-jit -j 1` | 3.068 s | 2.696 s |

The best number available is **2.523 s**, against 2.572 s before any of this --
so the two rounds together are worth about 2%, and nearly all of what looks
like progress in the table is the scheduler fix paying for the PGO that was
already there. They are not additive: the profile absorbs some of what the
worker cap removes.

Note that `-j 1` is no longer the fastest configuration and has not been since
the idle-worker walks were fixed, so the gate in §2 now measures the slowest of
the three. It should be restated against the default, which is what anybody
running the compiler actually gets.

The worker curve on the trained build is flat from 2 to 8 (2.459, 2.488, 2.491,
2.534 at 2, 4, 6, 8) and climbs after (2.576 at 12, 2.682 at 24). The cap is 8
rather than 4 deliberately: 4 would buy another 2% on this single-process
workload and halve the parallelism available to a program that actually uses
processes, which is the wrong trade for a default.

### Third pass: 2.32 s, and where the rest of it is

Interleaved seven rounds, `--no-jit` with default workers, each build from the
same source:

| | median | min |
| --- | ---: | ---: |
| before this pass | 2.874 s | 2.811 s |
| + inliner ceilings, `ContStack`, inline `builtin_def` | 2.760 s | 2.687 s |
| + `just vm-pgo` | **2.322 s** | **2.263 s** |

Against the VM as committed before any of this session's work, that is
**3.097 s -> 2.322 s, 25%**. The 2 s target is not met and is 16% away.

The order of the wins is the opposite of where the effort went, which is the
most useful thing in this section: PGO 12%, the GCC inliner ceilings 4%, the
scheduler's idle workers 9%, and **every source-level change to the interpreter
together under 1%**. Three separate hot symbols were attacked on the strength
of a `perf` profile and all three came back inside the noise floor. See "`step_eval`
is too big for the inliner" and the two entries after it in
`docs/notes/vm-performance.md` for why, and for what to read a profile *for* in this interpreter.

What the profile says now, at default workers, is that the collector is
**35% of all CPU samples** across three parallel lambdas -- but only 284 ms of
pause on a 2.3 s run, so most of that is helper threads burning cycles beside a
mutator they are not blocking. Reducing it means allocating less, not
collecting differently: three GC policy levers have now been measured here and
none of them moved this workload.

So the remaining 16% is where it has been all along: 75.4M reductions and
1.7 GB allocated, 39% of it frames. Two routes, both projects rather than
edits:

- **Frames that do not escape.** The §4 item below, unchanged.
- **Threaded dispatch.** `step_eval` is 16.6% and much of it is one indirect
  branch per Eval step that mispredicts by construction. Replicating the
  dispatch per opcode is the standard 10-25% for a bytecode interpreter -- but
  this one is a preemptible state machine whose loop does a safepoint and a
  limit check between every step, and those checks are what threading would
  have to move. Not attempted.

### Why 1.25 s is not reachable from here

In the gate configuration `--stats` reports **691 ms of the 2572 ms is GC
pause**. A collector that cost nothing would leave 1.88 s, still 50% over
target, and that was measured before the two changes above brought the best
configuration to 2.52 s -- which does not change the conclusion, only the
arithmetic. The remaining cost is 75.4M reductions and 1.7 GB allocated, of which
frames are 39% and thunks 20%. That is the frame-reuse and escape-analysis work
in §4 below, which is unimplemented and which the plan itself never argued was
safe under lazy evaluation. No combination of PGO, worker tuning and GC work
reaches the target; the closing sentence of §5 remains the honest summary.

## Status and scope

The target is not met. The GC fix, indexed heap verification, decoded array
accessors, benchmark gate, and optional GCC profile-guided build are implemented
in `dream/`. Compiler and standard library sources remain unchanged by this
work. Measurements are from this checkout on
2026-09-18, including the existing user edits in `dreams/src/ir.dr` and
`mind/std/src/dynamic.dr`. Those edits were preserved.

## Implementation results

The single-worker failure was a failed concurrent-mark launch leaving root
marks behind. The fallback full trace skipped those already-marked objects,
so their children could be freed. Failed launches now undo exactly the marks
acquired during seeding, including leaf roots. A unit regression covers a
declined launch, duplicate roots, and child survival through the fallback GC.

Heap verification formerly searched every allocation block for every object.
It now builds a sorted range index once per verification and checks ownership
before dereferencing a header. Full verified self-compilation completes and
reproduces stage 2 byte for byte.

The image loader now recognizes functions whose entire body reads a local
array argument at a constant or argument-supplied index. Saturated calls can
read an already-materialized array without allocating a callee frame. Suspended
inputs, invalid indexes, non-arrays, and fallbacks retain ordinary evaluation.
Selected elements are entered normally, so their laziness and errors remain
intact. No on-disk format change, unsafe frame reuse, or compiler change is
required. Profiling uses the ordinary path; the fast path accounts for the
body reduction and respects the remaining reduction budget.

Five-run single-worker interpreter medians before the combined final build:

| Configuration | Median | Maximum |
| --- | ---: | ---: |
| GC fix only | 3.092 s | 3.153 s |
| Bounded nested Return batching (discarded) | 3.115 s | 3.154 s |
| Recursive ready-field reads (discarded) | 3.178 s | 3.206 s |
| Decoded array accessors, ordinary build | 3.038 s | 3.071 s |
| GCC PGO, before accessor optimization, LLVM disabled | 2.577 s | 2.970 s |

An LTO-only trial also took about 3.10 seconds and was not retained. Native CPU
sampling using the installed gperftools library found distributed costs:
approximately 8.8% in evaluation dispatch, 6.9% resolving indirections, 6.0%
in continuation insertion, 5.2% in copying, and substantial GC work. These are
sampled locations (including inlined functions), not exclusive subsystem costs.
Single-worker GC pauses were about 0.83 seconds; the earlier 0.28–0.31-second
figures below describe default-worker parallel collection, not single-worker GC.

### Reproduce the benchmark

```sh
python3 dream/tests/self_compile_bench.py --no-jit --workers 1
python3 dream/tests/self_compile_bench.py
```

The script builds stage 2 from the seed, warms up, then performs at least five
sequential self-compiles, verifying byte equality each time. JSON includes the
CPU, build settings, image hashes, environment, individual wall times, median,
maximum, and gate status. Exit status 1 means the 1.25-second gate failed.
`--vm`, `--seed`, `--workers`, `--runs`, `--timeout`, and `--target` configure it.
First invocation is recorded separately without claiming OS caches are cold.

### Train a GCC build

Use the same fresh build directory for both phases. The chosen build type and
LLVM setting must remain unchanged between training and use:

```sh
cmake -S dream -B /tmp/dream-trained-build -DCMAKE_BUILD_TYPE=Release -DDREAM_PGO=GENERATE
cmake --build /tmp/dream-trained-build -j 4
/tmp/dream-trained-build/bin/dream_tests
/tmp/dream-trained-build/bin/dream dreams/bootstrap/dreams.dream -L mind -L . -o /tmp/dream-trained-stage2.dream dreams/src/main.dr
/tmp/dream-trained-build/bin/dream --no-jit -j 1 /tmp/dream-trained-stage2.dream -L mind -L . -o /tmp/dream-trained-stage3.dream dreams/src/main.dr
cmp /tmp/dream-trained-stage2.dream /tmp/dream-trained-stage3.dream
cmake -S dream -B /tmp/dream-trained-build -DDREAM_PGO=USE
cmake --build /tmp/dream-trained-build -j 4
python3 dream/tests/self_compile_bench.py --vm /tmp/dream-trained-build/bin/dream --no-jit --workers 1
```

Train in a **fresh** build directory, never in `build-dream`. `just vm` does not
pass `-DDREAM_PGO`, so a cached `USE` survives it, and the next edit to any VM
source then fails the build with `-Werror=coverage-mismatch` rather than
rebuilding. This has already happened once.

PGO defaults to OFF and requires GCC when enabled. Training uses atomic
counters; `os.exit!` explicitly flushes them because `_Exit` skips GCC's normal
shutdown writer. USE builds have no instrumentation and reject missing or
incompatible profiles. After changing VM sources, retrain in a fresh build
directory or configure `-DDREAM_PGO=OFF` before rebuilding. Broader training
workloads may give different performance; this profile targets self-compilation.

The numbered sections below retain the original investigation and remaining
work. In particular, the 1.25-second acceptance gate remains open; broad frame
reuse and general instruction fusion have not been implemented or proven safe.

## Original measured baseline

The existing RelWithDebInfo build was confirmed current with
`cmake --build build-dream -j 4`.

Run from the repository root (VM options must precede the image):

```sh
time ./build-dream/bin/dream dreams/bootstrap/dreams.dream -L mind -L . -o /tmp/dream-perf-stage2.dream dreams/src/main.dr
time ./build-dream/bin/dream --stats /tmp/dream-perf-stage2.dream -L mind -L . -o /tmp/dream-perf-stage3.dream dreams/src/main.dr
cmp /tmp/dream-perf-stage2.dream /tmp/dream-perf-stage3.dream
```

- Seed compilation: 3.335 seconds wall time.
- Unprofiled stage-2 self-compilation, with statistics: 3.101 seconds.
- Default-worker interpreter-only self-compilation: 3.097 seconds.
- Instrumented function profile: 3.478 seconds; not a benchmark result.
- Stage 2, stage 3, and a repeated output were byte-identical: 485,800 bytes,
  27 modules, 19,667 nodes, 1,670 functions.
- Approximately 73.27 million reductions and 1.69 GB allocated per compile.
- Allocation by kind: frames 39%, thunks 20%, lists 19%, maps 18%.
- Seven major and 53 minor collections; 0.28–0.31 seconds stopped in GC.
- Only 19 functions compiled by the JIT. These runs show no meaningful JIT
  advantage; repeated controlled measurements are needed to quantify it.
- Largest reduction counts: `lower_expr` 7.7%, `check_expr` 5.7%,
  `skip_trivia` 4.6%, `map` 2.8%, `stream` 2.5%, `parse_apply` 2.5%.
  Reduction counts are not CPU-time samples.
- Existing native unit tests: 64,782 checks, zero failures.

This needs about a 2.5x speedup. Eliminating the measured GC pauses entirely
would still leave approximately 2.8 seconds. Nursery tuning alone cannot
deliver the target.

## 1. Resolve the single-worker correctness failure

Before the fix, the newly generated compiler repeatedly failed with:

```sh
./build-dream/bin/dream --no-jit -j 1 /tmp/dream-perf-stage2.dream -L mind -L . -o /tmp/dream-perf-interp.dream dreams/src/main.dr
```

Observed error: `not` needs a bool, got a list. With JIT enabled and `-j 1`,
one run exited with status 139. Default-worker runs succeeded in both modes.
The cause is unproven; do not attribute it to JIT or GC without diagnosis.

Reproduce with heap verification, then a debug/sanitized build and a debugger
backtrace. Compare worker counts and collection modes. Inspect root lifetimes
across `force_whnf`, native calls, and collection, as well as worker-dependent
state. Reduce the failure to a regression under `tests/programs/` or
`tests/test_main.cpp`; fix it before relying on worker-count tuning or changing
allocation lifetimes. Success means repeated byte-identical self-compiles with
one and default workers, in both execution modes.

## 2. Establish a repeatable timing gate

Add `dream/tests/self_compile_bench.sh` with configurable VM/image paths, a
private temporary directory, one warmup and at least five sequential measured
runs. Record wall time, median, maximum, build type, CPU, worker count, image
hash, and relevant environment settings. Include VM startup, image loading,
compilation, and output writing. Exclude building the C++ executable. Report
seed compilation separately from self-compilation by the generated image.

Check every exit code and output equality. Do not benchmark competing runs
in parallel or use profiler timings for acceptance. Final acceptance: all five
measured self-compiles below 1.25 seconds on the agreed reference machine;
report cold-start timing separately. Also require interpreter-only success,
since the requested improvement concerns the interpreter.

## 3. Profile native execution before choosing optimizations

Collect CPU samples for default and interpreter-only runs using a native
sampling profiler. `perf` was not found on this session's PATH. Separate time
in `step_eval`, `step_return`, `force_whnf`, allocation, container operations,
GC, and JIT compilation. The language function profile above is insufficient
to choose among these costs.

If needed, add opt-in counters for opcode/continuation pairs, frame sizes,
thunk forcing, and function entries, with no atomic increments in ordinary
execution. Rank candidates by measured total cost and expected removable
work. Avoid adding counters permanently to every reduction.

## 4. Reduce dispatch and allocation costs incrementally

First inspect `src/interp.cpp`'s nested `force_whnf` loop. The outer loop already
drains Return continuations; investigate a corresponding bounded fast path
inside nested forcing. Preserve the continuation floor, parking behavior,
limit checks, and GC opportunities. Some returns allocate, so skipping all
safepoints through arbitrary return chains needs a proved bound.

Next measure frame allocation in `do_apply` and direct-call paths. Introduce
frame reuse only where escape/alias analysis proves no closure or suspended
thunk retains it. Unconditional tail-call frame reuse is unsound for lazy
evaluation. Keep the allocating path for unknown cases.

Then consider an internal decoded instruction representation in `image.*` and
`interp.*`: specialize common local reads, direct calls and continuation
pairs at load time without changing the on-disk image or compiler. Account
for decoding startup cost in the 1.25-second gate. Preserve lazy argument
sharing, error order, reductions, and scheduler fairness.

Finally optimize allocator hot paths in `heap.*` only if sampling supports it:
specialize frequent frame/thunk sizes, reduce redundant initialization, and
retain required write barriers. Existing code already skips many trivial
thunks and uses allocation size-class lookup tables; measure beyond those
optimizations rather than reimplementing them.

Land each optimization separately with before/after timings. Reject changes
whose benefit disappears across repeated runs. If interpreter improvements
remain insufficient, evaluate broader JIT coverage as a separate follow-up;
do not silently redefine an interpreter-only target as a JIT benchmark.

## 5. Verification and completion

For each semantic change, run native unit tests and `dream/tests/e2e.sh`, which
compares interpreted and JIT results. Exercise verified heaps and the existing
parallel/concurrent GC modes after changes to roots or allocation. Add focused
regressions for newly affected lazy arguments, thunk sharing, effects executed
once, tail recursion, limits, and parking. Build without LLVM as well.

Finish with seed → stage 2 → stage 3 byte equality, successful self-compilation
in both modes and worker configurations, and the timing gate. Publish actual
results and remaining costs; there is no evidence yet that a small isolated
patch can achieve 1.25 seconds.
