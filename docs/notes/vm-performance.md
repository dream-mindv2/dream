# Making the VM faster

The interpreter, the collector, the scheduler and the JIT: what was measured, what was kept, and what the measurements taught. Newest findings are nearer the end of each part. Moved out of `CLAUDE.md` on 2026-09-25; the prose is as it was, so a date or number in it is as of when it was written.

## Making it faster

Three tools, all of which had to exist before any of the speedups below could
be justified:

```
dreams --time FILE       # what each stage of a compile cost, and -- for a
                         # program with macros -- what each phase of expanding
                         # them cost
dream --profile [N] IMG  # the hottest functions, by reductions
dream --stats IMG        # reductions, collections, bytes allocated and
                         # promoted, milliseconds stopped in collection, and
                         # how many functions the JIT took
benchmark/benchmark/run.sh   # seven workloads, Dream against CPython
```

`--stats` answers two questions about memory that are easy to confuse. What a
program *allocated*, by kind, is what it made; what is *live at its largest
major*, by kind -- with the object count and average size beside each -- is
what it is still holding, which is the one that decides whether a large program
fits. They rarely name the same kind, and the gap between them is where a
representation problem lives: see "A lazy value stored in a map pins the map it
was made in" below, which is a finding nothing else here could have reached.

A VM option goes **before** the image: `dream --stats build/dreams.dream ...`,
because everything after the image name belongs to the program. All three
report from `os.exit!` as well as from the end of `main`, which matters because
every tool here ends by exiting -- a number only printed on the way out of
`main` is never printed for the runs worth measuring.

A compiled function that calls another compiled function makes the same kind of
hole a native does: the callee is a machine call inside the caller's compiled
body, so its reductions are charged to the caller and it does not appear in the
profile at all. `collatz_steps` under a fused `collatz` is the example.

`--time` forces each stage where it reads the clock, because a lazy stage that
has not been forced has not run: bind and force on one line and every stage
looks free except the last. `--profile` attributes each reduction to the
function whose frame is current. Natives do not reduce, so work inside a builtin
is charged to its caller -- which is why a member of a host module (`core.head`)
can be expensive without appearing anywhere in the profile.

A profile is only worth as much as its coverage, and this one's was wrong twice
over until 2026-09-12. The reductions a *nested* force makes -- which is most of
a compile, because `dreams` forces at every stage boundary -- were counted but
never attributed, and a JIT-compiled body was invisible however hot it got. The
first of those was hiding a quarter of a self-compile in one function. Check the
two totals agree: `--stats --profile 1` prints the attributed sum and the run's
own count, and they should be the same number.

What has already been learnt from them, so it is not learnt twice:

- Building a string with `+` is quadratic. `str.concat_all` (and `join_str`,
  `repeat`, built on it) copies each piece once. The image writer works in
  strings for this reason: as a list of bytes it was four million objects to
  produce a megabyte.
- A record read more often than it is built wants to be an array. A token was a
  six-element list, and reading one was a fifth of everything the compiler did.
- A membership test over a list of names wants to be a map. The keyword check
  was a tenth of it.
- A wrapper that only passes its arguments on -- `let head xs = core.head xs` --
  is emitted as the call it stands for. `scope` records which globals are
  wrappers (see "wrappers" there); `lower` rewrites saturated calls of them.
  Roughly 13% of the reductions in a compile, and ~8% off any program that runs.
  A body that is one `get` or `set` of the parameters is a wrapper too
  (`let map_get m k d = m.[k else d]`), and a call of it is the opcode.
  The rule is deliberately narrow: one application, every parameter passed on
  exactly once, literals allowed. "At most once" is not enough -- an argument
  the wrapper ignores would never be lowered, and a lambda in that position was
  already given a function record, so the image would carry a function with no
  body.
- `strict!` is linear in *data*, not in paths, and only because objects carry a
  "deeply forced" bit -- see `AUX_DEEP_FORCED` in [dream/src/value.hpp](../../src/value.hpp).
  Before that, forcing the compiler's own tables walked shared structure once
  per path to it.
- **A list searched more often than it is built wants to be a map** -- and the
  reason this appears twice in these notes is that it was found twice. The
  keyword check was the first. The second was `ir.find_op`, a linear scan of
  the 41-entry opcode table run once per node the lowerer emits and once per
  node an image is read back from: **a quarter of a self-compile**, 20M
  reductions, gone by indexing the same table three ways at module level. It
  had been there all along and no profile had ever named it, because the
  profiler could not see the loop it ran in.
- **The JIT was charging three reductions for every one it ran.** `--stats` said
  194M reductions for a compile that does 58M, because a compiled body folded
  the whole fall in the budget into the total -- including the reductions the
  interpreter it re-entered had already counted, and again for every compiled
  frame above them. Scheduling was never affected (the budget itself was always
  right), but every number derived from the count was. The JIT does 57.8M
  reductions where the interpreter does 60.6M, which is the honest measure of
  what it saves and was not previously knowable.
- **65% of what a program allocates is the call, not the work.** `--stats` says
  so by kind: on a self-compile, frames 34% and thunks 31%; on `fib 32` those
  two *are* the program. The cheapest part of that bill is an argument that is
  one addition, subtraction or multiplication of numbers already in hand --
  `f (n - 1)` -- which used to cost a Thunk, a Blackhole write, an Indirect
  write, a continuation and a reduction to save an add. `thunk_for` computes
  those instead, and `fib 32` lost 43% of its allocation and 35% of its time.
  It needs no strictness analysis and is not one: `operand_value` reads an
  operand only when it is *already* in normal form, so nothing is forced early,
  and two fixnums added cannot raise or diverge. Division is excluded because
  `x / 0` raises, and overflow falls back to the thunk --
  `dream/tests/programs/lazy_args.dr` is what holds that line. The remaining
  thunks are calls and data, and those *would* need the analysis.
- **A native's strict arguments are evaluated, not suspended.** A native
  declares which of its arguments it forces and `resume_native` forces exactly
  those -- but the call site had already built a Thunk for every argument it
  passed, so for a strict one that Thunk was a temporary and nothing else:
  blackholed, evaluated and overwritten with an Indirect on the very next step,
  never shared with anyone, an allocation and two writes spent carrying
  `(node, frame)` four steps. The machine's own continuation carries them
  instead -- `fill_native_args` and `ContKind::NativeArgs` in
  [dream/src/interp.cpp](../../src/interp.cpp), which read the callee back out
  of its own node rather than carrying it, on the same grounds `BinRight` reads
  its right operand after forcing its left. A self-compile makes 3.35M fewer
  thunks (10.3M -> 7.0M) and allocates 6.3% less (1256 MB -> 1177 MB) for an
  identical reduction count and a byte-identical image. A strict argument of a
  shape `thunk_for` already answers without allocating -- a constant, a bound
  local, `n - 1`, an in-range array read -- is still left to it, because
  computing one of those is cheaper than evaluating it. What the change did
  *not* do is make the self-compile finish sooner, and why not is the next
  section, which is the more useful half of this entry.
- **A list searched more often than it is built wants to be a map -- a fifth
  time.** `builtins.id` and `builtins.is_builtin` were linear scans of the
  builtin table, and `opt.specialize` asked `id` three times for every `apply`
  it rebuilt. `--profile` charged it to `index_of_from`, **7.2% of a
  self-compile**; the table is indexed once at module level now (`ids`) and a
  self-compile went 186.8M -> 172.8M reductions with a byte-identical image.
- **A linear walk the machine can do is worth ten of the same walk in Dream.**
  This is the largest single lesson so far: `list.append` and `list.nth`,
  written as the obvious recursions, were between them *a third of a
  self-compile*. Both are now the operation they stand for -- `xs + ys` and
  `xs.[n else ()]` -- and `list.length` is the `len` builtin. Same complexity,
  a tenth of the constant, because a recursion pays a call, a frame and a
  couple of natives per element where an opcode pays one machine step for the
  whole walk. The same argument gave the lexer `core.str_span` and
  `core.str_upto`: a byte class scanned a byte at a time in Dream is a dozen
  reductions per byte, and a line's indentation is one operation.
  Before reaching for a recursion over a list or a string, ask what opcode or
  builtin already means it.
- **What the profile says and what it costs are different questions.**
  `--profile` counts reductions and attributes them to the function whose frame
  is current, which is what found `append` and `nth`. It says nothing about the
  runtime's own time -- collection, allocation, the dispatch loop -- so a C++
  profile is the other half. On the self-compile the split is roughly 60%
  interpreter, 20% collector, 20% natives.
- **What the runtime rebuilt per call, and no longer does.** Each of these was
  found the same way, by counting calls rather than guessing: evaluating a
  `:name` constant interned the atom by *name*, taking a mutex and hashing a
  `std::string`, forty million times in one fold -- image atoms are mapped to
  runtime ids once, at load. Calling any `std` member built a `StrObj` for the
  name and a `NativeObj` around the function pointer at *every call* -- both
  are decided by which member it is, so a process builds each once
  (`Process::native_cache`). A string or float literal allocated a fresh object
  every time it was reached; both are immutable and now have one copy per
  process. And the allocator answered "which size class?" with a binary search
  three hundred million times, where the table inverts into 512 bytes.
- **Naming a thing is a read, not an evaluation.** The machine's shape is "push
  a continuation, evaluate the part, come back", which is what makes it
  interruptible -- but a global, a builtin, a parameter, a member of an
  imported module and a constant are none of them expressions. `operand_value`
  and `callee_operand` in [dream/src/interp.cpp](../../src/interp.cpp) answer
  "can I just read it?", and the operators, `if`, `.[ ]` and every call site
  ask before falling back to the general path. `acc + x` over two bound locals
  went from five machine steps to one.
- **The JIT must be free when it is not helping.** A function LLVM had refused
  took the JIT's global mutex and two hash lookups on *every entry, for ever*,
  to be told again that it could not be compiled -- in a program that enters a
  hundred and eighty million functions that is not a slow path, it is the
  program. The tier decision is now one inlined atomic load with three states
  (compiled, rejected, still cold): see `Jit::tier`. When that was written the
  JIT was roughly neutral on the benchmarks either way, which was its own
  finding at the time; it is not true any more, and the two entries below say
  why -- what was neutral was a tier that could only compile a loop, and could
  not stay inside one for more than 4000 iterations.
- **A quarter of a self-compile was the collector, and now a ninth is.**
  `--stats` reports the pause, which is what made the question askable: 831 ms
  of a 3197 ms compile. The collection of one process now divides across a
  pool of threads -- promotion behind a claim, marking behind an atomic mark
  bit, the sweep by block -- which takes it to 304 ms and the compile to
  2700 ms. [docs/gc.md](../gc.md) is the design and the log; what is worth
  carrying away from it is that the three bugs which made the first parallel
  collector *slower* than the serial one were all the same mistake -- paying a
  synchronization cost per object instead of per batch -- and that none of
  them were visible by reading the code.
- **The JIT compiles self recursion, not just self tail calls.** A tail call
  was a loop back-edge and everything else fell to the interpreter, which meant
  the two functions closest to being fast -- `fib` and `collatz_steps` -- were
  the two it refused. `fib (n - 1) + fib (n - 2)` is now a *machine* call, which
  requires the body to be a function of its arguments rather than of a heap
  frame: such a function is emitted twice, an inner body taking its parameters
  by value and an outer entry matching `CompiledFn` that unpacks the frame once.
  `fib 32` went from 593 ms to 45 ms and from **451 MB of allocation to 4.8 KB**
  -- 100% of what it allocated was frames, and a machine call has none. Against
  CPython 3.13 that is 0.20x where it was 2.8x. `collatz` went 1949 ms -> 265 ms.
  What it costs is machine stack, which is fixed where a heap continuation is
  not, so a compiled recursion carries its depth and returns `JIT_DEEP` rather
  than overflow; `enter_function` then runs the call interpreted and stops
  offering the compiled body (`Jit::deoptimize`), because a function that
  recurses deeper than the stack allows does it on every call. `recursion.dr`
  and `runaway.dr` are what hold that line -- `sum_to 100000` and a `blow` that
  never returns both still answer exactly what they answered before.
- **A compiled loop under a nested force yielded on every single iteration.**
  This is the largest thing a wall clock ever hid here. A compiled loop spends a
  reduction per iteration and yields to the interpreter when the budget runs
  out. But a *nested* force -- `force_whnf`, which is what `strict!`, `print!`
  and `to_string` all run -- cannot hand the process back, because the natives
  above it are waiting on the C++ stack, so it spends the budget below zero and
  leaves it there. After 4000 iterations `next > 0` is false for ever and every
  iteration yields: a heap frame allocated, the interpreter re-entered, the
  compiled body re-called. A ten-million-iteration loop took 429 ms and
  allocated 240 MB to do work that takes **43 ms and 122 KB**. So the nested
  loop re-arms the budget and records the slice as spent (`Process::slice_spent`)
  instead, and `run_process` ends the slice when the force returns. Note what
  this means about every JIT number taken before it: "a tail loop is 7x faster
  than CPython", which these notes carried for months, was only true when
  nothing was forcing the loop -- and a program that prints its answer is
  forcing it. It is true now.
- **`strict!` is the native that most needed to vouch.** `VouchesForGc` exists
  because a native running unbounded Dream work underneath itself pins the heap,
  and `strict!` is how a program says "do the whole of this now" -- so the work
  under it is not a detail of one native, it is the program. Without the vouch a
  fold of ten million elements grew a **2.3 GB nursery that no safepoint could
  reach**; with it the same fold holds 300 KB and runs 8-12% faster. The reason
  it could not simply be added is `force_deep`: its map case collected the
  entries into a `std::vector<Value>` and forced them one by one, and a
  collection rewrites what it can see -- which a C++ vector is not. The entries
  wait on the value stack now, like every list `force_deep` already walked there.
- **The VM is a shared library, and that is not free.** Without
  `-fno-semantic-interposition` a compiler must assume any global function in a
  `.so` can be interposed at load time, so every cross-TU call goes through the
  PLT and none of them inline -- and the interpreter's hot path is nothing but
  cross-TU calls. Link-time optimization on top of that was measured and bought
  nothing, so it is not enabled.
- **A compiled function may call another compiled function.** The tier used to
  compile exactly one Dream-level call -- this function calling itself -- and
  refuse any function that made another, which is why a fused `collatz` was
  interpreted while `collatz_steps` beside it was compiled: the loop body is
  `acc + collatz_steps x`, and that call refused the loop. The callee is now
  emitted into the same LLVM module, as a function of its arguments, and called
  with them in registers: no frame, no thunk, no trip through the interpreter.
  `collatz` went **205 ms -> 100 ms**, from 0.30x CPython to 0.15x, and from
  20.8 MB allocated over 317 minor collections to 570 KB over 8: the frame per
  element *was* the allocation, and a machine call has none. The self-compile does
  not move at all (2617-2662 ms before, 2609-2659 after, on the same machine in
  the same minute), which is the expected answer -- `dreams` is not a numeric
  loop anywhere -- and is worth having measured rather than assumed.
  What licenses it is the rule the tier already rested on -- a function is only
  compiled when it forces every parameter on every path -- asked of the callee
  rather than invented for the occasion, so the test is `Analyzer::run` and not
  a second set of rules. `PeerSet` in [dream/src/jit.cpp](../../src/jit.cpp) is
  where that is written, along with the two things a callee may not be: a
  function that reads a slot no parameter owns (there is no frame for one), and
  a *loop* (a compiled loop stays preemptible by writing its state back to its
  frame, and a callee has none, so a callee that looped would hold its worker
  for as long as the loop ran). Recursion is fine, because the machine stack
  bounds it and `JIT_DEEP` hands the call back when it runs out. Mutual
  recursion works, because every function in the closure is declared before any
  of them is written. One consequence for `--profile`: a callee's reductions are
  charged to the function that called it, the way a native's already are.
- **A `let` is written where its name is read.** A block containing one used to
  refuse the whole function, because a `let` in a compiled body has nowhere to
  put its value: there are no thunks, and the frame's slots are registers. So
  the value is emitted at the *read* instead -- which is exactly where the
  interpreter would have forced the thunk, so nothing moves. A binding nobody
  reads is never evaluated, which is what an unforced thunk comes to; one read
  once runs at the read. What is lost is the thunk's memory of its own answer,
  so a binding read twice runs twice -- allowed only when its value is small and
  contains no call, since a duplicated call could be a duplicated recursion and
  nesting duplications is how a linear body becomes an exponential one.
  `dream/tests/programs/jit_let.dr` is what holds the laziness: a binding whose
  value divides by zero and is never named must not raise, and one named below
  something that raises first must not get there.
- **Integer `/` and `%` are two instructions, not a call.** They were the only
  arithmetic with no inline fast path, on the stated grounds that the zero check
  and the integer/float split were not worth duplicating. Measured, they were: a
  ten-million-iteration loop took 92 ms dividing where the same loop multiplying
  took 48, and the difference was one `dream_rt_arith` call per operation. Both
  are now 46-49 ms. `collatz` pays it on every step for `n % 2` and again on half
  of them for `n / 2`, and `mapfilter`'s guard is `x % 2 == 1`.
- **The interpreter disagreed with itself about one quotient.** Found by writing
  the JIT's division against `arith` and then asking both tiers the awkward
  questions. `-2^62 / -1` is `2^62`, one past the largest fixnum: `arith` falls
  through to the double path and answers a float, and `finish_binary` -- the
  two-fixnums fast path every operator takes first -- built the fixnum anyway,
  which wraps to the *smallest* one. So `x / y` gave `-4611686018427387904` or
  `4.611686018427388e+18` depending on whether both operands happened to be in
  hand at the operator, which is a property of how the argument arrived and not
  of the program. `Op::Add`, `Op::Sub` and `Op::Mul` beside it all checked
  `fixnum_fits`; `Op::Div` did not. It does now, and `Op::Mod` is written out
  separately beside it with the note that a remainder always fits.
- **A compiled function may call a host native.** The tier wrote out five
  numeric natives -- `to_float`, `to_int`, `sqrt`, `abs`, `floor` -- and refused
  the whole function for any other, which is every member of `std.native`. That
  refusal, not the strictness rule, is what kept the tier to loops somebody had
  shaped for it: a lexer reading a byte, a walk over a list through `tail` and
  `is_empty`, an ordering written over `compare` are all arithmetic with one
  call in the middle. Such a call is now made the way the interpreter makes one
  -- the callee value this process already built, the arguments on the process's
  own value stack, the native itself -- so there is one implementation of what a
  native means and both tiers reach it (`run_native` in
  [dream/src/jit_rt.cpp](../../src/jit_rt.cpp)). `native_site` in
  [dream/src/jit.cpp](../../src/jit.cpp) is the admission rule and it is four
  questions, of which the first does most of the work:
  - **Is it pure?** Purity is spelling in this language, so this is one look at
    the last character of the name. That single rule is what makes the rest
    simple: nothing has to order an effect against anything, and nothing can
    park -- every native that answers `NativeOutcome::Block` is impure, and so is
    everything reachable from the Dream work a pure one runs underneath itself.
  - **Is it saturated, of fixed arity, and four arguments or fewer?** The call is
    one helper with a fixed signature; a partial application has no callee to
    enter and a variadic native has no argument positions for a mask to describe.
  - **Does it vouch for the collector?** If so it is refused -- see below, which
    is the one measurement that changed the design.
  - **May every argument be produced here?** A position the native's strict mask
    claims is evaluated, which is what the interpreter does a step later anyway.
    A position it does not claim is *read and not evaluated*: a literal, or a
    slot handed over as it stands. That is exactly `thunk_for`, so it is not an
    eager evaluation at all, and anything of another shape refuses the call.

  What it is worth is one new benchmark row, `bytescan` -- three million bytes
  through `core.str_byte` -- which went **454 ms to 59 ms**, from 3.06x CPython
  to **0.41x**. The other six do not move. What it *costs* is at the end of this
  entry, because it is not nothing.
- **A parameter every self call hands back is carried, not evaluated.** Written
  with the above, and without it the above reaches almost nothing. The tier
  admits a function only when every parameter is forced on every path, and
  `f s (i + 1) n (acc + core.str_byte s i)` fails it for `s`: the base case
  answers `acc` and never looks at the string. That refused every loop that
  carries the thing it is walking, which is most of the loops a native call is
  good for.

  The rule was always stronger than the reason behind it. What the strictness is
  for is *eager evaluation*: compiled code evaluates a call's arguments before
  making the call, and doing that to one the callee would never force can raise
  in a program that was going to finish quietly. A parameter every self call
  passes straight back -- slot `i` to slot `i` -- is not evaluated by one. It is
  moved, which is precisely what the interpreter does with it, so it needs no
  licence. `Analyzer::carried` is the set, `Emitter::load_slot_raw` is the read
  that does not force, and `dream/tests/programs/jit_natives.dr` holds the line
  that matters: a loop carrying `1 / 0` it never looks at must not raise.

  Note who still has to be strict in everything. A **peer** is entered by a
  caller that evaluates its arguments, so nothing here is weakened for one. The
  **root** of a compile is entered from a heap frame the interpreter filled, so
  the only eager evaluation in its picture is its own self calls'. The two
  answers differ and `Analyzer::run` says why in the one place it decides.
- **A native that vouches for the collector is one compiled code may not call.**
  The measurement that decided it, and the reason the widening above is not a
  trade of memory for speed. `VouchesForGc` is how a native says "I walk a lazy
  structure underneath myself and I am written to survive a collection while I
  do" -- `str_concat`, `str_of_chars`, `str_of_bytes`, and `strict!`, which is
  impure and was already refused. A *compiled* caller cannot honour that: its
  values are in machine registers, so the call pins the heap however the native
  is written. A loop calling `str_concat` over a 300,000-element lazy list held
  **163 MB of peak heap against 45 MB interpreted**, and collected twice against
  334 times, which is the same failure "`strict!` is the native that most needed
  to vouch" describes from the other side. So `NativeDef::vouches` says which
  they are and the tier declines them. Declining costs nothing that was not
  already being paid -- a function containing one was not compiled at all before
  this -- and false is the safe default: a native wrongly left unmarked costs
  memory in one compiled loop, where one wrongly marked costs only the compile.
- **A wider tier is not free for a program that does not use it.** The
  self-compile got **2.7% slower** -- 2805 ms to 2882 ms, mean of seven
  interleaved rounds, and the new build was slower in all seven, so this is not
  the placement noise the section below describes. It compiles 19 functions where
  it compiled 10, and nine functions at the ~8.5 ms apiece LLVM charges is the
  whole of the difference: run both with the threshold set high enough that
  nothing is compiled and they are 2798 ms against 2823 ms, which *is* noise, and
  with `--no-jit` they are 2800 against 2790. `dreams` does not spend its time in
  the nine -- its own profile is AST walkers that build lists and maps, which
  this tier will never take -- so it pays for them and gets nothing back. The
  image is byte-identical and the reduction count is within 0.2%. `--stats` now
  reports the number of functions compiled from `print_stats` rather than from
  the end of `main`, which is what makes that question askable at all for a tool
  that ends in `os.exit!`.
- **An idle worker is not free, and the default was one per core.** A
  self-compile is one green process, so on a 24-core machine twenty-three
  workers had nothing to do -- and "nothing to do" was a 500us poll that walked
  every *other* worker's queue looking for something to steal, and then walked
  every worker's queue again to ask whether the system had deadlocked. One
  mutex acquisition apiece, both walks, every worker, five hundred microseconds
  apart: O(workers^2) lock operations per poll, contending with the one worker
  that was running the program. Measured across worker counts, the self-compile
  went 2.67 s at `-j 1`, 2.46 s at `-j 4`, and 2.70 s at `-j 24` -- the default
  was the worst setting available. It is now cores capped at 8, which is the
  same compromise `GcPool` already made and for the same reason, and `steal`
  answers "is anything queued anywhere?" from a counter instead of from the
  locks. Together, 3.10 s -> 2.81 s interpreted and 3.22 s -> 2.95 s by default.
  `-j` still means what it said, so a program that really does have
  twenty-four runnable processes still gets twenty-four workers.

  **The deadlock walk beside it cannot be replaced by that counter**, which is
  the part worth carrying away, because it is the obvious next edit and it is
  wrong. `take_local` moves a process out of a queue and into `runnable_` as
  two separate relaxed writes, and what makes the pair safe is that it holds
  the queue lock across both -- a walker blocks on that lock and never observes
  the instant when the process is counted in neither. A lock-free read observes
  exactly that instant and declares a deadlock in a program that is running
  fine. The comment in `take_local` had said so all along. It cost one test to
  find out: `console.dr` under `just test-races`, which failed within a few
  dozen reductions while the rest of the suite passed. If it is ever worth
  removing that walk, the move is a single counter of "queued or running"
  incremented by `enqueue` and decremented after `run_slice` -- one atomic, so
  there is no instant between two of them to observe.

- **`step_eval` is too big for the inliner, and everything it calls pays.** The
  largest single finding of the 2026-09-18 round, and it is about the build
  rather than the code. `step_eval` is one switch over every opcode, which puts
  it far past `large-function-insns` -- past which GCC will not grow a function
  at all. So every helper it calls is outlined *however small*: pushing a
  continuation is a compare and two stores, and `perf` showed it as **6.2% of a
  self-compile** in a symbol of its own. The helpers are not expensive; the one
  function they serve is too big to be allowed to absorb them, so the hot path
  pays a call per operand read and per continuation push. Raising the five
  `--param` ceilings in [dream/CMakeLists.txt](../../CMakeLists.txt) is worth
  about 4% on its own. The lesson generalizes: in a switch-per-opcode
  interpreter, read the profile for *outlined helpers* before reading it for
  expensive ones, and check `nm`/`perf` rather than assuming `inline` did
  anything.
- **A `std::vector` cannot be appended to without a call.** `ContStack` in
  [dream/src/process.hpp](../../src/process.hpp) replaced
  `std::vector<Cont> conts` because there is no way to ask a vector for "bump
  the size, I have already checked the capacity" -- so the push stayed a call
  even with the ceilings raised, and needed `always_inline` on top. Note what
  the measurement said about it though: on its own the change is **inside the
  noise floor**, because the work was always the stores rather than the call,
  and inlining relocates work rather than removing it. `step_eval` went 13.7%
  -> 16.4% as `push_back` went 6.2% -> 0. It is kept because it is the right
  shape and because it compounds with the ceilings, not because it was a win by
  itself.
- **A profile that names a symbol has not told you what removing it saves.**
  Said already under "What the profile says and what it costs are different
  questions", re-learnt the hard way here, and worth the repetition: three of
  the four changes in this round were individually indistinguishable from noise
  on a seven-round interleaved A/B, and the round as a whole is 4%. Build the
  before-binary and interleave the runs; a single before-and-after on this
  machine cannot resolve anything smaller than about 3%.
- **A `match` used to pay for every arm it missed.** The arms are tried in
  order, and a tagged-list arm that does not fit -- `[:name, n, sp]` against an
  `:apply` -- is a `type_of`, a cons test, a head read and an equality, about
  a dozen reductions, before the next arm is asked. The compiler is built out
  of matches of forty such arms, which is why ordering `infer`'s arms by
  frequency was once worth 15%. Now a `match` with three or more tags starts
  with `switch_head` / `switch_atom` (docs/bytecode-format.md), which jumps into
  the ordinary chain of arm tests at the first arm that could match that tag.
  A twenty-arm match reaching its last arm went 1645 ms -> 350 ms, and the
  self-compile 176.0M -> 156.0M reductions (-11.4%), 5733 -> 5394 ms on three
  interleaved rounds (-5.9%). It is `dispatch` in
  [dreams/src/lower.dr](https://github.com/dream-mindv2/dreams/blob/main/src/lower.dr), and two properties make it safe. Every
  target is a suffix of the chain, so each arm is lowered once and a tag arm
  that does not match after all falls through as before. And `switch_head` is
  only emitted where the chain would have forced the cell's head first anyway.
  `dream/tests/programs/match_switch.dr` holds the orderings: a catch-all
  between tags, a head that raises, and a first arm that forces nothing. It
  needs no type: the patterns say which tags they take. The JIT does not
  compile a switch yet (`op_is_supported`), so a function containing one stays
  interpreted.
- **Where the self-compile now stands.** 3.10 s -> **2.32 s** interpreted,
  measured `--no-jit` with default workers against a VM built from the commit
  before any of this. **Do not trust this row as an absolute.** Re-measured
  2026-09-19 on the same machine and a `just vm` build, a self-compile is
  7 s -- and the two numbers were never measuring the same program, since
  `dreams` and `std` have both grown a great deal since. What the row is good
  for is its ratio; what it is not good for is deciding that today's compile
  has regressed. Measure the commit before, in a worktree, with the same
  binary -- `git worktree add /tmp/base HEAD` -- which is how the two
  image-per-call bugs below were found rather than argued about. The order of the wins is the opposite of where the effort
  went: PGO 12%, the inliner ceilings 4%, the scheduler's idle workers 9%, and
  every source-level interpreter change together under 1%.

## A spurious deadlock, and why it hid behind a slow walk

Found 2026-09-18 by `just test-races` failing on `console`, reported against a
process the dump described as `finished mode=halted` after 99 reductions --
which is not what a deadlocked process looks like, and was the clue.

`deadlocked_` is sticky. The detection happened at *startup* and the dump was
printed at exit, after the program had run correctly to completion. The window
is one line of `cli.cpp`:

```
sched.start();      // workers begin looking for work
sched.enqueue(root);
```

Between those two statements the root process is live, nothing is queued or
running, and every worker is idle -- which is precisely the condition the
deadlock check tests. Enqueueing first closes it, and there is no other instant
like it, because every later `enqueue` happens while something is already
running.

Two things kept it hidden. The check only runs when `idle_workers_ + 1 >=
workers_.size()`, which with one worker per core on a 24-core machine almost
never held -- capping the default at 8 made it hold routinely. And the check
used to walk every worker's queue under its lock, which took long enough that
the window had usually closed by the time it finished. Both of those are
accidents, not protections: **the check was always wrong and was being saved by
being slow.**

The walk is gone, and not because of speed. It could not be made correct: a
worker re-enqueueing its process pushes it onto a queue and only then
decrements `runnable_`, so a walk that has already passed that queue goes on to
read `runnable_` after the decrement and concludes that a healthy program is
deadlocked. Two counters cannot be read without observing a transition between
them. `Scheduler::active_` is one counter -- queued *or* running -- incremented
by `enqueue` and decremented when a slice ends without re-enqueueing, so a
slice that hands its process straight back increments before it decrements and
the count never dips. The check is one load.

**Later, 2026-09-30: there was another instant like it.** `eval_image!` and
`call_image!` run a compile-time expression on a scheduler of their own, and
`run_compile_time` still started it before enqueueing. With one worker the
check's `idle_workers_ + 1 >= workers_.size()` always holds, so the only thing
between it and a false deadlock was the worker's 500us wait: a starting thread
descheduled for longer than that on a loaded machine lost the race, and the
`comp` answered `:comp_failed the compile-time expression failed`. It never
showed under `just test`, which runs one compile at a time; `mind test` runs
them in parallel and hit `dreams.contracts` about one clean run in two.
Twenty-four copies of `dreams/tests/contracts.py` at once on four cores failed
16 times before and none in 72 after. The order is now `Scheduler::run`, and
the command line, the embedding API and compile-time evaluation all call it,
so there is one place to get it right rather than three.

## The tier's eager arguments can change *which* error a program raises

Found 2026-09-14, confirmed against the VM as it was before any of that day's
work, so it is old and it is not a consequence of anything above. It is written
down because the file header states the rule in a way that reads stronger than
the rule is.

Compiled code evaluates a self call's arguments before making the call, and what
licenses that is the admission test: every parameter is forced on every path, so
an argument evaluated here is one the callee was going to force anyway. That
keeps the guarantee the header claims -- a program that finishes quietly still
finishes quietly. It does **not** keep the argument being forced at the same
*point*, and when more than one error is reachable that decides which one the
program gets:

```
let rec ordered i n acc =
    if i > n { acc } else { ordered (i + 1) n (if i + () > 0 { acc } else { acc }) };
ordered 1 20000 0
```

`acc` is forced on every path, so the loop is compiled. Interpreted, the
accumulator is a chain of twenty thousand suspended `if`s and forcing it
evaluates the outermost condition first, so the error names `integer 20000`.
Compiled, each iteration evaluates its own condition, so the error names
`integer 32` -- the iteration the JIT threshold happened to fall on. Both raise,
both raise a `:type_error`, and the payloads differ. The same argument says a
program whose accumulator *diverges* where an earlier one raises can diverge
under one tier and raise under the other.

Not fixed, and the reason is the benchmark. The rule that would fix it is "the
argument must be forced before anything else that can raise", and `sum`'s fused
loop fails it -- its condition `i >= hi` is a comparison, which can raise a type
error, and it runs before the accumulator is forced. Refusing that would undo
the whole of the fusion result. This is the same trade every lazy language makes
under "imprecise exceptions"; what makes it worth writing down here rather than
shrugging at is that Dream has typed errors and `catch`, so a program *can* look
at which one it got.

## Two collector bugs that presented as a segfault a long way from home

Both were found in 2026-09 by a crash the fuzzer reached and nothing else did,
and both are the same species: a reference the collector could not see or could
not fix. Neither faults where the mistake is, which is what made them expensive.

**A pointer must not span a force.** `force_deep` walks a list in place. The
cell waits on the value stack precisely so that a collection can find it *and
rewrite the reference* -- the object moves. So a `ConsObj*` taken before a force
names the block the cell moved out of, and storing through it writes into
whatever that block was recycled into. `ConsObj::head` and `ThunkObj::node`
share an offset, so the usual shape of the damage is a thunk whose node field is
the low half of a pointer; the machine jumps to it some thousands of reductions
later and dies somewhere unrelated. The cell is now re-read from the stack at
every single use (`cell()` in [dream/src/interp.cpp](../../src/interp.cpp)), and
the array case beside it does the same.

**An object born old is remembered at birth.** An allocation over
`kMaxClassSize` is born in the old generation, and what its caller fills it with
is young -- `make_array` hands back an array and the caller writes the elements
straight in. That is an old-to-young edge made by no store the write barrier
ever sees, so the next minor collection does not scan the array and frees the
elements it cannot find; a 3000-cell array of unforced thunks comes back holding
whatever the nursery has since put there. Such objects are pushed onto the
remembered set in `alloc_bare` ([dream/src/heap.cpp](../../src/heap.cpp)). One
entry covers the whole object, and only allocations large enough to tenure pay
it. Nothing can collect between the allocation and the fill -- a collection runs
only at a safepoint, and there is none inside a reduction -- which is what makes
filling one safe without a barrier at every fill site.

The verifier now names the object's type and what it was reached *from* when it
reports a young object surviving a minor, because "a minor collection left a
young object reachable" without a path is a fact you cannot act on.

## Fixed: compiled code forcing a long thunk chain crashed

A JIT-compiled function that forced a long chain of suspensions died with
**SIGSEGV** where the interpreter raised `:stack_overflow`. Found 2026-09-13,
reproduced against the JIT exactly as it was before it learnt self recursion, so
it was never new -- nothing here had looked for it before. Fixed the same day;
`dream/tests/programs/force_chain.dr` is what holds the line.

```
import std.console;
let rec loop_f f i n acc = if i > n { acc } else { loop_f f (i + 1) n (f acc i) };
let main! = { console.print! (loop_f (fn a b -> a + b) 1 10000000 0) };
```

`acc` is lazy, so this builds ten million suspended applications rather than
adding anything -- the trap `list.fold_strict` exists to avoid, and its note in
[mind/std/src/list.dr](https://github.com/dream-mindv2/std/blob/main/src/list.dr) explains it. Forcing that chain is supposed
to raise at `DREAM_MAX_DEPTH`, and with `--no-jit` it always did, exactly:

```
dream: uncaught error: <error :stack_overflow recursion too deep: 4194305 pending frames>
```

With the JIT it segfaulted. The reason was one line of the machine: the
interpreter forces a slot by pushing a **continuation**, so its recursion is
heap and its limit is a number it can check, while compiled code forces a slot
by *calling* -- `load_slot` -> `force` -> `dream_rt_force` -> `force_whnf`,
which runs a whole nested machine loop, which enters the compiled body again for
the next link. One C++ frame per link of the chain, against an 8 MB thread
stack. The lambda was the whole trigger: `fn a b -> a + b` is arithmetic over
two strict parameters, which is precisely what this tier compiles.

The fix is not the depth argument that bounds compiled *self* recursion, because
the recursion here goes out through the runtime and back in. It is the same move
`Jit::deoptimize` makes, on a different trigger: `Process::force_nest` counts
the nested `force_whnf` loops on the process's machine stack, and
`enter_function` stops offering the compiled tier past
`kMaxForceNestForCompiled` (256). From there the interpreter handles the rest of
the chain, pushing continuations instead of C++ frames, and the walk finishes on
the heap where its limit can be enforced. Interpreted code never nests forces
deeply -- each one returns before the next -- so the bound costs every healthy
program one compare on function entry and nothing else.

The bound is free, which was worth measuring rather than asserting because
`enter_function` runs on every single call: one `uint32_t` load and a compare,
folded into the branch that was already testing whether the JIT exists. The six
benchmark workloads A/B'd against the same build without it on the same machine
came back 44/51/216/30/82/25 ms before and 43/49/217/29/82/25 after -- inside
the placement noise floor in "Two things that will lie to you about a change to
the interpreter" in every row.

What makes the test cheap enough to keep is `force_chain.env`: a small
`DREAM_MAX_DEPTH` and a 200,000-link chain, which is comfortably past the
roughly 16,000 links eight megabytes of stack allowed and finishes in a fraction
of a second. Note that the crash was *not* catchable -- `try!` did not see it --
which is why the e2e harness running every program under both tiers is the right
place for it: what it asserts is that the two tiers agree, and a segfault on one
side is the loudest possible disagreement.

## Measured, and not kept

- **A larger major-GC growth factor.** `gc_threshold_ = live * 3` makes the
  majors a geometric series whose last term dominates, so widening the ratio
  should mean less total marking. At 3, 4, 6 and 10 the self-compile does 7, 7,
  5 and 4 majors -- and the wall clock does not move (2.68 s, 2.75, 2.74, 2.74,
  with 3 nominally best). Peak heap does not improve either. This is the third
  GC policy lever measured here and the third that does nothing; see also the
  nursery note below. Whatever the collector costs this workload, it is not the
  number of collections.
- **`-march=native -mtune=native`.** 2.775 s against 2.737 for the same source
  without it -- nominally *worse*, certainly not better. The interpreter's hot
  loop is pointer chasing and indirect branches; there is no vector width or
  new instruction for it to find. Not worth the non-portable binary.
- **Turning the concurrent mark off** (`DREAM_GC_CONCURRENT=0`): 2.325 s
  against 2.360. Inside the noise floor, so the concurrent mark is neither
  earning its 7.8% of CPU samples nor costing anything measurable in wall
  clock. Left on.

- **The outer loop's run of Return steps, added to the nested loop.**
  `run_process` skips its safepoint over a run of Return steps, which is worth
  having because more than half of all steps are Returns. The nested loop
  (`force_whnf`) has no such run, and since that is the loop a program actually
  lives in, adding one looked free. It is 8% *slower* on a fold -- identical
  reductions, identical collections, identical bytes -- because a Return under a
  fold rarely comes in a run, so the extra loop condition is paid per step and
  the skipped checks are never skipped.
- **A bigger nursery for a program that allocates hard.** A fold that allocates
  4 GB collects tens of thousands of times with the default 64 KB nursery, so a
  floor on it looked obviously right. Measured at 64 KB, 256 KB, 1 MB, 4 MB and
  16 MB, the collection count falls from 70,459 to 276 and the wall clock does
  not move at all (4.0-4.4 s either way). The collector is 2-4% of that
  workload; what it costs is not what it collects.

## Two things that will lie to you about a change to the interpreter

Both of these were found on 2026-09-13, by a change that plainly does less work
and did not finish sooner -- the native arguments above. Neither is about that
change; both are about what a wall clock on this machine can and cannot resolve.

**Code placement moves a benchmark by 3%, in either direction.** `step_eval` is
one switch over every opcode, and adding a case to it re-allocates registers and
re-pads the whole function. Adding the native-argument path moved `fib` by 3% --
`fib`, which never reaches the new code, and whose `thunk_for` is a
byte-for-byte identical instruction stream in both builds. So 3% is the noise
floor for *any* edit to that function, and a CPU-benchmark result smaller than
that is about where the code landed rather than what it does. Diff the
instruction streams before believing one:
`objdump -dC build-dream/lib/libdream.so`.

**The collector's thresholds are geometric, so allocating less can mean marking
more.** `gc_threshold_` is refitted to `live * 3` after every major, so a run's
majors form a geometric series and the last one dominates everything before it.
Allocate 6% less and every crossing moves: on the self-compile the series went
from `60K, 253K, 889K, 3.9M, 13.9M, 41.9M, 111M` live to
`60K, 277K, 1.1M, 4.8M, 19.9M, 68.9M, 143M` -- seven majors either way, 238 MB
marked instead of 172 MB, and the last two majors getting almost no overlap out
of the concurrent mark where the shorter series got 24 ms. That is +3% on the
self-compile's wall clock and +250 ms of helper-thread CPU, and it is phase, not
cost: the same two binaries compiling `lucid` are within 1% of each other, and
the same self-compile with the collector's threads and the JIT out of the
picture (`DREAM_GC_THREADS=1 dream --no-jit -j 1 ...`) is *faster* -- 2% on the
best of seven runs, 1% on their mean, which is about as much as this machine can
resolve. `DREAM_GC_TRACE=1` prints the series, and it is the first thing to look
at when something that does less work takes longer.

## Where the remaining time and memory are, and the plan

Measured 2026-09-12, after the round of work that took the self-compile from
2.78 s to 2.02 s and its peak RSS from 859 MB to 499 MB. Both numbers now have
one dominant cause each, and neither is the collector.

*(Both halves of this section were overtaken on 2026-09-22, and by the same
tool. "Memory is fragmentation" was measured against a live set nobody could
break down -- 57% of it turned out to be map branches nothing could reach but
an unforced value, and removing a third of it moved the wall a large program
hits by 50%. Read "A lazy value stored in a map pins the map it was made in"
before acting on anything below.)*

**Memory is fragmentation, not policy.** The last full collection of a
self-compile finds 182 MB live in 386 MB of blocks, and only 422 of 5848 blocks
are entirely empty -- a 64 KiB block holds around 1300 objects and 45% of them
survive, so no block ever empties and sweeping cannot hand anything back. Four
different trigger policies were tried and all landed within 10% of each other;
`--stats` says why, by printing what was *allocated* at the high-water mark
alongside what was held (88% of it). The only thing that reaches a hole is a
collector that moves: compaction, or the opportunistic evacuation of sparse
blocks. See [docs/gc.md](../gc.md), "What is left: the holes".

**Time is the call, not the work.** `--stats` breaks allocation down by kind:
frames 40%, thunks 20%. Three fifths of everything a compile allocates is the
machinery of calling and suspending, not the data -- and the two have traded
places as each round of this work removed thunks and left the frames a larger
share of a smaller total. Counting the thunks by what they suspend, before and
after the two rounds so far:

| suspended node | count | what it is |
|---|---|---|
| `apply` | 8.2M -> 5.1M | an argument that is a function call |
| `get` | 2.6M -> 0.98M | an argument that is `c.[k]` or `c.[k else d]` |
| everything else | 1.2M -> 0.92M | `set`, `global`, `list`, `add`, `block`, `sub` |

`DREAM_PROBE_THUNK=1` with `--stats` is what prints that table: suspensions by
the kind of expression suspended. `--stats` alone says how much of a program's
allocation is thunks; this says what they are *of*, which is the question that
decides what to do about it.

So the order of work, most valuable first:

1. ~~**`get` arguments that cannot fail.**~~ **Done.** An in-range read of an
   array in normal form is a load, and the already-built prefix of a list is a
   pointer chase; `thunk_for` does both instead of suspending them. Neither
   needs a strictness analysis, for the same reason the arithmetic case does
   not: `operand_value` reads an operand only when it is already in normal
   form, and the list walk follows only cells that are already in normal form
   and gives up the instant it would have to force one -- which is what keeps
   an infinite list infinite. The depth is capped at eight because the walk
   runs whether or not the callee looks at the argument. `get` thunks fell from
   2.6M to 0.98M and a self-compile lost 6% of its allocation.
   `dream/tests/programs/lazy_args.dr` holds the line on all of it.
2. **NOT a strictness analysis.** This was the obvious next step and it was
   written, measured and thrown away; the numbers are here so it is not written
   again. Two versions were built against the image at load -- no compiler
   change and no image-format change are needed, because the VM has the whole
   IR -- and both were counted against a self-compile without changing
   behaviour:

   - The **sound** one answers for a single parameter: the one forced before
     anything else that could raise. That restriction is what makes acting on
     it invisible, because a caller evaluating such an argument raises exactly
     the error the callee would have raised, in the same place. It would
     remove **21,271** of the 8.2M `apply` thunks. 0.26%.
   - The **unsound** one is the classic set -- every parameter the body forces
     if nothing raises -- which is what most lazy languages ship under
     "imprecise exceptions" and which Dream cannot, because it has typed errors
     and `catch`. It would remove **44,912**. 0.55%.

   Both are worthless here, and the reason is visible once the thunks are
   attributed to where they are *made* rather than to what they suspend:

   | made at | count | |
   |---|---|---|
   | ~~a strict argument of a **native**~~ | ~~3.09M~~ | **done** -- the mask said so all along |
   | a `let` bound in a block | 2.14M | `let x = f y` |
   | an argument of a saturated call to a known function | 1.48M | the only place an analysis would apply |
   | a list, array or map literal's element | 0.54M | |

   The one worth taking was the first, and it needed no analysis at all: a
   native declares which arguments it forces, and the machine built a Thunk for
   one anyway, which `resume_native` then immediately blackholed, evaluated and
   overwrote. That is done -- see "A native's strict arguments are evaluated,
   not suspended" above. It was worth the 6% of allocation predicted here and
   3.35M of the thunks, which is more than this table's 3.09M because the
   `global`, `list` and `block` shapes turn up in that position too. It was
   worth none of the wall clock, for reasons that are not about the change and
   are written up under "Two things that will lie to you".

   What is left in the table is what an analysis would have to reach, and the
   measurements above say it would reach almost none of it.

3. **Frames, which are now 40%.** **Half done, from the other end.** A frame
   that never escapes its call could live on a stack rather than in the heap.
   That still needs escape analysis in the interpreter and is still the largest
   piece of the three -- but the JIT reaches one important case of it without
   any: a function it compiles has no frame at all, because its slots are
   registers, and now that a self call can be a machine call rather than a loop
   back-edge, a whole recursion can run with no frame anywhere. `fib 32` went
   from 451 MB to 4.8 KB for exactly that reason. What is left is every call the
   JIT cannot take, which is every call that allocates -- see the next section.

## Where it stands against CPython, and what the remaining gap is made of

`benchmark/benchmark/run.sh` runs the seven workloads of `main.dr` against the
transliteration of them in `bench.py`, best of `--repeat` runs each, and prints
the ratio. On this machine, 2026-09-15, best of five against CPython 3.13 (below
1.00x is Dream ahead; "was" is the same measurement before any of the work in
this section, "pre-fusion" is after the JIT work and before deforestation,
"pre-calls" is before the tier learnt to call another compiled function, and
"pre-natives" is before it learnt to call a host native):

| workload | dream | python | ratio | pre-natives | pre-calls | pre-fusion | was |
|---|---|---|---|---|---|---|---|
| `sum` | 42 ms | 310 ms | **0.14x** | 0.15x | 0.15x | 13.0x | 14.3x |
| `fib` | 40 ms | 214 ms | **0.19x** | 0.19x | 0.19x | 0.19x | 2.62x |
| `collatz` | 100 ms | 715 ms | **0.14x** | 0.15x | 0.30x | 0.33x | 2.83x |
| `mapfilter` | 22 ms | 29 ms | **0.77x** | 0.79x | 0.75x | 13.1x | 14.8x |
| `pi` | 28 ms | 230 ms | **0.12x** | 0.13x | 0.13x | 5.8x | 6.55x |
| `bytescan` | 59 ms | 144 ms | **0.41x** | 3.06x | -- | -- | -- |
| `strbuild` | 68 ms | 0.9 ms | 75x | 76x | 70x | 82x | 72x |

*(2026-10-03: `strbuild` is 32 ms and 37x since "Building ahead" below.)*

Six of the seven are ahead of CPython, and the seventh is a different algorithm
on each side. Read the six with the noise floor in mind: everything but
`bytescan` moved by less than the 3% that "Two things that will lie to you" says
a recompile of `step_eval` is worth on its own, and the `pre-natives` column is
an adjacent run of the same images under a VM built from the commit before.

`bytescan` is the newest row and the one the native-call work is judged by:
three million bytes read through `core.str_byte`, which is a wrapper the
compiler rewrites into `std.native`'s, so the loop body is one native call and
an addition. Before, a single such call refused the function and the whole scan
ran interpreted. Two things about how it is written are worth copying, because
both were found by getting them wrong: its accumulator is guarded (`if acc < 0`)
because otherwise it is three million suspended additions and forcing them runs
the machine out of depth -- the `fold_strict` trap -- and that guard sits
*after* the bounds test, because in front of it there is a path where `i` is
never looked at and the tier cannot admit the function at all.

`mapfilter` is the one row whose number is not about the loop. 500,000 elements
at 22 ms is 44 ns each, where `sum` runs at 5 -- and the difference is that the
JIT spends about **10 ms per function it compiles**, which at this size is most
of the measurement. Timed at ten times the size the per-element cost is 15 ns,
which is the loop's own: an `x % 2` and a guard. Measured by running the same
workload twice in one process, where the second run costs 6 ms against the
first's 25. That 10 ms is not the optimization pipeline -- `O1` and `O2` come
out the same -- so it is instruction selection and object emission, and nothing
short of a cheaper code path through LLJIT would move it.

`strbuild` is the one that went the wrong way, and it is worth knowing why: it
is the workload here whose data is *live* rather than garbage -- a 400,000-cell
list held from one end while a native walks it from the other -- so letting
`strict!` collect (above) buys it a 2 MB smaller peak for 20 ms of copying. The
folds, whose every cell dies the instant it is read, get 8-12% back for the same
change. That is the generational bet losing one hand and winning three. Its 82x
should not be read as a runtime comparison at all, either way: `"hello " * n` in
Python is a single C memcpy loop, and the Dream line beside it builds a
400,000-element list and concatenates it. Same answer, different algorithm.

The decomposition that made the case for fusion is kept here because it is what
predicted the result, and it predicted it closely. Before fusion the split was
two, not six: `fib` and `collatz` were recursion over numbers that the JIT
compiled, and the other four folded over a list `list.range` built one cell at a
time, an order of magnitude behind. Timing the same ten-million-element loop
with one thing added at a time, per element:

| | ns | what the step adds |
|---|---|---|
| a compiled loop | 4.6 | nothing: registers and a back-edge |
| an interpreted loop | 164 | frames, dispatch, the strictness dance |
| + a closure call | +52 | one more frame, entered and returned |
| + a lazy list | +256 | a `range` frame, a cons, a tail thunk, forcing it |

So the lazy list is three fifths of it, and the interpreter is the rest. Neither
is a constant factor anyone can tune away: producing one cons cell per element
costs a frame, a cell and a suspension **because that is what the program says**,
where `for x in range(...)` in CPython costs an increment in C. Closing it needed
one of two things, and both were projects rather than edits:

- ~~**Deforestation**, in `dreams`~~ -- **done**, the next section. A fused
  `sum` runs the top line of that table rather than the bottom one: 4.6 ns an
  element instead of 420, which is the whole of the distance between 13.0x and
  0.15x.
- **A JIT that can allocate** -- done; see the section of that name. The tier's scope was "the strict numeric spine"
  and the reason is not ambition, it is the collector: a compiled frame keeps
  its slots in registers the collector cannot find or rewrite, which is what
  `PinsTheHeap` says. So compiled code may not allocate, and until fusion
  removed it, a fold allocated three objects an element. What a compiled loop
  cannot do is *build* -- a list, a string, a map -- because building allocates
  and allocating means the collector will eventually run under it. The plan
  below is the response to this bullet, and it deliberately does **not** walk
  the machine stack with stack maps: a compiled frame is already a root (the
  interpreter put it on the value stack), so the move is to spill slots into
  that frame at every point where the collector can run and read them back
  after. That works with the concurrent collector and its helper threads, which
  a stack-map walk of the running thread's stack would not.

## List and map code in the JIT

Done 2026-09-25. Until then the tier compiled numbers: a function was admitted
only when every parameter was forced on every path and every op was in a short
numeric table, and that refused essentially every list or map function --
`cons`, `tail`, the empty test and `.[ ]` were not in the table at all, and any
argument the callee might not force (a builder's accumulator, the tail of a new
cell, a value stored under a key) refused the whole function. A self-compile
made 40 functions hot enough to compile; it compiles 422 now. What changed, in
[dream/src/jit.cpp](../../src/jit.cpp) unless it says otherwise:

- **The data ops are compiled.** `ListIsEmpty`, `ListTail`, `Cons`, `Get`,
  `Set`, `MakeList`, `MakeArray`, `ConstStr` and `Global`, and the three natives
  `match` lowers a list pattern to (`match_is_cons`, `match_head`,
  `match_tail`). The reads a list loop makes every iteration -- the empty test,
  a cell's head and tail, an in-range array element -- are written inline; the
  rest (a map lookup, a walk, every error) is `dream_rt_get`/`dream_rt_set` in
  [dream/src/jit_rt.cpp](../../src/jit_rt.cpp), which are `container_get` and
  `container_set` with the machine's continuation replaced by a nested force, and
  say the same words when they raise.
- **Strictness is a decision per argument, not a condition of admission.** A
  self-call or peer argument is evaluated where the callee forces that parameter
  on every path -- the standard strictness fixpoint, assume all and drop what the
  assumption does not make strict, in `Analyzer::run` under "Which arguments are
  evaluated" -- and is in a *lazy position* everywhere else.
- **A lazy position is built or suspended, never evaluated.** Built in place
  when that cannot be observed: a literal, a slot as it stands, a cell or a list
  literal (a cell is a value and building one cannot raise), and exactly the
  shapes `thunk_for` computes. Suspended otherwise, against a frame made on
  demand from what the slots hold right now -- which is the frame the
  interpreter would have been running this iteration in, so the thunk means what
  its thunk would have meant, and needs nothing from this file to run. Before a
  call's first back-edge the interpreter's own frame for the call *is* that
  frame, and is adopted rather than copied; that took a self-compile from 3.7M
  frames made to 0.6M. "Lazy positions" in jit.cpp is the design.
- **A call the tier cannot make itself, the machine makes.** A closure in a
  slot, a function the tier did not take, a partial application: `dream_rt_apply`
  applies it in a nested loop with its arguments lazy and forces the answer. That
  is what compiles a fold that calls the function it was given. An *impure*
  callee -- which a pure function can only have been handed through a
  parameter -- is never run from here: compiled code orders no effects and a
  call under it that parks is retried from the top, so the whole call is given
  back (`JIT_BAIL`) before anything has happened.
- **`e + f ..` recursion is a loop.** `1 + len (tail xs)` used to be a machine
  call a level and, past a few thousand levels, `JIT_DEEP` -- which gives the
  function to the interpreter *for good*. It is a loop carrying the pending sum
  now, and so is `let rest = f ..; .. rest` read as the answer (`filter`
  skipping an element). The soundness of the first is the part to read before
  touching it: the interpreter adds innermost first and a partial sum that
  leaves the fixnum range turns every later addition into a float addition, so
  the loop keeps the least and greatest prefix sum and proves at the end, in
  128 bits, that every suffix sum the interpreter would have made fits. Anything
  else -- a float, a string, a failed proof -- gives the call back (`JIT_BAIL`)
  to be run from its frame, which answers what the interpreter answers because
  nothing compiled code did was an effect. "Recursion that is a loop" in jit.cpp.
- **Hot functions compile on a background thread.** See the finding below; a
  threshold of 1, or `DREAM_JIT_SYNC`, keeps them synchronous.

What it is worth, 2,000,000-element lists, against the VM before any of it
(`JIT on`, best of a few runs, KB is what the call allocated):

| | before | after |
|---|---|---|
| `sum_list` (empty test, head, tail) | 457 ms, 177 MB | **35 ms, 67 KB** |
| `sum_match` (the same through `match`) | 173 ms, 345 KB | **47 ms, 104 KB** |
| `len_rec` (`1 + len (tail xs)`) | 291 ms, 112 MB | **30 ms, 1 KB** |
| `total` (`x + total rest` through `match`) | 922 ms, 208 MB | **45 ms, 3 KB** |
| `upto` (`upto (i+1) n (cons i acc)`) | 567 ms, 364 MB | **95 ms, 99 MB** |
| `rev` | 568 ms, 394 MB | **123 ms, 80 MB** |
| `evens` (a hand-written `filter`) | 595 ms, 240 MB | **235 ms, 192 MB** |
| fold calling a lambda | 422 ms, 272 MB | **110 ms, 80 MB** |
| `list.map` then a sum | 973 ms, 557 MB | **452 ms, 336 MB** |
| 200,000 map inserts | 380 ms, 367 MB | **127 ms, 194 MB** |
| 200,000 map reads | 111 ms, 42 MB | **56 ms, 9 KB** |

The consumers now allocate nothing at all, and a builder allocates its cells
and nothing else: `upto`'s 99 MB is 48 MB of cells and the 44 MB `--stats`
counts again when a minor collection promotes them, because the list is live.
What is left in the last four rows is the program's own laziness -- `map`
suspends `f x` and the rest of the list because the program says to -- and the
frame the interpreter makes for each call of a closure.

The self-compile: **9.35 s -> 7.37 s** (three interleaved rounds, same machine),
173M -> 155M reductions, 3.25 -> 3.09 GB allocated, 481 -> 389 MB promoted, and
a byte-identical image; `benchmark/benchmark/run.sh` moves `collatz` 134 ->
105 ms and `strbuild` 85 -> 65, and the other five inside the noise.

**Four findings, each of which cost more than the change that exposed it:**

- **Four hundred compiles cost more than they buy, unless nobody waits for
  them.** The first version made the self-compile 40% *slower*: `DREAM_JIT_TRACE=1`
  (every compile, and what LLVM charged for it) said 4.7 s of LLVM for 425
  functions, which was the entire regression -- the compiled code itself was
  already ahead. Cheaper code generation only halves it (`CodeGenOptLevel::None`
  2.6 s, `O1` 4.2 s). What removed it is that *when* a compiled body arrives is
  not observable -- the interpreter runs the function exactly as it did while it
  was cold -- so the compile moved to a thread of its own (`Jit::compile_worker`)
  and the program never waits. A self-compile is one green process on a machine
  with more cores than that. The cost is paid by short programs: a loop that is
  hot for 30 ms runs interpreted while its compile finishes.
- **A binding read both evaluated and lazily was computed twice.** A `let` is
  written where its name is read, which for an evaluated read computes it, and
  for a lazy read suspended it *again* -- so whoever forced the suspension
  computed it a second time. `dreams` lowering every function body twice is
  what it looked like: 24M extra reductions, found by bisecting the compiled
  functions down to `lower.set_func_body`'s caller with `--profile` showing
  `emit` and `lower_expr` exactly doubled. A `let` that calls anything is now
  remembered per iteration and shared with its lazy reads, as the interpreter's
  one slot thunk is ("A binding computed once" in jit.cpp).
- **A yield must not write the frame a suspension may read.** A yielding loop
  used to write its loop state back into the frame it was entered with. Once
  that frame is adopted as a snapshot, a thunk made in the first iteration reads
  its slots when forced, and the write-back hands it the *last* iteration's
  accumulator -- which is the thunk itself: `loop value depends on itself`. A
  yield returns a new frame now, and `enter_function` resumes in it.
- **Every alloca belongs in the entry block.** The emitter made its
  out-parameters where it needed them, and an alloca anywhere but the entry block
  is a *dynamic* one that grows the machine stack each time it is reached. A
  numeric loop reaches those blocks almost never. A list loop reaches `force`'s
  slow path every iteration, and would have grown the stack by a word an
  iteration for as long as it ran.

**What it does not do yet, and where the rest is:**

- *A call to a closure still makes the interpreter's frame.* The fold row above
  is 80 MB of the lambda's frames. Taking them away needs a compiled function to
  be callable with its arguments in registers from *outside* its own module --
  an entry per function taking values rather than a frame -- which is a registry
  the tier does not have.
- *Only `+` accumulates.* `*` has the same shape and a harder proof (suffix
  products), `-` is not associative, and `f .. + e` evaluates `e` after the
  recursion returns, so its errors would move. List `+` (`[x] + f rest`) is
  associative and falls back today because its left side is not a fixnum.
- *The depth at which a recursion overflows can differ between the tiers.* The
  loop gives a call back at `DREAM_MAX_DEPTH` pending levels, and the
  interpreter, which can spend more than one continuation a level, may overflow
  sooner. A recursion deep enough to overflow one tier and not the other is the
  one difference this admits; `runaway.dr` holds the case that matters, a
  recursion that never ends, to raising the same error in both.
- *An accumulating function is not a peer.* It gives a call back by restarting
  it from its frame, and a peer has none, so a compiled caller reaches one
  through `dream_rt_apply`.

`dream/tests/programs/jit_lists.dr` is what holds all of it to the interpreter's
answers: every function above compiled from its first call (`jit_lists.jit`),
and the awkward cases each written out -- a float in a sum, a suffix sum that
overflows though the total does not, an error from the middle of a recursion, a
lazy element that must never be evaluated, an infinite producer, a missing key
with and without a default.

**One bug this found that was not the JIT's.** `Scheduler::finish` notified
`wait_for_all` without the mutex it waits under, so a process finishing between
the main thread's test of `live_` and its going to sleep lost the wake-up and
the VM waited for ever on a finished program -- every worker idle, nothing left
to say so. Once in a few hundred runs on a loaded machine; the full e2e suite
under a watchdog that took `gdb` backtraces is what caught it, in an `--no-jit`
run. `Scheduler::notify_done` is the fix.

## Building ahead: a producer makes a run of cells a call

Done 2026-10-03. After "List and map code in the JIT" every list *consumer* ran
compiled and allocated nothing, and every *producer* still ran one cell per
call. `replicate n x = .. cons x (replicate (n - 1) x)` suspends its tail
because the program says to, so each cell cost a cell, a thunk, the frame the
thunk reads and a trip through the interpreter to force it: `strbuild` was
116 bytes and a call for 24 bytes of data, and it was the one row of the
CPython table nothing had moved.

The tail can be evaluated early wherever nobody could tell, and in a pure body
the only ways to tell are an error, a loop that never ends, and work or memory
nobody asked for. So a compiled `cons h (f ..)` answer takes the loop's
back-edge instead of suspending `f ..`, and every iteration after the first
runs *ahead*: it may do inline arithmetic, tests and reads on values in hand,
and build, and nothing else. Every helper call, raise, bail and yield is behind
`ahead_guard`, which leaves the run with its last tail suspended -- a thunk of
the whole body against a frame of the slots, which is what the interpreter's
thunk of the call would have run one `Apply` later. A run is at most
`kBuildAhead` (64) cells and one slice. "Building ahead" in
[dream/src/jit.cpp](../../src/jit.cpp) is the design;
`dream/tests/programs/jit_ahead.dr` holds it to the interpreter on each way it
could show -- an error twenty cells past the reader, a condition that stops
being a bool, a producer that spins for ever past cell fifty.

What made it a small change is that the abort is never a new kind of state.
The first iteration is the call the interpreter made, and runs as it always
did; an iteration ahead that gives up leaves exactly the suspension a
step-at-a-time producer would have left. The run length barely matters past
16 (`strbuild` 54 ms at 2, 37 at 16, 34 at 64, 32 at 4096).

The tail of a cell already in hand is now built in place in a lazy position
too -- a load that cannot raise -- which is what lets `map f (tail xs)` keep
going over a list that is already built. Without it, `map` stopped every step:
its next argument was a suspension the next iteration would force at once,
and a step whose strict argument is still suspended ends the run rather than
aborting a moment later.

| | before | after |
|---|---|---|
| `strbuild` (benchmark) | 45 ms, 46 MB | **32 ms, 22 MB** |
| `map (*2) (map (+1) (replicate 400000 1))` | 200 ms | **98 ms** |
| self-compile | 1.70 s | 1.70 s, byte-identical image |

`strbuild`'s 22 MB is its cells, counted twice because the list is live while
`_str_concat` walks it; 13 ms of its 32 is collecting them. What is left in the
map chain is the program's own laziness: `f x` is still suspended per cell,
with the frame it reads.

**One difference between the tiers this found, and it was not this change's.**
A non-bool condition compiled raised "`if` needs a bool" where the interpreter
says "`if` needs a bool, got integer 5"; `and`, `or` and `not` the same. An
error from an aborted run comes from the interpreter, so the test showed the
two messages side by side. `dream_rt_type_error_got` gives the compiled tier
the interpreter's words.

## Reusing the frame of a self tail call: measured, and not built

Asked 2026-10-03 as the cheap first step at the 40% of a compile that is
frames, needing no escape analysis: when the interpreter makes a saturated self
call in tail position and nothing captured the caller's frame -- no thunk was
made against it -- write the arguments into it and go round again, instead of
allocating the callee's. Counted before building, with a probe that marked a
frame whenever `make_thunk` was handed it:

| self-compile | JIT on | `--no-jit` |
|---|---|---|
| saturated calls of a closure (each a frame) | 14.68M | 16.45M |
| ... a self call in tail position | 1.12M | 1.85M |
| ... whose frame nothing had captured | **73.8K** | **73.8K** |

Half a percent of the frames, so not built. The reason is the same as the
reason frames are 40% in the first place: an argument is suspended against
the caller's frame unless `thunk_for` can compute it on the spot, so the call
that is about to replace a frame has almost always just captured it. A `let`
does the same. The tail calls a frame could be reused for are the strict
numeric loops, and those are the ones the JIT already takes with no frame at
all -- which is why JIT on and off leave the same 73.8K.

## A JIT that can allocate -- the plan

**Done, 2026-10-04, and not by either route below.** Compiled code allocates,
since "List and map code in the JIT", and since 2026-10-04 a collection may run
underneath it: the machine stack is scanned conservatively and whatever it
points at is pinned in place, so nothing compiled code holds ever moves and
nothing has to be reloaded. That sidesteps the PHI web "Spilling compiled
frames" below measured, rather than making it smaller, and it changes no
generated code. The design, and the two things the first version of it got
wrong -- promoting a pinned object in place, and an argument array that kept a
whole list alive -- are "Collecting under compiled code" in
[docs/gc.md](../gc.md). What follows is kept for the reasoning, which is still
why the precise route costs what it does.

*The plan below was drafted by an AI coding assistant (2026-09-13), not by the
human author of these notes. It is a proposal, not a record: update it as the
work changes it, the way every other section here records what was learnt.*

**The shape of the problem, measured.** A compiled frame is already a root: the
interpreter pushes the frame object on the value stack and passes it into
`CompiledFn`, and the collector finds it and rewrites it. What makes compiled
code unsafe is that its live values live in LLVM allocas (registers), *not in
that frame*. So a collection underneath a compiled frame loses whatever the
frame's canonical copy no longer agrees with -- which is why the four runtime

helpers pin the heap. (The draft also said the tier "refuses every function that
allocates". That is not what the tier checks and not why those functions are
refused; see the second correction below.) The
route chosen here: make something the collector rewrites the source of truth at
every point where the collector can run -- **root the live slots before a
collect-capable helper, read them back after** -- and de-pin the helpers. No
machine-stack walking, which is what keeps this sound with the concurrent
collector's helper threads: they never see the process's stack, but they never
need to, because the roots are the process's own. The draft of this said "spill
into the frame", and that is wrong for a recursing function -- see the first
correction under that stage below, which is the single most useful thing in
this section.


**What the goal concretely is.** Let the JIT compile self-recursive loops whose
values are objects: forward list builders (`list.replicate`, manual `upto`-style
recursions), string-concat folds, map-building folds. `pi` is not the driver any
more -- see the table and "What it did not reach" above; it is already compiled.
The work is all in `dream/src`; there is no compiler change, so no bootstrap
moves.


**The draft of it in stages**, with one caveat about their order. The two
corrections below say that the de-pinning stage is not a prerequisite for the
widening stage, and the two together say something about sequence: done in the
drafted order, de-pinning lands machinery that no workload exercises, because
the tier as it stands is numeric and the forces it makes are shallow. There is
nothing to measure it with until the tier is wider, and "what each stage has to
prove" cannot be satisfied for it in isolation. Widening first gives de-pinning
a workload to be judged by.

*What has been widened since this was drafted*, none of it about allocation: a
compiled function may call another compiled function (`PeerSet`), a `let` in a
compiled body is written where its name is read, and integer `/` and `%` are
inline. All three are in "Making it faster" above. They do not move this plan --
the values are still numbers -- but they do mean the tier reaches ordinary code
rather than only a hand-shaped loop, so the workloads the de-pinning stage was
missing are now easier to write: anything whose helper forces a lazy structure
is one.

- ~~**Fix the known compiled-force SIGSEGV first.**~~ **Done** (2026-09-13). A
  `force_nest` counter on the process, incremented at the top of `force_whnf`,
  and `enter_function` declining the compiled tier past
  `kMaxForceNestForCompiled`. Same shape and trigger as the `JitTooDeep` deopt.
  It was right that this had to come first, and for the reason given: every
  stage after it sends more nested forces out through the runtime. See "Fixed:
  compiled code forcing a long thunk chain crashed" above, and
  `dream/tests/programs/force_chain.dr`, which is the tier-agreement test that
  holds it.
- **Spill/reload and de-pin.** Before each call to `dream_rt_force/arith/
  compare/arith_f`, root every live slot somewhere the collector rewrites;
  after the call returns, read them back, because a mid-call collection moved
  objects. Then the helpers stop pinning. **Two corrections to this stage, both
  found by reading the emitter rather than by writing any of it, and both of
  which would have cost a day each to find the other way:**

  - **The frame cannot be the spill target for a recursing function.** This is
    the one that matters. In the recursive shape (`Emitter::emit_body`) the
    frame is *not* per-invocation: `recursive_call` passes the same `frame_`
    argument down at every depth, because the whole point of that shape is that
    a self call allocates nothing and the only frame in the picture is the one
    the interpreter made for the outermost call. So a depth-N invocation
    spilling its slots into that frame overwrites the depth-0 loop-carried
    state, and the `emit_yield` that reads it back at depth 0 resumes the
    iteration with the wrong values. Spilling to the frame is sound only in the
    loop shape (`emit_loop`), where there is exactly one invocation. What the
    recursive shape wants is a *stack*, and there is already one the collector
    rewrites: `p.pins` (see `Pin` in [dream/src/interp.hpp](../../src/interp.hpp)),
    which is precisely the mechanism a C++ frame uses to hold a `Value` across a
    force. Push the live slots, call, read back, pop. That also covers the case
    the frame never could: an intermediate LLVM temporary -- the left operand of
    `a + b` while the right is being forced -- which is live in a register and
    has no slot to be spilled into at all.
  - **Nothing here is a prerequisite for allocating.** The stage below reads as
    if de-pinning were what unlocks it. It is not. Allocation never collects: a
    collection runs only at a safepoint, and there are exactly two
    (`run_process`'s and `force_whnf`'s, the latter gated on `force_pins == 0`),
    neither of which an allocation passes through. `dream_rt_float` already
    allocates from compiled code for this reason and says so. What the pinning
    actually buys is that a collection cannot happen *underneath* a compiled
    frame while its values are in registers -- and de-pinning is what a compiled
    loop needs before it can force a long lazy structure without the nursery
    growing for the whole walk, which is the `strict!` failure in "Making it
    faster" a size away. So this stage is about memory under nested forcing, not
    about permission to allocate, and the stage below does not have to wait for
    it.

- ~~**Admit object-allocating self-recursion.**~~ **Done** (2026-09-25), by
  the route the last paragraph of this bullet names -- a parameter's argument is
  evaluated only where the callee forces it, and is in a lazy position otherwise
  -- and wider than it asked: an argument that is not "eagerly safe" is
  suspended rather than refused. See "List and map code in the JIT" above. What
  follows is kept because the reasoning is still the reason.

  The draft read as though this
  were a widening of `op_is_supported`. It is not, and the gate that actually
  refuses a list builder is worth knowing before any of it is written.

  Take the shape the stage is for:

  ```
  let rec upto i n acc = if i > n { acc } else { upto (i + 1) n (core.cons i acc) };
  ```

  Three separate things are true about it, and only the third is the blocker:

  - There is no `Op::Cons`. A list literal is `Op::MakeList`; a prepend is the
    `core.cons` **native**. So the list case is not an opcode at all, it is the
    old draft's third bullet, "natives compiled code calls".
  - `core.cons` is the easiest native in the table to admit: strict mask `0b0`,
    and a body that is one `make_cons` (`list_cons` in
    [dream/src/builtins.cpp](../../src/builtins.cpp)). It forces nothing, raises
    nothing and allocates -- which, per the correction above, compiled code is
    already allowed to do.
  - **`Analyzer::run` refuses the function, and would still refuse it with
    `core.cons` admitted.** Every parameter must be forced on every path, and
    `acc` is not: `strict_of` of the `if` is the condition's `{i, n}` union the
    *intersection* of the two arms, the `then` arm gives `{acc}`, the `else` arm
    gives `{i, n}` because a native contributes only what its strict mask
    claims and cons's claims nothing -- so the intersection is empty and `acc`
    never joins the set. Measured rather than argued: compiling the loop above
    beside a numeric one and running it under `--stats` reports `1 functions
    compiled`, and the one is the numeric loop.

  So the stage is a change to the tier's *soundness argument*, not to a table.
  The rule "every parameter is forced on every path" is what licenses compiling
  a self tail call by evaluating its argument expressions eagerly. What a list
  builder needs is the weaker licence: a parameter may be non-strict provided
  every self-call argument in that position is *eagerly safe* -- cannot raise,
  diverge, or perform an effect -- which `core.cons i acc` is, because it builds
  a cell around two values it does not look at. Both halves have to move
  together with `load_slot`, which today *forces* every slot it reads: that is
  exactly what must not happen to a lazy cons's argument, so an unforced read is
  needed beside it -- and `strict_of` counting a slot read as a force is only
  true *because* `load_slot` forces. Change one without the other and the
  strictness claim the whole tier rests on quietly stops being true.

  What bounds the nursery once this lands is the yield: a compiled loop spends
  its slice and returns to the interpreter, which collects at its safepoint, so
  a loop that allocates per iteration allocates for one slice and no more.

- ~~**Generalize the callable set only if measurement justifies it.**~~ **Done**
  (2026-09-15), and done *first* rather than last, because the staged version
  above turned out not to be its prerequisite. What this stage guessed at was
  "call any host native exactly as the interpreter enters it, with the compiled
  caller's slots spilled"; what it needed was the first half of that and none of
  the second. No spill: the arguments go on the process's value stack, which the
  collector already walks, and the heap is pinned for the call exactly as
  `dream_rt_force` pins it. The vetting did shrink from an allowlist to a
  property, and the property is purity plus a saturated fixed arity plus not
  vouching for the collector -- see "A compiled function may call a host native"
  in *Making it faster* for the rule, the numbers, and the one measurement
  (`str_concat` under a compiled frame) that added the third clause.

  Two things it changes about the stage above, which is still open. Its second
  bullet -- "`core.cons` is the easiest native in the table to admit" -- is now
  true in the sense that `cons` *is* admitted, being pure, binary and
  non-vouching. Its third is untouched and remains the whole blocker: `upto`
  above is still refused, because `acc` is neither forced on any path nor
  *carried* (the widening that landed beside this one covers a parameter handed
  straight back, and `core.cons i acc` computes a new one). So the list builder
  still wants exactly what this bullet said it wanted -- the weaker licence for
  an eagerly safe self-call argument -- and now wants only that.

**What each stage has to prove.** A repeated `pi` row that does not move (a
regression there is a spill paid on the hot path instead of only at true
boundaries); new rows for list builder, string fold and map fold, judged
JIT-on against JIT-off (the placement-noise floor applies); equivalence between
the tiers under `just test-all` (heap-verified run plus fuzzer) for the new

`consbuild`, `strfold` and `mapfold` cases; and the long-thunk-chain repro
raising `:stack_overflow` byte-identically with and without `--no-jit`.

## Spilling compiled frames: measured, and not kept

Built and reverted 2026-09-13. The mechanism was correct -- every tier-agreement
test passed -- and it cost 21% on `fib` and 40% on `mapfilter`, so it is not in
the tree. All of it is written down because the three things the plan above got
wrong are the three things anyone doing this again would get wrong, and because
the reason for the cost is not the one anybody would guess.

**Where the roots live.** Not in the heap frame, which is what the plan said.
Two reasons. In the recursive shape *every depth shares one frame* -- it is
passed down and only depth zero ever yields through it -- so a frame spilled at
depth three is the frame depth two reloads from. And a frame slot needs
`dream_rt_frame_store` for the write barrier, which is a call per slot. So it is
`Process::jit_roots` instead: a plain array the collector walks, `top` bumped by
compiled code itself with no call and no allocation.

**What has to be spilled is not just the slots.** The plan said slots. An
operand already evaluated while the operand beside it is still being evaluated
is a live reference in a register and nothing else -- `a + f b` is the shape,
and so is every argument of a self call but the last. Those are held in allocas
by `Emitter::RootSet`, a stack that mirrors the emitter's own recursive descent,
and they travel with the slots. The frame reference is one of them too: a
collection moves the frame, and the yield writes through it afterwards.

**`arith` and `compare` cannot be de-pinned, and the plan was wrong to say they
could.** Making the *caller* safe is not enough: `arith` on two lists is
`concat_lists`, which forces a whole spine holding both operands in C++ locals,
and `arith` holds them too. Rooting at the `dream_rt_*` boundary does not reach
either. So only `dream_rt_force` vouches -- which is the one that mattered
anyway, being the helper a compiled loop spends unbounded time under. The other
three keep `PinsTheHeap`, and their comments now name the real cause.

**One thing the interpreter had to give up.** `enter_function` held the frame in
a C++ local across the compiled call, and three of the five outcomes hand that
frame straight to `eval_node`. It is `Pin`ned now.

**The cost, and the reason.** Best of five on this machine, against the same
build with the spill emission switched off:

| | baseline | spilling | |
|---|---|---|---|
| `fib` | 39 ms | 47 ms | +21% |
| `mapfilter` | 25 ms | 35 ms | +40% |
| `pi` | 32 ms | 35 ms | +9% |
| `collatz` | 205 ms | 219 ms | +7% |
| `sum` | 49 ms | 50 ms | +2% |
| `strbuild` | 81 ms | 82 ms | +1% |

`mapfilter` is the one that explains it, and it explains it in a way no guess
would have reached: **its spill code never runs**. Its slots hold fixnums, so
every `force` takes the inline fast path and the slow block is dead. The
optimized IR says what happens instead -- the reload writes *every* root, so
each of the twelve force sites in that loop body contributes a fresh PHI per
root at its join, and the loop carries a web of fifty-odd PHIs where it used to
carry four. The register allocator pays for that on the path that does run.
So the cost is not the stores; it is that a reload site is a *merge point for
everything the frame holds*.

Two rounds of tuning went in before the numbers above: the capacity check
hoisted to the function entry (a frame's region base is invariant, so no spill
site has to go and find it), and `dream_rt_roots_state` and
`dream_rt_reduction_slot` marked as the pure address computations they are, so
LLVM sinks them to the paths that use them rather than emitting both at every
entry of a recursive body. Those took `pi` from +34% to +9% and `mapfilter` from
+56% to +40%. What is left is the PHI web, and tuning will not move it.

**So the next step is to make the sites fewer, not the spill cheaper.** Two
candidates, both of which stand on their own merits:

- **Force the parameters at entry and stop forcing them in the body.** The
  analyzer already proves every parameter is forced on every path -- that is the
  admission test -- and `Analyzer::force_order`/`entry_forces` already computes
  the definite prefix of the order for the float slots. Extend that to every
  tagged slot it can pin down, and `load_slot` needs no force at all for those.
  A tail call only ever writes values that are already in WHNF, so from the
  second iteration on *no* slot needs forcing. For a fused loop that removes
  every spill site from the hot loop rather than making it cheaper.
- **Restart the iteration instead of reloading.** A compiled loop body is pure,
  and re-forcing an updated thunk is a resolve. So on a collection the loop
  could branch back to its header with the reloaded values, which collapses
  every per-site PHI into the loop header's existing ones. It needs a collection
  counter on the process to test against, or it re-forces for ever.

Neither was worth doing on spec, which is why this was reverted rather than
carried: nothing compiled allocates yet, so all the spill buys today is that a
compiled loop forcing a long lazy structure can collect instead of growing a
nursery no safepoint can reach -- real, but not visible in any measurement here,
and not worth 40% of `mapfilter` to have early. Do the site-count work first;
then this becomes cheap enough to be worth having, and stage three has something
to stand on.

## Strict arguments are evaluated where they stand

2026-10-09. A saturated call to a native already evaluated its strict arguments
in the caller (`fill_native_args`); a call to a *closure* suspended every
argument, including one its function forces on entry. `let rec go i !acc = ..`
made a thunk for `acc` at every call and forced it one step later.

The loader now records, per function, the parameters its body forces before
anything else (`Image::strict_args`): the run of strict `local` reads at the
head of the body, which is exactly the shape `lower_body` emits for `!acc`. A
saturated closure call with any of them evaluates those arguments in the caller
through a `ClosureArgs` continuation -- the closure half of `NativeArgs` -- and
builds the frame when the last is in. Only that prefix counts, because it is
what makes the change unobservable: nothing of the callee's has run before its
strict parameters are forced on either path, and they are forced in the same
order, so the same argument raises first with the same error. Every image is
byte-identical.

**On today's code it is worth almost nothing**, and that is worth knowing
before reading anything into it: 0.14% of a self-compile's allocation. The
compiler's hot `!acc` loops run compiled, and `thunk_for` already computes an
`acc + x` whose operands are in hand without suspending it -- a three-million
step `!acc` loop under `--no-jit` allocates the same bytes on both VMs.

**On strict code it is the difference.** The all-strict compiler from the next
section allocates 4.25 GB on the old VM and 4.00 GB on this one (-5.8%), and
thunks fall from 23% of allocation to 18%. So it is what a strict parameter
costs, and it is kept for that.

## A pinned heap grew past its cap

Found 2026-10-09 by the experiment below, and not caused by it. A process's
heap is capped (`DREAM_MAX_HEAP`, 1 GB), and the cap fires only when the
*live* data measured by the last collection is past half of it -- deliberately,
so a process that merely allocates fast is not killed. But under a force
nobody vouched for (`force_pins`; see "Collecting underneath a native" in
docs/gc.md) nothing can be collected, so the live figure is never measured
again. `list.length (list.from 1)` with the list held is a native walking an
endless list in a nested machine, and it grew until the allocator threw and
`bad_alloc` took down the VM -- under the JIT and the interpreter alike, on the
VM before any of this work.

Under a pin, everything allocated since the last collection is held whether or
not it is reachable, so `check_limits` takes the allocation as the measure
there. The case now raises a catchable `:out_of_memory` in that process.

## Strict builtins and pinned forces

Found 2026-10-12, chasing what looked like a cost of `lazy.forced`. Replacing
`type_of (out s) != :nothing` with `lazy.forced (out s)` in `opt.settle` made a
self-compile allocate 2.7% more and run 3% slower, with byte-identical output,
the same reductions, and the same allocation by kind. Under `--no-jit` a compile
is deterministic, which is what made it findable:

- The "extra allocation" was promotion: allocated and promoted grew by the same
  126 MB, so the same objects were made and more of them survived a minor.
- Counting remembered-set entries by kind (`remember_if_old`) put it on
  `Blackhole`: half a million more thunks were already old when forced, spread
  over lowering and parsing rather than near `opt.settle` at all.
- Tagging the GC trace with a per-heap serial showed the *cheaper* run had seven
  minor collections far past the 32 MB cap -- 551 MB allocated with nothing
  able to collect -- against three (161 MB) in the other.
- Charging bytes allocated under each outermost unvouched force to the native
  that started it named `_array_of_list`: 808 MB of the compiler's work ran
  inside `array.of_list`'s spine walk in one run, 74 MB in the other.

`array.of_list (list.reverse acc)` at the end of a pass *is* the rest of the
pass, run under a native that did not vouch, so none of it could collect. Which
`if` forced what first decided how much of the program ended up in there, and
an uncollected stretch is a huge nursery: little promotion, a little faster,
and no bound on memory -- the failure that made `dreams --check --test` run out
of its heap the same day, through `to_string`. `_array_of_list` now vouches.
The two forms of `opt.settle` then cost the same; the self-compile is about 2%
slower than when that work ran pinned, and holds 27% less from the OS (1.30 GB
against 1.78 GB).

Two things to take from it. A native that walks a lazy structure must vouch,
or the work behind the structure runs uncollected; `bi_settled`'s comment and
`core_array_of_list`'s say how. And when a change to hot code moves allocation,
compare `--no-jit --stats` promotion before believing the change costs anything.

A bigger nursery does not buy the time back: at 64 MB a self-compile measured
2.737 s against 2.747 s at 32 MB, inside the noise. The cap is now a setting a
program's root package can make, `[vm] nursery_max` in its `mind.toml`, carried
in the image's `VMOP` section (docs/bytecode-format.md); `dream --nursery-max`
and `DREAM_NURSERY_MAX` overrule it. Under the JIT a compiled loop reaches the
collector's check less often, so its nursery overshoots any cap -- 3.4 MB and
climbing against 256 KB -- which the interpreter does not.

## Strict by default: the experiment

2026-10-09. The question: would Dream be better strict by default, with
laziness written where it is wanted? Half of what CLAUDE.md warns about is the
cost of laziness -- `fold_strict`, `strict!` inside a list being suspended
itself, a lint for accumulators nothing forces, a guard written only to make
the JIT admit a loop. "Strictness analysis" was measured earlier and found
worthless *for allocation*; a strict semantics is a different claim, and
nobody had measured it.

**How.** `-D strict_everything` makes the compiler rewrite every named
parameter of every `let` and `fn` in the program as though it were `!x`
(`strict_everything` in dreams/src/modules.dr). It is call-by-value in arguments
and nothing else: list cells, map values and a block's `let` stay lazy. Two
exceptions, both to model what a strict-by-default compiler would actually do
rather than to flatter the result: a pattern parameter, and a `let` whose body
is one call or one `.[ ]`. The second matters. That is the shape
`scope.wrapper_of` turns into the primitive it wraps, and it refuses a wrapper
with a strict parameter -- so without the exception `list.nth`, `list.head`
and every accessor stopped being an opcode, and the first measurement was 66%
more reductions, most of them that.

The compiler built with the switch compiles the compiler: **its output is
byte-identical** to the lazy compiler's. Self-compile, same VM, same source:

| | lazy | all-strict | |
|---|---|---|---|
| wall (JIT, best of 3, interleaved) | 2.36 s | 2.84 s | **+20%** |
| reductions | 254 M | 352 M | +39% |
| bytes allocated | 4.17 GB | 4.00 GB | -4% |
| promoted | 643 MB | 521 MB | -19% |
| live at the largest major | 309 MB | 205 MB | **-34%** |
| held from the OS at the peak | 1.09 GB | 0.73 GB | **-33%** |
| frames live at the largest major | 762 K | 221 K | -71% |

**The memory is the finding.** Allocation barely moves, which is what the
earlier analysis saw. *Retention* is cut by a third, and the by-kind line says
where: frames live at the largest major fall from 762 thousand to 221 thousand.
A suspension carries its whole frame, so a lazy argument nobody has forced yet
keeps everything its caller's frame reached. That is a space leak with a
specific shape, and it does not need strictness to fix: a thunk that kept only
the slots it reads would release the rest. That is the next thing to measure.

**The time is real work laziness was skipping**, not overhead. The largest
single item: the lexer's `char_len` walks a `//` comment to advance the column
(dreams/src/lexer.dr, `skip_trivia`), and the column is never read, because the
newline after the comment resets it. Lazily that walk never happens; strictly
it is 16.6 M reductions. The rest is spread thin over the folds, the checker
and the parser -- the same shape many times: an argument computed for a
branch that does not use it.

**What breaks.** `std --test` built the same way: 14 of roughly a thousand
cases fail, in three kinds.

- *The callee decides when, or whether, to evaluate.* `time.within! d
  (list.length (list.from 1))` hands a computation to a worker to run under a
  deadline; strictly the caller runs it first, forever. `http.get! "not a
  url"` raises before the callee's own handler is reached. A contract test
  asserts that an argument nobody uses is never evaluated. A strict Dream
  writes each of these with `$( )`.
- *The order of effects in impure code.* The registry, websocket, log and HTTP
  client cases: an impure call in argument position ran later lazily, after
  effects the test had arranged, and earlier strictly. Strictly is arguably the
  order a reader expects -- but every one of these programs was written and
  tested against the other order.
- *Nothing breaks, it is only slower:* the `char_len` kind.

**The verdict, for now: do not flip the default.** 20% is too much to pay for
code that works, and the second kind of break is the expensive one to migrate,
because nothing reports it. What the experiment did buy is a target: the
memory win comes from what suspensions *retain*, not from how many are made,
and that can be had without changing what any program means. The switch stays
in the compiler so this can be measured again.

## Direct calls, and where a precompiled run's calls actually come from

2026-10-11. **Measured, and not kept** -- except the collector's half, at the
end. Direct calls were commit a325e9e and were reverted after it; compiled
suspensions were never committed. The lessons are at the end of the section,
and they are the reason it is long.

`--precompile` compiles every function first, and a self-compile
still ran in about the same CPU as the interpreter alone. Frames were 40% of
what it allocated, and only a call between two functions of one compile -- a
peer, at most eight a module -- avoided one. So the first thing built was a way
for any compiled function to call any other without a frame ("Direct calls" in
jit.cpp at a325e9e). A function publishes its body as a peer in a table by
function index, beside the set of arguments it takes evaluated, and a call that
is not to a peer of its own compile reads the slot: a body published for the
set the caller assumed is called with the arguments in registers, and anything
else is the machine's call it always was.

**It is correct, and on a self-compile it buys nothing.** Same binary,
`DREAM_JIT_DIRECT=0` against the default, interleaved: 2.21 s against 2.22 s
of run, the image byte-identical, allocation unchanged to three figures. What
was counted -- temporarily, by call site -- says why:

| a precompiled self-compile | count |
|---|---|
| function entries, compiled | 9.7 M |
| function entries, interpreted | 6.4 M |
| -- of them because `force_nest` was past 256 | **6.0 M** |
| frames made by interpreted code's calls | 16.1 M |
| frames made for compiled code's suspensions | 3.7 M |
| thunks made | 42 M |
| calls compiled code handed to the machine | 3.4 M |
| calls compiled code made directly | 0.5 M |

Compiled code calling compiled code is rare. Almost every call starts in the
interpreter -- in a thunk's body, which is never compiled, or in a function
entered past `kMaxForceNestForCompiled`. That bound is the larger surprise.
Interpreted code nests forces three or four deep; precompiled code reaches the
256 bound constantly, at 4.5 MB of machine stack (17 KB a level -- a compiled
body with its peers inlined has a large frame), and past it every function is
interpreted. Two thirds of the interpreted entries are that, and none of it is
a function the tier refused.

**A published body has to be paid for.** Most functions are compiled in the
loop shape, which reads its frame and is not a body of its arguments, so
publishing one is a second compile -- 200 to 500 ms of LLVM for the larger
functions here. Compiling it beside every function made `fib`'s compile 11.5 ->
18 ms and the benchmark 38 -> 45 ms; asking for it on the 32nd miss made a
self-compile spend 4.5 s of CPU during the run on 23 of them, and at ten
thousand misses it was still 0.85 s for nothing the wall clock could see. So a
recursive root publishes the body it already has, for nothing, and any other
only when `DREAM_JIT_EXPORT_MISSES` asked.

Nor does anything else miss it. The seven workloads of benchmark/benchmark with
it on and off, twice each, interleaved, agree to the millisecond (fib 37/38,
collatz 63/64, ...), and a program written to favour it -- a hot loop calling
eleven helpers, two of them recursive roots in other compiles -- ran 500 ms
against 525. Five per cent at best, for 650 lines of the tier.

What this says about the next steps: a thunk compiled code makes should have a
compiled body (most calls are made from one), and the nesting bound wants to
be a measure of the machine stack rather than a count -- which is only
affordable once a collection does not scan that stack word by word.

**Compiled suspensions, measured the same day: off by default.** A suspension's
node forced often enough was compiled as a body of its frame. It was correct -- every e2e
program agrees with every suspension compiled on its first force -- and it
loses: with the 404 hottest nodes compiled before the run, so that no compile
is paid during it, the run took 2.52 s against 2.30 s. Counting every force of
every node not yet compiled was 2.5% of the run, and a compiled suspension runs
on the machine stack, which a collection under compiled code scans word by
word: `pin_conservative` and the held blocks went from 4% of the run to 7%.

**What the conservative scan costs, and what was cheap to take back.** On a
precompiled self-compile 7,200 collections run under compiled code; they scan
52 M words (7 K on average, 564 K at the deepest), pin 2 M objects and hold 52 K
nursery blocks. Of the scan's 640 ms: reading the words 162, sorting the hits
166, finding the object each hit is in 122, and marking the dead in every held
block 198 -- a walk of 3.2 GB. The last is gone (a held block records how far
it was filled and what was pinned in it; docs/gc.md), and a 256-entry table of
addresses already taken drops most repeated words before the sort. Against the
commit before, in a worktree, interleaved: 1.61 s stopped in collection against
1.79 s -- and the run's wall clock unchanged, 2.24 s against 2.21 s. Collection
time summed over a build's processes is not its critical path.

So none of the three -- direct calls, compiled suspensions, cheaper pins -- moved
a self-compile's wall clock, though each removed the cost it was aimed at. The
cheaper pins are kept: they are small, inside the collector, and take back CPU
every program spends. The other two were taken out; with `--precompile` and the
`run:` line of `--stats`, which stay, they can be measured again from a325e9e
if the picture changes.

**What to do differently next time.**

- *Find the critical path before choosing a target.* A self-compile is a
  parallel build: 8.7 s of CPU in 2.2 s of wall. A profile of where the CPU
  went, and collection time summed over every process, both say what the
  machine paid and nothing about what the clock waited for. Three changes were
  aimed by that profile and all three missed the clock.
- *Count why the thing happens before building the fix for it.* The table above
  took an hour of temporary counters and would have killed direct calls before
  a line of them was written: compiled code calling compiled code is 0.5 M of
  16 M calls. Count the origin of the events (who made the frame, who entered
  the function), not only their number.
- *Price the second compile.* Anything that needs a function compiled again in
  another shape pays LLVM's 200-500 ms a function, during the run, and needs a
  number of uses that pays for that before it is worth asking for.
- *A/B in a worktree, always.* The VM links `build-dream/lib/libdream.so`, so a
  copy of the binary taken before a rebuild runs the new library -- the first
  "baseline" here measured the change against itself.
- *Running Dream on the machine stack costs the collector.* Every frame of it
  is scanned conservatively under compiled code, so moving work there
  (compiled suspensions, deeper nesting) is paid for in pins until a
  collection has precise roots. That is the order: roots first, then more
  machine stack -- and roots only once the clock, not the CPU, says so.

The next measurement, then, is what sets a self-compile's wall clock -- the
slowest module, and the stages that wait on it -- not precise roots, whose
whole prize is the remaining 340 ms of scanning CPU.

## JIT stack bounds and selection policies (2026-10-13)

The widened slot masks admitted functions whose generated stack frames were
much larger than their parameter and local-slot counts suggested. Saved crashes
included generated code clearing a stack buffer and GC scanning across a
worker's stack guard page. A nesting count could not bound the bytes consumed
by those frames; nested interpreter forces also started a fresh compiled
recursion count.

The tier now checks actual stack addresses. Before entering compiled code it
reserves 256 KiB above the current OS thread's stack bottom. Nested compiled
calls share a 512 KiB budget from the outer machine loop, and recursive/peer
bodies compare their stack pointer against that floor inline. The OS bounds
are cached per thread, so a process resuming on another scheduler worker gets
that worker's bounds. Snapshot sites also share one scratch array per generated
invocation, instead of each reserving a full array of local slots.

`jit_wide_stack` reproduces the old SIGSEGV on a 1 MiB worker stack; both the
fixed JIT and interpreter produce the same result with heap verification and
a small nursery. `jit_branch_stack` exercises fallback from branching machine
recursion. `jit_recursion_loops` checks that tail recursion, additive
accumulation and tail calls through bindings emit loops and run deep on the
small stack. General branching recursion keeps its evaluation order and uses
the interpreter's heap continuations once the byte budget is spent.

Selection is available as `--jit-policy`, with the old `DREAM_JIT_RANK` as its
environment default. `density` (calls/body size) remains the default. The
`adaptive` option samples every 64th interpreted reduction and learns the CPU
cost of successful compilations. Its regularized, exponentially weighted model
uses body size and body size times local-slot width; queued candidates are
ranked by `max(calls, 64 * samples) / predicted compilation milliseconds`.
On systems without a thread CPU clock, the observations use elapsed time.
The model is fitted after a compile, away from the interpreter's hot loop;
the trace shows each selection's samples, score and estimated cost.

Measured on a Ryzen 9 5900X, with serial, interleaved runs and identical output
required for every policy. Self-builds must reproduce the compiler image byte
for byte. The first comparison used five rounds for each policy:

| policy | self-build median, seconds | one CPU, seconds |
|---|---:|---:|
| calls | 4.835 | 15.424 |
| density | 4.590 | 15.247 |
| calls times body size | 5.105 | 15.490 |
| sampled work/body size | 4.787 | 16.048 |

Dividing by body size squared also lost: 4.700 s against density's 4.616 s in
another five interleaved rounds. After adding the learned cost model and CLI,
five rounds of density/adaptive self-builds measured 4.663/4.764 s. Three rounds
restricted to one CPU measured 15.406/19.038 s. Adaptive improved the sum and
Collatz kernel medians by about 4% on the unrestricted machine, but compilation
and sampling overhead made it a worse default for self-builds, especially when
LLVM competed for one CPU. It remains an explicit option for workload-specific
measurement rather than an automatic assumption that more profiling is faster.

The reproducible comparison, including all seven kernels and optional CPU
affinity, is `dream/tests/jit_selection_bench.py`. It accepts `--policies`,
`--runs`, `--cpu`, and `--output` for a JSON record. Each policy is selected
through the CLI, so the benchmark exercises the same override users do.

## Frames freed on return: the most it could save

Measured on 2026-10-17 before building anything, with `DREAM_PROBE_FRAMES=1
dream --stats`: how many frames a run makes, and how many a suspension is ever
made against. A frame no suspension holds is done with when its call returns,
so the rest are the most that freeing frames on return -- proving it in the
compiler and emitting a free -- could take from the collector.

| | frames, of all allocation | never held by a suspension | of all allocation |
|---|---|---|---|
| self-compile | 50% | 37.7% of frame bytes | about 19% |
| compiled JSON grammar | 39% | 27.7% | about 11% |

And what that is worth in time is smaller than in bytes, because a dead
object in the nursery costs its allocation and nothing else -- it is never
copied. A self-compile spends 1.4 s of 27 s of CPU in minor collections, so
the ceiling is a fifth of that, about 1%. Not built. What holds frames is
suspensions, which is where the cost of laziness is, and freeing the frames
nobody held does not touch it.

## Strict bindings are loads

A strict binding -- every `match` on a call's result binds its subject so --
was compiled like a lazy one: its code written at each read behind a test of
whether it had been computed, and counted against `kMaxBindDuplication` and
`kMaxMemoCode` as though it might run at any of them. But its statement runs
before any read can (a binding is in scope only after it), so a read after it
in the same block is a load of what it remembered (`Emitter::settled_`), and
the analysis no longer counts it, nor a name for it (what a pattern binds a
subject to). Worth about 5% on a compiled JSON grammar's large document and
nothing measurable on a self-compile; it mattered most to inlined code, which
is made of such bindings. `DREAM_JIT_WHY` now names the binding it refuses.

## The timeline, and what a parallel compile was waiting on (2026-10-18)

A self-compile is a few hundred processes -- parse, three resolution walks,
lowering and two type-check passes, each a part per module -- joined by a
root that does the serial work between them. Its wall clock is the longest
chain of dependent work through those, and no profile of reductions can
see a chain: "Precompile experiment lessons" cut CPU and moved nothing.

`DREAM_TIMELINE=FILE` records, with no cost unless set: every slice a
process ran (and how long of it was collecting), every spawn (and what the
work's closure belongs to), every join that had to wait, every result copied
in by a join, and every native call of 2 ms or more. With `--profile N` and
`DREAM_PROFILE_PID=PID`, every 1024th step of that process is sampled too,
by time. `dream/tools/timeline.py FILE` walks back from the root's exit --
through a process's slices, across a join to the end of what it waited on,
from a first slice to the spawn -- and prints the path, what was work and
what was waiting, and for a sampled process what each of its stretches on
the path was doing. `--chrome out.json` is the whole run for Perfetto.

The first picture, 5.26 s of wall:

- 2.1 s was the root's own work between phases, and 1.0 s parts sat in a
  queue -- work and waiting nobody had seen.
- The root's long stretches were single slices of 200-325 ms: natives that
  do not yield. They were `vm.share!` -- not the copy into the shared area,
  which is 20-50 ms, but its deep force of lazy compiler state (the
  declarations, the checker's tables). So that was compiler work.
- Declaring the program (`scope.declare!`) ran three times: once for macro
  expansion's snapshot and once for each of the two resolution walks, which
  start from the same loader. Now declared once for both walks
  (`scope.resolve_declared!`). And `lend_shorthands` sorted every library's
  exports again for each of the program's 143 modules.
- Lowering was 141 parts averaging 30 ms and one of 870: one large module.
  `lower.balanced` cuts a part heavier than a 48th of the program into runs
  of its bodies, which number their functions from the module's offsets and
  link as parts do; `progress.join_heaviest_first!` starts the heaviest
  first. The image is byte for byte the one an unbalanced build writes.
- The root, dealing 200 parts, stalled 270 ms a third of the way through:
  `spawn!` dealt children round robin over every worker, its own included,
  and a parent that yields goes behind what is queued on its worker. A
  child now goes to some other worker.
- Each part runs as one slice -- inside `vm.share!`'s force -- so a worker
  queue is run to completion, and there were eight of them on a 24-thread
  machine: `default_workers` was capped at 8 because an idle worker woke
  every 500us, which a one-process program paid for. Idle workers now back
  off to 16 ms, woken through the lock they test the queues under so a wake
  is never lost, and the default is every usable core. Self-compile: 4.9 s
  with eight workers, 4.4 s with twenty-four; the benchmarks and a short
  program are unchanged.

5.37 s to 4.35 s in all, the images unchanged. What is left on the path is
work: 2.0 s of it the root's, and per phase the largest part. On 24 threads
the parts' CPU is 30 s -- SMT makes each slower than on 8 -- of which
parsing is 5.7 s, 47 reductions a byte of source.

## Suspensions that hold values, and frames on a stack: the measurement and the plan (2026-10-18)

A suspension the interpreter makes (`thunk_for`, a lazy `let`) is a node and
the frame it was made in, and it keeps that frame alive however few of the
frame's slots it reads. `DREAM_PROBE_FRAMES=1 dream --stats --no-jit` on a
self-compile, extended to ask the one question that decides whether the
alternative is possible:

| | |
|---|---|
| frames made | 59.1M, 3.64 GB -- 44% of everything allocated |
| held by a suspension | 29.1M, 2.01 GB |
| suspensions made | 83.6M, reading **1.69** values each |
| as suspensions that copy those values | 2.47 GB, against 2.01 GB now plus the frames |
| reading a slot not yet bound when made | **0** |

The last line is the one that matters. A suspension that copied the values
it reads, as a closure already does, would need its frame for nothing; and
since none read a slot bound after them, nothing is lost by copying at the
moment they are made. Then nothing on the heap points at a frame -- closures
copy, suspensions copy -- and a frame is reachable only from the
continuation stack and the current frame, so its life is exactly a call's.

The plan, in the order it can be measured:

1. **Flat suspensions.** `thunk_for` copies the node's free slots (worked out
   once per node, as the probe does) and the closure; forcing one runs the
   node in a frame filled from them. A slot not yet bound (none in a
   self-compile, but a `let rec` can) gets a placeholder the binding later
   fills.
2. **Frames on a per-process stack**, released when the machine returns below
   the depth a frame was made at, or at a tail call. The collector scans the
   live ones as roots. That takes 3.6 GB of 7.6 off the heap, and all the
   promotion of frames a suspension kept past a minor collection.
3. **The JIT's suspensions** (`dream_rt_snapshot`) copy the same way.

"Frames freed on return" above measured the ceiling of freeing only the
frames nobody held at about 1%: those were already dying young. What this
removes is the other half -- the held ones, which are the ones that cost.

## Frames on a stack, built (2026-10-18)

Steps 1 and 2 of the plan above, as `flat_thunk`/`run_suspension` in
interp.cpp and `FrameStack` in process.hpp. Step 3 was not needed: compiled
code still makes its snapshots on the heap, and copies the interpreter's
frame there only when it adopts one (`dream_rt_adopt`) or parks with one in
an `EnterRetry` -- not on every compiled call, which is what a first version
did, and which left frames at 19% of allocation.

A self-compile, against the commit before on the same machine, three runs
each:

| | before | after |
|---|---|---|
| allocated | 7.83 GB | 5.79 GB (-26%) |
| promoted | 1.38 GB | 1.08 GB (-22%) |
| live at the largest major | 590 MB | 430 MB (-27%) |
| collections | 1644 major, 13981 minor | 1379, 10208 |
| wall, all cores | 4.3-4.6 s | 4.3-4.6 s |
| wall, `-j 1` | 18.2 s | 17.8 s (-2%) |

Memory, not time. A `perf` profile of the single-worker run after it puts
the collector's scanning and forwarding at about 3% of the samples: the
interpreter's own dispatch and LLVM's optimizer are where the time is, so
taking a quarter of the allocation away moved the clock by about the share
the collector had left. The heap is now mostly flat suspensions (thunks
51% of what is allocated, at 37 bytes on average against 24), maps and lists.

Three things the build found, all about what may still point at a frame
once it is dropped:

- **The frame register and a nested force's saved frame** may name a
  dropped frame whose memory a newer one now uses, mid-object. The
  collector read that as a header and promoted garbage; it showed as
  integers where lists were, far from the cause. `Process::visit_roots`
  now asks `FrameStack::holds` before forwarding either. The way it was
  found is worth keeping: a run with flat suspensions over *heap* frames
  was correct, so the copying was sound and the fault was in reuse; a run
  that poisoned dropped frames and never reused their memory was correct
  too, so nothing used a dropped frame -- something only *pointed* at one.
- **The heap verifier** treats a pointer into no heap and no shared area as
  corruption, and the continuations hand it stack frames. A continuation
  names only live frames, so its header is safe to read and a stack frame
  there is skipped; the frame's slots are visited as roots of their own.
- **A call in tail position** makes its frame where its caller's was, so
  the saturated-call path builds the arguments' suspensions before the
  frame rather than writing them into it from the caller's slots.
