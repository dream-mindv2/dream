// The LLVM JIT tier.
//
// Scope, and why it is drawn where it is: this compiles a function's evaluated
// spine -- arithmetic, comparisons, branches, `let`, calls to itself and to
// other functions it can compile, primitive opcodes and host calls, reading and
// building lists, arrays and maps, and calls it hands to the machine -- and
// leaves what the program does not evaluate to the interpreter. The limit is a
// soundness requirement. Compiled code evaluates what it can reach eagerly, and
// doing that to something the program would never have forced can raise an
// error in a program that was going to terminate quietly.
//
// So nothing is evaluated here that the interpreter would not have evaluated.
// An argument is evaluated only where the callee forces that parameter on every
// path -- a strictness fixpoint, per parameter (`Analyzer::run`, "Which
// arguments are evaluated") -- and every other position is *lazy*: its value is
// built in place when building it cannot be observed, and suspended against a
// frame made on demand otherwise, exactly as `thunk_for` would have done (see
// "Lazy positions"). A parameter every self call hands straight back to itself
// is moved rather than computed (`Analyzer::carried`).
//
// This used to be a condition of *admission* -- a function was compiled only
// when every parameter was forced on every path -- which refused every list
// builder, every lazy accumulator and every function that calls what it is
// given. It is a decision per argument now, and those are compiled.
//
// What that rule does *not* promise is that an argument is forced at the same
// point the interpreter would have forced it, only that it is forced. Where two
// errors are reachable that can decide which one a program gets; the repro and
// why it is not fixed are under "The tier's eager arguments can change *which*
// error a program raises" in docs/notes/vm-performance.md.
//
// A `let` has nowhere to live here -- no thunks, and the frame's slots are
// registers -- so its value is emitted where its name is read, which is where
// the interpreter would have forced the thunk. `Analyzer::check_bind` is the
// rule and what it costs.
//
// Self recursion comes in two shapes and they are compiled differently. A tail
// call is a loop back-edge: the parameters are overwritten and control jumps to
// the top, which is what lets a compiled loop run in constant space and yield
// to the scheduler at every iteration. A *non-tail* call -- `fib (n - 1) +
// fib (n - 2)` -- is a machine call, which means the body has to be a function
// of its arguments rather than of a heap frame. So such a function is emitted
// twice over: an inner body taking its parameters by value, and an outer entry
// matching `CompiledFn` that unpacks the frame once and calls it. The frame the
// interpreter would have allocated per call disappears entirely, which on `fib`
// is the whole program -- 100% of what it allocates is frames.
//
// A machine call uses machine stack, a fixed resource where a heap
// continuation is not. Compiled recursion therefore carries its stack floor and
// gives up -- `JIT_DEEP` -- rather than overflow, and the interpreter, whose
// recursion is on the heap and bounded by `DREAM_MAX_DEPTH`, takes the call
// over. That is a fallback, not an error: the program sees what it would have
// seen with no JIT at all.
//
// The fast paths are inline; every slow path calls the interpreter's own
// helper, so the two tiers cannot drift apart on what an operation means.

#include <functional>
#include <pthread.h>
#include <sched.h>
#include "jit.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "builtins.hpp"
#include "image.hpp"
#include "interp.hpp"
#include "jit_rt.hpp"
#include "process.hpp"
#include "runtime.hpp"

namespace dream {

namespace {

/// Status codes a compiled body returns through its out-parameter. The same
/// four as `JitStatus` in jit.hpp, spelled again here because the emitter needs
/// them as constants it can put in the IR.
constexpr int JIT_OK = 0;
constexpr int JIT_RAISED = 1;
/// The reduction budget ran out mid-loop. The loop-carried values have been
/// written back to the frame, so the interpreter can pick the iteration up
/// from the top of the body -- an OSR exit that keeps JIT code preemptible.
constexpr int JIT_YIELD = 2;
/// Compiled self recursion ran out of machine stack. Nothing has been written
/// anywhere and no effect has happened -- the compiled body is arithmetic and
/// nothing else -- so the caller may simply run the call interpreted instead.
constexpr int JIT_DEEP = 3;
/// An entry guard did not hold: a parameter the body carries as an unboxed
/// double arrived as something that is not a float in hand. Like `JIT_DEEP`,
/// nothing has happened yet and the interpreter can simply run the call --
/// but unlike it, the function is not given up on, because this is a fact
/// about one call rather than about the function.
constexpr int JIT_BAIL = 4;

/// How many *other* functions one compilation may pull into its module.
///
/// A call to another global function is compiled by emitting that function's
/// body into the same module and calling it, so saying yes to one costs the
/// compile time of writing it. That time is real and already visible: the
/// first run of `mapfilter` spends about 19 ms in LLVM, which is most of what
/// the benchmark reports for it. A small closure buys the shapes that matter
/// -- a loop that calls a helper, a helper that calls one more -- and a large
/// one buys a compile nobody asked for.
constexpr uint32_t kMaxPeers = 8;
/// How deep the closure may be followed. A guard against a chain of calls,
/// not a budget: `kMaxPeers` is what bounds the size.
constexpr int kMaxPeerDepth = 8;
/// "Not a function index", for the answer to "which function does this call
/// name?" when it does not name one.
constexpr uint32_t kNoFunc = ~uint32_t(0);

/// How many arguments a native compiled code calls may take.
///
/// The call is one helper with a fixed signature and the arguments in
/// registers, so the widest native decides the signature. Four covers every one
/// the VM provides -- `ffi.bind!` and `ffi.load!` are the only four, and the
/// pure ones stop at three -- and a host module that registers a wider one is
/// simply not called from compiled code.
constexpr uint32_t kMaxNativeArgs = 4;

/// How big a `let` may be for its value to be written out more than once.
///
/// A binding read twice has no thunk to remember its answer in, so writing it
/// where each name is read runs it twice. Counted in nodes, after every `let`
/// inside it has itself been written out -- which is what stops
/// `let a = ..; let b = a + a; let c = b + b` from doubling at every step.
constexpr uint32_t kMaxBindDuplication = 8;
/// How big a `let` that calls something may be for its code to be written at
/// more than one read. Its *work* is done once whatever this says -- see "A
/// binding computed once" -- so this bounds the size of the compiled body.
constexpr uint32_t kMaxMemoCode = 64;

/// The most nodes a switch's targets may come to, together, before it is
/// written as the chain of arms it jumps into instead (`Emitter::switch_node`).
/// Each target is a suffix of one chain, so writing them all is quadratic in
/// the arms; `std.types.check` took nine seconds to compile that way.
constexpr uint32_t kMaxSwitchCode = 1024;

/// How many cells one compiled call of a list builder makes before it suspends
/// the rest. See "Building ahead". A bound on work and memory nobody may ever
/// ask for -- `take 1 (replicate n x)` pays for this many cells -- and on how
/// long a compiled call runs before the interpreter sees it again; large
/// enough that the suspension, the frame and the trip through the interpreter
/// it replaces are paid once a run rather than once a cell.
constexpr int64_t kBuildAhead = 64;

/// `DREAM_JIT_AHEAD=0` turns building ahead off, for measuring it.
bool build_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("DREAM_JIT_AHEAD");
        return !v || std::strcmp(v, "0") != 0;
    }();
    return on;
}

// ---------------------------------------------------------------------------
// Representation
//
// A compiled function carries its values in one of two shapes: a tagged
// `Value`, which is what every other part of the runtime speaks, and a raw
// `double`, which is what a float *is* once nobody has to be able to point at
// it. The second exists because a boxed float is an allocation, and a loop that
// allocates one per operation is a loop the collector runs rather than the CPU:
// `pi` in the benchmark allocated 866 MB to add two million terms, 44% of it
// float boxes, and spent its time in the nursery instead of the FPU.
//
// So the emitter decides, before it writes a line, which values are floats, and
// keeps those in registers. Boxing happens at exactly four places -- the value
// returned to the interpreter, the spill a yield makes into the heap frame, an
// operand handed to a slow-path helper, and a join where the two arms disagree
// -- and nowhere in between. A fused `pi` loop reaches all four zero times per
// iteration.
//
// `Ty` is what makes that safe. `Float` is a claim about the *value*, not about
// the expression: it means "if this reduces without raising, what it reduces to
// is a float". That is exactly what `arith` guarantees -- an operator with a
// float on one side of it answers a float or raises, and has no third case --
// which is why `Unknown + Float` is `Float` even though the left side could
// turn out to be a list. If it is a list, the operation raises, and a raise is
// a return, not a value of the wrong shape.
//
// A parameter is the one place the claim cannot be proved, because its value
// comes from the caller. So it is *guarded* instead: a slot the loop carries as
// a double is checked on entry, and a function entered with something else in
// it answers `JIT_BAIL` and is run by the interpreter, which is what would have
// happened had it never been compiled.
//
// Which parameters are floats has two sources. The fixpoint finds the ones a
// loop's own self calls make floats -- `acc + 1.0 / k` -- and a **signature**
// names the rest: a parameter declared `:float` starts the fixpoint as Float
// rather than as Unknown (the image's `TYPE` section, `Image::float_params`).
// The second is what reaches `go (x + dx) dx (acc + x * dx)`, where no float
// literal appears and the fixpoint alone would never say `x` is one. It is a
// guess and is treated as one, because the types are gradual: a caller with no
// signature may pass an integer, and the guard that makes the fixpoint's own
// answers safe makes this one safe too. A declared float that turns out to be
// an integer costs a bail, and a run of them gives the function up
// (`Jit::note_bail`); it never costs an answer.
// ---------------------------------------------------------------------------

enum class Ty : uint8_t {
    /// Nothing is known yet: the fixpoint has not reached this, or the
    /// expression transfers control and produces no value at all.
    Unknown = 0,
    Float = 1,
    /// A tagged `Value` of any kind, floats included. The top of the lattice.
    Any = 2,
};

inline Ty join(Ty a, Ty b) {
    if (a == Ty::Unknown) return b;
    if (b == Ty::Unknown) return a;
    return a == b ? a : Ty::Any;
}

/// The type of `a op b` for `+ - * / %`.
///
/// A float on either side makes the answer a float, whatever the other side is
/// -- see the note above. Otherwise nothing is known until both sides are.
inline Ty arith_ty(Ty a, Ty b) {
    if (a == Ty::Float || b == Ty::Float) return Ty::Float;
    if (a == Ty::Unknown || b == Ty::Unknown) return Ty::Unknown;
    return Ty::Any;
}

/// The host natives compiled code knows how to make, rather than call.
///
/// Every one of them is a pure function of one number: no effect, no Dream
/// code underneath it, and nothing it can do that the machine cannot do in a
/// few instructions. That is the whole admission test, and it is a narrow one
/// on purpose -- a native that runs Dream work underneath itself would do so
/// with this frame's values in registers the collector cannot see.
///
/// They matter out of proportion to their size because a call to *any* of them
/// used to refuse the whole function: the tier compiles self calls and nothing
/// else, so a fused `pi` loop, whose body is arithmetic and one `to_float`, was
/// never compiled at all.
enum class KnownNative : uint8_t {
    None, ToFloat, ToInt, Sqrt, Abs, Floor,
    // What `match` lowers a list pattern to: a test for a cell and the two
    // reads of one. Each is a type check and a load, and a list function makes
    // all three every iteration -- as helper calls, they were most of what a
    // compiled `match` over a list cost.
    MatchIsCons, MatchHead, MatchTail,
};

/// Which native `Apply`'s callee names, when it names one of the above.
///
/// The callee has to be `Field(Global module, name)` -- a member of a module
/// named at the top level, which is what every `std` call looks like -- and the
/// module has to be one this runtime provides. `module_for_import` is what says
/// so, and it is the same lookup the interpreter does; a module of the same
/// member name from anywhere else resolves to a different `ModuleDef`, or to
/// none, and is not recognised.
KnownNative known_native(Runtime& rt, const Image& img, uint32_t callee_node,
                         uint32_t argc, bool direct = false) {
    const Node c = direct ? Node{uint8_t(Op::Builtin), 0, 0, callee_node, NO_NODE, NO_NODE}
                          : img.node(callee_node);
    if (Op(c.op) == Op::Builtin && argc == 1 && c.a < builtin_count()) {
        const char* name = builtin_def(c.a).name;
        if (std::strcmp(name, "_to_float") == 0) return KnownNative::ToFloat;
        if (std::strcmp(name, "_to_int") == 0) return KnownNative::ToInt;
        if (std::strcmp(name, "_match_is_cons") == 0) return KnownNative::MatchIsCons;
        if (std::strcmp(name, "_match_head") == 0) return KnownNative::MatchHead;
        if (std::strcmp(name, "_match_tail") == 0) return KnownNative::MatchTail;
    }
    if (Op(c.op) != Op::Field) return KnownNative::None;
    const Node& m = img.node(c.a);
    if (Op(m.op) != Op::Global) return KnownNative::None;
    const GlobalRec& g = img.global(m.a);
    if (g.kind != GLOBAL_MODULE) return KnownNative::None;
    const ModuleDef* def = rt.module_for_import(g.target);
    if (!def || argc != 1) return KnownNative::None;
    StringRef name = img.str(c.b);
    if (def->name == "std.math") {
        if (name.equals("sqrt")) return KnownNative::Sqrt;
        if (name.equals("abs")) return KnownNative::Abs;
        if (name.equals("floor")) return KnownNative::Floor;
    }
    return KnownNative::None;
}

// ---------------------------------------------------------------------------
// Natives compiled code calls
//
// The five above are natives compiled code *makes*: each is a few instructions,
// so writing them out beats calling them. Everything else a program reaches
// through primitive opcodes or builtins -- `str_byte`, `compare`, `len`
// -- is a call, and until this existed a single one of them refused the whole
// function. That refusal was most of what kept the tier to hand-shaped numeric
// loops: a lexer that reads a byte, a resolver that looks a name up in a map and
// a comparison written over `compare` are all arithmetic with one call in
// the middle of them.
//
// A call is made the way the interpreter makes one -- same callee value, same
// arguments on the same stack, same native -- so the two tiers cannot disagree
// about what a native means. What decides whether a call may be *compiled* is
// below, and it is three questions:
//
//   * **Is it pure?** Purity is spelling in this language, so this is one look
//     at the last character of the name. An impure native is refused, and that
//     one rule is what makes the rest of this simple: nothing here has to order
//     an effect against anything, and nothing here can park -- every native that
//     answers `NativeOutcome::Block` is impure, and so is everything reachable
//     from the Dream work a pure one runs underneath itself.
//   * **Is it saturated, and narrow enough?** A partial application has no
//     callee to enter, and a variadic native has no fixed argument positions for
//     a strict mask to describe. `kMaxNativeArgs` is the rest.
//   * **What of the arguments it does not force?** A native forces the ones its
//     mask claims and is handed the others suspended. Those are lazy positions,
//     written as the interpreter would write them: built in place where that
//     cannot be observed, suspended otherwise. See "Lazy positions".
//
// What is deliberately *not* asked is whether the native allocates, or forces,
// or runs Dream code underneath itself. Allocation never collects, so compiled
// code has always been allowed to allocate -- `dream_rt_float` does -- and
// forcing is what `dream_rt_force` already does on every slot read. Both are the
// runtime's to get right, and `run_native` in jit_rt.cpp is where it does.
// ---------------------------------------------------------------------------

/// A native a compiled body may call, and everything needed to call it.
///
/// `builtin` says which of the two shapes it is: a builtin is an immediate
/// carrying an index into a static table, and a member is one function of one
/// host module, named by where it sits. Both are constants at compile time --
/// the module registry is fixed before a program runs -- so the call site is a
/// helper call with two literals in it and nothing to look up.
struct NativeSite {
    bool ok = false;
    bool builtin = false;
    /// Index into `rt.modules()`, when this is a member.
    uint32_t module = 0;
    /// Index within that module's members, or the builtin's table index.
    uint32_t member = 0;
    uint32_t strict_mask = 0;
};

/// Purity, the way the language spells it: a name ending in `!` is impure.
bool native_is_pure(const char* name) {
    const size_t n = std::strlen(name);
    return n > 0 && name[n - 1] != '!';
}

/// Which native this call names, when it names one this tier may call.
///
/// The callee has to be a `Builtin` node or `Field(Global module, name)` -- a
/// member of a module named at the top level, which is what every `std` call
/// looks like. `module_for_import` is the same lookup the interpreter does, so a
/// member of the same name from anywhere else resolves to a different
/// `ModuleDef`, or to none, and is not recognised.
NativeSite native_site(Runtime& rt, const Image& img, uint32_t callee_node,
                       uint32_t argc, bool direct = false) {
    NativeSite site;
    const Node c = direct ? Node{uint8_t(Op::Builtin), 0, 0, callee_node, NO_NODE, NO_NODE}
                          : img.node(callee_node);
    const char* name = nullptr;
    uint32_t arity = 0;
    bool vouches = false;

    if (Op(c.op) == Op::Builtin) {
        if (c.a >= builtin_count()) return site;
        const BuiltinDef& bd = builtin_def(c.a);
        site.builtin = true;
        site.member = c.a;
        site.strict_mask = bd.strict_mask;
        name = bd.name;
        arity = bd.arity;
        vouches = bd.vouches;
    } else if (Op(c.op) == Op::Field) {
        const Node& m = img.node(c.a);
        if (Op(m.op) != Op::Global) return site;
        const GlobalRec& g = img.global(m.a);
        if (g.kind != GLOBAL_MODULE) return site;
        const ModuleDef* def = rt.module_for_import(g.target);
        if (!def) return site;
        const NativeDef* nd = def->find(img.str(c.b));
        if (!nd) return site;
        // Two indices rather than either pointer: a module is found again at run
        // time by where it sits, which is fixed once a program is running and so
        // needs nothing to stay valid. `modules()` is filled while the runtime
        // is being built and never afterwards.
        site.module = uint32_t(def - rt.modules().data());
        site.member = uint32_t(nd - def->members.data());
        site.strict_mask = nd->strict_mask;
        name = nd->name;
        arity = nd->arity;
        vouches = nd->vouches;
    } else {
        return site;
    }

    // Match failure is a raise, which the helper reports through the ordinary
    // error status. It performs no external effect despite the bang suffix.
    if (!native_is_pure(name) && !(site.builtin && std::strcmp(name, "raise!") == 0)) return site;
    if (arity == NATIVE_VARIADIC || arity != argc || arity > kMaxNativeArgs) return site;
    // A native that vouches for the collector is one this tier must not call:
    // it walks a lazy structure underneath itself and is written so a
    // collection may happen while it does, and a call from compiled code pins
    // the heap instead. `NativeDef::vouches` has the measurement. Declining
    // costs nothing that was not already being paid -- such a function was not
    // compiled at all before this -- and it keeps the widening from being a
    // trade of memory for speed.
    if (vouches) return site;
    site.ok = true;
    return site;
}

// The lowerer's catch-all arm is `if true`, followed by a match-error
// fallback. That dead fallback must not weaken strictness or result types.
int constant_condition(const Image& img, const Node& n) {
    const Node& condition = img.node(n.a);
    return Op(condition.op) == Op::ConstBool ? int(condition.a != 0) : -1;
}

/// The ops the emitter writes itself. Three more are decided in `check`,
/// because whether they are written depends on more than the op: a capture
/// (only a root has a closure to read it from), a map literal (only an empty
/// one), and a big integer. Anything else pure is the interpreter's to
/// evaluate -- see "What the interpreter evaluates" -- so this list is no
/// longer what refuses a function, only what is written inline. The
/// emitter used to refuse unsupported ops *after* the analysis had said yes, which cost nothing while a
/// compile was one function -- the emitter gave up and the function was marked
/// rejected. It is not free now: an emitter that gives up on a peer gives up on
/// the whole closure, including a root that had nothing wrong with it. So the
/// two lists say the same thing.
bool op_is_supported(Op op) {
    switch (op) {
        case Op::ConstInt: case Op::ConstFloat: case Op::ConstBool:
        case Op::ConstChar: case Op::ConstAtom: case Op::Unit:
        case Op::SwitchAtom: case Op::SwitchHead:
        case Op::Local:
        case Op::If: case Op::Block: case Op::Force:
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs:
        // Data. See "Lists, arrays and maps" and "Lazy positions" below.
        case Op::ConstStr: case Op::Global: case Op::Builtin:
        case Op::ListIsEmpty: case Op::ListTail: case Op::Cons: case Op::ListIsCons: case Op::ListHead:
        case Op::MakeList: case Op::MakeArray: case Op::RecordMake: case Op::Get: case Op::Set:
        case Op::SetMany:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Lazy positions
//
// Most of what a list or a map function does is put something *somewhere
// nobody has looked yet*: the head and tail of a new cell, an element of a
// literal, the value stored under a key, an argument the callee may never
// force. The interpreter suspends each of those (`thunk_for`), and until this
// existed a compiled body had no way to, because a thunk is a node and a frame
// and a compiled body has no frame -- its slots are registers. So every such
// position refused the function, and with it every list builder, every map
// fold and every loop carrying a lazy accumulator.
//
// A position like that is now written one of two ways, and neither evaluates
// anything the interpreter would not have:
//
//   * **Built in place**, when producing the value cannot be observed: a
//     literal, a slot handed over as it stands, a cell or a literal list of
//     such things (a cell is already a value, and building one cannot raise),
//     and the shapes `thunk_for` itself computes -- `+ - *` of two fixnums in
//     hand, an array or list element already in hand. Those last are decided at
//     run time, exactly as `thunk_for` decides them, and fall to the second way
//     when the operands are not what they need to be.
//   * **Suspended**, otherwise: a frame is made on demand holding what the
//     slots hold right now (`dream_rt_snapshot`), and the expression is a thunk
//     against it. That is the frame the interpreter would have been running
//     this iteration in, so the thunk means what the interpreter's would have
//     meant -- and the expression inside it is the interpreter's to run, so it
//     needs no support from this file at all. One frame serves every
//     suspension an iteration makes.
//
// The first is what makes a builder cost nothing but its data:
// `upto (i + 1) n (cons i acc)` allocates one cell an iteration and no thunk,
// no frame, no trip through the interpreter. The second is what makes the
// rest *admissible*: `map f xs` still suspends `f x` and the recursion, because
// the program says to, but the loop around them is compiled.
//
// What decides whether a self call's or a peer's argument is in a lazy position
// is the callee's strictness, and that is now asked rather than required. See
// "Which arguments are evaluated" in `Analyzer::run`.
// ---------------------------------------------------------------------------

/// May `idx` be produced in a lazy position without allocating and without
/// forcing anything? The operands `+ - *` and `.[ ]` need in hand, and the
/// shapes whose production is free. Everything else is built or suspended.
bool cheap_lazy(const Image& img, const std::vector<uint32_t>& binds, uint32_t arity,
                uint32_t idx, int depth) {
    if (depth > 16) return false;
    const Node& n = img.node(idx);
    switch (Op(n.op)) {
        case Op::ConstInt:
            return fixnum_fits(img.integer(n.a));
        case Op::ConstBool: case Op::ConstChar: case Op::Unit: case Op::ConstAtom:
        case Op::Builtin: case Op::Capture:
            return true;
        case Op::Local:
            if (n.a < arity) return true;
            return n.a < binds.size() && binds[n.a] != NO_NODE &&
                   cheap_lazy(img, binds, arity, binds[n.a], depth + 1);
        case Op::Force:
            return cheap_lazy(img, binds, arity, n.a, depth + 1);
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Get:
            return cheap_lazy(img, binds, arity, n.a, depth + 1) &&
                   cheap_lazy(img, binds, arity, n.b, depth + 1);
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// What the interpreter evaluates
//
// An op this file does not write used to refuse the whole function. Measured on
// a self-compile run with every hot function compiled before it ran again
// (`DREAM_JIT_SYNC=1 DREAM_JIT_WHY=1`), that was 612 refusals against 855
// compiles, and nearly all of them were one node in an otherwise ordinary
// body: the `%{}` default in every accessor of a `mapping`, which is every
// accessor of the compiler's own state; a lambda reading what it captured; a
// lambda being made; a module member passed as a value; a `try`.
//
// A suspension already says how to have the interpreter evaluate a node:
// `suspend` makes a thunk of it against the frame this iteration would have
// had, and forcing that thunk runs the node exactly where and as the
// interpreter would have. A node in a *lazy* position has always been written
// that way. One in an evaluated position can be written the same way and then
// forced on the spot -- which is what the interpreter does when it reaches it,
// with one thunk more -- and the rest of the function is compiled around it.
//
// What that may not cover is an effect, for the reason everything in this file
// avoids one: compiled code orders nothing, and a call it makes may be run
// again from the top. So a node marked impure is refused as before, and the
// lowerer marks a node impure when anything under it is (`inherit` in
// dreams/src/lower.dr), so the mark on the node is the answer for all of it.
//
// What the interpreter evaluates is invisible to the analysis: none of it is a
// self call site, none of it counts as forcing a parameter, and none of it has
// a definite force order. Each of those walks stops at such a node, which
// claims less than the truth and is therefore always sound. The cost is the
// thunk and, where an iteration has none yet, the frame; an op that is hot
// enough to matter is worth writing out instead, as `Capture`, `Builtin` and
// the empty map literal are.
// ---------------------------------------------------------------------------

/// The head and tail of a cell `n` builds: the opcode, or a saturated call of
/// the builtin it is the opcode of, which is what an image `opt` has not
/// rewritten still holds. `{NO_NODE, NO_NODE}` for anything else.
std::pair<uint32_t, uint32_t> cons_parts(const Image& img, const Node& n) {
    if (Op(n.op) == Op::Cons) return {n.a, n.b};
    if (Op(n.op) != Op::Apply || n.c != 2) return {NO_NODE, NO_NODE};
    const Node& callee = img.node(n.a);
    if (Op(callee.op) != Op::Builtin || callee.a >= builtin_count()) return {NO_NODE, NO_NODE};
    if (std::strcmp(builtin_def(callee.a).name, "_list_cons") != 0) return {NO_NODE, NO_NODE};
    return {img.kid(n.b), img.kid(n.b + 1)};
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// The masks over a function's local slots, one bit each, in an array of words.
///
/// A single machine word held these once, and a function with more than 64
/// slots was refused for it. That was the wrong thing to give up on: the
/// compiler's own widest dispatchers -- `infer` at 115 slots, `hoist` at 125 --
/// are four of the five thousand functions in a self-compile and about 8% of
/// its reductions, and they were the four the tier could not take. So the mask
/// is wide enough for the functions that exist and then some, and `f_.slots` is
/// capped by the width rather than by a machine word. See `Analyzer::run`,
/// which reports the cap under `DREAM_JIT_WHY` like every other refusal.
constexpr uint32_t kSlotSetWords = 8;  // 512 slots; the widest is 125
constexpr uint32_t kSlotSetBits = 64 * kSlotSetWords;

struct SlotSet {
    uint64_t w[kSlotSetWords] = {};

    SlotSet() = default;
    SlotSet(uint64_t v) { w[0] = v; }

    static SlotSet bit(uint32_t i) {
        SlotSet s;
        s.set(i);
        return s;
    }
    /// The first `n` slots set, for a mask of a function's parameters.
    static SlotSet low_bits(uint32_t n) {
        SlotSet s;
        for (uint32_t i = 0; i < n && i < kSlotSetBits; ++i) s.set(i);
        return s;
    }

    bool test(uint32_t i) const {
        return i < kSlotSetBits && ((w[i >> 6] >> (i & 63)) & 1) != 0;
    }
    void set(uint32_t i) {
        if (i < kSlotSetBits) w[i >> 6] |= uint64_t(1) << (i & 63);
    }
    void clear(uint32_t i) {
        if (i < kSlotSetBits) w[i >> 6] &= ~(uint64_t(1) << (i & 63));
    }
    bool any() const {
        for (uint32_t i = 0; i < kSlotSetWords; ++i) {
            if (w[i]) return true;
        }
        return false;
    }
    explicit operator bool() const { return any(); }
    /// The low 32 bits, as a per-argument mask: call positions are capped at
    /// 32, so the argument masks the analysis answers are these low bits.
    uint32_t low32() const { return uint32_t(w[0]); }

    SlotSet& operator|=(const SlotSet& o) {
        for (uint32_t i = 0; i < kSlotSetWords; ++i) w[i] |= o.w[i];
        return *this;
    }
    SlotSet& operator&=(const SlotSet& o) {
        for (uint32_t i = 0; i < kSlotSetWords; ++i) w[i] &= o.w[i];
        return *this;
    }
    friend SlotSet operator|(SlotSet a, const SlotSet& b) { return a |= b; }
    friend SlotSet operator&(SlotSet a, const SlotSet& b) { return a &= b; }
    friend SlotSet operator~(SlotSet a) {
        for (uint32_t i = 0; i < kSlotSetWords; ++i) a.w[i] = ~a.w[i];
        return a;
    }
    friend bool operator==(const SlotSet& a, const SlotSet& b) {
        for (uint32_t i = 0; i < kSlotSetWords; ++i) {
            if (a.w[i] != b.w[i]) return false;
        }
        return true;
    }
    friend bool operator!=(const SlotSet& a, const SlotSet& b) { return !(a == b); }
};

/// How a call is made; see `Analyzer::check_apply`.
enum class CallKind : uint8_t { Self, Known, Native, Peer, Closure, Refused };

struct Analysis {
    bool compilable = false;
    SlotSet strict_params = 0;
    /// The body calls itself somewhere other than tail position, so it is
    /// emitted as a function of its arguments rather than of a frame. See the
    /// note at the top of this file.
    bool recurses = false;
    /// Slots the function carries as raw doubles: one bit per slot. Decided by
    /// the fixpoint in `Analyzer::float_slots`, guarded on entry, and boxed
    /// again at every boundary. See "Representation" above.
    SlotSet float_slots = 0;
    /// Slots to force on entry, in this order, before the float ones are
    /// unboxed. Empty unless `float_slots` is. See `Analyzer::force_order`.
    std::vector<uint32_t> force_first;
    /// The entry cannot force the float slots in the order the body would, so
    /// the first iteration is emitted a second time over tagged slots and only
    /// its tail call crosses into doubles. Loop shape only. See "The first
    /// iteration" above `Emitter::emit_loop`.
    bool peel = false;
    /// A peer only: this body returns its answer as a raw double, its bits in
    /// the return register. Decided by the type of the body, not by a
    /// signature -- see "Typed peers" above `PeerSet`.
    bool ret_dbl = false;
    /// A peer only: the parameters of its *typed* variant, which takes them as
    /// doubles, and whether that variant returns one. Zero when the signature
    /// declares no float parameter, and then there is no typed variant.
    SlotSet typed_slots = 0;
    bool typed_ret_dbl = false;
    /// The other global functions this body calls. Every one of them is a
    /// member of the compile's `PeerSet` and is emitted beside it.
    std::vector<uint32_t> calls;
    /// The body calls itself in tail position, so it is a loop. That is the
    /// best shape there is for the root of a compile and the one shape a peer
    /// may not have; `PeerSet::admit` says why.
    bool tail_self = false;
    /// Parameters every self call hands straight back to itself, one bit per
    /// slot. Such an argument is moved rather than evaluated, which is what
    /// lets it be neither forced nor proved strict. See `Analyzer::carried`.
    SlotSet carried = 0;
    /// For each slot, the statement of the one `let` that binds it, or
    /// `NO_NODE` -- for a slot no `let` binds, and for one bound in more than
    /// one place (`multi`), whose binding is whichever is in scope where it is
    /// read (`BindScope`). `bound` is every slot some `let` binds. A bound
    /// slot is never read: the expression is written where the name is. See
    /// `Analyzer::check_bind`.
    std::vector<uint32_t> unique_stmts;
    std::vector<uint8_t> bound;
    std::vector<uint8_t> multi;
    /// Parameters whose self-call arguments are evaluated before the call, one
    /// bit per slot. Every other argument is in a lazy position. See "Which
    /// arguments are evaluated" in `Analyzer::run`.
    SlotSet eager = 0;
    /// The `let` statements whose value is cheap enough to be written again at
    /// a lazy read of its name (`cheap_lazy`, and small). Otherwise such a read
    /// suspends the binding's expression, which is what the interpreter's slot
    /// would have held.
    std::unordered_set<uint32_t> inline_stmts;
    /// The `let` statements whose value is remembered once computed rather
    /// than written out again at every read. See "A binding computed once".
    std::unordered_set<uint32_t> memo_stmts;
    /// The body calls something that runs the machine underneath it -- a
    /// closure, or a function this tier did not take -- and so may spend the
    /// slice from inside the call. See `dream_rt_apply`.
    bool nested_calls = false;
    /// How each call the body evaluates is made, by node. The emitter reads
    /// this rather than deciding again.
    std::unordered_map<uint32_t, CallKind> kinds;
    /// Every recursion in the body is one of the two shapes that is a loop in
    /// disguise, so it is compiled as one, with an accumulator. `steps` are the
    /// `e + f ..` nodes and `let_steps` the reads of a `let f ..` as the answer.
    /// See "Recursion that is a loop".
    bool accumulate = false;
    std::unordered_set<uint32_t> steps;
    /// A let step is a read and the binding it reads (`let_step_key`): one
    /// node of a name is shared by every place it is written, and in another
    /// place it may name another binding.
    std::unordered_set<uint64_t> let_steps;
    /// Answers of the shape `cons h (f ..)`, a cell whose tail is a call of
    /// this function: the list is built ahead of whoever reads it, a run of
    /// cells a call. See "Building ahead".
    std::unordered_set<uint32_t> cons_steps;
    /// Nodes in evaluated positions that are suspended and forced on the spot
    /// rather than written out. See "What the interpreter evaluates".
    std::unordered_set<uint32_t> interpreted;
};

/// How many arguments a call handed to the machine may have. The helper takes
/// them in memory, so this is only a bound on the compile, not on a signature.
constexpr uint32_t kMaxApplyArgs = 16;

class PeerSet;
/// The analysis of an admitted peer, or null while it is still being decided.
const Analysis* peer_member(const PeerSet& peers, uint32_t gi);
/// `peers.admit(gi, depth)`, spelled as a free function because `Analyzer` and
/// `PeerSet` each have to name the other: deciding whether a call may be
/// compiled means analysing the callee, and analysing the callee means asking
/// the same question of everything *it* calls.
bool peer_admits(PeerSet& peers, uint32_t gi, int depth);
/// The type of a call to peer `gi` whose arguments in `float_args` are known to
/// be floats. Also a free function, for the same reason.
Ty peer_call_ty(const PeerSet& peers, uint32_t gi, SlotSet float_args);

/// `DREAM_JIT_DENY=name,#index,..` keeps the named functions out of the tier,
/// as roots and as peers alike: a name as `--profile` and `DREAM_JIT_TRACE`
/// print it, which may match several functions, or `#` and the index the
/// trace gives, which matches one. It is how a fault the interpreter does not
/// have is narrowed to the function whose compiled form has it -- halve the
/// trace's list until the fault goes -- without a rebuild per guess.
bool denied_by_env(const Image& img, uint32_t func_index) {
    static const std::vector<std::string> deny = [] {
        std::vector<std::string> out;
        const char* env = std::getenv("DREAM_JIT_DENY");
        if (!env) return out;
        std::string all(env);
        size_t start = 0;
        while (start <= all.size()) {
            size_t comma = all.find(',', start);
            if (comma == std::string::npos) comma = all.size();
            if (comma > start) out.push_back(all.substr(start, comma - start));
            start = comma + 1;
        }
        return out;
    }();
    if (deny.empty()) return false;
    StringRef name = img.str(img.func(func_index).name);
    const std::string index = "#" + std::to_string(func_index);
    for (const std::string& d : deny) {
        if (d == index || (d.size() == name.len && std::memcmp(d.data(), name.data, name.len) == 0)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Bindings in scope
//
// A slot may be bound by more than one `let`. The branches of an `if` and the
// arms of a `match` give their slots to each other, since only one of them
// runs in a call (dreams/src/scope.dr) -- which took a self-compile's frames
// down by 13% and let the tier take twice the functions, being smaller. So
// which binding a read means depends on the path to it, and the read cannot
// say: nodes are shared, and one node of `Local 3` stands for every read of
// slot 3 in the function. This used to be refused outright, "one slot, one
// binding", which after the slots were shared refused nearly every function
// with a `let` in a branch.
//
// So every walk follows the bindings in scope as it goes. Entering a block
// enters its `let`s, leaving it leaves them (`BindScope`), and a read means
// whatever `binds_[slot]` holds where the walk stands. On one path a slot is
// bound once, but for one case: an arm whose guard says no falls through to
// the arms below it, which are written inside it and bind its slots again --
// and since the failed arm made nothing that outlives its guard (scope.dr
// again), the inner binding simply shadows the outer. A fact about a binding
// is kept by its statement, not its slot; and one asked after the walk is
// asked in the scope the binding was made in, which is recorded with it.
//
// What the emitter adds is the frame: a suspension made against this
// iteration's snapshot reads its slots out of it, so a slot bound in more than
// one place holds, in each snapshot, the binding in scope where the snapshot
// was made (one bind table a site), and a `let` of such a slot met after the
// snapshot was made writes its own suspension in (`dream_rt_rebind`).

/// The bindings a walk has entered, left again when it is destroyed.
class BindScope {
public:
    BindScope(std::vector<uint32_t>& binds, std::vector<uint32_t>& stmts, uint64_t* sig)
        : binds_(binds), stmts_(stmts), sig_(sig) {}
    BindScope(const BindScope&) = delete;
    BindScope& operator=(const BindScope&) = delete;
    ~BindScope() {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) set(it->slot, it->value, it->stmt);
    }
    /// The `let` statement at `stmt`, entered.
    void enter(const Image& img, uint32_t stmt) {
        const Node& b = img.node(stmt);
        if (b.a >= binds_.size()) return;
        saved_.push_back({b.a, binds_[b.a], stmts_[b.a]});
        set(b.a, b.b, stmt);
    }
    /// Every `let` of a block, for a walk that does not visit them in order.
    void enter_block(const Image& img, const Node& block) {
        for (uint32_t i = 0; i < block.b; ++i) {
            const uint32_t stmt = img.kid(block.a + i);
            if (Op(img.node(stmt).op) == Op::Bind) enter(img, stmt);
        }
    }
    /// What `sig` is kept as: a value that differs whenever a slot's binding
    /// does, so that "this node, in this scope" can be remembered.
    static uint64_t mix(uint32_t slot, uint32_t stmt) {
        uint64_t x = (uint64_t(slot) << 32 | stmt) * 0x9E3779B97F4A7C15ull;
        return x ^ (x >> 29);
    }

private:
    void set(uint32_t slot, uint32_t value, uint32_t stmt) {
        if (sig_) *sig_ ^= mix(slot, stmts_[slot]) ^ mix(slot, stmt);
        binds_[slot] = value;
        stmts_[slot] = stmt;
    }
    struct Saved {
        uint32_t slot, value, stmt;
    };
    std::vector<uint32_t>& binds_;
    std::vector<uint32_t>& stmts_;
    uint64_t* sig_;
    std::vector<Saved> saved_;
};

/// A node, or a binding, with the scope it is met in (`BindScope::mix`).
struct PairHash {
    size_t operator()(const std::pair<uint32_t, uint64_t>& p) const {
        return size_t(p.second ^ (uint64_t(p.first) * 0x9E3779B97F4A7C15ull));
    }
};

/// A let step's key: the read, and the binding it reads there.
inline uint64_t let_step_key(uint32_t read, uint32_t stmt) { return uint64_t(read) << 32 | stmt; }

class Analyzer {
public:
    Analyzer(Runtime& rt, const Image& img, uint32_t func_index, PeerSet* peers = nullptr,
             int peer_depth = 0)
        : img_(img), fi_(func_index), f_(img.func(func_index)), rt_(rt), peers_(peers),
          peer_depth_(peer_depth) {}

    Analysis run() {
        Analysis a;
        if (f_.slots > kSlotSetBits) {
            why(img_.node(f_.body), "more slots than the tier tracks (raise kSlotSetWords)");
            return a;
        }
        if (f_.arity == 0) return a;
        if (denied_by_env(img_, fi_)) return a;
        binds_.assign(f_.slots, NO_NODE);
        bind_stmts_.assign(f_.slots, NO_NODE);
        reads_.assign(f_.slots, 0);
        const SlotSet params = SlotSet::low_bits(f_.arity);
        // Checked as though every self-call argument were evaluated, which asks
        // the most of the body: whatever the fixpoint below decides, everything
        // it leaves evaluated has been checked.
        eager_ = params;
        // A body with an effect anywhere in it -- the compiler carries
        // `F_IMPURE` up to the root from every node, lazy positions included --
        // is not compiled at all. The checks below refuse an impure call this
        // body makes, but not one it only *builds*: a lambda written inside an
        // impure function and handed to a pure peer (`str.join_str ""
        // (list.map (fn e -> .. run! e ..) es)`) sits in an argument nobody
        // walks. Compiled code that parks is retried from the top, and the
        // retry makes that lambda, and the thunks it is mapped into, afresh --
        // so whatever ran before the park ran again, on every retry: `mind
        // docs` started the same three example compiles until it was killed.
        if (img_.node(f_.body).flags & F_IMPURE) {
            why(img_.node(f_.body), "a body that performs an effect");
            return a;
        }
        if (!check(f_.body, 0)) return a;

        // A slot no parameter owns and no `let` bound holds state that lives in
        // the frame and nowhere this can reach. Nothing should produce one --
        // `Op::Bind` is the only thing that writes such a slot, and every bind
        // in the body has just been accounted for -- so this is checked rather
        // than argued, because the argument is about the compiler and this file
        // is not.
        if (reads_beyond_params_) {
            why(img_.node(f_.body), "a slot no parameter or binding owns");
            return a;
        }
        if (!bindings_substitutable()) {
            why(img_.node(f_.body), ("a binding that cannot be written where it is read: " + refused_bind_).c_str());
            return a;
        }

        // Which arguments are evaluated.
        //
        // The rule the tier rests on is about *eager evaluation*: compiled code
        // may evaluate a call's argument before making the call only where the
        // callee would have forced it anyway, because evaluating one it would
        // never have forced can raise, or diverge, in a program that was going
        // to finish quietly. This used to be a condition of admission -- every
        // parameter forced on every path, or the function was refused -- and
        // that refused every function carrying something it does not always
        // look at, which is every list builder and every lazy accumulator.
        //
        // It is a decision per parameter now. A parameter the body forces on
        // every path gets its self-call arguments evaluated; any other gets
        // them in a lazy position, built or suspended as the interpreter would
        // (see "Lazy positions"). What is circular is that evaluating an
        // argument is itself a force, so which parameters are strict depends on
        // which arguments are evaluated. That is the standard strictness
        // fixpoint: assume every argument evaluated, drop the parameters that
        // assumption does not make strict, and ask again with those arguments
        // lazy, until nothing changes. It descends, so it stops within `arity`
        // rounds, and what it stops on is consistent: every parameter left is
        // forced on every path given exactly the evaluation that is emitted.
        //
        // A parameter every self call hands straight back to itself is moved,
        // not evaluated (`carried`): slot `i` to slot `i`, whatever is in it.
        SlotSet strict = 0;
        for (int round = 0; round <= int(f_.arity) + 1; ++round) {
            call_sites_.clear();
            site_scopes_.clear();
            steps_.clear();
            let_steps_.clear();
            recurses_ = false;
            tail_self_ = false;
            collect_calls(f_.body, 0, true);
            carried_ = carried();
            strict = strict_of(f_.body, 0);
            const SlotSet next = eager_ & strict & params;
            if (next == eager_) break;
            eager_ = next;
        }

        // Each binding, in each scope it was met in: written again at a lazy
        // read only when that is cheap in all of them, and remembered once
        // computed when it calls anything in any.
        a.unique_stmts.assign(f_.slots, NO_NODE);
        a.bound.assign(f_.slots, 0);
        a.multi.assign(f_.slots, 0);
        for (uint32_t stmt : bind_order_) {
            const Node& b = img_.node(stmt);
            const bool strict = (b.flags & F_STRICT) != 0;
            bool cheap = !strict, calls = false;
            for (const Scope& sc : bind_scopes_[stmt]) {
                in_scope(sc, [&] {
                    bool c = false;
                    const uint32_t cost = expand_cost(b.b, 0, &c);
                    calls = calls || c;
                    if (cheap && (c || cost > kMaxBindDuplication || !cheap_lazy(img_, binds_, f_.arity, b.b, 0))) {
                        cheap = false;
                    }
                });
            }
            if (cheap) a.inline_stmts.insert(stmt);
            if (calls || strict) a.memo_stmts.insert(stmt);
            if (a.bound[b.a] && a.unique_stmts[b.a] != stmt) a.multi[b.a] = 1;
            a.bound[b.a] = 1;
            a.unique_stmts[b.a] = stmt;
        }
        for (uint32_t slot = 0; slot < f_.slots; ++slot) {
            if (a.multi[slot]) a.unique_stmts[slot] = NO_NODE;
        }
        a.carried = carried_;
        a.eager = eager_;
        a.nested_calls = nested_calls_;
        a.kinds = kinds_;
        a.compilable = true;
        a.strict_params = strict;
        a.recurses = recurses_;
        a.tail_self = tail_self_;
        // Only when *every* recursion is a step: one real non-tail call left
        // makes the body a function of its arguments, and a machine call is
        // then the only way to make any of them.
        if (!recurses_ && (!steps_.empty() || !let_steps_.empty())) {
            a.accumulate = true;
            a.steps = steps_;
            a.let_steps = let_steps_;
        }
        a.calls = calls_;
        a.interpreted = interp_;
        // A parameter the signature declares `:float` starts the fixpoint as
        // one instead of as nothing. That is a guess about the value the
        // caller hands in, and the entry guards it exactly as it guards a slot
        // the fixpoint found on its own; what it adds is the loop whose self
        // calls never mention a float literal -- `go (x + dx) dx (acc + x * dx)`
        // -- where nothing else would ever say that `x` is one.
        //
        // Only a parameter the body forces on every path. One it merely
        // carries may never be looked at, so the value it arrives with may be
        // a suspension nobody is allowed to force -- and a guard can only hand
        // such a call back, every time.
        const SlotSet hinted = img_.float_params(fi_) & params & strict;
        a.float_slots = float_slots(hinted, 0);
        if (a.float_slots && !entry_forces(a.float_slots, &a.force_first)) {
            // The body forces its parameters in an order the entry cannot pin
            // down, so unboxing one there would mean forcing it out of turn.
            a.force_first.clear();
            if (!recurses_) {
                // A loop runs its first iteration over tagged slots instead,
                // forcing in the body's own order, and crosses into doubles
                // at the back-edge. Nothing is forced early and nothing is
                // given up.
                a.peel = true;
            } else {
                // A recursion has no back-edge to cross at, so it keeps the
                // float slots the entry *can* reach and gives up only the
                // others -- which used to be all of them. Giving one up can
                // demote another (a slot that was a float because this one
                // was), so the fixpoint runs again with the given-up ones
                // pinned, until what is left is inside the prefix.
                std::vector<uint32_t> seq;
                force_order(f_.body, 0, seq);
                SlotSet prefix = 0;
                for (uint32_t slot : seq) prefix.set(slot);
                SlotSet pinned = 0;
                SlotSet fs = a.float_slots;
                while (fs & ~prefix) {
                    pinned |= fs & ~prefix;
                    fs = float_slots(hinted & ~pinned, pinned);
                }
                a.float_slots = fs;
                if (fs && !entry_forces(fs, &a.force_first)) {
                    a.float_slots = 0;
                    a.force_first.clear();
                }
            }
        }
        // Only the loop shape builds ahead: the run lives in its registers and
        // is handed back from its one frame. A float slot would want the
        // peeled first iteration as well, and nothing that builds a list has
        // asked for one yet.
        if (!recurses_ && !a.accumulate && !a.float_slots && peer_depth_ == 0 &&
            build_ahead_enabled()) {
            find_cons_steps(f_.body, 0, a.cons_steps);
        }
        if (peer_depth_ > 0) {
            // The two variants a peer may be emitted as, typed by the slots
            // each one has: every parameter tagged, and the declared floats
            // (plus whatever the fixpoint finds follows from them) as doubles.
            // Each returns a double when its body's type is Float -- a fact
            // about the body, true of the generic variant as much as the typed
            // one: `1.0 / (k * k + 1.0)` is a float whatever `k` is, or raises.
            float_slots(0, params);
            a.ret_dbl = ty(f_.body, 0) == Ty::Float;
            if (hinted) {
                a.typed_slots = float_slots(hinted, 0) & params;
                a.typed_ret_dbl = a.typed_slots && ty(f_.body, 0) == Ty::Float;
            }
        }
        return a;
    }

private:
    /// Is this node, and everything under it, something we can emit?
    bool check(uint32_t node, int depth) {
        if (depth > 256) return false;
        const Node& n = img_.node(node);
        Op op = Op(n.op);

        if (op == Op::Apply || is_primitive(op)) {
            if (check_apply(node, n, depth)) return true;
            // Only a call refused for what it *is*. One whose argument was
            // refused has already been offered to the interpreter at that
            // argument, and what failed there would fail here.
            if (kind_at(node) == CallKind::Refused && interpret(node, n)) return true;
            why(n, kind_at(node) == CallKind::Refused ? "a call this tier cannot make" : "an argument");
            return false;
        }
        // A capture is read out of the closure the frame holds, which a root
        // has and a peer -- a global function, entered with no frame -- does
        // not; nor would a peer have anything to capture.
        if (op == Op::Capture) {
            if (peer_depth_ == 0 && n.a < f_.n_captures) return true;
            return interpret(node, n);
        }
        // `%{}` is built in place; a literal with entries forces its keys, in
        // order, and is the interpreter's.
        if (op == Op::MakeMap) return n.b == 0 || interpret(node, n);
        // An integer too big for a fixnum is a boxed constant the emitter has
        // no way to name.
        if (op == Op::ConstInt && !fixnum_fits(img_.integer(n.a))) return interpret(node, n);
        if (!op_is_supported(op)) {
            if (interpret(node, n)) return true;
            why(n, "an operation this tier does not emit");
            return false;
        }
        if (op == Op::Local) {
            if (n.a >= f_.slots) return false;
            ++reads_[n.a];
            // A read of a slot that is neither a parameter nor bound here.
            if (n.a >= f_.arity && binds_[n.a] == NO_NODE) reads_beyond_params_ = true;
            if (bound(n.a)) ++stmt_reads_[bind_stmts_[n.a]];
            // A binding's value is written where its name is read, so it is
            // checked the first time its name is read *here* -- where it will
            // be evaluated. A binding only ever read in lazy positions is never
            // evaluated by compiled code at all: those reads suspend it, and
            // what it contains is the interpreter's business.
            //
            // A read of a binding whose check is still running is the binding
            // naming itself, which would expand for ever.
            if (bound(n.a)) {
                const uint32_t stmt = bind_stmts_[n.a];
                const uint8_t state = bind_state_[stmt];
                if (state == 1) return false;
                if (state == 0) {
                    bind_state_[stmt] = 1;
                    const bool ok = check(binds_[n.a], depth + 1);
                    bind_state_[stmt] = 2;
                    if (!ok) return false;
                }
            }
        }
        if (op == Op::Global && n.a >= img_.global_count()) return false;

        switch (op) {
            case Op::SwitchAtom: case Op::SwitchHead:
                if (!check(n.a, depth + 1)) return false;
                for (uint32_t i = 1; i < n.c; i += 2) {
                    if (!check(img_.kid(n.b + i), depth + 1)) return false;
                }
                return check(img_.kid(n.b + n.c - 1), depth + 1);
            case Op::If: {
                const int c = constant_condition(img_, n);
                if (c >= 0) return c ? check(n.b, depth + 1)
                                    : (n.c == NO_NODE || check(n.c, depth + 1));
                return check(n.a, depth + 1) && check(n.b, depth + 1) &&
                       (n.c == NO_NODE || check(n.c, depth + 1));
            }
            case Op::Block: {
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                for (uint32_t i = 0; i < n.b; ++i) {
                    uint32_t stmt = img_.kid(n.a + i);
                    if (Op(img_.node(stmt).op) == Op::Bind) {
                        if (!check_bind(stmt, scope, depth + 1)) return false;
                        continue;
                    }
                    if (!check(stmt, depth + 1)) return false;
                }
                return true;
            }
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs:
            case Op::ListIsEmpty: case Op::ListTail: case Op::ListIsCons: case Op::ListHead:
                return check(n.a, depth + 1);
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or:
                return check(n.a, depth + 1) && check(n.b, depth + 1);
            case Op::Set:
                // The value is in a lazy position and needs nothing.
                return check(n.a, depth + 1) && check(n.b, depth + 1);
            case Op::SetMany:
                // Every key is a constant and every value in a lazy position.
                return check(n.a, depth + 1);
            case Op::Get:
                return check(n.a, depth + 1) && check(n.b, depth + 1) &&
                       (n.c == NO_NODE || check(n.c, depth + 1));
            default:
                // Including `Cons`, `MakeList` and `MakeArray`, every part of
                // which is in a lazy position.
                return true;
        }
    }

    /// Have the interpreter evaluate `node` where it stands: suspended against
    /// this iteration's frame and forced at once. See "What the interpreter
    /// evaluates". False for an effect, which compiled code may not perform,
    /// and for a statement, which is not an expression with a value.
    bool interpret(uint32_t node, const Node& n) {
        if (n.flags & F_IMPURE) return false;
        if (Op(n.op) == Op::Bind || Op(n.op) == Op::Nop) return false;
        interp_.insert(node);
        nested_calls_ = true;
        return true;
    }
    bool interpreted(uint32_t node) const { return interp_.count(node) != 0; }

    /// `DREAM_JIT_WHY=1` says, for each function refused, the first node that
    /// refused it -- which is what to change in a library that wants a loop
    /// compiled. Reported innermost first, once per function.
    void why(const Node& n, const char* what) {
        static const bool on = std::getenv("DREAM_JIT_WHY") != nullptr;
        if (!on || said_why_) return;
        said_why_ = true;
        StringRef name = img_.str(img_.func(fi_).name);
        std::fprintf(stderr, "; jit why fn#%u %.*s: %s (%s)\n", fi_, int(name.len), name.data, what,
                     op_name(Op(n.op)));
    }
    bool said_why_ = false;

    /// This function calling itself with a full argument list. In tail position
    /// it becomes a loop back-edge; anywhere else a machine call, which is what
    /// `recurses_` records: the two need different shapes of function around
    /// them. A call to some *other* global function is a peer, below.
    bool is_self_call(const Node& n) const {
        if (Op(n.op) != Op::Apply || n.c != f_.arity) return false;
        const Node& callee = img_.node(n.a);
        if (Op(callee.op) != Op::Global) return false;
        const GlobalRec& g = img_.global(callee.a);
        return g.kind == GLOBAL_FUNCTION && g.target == fi_;
    }

    /// A call, and how it is made. Five shapes: this function calling itself;
    /// a numeric native compiled code makes rather than calls; a host native;
    /// a saturated call to another global function this tier can write (a
    /// peer); and anything else that can be applied at all -- a closure in a
    /// slot, a function this tier did not take, a partial or over-application
    /// -- which is handed to the machine (`dream_rt_apply`), its arguments in
    /// lazy positions and its answer forced, exactly as the interpreter's
    /// `Apply` would do it. That last shape is what lets a fold call the
    /// function it was given, and a loop call a helper that is not compiled.
    ///
    /// An impure call is refused: compiled code orders nothing, so an effect
    /// has nowhere to be ordered.
    bool check_apply(uint32_t idx, const Node& n, int depth) {
        // Decided once per node and recorded: every later walk, and the
        // emitter, read the record rather than asking again, because whether a
        // peer is admitted depends on how full the compile already is and
        // asking twice could answer twice.
        auto known = kinds_.find(idx);
        const CallKind kind = known != kinds_.end() ? known->second : classify(n, depth);
        kinds_[idx] = kind;
        if (kind == CallKind::Refused) return false;
        if (kind == CallKind::Closure) {
            if ((n.flags & F_IMPURE) || n.c > kMaxApplyArgs) return false;
            nested_calls_ = true;
            return check(n.a, depth + 1);
        }
        if (kind == CallKind::Peer) {
            const uint32_t gi = peer_target(n);
            calls_.push_back(gi);
        }
        const uint32_t mask = check_mask(n, kind);
        for (uint32_t i = 0; i < n.c; ++i) {
            if (i < 32 && !((mask >> i) & 1)) continue;
            if (!check(img_.kid(n.b + i), depth + 1)) return false;
        }
        return true;
    }

    /// Which shape a call is. Asked once per node, by `check`, which is the
    /// one walk allowed to decide a peer.
    CallKind classify(const Node& n, int depth) {
        if (is_self_call(n)) return CallKind::Self;
        const bool direct = is_primitive(Op(n.op));
        if (known_native(rt_, img_, n.a, n.c, direct) != KnownNative::None) return CallKind::Known;
        if (native_site(rt_, img_, n.a, n.c, direct).ok) return CallKind::Native;
        // A primitive node has no callee to apply; one that is not a native
        // this tier may call has no other way in.
        if (direct) return CallKind::Refused;
        const uint32_t gi = peer_target(n);
        if (gi != kNoFunc && peers_ && peer_admits(*peers_, gi, peer_depth_)) return CallKind::Peer;
        return CallKind::Closure;
    }

    /// What `check` decided about the call at `idx`. Every call a later walk
    /// reaches is one `check` walked, so the record is always there; a missing
    /// one answers the shape that evaluates nothing.
    CallKind kind_at(uint32_t idx) const {
        auto found = kinds_.find(idx);
        return found == kinds_.end() ? CallKind::Refused : found->second;
    }

    /// The arguments `check` must be able to write: every one that may be
    /// evaluated. For a peer still being decided -- a mutual recursion -- that
    /// is not known yet, so all of them.
    uint32_t check_mask(const Node& n, CallKind kind) const {
        if (kind == CallKind::Peer) {
            const Analysis* m = peer_member(*peers_, peer_target(n));
            return m ? m->eager.low32() : ~uint32_t(0);
        }
        return eval_mask(n, kind);
    }

    /// The arguments compiled code evaluates before making this call, one bit
    /// per position. What strictness may count as forced and what the emitter
    /// evaluates are both this, which is what keeps them the same set.
    uint32_t eval_mask(const Node& n, CallKind kind) const {
        switch (kind) {
            case CallKind::Self:
                return (eager_ & ~carried_).low32();
            case CallKind::Known:
                return ~uint32_t(0);
            case CallKind::Native:
                return native_site(rt_, img_, n.a, n.c, is_primitive(Op(n.op))).strict_mask;
            case CallKind::Peer: {
                // Unknown while a mutual recursion is being decided: then the
                // answer that claims nothing, which is always sound for the
                // caller's strictness. The emitter reads the final answer.
                const Analysis* m = peer_member(*peers_, peer_target(n));
                return m ? m->eager.low32() : 0;
            }
            default:
                return 0;
        }
    }

    /// The function this call names, when it is a saturated call to a global
    /// function other than this one. `kNoFunc` otherwise -- which covers a
    /// partial application, a call through a local or a closure, and a call of
    /// a module member.
    ///
    /// A self call is deliberately excluded: it has its own two shapes, and a
    /// function is not its own peer.
    uint32_t peer_target(const Node& n) const {
        if (Op(n.op) != Op::Apply) return kNoFunc;
        const Node& callee = img_.node(n.a);
        if (Op(callee.op) != Op::Global) return kNoFunc;
        const GlobalRec& g = img_.global(callee.a);
        if (g.kind != GLOBAL_FUNCTION || g.target == fi_) return kNoFunc;
        if (g.target >= img_.func_count()) return kNoFunc;
        if (n.c != img_.func(g.target).arity) return kNoFunc;
        return g.target;
    }

    // --- bindings ---------------------------------------------------------
    //
    // A `let` in a compiled body has nowhere to put its value. The interpreter
    // suspends it in a thunk against the frame and forces that thunk the first
    // time the name is read; a compiled body has no thunks, and its frame slots
    // are registers the collector cannot see.
    //
    // So the value is written where the name is *read* instead. That is exactly
    // where the interpreter would have forced the thunk, so nothing moves: a
    // binding nobody reads is never evaluated, which is what an unforced thunk
    // comes to, and a binding read once is evaluated at the read. What is lost
    // is the thunk's memory of its own answer -- read twice, the expression runs
    // twice -- which is why reading twice is only allowed for something small
    // that calls nothing.

    /// A `let` statement, entered into `scope`. Answers false to refuse the
    /// whole function.
    bool check_bind(uint32_t stmt, BindScope& scope, int depth) {
        const Node& n = img_.node(stmt);
        // Strict values are computed at the binding point and kept in the
        // same per-invocation storage used for memoized lazy bindings. This
        // admits computed match subjects without duplicating their work.
        // Into a parameter's slot, or past the end of the frame: neither is
        // something the compiler emits, and neither has a meaning here.
        if (n.a < f_.arity || n.a >= f_.slots) return false;
        // A slot bound here and elsewhere is whichever binding is in scope:
        // see "Bindings in scope". The same `bind` met again -- a `match`
        // that starts with a switch jumps *into* its chain of arms, so the
        // walk reaches an arm's binds once through the chain and once through
        // the switch -- is the same binding, in perhaps another scope.
        scope.enter(img_, stmt);
        note_binding(stmt);
        // A strict binding runs where it stands, so it is checked here. Any
        // other is checked at its first evaluated read (see `check`'s `Local`),
        // and one nobody evaluates is never checked at all.
        if (n.flags & F_STRICT) {
            const uint8_t state = bind_state_[stmt];
            if (state != 0) return state == 2;
            bind_state_[stmt] = 1;
            const bool ok = check(n.b, depth + 1);
            bind_state_[stmt] = 2;
            return ok;
        }
        return true;
    }

    /// The bindings in scope where a walk stands, kept to be stood in again.
    struct Scope {
        std::vector<uint32_t> binds, stmts;
        uint64_t sig = 0;
    };
    Scope here() const { return Scope{binds_, bind_stmts_, scope_sig_}; }

    /// `f` asked in `sc` rather than in the scope the walk is in.
    template <class F>
    void in_scope(const Scope& sc, F f) {
        std::vector<uint32_t> b = sc.binds, s = sc.stmts;
        std::swap(binds_, b);
        std::swap(bind_stmts_, s);
        const uint64_t sig = scope_sig_;
        scope_sig_ = sc.sig;
        f();
        std::swap(binds_, b);
        std::swap(bind_stmts_, s);
        scope_sig_ = sig;
    }

    /// Remember the binding `stmt` and the scope it was met in, once a scope.
    void note_binding(uint32_t stmt) {
        auto& scopes = bind_scopes_[stmt];
        if (scopes.empty()) bind_order_.push_back(stmt);
        for (const Scope& sc : scopes) {
            if (sc.sig == scope_sig_) return;
        }
        scopes.push_back(here());
    }

    /// Is `slot` a `let`'s here?
    bool bound(uint32_t slot) const { return slot >= f_.arity && slot < binds_.size() && binds_[slot] != NO_NODE; }
    /// Is it a strict one's?
    bool strict_bound(uint32_t slot) const {
        return bound(slot) && (img_.node(bind_stmts_[slot]).flags & F_STRICT);
    }

    /// May every `let` in the body be written where its name is read?
    ///
    /// Read once or not at all, always. Read more than once, only when the
    /// value is small and contains no call: "once per read" then costs a couple
    /// of instructions, where a duplicated call could be a duplicated
    /// recursion, and nested duplication is how a linear body becomes an
    /// exponential one.
    /// Which binding `bindings_substitutable` refused, for `DREAM_JIT_WHY`.
    std::string refused_bind_;

    bool bindings_substitutable() {
        for (uint32_t stmt : bind_order_) {
            const Node& b = img_.node(stmt);
            const uint32_t reads = stmt_reads_[stmt];
            // A strict binding is computed by its own statement, before any
            // read of it can run, and every read after is a load of what it
            // remembered (`Emitter::settled_`) -- no code is written twice,
            // so nothing here bounds it. Most are a `match` on a call.
            if (reads <= 1 || (b.flags & F_STRICT)) continue;
            for (const Scope& sc : bind_scopes_[stmt]) {
                bool calls = false;
                uint32_t cost = 0;
                in_scope(sc, [&] { cost = expand_cost(b.b, 0, &calls); });
                // A binding that calls anything is remembered once computed, so
                // reading it twice costs code, not work: the bound is on the code.
                if (cost > (calls ? kMaxMemoCode : kMaxBindDuplication)) {
                    refused_bind_ = "slot " + std::to_string(b.a) + ", read " + std::to_string(reads) +
                                    " times, " + std::to_string(cost) + " nodes of " +
                                    op_name(Op(img_.node(b.b).op)) + (calls ? " that calls" : "");
                    return false;
                }
            }
        }
        return true;
    }

    /// How many nodes `idx` becomes once every `let` inside it has been written
    /// out, and whether any of it is a call. Only asked of a body `check` has
    /// already accepted, so every op it meets is one this tier emits.
    uint32_t expand_cost(uint32_t idx, int depth, bool* calls) {
        if (depth > 64) {
            // Too deep to measure. Say the one thing that refuses it either way.
            *calls = true;
            return kMaxBindDuplication + 1;
        }
        if (interpreted(idx)) {
            // Run by the machine, so as much a call as any: a binding of one
            // is remembered rather than run again at every read.
            *calls = true;
            return 1;
        }
        const Node& n = img_.node(idx);
        switch (Op(n.op)) {
            case Op::Local:
                // A strict binding is a load wherever it is read (its statement
                // ran first: `Emitter::settled_`), so a name for one -- what a
                // pattern binds a `match`'s subject to -- costs what a load does.
                if (bound(n.a) && !strict_bound(n.a)) return expand_cost(binds_[n.a], depth + 1, calls);
                return 1;
            DREAM_PRIMITIVE_CASES
            case Op::Apply:
                *calls = true;
                return 1;
            case Op::SwitchAtom: case Op::SwitchHead: {
                uint32_t c = 1 + expand_cost(n.a, depth + 1, calls);
                for (uint32_t i = 1; i < n.c; i += 2)
                    c += expand_cost(img_.kid(n.b + i), depth + 1, calls);
                return c + expand_cost(img_.kid(n.b + n.c - 1), depth + 1, calls);
            }
            case Op::If: {
                if (constant_condition(img_, n) == 1) return expand_cost(n.b, depth + 1, calls);
                if (constant_condition(img_, n) == 0)
                    return n.c == NO_NODE ? 1 : expand_cost(n.c, depth + 1, calls);
                uint32_t c = 1 + expand_cost(n.a, depth + 1, calls) +
                             expand_cost(n.b, depth + 1, calls);
                if (n.c != NO_NODE) c += expand_cost(n.c, depth + 1, calls);
                return c;
            }
            case Op::Block: {
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                scope.enter_block(img_, n);
                uint32_t c = 1;
                for (uint32_t i = 0; i < n.b; ++i) {
                    const uint32_t stmt = img_.kid(n.a + i);
                    const Node& sn = img_.node(stmt);
                    c += expand_cost(Op(sn.op) == Op::Bind ? sn.b : stmt, depth + 1, calls);
                }
                return c;
            }
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs:
            case Op::ListIsEmpty: case Op::ListTail: case Op::ListIsCons: case Op::ListHead:
                return 1 + expand_cost(n.a, depth + 1, calls);
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or:
                return 1 + expand_cost(n.a, depth + 1, calls) +
                       expand_cost(n.b, depth + 1, calls);
            case Op::Get:
                return 1 + expand_cost(n.a, depth + 1, calls) +
                       expand_cost(n.b, depth + 1, calls) +
                       (n.c == NO_NODE ? 0 : expand_cost(n.c, depth + 1, calls));
            case Op::Set: case Op::Cons: case Op::MakeList: case Op::MakeArray:
            case Op::RecordMake: case Op::SetMany:
                // Each builds something, and building it twice is two of it
                // where the program had one -- sharing lost, not just time.
                *calls = true;
                return 1;
            default:
                return 1;
        }
    }

    /// Parameters every self call hands straight back to itself.
    ///
    /// `loop s (i + 1) n acc` in a function whose first parameter is `s`: the
    /// argument is slot 0 and the position is slot 0, so the call moves nothing
    /// and evaluates nothing. Every site has to agree -- one that computes a new
    /// value there is an evaluation like any other, and the parameter is back to
    /// needing the strictness the rest of the tier asks for.
    ///
    /// Asked of the self call sites only, which is what `collect_calls`
    /// gathers: a peer's arguments are evaluated at the call, so nothing is
    /// carried across one.
    SlotSet carried() const {
        if (call_sites_.empty()) return 0;
        SlotSet bits = SlotSet::low_bits(f_.arity);
        for (uint32_t site : call_sites_) {
            const Node& n = img_.node(site);
            for (uint32_t i = 0; i < f_.arity; ++i) {
                const Node& arg = img_.node(img_.kid(n.b + i));
                if (Op(arg.op) != Op::Local || arg.a != i) bits.clear(i);
            }
        }
        return bits;
    }

    /// Slots definitely forced when this node is reduced to WHNF.
    SlotSet strict_of(uint32_t node, int depth) {
        if (depth > 256) return 0;
        // The interpreter may force anything under it, but nothing it forces
        // is something compiled code was told it may evaluate early.
        if (interpreted(node)) return 0;
        const Node& n = img_.node(node);
        switch (Op(n.op)) {
            case Op::Bind:
                return (n.flags & F_STRICT) ? strict_of(n.b, depth + 1) : 0;
            case Op::Local:
                // A bound slot is the expression it was bound to, written here.
                if (bound(n.a)) return strict_of(binds_[n.a], depth + 1);
                return SlotSet::bit(n.a);
            case Op::SwitchAtom: case Op::SwitchHead: {
                SlotSet arms = strict_of(img_.kid(n.b + n.c - 1), depth + 1);
                for (uint32_t i = 1; i < n.c; i += 2)
                    arms &= strict_of(img_.kid(n.b + i), depth + 1);
                return strict_of(n.a, depth + 1) | arms;
            }
            case Op::If:
                if (constant_condition(img_, n) == 1) return strict_of(n.b, depth + 1);
                if (constant_condition(img_, n) == 0)
                    return n.c == NO_NODE ? 0 : strict_of(n.c, depth + 1);
                // The condition always runs; a branch only counts when both
                // arms force it.
                return strict_of(n.a, depth + 1) |
                       (n.c == NO_NODE
                            ? 0
                            : (strict_of(n.b, depth + 1) & strict_of(n.c, depth + 1)));
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
                return strict_of(n.a, depth + 1) | strict_of(n.b, depth + 1);
            case Op::And: case Op::Or:
                // Short-circuiting: only the left operand is certain.
                return strict_of(n.a, depth + 1);
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs:
            case Op::ListIsEmpty: case Op::ListTail: case Op::ListIsCons: case Op::ListHead:
                return strict_of(n.a, depth + 1);
            case Op::Get: case Op::Set:
                // The container and the key; a `Get`'s default only sometimes,
                // and a `Set`'s value never.
                return strict_of(n.a, depth + 1) | strict_of(n.b, depth + 1);
            case Op::SetMany:
                // The container; the keys are constants and the values lazy.
                return strict_of(n.a, depth + 1);
            case Op::Block: {
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                scope.enter_block(img_, n);
                SlotSet s = 0;
                for (uint32_t i = 0; i < n.b; ++i) {
                    uint32_t stmt = img_.kid(n.a + i);
                    bool last = (i + 1 == n.b);
                    if (last || (img_.node(stmt).flags & F_STRICT)) {
                        s |= strict_of(stmt, depth + 1);
                    }
                }
                return s;
            }
            DREAM_PRIMITIVE_CASES
            case Op::Apply: {
                // A call forces the arguments compiled code evaluates for it
                // and no others -- `eval_mask`, which is also what the emitter
                // evaluates. Taking the union of all of them would be claiming
                // a parameter strict on the strength of an argument nobody
                // looks at, and this set is what licenses evaluating one.
                //
                // A carried argument is moved, not evaluated, so it forces
                // nothing; `eval_mask` leaves it out for that reason. A closure
                // call forces its callee and nothing else.
                const CallKind kind = kind_at(node);
                if (kind == CallKind::Closure) return strict_of(n.a, depth + 1);
                if (kind == CallKind::Refused) return 0;
                const uint32_t mask = eval_mask(n, kind);
                SlotSet s = 0;
                for (uint32_t i = 0; i < n.c && i < 32; ++i) {
                    if ((mask >> i) & 1) s |= strict_of(img_.kid(n.b + i), depth + 1);
                }
                return s;
            }
            default:
                return 0;
        }
    }

    // --- which values are floats ------------------------------------------
    //
    // A fixpoint over the parameters, ascending the lattice in "Representation"
    // above. A slot's type is the join of what every self call puts in it; the
    // value it arrives with contributes nothing, because that one is guarded
    // rather than inferred. Each of the arity slots can rise at most twice
    // (Unknown to Float to Any), so the loop is bounded by that and terminates
    // on the round that changes nothing.

    /// One bit per slot the function can carry as a raw double. `seed` is
    /// where the fixpoint starts at Float rather than at Unknown -- the
    /// parameters a signature declared -- and `pinned` is where it starts at
    /// Any and so stays.
    SlotSet float_slots(SlotSet seed, SlotSet pinned) {
        slot_ty_.assign(f_.slots, Ty::Unknown);
        for (uint32_t i = f_.arity; i < f_.slots; ++i) slot_ty_[i] = Ty::Any;
        for (uint32_t i = 0; i < f_.arity; ++i) {
            // A parameter whose arguments arrive in a lazy position arrives
            // tagged, and may arrive suspended: it cannot be a double.
            const bool lazy = !(eager_ | carried_).test(i);
            if (pinned.test(i) || lazy) slot_ty_[i] = Ty::Any;
            else if (seed.test(i)) slot_ty_[i] = Ty::Float;
        }
        // The sites were gathered in `run`, which needs them before this does.

        const int rounds = int(2 * f_.arity) + 2;
        for (int round = 0; round < rounds; ++round) {
            bool changed = false;
            for (size_t k = 0; k < call_sites_.size(); ++k) {
                const Node& n = img_.node(call_sites_[k]);
                in_scope(site_scopes_[k], [&] {
                    for (uint32_t i = 0; i < n.c && i < f_.arity; ++i) {
                        Ty t = join(slot_ty_[i], ty(img_.kid(n.b + i), 0));
                        if (t != slot_ty_[i]) {
                            slot_ty_[i] = t;
                            changed = true;
                        }
                    }
                });
            }
            if (!changed) break;
        }

        SlotSet bits = 0;
        for (uint32_t i = 0; i < f_.slots; ++i) {
            if (slot_ty_[i] == Ty::Float) bits.set(i);
        }
        return bits;
    }

    // --- what the body forces, and in what order ---------------------------
    //
    // A slot carried as a double has to be a double before the loop starts, and
    // the value the caller left in the frame is usually a thunk -- the argument
    // of a fused fold is `acc + f x`, which `thunk_for` suspends because it is
    // not already a value. So the entry has to force it, and forcing is the one
    // thing the tier has been careful never to do out of turn: a parameter that
    // raises must raise where the interpreter would have raised it, which is
    // wherever the body first looks at it.
    //
    // So the entry forces a *prefix* of the body's own order and no more. What
    // is below computes that order for as far as it is the same on every path;
    // where two branches disagree the sequence stops, and if a float slot is
    // not inside the part that is definite, the specialization is dropped
    // rather than the order. Slots after the prefix are left to `load_slot`,
    // which forces each on first read exactly as before.

    /// Append the slots `idx` definitely forces, in order. Answers false once
    /// the order stops being the same on every path, after which the caller
    /// must stop appending.
    bool force_order(uint32_t idx, int depth, std::vector<uint32_t>& out) {
        if (depth > 128) return false;
        // Work the interpreter does, which can raise: nothing after it is
        // definite.
        if (interpreted(idx)) return false;
        const Node& n = img_.node(idx);
        switch (Op(n.op)) {
            case Op::Capture:
                // Forcing what was captured is work that can raise.
                return false;
            case Op::Bind:
                return !(n.flags & F_STRICT) || force_order(n.b, depth + 1, out);
            case Op::Local:
                if (bound(n.a)) return force_order(binds_[n.a], depth + 1, out);
                out.push_back(n.a);
                return true;
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs: case Op::ListIsEmpty:
            case Op::ListIsCons:
                return force_order(n.a, depth + 1, out);
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
                return force_order(n.a, depth + 1, out) && force_order(n.b, depth + 1, out);
            case Op::ListTail: case Op::ListHead:
                // Then the tail is forced, which is work that can raise, so
                // nothing after it is definite.
                force_order(n.a, depth + 1, out);
                return false;
            case Op::Get: case Op::Set:
                if (force_order(n.a, depth + 1, out)) force_order(n.b, depth + 1, out);
                return false;
            case Op::SetMany:
                force_order(n.a, depth + 1, out);
                return false;
            case Op::Global:
                // Naming a global value forces it the first time, and that can
                // raise.
                return false;
            case Op::And: case Op::Or:
                // Short-circuiting: whether the right side runs at all depends
                // on the left, so the order is only definite up to here.
                force_order(n.a, depth + 1, out);
                return false;
            case Op::SwitchAtom: case Op::SwitchHead:
                // The head may itself force work before any arm executes.
                // Keep the entry prefix to the subject; peeling handles the rest.
                force_order(n.a, depth + 1, out);
                return false;
            case Op::If: {
                if (constant_condition(img_, n) == 1) return force_order(n.b, depth + 1, out);
                if (constant_condition(img_, n) == 0)
                    return n.c == NO_NODE || force_order(n.c, depth + 1, out);
                if (!force_order(n.a, depth + 1, out)) return false;
                if (n.c == NO_NODE) return false;
                std::vector<uint32_t> t, e;
                force_order(n.b, depth + 1, t);
                force_order(n.c, depth + 1, e);
                size_t i = 0;
                while (i < t.size() && i < e.size() && t[i] == e[i]) out.push_back(t[i++]);
                return false;
            }
            case Op::Block: {
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                scope.enter_block(img_, n);
                for (uint32_t i = 0; i < n.b; ++i) {
                    uint32_t stmt = img_.kid(n.a + i);
                    bool last = (i + 1 == n.b);
                    if (!last && !(img_.node(stmt).flags & F_STRICT)) continue;
                    if (!force_order(stmt, depth + 1, out)) return false;
                }
                return true;
            }
            DREAM_PRIMITIVE_CASES
            case Op::Apply: {
                // Only what is evaluated, so a carried argument -- moved, not
                // evaluated -- lets the order carry on past it.
                const CallKind kind = kind_at(idx);
                if (kind == CallKind::Closure) {
                    force_order(n.a, depth + 1, out);
                    return false;
                }
                const uint32_t mask = kind == CallKind::Refused ? 0 : eval_mask(n, kind);
                for (uint32_t i = 0; i < n.c && i < 32; ++i) {
                    if (!((mask >> i) & 1)) continue;
                    if (!force_order(img_.kid(n.b + i), depth + 1, out)) return false;
                }
                // A tail call transfers control and a machine call comes back
                // with a value, not with a slot; either way nothing follows.
                return false;
            }
            default:
                return true;
        }
    }

    /// The slots to force on entry, in order, so that every float slot has been
    /// reached. False when one of them is not inside the definite part of the
    /// order, which is when the specialization has to be given up.
    bool entry_forces(SlotSet floats, std::vector<uint32_t>* out) {
        std::vector<uint32_t> seq;
        force_order(f_.body, 0, seq);

        SlotSet seen = 0;
        SlotSet want = floats;
        for (uint32_t slot : seq) {
            if (seen.test(slot)) continue;
            seen.set(slot);
            out->push_back(slot);
            want.clear(slot);
            if (!want) return true;
        }
        return false;
    }

    /// The self calls compiled code will make, and whether any of them is not
    /// a tail call. Follows exactly what is evaluated: a call inside a lazy
    /// position is suspended and made by the interpreter, if by anyone, so it
    /// is not a site, does not make the body recursive, and says nothing about
    /// what the loop carries.
    ///
    /// `tail` is whether `idx` is in result position -- its value is the
    /// body's -- which is what recognises the two shapes of recursion that are
    /// compiled as a loop after all. See "Recursion that is a loop" below.
    /// Every answer of the body that is `cons h (f ..)` with `f` this
    /// function, saturated. The walk is the one `collect_calls` makes of the
    /// answer positions.
    void find_cons_steps(uint32_t idx, int depth, std::unordered_set<uint32_t>& out) const {
        if (depth > 256) return;
        const Node& n = img_.node(idx);
        switch (Op(n.op)) {
            case Op::If: {
                const int k = constant_condition(img_, n);
                if (k >= 0) {
                    const uint32_t arm = k ? n.b : n.c;
                    if (arm != NO_NODE) find_cons_steps(arm, depth + 1, out);
                    return;
                }
                find_cons_steps(n.b, depth + 1, out);
                if (n.c != NO_NODE) find_cons_steps(n.c, depth + 1, out);
                return;
            }
            case Op::Block: {
                if (n.b == 0) return;
                const uint32_t last = img_.kid(n.a + n.b - 1);
                if (Op(img_.node(last).op) != Op::Bind) find_cons_steps(last, depth + 1, out);
                return;
            }
            case Op::SwitchAtom: case Op::SwitchHead:
                for (uint32_t i = 1; i < n.c; i += 2) find_cons_steps(img_.kid(n.b + i), depth + 1, out);
                find_cons_steps(img_.kid(n.b + n.c - 1), depth + 1, out);
                return;
            default: {
                const uint32_t tail = cons_parts(img_, n).second;
                if (tail != NO_NODE && is_self_call(img_.node(tail))) out.insert(idx);
                return;
            }
        }
    }

    /// A self call compiled code makes, and the scope it is made in, which is
    /// where its arguments' types are asked (`float_slots`).
    void add_site(uint32_t idx) {
        call_sites_.push_back(idx);
        site_scopes_.push_back(here());
    }

    void collect_calls(uint32_t idx, int depth, bool tail = false) {
        if (depth > 256) return;
        // A self call the interpreter makes is an ordinary call, entering
        // this function afresh; it is not a site of this body's.
        if (interpreted(idx)) return;
        const Node& n = img_.node(idx);
        if (Op(n.op) == Op::Apply || is_primitive(Op(n.op))) {
            const CallKind kind = kind_at(idx);
            if (kind == CallKind::Self) {
                add_site(idx);
                if (n.flags & F_TAIL) tail_self_ = true;
                else recurses_ = true;
            }
            collect_args(idx, n, kind, depth);
            return;
        }
        switch (Op(n.op)) {
            case Op::Local:
                // A `let` is written where its name is read -- here, since this
                // walk only reaches evaluated positions.
                if (bound(n.a)) {
                    const uint32_t value = binds_[n.a];
                    if (tail && kind_at(value) == CallKind::Self &&
                        !(img_.node(value).flags & F_TAIL)) {
                        // `let rest = f ..; .. rest` read as the answer: a tail
                        // call written through a name.
                        add_site(value);
                        let_steps_.insert(let_step_key(idx, bind_stmts_[n.a]));
                        collect_args(value, img_.node(value), CallKind::Self, depth);
                        return;
                    }
                    collect_calls(value, depth + 1);
                }
                return;
            case Op::SwitchAtom: case Op::SwitchHead:
                collect_calls(n.a, depth + 1);
                for (uint32_t i = 1; i < n.c; i += 2)
                    collect_calls(img_.kid(n.b + i), depth + 1, tail);
                collect_calls(img_.kid(n.b + n.c - 1), depth + 1, tail);
                return;
            case Op::If:
                if (constant_condition(img_, n) >= 0) {
                    uint32_t arm = constant_condition(img_, n) ? n.b : n.c;
                    if (arm != NO_NODE) collect_calls(arm, depth + 1, tail);
                    return;
                }
                collect_calls(n.a, depth + 1);
                collect_calls(n.b, depth + 1, tail);
                if (n.c != NO_NODE) collect_calls(n.c, depth + 1, tail);
                return;
            case Op::Block: {
                // A strict `let` runs where it stands; any other is reached
                // through its name, above. The statements that are not the last
                // run only when strict, as `block` emits them.
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                scope.enter_block(img_, n);
                for (uint32_t i = 0; i < n.b; ++i) {
                    const uint32_t stmt = img_.kid(n.a + i);
                    const Node& sn = img_.node(stmt);
                    const bool last = i + 1 == n.b;
                    if (Op(sn.op) == Op::Bind) {
                        if (sn.flags & F_STRICT) collect_calls(sn.b, depth + 1);
                        continue;
                    }
                    if (last || (sn.flags & F_STRICT)) collect_calls(stmt, depth + 1, last && tail);
                }
                return;
            }
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs:
            case Op::ListIsEmpty: case Op::ListTail: case Op::ListIsCons: case Op::ListHead:
                collect_calls(n.a, depth + 1);
                return;
            case Op::Add:
                if (tail) {
                    // `e + f ..` as the answer: the left side first, and when it
                    // makes no self call of its own, the right side is a step.
                    const size_t before = call_sites_.size();
                    collect_calls(n.a, depth + 1);
                    const Node& r = img_.node(n.b);
                    if (call_sites_.size() == before && kind_at(n.b) == CallKind::Self &&
                        !(r.flags & F_TAIL)) {
                        add_site(n.b);
                        steps_.insert(idx);
                        collect_args(n.b, r, CallKind::Self, depth);
                        return;
                    }
                    collect_calls(n.b, depth + 1);
                    return;
                }
                collect_calls(n.a, depth + 1);
                collect_calls(n.b, depth + 1);
                return;
            case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or: case Op::Set:
                collect_calls(n.a, depth + 1);
                collect_calls(n.b, depth + 1);
                return;
            case Op::SetMany:
                collect_calls(n.a, depth + 1);
                return;
            case Op::Get:
                collect_calls(n.a, depth + 1);
                collect_calls(n.b, depth + 1);
                if (n.c != NO_NODE) collect_calls(n.c, depth + 1);
                return;
            default:
                return;
        }
    }

    /// The evaluated parts of a call: its callee when the machine makes it,
    /// and the arguments compiled code evaluates for it.
    void collect_args(uint32_t idx, const Node& n, CallKind kind, int depth) {
        (void)idx;
        if (kind == CallKind::Closure) {
            collect_calls(n.a, depth + 1);
            return;
        }
        // A peer still being decided may evaluate any of them.
        const uint32_t mask = kind == CallKind::Refused ? 0 : check_mask(n, kind);
        for (uint32_t i = 0; i < n.c && i < 32; ++i) {
            if ((mask >> i) & 1) collect_calls(img_.kid(n.b + i), depth + 1);
        }
    }

    const Image& img_;
    uint32_t fi_;
    const FuncRec& f_;
    Runtime& rt_;
    PeerSet* peers_ = nullptr;
    int peer_depth_ = 0;
    std::vector<Ty> slot_ty_;
    std::vector<uint32_t> call_sites_;
    std::vector<uint32_t> calls_;
    /// Per slot, the value and the statement of the `let` in scope where a
    /// walk stands, or `NO_NODE` (see "Bindings in scope"); and how many
    /// times the body reads each slot, filled in by `check` as it walks.
    std::vector<uint32_t> binds_;
    std::vector<uint32_t> bind_stmts_;
    uint64_t scope_sig_ = 0;
    std::vector<uint32_t> reads_;
    /// Per `let` statement: how many times it is read, and every scope it
    /// was met in, in the order the statements were first met.
    std::unordered_map<uint32_t, uint32_t> stmt_reads_;
    std::unordered_map<uint32_t, std::vector<Scope>> bind_scopes_;
    std::vector<uint32_t> bind_order_;
    std::vector<Scope> site_scopes_;
    /// Parameters every self call moves rather than evaluates. Computed in
    /// `run` before `strict_of`, which has to know.
    SlotSet carried_ = 0;
    /// Parameters whose self-call arguments are evaluated. See "Which
    /// arguments are evaluated" in `run`.
    SlotSet eager_ = 0;
    /// Per `let` statement: 0 not yet checked, 1 being checked, 2 checked.
    std::unordered_map<uint32_t, uint8_t> bind_state_;
    /// How each call `check` reached is made. See `check_apply`.
    std::unordered_map<uint32_t, CallKind> kinds_;
    /// The two shapes of step found by `collect_calls`: an `e + f ..` in
    /// result position, and a `let` of a self call read as the answer.
    std::unordered_set<uint32_t> steps_;
    std::unordered_set<uint64_t> let_steps_;
    /// See "What the interpreter evaluates".
    std::unordered_set<uint32_t> interp_;
    bool nested_calls_ = false;
    bool recurses_ = false;
    bool tail_self_ = false;
    bool reads_beyond_params_ = false;

public:
    /// The type of one node, given what is known about the slots. Public
    /// because the emitter asks it the same question while it writes code, and
    /// the two must give the same answer at every node or a value would be
    /// carried in one representation and read in the other.
    Ty ty(uint32_t idx, int depth) {
        if (depth > 256) return Ty::Any;
        if (interpreted(idx)) return Ty::Any;
        const Node& n = img_.node(idx);
        switch (Op(n.op)) {
            case Op::ConstFloat: return Ty::Float;
            case Op::Local:
                if (bound(n.a)) return ty(binds_[n.a], depth + 1);
                return n.a < slot_ty_.size() ? slot_ty_[n.a] : Ty::Any;
            case Op::Force: case Op::Neg: return ty(n.a, depth + 1);
            case Op::SwitchAtom: case Op::SwitchHead: {
                Ty t = ty(img_.kid(n.b + n.c - 1), depth + 1);
                for (uint32_t i = 1; i < n.c; i += 2)
                    t = join(t, ty(img_.kid(n.b + i), depth + 1));
                return t;
            }
            case Op::If: {
                if (constant_condition(img_, n) == 1) return ty(n.b, depth + 1);
                if (constant_condition(img_, n) == 0)
                    return n.c == NO_NODE ? Ty::Any : ty(n.c, depth + 1);
                // An arm that transfers control -- a tail call -- contributes
                // nothing, which is what makes the type of a loop body the type
                // of its base case.
                Ty t = ty(n.b, depth + 1);
                Ty e = n.c == NO_NODE ? Ty::Any : ty(n.c, depth + 1);
                return join(t, e);
            }
            case Op::Block: {
                if (n.b == 0) return Ty::Any;
                BindScope scope(binds_, bind_stmts_, &scope_sig_);
                scope.enter_block(img_, n);
                return ty(img_.kid(n.a + n.b - 1), depth + 1);
            }
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
                return arith_ty(ty(n.a, depth + 1), ty(n.b, depth + 1));
            DREAM_PRIMITIVE_CASES
            case Op::Apply: {
                // A tail self call produces no value; a non-tail one comes back
                // through the return register, which is a tagged `Value`
                // whatever the body computed.
                const CallKind kind = kind_at(idx);
                if (kind == CallKind::Self) return (n.flags & F_TAIL) ? Ty::Unknown : Ty::Any;
                if (kind == CallKind::Closure || kind == CallKind::Refused) return Ty::Any;
                if (peers_ && kind == CallKind::Peer) {
                    const uint32_t gi = peer_target(n);
                    if (gi != kNoFunc) {
                        SlotSet floats = 0;
                        for (uint32_t i = 0; i < n.c; ++i) {
                            if (ty(img_.kid(n.b + i), depth + 1) == Ty::Float) floats.set(i);
                        }
                        return peer_call_ty(*peers_, gi, floats);
                    }
                }
                switch (known_native(rt_, img_, n.a, n.c, is_primitive(Op(n.op)))) {
                    case KnownNative::ToFloat:
                    case KnownNative::Sqrt:
                        return Ty::Float;
                    case KnownNative::Abs:
                        // `abs` keeps the shape it was given: an integer stays
                        // an integer and a float stays a float.
                        return ty(img_.kid(n.b), depth + 1) == Ty::Float ? Ty::Float : Ty::Any;
                    default:
                        return Ty::Any;
                }
            }
            default: return Ty::Any;
        }
    }
};

// ---------------------------------------------------------------------------
// Typed peers
//
// A peer used to be all-tagged, for a reason that was right: its signature has
// to be the same for every caller, and a double parameter is a decision about
// what callers pass. So a float helper called from a compiled float loop cost
// a box for every argument going in and one for the answer coming out -- three
// allocations an iteration for `acc + term (to_float i)`, which is the whole of
// what that loop allocated.
//
// A signature is a decision that does not depend on the caller, which is what
// the missing piece was. A peer whose signature declares a float parameter is
// emitted twice: the generic variant, every parameter tagged, as before; and a
// typed one, the declared floats (and whatever the fixpoint finds follows from
// them) as doubles. A call site whose arguments are already doubles calls the
// typed variant with them in registers. One holding a tagged value checks it
// -- it is evaluated, so a float is a float box -- and calls the typed variant
// if every such argument is one and the generic variant otherwise, which is
// where an integer the types allowed an unannotated caller to pass ends up,
// and it gets exactly what the interpreter would give it.
//
// Either variant hands back a raw double when its *body* is a float -- a fact
// the analysis establishes, not the signature, and so true of untyped peers
// too: `1.0 / (k * k + 1.0)` is a float whatever `k` is, or it raises. The
// bits travel in the return register the tagged value used, and are only read
// once the status says `JIT_OK`, because on any other status that register
// holds an error or a placeholder. The caller's analysis knows which variant
// answers what (`peer_call_ty`), so a loop accumulating a peer's answer carries
// that accumulator as a double as well.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The functions one compile reaches
// ---------------------------------------------------------------------------

/// The other global functions a compilation may call, and the analysis of each.
///
/// The tier used to compile exactly one Dream-level call: this function calling
/// itself. Everything else refused the whole function, which is why a fused
/// `collatz` -- a loop whose body is `acc + collatz_steps x` -- ran interpreted
/// while `collatz_steps` beside it was compiled. A call to another global
/// function is now compiled the way a non-tail self call already was: the
/// callee is emitted as a function of its arguments, into the same module, and
/// called. There is no frame, no thunk and no trip through the interpreter.
///
/// The soundness argument is the one the tier already rests on. Compiled code
/// evaluates an argument before making a call only where the callee would have
/// forced it anyway, because evaluating one it would never have forced can
/// raise in a program that was going to finish quietly -- so a caller needs the
/// callee's strictness, per parameter. Asking that of the callee is asking
/// exactly what the tier asks of itself, which is why the test below is
/// `Analyzer::run` and not a second set of rules.
///
/// Mutual recursion makes the question circular: `f` is admissible if `g` is,
/// and `g` is admissible if `f` is. What is being asked is "nothing anywhere in
/// this closure is unsupported", which is a greatest fixpoint, so a function
/// whose own decision is still running is assumed admissible. That assumption
/// can be wrong -- the decision underneath it may come back no -- and `sound()`
/// is what catches it, rather than an argument about which cases can happen.
class PeerSet {
  public:
    PeerSet(Runtime& rt, const Image& img) : rt_(rt), img_(img) {}

    bool admit(uint32_t gi, int depth);

    /// Every call every member makes, and every call the root makes, is to a
    /// member. False means the optimistic answer above was taken somewhere and
    /// turned out wrong, and the whole compile is given up.
    bool sound(const Analysis& root) const {
        for (uint32_t gi : root.calls) {
            if (!members.count(gi)) return false;
        }
        for (const auto& entry : members) {
            for (uint32_t gi : entry.second.calls) {
                if (!members.count(gi)) return false;
            }
        }
        return true;
    }

    /// Admitted functions, by image function index.
    std::unordered_map<uint32_t, Analysis> members;

  private:
    enum class State : uint8_t { Deciding, Yes, No };
    Runtime& rt_;
    const Image& img_;
    std::unordered_map<uint32_t, State> state_;
};

bool PeerSet::admit(uint32_t gi, int depth) {
    if (depth > kMaxPeerDepth) return false;
    auto seen = state_.find(gi);
    if (seen != state_.end()) return seen->second != State::No;
    if (members.size() >= kMaxPeers) return false;
    if (gi >= img_.func_count()) return false;

    state_[gi] = State::Deciding;
    Analysis a = Analyzer(rt_, img_, gi, this, depth + 1).run();

    // A peer is emitted as a function of its arguments and is given no frame of
    // its own. Reading a slot no parameter owns is already refused for every
    // function, peer or not, so what is left to refuse here is one thing:
    //
    // A peer may not be a loop. A compiled loop stays preemptible by writing its
    // loop-carried state back to its frame and handing control to the
    // interpreter, and a peer has no frame to write to -- so a peer that looped
    // would hold its worker for as long as the loop ran, which for a loop that
    // does not terminate is for ever. Recursion is allowed, because the machine
    // stack bounds it: past the stack floor the body answers `JIT_DEEP` and the
    // interpreter takes the call over. A loop has no such bound, and that is the
    // whole of the difference. A function refused here is still compiled in its
    // own right when something enters it directly; it is only refused as
    // somebody else's callee.
    // Nor a recursion compiled as a loop (`Analysis::accumulate`), for the same
    // reason and one more: it gives a call back to the interpreter by starting
    // it again from its frame, and a peer has none.
    const bool ok = a.compilable && !a.tail_self && !a.accumulate;
    state_[gi] = ok ? State::Yes : State::No;
    if (!ok) return false;

    // Emitted with no float specialization. A peer's signature has to be the
    // same whoever calls it, and a slot carried as a raw double is a decision
    // about one function's own loop -- keeping it would mean compiling the
    // callee once per shape of argument its callers happened to have.
    a.float_slots = 0;
    a.force_first.clear();
    a.peel = false;
    members.emplace(gi, std::move(a));
    return true;
}

bool peer_admits(PeerSet& peers, uint32_t gi, int depth) { return peers.admit(gi, depth); }

const Analysis* peer_member(const PeerSet& peers, uint32_t gi) {
    auto found = peers.members.find(gi);
    return found == peers.members.end() ? nullptr : &found->second;
}

Ty peer_call_ty(const PeerSet& peers, uint32_t gi, SlotSet float_args) {
    auto found = peers.members.find(gi);
    // Still being decided -- a mutual recursion -- so nothing is known yet.
    if (found == peers.members.end()) return Ty::Any;
    const Analysis& a = found->second;
    // Arguments the analysis already knows are floats are floats at run time
    // too, so the call site's check for them passes and the typed variant is
    // the one that runs. That is what licenses answering for it here.
    if (a.typed_slots && (a.typed_slots & ~float_args) == 0) {
        return a.typed_ret_dbl ? Ty::Float : Ty::Any;
    }
    // Otherwise either variant may run, so the answer is a float only when
    // both of them give one.
    const bool both = a.ret_dbl && (!a.typed_slots || a.typed_ret_dbl);
    return both ? Ty::Float : Ty::Any;
}


// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

/// The signature of a body emitted as a function of its arguments:
/// `(proc, depth, frame, stack_floor, a0..an, status)`. A peer's is always all-tagged, which
/// is what lets one declaration serve every caller of it.
llvm::FunctionType* body_signature(llvm::LLVMContext& ctx, uint32_t arity, SlotSet float_slots) {
    auto* i64t = llvm::Type::getInt64Ty(ctx);
    auto* i32t = llvm::Type::getInt32Ty(ctx);
    auto* ptrt = llvm::PointerType::getUnqual(ctx);
    auto* dblt = llvm::Type::getDoubleTy(ctx);
    std::vector<llvm::Type*> params{ptrt, i32t, i64t, i64t};
    for (uint32_t i = 0; i < arity; ++i) {
        params.push_back(float_slots.test(i) ? dblt : i64t);
    }
    params.push_back(ptrt);
    return llvm::FunctionType::get(i64t, params, false);
}

/// The LLVM function emitted for each peer of one compile, by image function
/// index. Every one of them is declared before any body is written, because a
/// call to a peer may be emitted before that peer's own body is -- which is
/// what mutual recursion needs.
struct PeerFn {
    /// Every parameter tagged. What a call gets when it cannot show the typed
    /// variant floats.
    llvm::Function* generic = nullptr;
    /// The declared floats as doubles; null when the signature declares none.
    llvm::Function* typed = nullptr;
    SlotSet typed_slots = 0;
    bool ret_dbl = false;
    bool typed_ret_dbl = false;
    /// The parameters a caller evaluates; every other argument is in a lazy
    /// position. The same set for both variants.
    SlotSet eager = 0;
};
using PeerFns = std::unordered_map<uint32_t, PeerFn>;

class Emitter {
public:
    /// `peers` is the compile's declarations, shared by every emitter in it.
    /// `preset` is non-null when this emitter is writing a peer: the function
    /// already exists, because somebody may already have called it.
    Emitter(llvm::LLVMContext& ctx, llvm::Module& mod, Runtime& rt, const Image& img, uint32_t fi,
            const Analysis& a, const PeerFns* peers = nullptr, llvm::Function* preset = nullptr)
        : ctx_(ctx), mod_(mod), rt_(rt), img_(img), fi_(fi), f_(img.func(fi)),
          recurses_(a.recurses), float_slots_(a.float_slots), carried_(a.carried),
          force_first_(a.force_first), binds_(f_.slots, NO_NODE), bind_stmts_(f_.slots, NO_NODE),
          unique_stmts_(a.unique_stmts), bound_(a.bound), multi_(a.multi),
          peel_first_(a.peel), ret_dbl_(preset && a.ret_dbl), eager_(a.eager), inline_stmts_(a.inline_stmts),
          memo_stmts_(a.memo_stmts), accumulate_(a.accumulate), steps_(a.steps),
          let_steps_(a.let_steps), cons_steps_(a.cons_steps),
          kinds_(a.kinds), interp_(a.interpreted), nested_calls_(a.nested_calls), peers_(peers),
          preset_(preset), b_(ctx) {}

    llvm::Function* emit(const std::string& name);

private:
    /// A value in hand, in one of the two representations of "Representation"
    /// above: a tagged `Value`, or a raw `double` nobody has boxed. A null `v`
    /// means control was transferred and there is no value at all -- what a
    /// tail call leaves behind.
    struct JV {
        llvm::Value* v = nullptr;
        bool dbl = false;
        // A layout hint only: integers can overflow, and gradual callers can
        // pass other types. Every fast path still checks its operands.
        bool integer_hint = false;
    };
    static JV tag(llvm::Value* v, bool integer_hint = false) { return JV{v, false, integer_hint}; }
    static JV flt(llvm::Value* v) { return JV{v, true}; }
    static JV none() { return JV{nullptr, false}; }

    /// Does the function carry this slot as a raw double? Decided by the
    /// fixpoint in the analyzer and guarded on entry.
    /// Always false while the first iteration of a peeled loop is being
    /// written: every slot is tagged there.
    bool slot_is_dbl(uint32_t i) const { return !peeling_ && float_slots_.test(i); }
    /// The same question about the loop proper, whichever is being written.
    bool loop_slot_is_dbl(uint32_t i) const { return float_slots_.test(i); }

    void declare_helpers();
    /// The body, as a function of `(proc, depth, frame, stack_floor, a0..an, status)`. Only
    /// emitted for a body that calls itself outside tail position; the frame
    /// comes along for the yield at depth zero and is untouched otherwise.
    llvm::Function* emit_body(const std::string& name);
    /// The `CompiledFn` the interpreter enters: unpack the frame once, and
    /// call `body` with depth zero.
    llvm::Function* emit_entry(const std::string& name, llvm::Function* body);
    /// Today's shape, for a body whose only self calls are tail calls: one
    /// function, slots read out of the frame, a loop and nothing else.
    llvm::Function* emit_loop(const std::string& name);
    /// `tail` is whether the value is the body's answer: the one context in
    /// which a step is a loop. See "Recursion that is a loop".
    JV node(uint32_t idx, bool tail = false);
    JV binary(const Node& n);
    JV logic(const Node& n);
    JV unary(const Node& n);
    JV type_test(const Node& n);
    JV conditional(const Node& n, bool tail = false);
    JV switch_node(const Node& n, bool tail = false);
    JV block(const Node& n, bool tail = false);
    // --- recursion that is a loop ------------------------------------------
    JV add_step(const Node& n);
    JV let_step(const Node& n);
    /// One more level the interpreter would have had pending. Past the depth
    /// it allows, the call is handed back (`JIT_DEEP`) to overflow as it would.
    void bump_pending();
    /// Hand the answer back, finishing the accumulation when there is one.
    void emit_return(JV r);
    // --- building ahead ------------------------------------------------------
    /// `cons h (f ..)` as the answer: a cell, and the call's iteration next.
    JV cons_step(uint32_t idx);
    /// Leave for `ahead_abort_` when a run of cells is being built. Written
    /// before anything an iteration ahead may not do.
    void ahead_guard();
    /// What `ahead_guard` leaves for: the rest of the list, suspended.
    void emit_ahead_abort();
    /// A helper call that may run code, raise or block, behind `ahead_guard`.
    llvm::CallInst* guarded(llvm::FunctionCallee callee, llvm::ArrayRef<llvm::Value*> args) {
        ahead_guard();
        return b_.CreateCall(callee, args);
    }
    /// The arguments of a self call, evaluated or lazy as the analysis said.
    bool self_args(const Node& n, std::vector<JV>* args);
    JV apply(uint32_t idx, const Node& n);
    JV native_call(KnownNative which, const Node& n);
    /// A call to a host native, made the way the interpreter makes one.
    JV host_call(const NativeSite& site, const Node& n);
    JV self_call(const Node& n);
    /// Anything else that can be applied, handed to the machine.
    JV closure_call(const Node& n);

    // --- data ---------------------------------------------------------------
    JV list_is_empty(const Node& n);
    JV list_tail(const Node& n);
    JV list_is_cons(const Node& n);
    JV list_head(const Node& n);
    JV get(const Node& n);
    JV set(const Node& n);
    JV set_many(const Node& n);
    /// `n` values in lazy positions, laid out in memory for a helper.
    llvm::Value* lazy_array(uint32_t kids_off, uint32_t n);

    // --- lazy positions -----------------------------------------------------
    /// A node the interpreter evaluates, where it stands. See "What the
    /// interpreter evaluates".
    JV interpreted(uint32_t idx, bool tail);
    /// What a lambda captured, as it stands in its closure: unforced.
    llvm::Value* capture_raw(uint32_t index);

    /// The value for an expression in a lazy position: built in place when
    /// that cannot be observed, suspended otherwise. Always tagged. See "Lazy
    /// positions" at the top of this file.
    JV lazy(uint32_t idx);
    /// `idx` as a thunk against this iteration's frame.
    llvm::Value* suspend(uint32_t idx);
    /// The frame the interpreter would be running this iteration in, made the
    /// first time an iteration needs one.
    llvm::Value* snapshot();
    /// What the slots hold, laid out for the runtime to copy into a frame.
    llvm::Value* snapshot_vals();
    /// A remembered `let`'s value, computed at most once an iteration.
    JV bound_value(uint32_t slot);
    bool is_memo(uint32_t slot) const {
        return slot < memo_.size() && memo_[slot] && bind_stmts_[slot] != NO_NODE &&
               memo_stmts_.count(bind_stmts_[slot]);
    }
    /// The binding a snapshot made here puts in `slot`: the one in scope, or
    /// the one `let` that binds the slot anywhere -- a thunk nobody before
    /// its statement can read, which is why it may be made early -- or none,
    /// for a slot bound in several places and in none of them here.
    uint32_t table_stmt(uint32_t slot) const {
        if (bind_stmts_[slot] != NO_NODE) return bind_stmts_[slot];
        return multi_[slot] ? NO_NODE : unique_stmts_[slot];
    }
    /// A `let` of a slot bound in more than one place, met: what the slot
    /// meant before is forgotten, here and in the snapshot.
    void rebind(uint32_t stmt);
    /// The slots of strict bindings whose statement has been written in the
    /// block being written: a read of one after it is a load of its memo,
    /// since the statement ran first on every path that reaches the read.
    /// Cleared when the block ends, as a sibling scope may reuse the slot.
    std::vector<uint8_t> settled_;
    /// Make the per-iteration state -- the snapshot and the remembered
    /// bindings -- and clear it, at the top of the body.
    void make_iteration_state();
    void reset_iteration_state();
    /// Note the slots `idx` may read when the interpreter evaluates it.
    void mark_refs(uint32_t idx, int depth);
    void mark_slot(uint32_t slot);
    /// Give `bind_table_` its contents, now that every suspension is known.
    void finish_bind_table();
    /// An alloca in the entry block. One anywhere else is a *dynamic* alloca,
    /// which grows the machine stack every time it is reached -- in a loop, by
    /// one slot an iteration, for as long as the loop runs.
    llvm::Value* entry_alloca(llvm::Type* t, const char* name = "", uint32_t n = 1);
    /// The kind the analysis recorded for the call at `idx`.
    CallKind kind_at(uint32_t idx) const {
        auto found = kinds_.find(idx);
        return found == kinds_.end() ? CallKind::Refused : found->second;
    }
    /// A saturated call to another global function, emitted beside this one.
    JV peer_call(uint32_t gi, const Node& n);
    /// What a callee is handed as its depth. One more than ours in a shape
    /// that has one, and one in the loop shape, which has no depth of its own
    /// because it makes no machine call.
    llvm::Value* callee_depth() { return depth_ ? b_.CreateAdd(depth_, i32c(1)) : i32c(1); }
    /// Take the callee's status: anything but `JIT_OK` is this invocation's
    /// answer too, and the value in hand is already the right one to return.
    JV take_call_status(llvm::Value* r, const char* what, bool dbl = false);
    /// A call of one variant of a peer, arguments already in hand.
    JV call_peer_variant(llvm::Function* fn, SlotSet dbl_slots, bool ret_dbl,
                         const std::vector<JV>& args);
    /// What a body hands back through the return register.
    llvm::Value* returned(JV v) {
        if (!ret_dbl_) return box(v);
        return b_.CreateBitCast(as_double(v, "a float function answered something else"), i64_);
    }
    JV tail_call(const Node& n, const std::vector<JV>& args);
    JV enter_loop(const Node& n, const std::vector<JV>& args);
    JV recursive_call(const std::vector<JV>& args);
    /// Spend one reduction on a call. Answers what is left, so the tail path
    /// can test it; the recursive path ignores it.
    llvm::Value* spend_reduction();
    /// Write the loop-carried parameters back to the heap frame and return
    /// `JIT_YIELD`, so the interpreter resumes the body from the top.
    void emit_yield();
    JV load_slot(uint32_t slot);
    /// A slot as it stands, unforced. What an argument the callee may never
    /// look at is made of, and the one read in this file that is not a demand.
    JV load_slot_raw(uint32_t slot);
    /// Is this parameter one every self call moves rather than evaluates?
    bool slot_is_carried(uint32_t i) const { return carried_.test(i); }
    llvm::Value* force(llvm::Value* v);

    // --- the two representations, and the crossings between them -----------

    /// `v` as a tagged `Value`, allocating a float box if it is not one yet.
    /// This is the only thing in the file that can allocate, and the four
    /// places it is reached from are listed under "Representation".
    llvm::Value* box(JV v);
    /// Arithmetic and ordering whose answer is a float, in doubles.
    JV float_arith(Op op, JV a, JV b);
    JV float_order(Op op, JV a, JV b);
    llvm::Value* fop(Op op, llvm::Value* x, llvm::Value* y);
    llvm::Value* order_bool(Op op, llvm::Value* x, llvm::Value* y);
    /// The double `v` stands for, raising `message` if it is not a number.
    /// Emits nothing when `v` is already a double, which is the case that pays.
    llvm::Value* as_double(JV v, const char* message);
    /// The double `v` stands for, branching to `slow` if it is not a number in
    /// hand. Leaves the insert point on the path where it is.
    llvm::Value* try_double(llvm::Value* v, llvm::BasicBlock* slow);
    JV int_of_double(llvm::Value* d);
    /// Read a frame slot that the loop carries as a double, or leave for
    /// `bail`. The check is deliberately not a *force*: forcing here would move
    /// where a raising argument raises, and an argument that arrives suspended
    /// is rare enough to be worth handing back rather than reordering for.
    llvm::Value* guard_float(llvm::Value* v, llvm::BasicBlock* bail);
    /// Store `JIT_BAIL` and return: this one call is the interpreter's.
    void emit_bail();
    llvm::Value* unary_intrinsic(llvm::Intrinsic::ID id, llvm::Value* x);

    /// Emit the raise path: store the error and return with JIT_RAISED.
    void emit_raise(llvm::Value* error);
    void emit_type_error(const char* message);
    /// A type error naming what was found, in the interpreter's words.
    void emit_type_error(const char* message, llvm::Value* got);

    llvm::Value* i64(uint64_t v) { return llvm::ConstantInt::get(i64_, v); }
    llvm::Value* i32c(int v) { return llvm::ConstantInt::get(i32_, v); }
    llvm::Value* is_fixnum(llvm::Value* v) {
        return b_.CreateICmpNE(b_.CreateAnd(v, i64(1)), i64(0));
    }
    /// Eight-byte aligned and not the "no value" sentinel: a heap object.
    llvm::Value* is_heap_ptr(llvm::Value* v) {
        return b_.CreateAnd(b_.CreateICmpEQ(b_.CreateAnd(v, i64(TAG_MASK)), i64(0)),
                            b_.CreateICmpNE(v, i64(0)));
    }
    /// The `double` inside a float box. Its header is one word, so the payload
    /// is at offset eight and is as aligned as the object.
    llvm::Value* load_float(llvm::Value* v) {
        llvm::Value* at = b_.CreateIntToPtr(b_.CreateAdd(v, i64(8)), ptr_);
        auto* load = b_.CreateLoad(dbl_, at);
        load->setAlignment(llvm::Align(8));
        return load;
    }
    llvm::BasicBlock* bb(const char* name) { return llvm::BasicBlock::Create(ctx_, name, fn_); }

    llvm::LLVMContext& ctx_;
    llvm::Module& mod_;
    Runtime& rt_;
    const Image& img_;
    uint32_t fi_;
    const FuncRec& f_;
    /// Emitted as a function of its arguments rather than of a frame.
    const bool recurses_;
    /// One bit per slot carried as a raw double.
    const SlotSet float_slots_;
    /// One bit per parameter every self call moves rather than evaluates.
    const SlotSet carried_;
    /// Slots to force on entry, in the order the body would have forced them.
    const std::vector<uint32_t> force_first_;
    /// Per slot, the value and statement of the `let` in scope where the
    /// emitter stands, or `NO_NODE`; see "Bindings in scope". `scope_sig_`
    /// differs whenever they do.
    std::vector<uint32_t> binds_;
    std::vector<uint32_t> bind_stmts_;
    uint64_t scope_sig_ = 0;
    /// See `Analysis::unique_stmts`.
    const std::vector<uint32_t> unique_stmts_;
    const std::vector<uint8_t> bound_;
    const std::vector<uint8_t> multi_;
    /// The loop's first iteration is written separately, over tagged slots.
    const bool peel_first_;
    /// A peer that answers a raw double, its bits in the return register.
    const bool ret_dbl_;
    /// Parameters whose self-call arguments are evaluated; see `Analysis`.
    const SlotSet eager_;
    const std::unordered_set<uint32_t> inline_stmts_;
    const std::unordered_set<uint32_t> memo_stmts_;
    /// See `Analysis::accumulate`.
    const bool accumulate_;
    const std::unordered_set<uint32_t> steps_;
    const std::unordered_set<uint64_t> let_steps_;
    /// See `Analysis::cons_steps`.
    const std::unordered_set<uint32_t> cons_steps_;
    /// The run being built ahead: its first cell and its last, zero until the
    /// first cell is made, and how many more it may have. Null unless the
    /// body has a `cons_step`. See "Building ahead".
    llvm::Value* ahead_first_ = nullptr;
    llvm::Value* ahead_last_ = nullptr;
    llvm::Value* ahead_left_ = nullptr;
    llvm::BasicBlock* ahead_abort_ = nullptr;
    /// The accumulation: the sum of the left sides so far (`acc_`), the least
    /// and greatest it has been before each addition (`minp_`, `maxp_`), how
    /// many levels the interpreter would have pending (`pend_`), and whether
    /// any of them added anything (`adds_`). Null unless `accumulate_`.
    llvm::Value* acc_ = nullptr;
    llvm::Value* minp_ = nullptr;
    llvm::Value* maxp_ = nullptr;
    llvm::Value* pend_ = nullptr;
    llvm::Value* adds_ = nullptr;
    llvm::Type* i128_ = nullptr;
    /// Per slot, the alloca a remembered `let` keeps its value in once it is
    /// computed this iteration -- zero until then -- or null.
    std::vector<llvm::Value*> memo_;
    /// Snapshot helpers copy their input before returning. All snapshot sites
    /// in one invocation can share this scratch array, even across iterations.
    llvm::Value* snapshot_vals_ = nullptr;
    /// How each call is made, as the analysis decided it.
    const std::unordered_map<uint32_t, CallKind> kinds_;
    /// See "What the interpreter evaluates".
    const std::unordered_set<uint32_t> interp_;
    const bool nested_calls_;
    /// The frame this iteration's suspensions share, or zero until one is
    /// needed. See `snapshot`.
    llvm::Value* snap_ = nullptr;
    /// The bindings a snapshot fills in, as constant arrays the runtime can
    /// read: one for each scope a snapshot is made in, the binding of each
    /// slot as `table_stmt` has it there. Restricted to the bindings a
    /// suspension can reach, which is only known once the whole body is
    /// written, so they are filled in by `finish_bind_table`.
    struct BindTable {
        llvm::GlobalVariable* table;
        std::vector<uint32_t> stmts;
    };
    std::vector<BindTable> bind_tables_;
    llvm::GlobalVariable* bind_table();
    /// Slots a suspended expression may read, found by `mark_refs`; and the
    /// nodes, and bindings, already followed, each in the scope it was
    /// followed in (a node means other bindings in another).
    SlotSet refs_ = 0;
    std::unordered_set<std::pair<uint32_t, uint64_t>, PairHash> ref_seen_;
    std::unordered_set<std::pair<uint32_t, uint64_t>, PairHash> binds_marked_;
    /// True while the slots still hold what the interpreter's own frame for
    /// this call holds -- before the first back-edge, in the invocation the
    /// interpreter entered. That frame can then be the snapshot. See
    /// `snapshot`.
    llvm::Value* fresh_ = nullptr;
    /// That first iteration is what is being written right now.
    bool peeling_ = false;
    /// The loop's own slots, typed as `float_slots_` says. `slots_` is these
    /// except while the first iteration is written, when it is a set of tagged
    /// ones and the back-edge stores into these.
    std::vector<llvm::Value*> loop_slots_;
    /// Where a first iteration goes when the value it hands the loop is not
    /// the float the loop carries.
    llvm::BasicBlock* peel_bail_ = nullptr;
    /// The compile's peer declarations, or null when it has none.
    const PeerFns* peers_ = nullptr;
    /// Non-null when this emitter is writing a peer, into a function that was
    /// declared before any body in this module was written.
    llvm::Function* preset_ = nullptr;
    llvm::IRBuilder<> b_;

    llvm::Type* i64_ = nullptr;
    llvm::Type* i32_ = nullptr;
    llvm::Type* i8_ = nullptr;
    llvm::Type* i1_ = nullptr;
    llvm::Type* dbl_ = nullptr;
    llvm::Type* ptr_ = nullptr;

    llvm::Function* fn_ = nullptr;
    llvm::Argument* proc_ = nullptr;
    llvm::Argument* frame_ = nullptr;
    llvm::Argument* status_ = nullptr;
    /// How many compiled frames of this function are already on the machine
    /// stack. Null in the loop shape, which has none. It is an argument rather
    /// than a counter in the process because an argument needs no restoring on
    /// the way out: a deopt unwinds through every frame at once, and a counter
    /// would have to be put back by each of them.
    llvm::Argument* depth_ = nullptr;
    llvm::Value* stack_floor_ = nullptr;

    llvm::BasicBlock* loop_header_ = nullptr;
    /// Emitting the loop shape, which only a root compile is: its answer goes
    /// straight back to `enter_function` and nothing compiled reads it.
    bool root_loop_ = false;
    std::vector<llvm::Value*> slots_;   // allocas, one per frame slot
    llvm::Value* reduction_slot_ = nullptr;
    bool failed_ = false;

    // Declarations of the runtime helpers.
    llvm::FunctionCallee rt_number_double_;
    llvm::FunctionCallee rt_force_, rt_arith_, rt_compare_, rt_float_, rt_arith_f_, rt_to_int_,
        rt_abs_, rt_floor_, rt_int_of_double_, rt_type_error_, rt_reduction_slot_,
        rt_frame_slots_, rt_native_, rt_builtin_, rt_switch_head_,
        rt_get_, rt_set_, rt_set_many_, rt_cons_, rt_make_list_, rt_make_array_, rt_make_record_, rt_make_map_,
        rt_literal_str_,
        rt_global_, rt_snapshot_, rt_thunk_, rt_apply_, rt_peek_, rt_adopt_, rt_yield_frame_,
        rt_cons_after_, rt_set_tail_, rt_type_error_got_, rt_settle_, rt_rebind_, rt_stack_floor_;
    /// The adopt arm of the `snapshot` being written, joined at its end.
    llvm::BasicBlock* pending_adopt_ = nullptr;
    /// The frame that arm adopted: the entry frame, or its copy on the heap
    /// when the entry frame is on the process's stack (`dream_rt_adopt`).
    llvm::Value* adopted_ = nullptr;
};

llvm::Function* Emitter::emit(const std::string& name) {
    declare_helpers();
    // A peer is always a function of its arguments, recursive or not: it has no
    // frame, so there is nothing for the loop shape to read its slots out of.
    if (preset_) return emit_body(name);
    if (!recurses_) return emit_loop(name);
    llvm::Function* body = emit_body(name + ".rec");
    if (!body) return nullptr;
    llvm::Function* entry = emit_entry(name, body);
    if (!entry) {
        body->eraseFromParent();
        return nullptr;
    }
    return entry;
}

void Emitter::declare_helpers() {
    i64_ = llvm::Type::getInt64Ty(ctx_);
    i32_ = llvm::Type::getInt32Ty(ctx_);
    i8_ = llvm::Type::getInt8Ty(ctx_);
    i1_ = llvm::Type::getInt1Ty(ctx_);
    dbl_ = llvm::Type::getDoubleTy(ctx_);
    ptr_ = llvm::PointerType::getUnqual(ctx_);

    rt_switch_head_ = mod_.getOrInsertFunction(
        "dream_rt_switch_head", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_force_ = mod_.getOrInsertFunction(
        "dream_rt_force", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_stack_floor_ = mod_.getOrInsertFunction(
        "dream_rt_stack_floor", llvm::FunctionType::get(i64_, {ptr_}, false));
    rt_arith_ = mod_.getOrInsertFunction(
        "dream_rt_arith", llvm::FunctionType::get(i32_, {ptr_, i32_, i64_, i64_, ptr_}, false));
    rt_compare_ = mod_.getOrInsertFunction(
        "dream_rt_compare", llvm::FunctionType::get(i32_, {ptr_, i32_, i64_, i64_, ptr_}, false));
    rt_float_ = mod_.getOrInsertFunction(
        "dream_rt_float", llvm::FunctionType::get(i64_, {ptr_, dbl_}, false));
    rt_arith_f_ = mod_.getOrInsertFunction(
        "dream_rt_arith_f",
        llvm::FunctionType::get(i32_, {ptr_, i32_, i64_, i64_, ptr_, ptr_}, false));
    rt_to_int_ = mod_.getOrInsertFunction(
        "dream_rt_to_int", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_abs_ = mod_.getOrInsertFunction(
        "dream_rt_abs", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_floor_ = mod_.getOrInsertFunction(
        "dream_rt_floor", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_int_of_double_ = mod_.getOrInsertFunction(
        "dream_rt_int_of_double", llvm::FunctionType::get(i32_, {ptr_, dbl_, ptr_}, false));
    rt_number_double_ = mod_.getOrInsertFunction(
        "dream_rt_number_double", llvm::FunctionType::get(i32_, {ptr_, i64_, ptr_}, false));
    rt_type_error_ = mod_.getOrInsertFunction(
        "dream_rt_type_error", llvm::FunctionType::get(i64_, {ptr_, ptr_}, false));
    rt_type_error_got_ = mod_.getOrInsertFunction(
        "dream_rt_type_error_got", llvm::FunctionType::get(i64_, {ptr_, ptr_, i64_}, false));
    rt_reduction_slot_ = mod_.getOrInsertFunction(
        "dream_rt_reduction_slot", llvm::FunctionType::get(ptr_, {ptr_}, false));
    rt_frame_slots_ = mod_.getOrInsertFunction(
        "dream_rt_frame_slots", llvm::FunctionType::get(ptr_, {i64_}, false));
    // `(proc, module, member, argc, a0..a3, out)` and the same without the
    // module, which a builtin does not have. Four argument words whatever the
    // native's arity: the unused ones are never read, and a fixed signature
    // keeps the call site free of anything to lay out in memory.
    rt_native_ = mod_.getOrInsertFunction(
        "dream_rt_native",
        llvm::FunctionType::get(
            i32_, {ptr_, i32_, i32_, i32_, i64_, i64_, i64_, i64_, ptr_}, false));
    rt_builtin_ = mod_.getOrInsertFunction(
        "dream_rt_builtin",
        llvm::FunctionType::get(i32_, {ptr_, i32_, i32_, i64_, i64_, i64_, i64_, ptr_}, false));
    rt_get_ = mod_.getOrInsertFunction(
        "dream_rt_get", llvm::FunctionType::get(i32_, {ptr_, i64_, i64_, i32_, ptr_}, false));
    rt_set_ = mod_.getOrInsertFunction(
        "dream_rt_set", llvm::FunctionType::get(i32_, {ptr_, i64_, i64_, i64_, ptr_}, false));
    rt_set_many_ = mod_.getOrInsertFunction(
        "dream_rt_set_many", llvm::FunctionType::get(i32_, {ptr_, i64_, i32_, ptr_, ptr_}, false));
    rt_peek_ = mod_.getOrInsertFunction(
        "dream_rt_peek", llvm::FunctionType::get(i64_, {i64_, i64_}, false));
    rt_cons_ = mod_.getOrInsertFunction(
        "dream_rt_cons", llvm::FunctionType::get(i64_, {ptr_, i64_, i64_}, false));
    rt_cons_after_ = mod_.getOrInsertFunction(
        "dream_rt_cons_after", llvm::FunctionType::get(i64_, {ptr_, i64_, i64_}, false));
    rt_set_tail_ = mod_.getOrInsertFunction(
        "dream_rt_set_tail",
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {ptr_, i64_, i64_}, false));
    rt_make_list_ = mod_.getOrInsertFunction(
        "dream_rt_make_list", llvm::FunctionType::get(i64_, {ptr_, i32_, ptr_}, false));
    rt_make_array_ = mod_.getOrInsertFunction(
        "dream_rt_make_array", llvm::FunctionType::get(i64_, {ptr_, i32_, ptr_}, false));
    rt_make_record_ = mod_.getOrInsertFunction(
        "dream_rt_make_record", llvm::FunctionType::get(i64_, {ptr_, i32_, i32_, ptr_}, false));
    rt_settle_ = mod_.getOrInsertFunction(
        "dream_rt_settle",
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {ptr_, i64_, i32_, i64_}, false));
    rt_rebind_ = mod_.getOrInsertFunction(
        "dream_rt_rebind",
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {ptr_, i64_, i32_, i32_}, false));
    rt_make_map_ = mod_.getOrInsertFunction(
        "dream_rt_make_map", llvm::FunctionType::get(i64_, {ptr_}, false));
    rt_literal_str_ = mod_.getOrInsertFunction(
        "dream_rt_literal_str", llvm::FunctionType::get(i64_, {ptr_, i32_}, false));
    rt_global_ = mod_.getOrInsertFunction(
        "dream_rt_global", llvm::FunctionType::get(i64_, {ptr_, i32_}, false));
    rt_snapshot_ = mod_.getOrInsertFunction(
        "dream_rt_snapshot", llvm::FunctionType::get(i64_, {ptr_, i64_, i32_, ptr_, ptr_}, false));
    rt_thunk_ = mod_.getOrInsertFunction(
        "dream_rt_thunk", llvm::FunctionType::get(i64_, {ptr_, i32_, i64_}, false));
    rt_yield_frame_ = mod_.getOrInsertFunction(
        "dream_rt_yield_frame", llvm::FunctionType::get(i64_, {ptr_, i64_, i32_, ptr_}, false));
    rt_adopt_ = mod_.getOrInsertFunction(
        "dream_rt_adopt",
        llvm::FunctionType::get(i64_, {ptr_, i64_, i32_, ptr_, ptr_}, false));
    rt_apply_ = mod_.getOrInsertFunction(
        "dream_rt_apply", llvm::FunctionType::get(i32_, {ptr_, i64_, i32_, ptr_, ptr_}, false));
}

// The first iteration.
//
// A float slot has to be a double before the loop starts, and the value the
// interpreter left in the frame is usually a suspension, so the entry forces
// the slots in the order the body would -- but only as far as that order is
// the same on every path (`Analyzer::force_order`). Past that prefix, forcing
// at the entry would be forcing out of turn, and a float slot out there used
// to cost the whole specialization.
//
// So the loop's first iteration is emitted a second time, over tagged slots,
// and runs exactly as an unspecialized body would: every slot forced on first
// read, in the body's own order. Only its back-edge (`enter_loop`) crosses
// into doubles, and it does so with values the iteration has already computed
// -- a guard, not a force. From there the loop is the specialized one.
// `integrate x hi dx acc` is the shape that needs it: the condition forces `x`
// and `hi`, and `acc` is forced by the base case on one side and by the self
// call on the other, so no prefix reaches it.
llvm::Function* Emitter::emit_loop(const std::string& name) {
    root_loop_ = true;
    auto* fty = llvm::FunctionType::get(i64_, {ptr_, i64_, ptr_}, false);
    fn_ = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, mod_);
    proc_ = fn_->getArg(0);
    frame_ = fn_->getArg(1);
    status_ = fn_->getArg(2);
    proc_->setName("proc");
    frame_->setName("frame");
    status_->setName("status");

    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn_);
    b_.SetInsertPoint(entry);
    stack_floor_ = b_.CreateCall(rt_stack_floor_, {proc_}, "stack.floor");

    // Frame slots become allocas so the loop can carry them in registers;
    // they are written back only when yielding. Every one of them is created
    // here, in the entry block, which is where LLVM looks for the allocas it
    // can promote -- and before any guard splits the block underneath us.
    slots_.resize(f_.slots);
    for (uint32_t i = 0; i < f_.slots; ++i) {
        slots_[i] = b_.CreateAlloca(slot_is_dbl(i) ? dbl_ : i64_, nullptr,
                                    "slot" + std::to_string(i));
    }
    loop_slots_ = slots_;
    make_iteration_state();
    fresh_ = b_.CreateAlloca(i1_, nullptr, "fresh");
    b_.CreateStore(b_.getTrue(), fresh_);
    if (!cons_steps_.empty() && !peel_first_) {
        ahead_first_ = b_.CreateAlloca(i64_, nullptr, "ahead.first");
        ahead_last_ = b_.CreateAlloca(i64_, nullptr, "ahead.last");
        ahead_left_ = b_.CreateAlloca(i64_, nullptr, "ahead.left");
        b_.CreateStore(i64(0), ahead_first_);
        b_.CreateStore(i64(0), ahead_last_);
        b_.CreateStore(i64(kBuildAhead), ahead_left_);
    }
    if (accumulate_) {
        i128_ = llvm::Type::getInt128Ty(ctx_);
        auto zero = llvm::ConstantInt::get(i128_, 0);
        acc_ = b_.CreateAlloca(i128_, nullptr, "acc");
        minp_ = b_.CreateAlloca(i128_, nullptr, "minp");
        maxp_ = b_.CreateAlloca(i128_, nullptr, "maxp");
        pend_ = b_.CreateAlloca(i64_, nullptr, "pending");
        adds_ = b_.CreateAlloca(i1_, nullptr, "adds");
        b_.CreateStore(zero, acc_);
        b_.CreateStore(zero, minp_);
        b_.CreateStore(zero, maxp_);
        b_.CreateStore(i64(0), pend_);
        b_.CreateStore(b_.getFalse(), adds_);
    }
    std::vector<llvm::Value*> first_slots;
    if (peel_first_) {
        first_slots.resize(f_.slots);
        for (uint32_t i = 0; i < f_.slots; ++i) {
            first_slots[i] = b_.CreateAlloca(i64_, nullptr, "first" + std::to_string(i));
        }
    }
    llvm::BasicBlock* bail = float_slots_ ? llvm::BasicBlock::Create(ctx_, "bail", fn_) : nullptr;
    peel_bail_ = bail;
    llvm::Value* slot_base = b_.CreateCall(rt_frame_slots_, {frame_}, "slots");
    std::vector<llvm::Value*> raw(f_.slots);
    for (uint32_t i = 0; i < f_.slots; ++i) {
        raw[i] = b_.CreateLoad(i64_, b_.CreateGEP(i64_, slot_base, {i64(i)}));
    }
    if (peel_first_) {
        // Handed over exactly as they arrived; the first iteration forces
        // each on first read, as the interpreter would.
        for (uint32_t i = 0; i < f_.slots; ++i) b_.CreateStore(raw[i], first_slots[i]);
    } else {
        // Force what the body would have forced, in the order it would have
        // forced it. Empty unless a slot is carried as a double; see
        // `Analyzer::force_order`.
        for (uint32_t slot : force_first_) raw[slot] = force(raw[slot]);
        for (uint32_t i = 0; i < f_.slots; ++i) {
            b_.CreateStore(slot_is_dbl(i) ? guard_float(raw[i], bail) : raw[i], slots_[i]);
        }
    }
    reduction_slot_ = b_.CreateCall(rt_reduction_slot_, {proc_}, "reductions");

    loop_header_ = llvm::BasicBlock::Create(ctx_, "loop", fn_);
    if (peel_first_) {
        // The first iteration, over tagged slots. Its base case returns like
        // any other; its tail call stores into the loop's own slots and
        // enters the loop (`tail_call`).
        auto* first = llvm::BasicBlock::Create(ctx_, "first", fn_);
        b_.CreateBr(first);
        b_.SetInsertPoint(first);
        slots_ = first_slots;
        peeling_ = true;
        JV r = node(f_.body, true);
        peeling_ = false;
        slots_ = loop_slots_;
        if (failed_) {
            fn_->eraseFromParent();
            return nullptr;
        }
        if (r.v) {
            emit_return(r);
        } else if (!b_.GetInsertBlock()->getTerminator()) {
            b_.CreateUnreachable();
        }
    } else {
        b_.CreateBr(loop_header_);
    }

    if (bail) {
        b_.SetInsertPoint(bail);
        emit_bail();
    }

    b_.SetInsertPoint(loop_header_);
    // Every back-edge lands here with new slots, so a frame made for the last
    // iteration's suspensions is not this one's, nor are its bindings.
    reset_iteration_state();
    JV result = node(f_.body, true);
    if (failed_) {
        fn_->eraseFromParent();
        return nullptr;
    }
    if (result.v) {
        emit_return(result);
    } else if (!b_.GetInsertBlock()->getTerminator()) {
        // Every path ended in a tail call; nothing falls through here.
        b_.CreateUnreachable();
    }
    if (ahead_abort_) emit_ahead_abort();
    if (failed_) {
        fn_->eraseFromParent();
        return nullptr;
    }

    finish_bind_table();
    if (llvm::verifyFunction(*fn_, &llvm::errs())) {
        fn_->eraseFromParent();
        return nullptr;
    }
    return fn_;
}

llvm::Function* Emitter::emit_body(const std::string& name) {
    // `(proc, depth, frame, stack_floor, a0..an, status)`. The parameters arrive as values
    // rather than through the frame, which is the whole point: a self call
    // allocates nothing, and the only frame in the picture is the one the
    // interpreter already made for the outermost call. A parameter the
    // function carries as a float arrives as a `double`; the entry below is
    // where that is checked, once.
    // Internal, so LLVM is free to inline the first level or two of the
    // recursion into itself and to pick its own calling convention for it. A
    // peer's declaration was made before any body in this module was written,
    // so that a call to it could be emitted first.
    fn_ = preset_ ? preset_
                  : llvm::Function::Create(body_signature(ctx_, f_.arity, float_slots_),
                                           llvm::Function::InternalLinkage, name, mod_);
    proc_ = fn_->getArg(0);
    depth_ = fn_->getArg(1);
    frame_ = fn_->getArg(2);
    stack_floor_ = fn_->getArg(3);
    status_ = fn_->getArg(fn_->arg_size() - 1);
    proc_->setName("proc");
    depth_->setName("depth");
    frame_->setName("frame");
    status_->setName("status");

    auto* entry = llvm::BasicBlock::Create(ctx_, "entry", fn_);
    b_.SetInsertPoint(entry);

    // Allocas first and in the entry block, which is where LLVM looks for the
    // ones it can promote to registers.
    slots_.resize(f_.slots);
    for (uint32_t i = 0; i < f_.slots; ++i) {
        slots_[i] = b_.CreateAlloca(slot_is_dbl(i) ? dbl_ : i64_, nullptr,
                                    "slot" + std::to_string(i));
    }
    for (uint32_t i = 0; i < f_.arity; ++i) {
        b_.CreateStore(fn_->getArg(4 + i), slots_[i]);
    }
    // Slots past the parameters belong to `let` bindings, and a bound slot is
    // never read here: its value is written where its name is. So they are
    // allocated to keep the slot indices lining up, filled with something
    // legible, and never touched again. A slot that is neither a parameter nor
    // bound refuses the whole function (`Analyzer::run`), which is what makes
    // that true.
    for (uint32_t i = f_.arity; i < f_.slots; ++i) b_.CreateStore(i64(UNIT), slots_[i]);
    make_iteration_state();
    if (!preset_) {
        // Only the outermost invocation was entered with a frame of its own;
        // every deeper one shares that frame and must make its own.
        fresh_ = b_.CreateAlloca(i1_, nullptr, "fresh");
        b_.CreateStore(b_.CreateICmpEQ(depth_, i32c(0)), fresh_);
    }
    reduction_slot_ = b_.CreateCall(rt_reduction_slot_, {proc_}, "reductions");

    // Out of machine stack: hand the call back. The floor accounts for the
    // worker's guard page and the whole compiled/nested-force chain, rather
    // than guessing a frame size from the number of local slots.
    auto* deep = llvm::BasicBlock::Create(ctx_, "deep", fn_);
    loop_header_ = llvm::BasicBlock::Create(ctx_, "loop", fn_);
    // Read the actual stack pointer inline. The floor travels with compiled
    // calls, so recursive kernels need no runtime call for each stack check.
    std::vector<llvm::Type*> stack_types;
    if (llvm::Intrinsic::isOverloaded(llvm::Intrinsic::stacksave)) stack_types.push_back(ptr_);
#if LLVM_VERSION_MAJOR >= 21
    auto* stack_read = llvm::Intrinsic::getOrInsertDeclaration(&mod_, llvm::Intrinsic::stacksave, stack_types);
#else
    auto* stack_read = llvm::Intrinsic::getDeclaration(&mod_, llvm::Intrinsic::stacksave, stack_types);
#endif
    llvm::Value* stack_at = b_.CreatePtrToInt(b_.CreateCall(stack_read), i64_);
    llvm::Value* stack_ok = b_.CreateICmpUGE(stack_at, stack_floor_);
    b_.CreateCondBr(stack_ok, loop_header_, deep);

    b_.SetInsertPoint(deep);
    b_.CreateStore(i32c(JIT_DEEP), status_);
    b_.CreateRet(i64(UNIT));

    b_.SetInsertPoint(loop_header_);
    reset_iteration_state();
    JV result = node(f_.body);
    // A peer is not erased on failure: another peer may already have emitted a
    // call to it, and a function with uses cannot be removed. Nothing is leaked
    // by leaving it -- every caller of this drops the whole module.
    if (failed_) {
        if (!preset_) fn_->eraseFromParent();
        return nullptr;
    }
    if (result.v) {
        b_.CreateStore(i32c(JIT_OK), status_);
        b_.CreateRet(returned(result));
    } else if (!b_.GetInsertBlock()->getTerminator()) {
        b_.CreateUnreachable();
    }

    finish_bind_table();
    if (llvm::verifyFunction(*fn_, &llvm::errs())) {
        if (!preset_) fn_->eraseFromParent();
        return nullptr;
    }
    return fn_;
}

llvm::Function* Emitter::emit_entry(const std::string& name, llvm::Function* body) {
    auto* fty = llvm::FunctionType::get(i64_, {ptr_, i64_, ptr_}, false);
    auto* fn = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, mod_);
    // The helpers below write into whatever `fn_` names, so the entry takes
    // the emitter over for as long as it is being written. The body is already
    // finished and nothing reaches back into it.
    fn_ = fn;
    depth_ = nullptr;
    slots_.clear();
    proc_ = fn->getArg(0);
    frame_ = fn->getArg(1);
    status_ = fn->getArg(2);
    proc_->setName("proc");
    frame_->setName("frame");
    status_->setName("status");

    b_.SetInsertPoint(llvm::BasicBlock::Create(ctx_, "entry", fn));
    stack_floor_ = b_.CreateCall(rt_stack_floor_, {proc_}, "stack.floor");
    llvm::BasicBlock* bail = float_slots_ ? llvm::BasicBlock::Create(ctx_, "bail", fn) : nullptr;
    llvm::Value* slot_base = b_.CreateCall(rt_frame_slots_, {frame_}, "slots");
    std::vector<llvm::Value*> raw(f_.arity);
    for (uint32_t i = 0; i < f_.arity; ++i) {
        raw[i] = b_.CreateLoad(i64_, b_.CreateGEP(i64_, slot_base, {i64(i)}));
    }
    // The arguments are handed over unforced, and the body forces each slot on
    // first read exactly as the loop shape does -- so an argument the body
    // never looks at is never touched here either. The exception is the prefix
    // a double-carrying slot obliges: it has to be a double before the body
    // starts, so the body's own force order is replayed up to it and no
    // further.
    for (uint32_t slot : force_first_) {
        if (slot < f_.arity) raw[slot] = force(raw[slot]);
    }
    std::vector<llvm::Value*> args{proc_, i32c(0), frame_, stack_floor_};
    for (uint32_t i = 0; i < f_.arity; ++i) {
        args.push_back(slot_is_dbl(i) ? guard_float(raw[i], bail) : raw[i]);
    }
    args.push_back(status_);
    b_.CreateRet(b_.CreateCall(body, args));

    if (bail) {
        b_.SetInsertPoint(bail);
        emit_bail();
    }

    if (llvm::verifyFunction(*fn, &llvm::errs())) {
        fn->eraseFromParent();
        return nullptr;
    }
    return fn;
}

void Emitter::emit_raise(llvm::Value* error) {
    ahead_guard();
    b_.CreateStore(i32c(JIT_RAISED), status_);
    b_.CreateRet(error);
}

void Emitter::emit_type_error(const char* message) {
    llvm::Value* msg = b_.CreateGlobalString(message);
    emit_raise(b_.CreateCall(rt_type_error_, {proc_, msg}));
}

void Emitter::emit_type_error(const char* message, llvm::Value* got) {
    llvm::Value* msg = b_.CreateGlobalString(message);
    emit_raise(b_.CreateCall(rt_type_error_got_, {proc_, msg, got}));
}

void Emitter::emit_bail() {
    ahead_guard();
    // Nothing has happened yet: no reduction spent, no slot written, no effect.
    // So the interpreter can simply run the call, which is what it would have
    // done had this function never been compiled.
    b_.CreateStore(i32c(JIT_BAIL), status_);
    b_.CreateRet(i64(UNIT));
}

llvm::Value* Emitter::force(llvm::Value* v) {
    // Anything that is not a heap pointer is already in normal form, and in a
    // numeric function that is the overwhelmingly common case.
    auto* need = bb("force.slow");
    auto* done = bb("force.done");
    auto* raise = bb("force.raise");

    llvm::Value* tagged = b_.CreateICmpNE(b_.CreateAnd(v, i64(7)), i64(0));
    llvm::Value* null = b_.CreateICmpEQ(v, i64(0));
    llvm::Value* whnf = b_.CreateOr(tagged, null);
    auto* fast = b_.GetInsertBlock();
    auto* object = bb("force.object");
    b_.CreateCondBr(whnf, done, object);

    // A heap object is in normal form unless it is one of the three kinds a
    // suspension goes through -- and in a list function almost every object
    // read is a cell, which used to cost a call to be told so.
    b_.SetInsertPoint(object);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
    llvm::Value* rel = b_.CreateSub(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Thunk)));
    static_assert(uint8_t(ObjType::Blackhole) == uint8_t(ObjType::Thunk) + 1 &&
                      uint8_t(ObjType::Indirect) == uint8_t(ObjType::Thunk) + 2,
                  "the suspension kinds are one range");
    llvm::Value* suspended = b_.CreateICmpULE(rel, llvm::ConstantInt::get(i8_, 2));
    auto* object_end = b_.GetInsertBlock();
    b_.CreateCondBr(suspended, need, done);

    b_.SetInsertPoint(need);
    ahead_guard();
    llvm::Value* out = entry_alloca(i64_, "forced");
    llvm::Value* ok = b_.CreateCall(rt_force_, {proc_, v, out});
    llvm::Value* forced = b_.CreateLoad(i64_, out);
    auto* slow = b_.GetInsertBlock();
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), done, raise);

    b_.SetInsertPoint(raise);
    emit_raise(forced);

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i64_, 3);
    phi->addIncoming(v, fast);
    phi->addIncoming(v, object_end);
    phi->addIncoming(forced, slow);
    return phi;
}

Emitter::JV Emitter::load_slot(uint32_t slot) {
    if (slot_is_dbl(slot)) {
        // Already a double: the entry guard saw a float box and nothing since
        // has written anything else here.
        return flt(b_.CreateLoad(dbl_, slots_[slot]));
    }
    llvm::Value* v = b_.CreateLoad(i64_, slots_[slot]);
    llvm::Value* forced = force(v);
    // Write the forced value back so a second read in the same iteration is
    // free; the thunk itself was already updated by the runtime.
    b_.CreateStore(forced, slots_[slot]);
    return tag(forced, (img_.integer_params(fi_) >> slot) & 1);
}

Emitter::JV Emitter::load_slot_raw(uint32_t slot) {
    // No force, and nothing written back. A slot the loop carries as a double
    // is already a number and there is nothing to force about it; every other
    // slot is handed on exactly as it arrived, which may be a suspension the
    // program never demands.
    if (slot_is_dbl(slot)) return flt(b_.CreateLoad(dbl_, slots_[slot]));
    return tag(b_.CreateLoad(i64_, slots_[slot]));
}

llvm::Value* Emitter::entry_alloca(llvm::Type* t, const char* name, uint32_t n) {
    llvm::BasicBlock& entry = fn_->getEntryBlock();
    llvm::IRBuilder<> at(&entry, entry.getFirstInsertionPt());
    return at.CreateAlloca(t, n == 1 ? nullptr : llvm::ConstantInt::get(i32_, n), name);
}

// ---------------------------------------------------------------------------
// Lazy positions
// ---------------------------------------------------------------------------

llvm::Value* Emitter::snapshot_vals() {
    if (!snapshot_vals_) snapshot_vals_ = entry_alloca(i64_, "snap.vals", f_.slots);
    llvm::Value* vals = snapshot_vals_;
    for (uint32_t i = 0; i < f_.slots; ++i) {
        llvm::Value* v = nullptr;
        if (i < f_.arity) {
            v = slot_is_dbl(i) ? box(flt(b_.CreateLoad(dbl_, slots_[i])))
                               : b_.CreateLoad(i64_, slots_[i]);
        } else if (i < memo_.size() && memo_[i] && table_stmt(i) != NO_NODE && memo_stmts_.count(table_stmt(i))) {
            // A binding this iteration has computed hands over its value; one
            // it has not, zero, and the runtime makes the thunk. The memo is
            // the slot's, and a slot bound in several places is in the table
            // only where its binding is the one in scope, which is the one the
            // memo holds.
            v = b_.CreateLoad(i64_, memo_[i]);
        } else {
            // A bound slot is filled by the runtime, with a thunk of its
            // binding; any other is one nothing here has written.
            v = i64(NIL_SLOT);
        }
        b_.CreateStore(v, b_.CreateGEP(i64_, vals, {i64(i)}));
    }
    return vals;
}

llvm::Value* Emitter::snapshot() {
    // Made at most once an iteration: `snap_` is cleared at the top of the
    // body, which is where every back-edge lands, and the slots do not change
    // between there and the next back-edge -- `load_slot` writes back what it
    // forced, which is the same value.
    auto* have = bb("snap.have");
    auto* make = bb("snap.make");
    auto* done = bb("snap.done");
    llvm::Value* cur = b_.CreateLoad(i64_, snap_);
    auto* from = b_.GetInsertBlock();
    b_.CreateCondBr(b_.CreateICmpNE(cur, i64(0)), have, make);

    b_.SetInsertPoint(have);
    b_.CreateBr(done);

    b_.SetInsertPoint(make);
    llvm::GlobalVariable* table = bind_table();
    // Before the first back-edge of the invocation the interpreter entered,
    // the slots hold exactly what that invocation's heap frame holds -- the
    // arguments as they arrived, some since forced, which is the same values.
    // So that frame is this iteration's frame, and the only thing it lacks is
    // what the interpreter would have written into it by now: the bindings.
    // Adopting it is what makes a function that does not loop cost no frame
    // beyond the one its call already made.
    llvm::Value* vals = snapshot_vals();
    if (fresh_) {
        auto* adopt = bb("snap.adopt");
        auto* copy = bb("snap.copy");
        b_.CreateCondBr(b_.CreateLoad(i1_, fresh_), adopt, copy);
        b_.SetInsertPoint(adopt);
        llvm::Value* adopted =
            b_.CreateCall(rt_adopt_, {proc_, frame_, i32c(int(f_.slots)), vals, table});
        b_.CreateStore(adopted, snap_);
        adopted_ = adopted;
        auto* adopt_end = b_.GetInsertBlock();
        b_.CreateBr(done);
        b_.SetInsertPoint(copy);
        make = b_.GetInsertBlock();
        pending_adopt_ = adopt_end;
    }
    // The closure is what a thunk against the frame reads a capture out of,
    // and what the profiler names the frame by. The root's is in the frame the
    // interpreter entered it with; a peer has no frame of its own, and a global
    // function has no captures, so it goes without.
    llvm::Value* closure = i64(UNIT);
    if (!preset_) {
        llvm::Value* at = b_.CreateIntToPtr(b_.CreateAdd(frame_, i64(8)), ptr_);
        closure = b_.CreateLoad(i64_, at);
    }
    llvm::Value* made = b_.CreateCall(
        rt_snapshot_, {proc_, closure, i32c(int(f_.slots)), vals, table});
    b_.CreateStore(made, snap_);
    auto* make_end = b_.GetInsertBlock();
    b_.CreateBr(done);

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i64_, 3);
    phi->addIncoming(cur, have);
    phi->addIncoming(made, make_end);
    if (pending_adopt_) phi->addIncoming(adopted_, pending_adopt_);
    pending_adopt_ = nullptr;
    (void)from;
    return phi;
}

// A binding computed once.
//
// A `let` has no slot here: its value is written where its name is read. For a
// small one that calls nothing that is free even when it happens twice. For one
// that calls something it is not, and the case that decides it is a binding
// read in *both* kinds of position -- evaluated at one read, and handed on
// unevaluated at another. The interpreter has one thunk in one slot and
// evaluates it once; written out naively, the evaluated read computed it and
// the lazy one suspended it again, and whoever forced the suspension computed
// it a second time. `dreams` lowering a function body twice over is what that
// looked like, and it was 14% of a self-compile.
//
// So such a binding is remembered: an alloca per binding per iteration, zero
// until the first evaluated read fills it. A lazy read takes the value when it
// is there and the snapshot's thunk when it is not, and the snapshot takes the
// value too. A snapshot made *before* the first evaluated read already holds
// the binding's thunk, and something may hold that thunk, so the evaluated read
// forces it rather than computing a second copy.
//
// And a snapshot made *during* it -- by a lazy position inside the binding's
// own expression, which is where `let run = match .. { spec => rebuild_run ..
// }` makes one -- holds a thunk of the binding too, made before there was a
// value to put there. Whatever reads that slot afterwards forces the thunk and
// runs the binding a second time. Pure, so no answer changes; but in a function
// that recursion re-enters through that binding, the work doubles at every
// level. dreams/src/opt.dr's `rebuild_node` is that function, and compiled, a
// four-second compile of a hello-world ran out of memory instead. So the value,
// once computed, settles that thunk in place (`dream_rt_settle`), as forcing it
// would have.

void Emitter::make_iteration_state() {
    snap_ = b_.CreateAlloca(i64_, nullptr, "snap");
    b_.CreateStore(i64(0), snap_);
    // One a slot, for whichever of its bindings is in scope: on one path a
    // slot is bound once, and where an arm's guard fails and the next arm
    // binds it again, `rebind` clears it first.
    memo_.assign(f_.slots, nullptr);
    for (uint32_t stmt : memo_stmts_) {
        const uint32_t i = img_.node(stmt).a;
        if (i >= f_.slots || memo_[i]) continue;
        memo_[i] = b_.CreateAlloca(i64_, nullptr, "memo" + std::to_string(i));
        b_.CreateStore(i64(0), memo_[i]);
    }
}

void Emitter::reset_iteration_state() {
    b_.CreateStore(i64(0), snap_);
    for (llvm::Value* m : memo_) {
        if (m) b_.CreateStore(i64(0), m);
    }
}

Emitter::JV Emitter::bound_value(uint32_t slot) {
    // The snapshot's slot must hold the binding for the path below that forces
    // it; marking it is what puts it in the table every snapshot is made from.
    mark_slot(slot);
    auto* have = bb("bind.have");
    auto* miss = bb("bind.miss");
    auto* from_snap = bb("bind.snap");
    auto* compute = bb("bind.compute");
    auto* done = bb("bind.done");
    llvm::Value* known = b_.CreateLoad(i64_, memo_[slot]);
    b_.CreateCondBr(b_.CreateICmpNE(known, i64(0)), have, miss);
    b_.SetInsertPoint(have);
    b_.CreateBr(done);

    b_.SetInsertPoint(miss);
    llvm::Value* snap = b_.CreateLoad(i64_, snap_);
    b_.CreateCondBr(b_.CreateICmpNE(snap, i64(0)), from_snap, compute);

    b_.SetInsertPoint(from_snap);
    llvm::Value* at = b_.CreateIntToPtr(
        b_.CreateAdd(snap, i64(sizeof(FrameObj) + 8 * uint64_t(slot))), ptr_);
    llvm::Value* forced = force(b_.CreateLoad(i64_, at));
    auto* snap_end = b_.GetInsertBlock();
    b_.CreateBr(done);

    b_.SetInsertPoint(compute);
    JV v = node(binds_[slot]);
    if (failed_ || !v.v) {
        // Control left the binding -- nothing reaches the join from here.
        if (!failed_ && !b_.GetInsertBlock()->getTerminator()) b_.CreateUnreachable();
        b_.SetInsertPoint(done);
        auto* phi = b_.CreatePHI(i64_, 2);
        phi->addIncoming(known, have);
        phi->addIncoming(forced, snap_end);
        b_.CreateStore(phi, memo_[slot]);
        return failed_ ? none() : tag(phi);
    }
    llvm::Value* computed = box(v);
    // A snapshot made while the value was being computed suspended the
    // binding before it had one; settle that suspension now.
    auto* settle = bb("bind.settle");
    auto* settled = bb("bind.settled");
    llvm::Value* late = b_.CreateLoad(i64_, snap_);
    b_.CreateCondBr(b_.CreateICmpNE(late, i64(0)), settle, settled);
    b_.SetInsertPoint(settle);
    b_.CreateCall(rt_settle_, {proc_, late, i32c(int(slot)), computed});
    b_.CreateBr(settled);
    b_.SetInsertPoint(settled);
    auto* compute_end = b_.GetInsertBlock();
    b_.CreateBr(done);

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i64_, 3);
    phi->addIncoming(known, have);
    phi->addIncoming(forced, snap_end);
    phi->addIncoming(computed, compute_end);
    b_.CreateStore(phi, memo_[slot]);
    return tag(phi);
}

void Emitter::mark_slot(uint32_t slot) {
    if (slot >= f_.slots) return;
    refs_.set(slot);
    // A binding's thunk reads whatever its own expression reads -- in the
    // scope it is read in, which for a binding of a shared slot is not
    // always the scope it was last followed in.
    const uint32_t stmt = table_stmt(slot);
    if (stmt == NO_NODE || !binds_marked_.insert({stmt, scope_sig_}).second) return;
    mark_refs(img_.node(stmt).b, 0);
}

void Emitter::mark_refs(uint32_t idx, int depth) {
    if (idx == NO_NODE || !ref_seen_.insert({idx, scope_sig_}).second) return;
    // Too deep to follow is too deep to be sure of: every binding, then.
    if (depth > 512) {
        refs_ = ~SlotSet();
        return;
    }
    const Node& n = img_.node(idx);
    auto run = [&](uint32_t off, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) mark_refs(img_.kid(off + i), depth + 1);
    };
    if (is_primitive(Op(n.op))) {
        run(n.b, n.c);
        return;
    }
    // The operand layouts are `opt.dr`'s `node_edges` and `node_runs`, and
    // any opcode not written out here is answered with every binding -- the
    // one mistake this must not make is to leave out a slot a thunk will read.
    switch (Op(n.op)) {
        case Op::ConstInt: case Op::ConstFloat: case Op::ConstStr: case Op::ConstChar:
        case Op::ConstBool: case Op::ConstAtom: case Op::Unit: case Op::Capture:
        case Op::Global: case Op::Builtin: case Op::Nop:
            return;
        case Op::Local:
            mark_slot(n.a);
            return;
        case Op::MakeClosure: case Op::MakeThunk: {
            const FuncRec& g = img_.func(n.a);
            for (uint32_t i = 0; i < g.n_captures; ++i) {
                const uint32_t desc = img_.kid(g.captures_off + i);
                if (!(desc & CAP_FROM_CAPTURE)) mark_slot(desc);
            }
            return;
        }
        case Op::Field: case Op::Force: case Op::TypeIs: case Op::ListTail:
        case Op::ListIsEmpty: case Op::Neg: case Op::Not: case Op::BitNot: case Op::ListIsCons: case Op::ListHead:
            mark_refs(n.a, depth + 1);
            return;
        case Op::Bind:
            mark_refs(n.b, depth + 1);
            return;
        case Op::Try: case Op::Ensure: case Op::Cons:
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or:
            mark_refs(n.a, depth + 1);
            mark_refs(n.b, depth + 1);
            return;
        case Op::If: case Op::Get: case Op::Set:
            mark_refs(n.a, depth + 1);
            mark_refs(n.b, depth + 1);
            mark_refs(n.c, depth + 1);
            return;
        case Op::SetMany:
            mark_refs(n.a, depth + 1);
            run(n.b, 2 * n.c);
            return;
        case Op::Apply: case Op::SwitchHead: case Op::SwitchAtom:
            mark_refs(n.a, depth + 1);
            run(n.b, n.c);
            return;
        case Op::Block: {
            // A name the block binds means its own binding inside it.
            BindScope scope(binds_, bind_stmts_, &scope_sig_);
            scope.enter_block(img_, n);
            run(n.a, n.b);
            return;
        }
        case Op::MakeList: case Op::MakeArray: case Op::RecordMake:
            run(n.a, n.b);
            return;
        case Op::MakeMap:
            run(n.a, 2 * n.b);
            return;
        default:
            refs_ = ~SlotSet();
            return;
    }
}

llvm::GlobalVariable* Emitter::bind_table() {
    std::vector<uint32_t> stmts(f_.slots, NO_NODE);
    for (uint32_t i = f_.arity; i < f_.slots; ++i) stmts[i] = table_stmt(i);
    for (const BindTable& t : bind_tables_) {
        if (t.stmts == stmts) return t.table;
    }
    auto* ty = llvm::ArrayType::get(i32_, f_.slots);
    auto* table = new llvm::GlobalVariable(mod_, ty, true, llvm::GlobalValue::PrivateLinkage, nullptr,
                                           fn_->getName() + ".binds" + std::to_string(bind_tables_.size()));
    bind_tables_.push_back({table, std::move(stmts)});
    return table;
}

void Emitter::finish_bind_table() {
    auto* ty = llvm::ArrayType::get(i32_, f_.slots);
    for (const BindTable& t : bind_tables_) {
        std::vector<llvm::Constant*> entries;
        for (uint32_t i = 0; i < f_.slots; ++i) {
            const uint32_t stmt = refs_.test(i) ? t.stmts[i] : NO_NODE;
            entries.push_back(llvm::ConstantInt::get(i32_, stmt == NO_NODE ? NO_NODE : img_.node(stmt).b));
        }
        t.table->setInitializer(llvm::ConstantArray::get(ty, entries));
    }
}

void Emitter::rebind(uint32_t stmt) {
    const uint32_t slot = img_.node(stmt).a;
    if (slot < memo_.size() && memo_[slot]) b_.CreateStore(i64(0), memo_[slot]);
    if (slot < settled_.size()) settled_[slot] = 0;
    // A snapshot already made holds the slot's other binding, or none: this
    // one goes in, for the suspensions made from here on to read.
    mark_slot(slot);
    auto* put = bb("rebind.put");
    auto* done = bb("rebind.done");
    llvm::Value* snap = b_.CreateLoad(i64_, snap_);
    b_.CreateCondBr(b_.CreateICmpNE(snap, i64(0)), put, done);
    b_.SetInsertPoint(put);
    b_.CreateCall(rt_rebind_, {proc_, snap, i32c(int(slot)), i32c(int(img_.node(stmt).b))});
    b_.CreateBr(done);
    b_.SetInsertPoint(done);
}

llvm::Value* Emitter::suspend(uint32_t idx) {
    mark_refs(idx, 0);
    return b_.CreateCall(rt_thunk_, {proc_, i32c(int(idx)), snapshot()});
}

Emitter::JV Emitter::lazy(uint32_t idx) {
    const Node& n = img_.node(idx);
    switch (Op(n.op)) {
        case Op::ConstInt:
            if (fixnum_fits(img_.integer(n.a))) return node(idx);
            return tag(suspend(idx));
        case Op::ConstBool: case Op::ConstChar: case Op::Unit: case Op::ConstAtom:
        case Op::ConstStr: case Op::Builtin:
            return node(idx);
        case Op::ConstFloat:
            return tag(box(node(idx)));
        case Op::Capture:
            // Handed on as it stands, as `thunk_for` hands on a capture.
            if (preset_) return tag(suspend(idx));
            return tag(capture_raw(n.a));
        case Op::MakeMap:
            // Building an empty map cannot be observed; one with entries
            // forces its keys, and is suspended.
            if (n.b == 0) return tag(b_.CreateCall(rt_make_map_, {proc_}));
            return tag(suspend(idx));
        case Op::Local:
            if (n.a < binds_.size() && binds_[n.a] != NO_NODE) {
                // A `let`'s name. Its value again when that is cheap, and
                // otherwise what the interpreter's slot would have held: the
                // binding, suspended.
                if (inline_stmts_.count(bind_stmts_[n.a])) return lazy(binds_[n.a]);
                // Its value, when this iteration has already computed it; and
                // otherwise the frame's own thunk for it, so that every lazy
                // read in an iteration shares one suspension -- and one
                // evaluation -- as they share the interpreter's slot.
                mark_slot(n.a);
                llvm::BasicBlock* known_end = nullptr;
                llvm::Value* known = nullptr;
                auto* done = bb("lazybind.done");
                if (is_memo(n.a)) {
                    known = b_.CreateLoad(i64_, memo_[n.a]);
                    auto* unknown = bb("lazybind.unknown");
                    known_end = b_.GetInsertBlock();
                    b_.CreateCondBr(b_.CreateICmpNE(known, i64(0)), done, unknown);
                    b_.SetInsertPoint(unknown);
                }
                llvm::Value* fr = snapshot();
                llvm::Value* at = b_.CreateIntToPtr(
                    b_.CreateAdd(fr, i64(sizeof(FrameObj) + 8 * uint64_t(n.a))), ptr_);
                llvm::Value* th = b_.CreateLoad(i64_, at);
                auto* th_end = b_.GetInsertBlock();
                b_.CreateBr(done);
                b_.SetInsertPoint(done);
                auto* phi = b_.CreatePHI(i64_, 2);
                phi->addIncoming(th, th_end);
                if (known_end) phi->addIncoming(known, known_end);
                return tag(phi);
            }
            if (n.a >= f_.arity) return tag(suspend(idx));
            return tag(box(load_slot_raw(n.a)));
        case Op::Force:
            // Asked for where it stands, and handed over unevaluated instead,
            // which is what the interpreter's `thunk_for` does with it too.
            return lazy(n.a);
        case Op::Cons: {
            // A cell is already a value, and building one cannot raise, so it
            // is built rather than suspended -- which is the whole saving on a
            // builder: a cell an iteration, and no thunk around it.
            llvm::Value* h = lazy(n.a).v;
            llvm::Value* t = lazy(n.b).v;
            if (failed_) return none();
            return tag(b_.CreateCall(rt_cons_, {proc_, h, t}));
        }
        case Op::MakeList: case Op::MakeArray: {
            if (Op(n.op) == Op::MakeList && n.b == 0) return tag(i64(NIL));
            llvm::Value* items = lazy_array(n.a, n.b);
            if (failed_) return none();
            return tag(b_.CreateCall(Op(n.op) == Op::MakeList ? rt_make_list_ : rt_make_array_,
                                     {proc_, i32c(int(n.b)), items}));
        }
        case Op::RecordMake: {
            llvm::Value* items = lazy_array(n.a, n.b);
            if (failed_) return none();
            return tag(b_.CreateCall(rt_make_record_,
                                     {proc_, i32c(int(n.c)), i32c(int(n.b)), items}));
        }
        case Op::Add: case Op::Sub: case Op::Mul: {
            // `thunk_for`'s arithmetic: two fixnums already in hand, and an
            // answer that is still one. Anything else -- an operand that is a
            // suspension, overflow, a float -- is the suspension it would have
            // been. Nothing is forced either way.
            if (!cheap_lazy(img_, binds_, f_.arity, idx, 0)) return tag(suspend(idx));
            llvm::Value* x = lazy(n.a).v;
            llvm::Value* y = lazy(n.b).v;
            if (failed_) return none();
            auto* fast = bb("lazy.arith");
            auto* slow = bb("lazy.suspend");
            auto* done = bb("lazy.done");
            b_.CreateCondBr(b_.CreateAnd(is_fixnum(x), is_fixnum(y)), fast, slow);
            b_.SetInsertPoint(fast);
            llvm::Value* xa = b_.CreateAShr(x, 1);
            llvm::Value* ya = b_.CreateAShr(y, 1);
            const llvm::Intrinsic::ID iid = Op(n.op) == Op::Add ? llvm::Intrinsic::sadd_with_overflow
                                            : Op(n.op) == Op::Sub ? llvm::Intrinsic::ssub_with_overflow
                                                                  : llvm::Intrinsic::smul_with_overflow;
            auto* fn = llvm::Intrinsic::getOrInsertDeclaration(&mod_, iid, {i64_});
            llvm::Value* pair = b_.CreateCall(fn, {xa, ya});
            llvm::Value* raw = b_.CreateExtractValue(pair, 0);
            llvm::Value* ovf = b_.CreateExtractValue(pair, 1);
            // A fixnum is a 63-bit integer: `raw * 2` must not overflow either.
            auto* dbl = llvm::Intrinsic::getOrInsertDeclaration(
                &mod_, llvm::Intrinsic::smul_with_overflow, {i64_});
            llvm::Value* p2 = b_.CreateCall(dbl, {raw, i64(2)});
            llvm::Value* ovf2 = b_.CreateExtractValue(p2, 1);
            llvm::Value* tagged = b_.CreateOr(b_.CreateExtractValue(p2, 0), i64(1));
            auto* fast_ok = bb("lazy.arith.ok");
            b_.CreateCondBr(b_.CreateOr(ovf, ovf2), slow, fast_ok);
            b_.SetInsertPoint(fast_ok);
            b_.CreateBr(done);
            b_.SetInsertPoint(slow);
            llvm::Value* th = suspend(idx);
            auto* slow_end = b_.GetInsertBlock();
            b_.CreateBr(done);
            b_.SetInsertPoint(done);
            auto* phi = b_.CreatePHI(i64_, 2);
            phi->addIncoming(tagged, fast_ok);
            phi->addIncoming(th, slow_end);
            return tag(phi);
        }
        case Op::ListTail: {
            // The tail of a cell already in hand is a load, and cannot raise;
            // it is handed over as it stands, suspension or not. Anything else
            // -- an operand still suspended, the empty list -- is the
            // suspension it would have been. This is what lets `map f (tail
            // xs)` build ahead over a list that is already built.
            if (!cheap_lazy(img_, binds_, f_.arity, n.a, 0)) return tag(suspend(idx));
            llvm::Value* x = lazy(n.a).v;
            if (failed_) return none();
            auto* obj = bb("lazy.tail.obj");
            auto* cell = bb("lazy.tail.cell");
            auto* miss = bb("lazy.tail.miss");
            auto* done = bb("lazy.tail.done");
            b_.CreateCondBr(is_heap_ptr(x), obj, miss);
            b_.SetInsertPoint(obj);
            llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(x, ptr_));
            b_.CreateCondBr(
                b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons))), cell,
                miss);
            b_.SetInsertPoint(cell);
            llvm::Value* tail =
                b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(x, i64(16)), ptr_));
            b_.CreateBr(done);
            b_.SetInsertPoint(miss);
            llvm::Value* th = suspend(idx);
            auto* miss_end = b_.GetInsertBlock();
            b_.CreateBr(done);
            b_.SetInsertPoint(done);
            auto* phi = b_.CreatePHI(i64_, 2);
            phi->addIncoming(tail, cell);
            phi->addIncoming(th, miss_end);
            return tag(phi);
        }
        case Op::Get: {
            // `thunk_for`'s read: an element of an array or of the part of a
            // list already built, found without forcing anything. `dream_rt_peek`
            // is that walk; it answers zero where `thunk_for` would suspend.
            if (n.c != NO_NODE || !cheap_lazy(img_, binds_, f_.arity, idx, 0)) {
                return tag(suspend(idx));
            }
            llvm::Value* c = lazy(n.a).v;
            llvm::Value* k = lazy(n.b).v;
            if (failed_) return none();
            llvm::Value* found = b_.CreateCall(rt_peek_, {c, k});
            auto* hit = bb("lazy.get.hit");
            auto* miss = bb("lazy.get.miss");
            auto* done = bb("lazy.get.done");
            b_.CreateCondBr(b_.CreateICmpNE(found, i64(0)), hit, miss);
            b_.SetInsertPoint(hit);
            b_.CreateBr(done);
            b_.SetInsertPoint(miss);
            llvm::Value* th = suspend(idx);
            auto* miss_end = b_.GetInsertBlock();
            b_.CreateBr(done);
            b_.SetInsertPoint(done);
            auto* phi = b_.CreatePHI(i64_, 2);
            phi->addIncoming(found, hit);
            phi->addIncoming(th, miss_end);
            return tag(phi);
        }
        default:
            return tag(suspend(idx));
    }
}

Emitter::JV Emitter::interpreted(uint32_t idx, bool tail) {
    llvm::Value* th = suspend(idx);
    // The answer of a root loop is forced by `enter_function` once the
    // compiled call returns, as a parameter that is the answer is; there is
    // no reason to run the machine underneath this call to do it first.
    if (tail && root_loop_ && !accumulate_ && !ahead_first_) return tag(th);
    return tag(force(th));
}

llvm::Value* Emitter::capture_raw(uint32_t index) {
    // `frame->closure`, then `closure->caps()[index]`: a frame's closure is
    // its first word after the header, and a closure's captures follow its
    // sixteen bytes of header, function and count.
    llvm::Value* closure = b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(frame_, i64(8)), ptr_));
    return b_.CreateLoad(
        i64_, b_.CreateIntToPtr(b_.CreateAdd(closure, i64(sizeof(ClosureObj) + 8 * uint64_t(index))), ptr_));
}

llvm::Value* Emitter::lazy_array(uint32_t kids_off, uint32_t n) {
    llvm::Value* items = entry_alloca(i64_, "items", std::max(1u, n));
    for (uint32_t i = 0; i < n; ++i) {
        JV v = lazy(img_.kid(kids_off + i));
        if (failed_) return nullptr;
        b_.CreateStore(v.v, b_.CreateGEP(i64_, items, {i64(i)}));
    }
    return items;
}

// ---------------------------------------------------------------------------
// Lists, arrays and maps
//
// The reads a list or map function makes on every iteration are written out:
// the empty test, the tail of a cell, the head of one and an in-range array
// element. Everything past those -- a map lookup, a walk down a list, every
// error -- is `dream_rt_get`, which is the interpreter's `container_get`.
// ---------------------------------------------------------------------------

Emitter::JV Emitter::list_is_empty(const Node& n) {
    JV v = node(n.a);
    if (failed_ || !v.v) return none();
    llvm::Value* x = box(v);
    return tag(b_.CreateSelect(b_.CreateICmpEQ(x, i64(NIL)), i64(TRUE_V), i64(FALSE_V)));
}

Emitter::JV Emitter::list_is_cons(const Node& n) {
    JV v = node(n.a);
    if (failed_ || !v.v) return none();
    llvm::Value* x = box(v);
    auto* obj = bb("cons.obj");
    auto* done = bb("cons.done");
    auto* from = b_.GetInsertBlock();
    b_.CreateCondBr(is_heap_ptr(x), obj, done);
    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(x, ptr_));
    llvm::Value* cell = b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons)));
    auto* obj_end = b_.GetInsertBlock();
    b_.CreateBr(done);
    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i1_, 2);
    phi->addIncoming(b_.getFalse(), from);
    phi->addIncoming(cell, obj_end);
    return tag(b_.CreateSelect(phi, i64(TRUE_V), i64(FALSE_V)));
}

/// `ListTail`'s twin: the head of a cell, forced.
Emitter::JV Emitter::list_head(const Node& n) {
    JV v = node(n.a);
    if (failed_ || !v.v) return none();
    llvm::Value* x = box(v);
    auto* obj = bb("head.obj");
    auto* cell = bb("head.cell");
    auto* bad = bb("head.bad");
    b_.CreateCondBr(is_heap_ptr(x), obj, bad);
    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(x, ptr_));
    b_.CreateCondBr(b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons))),
                    cell, bad);
    b_.SetInsertPoint(bad);
    emit_type_error("head needs a non-empty list");
    b_.SetInsertPoint(cell);
    llvm::Value* head = b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(x, i64(8)), ptr_));
    return tag(force(head));
}

Emitter::JV Emitter::list_tail(const Node& n) {
    JV v = node(n.a);
    if (failed_ || !v.v) return none();
    llvm::Value* x = box(v);
    auto* obj = bb("tail.obj");
    auto* cell = bb("tail.cell");
    auto* bad = bb("tail.bad");
    b_.CreateCondBr(is_heap_ptr(x), obj, bad);
    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(x, ptr_));
    b_.CreateCondBr(b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons))),
                    cell, bad);
    b_.SetInsertPoint(bad);
    emit_type_error("tail needs a non-empty list");
    b_.SetInsertPoint(cell);
    llvm::Value* tail = b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(x, i64(16)), ptr_));
    return tag(force(tail));
}

Emitter::JV Emitter::get(const Node& n) {
    JV cv = node(n.a);
    if (failed_ || !cv.v) return none();
    JV kv = node(n.b);
    if (failed_ || !kv.v) return none();
    llvm::Value* c = box(cv);
    llvm::Value* k = box(kv);

    auto* obj = bb("get.obj");
    auto* is_cell = bb("get.cell");
    auto* not_cell = bb("get.notcell");
    auto* is_arr = bb("get.array");
    auto* in_arr = bb("get.inarray");
    auto* slow = bb("get.slow");
    auto* found = bb("get.found");
    auto* done = bb("get.done");

    b_.CreateCondBr(is_heap_ptr(c), obj, slow);
    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(c, ptr_));
    b_.CreateCondBr(b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons))),
                    is_cell, not_cell);

    // `xs.[0]`: the head of a cell, which is how a list is read.
    b_.SetInsertPoint(is_cell);
    llvm::Value* head = b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(c, i64(8)), ptr_));
    auto* head_bb = b_.GetInsertBlock();
    b_.CreateCondBr(b_.CreateICmpEQ(k, i64(make_fixnum(0))), found, slow);

    // An array at a fixnum inside it. The index is signed, so one unsigned
    // compare against the length rules out both ends. A record is read the
    // same way: its header is an array's, a length at 8 and the fields from 16
    // (`RecordObj`), so a record's accessor is the same inline load a
    // `struct`'s was when it was an array.
    b_.SetInsertPoint(not_cell);
    llvm::Value* indexed = b_.CreateOr(
        b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Array))),
        b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Record))));
    b_.CreateCondBr(b_.CreateAnd(indexed, is_fixnum(k)), is_arr, slow);
    b_.SetInsertPoint(is_arr);
    llvm::Value* len = b_.CreateZExt(
        b_.CreateLoad(i32_, b_.CreateIntToPtr(b_.CreateAdd(c, i64(8)), ptr_)), i64_);
    llvm::Value* at = b_.CreateAShr(k, 1);
    b_.CreateCondBr(b_.CreateICmpULT(at, len), in_arr, slow);
    b_.SetInsertPoint(in_arr);
    llvm::Value* item = b_.CreateLoad(
        i64_, b_.CreateIntToPtr(b_.CreateAdd(b_.CreateAdd(c, i64(16)), b_.CreateShl(at, 3)), ptr_));
    auto* item_bb = b_.GetInsertBlock();
    b_.CreateBr(found);

    // Everything else: a map, a walk, a default, an error.
    b_.SetInsertPoint(slow);
    llvm::Value* out = entry_alloca(i64_, "got");
    llvm::Value* st =
        guarded(rt_get_, {proc_, c, k, i32c(n.c != NO_NODE ? 1 : 0), out});
    llvm::Value* slowv = b_.CreateLoad(i64_, out);
    auto* slow_end = b_.GetInsertBlock();
    auto* raise_bb = bb("get.raise");
    auto* dflt_bb = bb("get.default");
    auto* sw = b_.CreateSwitch(st, raise_bb, 2);
    sw->addCase(llvm::cast<llvm::ConstantInt>(i32c(1)), found);
    sw->addCase(llvm::cast<llvm::ConstantInt>(i32c(2)), dflt_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(slowv);

    // The element is forced where the machine's `enter` would force it.
    b_.SetInsertPoint(found);
    auto* elem = b_.CreatePHI(i64_, 3);
    elem->addIncoming(head, head_bb);
    elem->addIncoming(item, item_bb);
    elem->addIncoming(slowv, slow_end);
    llvm::Value* forced = force(elem);
    auto* found_end = b_.GetInsertBlock();
    b_.CreateBr(done);

    b_.SetInsertPoint(dflt_bb);
    llvm::Value* dv = nullptr;
    if (n.c != NO_NODE) {
        JV d = node(n.c);
        if (failed_) return none();
        if (d.v) dv = box(d);
    }
    auto* dflt_end = b_.GetInsertBlock();
    if (dv) {
        b_.CreateBr(done);
    } else if (!dflt_end->getTerminator()) {
        // No default (the status cannot be 2), or one that transferred control.
        b_.CreateUnreachable();
    }

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(forced, found_end);
    if (dv) phi->addIncoming(dv, dflt_end);
    return tag(phi);
}

Emitter::JV Emitter::set(const Node& n) {
    JV cv = node(n.a);
    if (failed_ || !cv.v) return none();
    JV kv = node(n.b);
    if (failed_ || !kv.v) return none();
    llvm::Value* c = box(cv);
    llvm::Value* k = box(kv);
    // The value is stored unforced, as the interpreter stores it.
    JV vv = lazy(n.c);
    if (failed_) return none();
    llvm::Value* out = entry_alloca(i64_, "set");
    llvm::Value* ok = guarded(rt_set_, {proc_, c, k, vv.v, out});
    llvm::Value* r = b_.CreateLoad(i64_, out);
    auto* cont = bb("set.ok");
    auto* raise_bb = bb("set.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), cont, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(r);
    b_.SetInsertPoint(cont);
    return tag(r);
}

/// The pairs are one array, keys and values interleaved as the node's kids
/// are: a key is a constant, which `lazy` answers as itself, and a value is in
/// a lazy position, as a `Set`'s is. One call then makes the one copy.
Emitter::JV Emitter::set_many(const Node& n) {
    JV cv = node(n.a);
    if (failed_ || !cv.v) return none();
    llvm::Value* c = box(cv);
    llvm::Value* pairs = lazy_array(n.b, 2 * n.c);
    if (failed_) return none();
    llvm::Value* out = entry_alloca(i64_, "set_many");
    llvm::Value* ok = guarded(rt_set_many_, {proc_, c, i32c(int(n.c)), pairs, out});
    llvm::Value* r = b_.CreateLoad(i64_, out);
    auto* cont = bb("set_many.ok");
    auto* raise_bb = bb("set_many.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), cont, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(r);
    b_.SetInsertPoint(cont);
    return tag(r);
}

// ---------------------------------------------------------------------------
// Crossing between the two representations
// ---------------------------------------------------------------------------

llvm::Value* Emitter::box(JV v) {
    if (!v.dbl) return v.v;
    return b_.CreateCall(rt_float_, {proc_, v.v});
}

llvm::Value* Emitter::try_double(llvm::Value* v, llvm::BasicBlock* slow) {
    auto* fix = bb("num.fix");
    auto* notfix = bb("num.notfix");
    auto* obj = bb("num.obj");
    auto* load = bb("num.load");
    auto* done = bb("num.done");

    b_.CreateCondBr(is_fixnum(v), fix, notfix);

    b_.SetInsertPoint(fix);
    llvm::Value* from_int = b_.CreateSIToFP(b_.CreateAShr(v, 1), dbl_);
    b_.CreateBr(done);

    // Reading the type byte is only legal once the value is known to be a
    // pointer, which is why this is two tests and not one.
    b_.SetInsertPoint(notfix);
    b_.CreateCondBr(is_heap_ptr(v), obj, slow);

    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
    b_.CreateCondBr(
        b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Float))), load, slow);

    b_.SetInsertPoint(load);
    llvm::Value* from_box = load_float(v);
    b_.CreateBr(done);

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(dbl_, 2);
    phi->addIncoming(from_int, fix);
    phi->addIncoming(from_box, load);
    return phi;
}

llvm::Value* Emitter::as_double(JV v, const char* message) {
    if (v.dbl) return v.v;
    // A fixnum or a float box is read inline. Anything else is asked of the
    // runtime, which answers a bignum's double or says it is not a number --
    // a call only a bignum's first conversion pays, never a fixnum's.
    auto* other = bb("num.other");
    llvm::Value* d = try_double(v.v, other);
    auto* fast = b_.GetInsertBlock();
    auto* join = bb("num.join");
    b_.CreateBr(join);

    b_.SetInsertPoint(other);
    llvm::Value* slot = entry_alloca(dbl_);
    llvm::Value* ok = guarded(rt_number_double_, {proc_, v.v, slot});
    auto* big = bb("num.big");
    auto* bad = bb("num.bad");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), big, bad);
    b_.SetInsertPoint(bad);
    emit_type_error(message);
    b_.SetInsertPoint(big);
    llvm::Value* from_big = b_.CreateLoad(dbl_, slot);
    b_.CreateBr(join);

    b_.SetInsertPoint(join);
    auto* phi = b_.CreatePHI(dbl_, 2);
    phi->addIncoming(d, fast);
    phi->addIncoming(from_big, big);
    return phi;
}

/// An integer from a double in a register: `to_int` and `math.floor` when the
/// number is unboxed. The runtime builds it -- a bignum past 2^62 -- and
/// refuses an infinity or a NaN, which has no integer to be.
Emitter::JV Emitter::int_of_double(llvm::Value* d) {
    llvm::Value* out = entry_alloca(i64_);
    llvm::Value* ok = guarded(rt_int_of_double_, {proc_, d, out});
    llvm::Value* v = b_.CreateLoad(i64_, out);
    auto* cont = bb("int.ok");
    auto* raise_bb = bb("int.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), cont, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(v);
    b_.SetInsertPoint(cont);
    return tag(v);
}

llvm::Value* Emitter::guard_float(llvm::Value* v, llvm::BasicBlock* bail) {
    auto* obj = bb("guard.obj");
    auto* ok = bb("guard.ok");
    b_.CreateCondBr(is_heap_ptr(v), obj, bail);

    b_.SetInsertPoint(obj);
    llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
    b_.CreateCondBr(
        b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Float))), ok, bail);

    b_.SetInsertPoint(ok);
    return load_float(v);
}

llvm::Value* Emitter::unary_intrinsic(llvm::Intrinsic::ID id, llvm::Value* x) {
    auto* fn = llvm::Intrinsic::getOrInsertDeclaration(&mod_, id, {dbl_});
    return b_.CreateCall(fn, {x});
}

llvm::Value* Emitter::fop(Op op, llvm::Value* x, llvm::Value* y) {
    switch (op) {
        case Op::Add: return b_.CreateFAdd(x, y);
        case Op::Sub: return b_.CreateFSub(x, y);
        case Op::Mul: return b_.CreateFMul(x, y);
        case Op::Div: return b_.CreateFDiv(x, y);
        default: return b_.CreateFRem(x, y);   // Mod, which `arith` does with fmod
    }
}

llvm::Value* Emitter::order_bool(Op op, llvm::Value* x, llvm::Value* y) {
    // `compare` turns two numbers into -1, 0 or 1 using `x < y` and `x > y` and
    // nothing else, so a NaN on either side falls to 0 -- which makes
    // `nan <= nan` *true* in this language. An ordered `fcmp ole` would say
    // false. The two predicates are written here exactly as `compare` writes
    // them, so that the tiers cannot disagree about it.
    llvm::Value* lt = b_.CreateFCmpOLT(x, y);
    llvm::Value* gt = b_.CreateFCmpOGT(x, y);
    llvm::Value* c = nullptr;
    switch (op) {
        case Op::Lt: c = lt; break;
        case Op::Gt: c = gt; break;
        case Op::Le: c = b_.CreateNot(gt); break;
        default: c = b_.CreateNot(lt); break;   // Ge
    }
    return b_.CreateSelect(c, i64(TRUE_V), i64(FALSE_V));
}

Emitter::JV Emitter::float_arith(Op op, JV a, JV bv) {
    // The case the whole thing is for: both sides in registers, and the answer
    // is one instruction with nothing around it.
    if (a.dbl && bv.dbl) return flt(fop(op, a.v, bv.v));

    auto* slow_bb = bb("fbin.slow");
    auto* join_bb = bb("fbin.end");
    llvm::Value* da = a.dbl ? a.v : try_double(a.v, slow_bb);
    llvm::Value* db = bv.dbl ? bv.v : try_double(bv.v, slow_bb);
    llvm::Value* fast = fop(op, da, db);
    auto* fast_end = b_.GetInsertBlock();
    b_.CreateBr(join_bb);

    // Not a number in hand. `arith` decides, and either raises -- which is
    // what all but always happens, with the error the interpreter would have
    // given -- or answers a number, because an operation with a float on one
    // side of it has no third case among numbers. So the answer comes back as
    // a double and the caller never learns there were two paths.
    //
    // The one value that is not a number and still combines with a float is
    // a tensor: `2.0 * t` is a tensor, and there is no register here for it.
    // The helper says so with a 2 and the call goes back to the interpreter
    // (`JIT_BAIL`), which is sound for the reason every bail is -- a compiled
    // body is pure and has written nothing but registers. `Ty::Float` stays
    // what it was: a claim about every value this can produce *here*.
    b_.SetInsertPoint(slow_bb);
    llvm::Value* out = entry_alloca(dbl_);
    llvm::Value* err = entry_alloca(i64_);
    llvm::Value* ok =
        guarded(rt_arith_f_, {proc_, i32c(int(op)), box(a), box(bv), out, err});
    llvm::Value* slowv = b_.CreateLoad(dbl_, out);
    auto* slow_end = b_.GetInsertBlock();
    auto* raise_bb = bb("fbin.raise");
    auto* other_bb = bb("fbin.other");
    auto* bail_bb = bb("fbin.bail");
    b_.CreateCondBr(b_.CreateICmpEQ(ok, i32c(1)), join_bb, other_bb);

    b_.SetInsertPoint(other_bb);
    b_.CreateCondBr(b_.CreateICmpEQ(ok, i32c(2)), bail_bb, raise_bb);

    b_.SetInsertPoint(bail_bb);
    emit_bail();

    b_.SetInsertPoint(raise_bb);
    emit_raise(b_.CreateLoad(i64_, err));

    b_.SetInsertPoint(join_bb);
    auto* phi = b_.CreatePHI(dbl_, 2);
    phi->addIncoming(fast, fast_end);
    phi->addIncoming(slowv, slow_end);
    return flt(phi);
}

Emitter::JV Emitter::float_order(Op op, JV a, JV bv) {
    if (a.dbl && bv.dbl) return tag(order_bool(op, a.v, bv.v));

    auto* slow_bb = bb("fcmp.slow");
    auto* join_bb = bb("fcmp.end");
    llvm::Value* da = a.dbl ? a.v : try_double(a.v, slow_bb);
    llvm::Value* db = bv.dbl ? bv.v : try_double(bv.v, slow_bb);
    llvm::Value* fast = order_bool(op, da, db);
    auto* fast_end = b_.GetInsertBlock();
    b_.CreateBr(join_bb);

    b_.SetInsertPoint(slow_bb);
    llvm::Value* out = entry_alloca(i64_);
    llvm::Value* ok = guarded(rt_compare_, {proc_, i32c(int(op)), box(a), box(bv), out});
    llvm::Value* slowv = b_.CreateLoad(i64_, out);
    auto* slow_end = b_.GetInsertBlock();
    auto* raise_bb = bb("fcmp.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), join_bb, raise_bb);

    b_.SetInsertPoint(raise_bb);
    emit_raise(slowv);

    b_.SetInsertPoint(join_bb);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(fast, fast_end);
    phi->addIncoming(slowv, slow_end);
    return tag(phi);
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

Emitter::JV Emitter::node(uint32_t idx, bool tail) {
    if (failed_) return none();
    if (interp_.count(idx)) return interpreted(idx, tail);
    if (tail && ahead_first_ && cons_steps_.count(idx)) return cons_step(idx);
    const Node& n = img_.node(idx);
    switch (Op(n.op)) {
        case Op::ConstInt: {
            int64_t v = img_.integer(n.a);
            if (!fixnum_fits(v)) {
                failed_ = true;
                return none();
            }
            return tag(i64(make_fixnum(v)), true);
        }
        case Op::ConstBool: return tag(i64(n.a ? TRUE_V : FALSE_V));
        case Op::ConstChar: return tag(i64(make_char(n.a)));
        case Op::Unit: return tag(i64(UNIT));
        case Op::ConstFloat:
            // The constant *is* the double. Boxing one was the largest single
            // source of garbage in a float loop: `pi` allocated a float box per
            // literal per iteration, four million of them, to hold two numbers
            // that never left the loop.
            return flt(llvm::ConstantFP::get(dbl_, img_.real(n.a)));
        case Op::ConstAtom:
            return tag(i64(make_atom(rt_.image_atom(n.a))));
        case Op::SwitchAtom: case Op::SwitchHead: return switch_node(n, tail);
        case Op::Local:
            // A `let` has no thunk and no slot here: its value is written where
            // its name is read, which is here. See `Analyzer::check_bind`.
            if (tail && accumulate_ && n.a < f_.slots && bind_stmts_[n.a] != NO_NODE &&
                let_steps_.count(let_step_key(idx, bind_stmts_[n.a]))) {
                return let_step(n);
            }
            if (n.a < binds_.size() && binds_[n.a] != NO_NODE) {
                if (is_memo(n.a) && n.a < settled_.size() && settled_[n.a]) {
                    // The snapshot still needs the slot, for whatever is
                    // suspended against it.
                    mark_slot(n.a);
                    return tag(b_.CreateLoad(i64_, memo_[n.a]));
                }
                return is_memo(n.a) ? bound_value(n.a) : node(binds_[n.a]);
            }
            // A parameter that *is* the answer is handed back as it stands,
            // and the interpreter forces it once the compiled call has
            // returned (`enter_function`). Forcing it here was not only
            // unnecessary but unsound: an accumulator is usually a chain of
            // suspensions this very loop built, and if forcing it reaches a
            // blocking native -- a `join!` in a fold's function -- the park
            // retries the whole call from its frame, which builds that chain
            // afresh and performs every effect in it a second time. A lazy
            // fold whose function spawned and joined printed each step twice.
            if (tail && root_loop_ && !accumulate_) return load_slot_raw(n.a);
            return load_slot(n.a);
        case Op::Capture:
            // The analysis admits one only in a root, which has a closure.
            if (preset_) {
                failed_ = true;
                return none();
            }
            return tag(force(capture_raw(n.a)));
        case Op::Builtin: return tag(i64(make_builtin(n.a)));
        case Op::MakeMap: return lazy(idx);
        case Op::Force: return node(n.a);
        case Op::TypeIs: return type_test(n);
        case Op::If: return conditional(n, tail);
        case Op::Block: return block(n, tail);
        case Op::And: case Op::Or: return logic(n);
        case Op::Neg: case Op::Not: case Op::BitNot: return unary(n);
        DREAM_PRIMITIVE_CASES
        case Op::Apply:
            // A pure higher-order callee may reach an effectful callback.
            // Return a tail call to the machine instead of forcing it under
            // this frame: a park would retry it and repeat earlier effects.
            if (tail && root_loop_ && !accumulate_ && !ahead_first_ &&
                kind_at(idx) == CallKind::Closure) return lazy(idx);
            return apply(idx, n);
        case Op::ConstStr: return tag(b_.CreateCall(rt_literal_str_, {proc_, i32c(int(n.a))}));
        case Op::Global: return tag(force(b_.CreateCall(rt_global_, {proc_, i32c(int(n.a))})));
        case Op::ListIsEmpty: return list_is_empty(n);
        case Op::ListTail: return list_tail(n);
        case Op::ListIsCons: return list_is_cons(n);
        case Op::ListHead: return list_head(n);
        case Op::Get: return get(n);
        case Op::Set: return set(n);
        case Op::SetMany: return set_many(n);
        case Op::Cons: case Op::MakeList: case Op::MakeArray: case Op::RecordMake:
            // Every part of these is in a lazy position, and so is the whole:
            // building one in place is the one way it is ever written.
            return lazy(idx);
        case Op::Add:
            if (tail && accumulate_ && steps_.count(idx)) return add_step(n);
            return binary(n);
        case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            return binary(n);
        default:
            // Nothing the analysis admitted reaches here. Refusing rather than
            // guessing is what keeps an op missing from one list from being
            // compiled as though it were in another.
            failed_ = true;
            return none();
    }
}

// A type test consumes its subject even when its representation is already
// known. In particular, do not erase an arithmetic error when folding a test
// of an unboxed double. The object header is read only after a pointer guard.
Emitter::JV Emitter::type_test(const Node& n) {
    JV subject = node(n.a);
    if (failed_ || !subject.v) return none();
    llvm::Value* test = nullptr;
    if (subject.dbl) {
        test = llvm::ConstantInt::getBool(ctx_, n.b == DREAM_TYPE_FLOAT);
    } else if (n.b == DREAM_TYPE_INTEGER) {
        // A fixnum, or a bignum -- whose header is read only behind the
        // pointer guard, as every other object test here reads its own.
        auto* object = bb("int.object");
        auto* done = bb("int.done");
        llvm::Value* fix = is_fixnum(subject.v);
        auto* here = b_.GetInsertBlock();
        b_.CreateCondBr(is_heap_ptr(subject.v), object, done);
        b_.SetInsertPoint(object);
        llvm::Value* header = b_.CreateLoad(i8_, b_.CreateIntToPtr(subject.v, ptr_));
        llvm::Value* big = b_.CreateICmpEQ(header, llvm::ConstantInt::get(i8_, uint8_t(ObjType::BigInt)));
        b_.CreateBr(done);
        b_.SetInsertPoint(done);
        auto* phi = b_.CreatePHI(llvm::Type::getInt1Ty(ctx_), 2, "int.matches");
        phi->addIncoming(fix, here);
        phi->addIncoming(big, object);
        test = phi;
    } else if (n.b == DREAM_TYPE_CHAR || n.b == DREAM_TYPE_BOOL ||
               n.b == DREAM_TYPE_UNIT || n.b == DREAM_TYPE_ATOM) {
        ImmKind kind = n.b == DREAM_TYPE_CHAR ? IMM_CHAR :
                       n.b == DREAM_TYPE_BOOL ? IMM_BOOL :
                       n.b == DREAM_TYPE_UNIT ? IMM_UNIT : IMM_ATOM;
        test = b_.CreateICmpEQ(b_.CreateAnd(subject.v, i64(255)), i64(make_imm(kind, 0)));
    } else {
        auto* object = bb("type.object");
        auto* done = bb("type.done");
        auto* immediate = b_.GetInsertBlock();
        llvm::Value* immediate_test = n.b == DREAM_TYPE_LIST
            ? b_.CreateICmpEQ(b_.CreateAnd(subject.v, i64(255)), i64(NIL))
            : llvm::ConstantInt::getFalse(ctx_);
        b_.CreateCondBr(is_heap_ptr(subject.v), object, done);
        b_.SetInsertPoint(object);
        llvm::Value* header = b_.CreateLoad(i8_, b_.CreateIntToPtr(subject.v, ptr_));
        ObjType kind = n.b == DREAM_TYPE_FLOAT ? ObjType::Float :
                       n.b == DREAM_TYPE_STRING ? ObjType::Str :
                       n.b == DREAM_TYPE_LIST ? ObjType::Cons :
                       n.b == DREAM_TYPE_ARRAY ? ObjType::Array :
                       n.b == DREAM_TYPE_TENSOR ? ObjType::Tensor :
                       n.b == DREAM_TYPE_RECORD ? ObjType::Record :
                       n.b == DREAM_TYPE_TRUNK ? ObjType::Trunk : ObjType::Map;
        llvm::Value* matches = b_.CreateICmpEQ(header, llvm::ConstantInt::get(i8_, uint8_t(kind)));
        if (n.b == DREAM_TYPE_MAP)
            matches = b_.CreateOr(matches, b_.CreateICmpEQ(
                header, llvm::ConstantInt::get(i8_, uint8_t(ObjType::MapLeaf))));
        b_.CreateBr(done);
        b_.SetInsertPoint(done);
        auto* phi = b_.CreatePHI(llvm::Type::getInt1Ty(ctx_), 2, "type.matches");
        phi->addIncoming(immediate_test, immediate);
        phi->addIncoming(matches, object);
        test = phi;
    }
    if (n.c) test = b_.CreateNot(test);
    return tag(b_.CreateSelect(test, i64(TRUE_V), i64(FALSE_V)));
}

Emitter::JV Emitter::block(const Node& n, bool tail) {
    JV last = tag(i64(UNIT));
    if (settled_.size() < f_.slots) settled_.resize(f_.slots, 0);
    // The strict bindings this block settles, unsettled again however it ends;
    // and what a binding of a shared slot unsettled, put back.
    std::vector<uint32_t> settled_here;
    std::vector<std::pair<uint32_t, uint8_t>> was_settled;
    struct Unsettle {
        std::vector<uint8_t>& s;
        std::vector<uint32_t>& slots;
        std::vector<std::pair<uint32_t, uint8_t>>& was;
        ~Unsettle() {
            for (uint32_t slot : slots) s[slot] = 0;
            for (auto it = was.rbegin(); it != was.rend(); ++it) s[it->first] = it->second;
        }
    } unsettle{settled_, settled_here, was_settled};
    BindScope scope(binds_, bind_stmts_, &scope_sig_);
    for (uint32_t i = 0; i < n.b; ++i) {
        uint32_t stmt = img_.kid(n.a + i);
        bool is_last = (i + 1 == n.b);
        // A `let` is not a statement here -- its value is written where its
        // name is read -- so there is nothing to emit where it stands. A block
        // whose *last* statement is one has the value a block of no statements
        // has, which is what the interpreter's `advance_block` ends with too.
        if (Op(img_.node(stmt).op) == Op::Bind) {
            const Node& binding = img_.node(stmt);
            if (binding.a >= f_.slots) return none();
            scope.enter(img_, stmt);
            if (multi_[binding.a]) {
                was_settled.push_back({binding.a, settled_[binding.a]});
                rebind(stmt);
            }
            if (binding.flags & F_STRICT) {
                JV v = is_memo(binding.a) ? bound_value(binding.a) : node(binding.b);
                if (!v.v) return none();
                if (is_memo(binding.a) && binding.a < settled_.size()) {
                    settled_[binding.a] = 1;
                    settled_here.push_back(binding.a);
                }
            }
            continue;
        }
        if (!is_last && !(img_.node(stmt).flags & F_STRICT)) continue;
        JV v = node(stmt, is_last && tail);
        if (failed_) return none();
        if (!v.v) return none();  // control transferred
        if (is_last) last = v;
    }
    return last;
}

/// The nodes under `node` as the emitter would write them, a tree and not a
/// graph, or `limit` once there are that many.
static uint32_t written_size(const Image& img, uint32_t node, uint32_t limit, int depth = 0) {
    if (node == NO_NODE || depth > 256) return 0;
    const Node& n = img.node(node);
    uint32_t size = 1;
    auto add = [&](uint32_t k) {
        if (size < limit) size += written_size(img, k, limit - size, depth + 1);
    };
    switch (Op(n.op)) {
        case Op::If: add(n.a); add(n.b); add(n.c); break;
        case Op::Block: for (uint32_t i = 0; i < n.b; ++i) add(img.kid(n.a + i)); break;
        case Op::SwitchAtom: case Op::SwitchHead:
            add(n.a);
            for (uint32_t i = 1; i < n.c; i += 2) add(img.kid(n.b + i));
            add(img.kid(n.b + n.c - 1));
            break;
        case Op::Bind: add(n.b); break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or: case Op::Get: case Op::Set:
            add(n.a); add(n.b); break;
        case Op::Force: case Op::Neg: case Op::Not: add(n.a); break;
        case Op::Apply: add(n.a); for (uint32_t i = 0; i < n.c; ++i) add(img.kid(n.b + i)); break;
        // The rest are leaves, or hold their parts in lazy positions the
        // emitter writes as a call: one node is near enough.
        default: break;
    }
    return std::min(size, limit);
}

/// The first arm of the chain a switch jumps into, or `NO_NODE`.
///
/// Every target is a suffix of that chain (`dispatch` in dreams/src/lower.dr): an
/// arm is an `if` whose `else` is the arm below it. So the head is the one
/// target no other target reaches down its `else`s. A target that is itself a
/// switch -- the atom switch under a head switch -- is looked through.
static uint32_t chain_head(const Image& img, const Node& sw) {
    std::vector<uint32_t> targets;
    std::function<void(const Node&)> collect = [&](const Node& s) {
        for (uint32_t i = 1; i <= s.c; i += 2) {
            uint32_t t = img.kid(s.b + (i < s.c ? i : s.c - 1));
            Op op = Op(img.node(t).op);
            if (op == Op::SwitchAtom || op == Op::SwitchHead) collect(img.node(t));
            else targets.push_back(t);
            if (i >= s.c) break;
        }
    };
    collect(sw);
    std::unordered_set<uint32_t> below;
    for (uint32_t t : targets) {
        for (uint32_t at = t; Op(img.node(at).op) == Op::If && img.node(at).c != NO_NODE;) {
            at = img.node(at).c;
            if (!below.insert(at).second) break;
        }
    }
    uint32_t head = NO_NODE;
    for (uint32_t t : targets) {
        if (below.count(t)) continue;
        if (head != NO_NODE && head != t) return NO_NODE;
        head = t;
    }
    return head;
}

Emitter::JV Emitter::switch_node(const Node& n, bool tail) {
    // Too many arms to write each target out: the chain once, from its head.
    // That is the program the switch was a shortcut through -- it skips only
    // arms that cannot match, and forces nothing the chain would not.
    {
        uint32_t total = 0;
        for (uint32_t i = 1; i < n.c && total < kMaxSwitchCode; i += 2)
            total += written_size(img_, img_.kid(n.b + i), kMaxSwitchCode);
        if (total >= kMaxSwitchCode) {
            uint32_t head = chain_head(img_, n);
            if (head != NO_NODE) return node(head, tail);
        }
    }
    JV subject = node(n.a);
    if (failed_ || !subject.v) return none();
    llvm::Value* key = box(subject);
    if (Op(n.op) == Op::SwitchHead) {
        auto* out = entry_alloca(i64_);
        auto* ok = guarded(rt_switch_head_, {proc_, key, out});
        key = b_.CreateLoad(i64_, out);
        auto* ready = bb("switch.head");
        auto* raised = bb("switch.raise");
        b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), ready, raised);
        b_.SetInsertPoint(raised);
        emit_raise(key);
        b_.SetInsertPoint(ready);
    } else {
        // A record is keyed by its id, as the interpreter's `take_switch`
        // keys it: the atom whose index is the header word beside the length,
        // without the variant bit. Built inline, so a `match` over a union is
        // one load and a jump table.
        auto* obj = bb("switch.obj");
        auto* rec = bb("switch.record");
        auto* keyed = bb("switch.keyed");
        auto* entry = b_.GetInsertBlock();
        b_.CreateCondBr(is_heap_ptr(key), obj, keyed);
        b_.SetInsertPoint(obj);
        llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(key, ptr_));
        b_.CreateCondBr(b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Record))),
                        rec, keyed);
        b_.SetInsertPoint(rec);
        llvm::Value* tag = b_.CreateZExt(
            b_.CreateLoad(i32_, b_.CreateIntToPtr(b_.CreateAdd(key, i64(12)), ptr_)), i64_);
        llvm::Value* id = b_.CreateOr(b_.CreateShl(b_.CreateAnd(tag, i64(~RECORD_VARIANT & 0xffffffffu)), 8),
                                      i64(make_atom(0)));
        b_.CreateBr(keyed);
        b_.SetInsertPoint(keyed);
        auto* k = b_.CreatePHI(i64_, 3);
        k->addIncoming(key, entry);
        k->addIncoming(key, obj);
        k->addIncoming(id, rec);
        key = k;
    }
    auto* fallback = bb("switch.default");
    auto* done = bb("switch.end");
    auto* dispatch = b_.CreateSwitch(key, fallback, n.c / 2);
    std::vector<std::pair<JV, llvm::BasicBlock*>> values;
    bool doubles = true;
    // Preserve first-match semantics even for hand-built tables with repeated
    // keys (including distinct image atoms interned to the same runtime id).
    std::unordered_set<Value> keys;
    for (uint32_t i = 0; i <= n.c / 2; ++i) {
        const bool last = i == n.c / 2;
        auto* arm = last ? fallback : bb("switch.arm");
        if (!last) {
            const Node& atom = img_.node(img_.kid(n.b + 2 * i));
            Value key_value = make_atom(rt_.image_atom(atom.a));
            if (keys.insert(key_value).second)
                dispatch->addCase(llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(i64_), key_value), arm);
        }
        b_.SetInsertPoint(arm);
        JV v = node(img_.kid(n.b + (last ? n.c - 1 : 2 * i + 1)), tail);
        if (failed_) return none();
        if (v.v) {
            doubles &= v.dbl;
            values.emplace_back(v, b_.GetInsertBlock());
        }
    }
    if (values.empty()) {
        done->eraseFromParent();
        return none();
    }
    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(doubles ? dbl_ : i64_, values.size());
    for (auto& [v, end] : values) {
        b_.SetInsertPoint(end);
        auto* value = doubles ? v.v : box(v);
        b_.CreateBr(done);
        phi->addIncoming(value, b_.GetInsertBlock());
    }
    b_.SetInsertPoint(done);
    return JV{phi, doubles};
}

Emitter::JV Emitter::conditional(const Node& n, bool tail) {
    if (constant_condition(img_, n) == 1) return node(n.b, tail);
    if (constant_condition(img_, n) == 0) return n.c == NO_NODE ? tag(i64(UNIT)) : node(n.c, tail);
    JV cond = node(n.a);
    if (failed_ || !cond.v) return none();
    // A condition that is a float is not a bool, and the tagged path below
    // says so with the message the interpreter would use.
    llvm::Value* c = box(cond);

    auto* then_bb = bb("then");
    auto* else_bb = bb("else");
    auto* bad_bb = bb("if.notbool");
    auto* join_bb = bb("endif");

    llvm::Value* is_true = b_.CreateICmpEQ(c, i64(TRUE_V));
    llvm::Value* is_false = b_.CreateICmpEQ(c, i64(FALSE_V));
    auto* pick = bb("if.pick");
    b_.CreateCondBr(b_.CreateOr(is_true, is_false), pick, bad_bb);

    b_.SetInsertPoint(bad_bb);
    emit_type_error("`if` needs a bool", c);

    b_.SetInsertPoint(pick);
    b_.CreateCondBr(is_true, then_bb, else_bb);

    // Both arms are written before either is closed, because whether they
    // agree about representation decides what each of them has to leave in the
    // register -- and a box belongs at the end of the arm that needs one, not
    // at the join where it would be paid on both paths.
    b_.SetInsertPoint(then_bb);
    JV tv = node(n.b, tail);
    if (failed_) return none();
    llvm::BasicBlock* tb = b_.GetInsertBlock();

    b_.SetInsertPoint(else_bb);
    JV ev = n.c == NO_NODE ? tag(i64(UNIT)) : node(n.c, tail);
    if (failed_) return none();
    llvm::BasicBlock* eb = b_.GetInsertBlock();

    if (!tv.v && !ev.v) {
        // Both arms tail-called; nothing reaches the join.
        join_bb->eraseFromParent();
        return none();
    }
    const bool as_dbl = tv.v && ev.v ? (tv.dbl && ev.dbl) : (tv.v ? tv.dbl : ev.dbl);
    llvm::Value* tval = nullptr;
    llvm::Value* eval = nullptr;
    if (tv.v) {
        b_.SetInsertPoint(tb);
        tval = as_dbl ? tv.v : box(tv);
        b_.CreateBr(join_bb);
        tb = b_.GetInsertBlock();
    }
    if (ev.v) {
        b_.SetInsertPoint(eb);
        eval = as_dbl ? ev.v : box(ev);
        b_.CreateBr(join_bb);
        eb = b_.GetInsertBlock();
    }

    b_.SetInsertPoint(join_bb);
    if (tval && eval) {
        auto* phi = b_.CreatePHI(as_dbl ? dbl_ : i64_, 2);
        phi->addIncoming(tval, tb);
        phi->addIncoming(eval, eb);
        return JV{phi, as_dbl};
    }
    return JV{tval ? tval : eval, as_dbl};
}

Emitter::JV Emitter::logic(const Node& n) {
    JV lhs = node(n.a);
    if (failed_ || !lhs.v) return none();
    llvm::Value* l = box(lhs);
    const bool is_and = (Op(n.op) == Op::And);

    auto* rhs_bb = bb("logic.rhs");
    auto* short_bb = bb("logic.short");
    auto* bad_bb = bb("logic.notbool");
    auto* join_bb = bb("logic.end");

    llvm::Value* is_true = b_.CreateICmpEQ(l, i64(TRUE_V));
    llvm::Value* is_false = b_.CreateICmpEQ(l, i64(FALSE_V));
    auto* pick = bb("logic.pick");
    b_.CreateCondBr(b_.CreateOr(is_true, is_false), pick, bad_bb);

    b_.SetInsertPoint(bad_bb);
    emit_type_error(is_and ? "`and` needs a bool" : "`or` needs a bool", l);

    b_.SetInsertPoint(pick);
    // `and` continues on true, `or` continues on false.
    b_.CreateCondBr(is_true, is_and ? rhs_bb : short_bb, is_and ? short_bb : rhs_bb);

    b_.SetInsertPoint(short_bb);
    llvm::Value* shortcut = i64(is_and ? FALSE_V : TRUE_V);
    b_.CreateBr(join_bb);

    b_.SetInsertPoint(rhs_bb);
    JV rhs = node(n.b);
    if (failed_) return none();
    // The short-circuit answer is a bool, so the two paths only meet as tagged
    // values; a float right-hand side is boxed here rather than at the join.
    llvm::Value* rv = rhs.v ? box(rhs) : nullptr;
    llvm::BasicBlock* rb = b_.GetInsertBlock();
    if (rv) b_.CreateBr(join_bb);

    b_.SetInsertPoint(join_bb);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(shortcut, short_bb);
    if (rv) phi->addIncoming(rv, rb);
    return tag(phi);
}

Emitter::JV Emitter::unary(const Node& n) {
    JV x = node(n.a);
    if (failed_ || !x.v) return none();

    if (Op(n.op) == Op::Not) {
        llvm::Value* v = box(x);
        auto* bad_bb = bb("not.notbool");
        auto* ok_bb = bb("not.ok");
        llvm::Value* is_true = b_.CreateICmpEQ(v, i64(TRUE_V));
        llvm::Value* is_false = b_.CreateICmpEQ(v, i64(FALSE_V));
        b_.CreateCondBr(b_.CreateOr(is_true, is_false), ok_bb, bad_bb);
        b_.SetInsertPoint(bad_bb);
        emit_type_error("`not` needs a bool", v);
        b_.SetInsertPoint(ok_bb);
        return tag(b_.CreateSelect(is_true, i64(FALSE_V), i64(TRUE_V)));
    }

    // Negation is `0 - x`, which is what the interpreter reaches for too --
    // and it is a subtraction rather than a sign flip because the two differ
    // on zero: `0 - 0.0` is `0.0` where `fneg 0.0` is `-0.0`.
    if (x.dbl && Op(n.op) != Op::BitNot) return flt(b_.CreateFSub(llvm::ConstantFP::get(dbl_, 0.0), x.v));

    // The integer case: -(2x+1) as a tagged value is 2 - v, with an overflow
    // check.
    llvm::Value* v = box(x);
    auto* fast_bb = bb("neg.fast");
    auto* slow_bb = bb("neg.slow");
    auto* join_bb = bb("neg.end");
    b_.CreateCondBr(is_fixnum(v), fast_bb, slow_bb);

    b_.SetInsertPoint(fast_bb);
    auto* ssub = llvm::Intrinsic::getOrInsertDeclaration(
        &mod_, llvm::Intrinsic::ssub_with_overflow, {i64_});
    llvm::Value* pair = b_.CreateCall(ssub, {i64(2), v});
    llvm::Value* fastv = b_.CreateExtractValue(pair, 0);
    llvm::Value* ovf = b_.CreateExtractValue(pair, 1);
    auto* fast_ok = bb("neg.fast.ok");
    if (Op(n.op) == Op::BitNot) {
        // Complement the payload bits while retaining the fixnum tag.
        fastv = b_.CreateXor(v, i64(uint64_t(-2)));
        b_.CreateBr(fast_ok);
    } else {
        b_.CreateCondBr(ovf, slow_bb, fast_ok);
    }
    b_.SetInsertPoint(fast_ok);
    b_.CreateBr(join_bb);

    b_.SetInsertPoint(slow_bb);
    llvm::Value* out = entry_alloca(i64_);
    llvm::Value* ok = guarded(
        rt_arith_, {proc_, i32c(int(Op(n.op) == Op::BitNot ? Op::BitNot : Op::Sub)),
                    Op(n.op) == Op::BitNot ? v : i64(make_fixnum(0)), v, out});
    llvm::Value* slowv = b_.CreateLoad(i64_, out);
    auto* slow_end = b_.GetInsertBlock();
    auto* raise_bb = bb("neg.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), join_bb, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(slowv);

    b_.SetInsertPoint(join_bb);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(fastv, fast_ok);
    phi->addIncoming(slowv, slow_end);
    return tag(phi);
}

Emitter::JV Emitter::binary(const Node& n) {
    Op op = Op(n.op);
    JV av = node(n.a);
    if (failed_ || !av.v) return none();
    JV bv = node(n.b);
    if (failed_ || !bv.v) return none();

    const bool is_order = (op == Op::Lt || op == Op::Le || op == Op::Gt || op == Op::Ge);
    const bool is_cmp = is_order || op == Op::Eq || op == Op::Ne;
    const bool is_arith = !is_cmp && op != Op::And && op != Op::Or;
    const bool is_bits = op == Op::BitAnd || op == Op::BitOr || op == Op::BitXor ||
                         op == Op::Shl || op == Op::Shr;

    if (av.dbl || bv.dbl) {
        if (is_arith && !is_bits) return float_arith(op, av, bv);
        if (is_order) return float_order(op, av, bv);
        // `==` and `!=` fall through to the tagged path on purpose. Two float
        // boxes that are the *same object* are equal in this language whatever
        // they hold, NaN included -- `values_equal` compares the words before
        // it compares the numbers -- and once a float is in a register there is
        // no object left to ask about.
    }

    llvm::Value* a = box(av);
    llvm::Value* bb_ = box(bv);

    auto* fast_bb = bb("bin.fast");
    auto* slow_bb = bb("bin.slow");
    auto* join_bb = bb("bin.end");
    auto* branch = b_.CreateCondBr(b_.CreateAnd(is_fixnum(a), is_fixnum(bb_)), fast_bb, slow_bb);
    if (av.integer_hint && bv.integer_hint)
        branch->setMetadata(llvm::LLVMContext::MD_prof,
                            llvm::MDBuilder(ctx_).createBranchWeights(2000, 1));

    b_.SetInsertPoint(fast_bb);
    llvm::Value* fastv = nullptr;
    llvm::BasicBlock* fast_end = nullptr;
    if (is_cmp) {
        // The tag is a left shift plus one, so tagged fixnums compare in the
        // same order as the integers they stand for -- no untagging needed.
        llvm::Value* c = nullptr;
        switch (op) {
            case Op::Lt: c = b_.CreateICmpSLT(a, bb_); break;
            case Op::Le: c = b_.CreateICmpSLE(a, bb_); break;
            case Op::Gt: c = b_.CreateICmpSGT(a, bb_); break;
            case Op::Ge: c = b_.CreateICmpSGE(a, bb_); break;
            case Op::Eq: c = b_.CreateICmpEQ(a, bb_); break;
            default: c = b_.CreateICmpNE(a, bb_); break;
        }
        fastv = b_.CreateSelect(c, i64(TRUE_V), i64(FALSE_V));
        fast_end = b_.GetInsertBlock();
        b_.CreateBr(join_bb);
    } else if (is_bits) {
        llvm::Value* x = b_.CreateAShr(a, 1);
        llvm::Value* y = b_.CreateAShr(bb_, 1);
        if (op == Op::BitAnd) fastv = b_.CreateAnd(a, bb_);
        else if (op == Op::BitOr) fastv = b_.CreateOr(a, bb_);
        else if (op == Op::BitXor) fastv = b_.CreateOr(b_.CreateXor(a, bb_), i64(1));
        else {
            auto* count_ok = bb("bits.count.ok");
            b_.CreateCondBr(b_.CreateICmpSLT(y, i64(0)), slow_bb, count_ok);
            b_.SetInsertPoint(count_ok);
            if (op == Op::Shr) {
                // Clamp before shifting: LLVM shifts by 64 or more are poison.
                llvm::Value* count = b_.CreateSelect(b_.CreateICmpUGT(y, i64(63)), i64(63), y);
                fastv = b_.CreateOr(b_.CreateShl(b_.CreateAShr(x, count), 1), i64(1));
            } else {
                auto* small = bb("bits.left.small");
                b_.CreateCondBr(b_.CreateICmpULT(y, i64(63)), small, slow_bb);
                b_.SetInsertPoint(small);
                // Shift the tagged payload; reversing it detects fixnum overflow.
                llvm::Value* payload = b_.CreateAnd(a, i64(uint64_t(-2)));
                llvm::Value* shifted = b_.CreateShl(payload, y);
                auto* fits = bb("bits.left.fits");
                b_.CreateCondBr(b_.CreateICmpEQ(b_.CreateAShr(shifted, y), payload), fits, slow_bb);
                b_.SetInsertPoint(fits);
                fastv = b_.CreateOr(shifted, i64(1));
            }
        }
        fast_end = b_.GetInsertBlock();
        b_.CreateBr(join_bb);
    } else if (op == Op::Div || op == Op::Mod) {
        // Two fixnums and a divisor that is not zero is one machine
        // instruction. Everything else -- a zero divisor, a float on either
        // side, something that is not a number at all -- is `arith`'s, and
        // goes down the slow path below to be given the error or the float the
        // interpreter would have given.
        //
        // These two used to be the only arithmetic that did not have a fast
        // path here, on the grounds that the zero check and the integer/float
        // split were not worth duplicating. Measured, they were: a
        // ten-million-iteration loop takes 92 ms dividing where the same loop
        // multiplying takes 48, and that difference is a helper call and
        // nothing else. `collatz` pays it on every step for `n % 2` and again
        // on half of them for `n / 2`.
        llvm::Value* x = b_.CreateAShr(a, 1);
        llvm::Value* y = b_.CreateAShr(bb_, 1);
        auto* nonzero = bb("bin.divmod.nz");
        b_.CreateCondBr(b_.CreateICmpEQ(y, i64(0)), slow_bb, nonzero);
        b_.SetInsertPoint(nonzero);
        if (op == Op::Mod) {
            // A remainder is smaller than its divisor, so it is a fixnum
            // whenever the divisor was one and there is nothing to check.
            // `srem` is C's `%`, which is what `arith` computes, so the sign
            // of the answer follows the dividend in both tiers.
            fastv = b_.CreateOr(b_.CreateShl(b_.CreateSRem(x, y), 1), i64(1));
            fast_end = b_.GetInsertBlock();
            b_.CreateBr(join_bb);
        } else {
            // A quotient leaves the fixnum range in exactly one place:
            // `-2^62 / -1` is `2^62`, one past the largest. Retagging is a
            // shift left, so the overflow check the multiply path already uses
            // decides it -- `n` is a fixnum exactly when `n * 2` does not
            // overflow -- and that one case goes to `arith`, which answers the
            // float the interpreter answers.
            llvm::Value* q = b_.CreateSDiv(x, y);
            auto* shl = llvm::Intrinsic::getOrInsertDeclaration(
                &mod_, llvm::Intrinsic::smul_with_overflow, {i64_});
            llvm::Value* pair = b_.CreateCall(shl, {q, i64(2)});
            llvm::Value* shifted = b_.CreateExtractValue(pair, 0);
            llvm::Value* ovf = b_.CreateExtractValue(pair, 1);
            auto* ok_bb = bb("bin.div.ok");
            b_.CreateCondBr(ovf, slow_bb, ok_bb);
            b_.SetInsertPoint(ok_bb);
            fastv = b_.CreateOr(shifted, i64(1));
            fast_end = b_.GetInsertBlock();
            b_.CreateBr(join_bb);
        }
    } else {
        llvm::Intrinsic::ID iid = op == Op::Add   ? llvm::Intrinsic::sadd_with_overflow
                                  : op == Op::Sub ? llvm::Intrinsic::ssub_with_overflow
                                                  : llvm::Intrinsic::smul_with_overflow;
        llvm::Value* lhs = a;
        llvm::Value* rhs = bb_;
        if (op == Op::Mul) {
            // Untag both, multiply, then retag: 2xy+1.
            lhs = b_.CreateAShr(a, 1);
            rhs = b_.CreateAShr(bb_, 1);
        } else if (op == Op::Add) {
            rhs = b_.CreateSub(bb_, i64(1));   // (2x+1) + (2y+1) - 1
        } else {
            rhs = b_.CreateSub(bb_, i64(1));   // (2x+1) - (2y+1) + 1 == a - (b-1)
        }
        auto* fn = llvm::Intrinsic::getOrInsertDeclaration(&mod_, iid, {i64_});
        llvm::Value* pair = b_.CreateCall(fn, {lhs, rhs});
        llvm::Value* raw = b_.CreateExtractValue(pair, 0);
        llvm::Value* ovf = b_.CreateExtractValue(pair, 1);

        auto* ok_bb = bb("bin.fast.ok");
        if (op == Op::Mul) {
            // Retagging shifts left by one, which can overflow independently.
            auto* shl = llvm::Intrinsic::getOrInsertDeclaration(
                &mod_, llvm::Intrinsic::smul_with_overflow, {i64_});
            auto* nofast = bb("bin.mul.retag");
            b_.CreateCondBr(ovf, slow_bb, nofast);
            b_.SetInsertPoint(nofast);
            llvm::Value* p2 = b_.CreateCall(shl, {raw, i64(2)});
            llvm::Value* shifted = b_.CreateExtractValue(p2, 0);
            llvm::Value* ovf2 = b_.CreateExtractValue(p2, 1);
            b_.CreateCondBr(ovf2, slow_bb, ok_bb);
            b_.SetInsertPoint(ok_bb);
            fastv = b_.CreateOr(shifted, i64(1));
        } else {
            b_.CreateCondBr(ovf, slow_bb, ok_bb);
            b_.SetInsertPoint(ok_bb);
            fastv = raw;
        }
        fast_end = b_.GetInsertBlock();
        b_.CreateBr(join_bb);
    }

    b_.SetInsertPoint(slow_bb);
    auto* out = entry_alloca(i64_);
    llvm::FunctionCallee helper = is_cmp ? rt_compare_ : rt_arith_;
    llvm::Value* ok = guarded(helper, {proc_, i32c(int(op)), a, bb_, out});
    llvm::Value* slowv = b_.CreateLoad(i64_, out);
    auto* slow_end = b_.GetInsertBlock();
    auto* raise_bb = bb("bin.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), join_bb, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(slowv);

    b_.SetInsertPoint(join_bb);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(fastv, fast_end);
    phi->addIncoming(slowv, slow_end);
    return tag(phi, is_arith && av.integer_hint && bv.integer_hint);
}

// ---------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------

Emitter::JV Emitter::apply(uint32_t idx, const Node& n) {
    // The shape the analysis recorded for this node, so the two cannot
    // disagree about which of the calls it is.
    switch (kind_at(idx)) {
        case CallKind::Known:
            return native_call(known_native(rt_, img_, n.a, n.c, is_primitive(Op(n.op))), n);
        case CallKind::Native:
            return host_call(native_site(rt_, img_, n.a, n.c, is_primitive(Op(n.op))), n);
        case CallKind::Peer: {
            const GlobalRec& g = img_.global(img_.node(n.a).a);
            return peer_call(g.target, n);
        }
        case CallKind::Self:
            return self_call(n);
        case CallKind::Closure:
            return closure_call(n);
        default:
            failed_ = true;
            return none();
    }
}

/// Anything that can be applied and is none of the shapes above: a closure in
/// a slot, a global function this tier did not take, a partial application.
///
/// The machine makes the call (`dream_rt_apply`), so it means exactly what the
/// interpreter's `Apply` means. The callee is evaluated first, as the
/// interpreter evaluates it; the arguments are in lazy positions, because the
/// callee is free not to look at them; and the answer is forced, because this
/// is an evaluated position. A reduction is spent for the `Apply` node.
///
/// What it costs is a nested machine loop per call. The loop may collect --
/// the machine stack is scanned for what this frame holds, see
/// `Process::jit_stack_hi` -- and the helper clears `items` once the arguments
/// are on the value stack, because nothing here reads it again and a scan
/// would otherwise keep them alive for as long as the call ran. The loop's
/// own reductions come off the same budget; when they exhaust the slice the
/// helper leaves it at zero, so the compiled loop around the call yields at
/// its next back-edge.
Emitter::JV Emitter::closure_call(const Node& n) {
    JV callee = node(n.a);
    if (failed_ || !callee.v) return none();
    llvm::Value* f = box(callee);
    llvm::Value* items = lazy_array(n.b, n.c);
    if (failed_) return none();
    spend_reduction();
    llvm::Value* out = entry_alloca(i64_, "applied");
    llvm::Value* st = guarded(rt_apply_, {proc_, f, i32c(int(n.c)), items, out});
    llvm::Value* v = b_.CreateLoad(i64_, out);
    auto* cont = bb("apply.ok");
    auto* raise_bb = bb("apply.raise");
    auto* bail_bb = bb("apply.impure");
    auto* sw = b_.CreateSwitch(st, raise_bb, 2);
    sw->addCase(llvm::cast<llvm::ConstantInt>(i32c(1)), cont);
    sw->addCase(llvm::cast<llvm::ConstantInt>(i32c(2)), bail_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(v);
    b_.SetInsertPoint(bail_bb);
    emit_bail();
    b_.SetInsertPoint(cont);
    return tag(v);
}

/// The tail every call in this file shares: anything but `JIT_OK` is this
/// invocation's answer too.
///
/// `JIT_RAISED` carries the error, `JIT_DEEP` carries the placeholder a deopt
/// returns -- either way the value in hand is the one to give back, and there
/// is nothing to unwind, because a compiled body has no effects and has written
/// nothing down but its own registers.
///
/// A `JIT_DEEP` from a *peer* is worth one note. It travels up to
/// `enter_function`, which deoptimizes the function it entered -- the root of
/// this compile, not the peer that ran out of stack. That is a misdirected
/// heuristic and not a wrong answer: the root is run by the interpreter, which
/// enters the peer through `enter_function` in its own right, and *that* entry
/// deoptimizes the peer. It corrects itself after one call.
Emitter::JV Emitter::take_call_status(llvm::Value* r, const char* what, bool dbl) {
    auto* ok_bb = bb((std::string(what) + ".ok").c_str());
    auto* out_bb = bb((std::string(what) + ".out").c_str());
    llvm::Value* st = b_.CreateLoad(i32_, status_);
    b_.CreateCondBr(b_.CreateICmpEQ(st, i32c(JIT_OK)), ok_bb, out_bb);

    b_.SetInsertPoint(out_bb);
    b_.CreateRet(r);

    b_.SetInsertPoint(ok_bb);
    // A callee that answers a double hands back its bits; anything but
    // `JIT_OK` is an error or a placeholder in the same register, which is why
    // the status is read before the bits are.
    if (dbl) return flt(b_.CreateBitCast(r, dbl_));
    return tag(r);
}

/// A saturated call to another global function.
///
/// The callee was emitted into this module by the same code that emitted this
/// function, so the call is a machine call with the arguments in registers: no
/// frame, no thunk, and no trip through the interpreter. An argument is
/// evaluated here only where the callee forces that parameter on every path --
/// its own analysis says which (`Analysis::eager`) -- and is in a lazy position
/// everywhere else.
///
/// The frame passed along is this function's, and the callee never writes it:
/// a peer has no tail self call, so it never yields, and it makes its own
/// frames for anything it suspends. It is passed because the shape takes one,
/// not because it means anything here.
Emitter::JV Emitter::peer_call(uint32_t gi, const Node& n) {
    auto found = peers_->find(gi);
    if (found == peers_->end()) {
        failed_ = true;
        return none();
    }
    const PeerFn& pf = found->second;
    // Every argument before the call, and left to right, which is the order the
    // interpreter's own argument list is built in. The ones the callee forces
    // on every path are evaluated; the others are in lazy positions.
    std::vector<JV> args(n.c);
    for (uint32_t i = 0; i < n.c; ++i) {
        const bool eager = pf.eager.test(i);
        args[i] = eager ? node(img_.kid(n.b + i)) : lazy(img_.kid(n.b + i));
        if (failed_ || !args[i].v) return none();
    }
    spend_reduction();

    if (!pf.typed) return call_peer_variant(pf.generic, 0, pf.ret_dbl, args);

    // Which of the typed variant's parameters are not doubles already. None is
    // the case that pays: the caller computed them as floats, and they go
    // across in registers without a box on either side.
    llvm::Value* all_floats = nullptr;
    for (uint32_t i = 0; i < n.c; ++i) {
        if (!pf.typed_slots.test(i) || args[i].dbl) continue;
        // A tagged value in hand -- every argument here is evaluated -- so a
        // float is a float box, and anything else (an integer the types let
        // an unannotated caller pass) goes to the generic variant, which does
        // with it what the interpreter would.
        llvm::Value* v = args[i].v;
        llvm::Value* ptr = is_heap_ptr(v);
        auto* here = b_.GetInsertBlock();
        auto* obj = bb("peer.obj");
        auto* joined = bb("peer.checked");
        b_.CreateCondBr(ptr, obj, joined);
        b_.SetInsertPoint(obj);
        llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
        llvm::Value* is_float =
            b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Float)));
        auto* obj_end = b_.GetInsertBlock();
        b_.CreateBr(joined);
        b_.SetInsertPoint(joined);
        auto* phi = b_.CreatePHI(i1_, 2);
        phi->addIncoming(b_.getFalse(), here);
        phi->addIncoming(is_float, obj_end);
        all_floats = all_floats ? b_.CreateAnd(all_floats, phi) : phi;
    }
    if (!all_floats) return call_peer_variant(pf.typed, pf.typed_slots, pf.typed_ret_dbl, args);

    auto* typed_bb = bb("peer.typed");
    auto* generic_bb = bb("peer.generic");
    auto* done_bb = bb("peer.done");
    b_.CreateCondBr(all_floats, typed_bb, generic_bb);

    b_.SetInsertPoint(typed_bb);
    std::vector<JV> unboxed = args;
    for (uint32_t i = 0; i < n.c; ++i) {
        if (pf.typed_slots.test(i) && !args[i].dbl) unboxed[i] = flt(load_float(args[i].v));
    }
    JV t = call_peer_variant(pf.typed, pf.typed_slots, pf.typed_ret_dbl, unboxed);

    // The two variants meet here. Both answer a double when both bodies are
    // floats; otherwise the double side is boxed, which is the one allocation
    // this path can make and only when the two variants disagree.
    const bool both_dbl = pf.typed_ret_dbl && pf.ret_dbl;
    llvm::Value* tv = both_dbl ? t.v : box(t);
    auto* typed_end = b_.GetInsertBlock();
    b_.CreateBr(done_bb);

    b_.SetInsertPoint(generic_bb);
    JV g = call_peer_variant(pf.generic, 0, pf.ret_dbl, args);
    llvm::Value* gv = both_dbl ? g.v : box(g);
    auto* generic_end = b_.GetInsertBlock();
    b_.CreateBr(done_bb);

    b_.SetInsertPoint(done_bb);
    auto* phi = b_.CreatePHI(both_dbl ? dbl_ : i64_, 2);
    phi->addIncoming(tv, typed_end);
    phi->addIncoming(gv, generic_end);
    return both_dbl ? flt(phi) : tag(phi);
}

Emitter::JV Emitter::call_peer_variant(llvm::Function* fn, SlotSet dbl_slots, bool ret_dbl,
                                       const std::vector<JV>& args) {
    std::vector<llvm::Value*> call{proc_, callee_depth(), frame_, stack_floor_};
    // A tagged parameter is handed a box -- a float in a register is boxed
    // here, the fourth crossing under "Representation" -- and a double one a
    // double. Every double one was either a double already or has just been
    // checked to be a float box and unboxed; nothing converts here.
    for (uint32_t i = 0; i < args.size(); ++i) {
        const bool dbl = dbl_slots.test(i);
        call.push_back(dbl ? args[i].v : box(args[i]));
    }
    call.push_back(status_);
    return take_call_status(guarded(fn, call), "peer", ret_dbl);
}

/// One of the numeric natives, made rather than called.
///
/// Each is the same few instructions the native runs, with the same error where
/// it raises one. What they replace is not the call -- a native call is cheap
/// -- but the *refusal*: a function containing one was not compiled at all.
Emitter::JV Emitter::native_call(KnownNative which, const Node& n) {
    JV x = node(img_.kid(n.b));
    if (failed_ || !x.v) return none();

    if (which == KnownNative::MatchIsCons || which == KnownNative::MatchHead ||
        which == KnownNative::MatchTail) {
        // The natives' own bodies: a cell or not, and the head or the tail of
        // one forced where the machine forces what a native hands back.
        llvm::Value* v = box(x);
        auto* obj = bb("cell.obj");
        auto* test = bb("cell.test");
        auto* here = b_.GetInsertBlock();
        b_.CreateCondBr(is_heap_ptr(v), obj, test);
        b_.SetInsertPoint(obj);
        llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
        llvm::Value* cons_here =
            b_.CreateICmpEQ(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Cons)));
        auto* obj_end = b_.GetInsertBlock();
        b_.CreateBr(test);
        b_.SetInsertPoint(test);
        auto* is_cons = b_.CreatePHI(i1_, 2);
        is_cons->addIncoming(b_.getFalse(), here);
        is_cons->addIncoming(cons_here, obj_end);
        if (which == KnownNative::MatchIsCons) {
            return tag(b_.CreateSelect(is_cons, i64(TRUE_V), i64(FALSE_V)));
        }
        auto* cell = bb("cell.read");
        auto* bad = bb("cell.bad");
        b_.CreateCondBr(is_cons, cell, bad);
        b_.SetInsertPoint(bad);
        emit_type_error(which == KnownNative::MatchHead ? "match_head needs a list cell"
                                                        : "match_tail needs a list cell");
        b_.SetInsertPoint(cell);
        const uint64_t at = which == KnownNative::MatchHead ? 8 : 16;
        llvm::Value* part = b_.CreateLoad(i64_, b_.CreateIntToPtr(b_.CreateAdd(v, i64(at)), ptr_));
        return tag(force(part));
    }

    switch (which) {
        case KnownNative::ToFloat:
            return flt(as_double(x, "to_float needs a number"));
        case KnownNative::Sqrt:
            return flt(unary_intrinsic(llvm::Intrinsic::sqrt,
                                       as_double(x, "sqrt needs a number")));
        case KnownNative::Abs:
            // `abs` keeps the shape it was given, so only the float case is
            // written here; an integer one goes to the native's own code, where
            // the fixnum range makes negating the smallest value a special
            // case not worth repeating.
            if (x.dbl) return flt(unary_intrinsic(llvm::Intrinsic::fabs, x.v));
            break;
        case KnownNative::Floor:
            if (x.dbl) {
                return int_of_double(unary_intrinsic(llvm::Intrinsic::floor, x.v));
            }
            break;
        case KnownNative::ToInt:
            if (x.dbl) return int_of_double(x.v);
            break;
        default:
            break;
    }

    // The tagged cases: hand the value to the runtime, which is the native's
    // own body, and take its answer or its error.
    llvm::FunctionCallee helper = which == KnownNative::Abs     ? rt_abs_
                                  : which == KnownNative::Floor ? rt_floor_
                                                                : rt_to_int_;
    auto* out = entry_alloca(i64_);
    llvm::Value* ok = guarded(helper, {proc_, x.v, out});
    llvm::Value* v = b_.CreateLoad(i64_, out);
    auto* cont = bb("native.ok");
    auto* raise_bb = bb("native.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), cont, raise_bb);
    b_.SetInsertPoint(raise_bb);
    emit_raise(v);
    b_.SetInsertPoint(cont);
    return tag(v);
}

/// A call to a host native.
///
/// One helper call with the arguments in registers. The helper does what the
/// interpreter's `resume_native` does -- put them on the process's value stack,
/// call the native, take its answer -- so there is one implementation of what a
/// native call means and both tiers reach it.
///
/// A reduction is spent, because the interpreter spends one evaluating the
/// `Apply` node this stands for. That keeps `--stats` honest, and it is also
/// what keeps a compiled loop whose body is mostly native calls preemptible:
/// the budget falls at the same rate either tier runs it.
///
/// Every argument is evaluated before the call and left to right, which is the
/// order the interpreter builds its own argument list in. A float in a register
/// is boxed here: a native speaks tagged values and knows nothing about the
/// representation a compiled body chose.
Emitter::JV Emitter::host_call(const NativeSite& site, const Node& n) {
    std::vector<JV> args(n.c);
    for (uint32_t i = 0; i < n.c; ++i) {
        const uint32_t arg = img_.kid(n.b + i);
        // A position the native's mask does not claim is one it may never look
        // at, so what goes there is a lazy position -- built or suspended, and
        // never evaluated.
        const bool strict = i < 32 && ((site.strict_mask >> i) & 1);
        args[i] = strict ? node(arg) : lazy(arg);
        if (failed_ || !args[i].v) return none();
    }
    spend_reduction();

    std::vector<llvm::Value*> call{proc_};
    if (!site.builtin) call.push_back(i32c(int(site.module)));
    call.push_back(i32c(int(site.member)));
    call.push_back(i32c(int(n.c)));
    for (uint32_t i = 0; i < kMaxNativeArgs; ++i) {
        call.push_back(i < n.c ? box(args[i]) : i64(UNIT));
    }
    auto* out = entry_alloca(i64_);
    call.push_back(out);

    llvm::Value* ok = guarded(site.builtin ? rt_builtin_ : rt_native_, call);
    llvm::Value* v = b_.CreateLoad(i64_, out);
    auto* cont = bb("host.ok");
    auto* raise_bb = bb("host.raise");
    b_.CreateCondBr(b_.CreateICmpNE(ok, i32c(0)), cont, raise_bb);

    b_.SetInsertPoint(raise_bb);
    emit_raise(v);

    b_.SetInsertPoint(cont);
    // In weak head normal form, which is the invariant every value in a
    // compiled body has: the helper forces a result the native handed back
    // unforced, at the point the machine would have forced it.
    return tag(v);
}

Emitter::JV Emitter::self_call(const Node& n) {
    std::vector<JV> args;
    if (!self_args(n, &args)) return none();
    if (n.flags & F_TAIL) return tail_call(n, args);
    return recursive_call(args);
}

bool Emitter::self_args(const Node& n, std::vector<JV>* out) {
    // Evaluate every argument before touching any slot: an argument may read a
    // parameter this call is about to overwrite.
    std::vector<JV>& args = *out;
    args.assign(n.c, JV{});
    for (uint32_t i = 0; i < n.c; ++i) {
        // A carried parameter is moved, not evaluated: slot `i` to slot `i`,
        // whatever is in it. That is what the interpreter does with it, and it
        // is why such a parameter needs no strictness -- see `Analyzer::carried`
        // and the note in `Analyzer::run` about who has to force what.
        if (i < f_.arity && slot_is_carried(i)) {
            args[i] = load_slot_raw(i);
            continue;
        }
        // Evaluated where the parameter is one the body forces on every path,
        // and in a lazy position everywhere else. See "Which arguments are
        // evaluated" in `Analyzer::run`.
        const bool eager = eager_.test(i);
        args[i] = eager ? node(img_.kid(n.b + i)) : lazy(img_.kid(n.b + i));
        if (failed_ || !args[i].v) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Recursion that is a loop
//
// `len xs = if is_empty xs { 0 } else { 1 + len (tail xs) }` is recursion, and
// until this it was compiled as recursion: a machine call per level, and past
// the machine stack floor -- `JIT_DEEP`, which gives the
// function to the interpreter for good. Over a long list that is every list
// function written the natural way, and the interpreter then spends a heap
// continuation per element to hold the pending additions.
//
// Two shapes of recursion are loops in disguise, and are compiled as loops:
//
//   * `e + f ..` as the answer, where `e` makes no self call. Every level adds
//     its `e` to what the next level answers, so what is pending is one number:
//     the sum of the `e`s so far. The loop carries it and adds the base case's
//     answer at the end.
//   * `let r = f ..; .. r` with `r` read as the answer, which is a tail call
//     written through a name -- `filter` skipping an element is the shape.
//     Nothing is pending but the level itself.
//
// What makes the first sound is the part worth reading twice. The interpreter
// evaluates `e1 + (e2 + (e3 + b))`: every `e` in order on the way down, then
// the additions innermost first on the way back. The loop evaluates the `e`s in
// the same order -- so anything they raise, they raise at the same point -- and
// sums them as it goes. For fixnums the two sums are the same number *only if
// no partial sum the interpreter makes leaves the fixnum range*, because one
// that does becomes a float and every addition after it is a float addition.
// Those partial sums are the suffix sums `T - P(j-1)`, where `T` is the total
// and `P` the prefix sums the loop has, so the loop keeps the least and the
// greatest prefix sum and checks both ends at the end, in 128 bits. When any `e`
// or the base is not a fixnum, or the check fails, the call is given back to
// the interpreter (`JIT_BAIL`) and run from its frame: nothing compiled code
// did was an effect, and what it forced stays forced, so running it again
// answers exactly what the interpreter would have answered the first time.
//
// A yield is the one thing a pending level cannot survive -- the interpreter
// resumes the body in a frame, and a frame has no room for the accumulator --
// so the loop does not yield while a level is pending. What bounds that is the
// interpreter's own depth limit: past `DREAM_MAX_DEPTH` pending levels the call
// is handed back with `JIT_DEEP`, and the interpreter, doing it on the heap,
// overflows where it always did. Below the limit the loop answers; the
// interpreter, which may spend more than one continuation a level, can run out
// sooner, and that difference -- a recursion deep enough to overflow one tier
// and not the other -- is the one this admits.
// ---------------------------------------------------------------------------

void Emitter::bump_pending() {
    llvm::Value* p = b_.CreateAdd(b_.CreateLoad(i64_, pend_), i64(1));
    b_.CreateStore(p, pend_);
    auto* deep = bb("pending.deep");
    auto* ok = bb("pending.ok");
    b_.CreateCondBr(b_.CreateICmpUGT(p, i64(max_depth_limit())), deep, ok);
    b_.SetInsertPoint(deep);
    b_.CreateStore(i32c(JIT_DEEP), status_);
    b_.CreateRet(i64(UNIT));
    b_.SetInsertPoint(ok);
}

Emitter::JV Emitter::add_step(const Node& n) {
    JV e = node(n.a);
    if (failed_ || !e.v) return none();
    llvm::Value* ev = box(e);
    auto* num = bb("step.fix");
    auto* bail = bb("step.bail");
    b_.CreateCondBr(is_fixnum(ev), num, bail);
    b_.SetInsertPoint(bail);
    emit_bail();
    b_.SetInsertPoint(num);
    llvm::Value* p = b_.CreateLoad(i128_, acc_);
    llvm::Value* lo = b_.CreateLoad(i128_, minp_);
    llvm::Value* hi = b_.CreateLoad(i128_, maxp_);
    b_.CreateStore(b_.CreateSelect(b_.CreateICmpSLT(p, lo), p, lo), minp_);
    b_.CreateStore(b_.CreateSelect(b_.CreateICmpSGT(p, hi), p, hi), maxp_);
    b_.CreateStore(b_.CreateAdd(p, b_.CreateSExt(b_.CreateAShr(ev, 1), i128_)), acc_);
    b_.CreateStore(b_.getTrue(), adds_);
    bump_pending();
    const Node& r = img_.node(n.b);
    std::vector<JV> args;
    if (!self_args(r, &args)) return none();
    return tail_call(r, args);
}

Emitter::JV Emitter::let_step(const Node& n) {
    const uint32_t slot = n.a;
    const Node& r = img_.node(binds_[slot]);
    // Computed already this iteration, or suspended in a frame somebody may
    // hold: then it is a value to hand back, as any read of it is. Only when
    // neither is it a tail call -- and then nobody can tell it was a name.
    if (!is_memo(slot)) {
        failed_ = true;
        return none();
    }
    mark_slot(slot);
    auto* known_bb = bb("letstep.known");
    auto* miss = bb("letstep.miss");
    auto* from_snap = bb("letstep.snap");
    auto* loop = bb("letstep.loop");
    auto* done = bb("letstep.done");
    llvm::Value* known = b_.CreateLoad(i64_, memo_[slot]);
    b_.CreateCondBr(b_.CreateICmpNE(known, i64(0)), known_bb, miss);
    b_.SetInsertPoint(known_bb);
    b_.CreateBr(done);
    b_.SetInsertPoint(miss);
    llvm::Value* snap = b_.CreateLoad(i64_, snap_);
    b_.CreateCondBr(b_.CreateICmpNE(snap, i64(0)), from_snap, loop);
    b_.SetInsertPoint(from_snap);
    llvm::Value* at = b_.CreateIntToPtr(
        b_.CreateAdd(snap, i64(sizeof(FrameObj) + 8 * uint64_t(slot))), ptr_);
    llvm::Value* forced = force(b_.CreateLoad(i64_, at));
    auto* snap_end = b_.GetInsertBlock();
    b_.CreateBr(done);

    b_.SetInsertPoint(loop);
    bump_pending();
    std::vector<JV> args;
    if (!self_args(r, &args)) return none();
    tail_call(r, args);

    b_.SetInsertPoint(done);
    auto* phi = b_.CreatePHI(i64_, 2);
    phi->addIncoming(known, known_bb);
    phi->addIncoming(forced, snap_end);
    return tag(phi);
}

void Emitter::emit_return(JV r) {
    if (ahead_first_) {
        // The end of a run: the answer is the last cell's tail, and the first
        // cell is the answer. With no run, it is the answer as it stands.
        llvm::Value* v = box(r);
        llvm::Value* first = b_.CreateLoad(i64_, ahead_first_);
        auto* plain = bb("ret.plain");
        auto* run = bb("ret.run");
        b_.CreateCondBr(b_.CreateICmpEQ(first, i64(0)), plain, run);
        b_.SetInsertPoint(plain);
        b_.CreateStore(i32c(JIT_OK), status_);
        b_.CreateRet(v);
        b_.SetInsertPoint(run);
        b_.CreateCall(rt_set_tail_, {proc_, b_.CreateLoad(i64_, ahead_last_), v});
        b_.CreateStore(i32c(JIT_OK), status_);
        b_.CreateRet(first);
        return;
    }
    if (!accumulate_) {
        b_.CreateStore(i32c(JIT_OK), status_);
        b_.CreateRet(box(r));
        return;
    }
    llvm::Value* v = box(r);
    auto* plain = bb("ret.plain");
    auto* sum = bb("ret.sum");
    auto* check = bb("ret.check");
    auto* fits = bb("ret.fits");
    auto* bail = bb("ret.bail");
    b_.CreateCondBr(b_.CreateLoad(i1_, adds_), sum, plain);

    b_.SetInsertPoint(plain);
    b_.CreateStore(i32c(JIT_OK), status_);
    b_.CreateRet(v);

    b_.SetInsertPoint(sum);
    b_.CreateCondBr(is_fixnum(v), check, bail);
    b_.SetInsertPoint(check);
    llvm::Value* total =
        b_.CreateAdd(b_.CreateLoad(i128_, acc_), b_.CreateSExt(b_.CreateAShr(v, 1), i128_));
    // Every sum the interpreter would have made on the way back up is
    // `total - P` for a prefix sum `P` the loop passed through. All of them
    // are fixnums exactly when the two extremes are.
    llvm::Value* least = b_.CreateSub(total, b_.CreateLoad(i128_, maxp_));
    llvm::Value* most = b_.CreateSub(total, b_.CreateLoad(i128_, minp_));
    llvm::Value* ok = b_.CreateAnd(
        b_.CreateICmpSGE(least, llvm::ConstantInt::getSigned(i128_, -(int64_t(1) << 62))),
        b_.CreateICmpSLE(most, llvm::ConstantInt::getSigned(i128_, (int64_t(1) << 62) - 1)));
    b_.CreateCondBr(ok, fits, bail);
    b_.SetInsertPoint(fits);
    b_.CreateStore(i32c(JIT_OK), status_);
    b_.CreateRet(b_.CreateOr(b_.CreateShl(b_.CreateTrunc(total, i64_), 1), i64(1)));
    b_.SetInsertPoint(bail);
    emit_bail();
}

// ---------------------------------------------------------------------------
// Building ahead
//
// `replicate n x = if n <= 0 { [] } else { cons x (replicate (n - 1) x) }` is
// the shape of every list producer, and the interpreter runs it one cell at a
// time: the tail is a suspension of the next call, so each cell costs a cell,
// a thunk, a frame for the thunk to read and a trip through the interpreter to
// force it -- 116 bytes and a call, for 24 bytes of data. Compiling the body
// made each trip cheaper and kept all four.
//
// The tail could be evaluated now instead, and for most producers nothing
// would show: the body is pure, so evaluating it early is invisible *unless it
// raises, diverges or does unbounded work* -- and a list nobody reads past its
// first cell is exactly where an eager tail would do all three. So the loop
// evaluates it speculatively, and gives up the moment it cannot be sure:
//
//   * The first iteration -- the call the interpreter actually made -- runs as
//     it always did. When its answer is `cons h (f args)`, it makes the cell
//     with no tail yet and, instead of suspending `f args`, takes the loop's
//     back-edge with `args` as the slots: in lazy positions, built or
//     suspended exactly as the interpreter's `Apply` would hand them over.
//   * Every iteration after that is *ahead*, and an iteration ahead may only
//     do what cannot be observed: inline arithmetic and tests on values in
//     hand, reads of cells and arrays, building. Before everything else --
//     every helper call, since any of them may raise, run code, force a
//     suspension or block; every raise; a bail; a yield -- `ahead_guard`
//     leaves for `ahead_abort_` instead, which makes the last cell's tail a
//     suspension of the *whole body* against a frame of the current slots.
//     That is what the interpreter's thunk of the call would have evaluated,
//     one `Apply` later, and the run so far is the answer. Nothing the
//     iteration did is lost but registers.
//   * A run ends at `kBuildAhead` cells, or when the slice runs out, by
//     suspending the call exactly as before; and an iteration whose answer is
//     not a cell ends it by becoming the last cell's tail.
//
// A strict parameter is one the next iteration will force at once, so a step
// whose argument there is still a suspension would only abort. The run stops
// at that step instead, which costs what a step always cost.
// ---------------------------------------------------------------------------

void Emitter::ahead_guard() {
    if (!ahead_first_) return;
    if (!ahead_abort_) ahead_abort_ = bb("ahead.abort");
    auto* cont = bb("ahead.ok");
    b_.CreateCondBr(b_.CreateICmpNE(b_.CreateLoad(i64_, ahead_first_), i64(0)), ahead_abort_,
                    cont);
    b_.SetInsertPoint(cont);
}

void Emitter::emit_ahead_abort() {
    b_.SetInsertPoint(ahead_abort_);
    llvm::Value* rest = suspend(f_.body);
    b_.CreateCall(rt_set_tail_, {proc_, b_.CreateLoad(i64_, ahead_last_), rest});
    b_.CreateStore(i32c(JIT_OK), status_);
    b_.CreateRet(b_.CreateLoad(i64_, ahead_first_));
}

Emitter::JV Emitter::cons_step(uint32_t idx) {
    const auto [head, call] = cons_parts(img_, img_.node(idx));
    const Node& c = img_.node(call);
    llvm::Value* h = lazy(head).v;
    if (failed_) return none();
    std::vector<llvm::Value*> args(c.c);
    for (uint32_t i = 0; i < c.c; ++i) {
        args[i] = lazy(img_.kid(c.b + i)).v;
        if (failed_) return none();
    }
    llvm::Value* cell =
        b_.CreateCall(rt_cons_after_, {proc_, b_.CreateLoad(i64_, ahead_last_), h});
    llvm::Value* first = b_.CreateLoad(i64_, ahead_first_);
    b_.CreateStore(b_.CreateSelect(b_.CreateICmpEQ(first, i64(0)), cell, first), ahead_first_);
    b_.CreateStore(cell, ahead_last_);
    llvm::Value* left = b_.CreateSub(b_.CreateLoad(i64_, ahead_left_), i64(1));
    b_.CreateStore(left, ahead_left_);
    llvm::Value* go = b_.CreateAnd(b_.CreateICmpSGT(left, i64(0)),
                                   b_.CreateICmpSGT(spend_reduction(), i64(0)));
    for (uint32_t i = 0; i < c.c; ++i) {
        if (!eager_.test(i)) continue;
        // Settled: what `force` lets through without a call.
        llvm::Value* v = args[i];
        auto* test = b_.GetInsertBlock();
        auto* obj = bb("ahead.obj");
        auto* next = bb("ahead.arg");
        b_.CreateCondBr(is_heap_ptr(v), obj, next);
        b_.SetInsertPoint(obj);
        llvm::Value* type = b_.CreateLoad(i8_, b_.CreateIntToPtr(v, ptr_));
        llvm::Value* rel =
            b_.CreateSub(type, llvm::ConstantInt::get(i8_, uint8_t(ObjType::Thunk)));
        llvm::Value* settled = b_.CreateICmpUGT(rel, llvm::ConstantInt::get(i8_, 2));
        auto* obj_end = b_.GetInsertBlock();
        b_.CreateBr(next);
        b_.SetInsertPoint(next);
        auto* phi = b_.CreatePHI(i1_, 2);
        phi->addIncoming(b_.getTrue(), test);
        phi->addIncoming(settled, obj_end);
        go = b_.CreateAnd(go, phi);
    }
    auto* iterate = bb("ahead.iterate");
    auto* stop = bb("ahead.stop");
    b_.CreateCondBr(go, iterate, stop);

    b_.SetInsertPoint(stop);
    b_.CreateCall(rt_set_tail_, {proc_, cell, suspend(call)});
    b_.CreateStore(i32c(JIT_OK), status_);
    b_.CreateRet(b_.CreateLoad(i64_, ahead_first_));

    b_.SetInsertPoint(iterate);
    for (uint32_t i = 0; i < c.c; ++i) b_.CreateStore(args[i], slots_[i]);
    if (fresh_) b_.CreateStore(b_.getFalse(), fresh_);
    b_.CreateBr(loop_header_);
    return none();
}

llvm::Value* Emitter::spend_reduction() {
    llvm::Value* left = b_.CreateLoad(i64_, reduction_slot_);
    llvm::Value* next = b_.CreateSub(left, i64(1));
    b_.CreateStore(next, reduction_slot_);
    return next;
}

void Emitter::emit_yield() {
    // Hand the loop-carried values to the interpreter, which resumes at the top
    // of this body -- in a *new* frame holding them, returned in place of a
    // value. The frame the interpreter entered this call with is not written:
    // once a suspension has been made against it (see `snapshot`, which adopts
    // it), a thunk reads its slots when forced, and overwriting a parameter
    // there would hand an earlier iteration's thunk a later iteration's value
    // -- which for a lazy accumulator is the thunk itself, and a loop that
    // depends on itself. One frame a slice is the price, and it is paid once
    // every few thousand iterations.
    //
    // A slot the loop carried as a double is boxed here, which is the one
    // allocation a fused float loop makes -- once a slice, when it runs out,
    // rather than once per operation.
    //
    // Only the parameters in the recursive shape: the slots past them are
    // never read there. A slot a `let` bound holds nothing worth carrying: the
    // interpreter resumes at the top of the body and runs the `let` again.
    ahead_guard();
    const uint32_t n = recurses_ ? f_.arity : f_.slots;
    llvm::Value* vals = entry_alloca(i64_, "yield.vals", std::max(1u, n));
    for (uint32_t i = 0; i < n; ++i) {
        llvm::Value* v = nullptr;
        if (i < bound_.size() && bound_[i]) {
            v = i64(NIL_SLOT);
        } else {
            v = slot_is_dbl(i) ? box(flt(b_.CreateLoad(dbl_, slots_[i])))
                               : b_.CreateLoad(i64_, slots_[i]);
        }
        b_.CreateStore(v, b_.CreateGEP(i64_, vals, {i64(i)}));
    }
    llvm::Value* fresh = b_.CreateCall(rt_yield_frame_, {proc_, frame_, i32c(int(n)), vals});
    b_.CreateStore(i32c(JIT_YIELD), status_);
    b_.CreateRet(fresh);
}

/// A tail call is the loop back-edge: overwrite the parameters and jump.
Emitter::JV Emitter::tail_call(const Node& n, const std::vector<JV>& args) {
    if (peeling_) return enter_loop(n, args);
    for (uint32_t i = 0; i < n.c; ++i) {
        // The coercions here should both be no-ops: the analyzer decided a slot
        // was a double by looking at exactly these expressions. They are
        // written anyway, so that a disagreement between the two would be a
        // conversion rather than a value read in the wrong shape.
        b_.CreateStore(slot_is_dbl(i)
                           ? as_double(args[i], "a loop carrying a float was given something else")
                           : box(args[i]),
                       slots_[i]);
    }

    // Spend a reduction and check the budget, so a compiled loop is just as
    // preemptible as an interpreted one.
    llvm::Value* next = spend_reduction();

    auto* yield_bb = bb("yield");
    auto* cont_bb = bb("iterate");
    llvm::Value* carry_on = b_.CreateICmpSGT(next, i64(0));
    if (accumulate_) {
        // Levels are pending that exist nowhere but in the accumulator, and a
        // yield hands the interpreter a frame, which has no room for them. So
        // the loop carries on until they are paid, which `bump_pending` bounds.
        carry_on = b_.CreateOr(carry_on, b_.CreateICmpNE(b_.CreateLoad(i64_, pend_), i64(0)));
    }
    if (recurses_) {
        // A yield leaves the interpreter to resume *this* invocation from the
        // top of the body -- which it can only do for the outermost one. Below
        // that there are compiled frames on the machine stack waiting for a
        // value, and returning to the interpreter would abandon them. So the
        // loop keeps going instead, and the slice ends when the whole chain
        // comes back. The budget still falls, so nothing runs unaccounted.
        carry_on = b_.CreateOr(carry_on, b_.CreateICmpNE(depth_, i32c(0)));
    }
    b_.CreateCondBr(carry_on, cont_bb, yield_bb);

    b_.SetInsertPoint(yield_bb);
    emit_yield();

    b_.SetInsertPoint(cont_bb);
    if (fresh_) b_.CreateStore(b_.getFalse(), fresh_);
    b_.CreateBr(loop_header_);
    return none();  // control transferred
}

/// The back-edge of a peeled first iteration: into the loop proper.
///
/// The arguments were evaluated over tagged slots, so a slot the loop carries
/// as a double is handed whatever the first iteration computed, and that is
/// only a float if the program's values are. The fixpoint's claim is about the
/// loop *given* floats on entry -- by induction, every iteration after -- so
/// this is where the induction starts, and it is a guard: a float box goes in
/// as its double, anything else hands the call back (`JIT_BAIL`). Nothing has
/// been written but registers, so the interpreter can run the call as though
/// it had never been compiled; what the first iteration forced stays forced,
/// which is the one thing it did and is what the interpreter would have done.
///
/// A carried slot is read again here rather than taken from `args`. `self_call`
/// reads it before evaluating the arguments after it, and those are usually
/// what force it -- `go cr (zr * zr + cr)` hands `cr` over as the caller's
/// thunk and then forces it one argument later -- so the value it read is the
/// suspension and the slot now holds the float. Reading the slot is still a
/// move and forces nothing.
Emitter::JV Emitter::enter_loop(const Node& n, const std::vector<JV>& args) {
    for (uint32_t i = 0; i < n.c; ++i) {
        JV a = (i < f_.arity && slot_is_carried(i)) ? load_slot_raw(i) : args[i];
        llvm::Value* v = !loop_slot_is_dbl(i) ? box(a)
                         : a.dbl              ? a.v
                                              : guard_float(a.v, peel_bail_);
        b_.CreateStore(v, loop_slots_[i]);
    }
    llvm::Value* next = spend_reduction();
    auto* yield_bb = bb("yield");
    auto* cont_bb = bb("iterate");
    llvm::Value* carry_on = b_.CreateICmpSGT(next, i64(0));
    if (accumulate_) {
        carry_on = b_.CreateOr(carry_on, b_.CreateICmpNE(b_.CreateLoad(i64_, pend_), i64(0)));
    }
    b_.CreateCondBr(carry_on, cont_bb, yield_bb);

    // The yield writes the loop's slots, which are the ones just stored.
    b_.SetInsertPoint(yield_bb);
    std::vector<llvm::Value*> first = slots_;
    slots_ = loop_slots_;
    peeling_ = false;
    emit_yield();
    peeling_ = true;
    slots_ = first;

    b_.SetInsertPoint(cont_bb);
    if (fresh_) b_.CreateStore(b_.getFalse(), fresh_);
    b_.CreateBr(loop_header_);
    return none();
}

/// A self call anywhere but tail position is a machine call, one frame deeper.
/// Its status is the callee's -- see `take_call_status`.
Emitter::JV Emitter::recursive_call(const std::vector<JV>& args) {
    spend_reduction();

    std::vector<llvm::Value*> call{proc_, callee_depth(), frame_, stack_floor_};
    for (uint32_t i = 0; i < args.size(); ++i) {
        call.push_back(slot_is_dbl(i)
                           ? as_double(args[i], "a call carrying a float was given something else")
                           : box(args[i]));
    }
    call.push_back(status_);
    // Whatever the body computes comes back through the return register -- a
    // tagged `Value`, or a peer's double as its bits. The analysis calls a
    // non-tail self call `Ty::Any` either way, which is the cautious answer:
    // the emitter handles whichever representation arrives.
    return take_call_status(b_.CreateCall(fn_, call), "call", ret_dbl_);
}

/// Analyze `func_index` and write it, and everything it calls, into `mod`.
///
/// One module holds the whole closure: the root under `name`, and one internal
/// function per peer. They are all declared before any of them is written,
/// because a call to a peer may be emitted before that peer's own body is --
/// which is what makes mutual recursion between two compiled functions work.
///
/// Answers the root's function, or null if anything in the closure could not be
/// written, in which case the caller drops the module whole.
llvm::Function* emit_closure(llvm::LLVMContext& ctx, llvm::Module& mod, Runtime& rt,
                             const Image& img, uint32_t func_index, const std::string& name) {
    PeerSet peers(rt, img);
    Analysis root = Analyzer(rt, img, func_index, &peers, 0).run();
    // `DREAM_JIT_WHY` (see `Analyzer::why`) for the refusals made here.
    static const bool say = std::getenv("DREAM_JIT_WHY") != nullptr;
    auto refuse = [&](const char* what) -> llvm::Function* {
        if (say) {
            StringRef n = img.str(img.func(func_index).name);
            std::fprintf(stderr, "; jit why fn#%u %.*s: %s\n", func_index, int(n.len), n.data, what);
        }
        return nullptr;
    };
    if (!root.compilable) return nullptr;
    if (!peers.sound(root)) return refuse("a function it calls was refused (`DREAM_JIT_WHY` names it)");

    PeerFns fns;
    for (const auto& member : peers.members) {
        const Analysis& a = member.second;
        const uint32_t arity = img.func(member.first).arity;
        PeerFn& pf = fns[member.first];
        pf.generic = llvm::Function::Create(
            body_signature(ctx, arity, 0), llvm::Function::InternalLinkage,
            "dream_peer_" + std::to_string(member.first), &mod);
        pf.ret_dbl = a.ret_dbl;
        pf.eager = a.eager;
        if (a.typed_slots) {
            pf.typed = llvm::Function::Create(
                body_signature(ctx, arity, a.typed_slots), llvm::Function::InternalLinkage,
                "dream_peer_" + std::to_string(member.first) + ".typed", &mod);
            pf.typed_slots = a.typed_slots;
            pf.typed_ret_dbl = a.typed_ret_dbl;
        }
    }
    // Both variants are written whether or not anything calls them; the
    // optimizer drops an internal function nobody calls before code is
    // generated for it, which is where the time goes.
    for (const auto& member : peers.members) {
        const PeerFn& pf = fns[member.first];
        Emitter generic(ctx, mod, rt, img, member.first, member.second, &fns, pf.generic);
        if (!generic.emit(pf.generic->getName().str())) return refuse("the emitter refused a function it calls");
        if (pf.typed) {
            Analysis t = member.second;
            t.float_slots = pf.typed_slots;
            t.ret_dbl = pf.typed_ret_dbl;
            Emitter typed(ctx, mod, rt, img, member.first, t, &fns, pf.typed);
            if (!typed.emit(pf.typed->getName().str())) return nullptr;
        }
    }
    llvm::Function* f = Emitter(ctx, mod, rt, img, func_index, root, &fns).emit(name);
    return f ? f : refuse("the emitter refused it");
}

}  // namespace

// ---------------------------------------------------------------------------
// Jit
// ---------------------------------------------------------------------------

// Compiling in the background.
//
// LLVM charges about ten milliseconds a function, and a program that is not a
// numeric kernel has hundreds of functions worth compiling -- the compiler's own
// self-compile makes four hundred of them hot, which is four seconds, and every
// one of them used to be paid by the program at the moment the function got
// hot, with the program stopped. Nothing needs it to be: until a compiled body
// exists the interpreter runs the function exactly as it did while it was cold,
// and the two tiers agree on every answer, so *when* the compiled body arrives
// is not observable. So a hot function is queued, a thread compiles the queue,
// and each body is published into `cached_` the moment it is done. A process
// is one green thread and the machine has more cores than that; the compile is
// paid on one of the others.
//
// Synchronous compilation is kept for a threshold of one, which is what the
// tests use to make sure a function's *compiled* body is the one they exercise
// -- and for `DREAM_JIT_SYNC`, when that is wanted anywhere else.

// Several compile threads, not one.
//
// A self-compile makes about a thousand functions hot, and compiling all of
// them costs LLVM a minute of CPU against a program that runs for under three
// seconds -- so one thread working through the queue reaches the first few
// dozen before the program is over, and the rest are interpreted to the end.
// Most machines have cores the program is not using. So there is a pool of
// compile threads (`DREAM_JIT_THREADS`, `--jit-threads`), each taking the
// hottest function waiting rather than the oldest (`take_hottest`), and the CPU
// budget below is shared between them: the pool runs at normal priority while
// what it has spent together is inside it, and at `SCHED_IDLE` past it, where
// it gets the idle cores and nothing else.
//
// Every compile has a context and a module of its own, and the program's image
// and runtime are only read, so the threads share nothing while they work but
// the two tables a finished compile is written into (`mutex`) and LLJIT,
// which is built to be added to from several threads. Its compiler is the one
// thing that was not, which is why it is ours (`SizedCompiler`).

/// The code generator: a target machine per module, made fresh for each one,
/// which is what lets several threads generate code at once -- LLJIT's own
/// compiler shares one -- and at the level the module's size calls for.
///
/// By the time a module reaches code generation, O2 has already done the
/// optimizing that matters, and what the generator's own optimization buys is
/// register allocation and instruction selection for tight loops. Measured on
/// the benchmark loops, that is real: `sum` 43 -> 68 ms and `collatz` 66 -> 107
/// with the fast generator. But it is paid in proportion to size, and on a
/// self-compile the 232 modules over `kFastCodegenInsts` were 88% of the 29 s
/// spent generating code; the fast generator did all of it in 3.6 s. Every
/// benchmark kernel is under 400 instructions. So a large module is generated
/// fast and a small one well -- marked by the flag `dream.fast`.
class SizedCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
public:
    explicit SizedCompiler(llvm::orc::JITTargetMachineBuilder jtmb)
        : IRCompiler(llvm::orc::irManglingOptionsFromTargetOptions(jtmb.getOptions())),
          jtmb_(std::move(jtmb)) {}

    llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(llvm::Module& m) override {
        llvm::orc::JITTargetMachineBuilder b = jtmb_;
        b.setCodeGenOptLevel(m.getModuleFlag("dream.fast") ? llvm::CodeGenOptLevel::None
                                                           : llvm::CodeGenOptLevel::Default);
        auto tm = b.createTargetMachine();
        if (!tm) return tm.takeError();
        return llvm::orc::SimpleCompiler(**tm)(m);
    }

private:
    const llvm::orc::JITTargetMachineBuilder jtmb_;
};

/// Instructions, after O2, past which a module's code is generated fast. See
/// `SizedCompiler`.
constexpr size_t kFastCodegenInsts = 1000;

/// How many compile threads, unless `--jit-threads` says: `DREAM_JIT_THREADS`,
/// or two. More is not faster for a program of a few seconds: on a
/// self-compile (24 threads, 12 cores) one, two and four threads finished
/// within the noise of each other, eight was 8% slower and sixteen 60% slower
/// -- they compile more, and what they compile cannot repay itself in a run
/// that short, while they take cores, cache and the hyperthread beside a
/// program thread. A run long enough to want everything compiled wants
/// `--precompile` instead.
unsigned default_jit_threads() {
    if (const char* v = std::getenv("DREAM_JIT_THREADS")) {
        const long n = std::atol(v);
        if (n > 0) return unsigned(n);
    }
    return 2;
}

/// Queue policy from the environment, overridden by `--jit-policy`. Density
/// won the self-build comparison; the sampled and learned policies remain
/// available for workloads where their profiling cost repays itself.
using JitRank = JitPolicy;

JitRank default_jit_rank() {
    JitRank policy = JitRank::CallsPerSize;
    if (const char* v = std::getenv("DREAM_JIT_RANK")) {
        parse_jit_policy(v, policy);
    }
    return policy;
}

// The rough body size the queue is ranked by.
//
// What a compile costs LLVM is proportional to the module it is handed, which
// is the function's body once `let` has been written out. That is what the
// ranking should divide the call count by -- a big dispatcher that costs a
// second of LLVM is not worth the same slice as a tight loop that costs five
// milliseconds -- but it is not a number the image stores. So it is counted
// here, once per function, walking the body's nodes. The layout is `mark_refs`'s
// (and its comment, which is the authority); the shapes it leaves out are
// counted as one node rather than followed, and a count that is low does no
// harm. It is a ranking, not a promise.
struct SizeScratch {
    /// `stamp[i] == gen` means node `i` was counted this walk, so the scratch
    /// needs no clearing between calls.
    std::vector<uint32_t> stamp;
    uint32_t gen = 0;
};

// The advanced policy learns CPU milliseconds from completed compilations.
// Features are body size and its interaction with local-slot width: wide
// functions emit more snapshot stores and spill more values. Fit only when a
// compile finishes; ranking a queued candidate is three multiplies and adds.
struct CompileCostModel {
    double normal[3][3] = {{4, 0, 0}, {0, 4, 0}, {0, 0, 4}};
    double rhs[3] = {8, 4, 4};
    double coeff[3] = {2, 1, 1};

    static std::array<double, 3> features(uint32_t nodes, uint32_t slots) {
        const double n = double(nodes) / 64.0;
        return {1.0, n, n * double(slots > 16 ? slots - 16 : 0) / 64.0};
    }
    double predict(uint32_t nodes, uint32_t slots) const {
        auto x = features(nodes, slots);
        return std::max(0.25, coeff[0] + coeff[1] * x[1] + coeff[2] * x[2]);
    }
    void observe(uint32_t nodes, uint32_t slots, double ms) {
        auto x = features(nodes, slots);
        // Forget older observations gradually as the program changes phase.
        for (unsigned i = 0; i < 3; ++i) {
            rhs[i] = 0.98 * rhs[i] + x[i] * ms;
            for (unsigned j = 0; j < 3; ++j)
                normal[i][j] = 0.98 * normal[i][j] + x[i] * x[j];
        }
        double a[3][4];
        for (unsigned i = 0; i < 3; ++i) {
            for (unsigned j = 0; j < 3; ++j) a[i][j] = normal[i][j];
            a[i][3] = rhs[i];
        }
        for (unsigned col = 0; col < 3; ++col) {
            unsigned pivot = col;
            for (unsigned row = col + 1; row < 3; ++row)
                if (std::abs(a[row][col]) > std::abs(a[pivot][col])) pivot = row;
            if (std::abs(a[pivot][col]) < 1e-9) return;
            for (unsigned j = col; j < 4; ++j) std::swap(a[col][j], a[pivot][j]);
            const double scale = a[col][col];
            for (unsigned j = col; j < 4; ++j) a[col][j] /= scale;
            for (unsigned row = 0; row < 3; ++row) {
                if (row == col) continue;
                const double scale = a[row][col];
                for (unsigned j = col; j < 4; ++j) a[row][j] -= scale * a[col][j];
            }
        }
        // Compilation cost cannot decrease just because a body gets wider.
        for (unsigned i = 0; i < 3; ++i) coeff[i] = std::max(0.0, a[i][3]);
    }
};

uint32_t body_size(const Image& img, uint32_t root, SizeScratch& s) {
    if (s.stamp.size() < img.node_count()) s.stamp.assign(img.node_count(), 0);
    if (++s.gen == 0) {
        std::fill(s.stamp.begin(), s.stamp.end(), 0);
        s.gen = 1;
    }
    const uint32_t gen = s.gen;
    std::vector<uint32_t> stack;
    auto push = [&](uint32_t i) {
        if (i != NO_NODE && i < img.node_count() && s.stamp[i] != gen) stack.push_back(i);
    };
    push(root);
    uint32_t count = 0;
    while (!stack.empty()) {
        const uint32_t idx = stack.back();
        stack.pop_back();
        if (s.stamp[idx] == gen) continue;
        s.stamp[idx] = gen;
        ++count;
        const Node& n = img.node(idx);
        auto run = [&](uint32_t off, uint32_t count) {
            for (uint32_t i = 0; i < count; ++i) push(img.kid(off + i));
        };
        switch (Op(n.op)) {
            case Op::ConstInt: case Op::ConstFloat: case Op::ConstStr: case Op::ConstChar:
            case Op::ConstBool: case Op::ConstAtom: case Op::Unit: case Op::Capture:
            case Op::Local: case Op::Global: case Op::Builtin: case Op::Nop:
            case Op::MakeClosure: case Op::MakeThunk:
                break;
            case Op::Field: case Op::Force: case Op::TypeIs: case Op::ListTail:
            case Op::ListIsEmpty: case Op::Neg: case Op::Not: case Op::BitNot: case Op::ListIsCons: case Op::ListHead:
                push(n.a);
                break;
            case Op::Bind:
                push(n.b);
                break;
            case Op::Try: case Op::Ensure: case Op::Cons:
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or:
                push(n.a);
                push(n.b);
                break;
            case Op::If: case Op::Get: case Op::Set:
                push(n.a);
                push(n.b);
                push(n.c);
                break;
            case Op::SetMany:
                push(n.a);
                run(n.b, 2 * n.c);
                break;
            case Op::Apply: case Op::SwitchHead: case Op::SwitchAtom:
                push(n.a);
                run(n.b, n.c);
                break;
            case Op::Block: case Op::MakeList: case Op::MakeArray: case Op::RecordMake:
                run(n.a, n.b);
                break;
            case Op::MakeMap:
                run(n.a, 2 * n.b);
                break;
            default:
                break;
        }
    }
    return count;
}

struct Jit::Impl {
    explicit Impl(Runtime& r) : rt(r) {}

    Runtime& rt;
    std::unique_ptr<llvm::orc::LLJIT> lljit;
    std::once_flag jit_once;
    std::string jit_error;
    /// Guards `compiled` and `rejected`, and nothing that runs long.
    std::mutex mutex;
    std::unordered_map<uint32_t, CompiledFn> compiled;
    std::unordered_set<uint32_t> rejected;
    uint32_t threshold = 32;
    std::atomic<uint64_t> compiled_count{0};
    /// Makes each compile's entry symbol unique, so two compiles of one
    /// function -- two workers forcing it synchronously at once -- cannot
    /// define the same name twice.
    std::atomic<uint64_t> serial{0};

    // The compile threads and their queue, guarded by `queue_mutex` -- a lock
    // of their own, so an entry that queues a function never waits on a
    // compile.
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<uint32_t> queue;
    std::vector<std::thread> workers;
    unsigned threads = default_jit_threads();
    bool stopping = false;
    /// How many compile threads are inside LLVM, which cannot be interrupted.
    std::atomic<int> compiling{0};
    /// CPU the compile threads have spent between them. See `jit_within_budget`.
    std::atomic<int64_t> jit_ns{0};
    const bool force_sync = std::getenv("DREAM_JIT_SYNC") != nullptr;
    /// The queue's ranking costs and the scratch that counts them, guarded by
    /// `queue_mutex` like the queue itself. A size of zero is not yet counted;
    /// see `take_hottest`.
    std::vector<uint32_t> sizes;
    SizeScratch scratch;
    JitRank rank = default_jit_rank();
    CompileCostModel cost_model;  // guarded by queue_mutex

    bool ensure_jit(std::string* error);
    bool create_jit();
};

bool Jit::Impl::ensure_jit(std::string* error) {
    std::call_once(jit_once, [this] { create_jit(); });
    if (!lljit && error) *error = jit_error;
    return lljit != nullptr;
}

bool Jit::Impl::create_jit() {
    std::string* error = &jit_error;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    // Code generation is ours rather than LLJIT's default, for two reasons:
    // the default compiler is not safe to call from several threads at once,
    // and several compile threads is the point (see "Compiling in the
    // background"); and the level it generates code at is chosen per module,
    // by size (see `SizedCompiler`).
    auto built = llvm::orc::LLJITBuilder()
                     .setCompileFunctionCreator(
                         [](llvm::orc::JITTargetMachineBuilder jtmb)
                             -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                             return std::make_unique<SizedCompiler>(std::move(jtmb));
                         })
                     .create();
    if (!built) {
        if (error) *error = llvm::toString(built.takeError());
        return false;
    }
    lljit = std::move(*built);

    // Expose the runtime helpers to compiled code by absolute address; they
    // are in this shared object, not something the JIT can look up by name.
    auto& jd = lljit->getMainJITDylib();
    llvm::orc::SymbolMap syms;
    auto add = [&](const char* name, void* addr) {
        syms[lljit->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(addr), llvm::JITSymbolFlags::Exported);
    };
    add("dream_rt_switch_head", reinterpret_cast<void*>(&dream_rt_switch_head));
    add("dream_rt_force", reinterpret_cast<void*>(&dream_rt_force));
    add("dream_rt_arith", reinterpret_cast<void*>(&dream_rt_arith));
    add("dream_rt_compare", reinterpret_cast<void*>(&dream_rt_compare));
    add("dream_rt_float", reinterpret_cast<void*>(&dream_rt_float));
    add("dream_rt_arith_f", reinterpret_cast<void*>(&dream_rt_arith_f));
    add("dream_rt_to_int", reinterpret_cast<void*>(&dream_rt_to_int));
    add("dream_rt_abs", reinterpret_cast<void*>(&dream_rt_abs));
    add("dream_rt_number_double", reinterpret_cast<void*>(&dream_rt_number_double));
    add("dream_rt_floor", reinterpret_cast<void*>(&dream_rt_floor));
    add("dream_rt_int_of_double", reinterpret_cast<void*>(&dream_rt_int_of_double));
    add("dream_rt_type_error", reinterpret_cast<void*>(&dream_rt_type_error));
    add("dream_rt_type_error_got", reinterpret_cast<void*>(&dream_rt_type_error_got));
    add("dream_rt_reduction_slot", reinterpret_cast<void*>(&dream_rt_reduction_slot));
    add("dream_rt_stack_floor", reinterpret_cast<void*>(&dream_rt_stack_floor));
    add("dream_rt_frame_slots", reinterpret_cast<void*>(&dream_rt_frame_slots));
    add("dream_rt_native", reinterpret_cast<void*>(&dream_rt_native));
    add("dream_rt_builtin", reinterpret_cast<void*>(&dream_rt_builtin));
    add("dream_rt_get", reinterpret_cast<void*>(&dream_rt_get));
    add("dream_rt_set", reinterpret_cast<void*>(&dream_rt_set));
    add("dream_rt_set_many", reinterpret_cast<void*>(&dream_rt_set_many));
    add("dream_rt_peek", reinterpret_cast<void*>(&dream_rt_peek));
    add("dream_rt_cons", reinterpret_cast<void*>(&dream_rt_cons));
    add("dream_rt_cons_after", reinterpret_cast<void*>(&dream_rt_cons_after));
    add("dream_rt_set_tail", reinterpret_cast<void*>(&dream_rt_set_tail));
    add("dream_rt_make_list", reinterpret_cast<void*>(&dream_rt_make_list));
    add("dream_rt_make_array", reinterpret_cast<void*>(&dream_rt_make_array));
    add("dream_rt_make_record", reinterpret_cast<void*>(&dream_rt_make_record));
    add("dream_rt_make_map", reinterpret_cast<void*>(&dream_rt_make_map));
    add("dream_rt_settle", reinterpret_cast<void*>(&dream_rt_settle));
    add("dream_rt_rebind", reinterpret_cast<void*>(&dream_rt_rebind));
    add("dream_rt_literal_str", reinterpret_cast<void*>(&dream_rt_literal_str));
    add("dream_rt_global", reinterpret_cast<void*>(&dream_rt_global));
    add("dream_rt_snapshot", reinterpret_cast<void*>(&dream_rt_snapshot));
    add("dream_rt_thunk", reinterpret_cast<void*>(&dream_rt_thunk));
    add("dream_rt_apply", reinterpret_cast<void*>(&dream_rt_apply));
    add("dream_rt_adopt", reinterpret_cast<void*>(&dream_rt_adopt));
    add("dream_rt_yield_frame", reinterpret_cast<void*>(&dream_rt_yield_frame));
    if (auto err = jd.define(llvm::orc::absoluteSymbols(std::move(syms)))) {
        if (error) *error = llvm::toString(std::move(err));
        lljit.reset();
        return false;
    }
    return true;
}

bool Jit::available() { return true; }

Jit::Jit(Runtime& rt) : impl_(std::make_unique<Impl>(rt)) {
    const size_t funcs = impl_->rt.image().func_count();
    counts_ = std::vector<std::atomic<uint32_t>>(funcs);
    work_ = std::vector<std::atomic<uint32_t>>(funcs);
    work_ranking_ = impl_->rank == JitRank::WorkPerSize || impl_->rank == JitRank::Adaptive;
    cached_ = std::vector<std::atomic<CompiledFn>>(funcs);
    bails_ = std::vector<std::atomic<uint8_t>>(funcs);
    threshold_.store(impl_->threshold, std::memory_order_relaxed);
    rt.set_jit(this);
}
bool Jit::compiling() const { return impl_->compiling.load(std::memory_order_acquire) > 0; }

Jit::~Jit() {
    // Whatever is still queued is dropped; a compile in progress is finished,
    // because LLVM cannot be interrupted, and then the thread goes.
    {
        std::lock_guard<std::mutex> g(impl_->queue_mutex);
        impl_->stopping = true;
        impl_->queue.clear();
    }
    impl_->queue_cv.notify_all();
    for (std::thread& w : impl_->workers) {
        if (w.joinable()) w.join();
    }
    impl_->rt.set_jit(nullptr);
}

namespace {

#if defined(__linux__)
int64_t cpu_nanos(clockid_t clock) {
    timespec ts{};
    clock_gettime(clock, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

/// How much CPU the compile threads may take between them at normal priority:
/// an allowance every program gets, plus a share of what the process has spent
/// so far. See `compile_worker`. `DREAM_JIT_SHARE` is the share as a
/// percentage.
bool jit_within_budget(int64_t jit_ns) {
    static const int64_t share_pct = [] {
        if (const char* v = std::getenv("DREAM_JIT_SHARE")) return int64_t(std::atoll(v));
        return int64_t(10);
    }();
    constexpr int64_t kAllowance = 100 * 1000000;  // 100 ms
    const int64_t process_ns = cpu_nanos(CLOCK_PROCESS_CPUTIME_ID);
    return jit_ns * 100 <= kAllowance * 100 + share_pct * (process_ns - jit_ns);
}
#endif

}  // namespace

void Jit::compile_worker() {
#if defined(__linux__)
    // Idle cycles first, and a budget when there are none.
    //
    // When a compiled body arrives is not observable -- the interpreter runs
    // the function exactly as it did while it was cold -- so there is no
    // reason for LLVM to take a core from anything doing the program's work.
    // A self-compile keeps every core busy at once (the loader's parses, the
    // arena being shared beside the checker, the collector's helpers), and at
    // normal priority without limit this thread, which spends seconds of CPU
    // in a compile, made the compile *slower* with the JIT on than off.
    //
    // But `SCHED_IDLE` alone means "never" on a machine with no idle cycles.
    // Pinned to one core, a self-compile compiled 2 functions of the 400 it
    // made hot, and every benchmark ran interpreted from start to finish --
    // `fib` 55 ms -> 900, `collatz` 115 -> 2500. The same happens on any number
    // of cores once the program keeps them all busy. So the thread runs at
    // normal priority while what it has spent is inside a budget -- 100 ms,
    // plus a tenth of what the rest of the process has spent -- and at
    // `SCHED_IDLE` past it. A numeric kernel's one or two functions fit in
    // the allowance and arrive within milliseconds; a program with hundreds of
    // hot functions gets the hottest of them promptly and the rest from idle
    // cycles, and pays at most a tenth for the JIT on a machine that has
    // none. An idle thread still gets the odd slice, which is where it
    // notices the budget has grown back and promotes itself again.
    const clockid_t self_clock = [] {
        clockid_t c;
        return pthread_getcpuclockid(pthread_self(), &c) == 0 ? c : CLOCK_THREAD_CPUTIME_ID;
    }();
    bool idle = false;
    auto set_idle = [&](bool want) {
        if (want == idle) return;
        sched_param param{};
        if (pthread_setschedparam(pthread_self(), want ? SCHED_IDLE : SCHED_OTHER, &param) == 0) {
            idle = want;
        }
    };
#endif
    for (;;) {
        uint32_t func_index = 0;
        {
            std::unique_lock<std::mutex> g(impl_->queue_mutex);
            impl_->queue_cv.wait(g, [&] { return impl_->stopping || !impl_->queue.empty(); });
            if (impl_->stopping) return;
            func_index = take_hottest();
        }
#if defined(__linux__)
        set_idle(!jit_within_budget(impl_->jit_ns.load(std::memory_order_relaxed)));
        const int64_t spent_before = cpu_nanos(self_clock);
#endif
        impl_->compiling.fetch_add(1, std::memory_order_acq_rel);
        CompiledFn fn = compile_one(func_index, nullptr);
        impl_->compiling.fetch_sub(1, std::memory_order_acq_rel);
        if (!fn) {
            std::lock_guard<std::mutex> g(impl_->mutex);
            impl_->rejected.insert(func_index);
        }
#if defined(__linux__)
        impl_->jit_ns.fetch_add(cpu_nanos(self_clock) - spent_before, std::memory_order_relaxed);
#endif
        // From "queued" only: nothing else moves a queued function, but a
        // compare-exchange says so rather than relying on it.
        CompiledFn expected = queued();
        cached_[func_index].compare_exchange_strong(expected, fn ? fn : rejected(),
                                                    std::memory_order_acq_rel);
    }
}

uint32_t Jit::take_hottest() {
    // The queue is in the order functions *became* hot, which is not the
    // order they matter in: a function that crossed the threshold early may
    // be called a handful of times more, and one that crossed it late may
    // be the loop the rest of the program runs in. `tier` keeps counting a
    // function while it waits, so the count says which is which. A linear
    // walk, because it is short beside a compile -- a thousand entries at the
    // very most, against milliseconds of LLVM.
    //
    // The count alone is not the ranking, though: what a compile costs LLVM
    // grows with the body it is handed, from five milliseconds to over a
    // second on the compiler's own widest dispatchers, and under the CPU
    // budget only so many compiles happen at all. So the queue is ranked by
    // calls per node of body -- the count divided by `body_size`, which
    // repays itself in calls the soonest -- and a cheap hot loop beats a big
    // dispatcher of the same count. Each size is counted once and kept.
    const Image& img = impl_->rt.image();
    if (impl_->sizes.size() < img.func_count()) impl_->sizes.assign(img.func_count(), 0);
    auto best = impl_->queue.begin();
    double best_value = -1.0;
    for (auto it = impl_->queue.begin(); it != impl_->queue.end(); ++it) {
        const uint32_t fi = *it;
        const uint32_t n = counts_[fi].load(std::memory_order_relaxed);
        double value = double(n);
        if (work_ranking_) {
            value = std::max(value, 64.0 * work_[fi].load(std::memory_order_relaxed));
        }
        if (impl_->rank != JitRank::Calls) {
            uint32_t& size = impl_->sizes[fi];
            if (size == 0) size = body_size(img, img.func(fi).body, impl_->scratch) + 1;
            if (impl_->rank == JitRank::Adaptive) {
                value /= impl_->cost_model.predict(size, img.func(fi).slots);
            } else {
                value = impl_->rank == JitRank::CallsTimesSize
                            ? value * double(size) : value / double(size);
            }
            if (impl_->rank == JitRank::CallsPerSizeSquared) value /= double(size);
        }
        if (value > best_value) {
            best_value = value;
            best = it;
        }
    }
    const uint32_t func_index = *best;
    static const bool trace = std::getenv("DREAM_JIT_TRACE") != nullptr;
    if (trace) {
        uint32_t& size = impl_->sizes[func_index];
        if (!size) size = body_size(img, img.func(func_index).body, impl_->scratch) + 1;
        StringRef name = img.str(img.func(func_index).name);
        std::fprintf(stderr, "; jit pick fn#%u %.*s policy=%s calls=%u work=%llu nodes=%u "
                             "estimated_cost=%.3f ms score=%.3f\n",
                     func_index, int(name.len), name.data, jit_policy_name(impl_->rank),
                     counts_[func_index].load(std::memory_order_relaxed),
                     static_cast<unsigned long long>(work_[func_index].load(std::memory_order_relaxed)) * 64,
                     size, impl_->cost_model.predict(size, img.func(func_index).slots), best_value);
    }
    impl_->queue.erase(best);
    return func_index;
}

void Jit::set_threshold(uint32_t calls) {
    impl_->threshold = calls;
    threshold_.store(calls, std::memory_order_relaxed);
}
uint32_t Jit::threshold() const { return impl_->threshold; }
uint64_t Jit::compiled_count() const { return impl_->compiled_count.load(); }

CompiledFn Jit::on_enter(uint32_t func_index) {
    // Reached once per function: `tier` has counted the entries inline and
    // this is the one that crossed the threshold. The answer is written back
    // into `cached_` either way -- a compiled body, or the rejected marker --
    // so no later entry comes anywhere near this lock.
    if (!impl_->force_sync && impl_->threshold > 1) {
        // Claimed by exactly one entry: the one that moves it from cold to
        // queued. Every other entry, on any worker, sees the marker and runs
        // the interpreter until the compile thread publishes the body.
        CompiledFn expected = nullptr;
        if (!cached_[func_index].compare_exchange_strong(expected, queued(),
                                                         std::memory_order_acq_rel)) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> g(impl_->queue_mutex);
            if (impl_->stopping) return nullptr;
            impl_->queue.push_back(func_index);
            // The pool is started by the first function to get hot, all of
            // it at once: a thread waiting on an empty queue costs nothing.
            if (impl_->workers.empty()) {
                for (unsigned i = 0; i < impl_->threads; ++i) {
                    impl_->workers.emplace_back([this] { compile_worker(); });
                }
            }
        }
        impl_->queue_cv.notify_one();
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        auto it = impl_->compiled.find(func_index);
        if (it != impl_->compiled.end()) return it->second;
        if (impl_->rejected.count(func_index)) return nullptr;
    }

    CompiledFn fn = compile_one(func_index, nullptr);
    if (!fn) {
        {
            std::lock_guard<std::mutex> g(impl_->mutex);
            impl_->rejected.insert(func_index);
        }
        cached_[func_index].store(rejected(), std::memory_order_release);
        return nullptr;
    }
    cached_[func_index].store(fn, std::memory_order_release);
    return fn;
}

void Jit::deoptimize(uint32_t func_index) {
    if (func_index >= cached_.size()) return;
    // Only the tier table is touched. The compiled body stays in `compiled`
    // and stays valid -- `--dump-jit` and a second runtime over the same image
    // still want it -- but no entry will reach it again, which is the point:
    // the recursion that ran out of machine stack would run out of it every
    // time, and each attempt would throw away the work it did before finding
    // out.
    cached_[func_index].store(rejected(), std::memory_order_release);
}

CompiledFn Jit::compile(uint32_t func_index, std::string* error) {
    return compile_one(func_index, error);
}

// Compiling ahead.
//
// A function is compiled once it is hot because most functions never are, and
// compiling one is not cheap: a self-compile's thousand hot functions cost LLVM
// a minute of CPU, against three seconds of program. But compiled, those
// functions do the same work in about two thirds of the CPU. So a program that
// runs long enough to repay the compile -- or a run where only the run's own
// speed matters -- can have everything compiled first, on every core, while
// nothing else wants them. That is `--precompile`.
//
// Nothing about what is compiled changes. A function the tier refuses is
// marked so, which is what being refused when hot would have done, and the
// interpreter runs it.
Jit::Precompiled Jit::precompile(unsigned threads) {
    const auto t0 = std::chrono::steady_clock::now();
    const Image& img = impl_->rt.image();
    const uint32_t n = uint32_t(img.func_count());
    std::atomic<uint32_t> next{0};
    std::atomic<uint32_t> compiled{0};
    std::atomic<uint32_t> refused{0};
    auto work = [&] {
        for (;;) {
            const uint32_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) return;
            // A thunk or a parameterless global: the tier never takes one, and
            // asking costs an analysis.
            if (img.func(i).arity == 0) continue;
            CompiledFn fn = compile_one(i, nullptr);
            if (fn) {
                compiled.fetch_add(1, std::memory_order_relaxed);
            } else {
                refused.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> g(impl_->mutex);
                impl_->rejected.insert(i);
            }
            cached_[i].store(fn ? fn : rejected(), std::memory_order_release);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned k = 1; k < std::max(1u, threads); ++k) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
    Precompiled out;
    out.compiled = compiled.load();
    out.refused = refused.load();
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

void Jit::set_threads(unsigned n) {
    std::lock_guard<std::mutex> g(impl_->queue_mutex);
    // Only before the pool has started; it is started once, whole.
    if (impl_->workers.empty() && n > 0) impl_->threads = n;
}

void Jit::set_policy(JitPolicy policy) {
    std::lock_guard<std::mutex> g(impl_->queue_mutex);
    if (!impl_->workers.empty()) return;
    impl_->rank = policy;
    work_ranking_ = policy == JitPolicy::WorkPerSize || policy == JitPolicy::Adaptive;
}

JitPolicy Jit::policy() const { return impl_->rank; }

CompiledFn Jit::compile_one(uint32_t func_index, std::string* error) {
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        auto it = impl_->compiled.find(func_index);
        if (it != impl_->compiled.end()) return it->second;
    }
    // `DREAM_JIT_TRACE=1` prints every compile, whether it was taken and what
    // LLVM charged for it -- the number that decided compiling in the
    // background, and the one to read before widening the tier again.
    // The function's name is printed beside its index, as `--profile` names
    // it, so that a trace says which of a library's functions the tier took.
    // With the time split three ways -- writing the IR, optimizing it, and
    // generating code -- and the module's size before and after O2, which is
    // what the last two are proportional to.
    struct Trace {
        using Clock = std::chrono::steady_clock;
        uint32_t fi;
        CompiledFn* result;
        const Image* img;
        Impl* impl;
#if defined(__linux__)
        int64_t cpu_start = impl->rank == JitRank::Adaptive ? cpu_nanos(CLOCK_THREAD_CPUTIME_ID) : 0;
#endif
        Clock::time_point t0 = Clock::now();
        Clock::time_point emitted{}, optimized{};
        size_t insts_in = 0, insts_out = 0;
        static double ms(Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        }
        ~Trace() {
            const Clock::time_point t3 = Clock::now();
            if (*result && impl->rank == JitRank::Adaptive) {
#if defined(__linux__)
                const double spent = double(cpu_nanos(CLOCK_THREAD_CPUTIME_ID) - cpu_start) / 1e6;
#else
                const double spent = ms(t0, t3);
#endif
                std::lock_guard<std::mutex> g(impl->queue_mutex);
                if (impl->sizes.size() < img->func_count()) impl->sizes.assign(img->func_count(), 0);
                uint32_t& size = impl->sizes[fi];
                if (!size) size = body_size(*img, img->func(fi).body, impl->scratch) + 1;
                impl->cost_model.observe(size, img->func(fi).slots, spent);
            }
            static const bool on = std::getenv("DREAM_JIT_TRACE") != nullptr;
            if (!on) return;
            StringRef name = img->str(img->func(fi).name);
            if (!*result) {
                std::fprintf(stderr, "; jit fn#%u %.*s refused %.2f ms\n", fi, int(name.len),
                             name.data, ms(t0, t3));
                return;
            }
            std::fprintf(stderr,
                         "; jit fn#%u %.*s compiled %.2f ms (emit %.2f, opt %.2f, codegen %.2f; "
                         "%zu -> %zu insts%s)\n",
                         fi, int(name.len), name.data, ms(t0, t3), ms(t0, emitted),
                         ms(emitted, optimized), ms(optimized, t3), insts_in, insts_out,
                         insts_out > kFastCodegenInsts ? ", fast codegen" : "");
        }
    };
    CompiledFn traced = nullptr;
    Trace trace{func_index, &traced, &impl_->rt.image(), impl_.get()};

    const Image& img = impl_->rt.image();
    if (!impl_->ensure_jit(error)) return nullptr;
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("dream.jit", *ctx);
    mod->setDataLayout(impl_->lljit->getDataLayout());

    std::string name = "dream_fn_" + std::to_string(func_index) + "_" +
                       std::to_string(impl_->serial.fetch_add(1, std::memory_order_relaxed));
    if (!emit_closure(*ctx, *mod, impl_->rt, img, func_index, name)) {
        if (error) *error = "not compilable by this tier";
        return nullptr;
    }

    // Standard optimization pipeline: without it the tag arithmetic and the
    // repeated slot loads dominate, and the compiled loop is barely faster
    // than the interpreter.
    auto instructions = [&] {
        size_t n = 0;
        for (const llvm::Function& f : *mod) {
            for (const llvm::BasicBlock& b : f) n += b.size();
        }
        return n;
    };
    trace.emitted = std::chrono::steady_clock::now();
    trace.insts_in = instructions();
    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    auto mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    mpm.run(*mod, mam);
    trace.optimized = std::chrono::steady_clock::now();
    trace.insts_out = instructions();
    if (trace.insts_out > kFastCodegenInsts) {
        mod->addModuleFlag(llvm::Module::Warning, "dream.fast", 1);
    }

    if (auto err = impl_->lljit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(mod), std::move(ctx)))) {
        if (error) *error = llvm::toString(std::move(err));
        return nullptr;
    }
    auto sym = impl_->lljit->lookup(name);
    if (!sym) {
        if (error) *error = llvm::toString(sym.takeError());
        return nullptr;
    }
    auto fn = sym->toPtr<CompiledFn>();
    {
        // Two compiles of one function can race only on the synchronous
        // path; the first to finish is the one every later caller gets.
        std::lock_guard<std::mutex> g(impl_->mutex);
        auto [at, fresh] = impl_->compiled.emplace(func_index, fn);
        if (fresh) ++impl_->compiled_count;
        fn = at->second;
    }
    traced = fn;
    return fn;
}

std::string Jit::dump_ir(uint32_t func_index) {
    const Image& img = impl_->rt.image();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("dream.jit", *ctx);
    if (!emit_closure(*ctx, *mod, impl_->rt, img, func_index,
                      "dream_fn_" + std::to_string(func_index))) {
        return "; fn#" + std::to_string(func_index) +
               " is not compilable by this tier (it is interpreted)\n";
    }
    std::string out;
    llvm::raw_string_ostream os(out);
    mod->print(os, nullptr);
    return out;
}

}  // namespace dream
