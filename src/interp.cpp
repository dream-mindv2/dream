#include "arithmetic.hpp"
#include <cstdlib>

#include "interp.hpp"
#include "tensor.hpp"
#include "bigint.hpp"

#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <mutex>

#include "builtins.hpp"
#include "jit.hpp"
#include "scheduler.hpp"

namespace dream {

std::atomic<uint64_t> g_thunk_counts[64];
const bool g_probe_thunk = std::getenv("DREAM_PROBE_THUNK") != nullptr;

bool nested_whnf(Process& p, Value v, const Value* args, uint32_t argc, Value* out,
                 Value* taken = nullptr);

namespace {

inline const Image& img_of(Process& p) { return *p.code; }

/// A nested `force_whnf` loop is in progress on this process's machine stack.
/// See `Process::force_nest` and `kMaxForceNestForCompiled`: the depth is what
/// `enter_function` reads before offering the compiled tier, because each link
/// of a lazy chain a *compiled* body forces is a C++ frame.
struct ForceNest {
    Process& p;
    explicit ForceNest(Process& proc) : p(proc) { ++p.force_nest; }
    ~ForceNest() { --p.force_nest; }
    ForceNest(const ForceNest&) = delete;
    ForceNest& operator=(const ForceNest&) = delete;
};

/// The machine loop now running on this C++ stack, for a collection under
/// compiled code to scan up to (`Process::jit_stack_hi`). Its frame address
/// is passed in rather than taken here, because it has to be the loop's.
struct LoopStack {
    Process& p;
    void* const saved;
    LoopStack(Process& proc, void* frame) : p(proc), saved(proc.loop_stack_hi) {
        p.loop_stack_hi = frame;
    }
    ~LoopStack() { p.loop_stack_hi = saved; }
    LoopStack(const LoopStack&) = delete;
    LoopStack& operator=(const LoopStack&) = delete;
};

inline void eval_node(Process& p, uint32_t node, Value frame) {
    p.mode = Mode::Eval;
    p.node = node;
    p.frame = frame;
}

inline void ret(Process& p, Value v) {
    p.mode = Mode::Return;
    p.result = v;
}

inline void do_raise(Process& p, Value err) {
    p.mode = Mode::Raise;
    p.result = err;
}

/// One of a process's interpreter frames, which die with their call (see
/// `FrameStack`), rather than one on the heap.
inline bool stack_frame(Value frame) { return (as_obj(frame)->gc & GC_SHARED) != 0; }

/// A suspension of `node` that carries the values it reads from `frame`
/// instead of the frame (see `ThunkObj`). Nothing is lost by copying them now:
/// a slot is not bound again while something made against its binding can
/// still read it, which is the rule the compiler's slot allocation keeps
/// already, since a suspension used to read the frame when it was forced.
///
/// The one slot that can be read *before* it is bound is a `let rec`'s: a
/// closure that refers to itself, or to one bound after it. `self` is the
/// slot this suspension is about to be bound to, and reads as the suspension;
/// any other slot not bound yet gets a hole, which the slot keeps until its
/// binding fills it (`bind_slot`). A self-compile makes 84 million of these
/// and not one reads a slot forward outside a `let rec`.
Value flat_thunk(Process& p, uint32_t node, Value frame, uint32_t self = NO_NODE) {
    const Image& img = img_of(p);
    const uint32_t* shape = img.flat_shape(node);
    const uint32_t k = shape[0];
    auto* fo = static_cast<FrameObj*>(as_obj(frame));
    // No collection can happen between here and the return: allocation never
    // collects, so `fo` and `t` stay good across the holes made below.
    Value th = p.heap().make_flat_thunk(node, fo->closure, k);
    auto* t = static_cast<ThunkObj*>(as_obj(th));
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t slot = shape[1 + i] & ~Image::FLAT_BOUND;
        Value v = slot < fo->nslots ? fo->slots()[slot] : NIL_SLOT;
        if (slot == self) {
            v = th;
        } else if (v == NIL_SLOT && !(shape[1 + i] & Image::FLAT_BOUND) && slot < fo->nslots) {
            v = p.heap().make_thunk(HOLE_NODE, UNIT);
            fo->slots()[slot] = v;
        }
        t->values()[i] = v;
    }
    return th;
}

/// Bind a slot, filling the hole a flat suspension left in it if one read the
/// slot before this binding (see `flat_thunk`): the hole becomes an
/// indirection to the value, so the suspension holding it sees the binding.
inline void bind_slot(Process& p, FrameObj* fo, uint32_t slot, Value v) {
    const Value old = fo->slots()[slot];
    if (is_ptr(old) && obj_type(as_obj(old)) == ObjType::Thunk &&
        static_cast<ThunkObj*>(as_obj(old))->node == HOLE_NODE) {
        Obj* hole = as_obj(old);
        p.heap().remember_if_old(hole, v);
        std::atomic_ref<Value>(static_cast<IndirectObj*>(hole)->target).store(v, std::memory_order_relaxed);
        obj_type_store_release(hole, ObjType::Indirect);
    }
    value_slot_store(&fo->slots()[slot], v);
    p.heap().remember_if_old(fo, v);
}

/// Evaluate a suspension that `enter` has blackholed and put an update under.
/// A flat one gets a frame of its function's size, holding the values it
/// carried in the slots they came from.
inline void run_suspension(Process& p, ThunkObj* th) {
    if (!thunk_flat(th)) {
        if (th->node == HOLE_NODE) {
            do_raise(p, raise_error(p, well_known(p.runtime()).loop, "value depends on itself"));
            return;
        }
        eval_node(p, th->node, th->frame);
        return;
    }
    const Image& img = img_of(p);
    const Value closure = th->frame;
    if (!is_obj(closure, ObjType::Closure)) p.unreachable("a flat suspension without a closure");
    const uint32_t nslots = img.func(static_cast<ClosureObj*>(as_obj(closure))->func).slots;
    Value fr = p.push_frame(closure, nslots);
    Value* slots = static_cast<FrameObj*>(as_obj(fr))->slots();
    const uint32_t* shape = img.flat_shape(th->node);
    const uint32_t k = shape[0];
    for (uint32_t i = 0; i < k; ++i) slots[shape[1 + i] & ~Image::FLAT_BOUND] = th->values()[i];
    eval_node(p, th->node, fr);
}

inline void push_cont(Process& p, ContKind k, uint32_t a, uint32_t b, uint32_t c, Value v1) {
    p.conts.push_back(Cont{k, 0, 0, a, b, c, v1});
}

/// Arrange for an operation to be run again when the process is woken.
///
/// Normally that means pushing its continuation on top. But when a nested
/// force inside the operation was *suspended* rather than abandoned (see
/// `force_whnf`), the work it had already done is still on the continuation
/// stack and has to finish first -- so the retry is spliced in underneath it.
/// The operation is then re-run against a value that is already forced, which
/// is what stops it repeating any effect the suspended work performed.
inline void push_retry(Process& p, ContKind k, uint32_t a, uint32_t b, uint32_t c, Value v1) {
    Cont retry{k, 0, 0, a, b, c, v1};
    if (p.force_blocked) {
        p.conts.insert_at(p.force_resume_at, retry);
        p.force_blocked = false;
    } else {
        p.conts.push_back(retry);
    }
}

// ---------------------------------------------------------------------------
// The effect check
// ---------------------------------------------------------------------------
//
// Compiled code that parks is retried from the top (`EnterRetry`), and the
// whole of what makes that sound is two assumptions: a compiled body performs
// no effect itself, and anything it forces that does is memoised, so the
// second attempt reads what the first one did instead of doing it again. Both
// have had gaps. The last one was a lambda built inside an impure function and
// handed to a pure peer: the retry built it afresh, and `mind docs` started the
// same example compiles until it was killed -- which looked like a hang in a
// tool, a long way from the JIT.
//
// `DREAM_JIT_CHECK_EFFECTS=1` turns the next such gap into an abort at the
// place it happens. Every call of a `!` native made while compiled code is on
// the machine stack is fingerprinted -- its name and a shallow hash of its
// arguments -- and kept against each compiled call it is under; when one of
// those calls parks, its retry carries the fingerprints of every attempt so
// far, and the same effect turning up again under the retry stops the VM,
// naming the function and the native. The legitimate repeat -- the native that
// parked, called again when it is woken -- is never under the retry: it is in
// the suspended work the retry is spliced beneath (`push_retry`), which runs
// after the compiled frames are gone.
//
// Off, all of this is one load of a static and a branch per compiled call and
// per native call.

static bool effect_check_on() {
    static const bool on = [] {
        const char* v = std::getenv("DREAM_JIT_CHECK_EFFECTS");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

namespace {

uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

/// What an argument looks like without forcing anything: the bytes of a
/// string, an immediate as it is, the first few elements of a list, and for
/// anything else its type. A thunk is not looked into -- a retry makes a new
/// one each time, so its address would make every call look different, and
/// forcing it here would itself be an effect.
uint64_t shallow_hash(Value v, int depth) {
    v = resolve(v);
    if (!is_ptr(v)) return mix(1, uint64_t(v));
    Bytes b;
    if (string_bytes(v, &b)) {
        uint64_t h = 2;
        for (uint64_t i = 0; i < b.len; ++i) h = mix(h, uint8_t(b.data[i]));
        return h;
    }
    Obj* o = as_obj(v);
    uint64_t h = mix(3, uint64_t(o->type));
    if (o->type == ObjType::Cons && depth > 0) {
        int n = 0;
        for (Value c = v; n < 16; ++n) {
            c = resolve(c);
            if (!is_obj(c, ObjType::Cons)) break;
            auto* cell = static_cast<ConsObj*>(as_obj(c));
            h = mix(h, shallow_hash(cell->head, depth - 1));
            c = cell->tail;
        }
    }
    return h;
}

std::string native_name(Value callee) {
    if (is_builtin(callee)) return builtin_def(uint32_t(imm_payload(callee))).name;
    Bytes b;
    auto* nat = static_cast<NativeObj*>(as_obj(callee));
    if (string_bytes(nat->name, &b)) return std::string(b.data, b.len);
    return "<native>";
}

std::string function_name(Process& p, uint32_t fi) {
    StringRef name = img_of(p).str(img_of(p).func(fi).name);
    return std::string(name.data, name.len);
}

/// A `!` native is about to be called under compiled code: abort if an
/// earlier attempt of a retried compiled call made the same call, and
/// otherwise record it against every compiled call it is under.
void effect_check_native(Process& p, Value callee, const Value* args, uint32_t argc) {
    const std::string name = native_name(callee);
    if (name.empty() || name.back() != '!') return;
    uint64_t fp = 4;
    for (char ch : name) fp = mix(fp, uint8_t(ch));
    for (uint32_t i = 0; i < argc; ++i) fp = mix(fp, shallow_hash(args[i], 2));
    for (const auto& a : p.effect_attempts) {
        for (uint64_t seen : a.prior) {
            if (seen != fp) continue;
            std::fprintf(stderr,
                "dream: DREAM_JIT_CHECK_EFFECTS: `%s` was called again by a retry of "
                "compiled `%s` (function #%u).\n"
                "  A compiled call that parks is run again from the top, and this effect\n"
                "  had already happened in an earlier attempt: something under the call\n"
                "  is not memoised. `DREAM_JIT_DENY=%s` keeps the function out of the JIT.\n",
                name.c_str(), function_name(p, a.func).c_str(), a.func,
                function_name(p, a.func).c_str());
            std::fflush(stderr);
            std::abort();
        }
    }
    p.effect_attempts.back().seen.push_back(fp);
}

/// Entering a compiled call: its record, and what earlier attempts did if
/// this is a retry.
void effect_check_enter(Process& p, uint32_t fi) {
    Process::EffectAttempt a{fi, {}, {}};
    if (p.effect_retry_id) {
        auto it = p.effect_retries.find(p.effect_retry_id);
        if (it != p.effect_retries.end()) {
            a.prior = std::move(it->second);
            p.effect_retries.erase(it);
        }
        p.effect_retry_id = 0;
    }
    p.effect_attempts.push_back(std::move(a));
}

/// Leaving one. An inner call's effects are the outer's too, so they move up;
/// a call that parked keeps every attempt's for its retry, under the id it
/// answers.
uint32_t effect_check_leave(Process& p, bool parked) {
    Process::EffectAttempt a = std::move(p.effect_attempts.back());
    p.effect_attempts.pop_back();
    if (!p.effect_attempts.empty()) {
        auto& up = p.effect_attempts.back().seen;
        up.insert(up.end(), a.seen.begin(), a.seen.end());
    }
    if (!parked) return 0;
    a.prior.insert(a.prior.end(), a.seen.begin(), a.seen.end());
    const uint32_t id = p.effect_retry_next++;
    if (p.effect_retry_next == 0) p.effect_retry_next = 1;
    p.effect_retries[id] = std::move(a.prior);
    return id;
}

}  // namespace

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

inline bool is_number(Value v) {
    return is_fixnum(v) || is_obj(v, ObjType::Float) || is_obj(v, ObjType::BigInt);
}
inline double to_double(Value v) {
    if (is_fixnum(v)) return double(fixnum_value(v));
    if (is_obj(v, ObjType::Float)) return static_cast<FloatObj*>(as_obj(v))->value;
    return bigint::to_double(v);
}

}  // namespace

Value make_native(Process& p, NativeFn fn, Value name, uint32_t arity,
                  uint32_t strict_mask, uint64_t user) {
    auto* nat = static_cast<NativeObj*>(p.heap().alloc(
        ObjType::Native, sizeof(void*) + sizeof(Value) + 8 + sizeof(uint64_t)));
    nat->fn = reinterpret_cast<void*>(fn);
    nat->name = name;
    nat->arity = arity;
    nat->strict_mask = strict_mask;
    nat->user = user;
    return from_obj(nat);
}

Value make_integer(Process& p, int64_t v) {
    if (fixnum_fits(v)) return make_fixnum(v);
    return bigint::from_int64(p.heap(), v);
}

Value raise_error(Process& p, uint32_t kind_atom, const std::string& message) {
    Value msg = p.heap().make_string(message.data(), uint32_t(message.size()));
    return p.heap().make_error(make_atom(kind_atom), msg);
}

namespace {

[[nodiscard]] Value type_error(Process& p, const std::string& msg) {
    return raise_error(p, well_known(p.runtime()).type_error, msg);
}

/// The `StrObj` for image string constant `index`, made once per process.
///
/// A literal is immutable and a program cannot observe which copy of it it
/// holds -- strings compare and hash by their bytes -- so evaluating `"("` in
/// a loop has no business allocating. Nothing in the VM ever writes into a
/// string it did not just make (concatenation and slicing both build a fresh
/// one), which is what makes sharing safe rather than merely cheap.
///
/// The cache is a process root, so a shared literal survives collection and
/// keeps its identity across one.
inline Value literal_string(Process& p, uint32_t index) {
    if (index < p.string_cache.size()) {
        Value hit = p.string_cache[index];
        if (hit != NIL_SLOT) return hit;
    } else {
        p.string_cache.assign(img_of(p).string_count(), NIL_SLOT);
    }
    StringRef s = img_of(p).str(index);
    Value v = p.heap().make_string(s.data, s.len);
    if (index < p.string_cache.size()) p.string_cache[index] = v;
    return v;
}

/// The boxed float for image float constant `index`, made once per process.
/// Same argument as `literal_string`: immutable, no observable identity, and
/// otherwise one allocation every time the constant is reached.
inline Value literal_float(Process& p, uint32_t index) {
    if (index < p.float_cache.size()) {
        Value hit = p.float_cache[index];
        if (hit != NIL_SLOT) return hit;
    } else {
        p.float_cache.assign(img_of(p).float_count(), NIL_SLOT);
    }
    Value v = p.heap().make_float(img_of(p).real(index));
    if (index < p.float_cache.size()) p.float_cache[index] = v;
    return v;
}

// ---------------------------------------------------------------------------
// Structural equality
//
// Forces as it descends, so `[1,2] == [1,2]` compares element by element even
// though both lists arrive as unevaluated thunks.
// ---------------------------------------------------------------------------

bool values_equal(Process& p, Value a, Value b, bool* raised, int depth);

bool equal_children(Process& p, Value* xs, Value* ys, uint32_t n, bool* raised, int depth) {
    for (uint32_t i = 0; i < n; ++i) {
        if (!values_equal(p, xs[i], ys[i], raised, depth + 1)) return false;
        if (*raised) return false;
    }
    return true;
}

bool values_equal(Process& p, Value a, Value b, bool* raised, int depth) {
    if (depth > 512) {
        p.result = type_error(p, "structure is too deeply nested to compare");
        *raised = true;
        return false;
    }
    // `resolve` and the WHNF test are inline; `force_whnf` is a call into a
    // nested machine loop. Both operands of a comparison have usually been
    // forced by the machine already, so ask before calling.
    Value fa = resolve(a), fb = resolve(b);
    if ((!is_whnf(fa) && !force_whnf(p, fa, &fa)) ||
        (!is_whnf(fb) && !force_whnf(p, fb, &fb))) {
        *raised = true;
        return false;
    }
    if (fa == fb) return true;

    if (is_number(fa) && is_number(fb)) {
        if (is_fixnum(fa) && is_fixnum(fb)) return fa == fb;
        // Both canonical, so a fixnum and a bignum are never the same integer.
        if (bigint::is_integer(fa) && bigint::is_integer(fb)) {
            return bigint::is_big(fa) && bigint::is_big(fb) &&
                   bigint::equal(static_cast<BigIntObj*>(as_obj(fa)),
                                 static_cast<BigIntObj*>(as_obj(fb)));
        }
        return to_double(fa) == to_double(fb);
    }
    if (!is_ptr(fa) || !is_ptr(fb)) return false;
    Obj* oa = as_obj(fa);
    Obj* ob = as_obj(fb);

    // Strings compare by their bytes, and a big string is a string whose bytes
    // happen to live in the image. Deciding otherwise would make
    // `str.slice 0 4 payload == "%PDF"` -- the first thing anyone does with a
    // payload -- quietly false, which is a worse surprise than the two
    // representations being visible to `type_of`.
    Bytes xs, ys;
    if (string_bytes(fa, &xs) && string_bytes(fb, &ys)) return bytes_equal(xs, ys);

    if (oa->type != ob->type) return false;

    switch (oa->type) {
        case ObjType::Cons: {
            // Down the spine in a loop, as `compare` walks it: recursing on
            // the tail made a list deep as well as long, and two lists of a
            // thousand elements could not be compared at all -- the depth
            // limit is for nesting, and only the elements nest.
            Value xa = fa, xb = fb;
            for (;;) {
                auto* x = static_cast<ConsObj*>(as_obj(xa));
                auto* y = static_cast<ConsObj*>(as_obj(xb));
                if (!values_equal(p, x->head, y->head, raised, depth + 1) || *raised) return false;
                xa = resolve(x->tail);
                xb = resolve(y->tail);
                if ((!is_whnf(xa) && !force_whnf(p, xa, &xa)) ||
                    (!is_whnf(xb) && !force_whnf(p, xb, &xb))) {
                    *raised = true;
                    return false;
                }
                if (xa == xb) return true;
                if (!is_obj(xa, ObjType::Cons) || !is_obj(xb, ObjType::Cons)) {
                    return values_equal(p, xa, xb, raised, depth + 1);
                }
            }
        }
        case ObjType::Trunk:
            return trunk_equal(p, fa, fb, raised, depth);
        case ObjType::Array: {
            auto* x = static_cast<ArrayObj*>(oa);
            auto* y = static_cast<ArrayObj*>(ob);
            if (x->len != y->len) return false;
            return equal_children(p, x->items(), y->items(), x->len, raised, depth);
        }
        case ObjType::Record: {
            // The same declaration, then the same fields. The id is the whole
            // of "the same declaration": it is qualified by module, and a
            // variant's names its union as well as itself.
            auto* x = static_cast<RecordObj*>(oa);
            auto* y = static_cast<RecordObj*>(ob);
            if (x->tag != y->tag || x->len != y->len) return false;
            return equal_children(p, x->items(), y->items(), x->len, raised, depth);
        }
        case ObjType::Pid:
            return static_cast<PidObj*>(oa)->id == static_cast<PidObj*>(ob)->id;
        case ObjType::Tensor:
            return tensor_equal(p, fa, fb, raised);
        case ObjType::Map: {
            auto* x = static_cast<MapObj*>(oa);
            auto* y = static_cast<MapObj*>(ob);
            if (x->count != y->count) return false;
            // Two maps holding the same entries may have different shapes only
            // if their hashes differ, which they cannot -- but comparing by
            // lookup rather than by shape is what makes that not something to
            // rely on.
            std::vector<std::pair<Value, Value>> entries;
            map_collect(fa, entries);
            for (auto& [k, v] : entries) {
                Value found;
                if (!map_lookup(p, fb, k, &found)) return false;
                if (!values_equal(p, v, found, raised, depth + 1)) return false;
                if (*raised) return false;
            }
            return true;
        }
        default:
            return false;  // functions, modules: identity only, already checked
    }
}

// ---------------------------------------------------------------------------
// Applying
// ---------------------------------------------------------------------------

void resume_native(Process& p, Value callee, uint32_t base, uint32_t argc, uint32_t from);
void enter(Process& p, Value v);
bool callee_operand(Process& p, const Image& img, uint32_t node, Value frame, Value* out);

/// The kinds of node `thunk_for` can answer with a value instead of a
/// suspension. A strict argument shaped like one of these is left to it: those
/// are the cheap cases it exists for, and most of them cost no allocation at
/// all. Everything else -- a call, a block, a member, a conditional -- it
/// always suspends, and a suspension the native is about to force on the very
/// next step is what `fill_native_args` is about.
///
/// The two lists are allowed to drift. A kind named here that `thunk_for`
/// suspends anyway is a thunk built where one need not have been, which is
/// what every call did before this existed; a kind missing from here is an
/// argument the machine evaluates rather than computes, which is what the
/// general path does. Neither is a change in meaning, which is why this can
/// stay a plain list rather than a second copy of the rules.
inline bool thunk_for_may_answer(Op op) {
    switch (op) {
        case Op::ConstInt: case Op::ConstStr: case Op::ConstFloat:
        case Op::ConstChar: case Op::ConstBool: case Op::ConstAtom:
        case Op::Unit: case Op::Builtin: case Op::Local: case Op::Capture:
        case Op::MakeClosure: case Op::MakeThunk:
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Get:
            return true;
        default:
            return false;
    }
}

/// A builtin is an immediate carrying an index into a static table; a host
/// native is an object carrying the same two numbers. Every caller wants one
/// or both of them and none of them care which kind it has.
inline uint32_t native_arity(Value callee) {
    if (is_builtin(callee)) return builtin_def(uint32_t(imm_payload(callee))).arity;
    return static_cast<NativeObj*>(as_obj(callee))->arity;
}
inline uint32_t native_strict_mask(Value callee) {
    if (is_builtin(callee)) return builtin_def(uint32_t(imm_payload(callee))).strict_mask;
    return static_cast<NativeObj*>(as_obj(callee))->strict_mask;
}


/// The depth of nested `force_whnf` loops past which `enter_function` stops
/// offering the compiled tier. See `Process::force_nest`: interpreted code
/// nests forces only as deep as natives on the same C++ chain actually sit one
/// inside another (a `strict!` over nested natives, tens at most), while a
/// *compiled* chain of lazy links spends one C++ frame per link and would blow
/// the machine stack long before `DREAM_MAX_DEPTH`. Small enough to keep the
/// interpreter in charge of anything pathological, large enough that every
/// healthy program nests well under it.
constexpr uint32_t kMaxForceNestForCompiled = 256;
/// Enter a function body, taking the compiled tier when one is available.
void enter_function(Process& p, uint32_t func_index, const FuncRec& f, Value frame) {
    Jit* jit = p.runtime().jit();
    // When a nested force is already deep on the process's machine stack, the
    // compiled tier is not offered at all. A compiled body forces a slot by
    // *calling*, so a chain of lazy links under a force is a C++ frame per link
    // -- one compiled body, one `dream_rt_force`, one `force_whnf` -- and a
    // long enough chain overflows the machine stack before `DREAM_MAX_DEPTH` is
    // anywhere in sight. The interpreter makes those frames on the heap and can
    // reach its limit, so it takes the chain over. Interpreted code never nests
    // this deep (each of its forces returns before the next), which is what
    // lets a small bound here be free for every healthy program. See

    // `Process::force_nest`, `dream/tests/programs/force_chain.dr`, and "Fixed:
    // compiled code forcing a long thunk chain crashed" in docs/notes/vm-performance.md.
    if (jit && p.force_nest <= kMaxForceNestForCompiled) {
        // One inlined, lock-free read of the tier table (see `Jit::tier`), so
        // a process whose functions never grow hot -- most of a compile --
        // pays a load and a branch for the JIT being present.
        CompiledFn fn = jit->tier(func_index);
        if (fn && p.compiled_stack_has_room(DREAM_FRAME_ADDRESS())) {
            int status = 0;
            // Compiled code spends the same budget the interpreter does, so
            // fold what it used into the process's total -- but only what *it*
            // spent. A compiled body re-enters the interpreter for anything it
            // cannot do itself, and those reductions have already counted
            // themselves; charging the whole fall in the budget counted them
            // twice, and twice again for every compiled frame above them. On a
            // self-compile that reported 194M reductions where the same work
            // without the JIT reports 61M, which made `--stats` and every
            // number derived from it meaningless whenever the JIT was on.
            const int64_t before = p.reductions;
            const uint64_t before_total = p.total_reductions;
            const size_t stack_before = p.stack.size();
            // The outermost compiled frame on this stack says where a scan of
            // it ends: at the machine loop that is running this call. Inner
            // ones, entered from forces under it, are inside that already.
            void* const outer_hi = p.jit_stack_hi;
            if (!outer_hi) p.jit_stack_hi = p.loop_stack_hi;
            const bool checking = effect_check_on();
            if (checking) effect_check_enter(p, func_index);
            Value r = fn(&p, frame, &status);
            p.jit_stack_hi = outer_hi;
            const uint32_t effect_retry =
                checking ? effect_check_leave(p, p.park_requested) : 0;
            // Clamped, because a nested force underneath this call may have
            // re-armed the budget (`Process::slice_spent`), which leaves more
            // in it than there was to start with.
            const int64_t fell = before - p.reductions;
            const uint64_t spent = fell > 0 ? uint64_t(fell) : 0;
            const uint64_t nested = p.total_reductions - before_total;
            const uint64_t own = spent > nested ? spent - nested : 0;
            p.total_reductions += own;
            // And attributed, which nothing did before: a function the JIT
            // compiled was invisible to `--profile` however hot it was.
            if (own && p.runtime().profiling_for(p.id())) p.runtime().note_reductions(func_index, own);
            // A force underneath the compiled body reached a blocking native
            // -- a `receive` inside an impure call that a pure function was
            // handed unevaluated, which is how `agent.value_as!` reaches
            // `types.check` -- and `force_whnf` suspended it rather than
            // finishing. Compiled code sees that as a failed force and
            // unwinds with whatever `p.result` held, which is not an error:
            // treating it as a raise killed the process with `()`. The answer
            // is "not yet", exactly as for a native (`apply_native`): leave
            // the suspended work where it is and run the whole call again
            // once it has finished. Compiled bodies are pure, and forcing is
            // memoised, so the second run repeats no effect.
            if (p.park_requested) {
                // The retry goes in below the suspended force, where a frame
                // on the process's stack could be dropped under it, so it
                // takes a copy on the heap.
                push_retry(p, ContKind::EnterRetry, func_index, uint32_t(stack_before),
                           effect_retry, stack_frame(frame) ? heap_frame(p, frame) : frame);
                return;  // the park left the mode at Return
            }
            if (status == JitOk) {
                jit->note_ran(func_index);
                // A loop may answer a parameter as it stands, unforced -- see
                // `Emitter::node` -- and it is forced here, by the machine,
                // where a park suspends the force rather than retrying the
                // call that produced it.
                r = resolve(r);
                if (is_whnf(r)) ret(p, r); else enter(p, r);
                return;
            }
            if (status == JitRaised) { do_raise(p, r); return; }
            if (status == JitTooDeep) {
                // Compiled self recursion ran out of machine stack. The body
                // is arithmetic, so nothing has happened that has to be undone
                // -- the call simply has not been made yet, and the
                // interpreter, whose recursion is heap continuations bounded
                // by `DREAM_MAX_DEPTH`, makes it instead.
                //
                // And it takes the function over for good. A function that
                // recurses deeper than the machine stack allows does it on
                // every call, and each attempt would run a megabyte of frames
                // before finding out; worse, the interpreted recursion would
                // re-enter the compiled tier at every level and do it again
                // per level. One store here is what keeps the fallback linear.
                jit->deoptimize(func_index);
                eval_node(p, f.body, frame);
                return;
            }
            if (status == JitBailed) {
                // A guard did not hold: a slot the compiled body carries as an
                // unboxed double was handed something that is not a float in
                // hand -- on entry, or at the end of a peeled first iteration.
                // Nothing has been written but registers, so the interpreter
                // runs this one call, and the tier stays: one bail is a
                // property of the call. A run of them is a property of the
                // program, and `note_bail` gives the function up after one.
                jit->note_bail(func_index);
                eval_node(p, f.body, frame);
                return;
            }
            jit->note_ran(func_index);
            // Yielded: the compiled loop spent its budget and handed back a
            // frame holding its loop-carried state -- a new one, because the
            // frame it was entered with may already be what suspensions it
            // made read their slots from. Resume the body in it, and let the
            // scheduler preempt at the next safepoint.
            eval_node(p, f.body, r);
            return;
        }
    }
    eval_node(p, f.body, frame);
}
inline size_t max(size_t a, size_t b) { return a > b ? a : b; }

void do_apply(Process& p, Value callee, uint32_t argc) {
    callee = resolve(callee);
    const Image& img = img_of(p);
    const size_t base = max(p.stack.size() - argc, 0);

    if (!is_whnf(callee)) {
        // Reached through a path that did not evaluate the callee node, such
        // as priming a process with a value read out of another heap.
        push_cont(p, ContKind::ApplyTo, argc, 0, 0, UNIT);
        enter(p, callee);
        return;
    }

    if (is_obj(callee, ObjType::Pap)) {
        // Fold the earlier arguments in ahead of the new ones and retry.
        auto* pap = static_cast<PapObj*>(as_obj(callee));
        uint32_t have = pap->nargs;
        p.stack.insert(p.stack.begin() + long(base), pap->args(), pap->args() + have);
        do_apply(p, pap->fn, argc + have);
        return;
    }

    if (is_obj(callee, ObjType::Closure)) {
        auto* cl = static_cast<ClosureObj*>(as_obj(callee));
        const FuncRec& f = img.func(cl->func);
        uint32_t arity = f.arity;

        if (arity == 0) {
            // A zero-arity function -- a thunk object or an impure action.
            // It consumes no arguments; any that were supplied apply to its
            // result, so `main! ()` and `go!` both do the right thing.
            Value fr = p.push_frame(callee, f.slots);
            if (argc > 0) push_cont(p, ContKind::ApplyTo, argc, 0, 0, UNIT);
            enter_function(p, cl->func, f, fr);
            return;
        }
        if (argc < arity) {
            Value pap = p.heap().make_pap(callee, argc);
            auto* po = static_cast<PapObj*>(as_obj(pap));
            for (uint32_t i = 0; i < argc; ++i) po->args()[i] = p.stack[base + i];
            p.stack.resize(base);
            ret(p, pap);
            return;
        }

        Value fr = p.push_frame(callee, f.slots, arity);
        auto* fo = static_cast<FrameObj*>(as_obj(fr));
        for (uint32_t i = 0; i < arity; ++i) fo->slots()[i] = p.stack[base + i];

        // Over-application: the extra arguments apply to whatever comes back.
        uint32_t extra = argc - arity;
        p.stack.erase(p.stack.begin() + long(base), p.stack.begin() + long(base + arity));
        if (extra > 0) push_cont(p, ContKind::ApplyTo, extra, 0, 0, UNIT);
        enter_function(p, cl->func, f, fr);
        return;
    }

    if (is_builtin(callee) || is_obj(callee, ObjType::Native)) {
        const uint32_t arity = native_arity(callee);
        if (arity == NATIVE_VARIADIC) {
            // Takes the whole application. There is nothing to under- or
            // over-apply: `print! a b c` hands the native all three.
            resume_native(p, callee, uint32_t(base), argc, 0);
            return;
        }
        if (argc < arity) {
            Value pap = p.heap().make_pap(callee, argc);
            auto* po = static_cast<PapObj*>(as_obj(pap));
            for (uint32_t i = 0; i < argc; ++i) po->args()[i] = p.stack[base + i];
            p.stack.resize(base);
            ret(p, pap);
            return;
        }
        if (argc > arity) {
            // Apply exactly `arity`, then apply the result to the rest.
            push_cont(p, ContKind::ApplyTo, argc - arity, 0, 0, UNIT);
            // The extra arguments sit above the ones we are about to consume,
            // so move them out of the way first -- into `pins` rather than a
            // C++ vector, because the native is about to run Dream code that
            // may collect and these are live references like any other.
            const size_t spilled = p.pins.size();
            p.pins.insert(p.pins.end(), p.stack.begin() + long(base + arity), p.stack.end());
            p.stack.resize(base + arity);
            resume_native(p, callee, uint32_t(base), arity, 0);
            // Re-push the leftovers underneath the pending ApplyTo.
            p.stack.insert(p.stack.end(), p.pins.begin() + long(spilled), p.pins.end());
            p.pins.resize(spilled);
            return;
        }
        resume_native(p, callee, uint32_t(base), arity, 0);
        return;
    }

    do_raise(p, raise_error(p, well_known(p.runtime()).not_a_function,
                            describe(p, callee) + " is not a function"));
}

/// Force the strict arguments of a native one at a time, then call it.
void resume_native(Process& p, Value callee, uint32_t base, uint32_t argc, uint32_t from) {
    uint32_t mask, arity;
    NativeFn fn;
    if (is_builtin(callee)) {
        const BuiltinDef& bd = builtin_def(uint32_t(imm_payload(callee)));
        mask = bd.strict_mask;
        arity = bd.arity;
        fn = bd.fn;
    } else {
        auto* nat = static_cast<NativeObj*>(as_obj(callee));
        mask = nat->strict_mask;
        arity = nat->arity;
        fn = reinterpret_cast<NativeFn>(nat->fn);
    }
    // A variadic native forces every argument: `strict_mask` is a 32-bit map of
    // argument positions, and an unbounded call has no fixed positions to map.
    // The `i < 32` guard is the same point from the other side -- it keeps a
    // 33rd argument to a fixed-arity native from shifting out of range.
    const bool all_strict = arity == NATIVE_VARIADIC;

    for (uint32_t i = from; i < argc; ++i) {
        Value v = resolve(p.stack[base + i]);
        p.stack[base + i] = v;
        const bool strict = all_strict || (i < 32 && (mask & (1u << i)));
        if (strict && !is_whnf(v)) {
            push_cont(p, ContKind::NativeArg, base, argc, i, callee);
            // `enter` sets up the machine to force it; the NativeArg
            // continuation writes the result back into the argument slot.
            p.mode = Mode::Return;
            p.result = v;
            // Fall through to the forcing path below by re-entering the value.
            Value thunk = v;
            auto* t = static_cast<ThunkObj*>(as_obj(thunk));
            if (as_obj(thunk)->type == ObjType::Blackhole) {
                p.conts.pop_back();
                do_raise(p, raise_error(p, well_known(p.runtime()).loop,
                                        "value depends on itself"));
                return;
            }
            obj_type_store(as_obj(thunk), ObjType::Blackhole);
            push_cont(p, ContKind::UpdateThunk, 0, 0, 0, thunk);
            run_suspension(p, t);
            return;
        }
    }

    // The callee is named again after the call -- `NativeRetry` has to record
    // which native to re-enter -- so it has to survive one. A builtin is an
    // immediate and cannot move; a host native is an object, and although it
    // is reachable from `native_cache` and so can never be *freed* under us,
    // a minor collection would promote it and leave this copy pointing at the
    // forwarded husk. Pinning is a push and a pop, and only on the host path.
    const bool pinned = !is_builtin(callee);
    Pin callee_pin(p, pinned ? callee : UNIT);

    if (!p.effect_attempts.empty()) effect_check_native(p, callee, p.stack.data() + base, argc);
    static const bool timed = timeline_on();
    const int64_t native_start = timed ? timeline_now() : 0;
    NativeResult r = fn(p, callee, p.stack.data() + base, argc);
    if (pinned) callee = callee_pin.get();
    if (timed && timeline_now() - native_start >= 2000) timeline_native(p, native_name(callee), native_start);

    // A nested force inside the native hit a blocking operation and gave up
    // (see `force_whnf`). Whatever the native decided to return is built on a
    // value it never actually got, so the answer is not "this failed" -- it is
    // "not yet". Park and call it again, rather than trusting the result.
    //
    // Catching it here rather than in each native means a native cannot forget
    // to: the ones that force a value deeply are exactly the ones that would
    // not think to check.
    if (p.park_requested && r.outcome != NativeOutcome::Block) {
        r = NativeResult::block();
    }

    switch (r.outcome) {
        case NativeOutcome::Value:
            p.stack.resize(base);
            ret(p, r.value);
            break;
        case NativeOutcome::Raise:
            p.stack.resize(base);
            do_raise(p, r.value);
            break;
        case NativeOutcome::Enter:
            // Something the native found and did not force. Forced here, as the
            // continuation of the call, however deep its evaluation goes.
            p.stack.resize(base);
            enter(p, r.value);
            break;
        case NativeOutcome::Block:
            // The native parked the process. Leave the arguments in place and
            // arrange to call it again when the scheduler wakes us.
            push_retry(p, ContKind::NativeRetry, base, argc, 0, callee);
            p.mode = Mode::Return;
            p.result = UNIT;
            break;
    }
}

/// Evaluate a native's arguments where they stand, suspending only the lazy
/// ones, and then call it.
///
/// A native declares which of its arguments it forces, and `resume_native`
/// forces exactly those -- but by then each one is already a Thunk, because the
/// call site builds a Thunk for every argument it passes. For a strict argument
/// that Thunk is a temporary and nothing else: `resume_native` immediately
/// blackholes it, evaluates it, and overwrites it with an Indirect, so an
/// allocation, two writes and a continuation have been spent carrying `(node,
/// frame)` four steps. It is never shared, either, because the native forces it
/// before it can hand it to anyone. On a self-compile three million of the ten
/// million thunks a compile makes are exactly this.
///
/// So the machine's own continuation carries what the Thunk carried: the call
/// node says which arguments are left and where to find them, and the frame is
/// what they are evaluated in. Nothing is forced here that `resume_native`
/// would not have forced, and in the same order, so an argument that raises
/// raises in the same place and with the same error -- this is a change of
/// representation, not of strictness. A lazy argument is suspended exactly as
/// before.
///
/// The callee is read again on the way back rather than carried across, because
/// a `Cont` has one value field and the frame is in it. Reading it is cheap and
/// it is the same callee: naming one is a read, not an evaluation (see
/// `callee_operand`), and a frame's slots do not change while one of its own
/// sub-expressions is being evaluated -- the same assumption `BinRight` makes
/// when it reads its right operand after forcing its left. It is also what
/// keeps this safe across a collection, which a carried copy would not be.
void fill_native_args(Process& p, uint32_t call_node, Value callee, uint32_t base,
                      uint32_t from, Value frame) {
    const Image& img = img_of(p);
    const Node& n = img.node(call_node);
    const uint32_t* kids = img.kids_at(n.b);
    const uint32_t mask = native_strict_mask(callee);
    // The same rule `resume_native` states, for the same reason: a variadic
    // native has no fixed argument positions for a 32-bit mask to describe, so
    // it forces all of them.
    const bool all_strict = native_arity(callee) == NATIVE_VARIADIC;

    for (uint32_t i = from; i < n.c; ++i) {
        // A lazy argument is suspended exactly as before, and so is a strict
        // one of a shape `thunk_for` answers without allocating -- computing
        // those is cheaper than evaluating them, and taking them away from it
        // would undo its work rather than add to it.
        const bool strict = all_strict || (i < 32 && (mask & (1u << i)));
        if (!strict || thunk_for_may_answer(Op(img.node(kids[i]).op))) {
            p.stack.push_back(thunk_for(p, kids[i], frame));
            continue;
        }
        push_cont(p, ContKind::NativeArgs, call_node, base, i, frame);
        eval_node(p, kids[i], frame);
        return;
    }
    resume_native(p, callee, base, n.c, 0);
}

/// A saturated call to a closure whose function forces some of its parameters
/// before anything else (`Image::strict_args`): those arguments are evaluated
/// here, in the caller, rather than suspended and forced one step into the
/// callee. The rest are suspended as ever.
///
/// It is the closure half of `fill_native_args`, and it is safe for the same
/// reason: nothing of the callee's has run before its strict parameters are
/// forced, and they are forced in parameter order on both paths, so the same
/// argument raises first with the same error. What is saved is a thunk and its
/// update per strict argument per call -- every `!acc` loop paid both.
/// The arguments gather on the value stack, where the collector sees them,
/// and the frame is built once the last is in.
void fill_closure_args(Process& p, uint32_t call_node, uint32_t mask, uint32_t base,
                       uint32_t from, Value frame) {
    const Image& img = img_of(p);
    const Node& n = img.node(call_node);
    const uint32_t* kids = img.kids_at(n.b);
    for (uint32_t i = from; i < n.c; ++i) {
        const bool strict = i < 32 && (mask & (1u << i));
        if (!strict || thunk_for_may_answer(Op(img.node(kids[i]).op))) {
            p.stack.push_back(thunk_for(p, kids[i], frame));
            continue;
        }
        push_cont(p, ContKind::ClosureArgs, call_node, base, i, frame);
        eval_node(p, kids[i], frame);
        return;
    }
    Value callee;
    if (!callee_operand(p, img, n.a, frame, &callee)) {
        p.unreachable("the callee of a closure call stopped being a name");
    }
    Value fn = resolve(callee);
    auto* cl = static_cast<ClosureObj*>(as_obj(fn));
    const FuncRec& f = img.func(cl->func);
    Value fr = p.push_frame(fn, f.slots, f.arity);
    auto* fo = static_cast<FrameObj*>(as_obj(fr));
    const Value* args = &p.stack[base];
    for (uint32_t i = 0; i < n.c; ++i) fo->slots()[i] = args[i];
    p.stack.resize(base);
    enter_function(p, cl->func, f, fr);
}

// ---------------------------------------------------------------------------
// Forcing
// ---------------------------------------------------------------------------

/// Build a closure for `func_index`, capturing from `frame`. Creating a
/// closure allocates but cannot fail at the language level and has no effects,
/// which is what lets `thunk_for` do it eagerly instead of suspending it.
bool build_closure(Process& p, uint32_t func_index, Value frame, Value* out) {
    const Image& img = img_of(p);
    const FuncRec& f = img.func(func_index);
    Value cl = p.heap().make_closure(func_index, f.n_captures);
    auto* co = static_cast<ClosureObj*>(as_obj(cl));
    auto* fo = static_cast<FrameObj*>(as_obj(frame));
    auto* parent = is_obj(fo->closure, ObjType::Closure)
                       ? static_cast<ClosureObj*>(as_obj(fo->closure))
                       : nullptr;
    for (uint32_t i = 0; i < f.n_captures; ++i) {
        uint32_t desc = img.kid(f.captures_off + i);
        uint32_t idx = desc & ~CAP_FROM_CAPTURE;
        if (desc & CAP_FROM_CAPTURE) {
            if (!parent || idx >= parent->ncaps) {
                *out = type_error(p, "malformed capture descriptor");
                return false;
            }
            co->caps()[i] = parent->caps()[idx];
        } else {
            if (idx >= fo->nslots) {
                *out = type_error(p, "malformed capture descriptor");
                return false;
            }
            // Capture the slot's current contents -- normally a thunk. Sharing
            // that thunk is what lets a `let rec` closure see itself: the slot
            // is filled in before anyone forces it.
            co->caps()[i] = fo->slots()[idx];
        }
    }
    *out = cl;
    return true;
}

void enter(Process& p, Value v) {
    v = resolve(v);
    if (!is_ptr(v)) {
        ret(p, v);
        return;
    }
    ObjType t = as_obj(v)->type;
    if (t == ObjType::Thunk) {
        auto* th = static_cast<ThunkObj*>(as_obj(v));
        // Blackhole first: if evaluating this thunk reaches it again, we have
        // a value defined in terms of itself and must say so rather than spin.
        obj_type_store(as_obj(v), ObjType::Blackhole);
        push_cont(p, ContKind::UpdateThunk, 0, 0, 0, v);
        run_suspension(p, th);
        return;
    }
    if (t == ObjType::Blackhole) {
        do_raise(p, raise_error(p, well_known(p.runtime()).loop, "value depends on itself"));
        return;
    }
    ret(p, v);
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

void advance_block(Process& p, uint32_t kids_off, uint32_t count, uint32_t index, Value frame) {
    const Image& img = img_of(p);
    while (index < count) {
        uint32_t stmt = img.kid(kids_off + index);
        const Node& sn = img.node(stmt);
        const bool last = (index + 1 == count);

        if (Op(sn.op) == Op::Bind) {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            const Node& vn = img.node(sn.b);
            // A strict bind -- a `match`'s subject, an effect's result -- is
            // evaluated where it stands and its value put in the slot
            // (`ContKind::BindThen`). It was suspended and the suspension
            // entered at once: a thunk made, blackholed and overwritten for
            // every `match` on a call, which is most of them, and in a parser
            // that matches a reply at every step, a large share of all it
            // allocated. A closure or a suspension still takes the path below,
            // because `let rec go = ..` must find its own slot filled while it
            // is being built.
            if ((sn.flags & F_STRICT) && Op(vn.op) != Op::MakeClosure && Op(vn.op) != Op::MakeThunk) {
                push_cont(p, ContKind::BindThen, kids_off, count, index, frame);
                eval_node(p, sn.b, frame);
                return;
            }
            Value bound;
            if (Op(vn.op) == Op::MakeClosure || Op(vn.op) == Op::MakeThunk) {
                // A local binding may refer to itself -- `let rec go x = .. go ..`
                // -- and a closure built right now would capture the slot while
                // it is still empty. Suspending it instead means the slot
                // already holds this thunk by the time the closure is made, so
                // the capture sees something that resolves to the closure.
                bound = stack_frame(frame) ? flat_thunk(p, sn.b, frame, sn.a)
                                           : p.heap().make_thunk(sn.b, frame);
            } else {
                bound = thunk_for(p, sn.b, frame);
            }
            // A heap frame may have outlived its call -- a global's, held by
            // its value -- so the bind can point an old object at a young one.
            bind_slot(p, fo, sn.a, bound);
            // A binding is lazy unless the compiler marked it strict, which it
            // does exactly when the bound expression has effects that must
            // happen where they were written.
            if (sn.flags & F_STRICT) {
                push_cont(p, ContKind::BlockNext, kids_off, count, index + 1, frame);
                enter(p, bound);
                return;
            }
            ++index;
            continue;
        }

        if (last) {
            eval_node(p, stmt, frame);
            return;
        }
        // A non-final statement exists only for its effects. The compiler marks
        // the ones that have any; the rest are dead and forcing them could
        // raise an error the program never asked for.
        if (sn.flags & F_STRICT) {
            push_cont(p, ContKind::BlockNext, kids_off, count, index + 1, frame);
            eval_node(p, stmt, frame);
            return;
        }
        ++index;
    }
    ret(p, UNIT);
}

void advance_map(Process& p, uint32_t kids_off, uint32_t pairs, uint32_t index, Value frame) {
    if (index >= pairs) {
        // Follow any indirection an insert left behind before handing the map
        // out; callers expect a value in normal form.
        Value map = resolve(p.stack.back());
        p.stack.pop_back();
        ret(p, map);
        return;
    }
    push_cont(p, ContKind::MapEntry, kids_off, pairs, index, frame);
    eval_node(p, img_of(p).kid(kids_off + index * 2), frame);
}

// ---------------------------------------------------------------------------
// Arithmetic and comparison
// ---------------------------------------------------------------------------

bool arith(Process& p, Op op, Value a, Value b, Value* out);
bool compare(Process& p, Op op, Value a, Value b, Value* out);

/// Is this the head of a list -- a cell, or the empty list?
inline bool is_listish(Value v) { return is_nil(v) || is_obj(v, ObjType::Cons); }

/// `xs + ys`. The spine of `xs` is copied and its end pointed at `ys`;
/// elements are never forced, so concatenating lists of unevaluated work stays
/// lazy in the elements.
bool concat_lists(Process& p, Value a, Value b, Value* out) {
    if (is_nil(a)) {
        *out = b;
        return true;
    }
    // Forcing the spine can collect, so the working set lives on the process
    // stack rather than in C++ locals: slot 0 holds the tail to append, slot 1
    // is the cursor, and the heads pile up above them.
    //
    // It does *not* vouch (`VouchesForGc`), though everything below it is
    // written for one. This is not reached through `resume_native` the way a
    // native is -- it is the `+` operator, reached from `arith`, which
    // `finish_binop` calls while holding both operands in C++ locals and which
    // JIT-compiled code calls while holding its entire frame in machine
    // registers. Neither of those survives a collection, and the frame that
    // cannot survive one is the frame that decides. Vouching here made
    // `effects_once` hang about one run in twelve.
    const size_t base = p.stack.size();
    p.stack.push_back(b);
    p.stack.push_back(a);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base + 1], &w)) {
            *out = p.result;
            // Not when the force was only suspended: its stack is above ours.
            if (!p.force_blocked) p.stack.resize(base);
            return false;
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            *out = type_error(p, "the left side of a list `+` is not a proper list");
            p.stack.resize(base);
            return false;
        }
        auto* c = static_cast<ConsObj*>(as_obj(w));
        Value head = c->head;
        p.stack[base + 1] = c->tail;
        p.stack.push_back(head);
    }

    Value result = p.stack[base];
    for (size_t i = p.stack.size(); i-- > base + 2;) {
        result = p.heap().make_cons(p.stack[i], result);
    }
    p.stack.resize(base);
    *out = result;
    return true;
}

/// The bitwise operators, past the fixnum fast paths: an operand that is a
/// bignum, a shift that leaves the fixnum range, or something that is not an
/// integer at all.
bool bitwise_arith(Process& p, Op op, Value a, Value b, Value* out) {
    const char* name = op == Op::BitAnd ? "&" : op == Op::BitOr ? "|" : op == Op::BitXor ? "^"
                     : op == Op::Shl ? "<<" : ">>";
    if (!bigint::is_integer(a) || !bigint::is_integer(b)) {
        *out = type_error(p, std::string("`") + name + "` needs integers, got " + describe(p, a) + " and " +
                                 describe(p, b));
        return false;
    }
    Heap& h = p.heap();
    switch (op) {
        case Op::BitAnd: *out = bigint::bit_and(h, a, b); return true;
        case Op::BitOr: *out = bigint::bit_or(h, a, b); return true;
        case Op::BitXor: *out = bigint::bit_xor(h, a, b); return true;
        default: break;
    }
    if (bigint::compare(b, make_fixnum(0)) < 0) {
        *out = raise_error(p, well_known(p.runtime()).out_of_bounds, std::string("a negative shift count for `") + name + "`");
        return false;
    }
    if (op == Op::Shr) {
        // A count past every bit leaves only the sign: 0 or -1.
        int64_t k = 0;
        if (!bigint::to_int64(b, &k)) k = INT64_MAX;
        *out = bigint::shift_right(h, a, uint64_t(k));
        return true;
    }
    // A left shift is as large as its count says. Past 2^32 bits -- half a
    // gigabyte for the one number -- it is a mistake rather than a value, and
    // is refused before it is allocated.
    int64_t k = 0;
    if (a == make_fixnum(0)) { *out = make_fixnum(0); return true; }
    if (!bigint::to_int64(b, &k) || k > (int64_t(1) << 32)) {
        *out = raise_error(p, well_known(p.runtime()).out_of_memory,
                           "a left shift by " + bigint::to_string(b) + " bits is too large to make");
        return false;
    }
    *out = bigint::shift_left(h, a, uint64_t(k));
    return true;
}

bool arith(Process& p, Op op, Value a, Value b, Value* out) {
    const auto& wk = well_known(p.runtime());

    if (op == Op::BitNot) {
        if (!bigint::is_integer(a)) {
            *out = type_error(p, "`~` needs an integer, got " + describe(p, a));
            return false;
        }
        *out = bigint::bit_not(p.heap(), a);
        return true;
    }

    if (op == Op::BitAnd || op == Op::BitOr || op == Op::BitXor || op == Op::Shl || op == Op::Shr) {
        return bitwise_arith(p, op, a, b, out);
    }

    // Elementwise, and the reason this is a VM operation at all: `a * b` on
    // two tensors is one pass over packed numbers. Only reached once the
    // fixnum fast paths have declined, so asking costs ordinary arithmetic
    // nothing.
    if (is_obj(a, ObjType::Tensor) || is_obj(b, ObjType::Tensor)) {
        return tensor_arith(p, op, a, b, out);
    }

    if (op == Op::Add && is_listish(a) && is_listish(b)) {
        return concat_lists(p, a, b, out);
    }

    if (op == Op::Add && is_obj(a, ObjType::Str) && is_obj(b, ObjType::Str)) {
        auto* x = static_cast<StrObj*>(as_obj(a));
        auto* y = static_cast<StrObj*>(as_obj(b));
        Value s = p.heap().make_string(nullptr, x->len + y->len);
        auto* z = static_cast<StrObj*>(as_obj(s));
        std::memcpy(z->data(), static_cast<StrObj*>(as_obj(a))->data(), x->len);
        std::memcpy(z->data() + x->len, static_cast<StrObj*>(as_obj(b))->data(), y->len);
        *out = s;
        return true;
    }

    // A big string is deliberately not concatenable. Joining one would have to
    // produce a `StrObj`, which copies the bytes into the heap and caps them at
    // 4 GiB -- undoing both halves of what the payload is for. The error names
    // the two ways out rather than leaving the caller to guess.
    if (is_obj(a, ObjType::BigStr) || is_obj(b, ObjType::BigStr)) {
        *out = type_error(p, std::string("cannot apply `") + op_name(op) +
                                 "` to a big string: it is a view into the image, not heap "
                                 "memory. Take a `str.slice` of it, or hand the whole of it "
                                 "to `io.write!`.");
        return false;
    }

    if (!is_number(a) || !is_number(b)) {
        *out = type_error(p, std::string("cannot apply `") + op_name(op) + "` to " +
                                 describe(p, a) + " and " + describe(p, b));
        return false;
    }

    if (bigint::is_integer(a) && bigint::is_integer(b)) {
        if (is_fixnum(a) && is_fixnum(b)) {
            int64_t x = fixnum_value(a), y = fixnum_value(b), r = 0;
            bool overflow = false;
            switch (op) {
                case Op::Add: overflow = add_overflow(x, y, &r); break;
                case Op::Sub: overflow = sub_overflow(x, y, &r); break;
                case Op::Mul: overflow = mul_overflow(x, y, &r); break;
                case Op::Div:
                    if (y == 0) {
                        *out = raise_error(p, wk.divide_by_zero, "division by zero");
                        return false;
                    }
                    r = x / y;
                    break;
                case Op::Mod:
                    if (y == 0) {
                        *out = raise_error(p, wk.divide_by_zero, "remainder by zero");
                        return false;
                    }
                    r = x % y;
                    break;
                default: break;
            }
            if (!overflow && fixnum_fits(r)) {
                *out = make_fixnum(r);
                return true;
            }
        }
        // Past 63 bits, or a bignum already: exact, and a fixnum again as soon
        // as the answer fits. This is where an overflow used to become a
        // float, and the only way into bigint.cpp from arithmetic.
        Heap& h = p.heap();
        switch (op) {
            case Op::Add: *out = bigint::add(h, a, b); return true;
            case Op::Sub: *out = bigint::sub(h, a, b); return true;
            case Op::Mul: *out = bigint::mul(h, a, b); return true;
            case Op::Div:
            case Op::Mod:
                if (b == make_fixnum(0)) {
                    *out = raise_error(p, wk.divide_by_zero,
                                       op == Op::Div ? "division by zero" : "remainder by zero");
                    return false;
                }
                *out = op == Op::Div ? bigint::quot(h, a, b) : bigint::rem(h, a, b);
                return true;
            default: break;
        }
    }

    double x = to_double(a), y = to_double(b), r = 0;
    switch (op) {
        case Op::Add: r = x + y; break;
        case Op::Sub: r = x - y; break;
        case Op::Mul: r = x * y; break;
        case Op::Div:
            if (y == 0.0 && is_fixnum(b)) {
                *out = raise_error(p, wk.divide_by_zero, "division by zero");
                return false;
            }
            r = x / y;
            break;
        case Op::Mod: r = std::fmod(x, y); break;
        default: break;
    }
    *out = p.heap().make_float(r);
    return true;
}

bool compare(Process& p, Op op, Value a, Value b, Value* out) {
    if (op == Op::Eq || op == Op::Ne) {
        bool raised = false;
        bool eq = values_equal(p, a, b, &raised, 0);
        if (raised) {
            *out = p.result;
            return false;
        }
        *out = make_bool(op == Op::Eq ? eq : !eq);
        return true;
    }

    int cmp;
    if (is_number(a) && is_number(b)) {
        if (bigint::is_integer(a) && bigint::is_integer(b)) {
            cmp = bigint::compare(a, b);
        } else {
            double x = to_double(a), y = to_double(b);
            cmp = x < y ? -1 : (x > y ? 1 : 0);
        }
    } else if (is_stringish(a) && is_stringish(b)) {
        Bytes x, y;
        string_bytes(a, &x);
        string_bytes(b, &y);
        cmp = bytes_compare(x, y);
    } else if (is_char(a) && is_char(b)) {
        uint64_t x = imm_payload(a), y = imm_payload(b);
        cmp = x < y ? -1 : (x > y ? 1 : 0);
    } else {
        *out = type_error(p, std::string("cannot order ") + describe(p, a) + " against " +
                                 describe(p, b));
        return false;
    }

    bool r = false;
    switch (op) {
        case Op::Lt: r = cmp < 0; break;
        case Op::Le: r = cmp <= 0; break;
        case Op::Gt: r = cmp > 0; break;
        case Op::Ge: r = cmp >= 0; break;
        default: break;
    }
    *out = make_bool(r);
    return true;
}

// ---------------------------------------------------------------------------
// Module member lookup
// ---------------------------------------------------------------------------

/// `obj.member`, where `obj` is a host module.
///
/// Host-module calls share an import cache. The
/// module is found through a cache indexed by the import record, and which
/// member was wanted is remembered against the node that asked -- packed as
/// the import index and the member's position in one word, so a node that
/// somehow meets a different module falls back to the search rather than
/// reading the wrong member out of the right one.
/// The slow half: find the member, number it, and build its function value.
/// Reached once per call site, and then never again.
#if defined(_MSC_VER)
__declspec(noinline)
#else
[[gnu::noinline]]
#endif
bool member_value_slow(Process& p, uint32_t imp, uint32_t name_index,
                       uint32_t node_index, Value* out) {
    const Image& img = img_of(p);
    StringRef member = img.str(name_index);
    const ModuleDef* def = p.runtime().module_for_import(imp);
    if (!def) {
        *out = raise_error(p, well_known(p.runtime()).no_such_member,
                           "module `" + img.str(img.import(imp).path).str() +
                               "` is not provided by this runtime");
        return false;
    }
    uint32_t at = UINT32_MAX;
    for (uint32_t i = 0; i < def->members.size(); ++i) {
        if (member.equals(def->members[i].name)) {
            at = i;
            break;
        }
    }
    // Match Dream's name resolution: an explicit `f!` wins, otherwise the
    // same native implementation can be called impurely as `f!`.
    if (at == UINT32_MAX && member.len && member.data[member.len - 1] == '!') {
        const StringRef pure{member.data, member.len - 1};
        for (uint32_t i = 0; i < def->members.size(); ++i) {
            if (pure.equals(def->members[i].name)) {
                at = i;
                break;
            }
        }
    }
    if (at == UINT32_MAX) {
        *out = raise_error(p, well_known(p.runtime()).no_such_member,
                           "module `" + img.str(img.import(imp).path).str() +
                               "` has no member `" + member.str() + "`");
        return false;
    }

    // The value handed back is decided by which member this is and nothing
    // else, so this process builds it once. What that replaces is two
    // allocations -- a copy of the name and a `NativeObj` -- at every call of
    // every `std` member, which in a program of any size is most of what the
    // allocator does.
    const uint32_t id = def->member_base + at;
    if (p.native_cache.empty()) {
        p.native_cache.assign(p.runtime().native_member_count(), NIL_SLOT);
    }
    // The site remembers the *cache slot*, not the member's position in its
    // module: that is what lets the fast path answer without asking which
    // module this import names. The import index rides along, so a node that
    // somehow meets a different module falls back here rather than reading
    // the wrong member out of the right one.
    p.runtime().set_field_cache(node_index, (uint64_t(imp + 1) << 32) | id);

    if (id < p.native_cache.size() && p.native_cache[id] != NIL_SLOT) {
        *out = p.native_cache[id];
        return true;
    }
    const NativeDef& m = def->members[at];
    // Both spellings share this cache entry. Keep the implementation's name,
    // so an effectful call does not change later reads of its pure spelling.
    Value name = p.heap().make_string(m.name, uint32_t(std::strlen(m.name)));
    Value fn = make_native(p, m.fn, name, m.arity, m.strict_mask, m.user);
    if (id < p.native_cache.size()) p.native_cache[id] = fn;
    *out = fn;
    return true;
}

/// `obj.member`, where `obj` is a host module.
///
/// Host-module calls use a cache. Warm,
/// it is three loads and a compare: the module's import index out of the
/// object, the cache slot out of the site that asked, and the function value
/// out of this process's table. No name is copied, no module is searched, and
/// nothing is allocated.
inline bool member_value(Process& p, Value obj, uint32_t name_index, uint32_t node_index,
                         Value* out) {
    if (!is_obj(obj, ObjType::Module)) {
        *out = type_error(p, describe(p, obj) + " has no member `" +
                                 img_of(p).str(name_index).str() + "`");
        return false;
    }
    const uint32_t imp = static_cast<ModuleObj*>(as_obj(obj))->import_index;
    const uint64_t cached = p.runtime().field_cache(node_index);
    if (uint32_t(cached >> 32) == imp + 1) {
        const uint32_t id = uint32_t(cached);
        if (id < p.native_cache.size()) {
            Value hit = p.native_cache[id];
            if (hit != NIL_SLOT) {
                *out = hit;
                return true;
            }
        }
    }
    return member_value_slow(p, imp, name_index, node_index, out);
}

void resolve_field(Process& p, Value obj, uint32_t name_index, uint32_t node_index) {
    Value v;
    if (member_value(p, obj, name_index, node_index, &v)) ret(p, v);
    else do_raise(p, v);
}

// ---------------------------------------------------------------------------
// Operands that need no evaluation
//
// The machine's general shape is "push a continuation, evaluate the part,
// come back" -- which is what makes it interruptible and what keeps its depth
// on the heap. But a great many operands are not expressions at all. A global,
// a builtin, a parameter, a member of an imported module: each of these is a
// *read*, and running it through the machine costs a continuation, a reduction
// and a return trip to learn what a load would have said. These two answer
// "can I just read it?", and the sites that ask fall back to the general path
// whenever the answer is no.
// ---------------------------------------------------------------------------

/// The module a node names, when it names one already in normal form.
bool module_operand(Process& p, const Image& img, uint32_t node, Value frame, Value* out) {
    const Node& n = img.node(node);
    if (Op(n.op) == Op::Global) {
        const GlobalRec& g = img.global(n.a);
        if (g.kind != GLOBAL_MODULE) return false;
        *out = global_value(p, n.a);
        return true;
    }
    // A module can also sit in a slot -- `import a.{b}` binds one, and so does
    // passing a module to a function.
    Value v;
    if (Op(n.op) == Op::Local) {
        v = static_cast<FrameObj*>(as_obj(frame))->slots()[n.a];
    } else if (Op(n.op) == Op::Capture) {
        auto* fo = static_cast<FrameObj*>(as_obj(frame));
        v = static_cast<ClosureObj*>(as_obj(fo->closure))->caps()[n.a];
    } else {
        return false;
    }
    if (v == NIL_SLOT) return false;
    v = resolve(v);
    if (!is_obj(v, ObjType::Module)) return false;
    *out = v;
    return true;
}

/// The value of a node that already is one: a constant, or a binding whose
/// contents are in weak head normal form.
///
/// This is what lets an operator finish where it starts. `acc + x` over two
/// bound locals is the whole of a fold's inner loop, and running each side
/// through the machine costs a continuation, an Eval and a Return to discover
/// that the slot held a number all along -- five steps to add two integers.
/// A local still holding a thunk answers false, and the general path forces it
/// exactly as before, so nothing is evaluated here that was not evaluated
/// there.
bool operand_value(Process& p, const Image& img, uint32_t node, Value frame, Value* out) {
    const Node& n = img.node(node);
    switch (Op(n.op)) {
        case Op::ConstInt: *out = make_integer(p, img.integer(n.a)); return true;
        case Op::ConstFloat: *out = literal_float(p, n.a); return true;
        case Op::ConstStr: *out = literal_string(p, n.a); return true;
        case Op::ConstChar: *out = make_char(uint32_t(n.a)); return true;
        case Op::ConstBool: *out = make_bool(n.a != 0); return true;
        case Op::ConstAtom: *out = make_atom(p.runtime().image_atom(n.a)); return true;
        case Op::Unit: *out = UNIT; return true;
        case Op::Builtin: *out = make_builtin(n.a); return true;
        case Op::Local: {
            Value v = static_cast<FrameObj*>(as_obj(frame))->slots()[n.a];
            // An unbound slot is an error the general path reports; saying so
            // in two places would be saying it differently in two places.
            if (v == NIL_SLOT) return false;
            v = resolve(v);
            if (!is_whnf(v)) return false;
            *out = v;
            return true;
        }
        case Op::Capture: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            Value v = resolve(static_cast<ClosureObj*>(as_obj(fo->closure))->caps()[n.a]);
            if (!is_whnf(v)) return false;
            *out = v;
            return true;
        }
        default:
            return false;
    }
}

/// The callee a node names, when naming it needs no evaluation. `*out` is
/// `NIL_SLOT` for a local that has not been bound yet, which is the caller's
/// error to report.
bool callee_operand(Process& p, const Image& img, uint32_t node, Value frame, Value* out) {
    const Node& n = img.node(node);
    switch (Op(n.op)) {
        case Op::Local:
            *out = static_cast<FrameObj*>(as_obj(frame))->slots()[n.a];
            return true;
        case Op::Capture: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            *out = static_cast<ClosureObj*>(as_obj(fo->closure))->caps()[n.a];
            return true;
        }
        case Op::Builtin:
            *out = make_builtin(n.a);
            return true;
        case Op::Global: {
            const GlobalRec& g = img.global(n.a);
            // A parameterless global is a memoized value or an action to
            // perform, and both have rules of their own; only a function with
            // parameters is a closure that is simply there.
            if (g.kind != GLOBAL_FUNCTION || img.func(g.target).arity == 0) return false;
            *out = global_value(p, n.a);
            return true;
        }
        case Op::Field: {
            Value obj;
            if (!module_operand(p, img, n.a, frame, &obj)) return false;
            // A member that cannot be resolved is left to the general path,
            // which raises the same error where the machine can see it.
            return member_value(p, obj, n.b, node, out);
        }
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// One step
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Containers: `get` and `set`
//
// One pair of operations over every container a program builds: a map by key,
// an array or a list by position. They are opcodes rather than natives because
// of what a native must do with the element it hands back. A native's result
// has to be a value, and an element is usually still a thunk, so a native
// forces it underneath itself -- in a nested machine loop, on the C++ stack. A
// map field updated twenty thousand times and never read is twenty thousand of
// those nested inside each other the first time it is read, which was deep
// enough to run the 8 MiB stack of a Nix builder out while the compiler compiled
// itself. Here the element is entered like any other value, so its depth is
// continuations: heap, with a limit that raises rather than a guard page.
// ---------------------------------------------------------------------------

/// Nothing at the key. When the `get` has a fallback, evaluate it and say so.
bool get_fallback(Process& p, uint32_t at, Value frame) {
    const Node& n = img_of(p).node(at);
    if (n.c == NO_NODE) return false;
    eval_node(p, n.c, frame);
    return true;
}

inline uint32_t out_of_bounds_atom(Process& p) { return well_known(p.runtime()).out_of_bounds; }

/// Step `remaining` more cells down a list and answer the head found there. A
/// tail that is not a value yet is entered with a continuation to come back to,
/// so a long lazy list is walked by the machine rather than by recursion.
void list_get(Process& p, uint32_t at, Value cur, uint32_t remaining, uint32_t index, Value frame) {
    for (;;) {
        cur = resolve(cur);
        if (!is_obj(cur, ObjType::Cons)) {
            if (get_fallback(p, at, frame)) return;
            do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                    "index " + std::to_string(index) +
                                        " is past the end of a list of " +
                                        std::to_string(index - remaining)));
            return;
        }
        auto* cell = static_cast<ConsObj*>(as_obj(cur));
        if (remaining == 0) {
            enter(p, cell->head);
            return;
        }
        --remaining;
        Value tail = resolve(cell->tail);
        if (!is_whnf(tail)) {
            push_cont(p, ContKind::GetWalk, remaining, at, index, frame);
            enter(p, tail);
            return;
        }
        cur = tail;
    }
}

/// Step `remaining` more cells down a list, keeping every cell passed on the
/// value stack, and answer the list with the element at `index` replaced by
/// what `value_node` suspends to. The cells in front of it are new; everything
/// behind it is shared, and so is every element but the one replaced.
void list_set(Process& p, uint32_t value_node, Value cur, uint32_t remaining, uint32_t index,
              Value frame) {
    for (;;) {
        cur = resolve(cur);
        const uint32_t walked = index - remaining;
        if (!is_obj(cur, ObjType::Cons)) {
            p.stack.resize(p.stack.size() - walked);
            do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                    "index " + std::to_string(index) +
                                        " is past the end of a list of " + std::to_string(walked)));
            return;
        }
        if (remaining == 0) {
            Value value = thunk_for(p, value_node, frame);
            Value out = p.heap().make_cons(value, static_cast<ConsObj*>(as_obj(cur))->tail);
            for (uint32_t i = 0; i < index; ++i) {
                Value head = static_cast<ConsObj*>(as_obj(p.stack.back()))->head;
                out = p.heap().make_cons(head, out);
                p.stack.pop_back();
            }
            ret(p, out);
            return;
        }
        p.stack.push_back(cur);
        --remaining;
        Value tail = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
        if (!is_whnf(tail)) {
            push_cont(p, ContKind::SetWalk, remaining, value_node, index, frame);
            enter(p, tail);
            return;
        }
        cur = tail;
    }
}

inline bool is_sequence(Value v) {
    return is_obj(v, ObjType::Array) || is_nil(v) || is_obj(v, ObjType::Cons);
}

/// `c.[k]` and `c.[k else d]`, with the container and the key both in hand.
void container_get(Process& p, uint32_t at, Value container, Value key, Value frame) {
    container = resolve(container);
    key = resolve(key);
    if (is_obj(container, ObjType::Map)) {
        Value found;
        if (map_lookup(p, container, key, &found)) {
            enter(p, found);
            return;
        }
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, well_known(p.runtime()).no_such_key,
                                "the map has no key " + describe(p, key)));
        return;
    }
    if (is_obj(container, ObjType::Tensor) && is_fixnum(key)) {
        Value got;
        switch (tensor_index(p, container, fixnum_value(key), &got)) {
            case 1: ret(p, got); return;
            case 0: do_raise(p, got); return;
            default: break;
        }
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "index " + std::to_string(fixnum_value(key)) +
                                    " is outside a tensor of length " +
                                    std::to_string(tensor_len(container))));
        return;
    }
    if (is_obj(container, ObjType::Record)) {
        if (!is_fixnum(key)) {
            do_raise(p, type_error(p, "a field of a record is read by position, not " +
                                          describe(p, key)));
            return;
        }
        auto* r = static_cast<RecordObj*>(as_obj(container));
        const int64_t k = fixnum_value(key);
        if (k >= 0 && k < int64_t(r->len)) {
            enter(p, r->items()[k]);
            return;
        }
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "field " + std::to_string(k) + " is outside " +
                                    describe(p, container)));
        return;
    }
    if (is_obj(container, ObjType::Trunk)) {
        if (!is_fixnum(key)) {
            do_raise(p, type_error(p, "a position is an integer, not " + describe(p, key)));
            return;
        }
        const int64_t k = fixnum_value(key);
        if (k >= 0 && k < int64_t(trunk_length(container))) {
            enter(p, trunk_element(container, uint32_t(k)));
            return;
        }
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "index " + std::to_string(k) + " is outside a trunk of " +
                                    std::to_string(trunk_length(container))));
        return;
    }
    if (!is_sequence(container)) {
        do_raise(p, type_error(p, "`.[ ]` reads a map, an array, a list or a trunk, not " +
                                      describe(p, container)));
        return;
    }
    if (!is_fixnum(key)) {
        do_raise(p, type_error(p, "a position is an integer, not " + describe(p, key)));
        return;
    }
    const int64_t k = fixnum_value(key);
    if (is_obj(container, ObjType::Array)) {
        auto* a = static_cast<ArrayObj*>(as_obj(container));
        if (k >= 0 && k < int64_t(a->len)) {
            enter(p, a->items()[k]);
            return;
        }
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "index " + std::to_string(k) + " is outside an array of " +
                                    std::to_string(a->len)));
        return;
    }
    if (k < 0 || k > int64_t(0xFFFFFFFFu)) {
        if (get_fallback(p, at, frame)) return;
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "index " + std::to_string(k) + " is not a position in a list"));
        return;
    }
    list_get(p, at, container, uint32_t(k), uint32_t(k), frame);
}

/// `c.[k => v]`, with the container and the key in hand. The value is stored
/// unforced, as a map literal's values are.
/// `c.[k => v]`, with the container and the key in hand and `value_node` the
/// node `v` is: a `set`'s `c`, or one of a `set_many`'s values.
void container_set(Process& p, uint32_t value_node, Value container, Value key, Value frame) {
    container = resolve(container);
    key = resolve(key);
    if (is_obj(container, ObjType::Map)) {
        Value value = thunk_for(p, value_node, frame);
        ret(p, map_insert(p, container, key, value));
        return;
    }
    if (is_obj(container, ObjType::Record)) {
        if (!is_fixnum(key)) {
            do_raise(p, type_error(p, "a field of a record is changed by position, not " +
                                          describe(p, key)));
            return;
        }
        auto* src = static_cast<RecordObj*>(as_obj(container));
        const uint32_t n = src->len;
        const int64_t k = fixnum_value(key);
        if (k < 0 || k >= int64_t(n)) {
            do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                    "field " + std::to_string(k) + " is outside " +
                                        describe(p, container)));
            return;
        }
        // A copy, with the id it had: changing a field of a `Point` answers
        // a `Point`.
        Value value = thunk_for(p, value_node, frame);
        Value out = p.heap().make_record(src->tag, n);
        src = static_cast<RecordObj*>(as_obj(container));
        auto* dst = static_cast<RecordObj*>(as_obj(out));
        for (uint32_t i = 0; i < n; ++i) {
            Value item = uint32_t(k) == i ? value : src->items()[i];
            dst->items()[i] = item;
            p.heap().remember_if_old(dst, item);
        }
        ret(p, out);
        return;
    }
    if (!is_sequence(container)) {
        do_raise(p, type_error(p, "`.[ => ]` changes a map, an array or a list, not " +
                                      describe(p, container)));
        return;
    }
    if (!is_fixnum(key)) {
        do_raise(p, type_error(p, "a position is an integer, not " + describe(p, key)));
        return;
    }
    const int64_t k = fixnum_value(key);
    if (is_obj(container, ObjType::Array)) {
        const uint32_t n = static_cast<ArrayObj*>(as_obj(container))->len;
        if (k < 0 || k >= int64_t(n)) {
            do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                    "index " + std::to_string(k) + " is outside an array of " +
                                        std::to_string(n)));
            return;
        }
        // A copy: the array given is a value, and someone else may hold it.
        // A large array is allocated straight into old space, so every store
        // into it goes through the barrier.
        Value value = thunk_for(p, value_node, frame);
        Value out = p.heap().make_array(n);
        auto* src = static_cast<ArrayObj*>(as_obj(container));
        auto* dst = static_cast<ArrayObj*>(as_obj(out));
        for (uint32_t i = 0; i < n; ++i) {
            Value item = uint32_t(k) == i ? value : src->items()[i];
            dst->items()[i] = item;
            p.heap().remember_if_old(dst, item);
        }
        ret(p, out);
        return;
    }
    if (k < 0 || k > int64_t(0xFFFFFFFFu)) {
        do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                "index " + std::to_string(k) + " is not a position in a list"));
        return;
    }
    list_set(p, value_node, container, uint32_t(k), uint32_t(k), frame);
}

/// The key of a `set_many`'s pair `i`: a constant, which the loader checked
/// (`SetMany` in image.cpp), so it is read rather than evaluated.
Value set_many_key(Process& p, const uint32_t* kids, uint32_t i, Value frame) {
    Value key = UNIT;
    operand_value(p, img_of(p), kids[2 * i], frame, &key);
    return resolve(key);
}

/// A `set_many`'s pairs from `from` on, applied to `container` in order.
///
/// A record or an array is copied once and every pair written into the copy,
/// which is the point of the opcode: a chain of `set`s copied all n fields per
/// key. The keys are all checked before anything is built, against the same
/// messages `container_set` raises, so a bad key fails as the chain would have
/// failed at it -- a pair before it changing nothing anyone could see. A map
/// takes its pairs as one batch of inserts. Anything else -- a list, or a value
/// `set` refuses -- goes through `container_set` a pair at a time, since a list
/// is walked with continuations of its own, and the rest of the pairs wait in
/// a `SetManyStep`.
void set_many(Process& p, uint32_t at, Value container, uint32_t from, Value frame) {
    const Image& img = img_of(p);
    const Node& n = img.node(at);
    const uint32_t* kids = img.kids_at(n.b);
    container = resolve(container);
    const bool record = is_obj(container, ObjType::Record);
    if (record || is_obj(container, ObjType::Array)) {
        const uint32_t len = record ? static_cast<RecordObj*>(as_obj(container))->len
                                    : static_cast<ArrayObj*>(as_obj(container))->len;
        for (uint32_t i = from; i < n.c; ++i) {
            Value key = set_many_key(p, kids, i, frame);
            if (!is_fixnum(key)) {
                do_raise(p, type_error(p, record ? "a field of a record is changed by position, not " +
                                                       describe(p, key)
                                                 : "a position is an integer, not " + describe(p, key)));
                return;
            }
            const int64_t k = fixnum_value(key);
            if (k < 0 || k >= int64_t(len)) {
                do_raise(p, raise_error(p, out_of_bounds_atom(p),
                                        record ? "field " + std::to_string(k) + " is outside " +
                                                     describe(p, container)
                                               : "index " + std::to_string(k) +
                                                     " is outside an array of " + std::to_string(len)));
                return;
            }
        }
        // Nothing here collects (allocation never does), so the source's
        // items can be read after the copy is made.
        Value out = record ? p.heap().make_record(static_cast<RecordObj*>(as_obj(container))->tag, len)
                           : p.heap().make_array(len);
        Value* dst = record ? static_cast<RecordObj*>(as_obj(out))->items()
                            : static_cast<ArrayObj*>(as_obj(out))->items();
        const Value* src = record ? static_cast<RecordObj*>(as_obj(container))->items()
                                  : static_cast<ArrayObj*>(as_obj(container))->items();
        Obj* holder = as_obj(out);
        for (uint32_t i = 0; i < len; ++i) {
            dst[i] = src[i];
            p.heap().remember_if_old(holder, src[i]);
        }
        for (uint32_t i = from; i < n.c; ++i) {
            const int64_t k = fixnum_value(set_many_key(p, kids, i, frame));
            Value value = thunk_for(p, kids[2 * i + 1], frame);
            dst[k] = value;
            p.heap().remember_if_old(holder, value);
        }
        ret(p, out);
        return;
    }
    if (is_obj(container, ObjType::Map)) {
        // A `mapping`'s setters: one batch, which copies each branch on the
        // keys' paths once (`map_insert_many`).
        // The pairs on the stack while there are few, which is always for a
        // pipeline of setters; a heap buffer costs more than the batch saves.
        const uint32_t count = n.c - from;
        Value small[32];
        std::vector<Value> large;
        Value* pairs = small;
        if (count > 16) {
            large.resize(2 * size_t(count));
            pairs = large.data();
        }
        for (uint32_t i = 0; i < count; ++i) {
            pairs[2 * i] = set_many_key(p, kids, from + i, frame);
            pairs[2 * i + 1] = thunk_for(p, kids[2 * (from + i) + 1], frame);
        }
        ret(p, map_insert_many(p, container, count, pairs));
        return;
    }
    if (from + 1 < n.c) push_cont(p, ContKind::SetManyStep, 0, at, from + 1, frame);
    container_set(p, kids[2 * from + 1], container, set_many_key(p, kids, from, frame), frame);
}

// ---------------------------------------------------------------------------
// Finishing an operator
//
// Each of these is the tail of a continuation, factored out so that the same
// code runs whether the operands arrived through the machine or were read
// straight out of the frame (see `operand_value`).
// ---------------------------------------------------------------------------

void finish_binary(Process& p, Op op, Value lhs, Value rhs) {
    // Two fixnums is what an operator almost always has, and answering it here
    // skips the whole of `arith` and `compare`: the list, string and big-string
    // cases each of them tries first, and the `values_equal` recursion behind
    // `==`. Overflow and division by zero fall through to the general code
    // rather than being handled twice.
    if (is_fixnum(lhs) && is_fixnum(rhs)) {
        const int64_t x = fixnum_value(lhs), y = fixnum_value(rhs);
        int64_t r;
        switch (op) {
            case Op::Add:
                if (!add_overflow(x, y, &r) && fixnum_fits(r)) {
                    ret(p, make_fixnum(r));
                    return;
                }
                break;
            case Op::Sub:
                if (!sub_overflow(x, y, &r) && fixnum_fits(r)) {
                    ret(p, make_fixnum(r));
                    return;
                }
                break;
            case Op::Mul:
                if (!mul_overflow(x, y, &r) && fixnum_fits(r)) {
                    ret(p, make_fixnum(r));
                    return;
                }
                break;
            case Op::Div:
                // A quotient leaves the fixnum range in exactly one place:
                // `-2^62 / -1` is `2^62`, one past the largest. `arith` hands
                // that to the double path and answers a float; this one used
                // to build the fixnum anyway, which wraps it to the *smallest*
                // fixnum. It was the only number the two paths disagreed
                // about, and which one a program got depended on whether both
                // operands happened to be in hand -- so `12 / n` and a `12 / n`
                // whose `n` arrived as a thunk gave different answers.
                if (y != 0) {
                    r = x / y;
                    if (fixnum_fits(r)) {
                        ret(p, make_fixnum(r));
                        return;
                    }
                }
                break;
            case Op::Mod:
                // A remainder is smaller than its divisor, so it is a fixnum
                // whenever the divisor was one: nothing to check.
                if (y != 0) {
                    ret(p, make_fixnum(x % y));
                    return;
                }
                break;
            // Two fixnums' bits combine into a fixnum: both are sign-extended
            // 63-bit numbers, and so is anything `&`, `|` or `^` makes of them.
            case Op::BitAnd: ret(p, make_fixnum(x & y)); return;
            case Op::BitOr: ret(p, make_fixnum(x | y)); return;
            case Op::BitXor: ret(p, make_fixnum(x ^ y)); return;
            case Op::Shl:
                // Shifting left is multiplying by a power of two, and overflows
                // the same way; a count past 62 always does, unless x is 0.
                if (y >= 0 && y < 63 && !mul_overflow(x, int64_t(1) << y, &r) && fixnum_fits(r)) {
                    ret(p, make_fixnum(r));
                    return;
                }
                break;
            case Op::Shr:
                // Arithmetic, as C++20 defines `>>` of a negative number:
                // floor division by 2^y, ending at -1 rather than 0.
                if (y >= 0) {
                    ret(p, make_fixnum(y >= 63 ? (x < 0 ? -1 : 0) : x >> y));
                    return;
                }
                break;
            case Op::Eq: ret(p, make_bool(x == y)); return;
            case Op::Ne: ret(p, make_bool(x != y)); return;
            case Op::Lt: ret(p, make_bool(x < y)); return;
            case Op::Le: ret(p, make_bool(x <= y)); return;
            case Op::Gt: ret(p, make_bool(x > y)); return;
            case Op::Ge: ret(p, make_bool(x >= y)); return;
            default: break;
        }
    }
    Value out;
    bool ok = (op == Op::Eq || op == Op::Ne || op == Op::Lt || op == Op::Le ||
               op == Op::Gt || op == Op::Ge)
                  ? compare(p, op, lhs, rhs, &out)
                  : arith(p, op, lhs, rhs, &out);

    // A nested force inside the operator hit a blocking operation and
    // gave up (see `force_whnf`) -- comparing two lists whose elements
    // are still `join!`s, say. Both operands are ordinary values and
    // both operators force, so this is reachable without any effect
    // being written at the comparison itself.
    //
    // The answer is not "this failed" but "not yet", so park and redo
    // the whole operation when the scheduler wakes us. Forcing is
    // memoised, so the retry pays only for what had not been forced
    // yet. This is the same handshake `apply_native` performs for a
    // native whose nested force gave up; the operators need their own
    // because they are the machine rather than a native call.
    //
    // Without it `ok` is false with a `p.result` that was never an
    // error, and the process dies with an error that has no kind and
    // no message.
    if (p.park_requested) {
        // Both operands travel in a cell: this retry may be spliced in
        // below a suspended force, and that work overwrites `p.result`
        // before the retry is reached.
        Value pair = p.heap().make_cons(lhs, rhs);
        push_retry(p, ContKind::BinFinish, uint32_t(op), 1, 0, pair);
        return;  // the mode is already Return
    }
    if (ok) ret(p, out); else do_raise(p, out);
}

/// `if cond { a } else { b }`, with the condition in hand.
/// The target a switch names for this key, or its default. The table is short
/// -- one entry per tag a `match` names -- and each probe is a compare of two
/// immediates, so a scan is the right shape for it; what it replaces is a
/// dozen reductions per arm the chain would otherwise have tested and failed.
uint32_t switch_target(Process& p, const Node& n, Value key) {
    const Image& img = img_of(p);
    const uint32_t* kids = img.kids_at(n.b);
    const uint32_t pairs = n.c / 2;
    if (is_atom(key)) {
        for (uint32_t k = 0; k < pairs; ++k) {
            if (make_atom(p.runtime().image_atom(img.node(kids[2 * k]).a)) == key) return kids[2 * k + 1];
        }
    }
    return kids[n.c - 1];
}

/// A switch whose subject is in hand. `switch_atom` keys on the subject;
/// `switch_head` on the head of a list cell, which it forces first -- the
/// compiler only emits one where the arm chain would have forced that head
/// before looking at anything else (see `lower_match`), so nothing is forced
/// that was not going to be, and nothing earlier.
void take_switch(Process& p, uint32_t node, Value subject, Value frame) {
    const Node& n = img_of(p).node(node);
    subject = resolve(subject);
    if (Op(n.op) == Op::SwitchAtom) {
        // A record is keyed by its id, which is how a `match` over a union
        // jumps to its variant's arm (`arm_key` in dreams/src/lower.dr). The id
        // is in the header, so nothing is forced to read it.
        if (is_obj(subject, ObjType::Record)) subject = static_cast<RecordObj*>(as_obj(subject))->id();
        eval_node(p, switch_target(p, n, subject), frame);
        return;
    }
    // The head of a list cell, or the first element of an array: a variant of
    // a `union struct` is `#[:tag, ..]` and is told apart the same way.
    Value head;
    if (is_obj(subject, ObjType::Cons)) {
        head = resolve(static_cast<ConsObj*>(as_obj(subject))->head);
    } else if (is_obj(subject, ObjType::Array) && static_cast<ArrayObj*>(as_obj(subject))->len > 0) {
        head = resolve(static_cast<ArrayObj*>(as_obj(subject))->items()[0]);
    } else {
        eval_node(p, img_of(p).kid(n.b + n.c - 1), frame);
        return;
    }
    if (is_whnf(head)) {
        eval_node(p, switch_target(p, n, head), frame);
        return;
    }
    push_cont(p, ContKind::SwitchKey, node, 0, 0, frame);
    enter(p, head);
}

void take_branch(Process& p, Value cond, uint32_t then_node, uint32_t else_node, Value frame) {
    if (!is_bool(cond)) {
        do_raise(p, type_error(p, "`if` needs a bool, got " + describe(p, cond)));
        return;
    }
    if (truthy(cond)) {
        eval_node(p, then_node, frame);
    } else if (else_node == NO_NODE) {
        ret(p, UNIT);
    } else {
        eval_node(p, else_node, frame);
    }
}

/// `a && b` and `a || b`, with the left operand in hand. The right is entered
/// only when the left has not already decided the answer.
void finish_logic(Process& p, Op op, Value lhs, uint32_t right_node, Value frame) {
    if (!is_bool(lhs)) {
        do_raise(p, type_error(p, std::string("`") + op_name(op) + "` needs a bool, got " +
                                      describe(p, lhs)));
        return;
    }
    if (op == Op::And && !truthy(lhs)) { ret(p, FALSE_V); return; }
    if (op == Op::Or && truthy(lhs)) { ret(p, TRUE_V); return; }
    eval_node(p, right_node, frame);
}

/// Negation, logical not and surface type tests, with the operand in hand.
void finish_unary(Process& p, Op op, Value v, uint32_t type, uint32_t invert) {
    if (op == Op::ListIsEmpty) {
        ret(p, make_bool(is_nil(v)));
        return;
    }
    if (op == Op::ListTail) {
        if (!is_obj(v, ObjType::Cons)) {
            do_raise(p, type_error(p, "tail needs a non-empty list"));
            return;
        }
        enter(p, static_cast<ConsObj*>(as_obj(v))->tail);
        return;
    }
    if (op == Op::TypeIs) {
        ret(p, make_bool((surface_type(v) == dream_type(type)) != bool(invert)));
        return;
    }
    if (op == Op::BitNot) {
        // ~x is -x - 1, which for a fixnum is a fixnum.
        if (is_fixnum(v)) { ret(p, make_fixnum(~fixnum_value(v))); return; }
        if (bigint::is_big(v)) { ret(p, bigint::bit_not(p.heap(), v)); return; }
        do_raise(p, type_error(p, "`~` needs an integer, got " + describe(p, v)));
        return;
    }
    if (op == Op::Neg) {
        if (is_fixnum(v)) { ret(p, make_integer(p, -fixnum_value(v))); return; }
        if (bigint::is_big(v)) { ret(p, bigint::negate(p.heap(), v)); return; }
        if (is_obj(v, ObjType::Float)) {
            ret(p, p.heap().make_float(-static_cast<FloatObj*>(as_obj(v))->value));
            return;
        }
        if (is_obj(v, ObjType::Tensor)) {
            // As `0 - t`, because that is what compiled code makes of a
            // negation it cannot do in registers, and the two tiers must
            // print the same numbers.
            Value r;
            if (tensor_arith(p, Op::Sub, make_fixnum(0), v, &r)) ret(p, r); else do_raise(p, r);
            return;
        }
        do_raise(p, type_error(p, "cannot negate " + describe(p, v)));
        return;
    }
    if (!is_bool(v)) {
        do_raise(p, type_error(p, "`not` needs a bool, got " + describe(p, v)));
        return;
    }
    ret(p, make_bool(!truthy(v)));
}

void step_eval(Process& p) {
    const Image& img = img_of(p);
    const Node& n = img.node(p.node);
    Value frame = p.frame;

    switch (Op(n.op)) {
        case Op::ConstInt: ret(p, make_integer(p, img.integer(n.a))); return;
        case Op::ConstFloat: ret(p, literal_float(p, n.a)); return;
        case Op::ConstStr: ret(p, literal_string(p, n.a)); return;
        case Op::ConstChar: ret(p, make_char(uint32_t(n.a))); return;
        case Op::ConstBool: ret(p, make_bool(n.a != 0)); return;
        case Op::ConstAtom:
            // Image atom indices are remapped onto the runtime's global table,
            // so atoms from different modules with the same name compare equal.
            // The map is built at load; here it is one array read.
            ret(p, make_atom(p.runtime().image_atom(n.a)));
            return;
        case Op::Unit: ret(p, UNIT); return;

        case Op::Local: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            Value v = fo->slots()[n.a];
            if (v == NIL_SLOT) {
                do_raise(p, type_error(p, "binding used before it was bound"));
                return;
            }
            enter(p, v);
            return;
        }
        case Op::Capture: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            auto* cl = static_cast<ClosureObj*>(as_obj(fo->closure));
            enter(p, cl->caps()[n.a]);
            return;
        }
        case Op::Global: {
            const GlobalRec& g = img.global(n.a);
            if (g.kind == GLOBAL_FUNCTION) {
                const FuncRec& f = img.func(g.target);
                // A parameterless impure binding is an action, not a value:
                // `let go! = { .. }` is performed by naming it, and is not
                // memoized, so each mention runs it again. That is the same
                // rule the entry point `main!` follows.
                if (f.arity == 0 && (f.flags & FN_IMPURE) && !(f.flags & FN_GLOBAL_VALUE)) {
                    Value cl = global_value(p, n.a);
                    Value fr = p.push_frame(cl, f.slots);
                    eval_node(p, f.body, fr);
                    return;
                }
            }
            enter(p, global_value(p, n.a));
            return;
        }
        case Op::Builtin: ret(p, make_builtin(n.a)); return;

        case Op::Field: {
            // `mod.member`, where `mod` is a module named at the top level, is
            // what every call of a `std` function looks like -- and a module
            // value is a per-process constant, so there is nothing here to
            // evaluate. Resolving it in place saves the pair of steps the
            // general path costs: one Eval to reach the module, one Return to
            // come back with it.
            Value obj;
            if (!module_operand(p, img, n.a, frame, &obj)) {
                // The node index rides along so the member lookup can be
                // cached against the site that asked for it.
                push_cont(p, ContKind::FieldOf, n.b, p.node, 0, UNIT);
                eval_node(p, n.a, frame);
                return;
            }
            resolve_field(p, obj, n.b, p.node);
            return;
        }

        DREAM_PRIMITIVE_CASES
            fill_native_args(p, p.node, make_builtin(n.a), uint32_t(p.stack.size()), 0, frame);
            return;
        case Op::Apply: {
            const uint32_t* kids = img.kids_at(n.b);
            // Almost every call names its callee -- a global, a builtin, a
            // parameter, a member of an imported module -- and naming one is
            // a read, not an evaluation. Reading it here goes straight to the
            // application; the general path would push a continuation, spend a
            // reduction evaluating the name, and come back to do this anyway.
            Value callee;
            if (!callee_operand(p, img, n.a, frame, &callee)) {
                for (uint32_t i = 0; i < n.c; ++i) {
                    p.stack.push_back(thunk_for(p, kids[i], frame));
                }
                push_cont(p, ContKind::ApplyTo, n.c, 0, 0, UNIT);
                eval_node(p, n.a, frame);
                return;
            }
            if (callee == NIL_SLOT) {
                do_raise(p, type_error(p, "binding used before it was bound"));
                return;
            }
            // A saturated call to a function we already have: the arguments
            // are the frame's first slots, so build the frame and write them
            // there. The general path pushes each one onto the value stack for
            // `do_apply` to copy out again and erase -- three passes over the
            // arguments where a call needs one.
            Value fn = resolve(callee);
            if (is_obj(fn, ObjType::Closure)) {
                auto* cl = static_cast<ClosureObj*>(as_obj(fn));
                const FuncRec& f = img.func(cl->func);
                if (f.arity == n.c && n.c > 0) {
                    // A trivial accessor needs no environment of its own.
                    // Read only an already-built container; suspended inputs,
                    // out-of-range indexes and fallbacks use the normal call.
                    const auto& accessor = img.accessor(cl->func);
                    if (accessor.container_slot != NO_NODE &&
                        !p.runtime().profiling() && p.reductions > 1) {
                        Value container, key = UNIT;
                        bool have_key = false;
                        if (operand_value(p, img, kids[accessor.container_slot], frame, &container)) {
                            if (accessor.key_slot == NO_NODE) {
                                if (fixnum_fits(accessor.key)) {
                                    key = make_fixnum(accessor.key);
                                    have_key = true;
                                }
                            } else {
                                have_key = operand_value(p, img, kids[accessor.key_slot], frame, &key);
                            }
                            // A record's accessor is the same `.[i]` an
                            // array's was, and earns the same shortcut.
                            const bool positional = is_obj(container, ObjType::Array) ||
                                                    is_obj(container, ObjType::Record);
                            if (have_key && is_fixnum(key) && positional) {
                                const bool rec = is_obj(container, ObjType::Record);
                                const uint32_t len =
                                    rec ? static_cast<RecordObj*>(as_obj(container))->len
                                        : static_cast<ArrayObj*>(as_obj(container))->len;
                                Value* items =
                                    rec ? static_cast<RecordObj*>(as_obj(container))->items()
                                        : static_cast<ArrayObj*>(as_obj(container))->items();
                                const int64_t index = fixnum_value(key);
                                if (index >= 0 && uint64_t(index) < len &&
                                    items[index] != NIL_SLOT) {
                                    --p.reductions;
                                    ++p.total_reductions;
                                    enter(p, items[index]);
                                    return;
                                }
                            }
                        }
                    }
                    if (const uint32_t mask = img.strict_args(cl->func)) {
                        fill_closure_args(p, p.node, mask, uint32_t(p.stack.size()), 0, frame);
                        return;
                    }
                    // The arguments are made before the frame, because a call
                    // in tail position makes its frame where the caller's was
                    // (see `FrameStack`) and they are read out of the caller's.
                    // Held in a local array, since allocation never collects
                    // -- the rule the whole machine rests on.
                    if (n.c <= 8) {
                        Value args[8];
                        for (uint32_t i = 0; i < n.c; ++i) args[i] = thunk_for(p, kids[i], frame);
                        Value fr = p.push_frame(fn, f.slots, f.arity);
                        auto* fo = static_cast<FrameObj*>(as_obj(fr));
                        for (uint32_t i = 0; i < n.c; ++i) fo->slots()[i] = args[i];
                        enter_function(p, cl->func, f, fr);
                        return;
                    }
                    const size_t base = p.stack.size();
                    for (uint32_t i = 0; i < n.c; ++i) p.stack.push_back(thunk_for(p, kids[i], frame));
                    Value fr = p.push_frame(fn, f.slots, f.arity);
                    auto* fo = static_cast<FrameObj*>(as_obj(fr));
                    for (uint32_t i = 0; i < n.c; ++i) fo->slots()[i] = p.stack[base + i];
                    p.stack.resize(base);
                    enter_function(p, cl->func, f, fr);
                    return;
                }
            }
            // A saturated call to a native: its strict arguments are
            // evaluated where they stand rather than suspended and forced one
            // step later. See `fill_native_args`.
            if (is_builtin(fn) || is_obj(fn, ObjType::Native)) {
                const uint32_t arity = native_arity(fn);
                if (arity == n.c || arity == NATIVE_VARIADIC) {
                    fill_native_args(p, p.node, fn, uint32_t(p.stack.size()), 0, frame);
                    return;
                }
            }
            for (uint32_t i = 0; i < n.c; ++i) {
                p.stack.push_back(thunk_for(p, kids[i], frame));
            }
            do_apply(p, fn, n.c);
            return;
        }

        case Op::If: {
            Value cond;
            if (!operand_value(p, img, n.a, frame, &cond)) {
                push_cont(p, ContKind::IfBranch, n.b, n.c, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            take_branch(p, cond, n.b, n.c, frame);
            return;
        }

        case Op::Block: advance_block(p, n.a, n.b, 0, frame); return;

        case Op::SwitchHead:
        case Op::SwitchAtom: {
            Value subject;
            if (!operand_value(p, img, n.a, frame, &subject)) {
                push_cont(p, ContKind::SwitchOn, p.node, 0, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            take_switch(p, p.node, subject, frame);
            return;
        }

        case Op::Bind: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            Value bound = thunk_for(p, n.b, frame);
            bind_slot(p, fo, n.a, bound);
            ret(p, UNIT);
            return;
        }

        case Op::MakeClosure:
        case Op::MakeThunk: {
            Value cl;
            if (!build_closure(p, n.a, frame, &cl)) {
                do_raise(p, cl);
                return;
            }
            ret(p, cl);
            return;
        }

        case Op::Try:
            push_cont(p, ContKind::Catch, n.b, n.c, uint32_t(p.stack.size()), frame);
            eval_node(p, n.a, frame);
            return;

        case Op::Ensure:
            push_cont(p, ContKind::Finalize, n.b, 0, uint32_t(p.stack.size()), frame);
            eval_node(p, n.a, frame);
            return;

        case Op::Force: eval_node(p, n.a, frame); return;

        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::Eq: case Op::Ne: {
            Value lhs;
            if (!operand_value(p, img, n.a, frame, &lhs)) {
                push_cont(p, ContKind::BinRight, n.op, n.b, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            Value rhs;
            if (!operand_value(p, img, n.b, frame, &rhs)) {
                push_cont(p, ContKind::BinFinish, n.op, 0, 0, lhs);
                eval_node(p, n.b, frame);
                return;
            }
            finish_binary(p, Op(n.op), lhs, rhs);
            return;
        }

        case Op::And: case Op::Or: {
            Value lhs;
            if (!operand_value(p, img, n.a, frame, &lhs)) {
                push_cont(p, ContKind::LogicRight, n.op, n.b, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            finish_logic(p, Op(n.op), lhs, n.b, frame);
            return;
        }

        case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs: case Op::ListTail: case Op::ListIsEmpty: {
            Value v;
            if (!operand_value(p, img, n.a, frame, &v)) {
                push_cont(p, ContKind::UnaryFinish, n.op, n.b, n.c, UNIT);
                eval_node(p, n.a, frame);
                return;
            }
            finish_unary(p, Op(n.op), v, n.b, n.c);
            return;
        }

        case Op::Cons: {
            Value head = thunk_for(p, n.a, frame);
            Value tail = thunk_for(p, n.b, frame);
            ret(p, p.heap().make_cons(head, tail));
            return;
        }
        case Op::MakeList: {
            const uint32_t* kids = img.kids_at(n.a);
            Value list = NIL;
            for (uint32_t i = n.b; i > 0; --i) {
                list = p.heap().make_cons(thunk_for(p, kids[i - 1], frame), list);
            }
            ret(p, list);
            return;
        }
        case Op::MakeArray: {
            const uint32_t* kids = img.kids_at(n.a);
            Value arr = p.heap().make_array(n.b);
            for (uint32_t i = 0; i < n.b; ++i) {
                Value item = thunk_for(p, kids[i], frame);
                static_cast<ArrayObj*>(as_obj(arr))->items()[i] = item;
            }
            ret(p, arr);
            return;
        }
        case Op::MakeMap: {
            Value m = p.heap().make_map(0);
            p.stack.push_back(m);
            advance_map(p, n.a, n.b, 0, frame);
            return;
        }
        case Op::RecordMake: {
            const uint32_t* kids = img.kids_at(n.a);
            Value rec = p.heap().make_record(
                p.runtime().image_atom(n.c & RECORD_MAKE_ATOM) | (n.c & RECORD_MAKE_VARIANT), n.b);
            for (uint32_t i = 0; i < n.b; ++i) {
                Value item = thunk_for(p, kids[i], frame);
                static_cast<RecordObj*>(as_obj(rec))->items()[i] = item;
            }
            ret(p, rec);
            return;
        }

        // The container and then the key, each forced where everything else
        // is: by the machine, with a continuation, and not underneath a native.
        // Neither is usually an expression, though -- `xs.[0]` is a binding and
        // a literal -- so both are read directly when they can be.
        case Op::Get: case Op::Set: {
            Value container;
            if (!operand_value(p, img, n.a, frame, &container)) {
                push_cont(p, ContKind::IndexKey, 0, p.node, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            Value key;
            if (!operand_value(p, img, n.b, frame, &key)) {
                // The container waits on the value stack, where the collector
                // can see it, while the key is forced.
                p.stack.push_back(container);
                push_cont(p, ContKind::IndexApply, 0, p.node, 0, frame);
                eval_node(p, n.b, frame);
                return;
            }
            if (Op(n.op) == Op::Get) container_get(p, p.node, container, key, frame);
            else container_set(p, n.c, container, key, frame);
            return;
        }
        case Op::SetMany: {
            Value container;
            if (!operand_value(p, img, n.a, frame, &container)) {
                push_cont(p, ContKind::IndexKey, 0, p.node, 0, frame);
                eval_node(p, n.a, frame);
                return;
            }
            set_many(p, p.node, container, 0, frame);
            return;
        }

        case Op::Nop: ret(p, UNIT); return;
        default:
            do_raise(p, type_error(p, "unimplemented opcode"));
            return;
    }
}

/// Hand `p.result` to the top continuation.
///
/// `floor` is the depth this loop belongs to: `run_process` owns the whole
/// stack and passes zero, and a nested force (`force_whnf`) owns only what it
/// pushed and passes the depth it found. It matters because of the thunk
/// updates below, which are the one continuation this runs several of in a
/// row -- and running one that belongs to the machine *outside* would be
/// finishing someone else's work under them.
void step_return(Process& p, size_t floor) {
  // Thunk updates are handled here rather than by returning to the machine for
  // each one. Forcing a value nested n deep leaves n of them stacked, and each
  // is a single store -- so what the machine would be dispatching between is
  // one write and the next.
  for (;;) {
    if (p.conts.size() <= floor) {
        if (p.conts.empty()) p.mode = Mode::Halted;
        return;
    }
    Cont c = p.conts.back();
    p.conts.pop_back();
    // Nothing still to run was pushed after the frames made deeper than this,
    // so they are done with: see `FrameStack`.
    p.frames.returned_to(p.conts.size());

    if (c.kind == ContKind::UpdateThunk) {
        // Overwrite the thunk in place so every sharer sees the result.
        Obj* o = as_obj(c.v1);
        // A thunk that has survived into old space now gains a young target,
        // so the write barrier notes the edge for the next minor collection.
        p.heap().remember_if_old(o, p.result);
        // The target goes in before the type flips, so the flip -- which is
        // what a concurrent mark's helper reads first -- cannot be seen with
        // the target still unset. The flip store is the one that releases.
        std::atomic_ref<Value>(static_cast<IndirectObj*>(o)->target)
            .store(p.result, std::memory_order_relaxed);
        obj_type_store_release(o, ObjType::Indirect);
        continue;  // the value flows on to the next continuation
    }

    switch (c.kind) {
        case ContKind::Halt:
            p.mode = Mode::Halted;
            return;

        case ContKind::UpdateThunk:
            p.unreachable("thunk update is handled above");

        case ContKind::ApplyTo:
            do_apply(p, p.result, c.a);
            return;

        case ContKind::IfBranch:
            take_branch(p, p.result, c.a, c.b, c.v1);
            return;

        case ContKind::SwitchOn:
            take_switch(p, c.a, p.result, c.v1);
            return;

        case ContKind::SwitchKey:
            eval_node(p, switch_target(p, img_of(p).node(c.a), resolve(p.result)), c.v1);
            return;

        case ContKind::EnterRetry:
            // The suspended work has finished and popped what it pushed, so
            // anything above the recorded height is what the abandoned
            // compiled call left there (a native's arguments).
            p.stack.resize(c.b);
            p.effect_retry_id = c.c;
            enter_function(p, c.a, img_of(p).func(c.a), c.v1);
            // Not taken: the function has left the tier since it parked.
            if (p.effect_retry_id) p.effect_retries.erase(p.effect_retry_id);
            p.effect_retry_id = 0;
            return;

        case ContKind::BinRight: {
            // The right operand is often a constant or a bound local, in which
            // case the operator finishes here rather than after another round
            // trip through the machine.
            Value lhs = p.result;
            Value rhs;
            if (!operand_value(p, img_of(p), c.b, c.v1, &rhs)) {
                push_cont(p, ContKind::BinFinish, c.a, 0, 0, lhs);
                eval_node(p, c.b, c.v1);
                return;
            }
            finish_binary(p, Op(c.a), lhs, rhs);
            return;
        }

        case ContKind::BinFinish: {
            Op op = Op(c.a);
            // Held across the call: comparing forces, and `force_whnf` leaves
            // its own value in `p.result` on the way out.
            Value lhs = c.v1;
            Value rhs = p.result;
            // `b == 1` marks a retry that was spliced in below a suspended
            // force (see below). That work finishes first and overwrites
            // `p.result` on its way out, so both operands travelled here in a
            // cell of their own instead.
            if (c.b == 1) {
                auto* pair = static_cast<ConsObj*>(as_obj(c.v1));
                lhs = pair->head;
                rhs = pair->tail;
            }
            finish_binary(p, op, lhs, rhs);
            return;
        }

        case ContKind::LogicRight:
            finish_logic(p, Op(c.a), p.result, c.b, c.v1);
            return;

        case ContKind::UnaryFinish:
            finish_unary(p, Op(c.a), p.result, c.b, c.c);
            return;

        case ContKind::BlockNext:
            advance_block(p, c.a, c.b, c.c, c.v1);
            return;

        case ContKind::BindThen: {
            const Image& img = img_of(p);
            const Node& sn = img.node(img.kid(c.a + c.c));
            auto* fo = static_cast<FrameObj*>(as_obj(c.v1));
            bind_slot(p, fo, sn.a, p.result);
            advance_block(p, c.a, c.b, c.c + 1, c.v1);
            return;
        }

        case ContKind::Catch:
            // The protected body finished normally; the handler is not needed.
            ret(p, p.result);
            return;

        case ContKind::Finalize:
            push_cont(p, ContKind::Finalized, 0, 0, c.c, p.result);
            ++p.cleanup_depth;
            eval_node(p, c.a, c.v1);
            return;

        case ContKind::Finalized:
            --p.cleanup_depth;
            p.stack.resize(c.c);
            if (c.a) do_raise(p, c.v1);
            else ret(p, c.v1);
            return;

        case ContKind::FieldOf:
            resolve_field(p, p.result, c.a, c.b);
            return;

        case ContKind::NativeArg:
            p.stack[c.a + c.c] = p.result;
            resume_native(p, c.v1, c.a, c.b, c.c + 1);
            return;

        case ContKind::NativeArgs: {
            // The argument just evaluated joins the ones already at the base,
            // and the rest carry on from there.
            p.stack.push_back(p.result);
            const Image& img = img_of(p);
            Value callee;
            if (is_primitive(Op(img.node(c.a).op))) {
                callee = make_builtin(img.node(c.a).a);
            } else if (!callee_operand(p, img, img.node(c.a).a, c.v1, &callee)) {
                p.unreachable("the callee of a native call stopped being a name");
            }
            fill_native_args(p, c.a, resolve(callee), c.b, c.c + 1, c.v1);
            return;
        }

        case ContKind::ClosureArgs: {
            p.stack.push_back(p.result);
            const Image& img = img_of(p);
            Value callee;
            if (!callee_operand(p, img, img.node(c.a).a, c.v1, &callee)) {
                p.unreachable("the callee of a closure call stopped being a name");
            }
            auto* cl = static_cast<ClosureObj*>(as_obj(resolve(callee)));
            fill_closure_args(p, c.a, img.strict_args(cl->func), c.b, c.c + 1, c.v1);
            return;
        }

        case ContKind::NativeRetry:
            resume_native(p, c.v1, c.a, c.b, 0);
            return;

        case ContKind::MapEntry: {
            Value key = p.result;
            Value valth = thunk_for(p, img_of(p).kid(c.a + c.c * 2 + 1), c.v1);
            p.stack.back() = map_insert(p, resolve(p.stack.back()), key, valth);
            advance_map(p, c.a, c.b, c.c + 1, c.v1);
            return;
        }

        case ContKind::IndexKey: {
            Value container = p.result;
            Value key;
            const Node& idx = img_of(p).node(c.b);
            if (Op(idx.op) == Op::SetMany) {
                set_many(p, c.b, container, 0, c.v1);
                return;
            }
            if (!operand_value(p, img_of(p), idx.b, c.v1, &key)) {
                // The container waits on the value stack, where the collector
                // can see it, while the key is forced.
                p.stack.push_back(container);
                push_cont(p, ContKind::IndexApply, 0, c.b, 0, c.v1);
                eval_node(p, idx.b, c.v1);
                return;
            }
            if (Op(idx.op) == Op::Get) container_get(p, c.b, container, key, c.v1);
            else container_set(p, idx.c, container, key, c.v1);
            return;
        }

        case ContKind::IndexApply: {
            Value container = p.stack.back();
            p.stack.pop_back();
            if (Op(img_of(p).node(c.b).op) == Op::Get) {
                container_get(p, c.b, container, p.result, c.v1);
            } else {
                container_set(p, img_of(p).node(c.b).c, container, p.result, c.v1);
            }
            return;
        }

        case ContKind::SetManyStep:
            set_many(p, c.b, p.result, c.c, c.v1);
            return;

        case ContKind::GetWalk:
            list_get(p, c.b, p.result, c.a, c.c, c.v1);
            return;

        case ContKind::SetWalk:
            list_set(p, c.b, p.result, c.a, c.c, c.v1);
            return;
    }
    return;
  }
}

/// The function a frame belongs to, or `NO_NODE` for a frame that has none
/// to name -- one copied from another heap before its closure was.
uint32_t frame_function(Value frame) {
    // A flat suspension holds its frame's closure where the frame would be.
    if (is_obj(frame, ObjType::Closure)) return static_cast<ClosureObj*>(as_obj(frame))->func;
    if (!is_obj(frame, ObjType::Frame)) return NO_NODE;
    Value cl = static_cast<FrameObj*>(as_obj(frame))->closure;
    if (!is_obj(cl, ObjType::Closure)) return NO_NODE;
    return static_cast<ClosureObj*>(as_obj(cl))->func;
}

/// Say where an error is being raised, the first time it is: which functions
/// were running, innermost first, and which suspensions were being forced.
///
/// A lazy program's failure is rarely where it was written. The function that
/// raised is the one whose frame is current, and a suspension being forced
/// names the function that *made* it -- the place the value was written, and
/// usually the line a reader wants, since that is the code that asked for the
/// work and not the code that happened to need it first. Both are on the
/// continuation stack already: a frame rides with most continuations, and an
/// update carries the thunk it will overwrite. So locating an error is a walk
/// down the top of that stack, and it costs nothing until something raises.
///
/// The answer is a list of fixnums, `fi + 1` for a function that was running
/// and `-(fi + 1)` for one that made a value being forced, which
/// `error_trace` turns into places from the image's function records. It is
/// written once: a `catch` that raises the same error again keeps where it
/// started, and a nested force that fails hands its outer machine an error
/// that is already located, more precisely than the outer one could.
///
/// Granularity is the function, because that is what the image records: the
/// optimizer shares identical nodes across the whole program, so a node has
/// no one place to report.
void locate_error(Process& p) {
    Value v = resolve(p.result);
    if (!is_obj(v, ObjType::ErrorBox)) return;
    auto* e = static_cast<ErrorObj*>(as_obj(v));
    if (e->where != UNIT || is_shared_obj(e)) return;

    constexpr size_t kMaxEntries = 12;
    constexpr size_t kMaxScan = 512;
    int64_t found[kMaxEntries];
    size_t n = 0;
    auto note = [&](int64_t entry) {
        if (n < kMaxEntries && (n == 0 || found[n - 1] != entry)) found[n++] = entry;
    };
    uint32_t fi = frame_function(p.frame);
    if (fi != NO_NODE) note(int64_t(fi) + 1);
    size_t scanned = 0;
    for (size_t i = p.conts.size(); i > 0 && n < kMaxEntries && scanned < kMaxScan; --i, ++scanned) {
        const Cont& c = p.conts.begin()[i - 1];
        switch (c.kind) {
            case ContKind::UpdateThunk: {
                // A blackhole while it is forced, and still a thunk's shape.
                Obj* o = as_obj(c.v1);
                ObjType t = obj_type(o);
                if (t != ObjType::Thunk && t != ObjType::Blackhole) break;
                uint32_t made = frame_function(static_cast<ThunkObj*>(o)->frame);
                if (made != NO_NODE) note(-(int64_t(made) + 1));
                break;
            }
            case ContKind::IfBranch: case ContKind::BinRight: case ContKind::LogicRight:
            case ContKind::BlockNext: case ContKind::BindThen: case ContKind::Catch: case ContKind::Finalize:
            case ContKind::NativeArgs:
            case ContKind::ClosureArgs:
            case ContKind::IndexKey: case ContKind::IndexApply: case ContKind::GetWalk:
            case ContKind::SetWalk: case ContKind::SwitchOn: case ContKind::SwitchKey:
            case ContKind::EnterRetry: case ContKind::SetManyStep: {
                uint32_t at = frame_function(c.v1);
                if (at != NO_NODE) note(int64_t(at) + 1);
                break;
            }
            default: break;
        }
    }
    if (n == 0) return;
    // Built from the end. Nothing here collects, so `e` stays good.
    Heap& h = p.heap();
    Value list = NIL;
    for (size_t i = n; i > 0; --i) list = h.make_cons(make_fixnum(found[i - 1]), list);
    e->where = list;
    h.remember_if_old(e, list);
}

/// Unwind to the nearest catch at or above `floor`. Returns false when the
/// error escapes it.
bool unwind(Process& p, size_t floor) {
    locate_error(p);
    while (p.conts.size() > floor) {
        Cont c = p.conts.back();
        p.conts.pop_back();

        if (c.kind == ContKind::UpdateThunk) {
            // The thunk did not produce a value. Put it back the way it was so
            // a later `try!` can attempt it again and see the same error,
            // rather than finding a blackhole and reporting a false loop.
            Obj* o = as_obj(c.v1);
            if (obj_type(o) == ObjType::Blackhole) obj_type_store(o, ObjType::Thunk);
            continue;
        }
        if (c.kind == ContKind::Catch) {
            // A process being killed is not unwound to a handler: `kill!`
            // ends it, and a `try!` that could catch that would be a process
            // that could refuse. Its stack goes with the frame.
            if (p.kill_unwinding && p.cleanup_depth == 0) {
                p.stack.resize(c.c);
                continue;
            }
            p.stack.resize(c.c);
            auto* fo = static_cast<FrameObj*>(as_obj(c.v1));
            value_slot_store(&fo->slots()[c.b], p.result);
            p.heap().remember_if_old(fo, p.result);
            eval_node(p, c.a, c.v1);
            return true;
        }
        if (c.kind == ContKind::Finalize) {
            p.stack.resize(c.c);
            push_cont(p, ContKind::Finalized, 1, 0, c.c, p.result);
            ++p.cleanup_depth;
            eval_node(p, c.a, c.v1);
            return true;
        }
        if (c.kind == ContKind::Finalized) {
            --p.cleanup_depth;
            p.stack.resize(c.c);
            // Keep the original failure; an error on normal exit becomes
            // the failure and still unwinds through the remaining cleanups.
            if (c.a) p.result = c.v1;
        }
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

std::vector<TracePlace> error_trace(Process& p, Value err) {
    std::vector<TracePlace> out;
    Value v = resolve(err);
    if (!is_obj(v, ObjType::ErrorBox)) return out;
    const Image& img = img_of(p);
    for (Value cur = resolve(static_cast<ErrorObj*>(as_obj(v))->where);
         is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        Value entry = resolve(static_cast<ConsObj*>(as_obj(cur))->head);
        if (!is_fixnum(entry)) break;
        int64_t n = fixnum_value(entry);
        bool made = n < 0;
        uint64_t fi = uint64_t(made ? -n : n) - 1;
        if (fi >= img.func_count()) break;
        const FuncRec& f = img.func(uint32_t(fi));
        TracePlace place{made, img.str(f.name).str(), "", 0, 0};
        if (f.source != 0 && f.source - 1 < img.string_count()) {
            place.path = img.str(f.source - 1).str();
            place.line = f.line;
            place.col = f.col;
        }
        out.push_back(std::move(place));
    }
    return out;
}

/// `DREAM_PROBE_THUNK`: count the suspensions a run makes, by the kind of
/// expression suspended, and report them with `--stats`.
///
/// `--stats` says how much of a program's allocation is thunks; this says what
/// they are *of*, which is the question that decides what to do about it. On a
/// self-compile it is what showed that two node kinds -- `apply` and `get` --
/// were 90% of every thunk made, and so that an in-range array read and an
/// already-built list prefix were worth handling in `thunk_for` while the rest
/// waits for a strictness analysis. One predictable branch per suspension, and
/// only on the path that was about to allocate anyway.

// The same counts by *site* -- the function a suspension was made in and the
// node it suspends -- since the kind alone says that `apply` is the problem
// and not which of a program's thousands of applications. Keyed by the
// function's index in the high half and the node in the low; counted under a
// lock, which only a probing run ever takes.
std::mutex g_thunk_sites_mutex;
std::unordered_map<uint64_t, uint64_t> g_thunk_sites;
const Image* g_thunk_sites_image = nullptr;

extern const bool g_probe_frames;
extern std::atomic<uint64_t> g_frames_made, g_frame_bytes, g_frames_held, g_held_bytes;

void dbg_dump_thunks() {
    if (g_probe_frames) {
        const uint64_t made = g_frames_made.load(), bytes = g_frame_bytes.load();
        const uint64_t held = g_frames_held.load(), held_bytes = g_held_bytes.load();
        std::fprintf(stderr, "; frames: %llu made, %llu bytes; %llu held by a suspension (%llu bytes); "
                             "%llu never held (%llu bytes, %.1f%% of frame bytes)\n",
                     (unsigned long long)made, (unsigned long long)bytes, (unsigned long long)held,
                     (unsigned long long)held_bytes, (unsigned long long)(made - held),
                     (unsigned long long)(bytes - held_bytes), bytes ? 100.0 * double(bytes - held_bytes) / double(bytes) : 0.0);
    }
    if (!g_probe_thunk) return;
    for (size_t i = 0; i < 64; ++i) {
        uint64_t c = g_thunk_counts[i].load();
        if (c > 50000)
            std::fprintf(stderr, "; thunked %-14s %llu\n", op_name(Op(i)),
                         (unsigned long long)c);
    }
    std::lock_guard<std::mutex> g(g_thunk_sites_mutex);
    if (!g_thunk_sites_image) return;
    std::vector<std::pair<uint64_t, uint64_t>> sites(g_thunk_sites.begin(), g_thunk_sites.end());
    std::sort(sites.begin(), sites.end(), [](auto& a, auto& b) { return a.second > b.second; });
    const Image& img = *g_thunk_sites_image;
    for (size_t i = 0; i < sites.size() && i < 25; ++i) {
        const uint32_t func = uint32_t(sites[i].first >> 32), node = uint32_t(sites[i].first);
        const std::string name = func < img.func_count() ? img.str(img.func(func).name).str() : "(top)";
        std::fprintf(stderr, ";   %10llu  %-14s node %-7u in %s\n", (unsigned long long)sites[i].second,
                     op_name(Op(img.node(node).op)), node, name.c_str());
    }
}

/// The builtin id of `_match_at`, found once: the table is the VM's own and
/// does not change while it runs.
uint32_t match_at_builtin() {
    static const uint32_t id = [] {
        for (uint32_t i = 0; i < builtin_count(); ++i) {
            if (std::strcmp(builtin_def(i).name, "_match_at") == 0) return i;
        }
        return UINT32_MAX;
    }();
    return id;
}

/// A copy on the heap of a frame on the process's stack, for what has to
/// keep a frame past the point its stack could drop it: compiled code's
/// snapshot (`dream_rt_adopt`) and a parked compiled call's retry.
Value heap_frame(Process& p, Value frame) {
    auto* from = static_cast<FrameObj*>(as_obj(frame));
    Value fr = p.heap().make_frame_filling(from->closure, from->nslots, from->nslots);
    auto* to = static_cast<FrameObj*>(as_obj(fr));
    for (uint32_t i = 0; i < from->nslots; ++i) to->slots()[i] = from->slots()[i];
    return fr;
}

Value thunk_for(Process& p, uint32_t node, Value frame) {
    const Image& img = img_of(p);
    const Node& n = img.node(node);
    // Wrapping something already in normal form in a thunk would cost an
    // allocation and, for a variable, would break sharing with the binding.
    switch (Op(n.op)) {
        case Op::ConstInt: return make_integer(p, img.integer(n.a));
        case Op::ConstStr: return literal_string(p, n.a);
        case Op::ConstFloat: return literal_float(p, n.a);
        case Op::ConstChar: return make_char(uint32_t(n.a));
        case Op::ConstBool: return make_bool(n.a != 0);
        case Op::ConstAtom: return make_atom(p.runtime().image_atom(n.a));
        case Op::Unit: return UNIT;
        case Op::Builtin: return make_builtin(n.a);
        case Op::Local: {
            Value v = static_cast<FrameObj*>(as_obj(frame))->slots()[n.a];
            if (v != NIL_SLOT) return v;
            break;
        }
        case Op::Capture: {
            auto* fo = static_cast<FrameObj*>(as_obj(frame));
            return static_cast<ClosureObj*>(as_obj(fo->closure))->caps()[n.a];
        }
        case Op::MakeClosure:
        case Op::MakeThunk: {
            // A closure is already a value. Wrapping it in a thunk would mean
            // `spawn! $( .. )` hands the child a suspension that evaluates to
            // the work rather than the work itself.
            Value cl;
            if (build_closure(p, n.a, frame, &cl)) return cl;
            break;
        }
        case Op::Add:
        case Op::Sub:
        case Op::Mul: {
            // `f (n - 1)`: one arithmetic step on two numbers already in hand.
            //
            // Suspending it costs a Thunk now and, when the callee looks at it,
            // a Blackhole write, an Indirect write, a continuation and a
            // reduction -- to save an add. On a self-compile thunks are 31% of
            // everything allocated and frames another 34%, and on `fib` the two
            // of them are the entire program, so this is the cheapest shape of
            // the biggest cost there is.
            //
            // It is not a strictness analysis and needs none, because nothing
            // here can be observed. `operand_value` reads an operand only when
            // it is *already* in weak head normal form -- it never forces, so
            // no evaluation is brought forward. Two fixnums added, subtracted
            // or multiplied cannot raise, cannot diverge and cannot depend on
            // anything; overflow and every other shape fall through to the
            // thunk. So the only difference is that an argument the callee
            // never looks at has been computed, which is one machine
            // instruction and no change in meaning.
            //
            // Division is deliberately absent: `x / 0` raises, and a raise the
            // program would not otherwise have reached is exactly the
            // observable difference this must not make.
            Value a, b;
            if (operand_value(p, img, n.a, frame, &a) && is_fixnum(a) &&
                operand_value(p, img, n.b, frame, &b) && is_fixnum(b)) {
                const int64_t x = fixnum_value(a), y = fixnum_value(b);
                int64_t r;
                bool over = true;
                switch (Op(n.op)) {
                    case Op::Add: over = add_overflow(x, y, &r); break;
                    case Op::Sub: over = sub_overflow(x, y, &r); break;
                    default: over = mul_overflow(x, y, &r); break;
                }
                if (!over && fixnum_fits(r)) return make_fixnum(r);
            }
            break;
        }
        case Op::Apply: {
            // A pattern's field: `#[:ok, v, next, pr]` and `Shape.circle(r)`
            // bind each name to `_match_at subject i`, lazily, after the test
            // has already said the subject is an array or record of that
            // shape. Suspending each of those reads was three or four thunks
            // per match -- in a parser that destructures a reply at every
            // step, most of everything it allocated -- for what is a load.
            // It is the `get` case below, reached through the builtin the
            // pattern compiler uses, and safe for the same reasons: the
            // subject and index are read only when already in normal form,
            // an in-range read cannot raise, and the element comes back as
            // stored, thunk and all.
            if (n.c != 2) break;
            const Node& callee = img.node(n.a);
            if (Op(callee.op) != Op::Builtin || callee.a != match_at_builtin()) break;
            const uint32_t* kids = img.kids_at(n.b);
            Value c, k;
            if (!operand_value(p, img, kids[0], frame, &c) ||
                !operand_value(p, img, kids[1], frame, &k) || !is_fixnum(k)) {
                break;
            }
            const int64_t i = fixnum_value(k);
            if (i < 0) break;
            if (is_obj(c, ObjType::Array)) {
                auto* a = static_cast<ArrayObj*>(as_obj(c));
                if (i < int64_t(a->len) && a->items()[i] != NIL_SLOT) return a->items()[i];
            } else if (is_obj(c, ObjType::Record)) {
                auto* r = static_cast<RecordObj*>(as_obj(c));
                if (i < int64_t(r->len) && r->items()[i] != NIL_SLOT) return r->items()[i];
            }
            break;
        }
        case Op::Get: {
            // `a.[i]` on an array already in hand, at an index inside it, is a
            // load. Suspending one is the second most common thunk a compile
            // makes -- 2.6M of them -- because a record here is an array and
            // reading a field of one is how everything downstream of the lexer
            // talks about a token.
            //
            // Safe for the same reason the arithmetic above is: `operand_value`
            // reads an operand only when it is *already* in normal form, so
            // nothing is forced early, and an in-range read of an array cannot
            // raise. Out of range falls through to the thunk, which is what
            // knows about `else d` and about raising.
            //
            // Only an array. A *list* index walks the spine, and walking forces
            // it -- on a lazy or infinite list that is a different program, not
            // a cheaper one. A map lookup hashes and compares the key, and for
            // a compound key that forces too. Neither belongs here.
            //
            // What comes back is the element as it is stored, thunk and all,
            // exactly as `Op::Local` hands back a slot: it is the same value
            // the suspension would have produced, and whoever forces it updates
            // the array's own copy, so sharing improves rather than suffers.
            Value c, k;
            if (!operand_value(p, img, n.a, frame, &c) ||
                !operand_value(p, img, n.b, frame, &k) || !is_fixnum(k)) {
                break;
            }
            const int64_t i = fixnum_value(k);
            if (i < 0) break;
            if (is_obj(c, ObjType::Array)) {
                auto* a = static_cast<ArrayObj*>(as_obj(c));
                if (i < int64_t(a->len)) {
                    Value v = a->items()[i];
                    if (v != NIL_SLOT) return v;
                }
                break;
            }
            if (is_obj(c, ObjType::Record)) {
                auto* r = static_cast<RecordObj*>(as_obj(c));
                if (i < int64_t(r->len)) {
                    Value v = r->items()[i];
                    if (v != NIL_SLOT) return v;
                }
                break;
            }
            // A list, walked only as far as it has already been built.
            //
            // Indexing a list is a walk, and a walk normally *forces* -- which
            // is why this cannot simply do what the array case does: on a lazy
            // or infinite list, forcing the spine early is a different program.
            // So the walk follows only cells that are already in normal form
            // and gives up the moment it would have to force one, which makes
            // it a pointer chase and nothing more. `list.nth` is
            // `xs.[n else ()]` and is how the compiler reads every record that
            // is still a list, so most of these are index 0, 1 or 2 into a
            // spine that was built several stages ago.
            //
            // The depth is capped because this runs whether or not the callee
            // ever looks at the argument. A short walk is cheaper than the
            // Thunk it replaces; a long one would be work done on spec.
            if (is_obj(c, ObjType::Cons) && i <= 8) {
                Value cur = c;
                for (int64_t step = 0; step < i; ++step) {
                    Value tail = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
                    if (!is_obj(tail, ObjType::Cons)) { cur = UNIT; break; }
                    cur = tail;
                }
                if (is_obj(cur, ObjType::Cons)) {
                    Value v = static_cast<ConsObj*>(as_obj(cur))->head;
                    if (v != NIL_SLOT) return v;
                }
            }
            break;
        }
        default: break;
    }
    if (g_probe_thunk) {
        g_thunk_counts[size_t(n.op) & 63].fetch_add(1, std::memory_order_relaxed);
        uint64_t func = 0xffffffffu;
        if (is_obj(frame, ObjType::Frame)) {
            Value cl = static_cast<FrameObj*>(as_obj(frame))->closure;
            if (is_obj(cl, ObjType::Closure)) func = static_cast<ClosureObj*>(as_obj(cl))->func;
        }
        std::lock_guard<std::mutex> g(g_thunk_sites_mutex);
        g_thunk_sites_image = &img;
        ++g_thunk_sites[(func << 32) | node];
    }
    if (stack_frame(frame)) return flat_thunk(p, node, frame);
    return p.heap().make_thunk(node, frame);
}

Value global_value(Process& p, uint32_t index) {
    if (p.globals.empty()) p.globals.assign(img_of(p).global_count(), NIL_SLOT);
    if (p.globals[index] != NIL_SLOT) return p.globals[index];

    const Image& img = img_of(p);
    const GlobalRec& g = img.global(index);
    Value v;
    if (g.kind == GLOBAL_MODULE) {
        const ImportRec& ir = img.import(g.target);
        StringRef alias = img.str(ir.alias);
        v = p.heap().make_module(g.target, p.heap().make_string(alias.data, alias.len));
    } else {
        const FuncRec& f = img.func(g.target);
        Value cl = p.heap().make_closure(g.target, 0);
        if (f.arity == 0 && (f.flags & FN_GLOBAL_VALUE)) {
            // A pure top-level value: force once, then memoize. The thunk is
            // the memo, so every later reference sees the computed result.
            Value fr = p.heap().make_frame(cl, f.slots);
            v = p.heap().make_thunk(f.body, fr);
        } else {
            v = cl;
        }
    }
    p.globals[index] = v;
    return v;
}

void prime_apply(Process& p, Value callee, uint32_t argc) {
    p.conts.clear();
    p.frames.clear();
    push_cont(p, ContKind::Halt, 0, 0, 0, UNIT);
    push_cont(p, ContKind::ApplyTo, argc, 0, 0, UNIT);
    // `enter` returns it immediately when it is already a function, and forces
    // it first when it is not, so both spawn shapes reach the same place.
    enter(p, callee);
}

void prime_entry(Process& p, const Image& img, uint32_t func) {
    // `let main! () = ..` is a function of one argument, and a program
    // that writes it means it to be called -- applied to nothing, it was
    // a closure nobody entered, and the run printed nothing and succeeded.
    const uint32_t arity = img.func(func).arity;
    for (uint32_t i = 0; i < arity; ++i) p.stack.push_back(UNIT);
    prime_apply(p, p.heap().make_closure(func, 0), arity);
}

void prime_force(Process& p, Value v) {
    p.conts.clear();
    p.frames.clear();
    push_cont(p, ContKind::Halt, 0, 0, 0, UNIT);
    enter(p, v);
}

/// The limits that keep one runaway process from taking the runtime with it.
///
/// A process is the unit of failure here, so exhausting memory has to kill the
/// process that did it and nothing else. Without these, `std::bad_alloc`
/// escapes the allocator and calls `terminate`, which loses every other
/// process, the scheduler, and any work already done.
///
/// Each is generous enough that ordinary programs never approach it -- a tail
/// call pops its continuation, so a loop runs in constant space, and
/// `sum_to 100000` needs 100k frames against a limit of four million -- and
/// each can be moved for a program that genuinely needs more.
struct Limits {
    size_t conts;
    size_t stack;
    size_t heap_bytes;

    /// What the command line said, which wins over the environment; zero for
    /// a limit it did not mention. See `override_limits`.
    static Limits& overrides() {
        static Limits o{0, 0, 0};
        return o;
    }

    static const Limits& get() {
        static Limits l = [] {
            auto pick = [](size_t given, const char* name, size_t fallback) {
                if (given) return given;
                size_t n = 0;
                if (const char* env = std::getenv(name); env && parse_size(env, &n)) return n;
                return fallback;
            };
            const Limits& o = overrides();
            return Limits{
                pick(o.conts, "DREAM_MAX_DEPTH", size_t(4u) << 20),      // ~96 MB of continuations
                pick(o.stack, "DREAM_MAX_STACK", size_t(4u) << 20),      // ~32 MB of values
                pick(o.heap_bytes, "DREAM_MAX_HEAP", size_t(1u) << 30),  // 1 GB per process
            };
        }();
        return l;
    }
};

bool parse_size(const char* text, size_t* out) {
    if (!text || !std::isdigit(static_cast<unsigned char>(*text))) return false;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(text, &end, 10);
    unsigned shift = 0;
    switch (*end) {
        case 'k': case 'K': shift = 10; ++end; break;
        case 'm': case 'M': shift = 20; ++end; break;
        case 'g': case 'G': shift = 30; ++end; break;
        default: break;
    }
    if (shift && (*end == 'b' || *end == 'B')) ++end;
    if (*end != '\0' || n == 0 || n > (~0ull >> shift)) return false;
    *out = size_t(n << shift);
    return true;
}

void override_limits(size_t conts, size_t stack, size_t heap_bytes) {
    Limits::overrides() = Limits{conts, stack, heap_bytes};
}

size_t max_depth_limit() { return Limits::get().conts; }

/// The raises behind the checks below, kept here rather than in the loops so
/// the per-reduction path stays a couple of compares and the error-string
/// construction stays in `.cold` clones.
void raise_depth_limit(Process& p, const WellKnownAtoms& wk, size_t depth) {
    do_raise(p, raise_error(p, wk.stack_overflow,
                            "recursion too deep: " + std::to_string(depth) + " pending frames"));
}
void raise_stack_limit(Process& p, const WellKnownAtoms& wk, size_t depth) {
    do_raise(p, raise_error(p, wk.stack_overflow,
                            "value stack too deep: " + std::to_string(depth) + " entries"));
}
void raise_heap_limit(Process& p, const WellKnownAtoms& wk, size_t max_heap) {
    do_raise(p, raise_error(p, wk.out_of_memory,
                            "process heap grew past " + std::to_string(max_heap) + " bytes"));
}

/// Raise in `p` if it has outgrown what one process may use. Returns true when
/// it did, so the caller can stop what it was doing.
///
/// Called at safepoints rather than from `push_cont` or `alloc`: here the
/// process is already in a consistent state, and safepoints are one reduction
/// apart, so a limit can only be overshot by a bounded amount. Making the
/// allocator itself fail would mean every caller of `alloc` -- most of which
/// hold raw object pointers -- had to cope with a null.
///
/// The limits and well-known atoms are passed in because this runs once a
/// reduction: reading them here would pay the thread-safe static initializer
/// guard and a handful of loads on every step of every loop. The callers hoist
/// them into local registers instead, which is what keeps this cheap enough to
/// run unconditionally.
static inline bool check_limits(Process& p, const WellKnownAtoms& wk,
                                size_t max_conts, size_t max_stack, size_t max_heap) {
    if (p.mode == Mode::Raise) return false;  // already unwinding; let it finish
    if (p.conts.size() > max_conts) {
        raise_depth_limit(p, wk, p.conts.size());
        return true;
    }
    if (p.stack.size() > max_stack) {
        raise_stack_limit(p, wk, p.stack.size());
        return true;
    }
    // Checked after a collection has had its chance, so this fires only for a
    // process whose *live* data is too big, not one that merely allocates fast.
    //
    // Once raised, it is not raised again until a collection has measured
    // the live data afresh: what the last one found is stale the moment the
    // error unwinds the frames that held it, and a `catch` must be able to run.
    //
    // A force nobody vouched for (`force_pins`) is the exception: nothing can
    // be collected under it, so the live figure is never measured afresh and
    // everything allocated since is held whether or not it is reachable. A
    // native walking an endless list -- `list.length (list.from 1)` -- grew
    // that way past the cap until the allocator failed and took the whole VM
    // down with `bad_alloc`. Under a pin, what has been allocated is the
    // measure.
    const bool over = p.force_pins > 0 ? p.heap().bytes_allocated() > max_heap
                                       : p.heap().bytes_allocated() > max_heap &&
                                             p.heap().bytes_live() > max_heap / 2;
    if (over && p.heap_limit_raised_at != p.heap().collections()) {
        p.heap_limit_raised_at = p.heap().collections();
        raise_heap_limit(p, wk, max_heap);
        return true;
    }
    return false;
}

/// The function whose frame is current, or `UINT32_MAX` when there is none --
/// at the very start of a process, or inside a native.
/// Charge one step's *own* allocation to `fi`.
///
/// The bytes a step allocates are the heap's total before and after -- but a
/// step that enters a native which forces runs whole machine loops underneath
/// itself, and each of those has already charged its own bytes. Subtract what
/// they claimed, and claim the rest. Without this a self-compile's allocation
/// profile totalled 36 GB against a heap that handed out 1.5 GB.
inline void note_own_alloc(Process& p, uint32_t fi, uint64_t a0, uint64_t n0) {
    const uint64_t delta = p.heap().bytes_total() - a0;
    const uint64_t nested = p.alloc_attributed - n0;
    const uint64_t own = delta > nested ? delta - nested : 0;
    if (own) {
        p.runtime().note_alloc(fi, own);
        p.alloc_attributed += own;
    }
}

uint32_t current_func(Process& p) {
    if (!is_ptr(p.frame)) return UINT32_MAX;
    Value c = static_cast<FrameObj*>(as_obj(p.frame))->closure;
    if (!is_obj(c, ObjType::Closure)) return UINT32_MAX;
    return static_cast<ClosureObj*>(as_obj(c))->func;
}

/// `step_return` with its allocation charged to whoever is returning.
///
/// A Return step is not a reduction -- it spends no budget, and the profile's
/// first table is right not to count it -- but it does allocate: it is where a
/// thunk's result is published and where the list builder conses. Leaving it
/// out put a third of a compile's bytes in no function at all.
inline void step_return_counted(Process& p, size_t floor) {
    const uint32_t fi = current_func(p);
    const uint64_t a0 = p.heap().bytes_total();
    const uint64_t n0 = p.alloc_attributed;
    step_return(p, floor);
    note_own_alloc(p, fi, a0, n0);
}

void run_process(Process& p, int64_t budget) {
    LoopStack loop(p, DREAM_FRAME_ADDRESS());
    p.reductions = budget;
    p.slice = budget;
    p.slice_spent = false;
    // The limits are loop-invariant over the whole run; read them once so the
    // per-reduction check stays a couple of compares against registers, not a
    // thread-safe static init guarded call.
    const Limits lim = Limits::get();
    const size_t max_conts = lim.conts;
    const size_t max_stack = lim.stack;
    const size_t max_heap = lim.heap_bytes;
    const WellKnownAtoms& wk = well_known(p.runtime());
    // Profiling never turns on mid-run, so this is a register for the whole
    // loop rather than a load per step. That matters most for the Return run
    // below, which is more than half of all the steps a program takes.
    const bool prof = p.runtime().profiling_for(p.id());
    Jit* const work_jit = p.runtime().jit();
    const bool sample_work = work_jit && work_jit->samples_work();
    // `slice_spent` is how a nested force says the budget ran out while it was
    // running: it re-armed the counter so that what reads it keeps working,
    // and this is where the slice it could not end actually ends.
    while (p.reductions > 0 && !p.slice_spent) {
        // Safepoint. Every live value is reachable from the process's stacks,
        // its frame and its result -- nothing is stranded in a C++ local.
        if (p.heap().should_collect()) p.maybe_collect();

        // A raised limit unwinds at the next dispatch, below.
        if (check_limits(p, wk, max_conts, max_stack, max_heap)) continue;

        switch (p.mode) {
            case Mode::Eval:
                --p.reductions;
                ++p.total_reductions;
                if (sample_work && (p.total_reductions & 63) == 0) {
                    work_jit->note_work(current_func(p));
                }
                // Attributed to whichever function's frame is current, which
                // is what makes a profile read like the source: a reduction
                // belongs to the code that asked for it.
                if (prof) {
                    // The frame is read before the step, because the step is
                    // what changes it, and the bytes after, because they are
                    // what the step allocated.
                    const uint32_t fi = current_func(p);
                    const uint64_t a0 = p.heap().bytes_total();
                    const uint64_t n0 = p.alloc_attributed;
                    p.runtime().note_reduction(fi);
                    if ((p.total_reductions & 1023) == 0 && timeline_on()) timeline_sample(p, fi);
                    step_eval(p);
                    note_own_alloc(p, fi, a0, n0);
                } else {
                    step_eval(p);
                }
                break;
            case Mode::Return:
                if (prof) step_return_counted(p, 0);
                else step_return(p, 0);
                break;
            case Mode::Raise:
                if (!unwind(p, 0)) {
                    p.failed = true;
                    p.exit_value = p.result;
                    p.mode = Mode::Halted;
                    return;
                }
                break;
            case Mode::Halted:
                p.exit_value = p.result;
                return;
        }

        // Hand the value that came back to whoever was waiting for it, and
        // keep doing so while there is nothing else to do.
        //
        // A Return step is bookkeeping, not work: it pops the continuation
        // stack it walks, so a run of them is bounded by the depth already
        // there and cannot spin. Leaving the safepoint check out of that run
        // costs at most a slightly late collection -- a safepoint is a place
        // collection *may* happen, not one where it must -- and saves the
        // check on more than half of all the steps a program takes.
        if (prof) {
            while (p.mode == Mode::Return && !p.park_requested) step_return_counted(p, 0);
        } else {
            while (p.mode == Mode::Return && !p.park_requested) step_return(p, 0);
        }

        // A blocking builtin asked to be parked; stop here and let the
        // scheduler complete the handshake.
        if (p.park_requested) return;
    }
}

bool force_whnf(Process& p, Value v, Value* out) {
    v = resolve(v);
    if (is_whnf(v)) {
        *out = v;
        return true;
    }
    return nested_whnf(p, v, nullptr, 0, out);
}

bool apply_whnf(Process& p, Value callee, const Value* args, uint32_t argc, Value* out) {
    return nested_whnf(p, callee, args, argc, out);
}

bool apply_whnf_taking(Process& p, Value callee, Value* args, uint32_t argc, Value* out) {
    return nested_whnf(p, callee, args, argc, out, args);
}

Value literal_string_value(Process& p, uint32_t index) { return literal_string(p, index); }

/// The nested machine loop behind `force_whnf` and `apply_whnf`: force `v`,
/// or -- when `args` is given -- apply `v` to them, which is the same loop
/// primed the way `prime_apply` primes a fresh process. The arguments go on the
/// value stack under an `ApplyTo`, where `do_apply` looks for them and where
/// the collector can see them.
bool nested_whnf(Process& p, Value v, const Value* args, uint32_t argc, Value* out,
                 Value* taken) {
    ForceNest nest(p);
    LoopStack loop(p, DREAM_FRAME_ADDRESS());
    // Run a nested machine loop down to the current continuation depth.
    //
    // Whether it may collect is decided here and nowhere else. The caller
    // either vouched for its own locals or did not (`VouchesForGc`); a force
    // nobody vouched for raises `force_pins` for as long as it runs, and a
    // collection needs every frame in the chain to have vouched, which is
    // exactly `force_pins == 0`. The flag is consumed rather than inherited:
    // what this force goes on to reach starts unvouched, because a vouch is a
    // claim about the frame that made it.
    const bool vouched = p.force_vouched;
    p.force_vouched = false;
    if (!vouched) ++p.force_pins;

    const Limits lim = Limits::get();
    const size_t max_conts = lim.conts;
    const size_t max_stack = lim.stack;
    const size_t max_heap = lim.heap_bytes;
    const WellKnownAtoms& wk = well_known(p.runtime());
    const bool prof = p.runtime().profiling_for(p.id());
    Jit* const work_jit = p.runtime().jit();
    const bool sample_work = work_jit && work_jit->samples_work();
    const size_t floor = p.conts.size();
    const Mode saved_mode = p.mode;
    const uint32_t saved_node = p.node;
    // The code that asked for this force may still be using its frame at the
    // depth this loop starts from, so the frames under it stay (`FrameStack`).
    struct FrameFloor {
        Process& p;
        size_t was;
        explicit FrameFloor(Process& q) : p(q), was(q.frames.raise_floor()) {}
        ~FrameFloor() { p.frames.restore_floor(was); }
    } frame_floor(p);

    // The machine state this loop is about to overwrite. It is two live
    // references, and a collection inside the loop would move both -- so they
    // are pinned rather than kept in a local, and read back below. Cheap, and
    // only on the path that actually runs a loop: a value already in WHNF
    // returned above without touching any of this.
    Pin saved_frame(p, p.frame);
    Pin saved_result(p, p.result);

    if (args) {
        for (uint32_t i = 0; i < argc; ++i) p.stack.push_back(args[i]);
        if (taken) std::fill(taken, taken + argc, UNIT);
        push_cont(p, ContKind::ApplyTo, argc, 0, 0, UNIT);
    }
    enter(p, v);
    bool ok = true;
    bool blocked = false;
    for (;;) {
        if (p.mode == Mode::Return && p.conts.size() <= floor) break;
        if (p.mode == Mode::Halted) break;
        if (p.mode == Mode::Raise) {
            if (!unwind(p, floor)) { ok = false; break; }
            continue;
        }
        if (p.mode == Mode::Eval) {
            --p.reductions;
            ++p.total_reductions;
            if (sample_work && (p.total_reductions & 63) == 0) {
                work_jit->note_work(current_func(p));
            }
            // Attributed like any other reduction. Leaving this out made the
            // profiler blind to most of a compile: `dreams` forces at every
            // stage boundary, so the machine spends the bulk of its life in
            // *this* loop rather than the outer one -- 179M of a self-compile's
            // 214M reductions -- and a profile that cannot see them names the
            // wrong functions.
            if (prof) {
                const uint32_t fi = current_func(p);
                const uint64_t a0 = p.heap().bytes_total();
                const uint64_t n0 = p.alloc_attributed;
                p.runtime().note_reduction(fi);
                if ((p.total_reductions & 1023) == 0 && timeline_on()) timeline_sample(p, fi);
                step_eval(p);
                note_own_alloc(p, fi, a0, n0);
            } else {
                step_eval(p);
            }
        } else if (prof) {
            step_return_counted(p, floor);
        } else {
            step_return(p, floor);
        }

        // The outer loop's limit check, and -- when the whole chain of frames
        // below `run_process` has vouched -- its safepoint too. Without the
        // limit check a runaway recursion reached through a native
        // (`console.print!` forcing a value that recurses for ever) grows until
        // the allocator throws, because this loop is where that forcing
        // happens. Without the safepoint a native that walks a long lazy
        // structure allocates for as long as the walk takes with nothing able
        // to collect any of it, which is the same failure a size away.
        if (check_limits(p, wk, max_conts, max_stack, max_heap)) continue;
        if (p.force_pins == 0 && p.heap().should_collect()) p.maybe_collect();
        // The slice ran out somewhere this loop cannot stop. See
        // `Process::slice_spent`: re-arm it, and let `run_process` end the
        // slice when this force finally returns.
        if (p.reductions <= 0) {
            p.slice_spent = true;
            p.reductions = p.slice;
            // A kill waits for the slice to end, and a force this long may not
            // end at all -- `strict!` over something endless, a native walking
            // a list that never stops. So the kill is raised here, where the
            // slice would have ended; `unwind` passes every handler, and the
            // natives above give the raise back as they would any other.
            if (p.mode != Mode::Raise && !p.kill_unwinding && p.cleanup_depth == 0 &&
                p.kill_requested.load(std::memory_order_relaxed)) {
                p.kill_unwinding = true;
                do_raise(p, p.kill_error());
                continue;
            }
        }

        // A blocking native inside the value being forced -- `join!`, `recv!`,
        // a read on a socket -- asked to be parked. This loop cannot park: it
        // is running underneath a native, whose C++ frame has to return before
        // the scheduler can touch the process. Carrying on would re-enter that
        // native for ever, which is a livelock, not a hang: the loop spins
        // without spending a reduction, so it never even yields its slice.
        //
        // So abandon the force and let the *outer* machine park instead. The
        // native above us returns Block, and is called again once the thing it
        // was waiting for has arrived.
        if (p.park_requested) {
            blocked = true;
            ok = false;
            break;
        }
    }

    // Both counters back as they were. The vouch is restored rather than left
    // consumed because a native that forces in a loop vouches once: taking it
    // away after the first force would quietly pin the rest of the walk, which
    // is the bug this whole mechanism exists to remove.
    if (!vouched) --p.force_pins;
    p.force_vouched = vouched;

    if (blocked) {
        // The force is *suspended*, not abandoned. Everything it pushed -- its
        // continuations, its half-forced blackholed thunks, its stack -- stays
        // exactly as it is, and the process resumes into it when the scheduler
        // wakes it. `force_resume_at` records where the outer operation's
        // retry belongs: underneath all of it.
        //
        // This used to unwind instead: un-blackhole the thunks, drop the
        // continuations, and let the outer native re-force from scratch. That
        // is only sound while forcing is pure, and it is not. A computation
        // that parks may already have performed effects, and re-forcing
        // performs them again -- `strict! { print! "x"; join! p }` printed
        // twice. Worse, `exec!` records its pending child in a single slot on
        // the process, so a re-force made the *first* call take the result the
        // *second* had started and begin another; that one never terminated.
        //
        // The machine state is left as the park set it (Return, on the
        // blocking native's own retry), so nothing below restores it.
        p.force_blocked = true;
        p.force_resume_at = floor;
        *out = p.result;
        return false;
    }

    *out = p.result;
    Value error = p.result;
    p.mode = saved_mode;
    p.node = saved_node;
    p.frame = saved_frame.get();
    if (ok) {
        p.result = saved_result.get();
    } else {
        p.result = error;
    }
    return ok;
}

bool force_deep(Process& p, Value v, Value* out) {
    Value head;
    if (!force_whnf(p, v, &head)) { *out = p.result; return false; }
    if (!is_ptr(head)) { *out = head; return true; }
    // Walked already, by this call or by an earlier one. See AUX_DEEP_FORCED:
    // this is what keeps a deep force linear in the data rather than in the
    // number of paths through it.
    if (as_obj(head)->aux & AUX_DEEP_FORCED) { *out = head; return true; }

    const size_t base = p.stack.size();
    auto unwind = [&](Value* out_) {
        if (!p.force_blocked) p.stack.resize(base);
        *out_ = p.result;
        return false;
    };

    switch (as_obj(head)->type) {
        case ObjType::Cons: {
            // A list is a chain and is walked as one. Recursing on the tail
            // would cost one C++ frame per element, and a list long enough to
            // be worth forcing -- the bytes of an image, say -- runs out of
            // stack long before it runs out of list.
            //
            // Every cell is kept on the value stack, both so the collector can
            // see them while the heads are being forced and so they can be
            // marked afterwards: a cell is deeply forced only once everything
            // after it is, so the marking runs backwards, from the end.
            p.stack.push_back(head);
            // The cell being walked waits on the value stack so that a
            // collection can find it and rewrite the reference -- which means
            // no C++ pointer to it may outlive a force. The object moves, and
            // a pointer taken before the force still names the block it moved
            // out of. So the cell is re-read from the stack at every single
            // use rather than once per turn round the loop, and this is the
            // only way to touch it.
            //
            // Storing through a stale one does not fault where it happens,
            // which is what made this expensive to find. It writes into
            // whatever that block has since been recycled into, and since
            // `ConsObj::head` and `ThunkObj::node` share an offset, the shape
            // the damage usually takes is a thunk whose node is the low half
            // of a pointer. The machine jumps to it some thousands of
            // reductions later and dies a long way from here.
            auto cell = [&] { return static_cast<ConsObj*>(as_obj(p.stack.back())); };
            for (;;) {
                Value tmp;
                if (!force_deep(p, cell()->head, &tmp)) {
                    return unwind(out);
                }
                value_slot_store(&cell()->head, tmp);
                p.heap().remember_if_old(cell(), tmp);

                Value tail;
                if (!force_whnf(p, cell()->tail, &tail)) {
                    return unwind(out);
                }
                value_slot_store(&cell()->tail, tail);
                p.heap().remember_if_old(cell(), tail);

                const bool more = is_ptr(tail) && as_obj(tail)->type == ObjType::Cons
                                  && !(as_obj(tail)->aux & AUX_DEEP_FORCED);
                if (more) {
                    p.stack.push_back(tail);
                    continue;
                }
                // What ends the chain is not always `[]`: a tail may hold any
                // value, and one that is not a cell still has to be forced.
                if (!is_ptr(tail) || as_obj(tail)->type != ObjType::Cons) {
                    Value done;
                    if (!force_deep(p, tail, &done)) return unwind(out);
                    auto* last = static_cast<ConsObj*>(as_obj(p.stack.back()));
                    value_slot_store(&last->tail, done);
                    p.heap().remember_if_old(last, done);
                }
                break;
            }
            // Nothing below allocates, so nothing moves: the cell at the
            // bottom of this run is still the one that was pushed.
            Value first = p.stack[base];
            while (p.stack.size() > base) {
                as_obj(p.stack.back())->aux |= AUX_DEEP_FORCED;
                p.stack.pop_back();
            }
            *out = first;
            return true;
        }
        case ObjType::Trunk: {
            // Each element forced where it lies: a forced suspension reads as
            // its value from every slot and diff that holds it, so nothing is
            // written back, and the version is the one it was.
            p.stack.push_back(head);
            const uint32_t n = trunk_length(head);
            for (uint32_t i = 0; i < n; ++i) {
                Value tmp;
                if (!force_deep(p, trunk_element(p.stack[base], i), &tmp)) return unwind(out);
            }
            as_obj(p.stack[base])->aux |= AUX_DEEP_FORCED;
            *out = p.stack[base];
            p.stack.resize(base);
            return true;
        }
        case ObjType::Array: {
            p.stack.push_back(head);
            // Re-read on both sides of the force, for the reason the list
            // above is: the array is on the value stack precisely so that a
            // collection may move it, and a pointer that spans the force is a
            // pointer to where it used to be. The length is safe to keep --
            // it is a number, and an array's does not change.
            auto arr = [&] { return static_cast<ArrayObj*>(as_obj(p.stack.back())); };
            const uint32_t len = arr()->len;
            for (uint32_t i = 0; i < len; ++i) {
                Value tmp;
                if (!force_deep(p, arr()->items()[i], &tmp)) return unwind(out);
                value_slot_store(&arr()->items()[i], tmp);
                p.heap().remember_if_old(arr(), tmp);
            }
            as_obj(p.stack.back())->aux |= AUX_DEEP_FORCED;
            *out = p.stack.back();
            p.stack.pop_back();
            return true;
        }
        case ObjType::Record: {
            // As an array, and for the same reason re-read across each force.
            p.stack.push_back(head);
            auto rec = [&] { return static_cast<RecordObj*>(as_obj(p.stack.back())); };
            const uint32_t len = rec()->len;
            for (uint32_t i = 0; i < len; ++i) {
                Value tmp;
                if (!force_deep(p, rec()->items()[i], &tmp)) return unwind(out);
                value_slot_store(&rec()->items()[i], tmp);
                p.heap().remember_if_old(rec(), tmp);
            }
            as_obj(p.stack.back())->aux |= AUX_DEEP_FORCED;
            *out = p.stack.back();
            p.stack.pop_back();
            return true;
        }
        case ObjType::Map: {
            p.stack.push_back(head);
            // The leaves are collected once and then forced in place. A leaf
            // may be shared with another version of the map, which is fine:
            // forcing a thunk yields the same value to everyone holding it.
            //
            // The values wait on the value stack rather than in a C++ vector,
            // for the reason every list above is walked there: forcing one of
            // them can collect, and a collection rewrites what it can see. A
            // vector of `Value` is not something it can see, so the entries
            // after the one being forced would be husks of objects that had
            // moved. That was safe only while a deep force could not collect,
            // and `strict!` vouching (see `bi_strict`) is exactly what makes
            // it collect.
            std::vector<std::pair<Value, Value>> entries;
            map_collect(p.stack.back(), entries);
            const size_t first = p.stack.size();
            for (auto& [k, val] : entries) {
                (void)k;
                p.stack.push_back(val);
            }
            const size_t count = entries.size();
            entries.clear();
            for (size_t i = 0; i < count; ++i) {
                Value tmp;
                if (!force_deep(p, p.stack[first + i], &tmp)) return unwind(out);
            }
            p.stack.resize(first);
            as_obj(p.stack.back())->aux |= AUX_DEEP_FORCED;
            *out = p.stack.back();
            p.stack.pop_back();
            return true;
        }
        case ObjType::Tensor: {
            // A deferred tensor's numbers are computed here, and kept in it:
            // `strict!` on a tensor means the arithmetic has been done, which
            // is what a program timing it or sharing it is asking for.
            Value err;
            if (!tensor_force(p, head, &err)) {
                p.result = err;
                *out = err;
                return false;
            }
            *out = head;
            return true;
        }
        default:
            *out = head;
            return true;
    }
}

bool jit_arith(Process& p, Op op, Value a, Value b, Value* out) {
    return arith(p, op, a, b, out);
}
bool jit_compare(Process& p, Op op, Value a, Value b, Value* out) {
    return compare(p, op, a, b, out);
}

/// What a record is, as `type_of` says it: the declaration's own name,
/// without the module it was declared in. `ship.item.Item.file` is an `Item`,
/// and `ship.item.Point` a `Point`.
std::string record_type_name(Process& p, Value v) {
    auto* r = static_cast<RecordObj*>(as_obj(resolve(v)));
    const std::string& id = p.runtime().atom_name(uint32_t(imm_payload(r->id())));
    size_t end = id.size();
    if (r->variant()) {
        size_t dot = id.rfind('.');
        if (dot != std::string::npos) end = dot;
    }
    size_t start = id.rfind('.', end == 0 ? 0 : end - 1);
    start = start == std::string::npos || start >= end ? 0 : start + 1;
    return id.substr(start, end - start);
}

/// How a record is written: `Item.file` for a variant, `Point` for a record.
std::string record_shown_name(Process& p, Value v) {
    auto* r = static_cast<RecordObj*>(as_obj(resolve(v)));
    std::string type = record_type_name(p, v);
    if (!r->variant()) return type;
    const std::string& id = p.runtime().atom_name(uint32_t(imm_payload(r->id())));
    return type + id.substr(id.rfind('.'));
}

std::string describe(Process& p, Value v) {
    v = resolve(v);
    if (is_fixnum(v)) return "integer " + std::to_string(fixnum_value(v));
    if (is_imm(v)) {
        switch (imm_kind(v)) {
            case IMM_UNIT: return "unit";
            case IMM_BOOL: return truthy(v) ? "bool true" : "bool false";
            case IMM_CHAR: return "a char";
            case IMM_ATOM: return ":" + p.runtime().atom_name(uint32_t(imm_payload(v)));
            case IMM_NIL: return "the empty list";
            case IMM_BUILTIN: return "a builtin function";
            default: return "a value";
        }
    }
    if (!is_ptr(v)) return "a value";
    return std::string("a ") + obj_type_name(as_obj(v)->type);
}

std::string image_function_name(Process& p, uint32_t fi) { return function_name(p, fi); }

bool equal_values(Process& p, Value a, Value b, bool* raised, int depth) {
    return values_equal(p, a, b, raised, depth);
}

}  // namespace dream
