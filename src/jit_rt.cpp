// Slow paths for JIT-compiled code. These are the interpreter's own helpers,
// so a compiled function and an interpreted one cannot disagree about what an
// operation means.

#include "jit_rt.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <string>

#include "bigint.hpp"
#include "builtins.hpp"
#include "interp.hpp"
#include "tensor.hpp"
#include "process.hpp"
#include "runtime.hpp"

using namespace dream;

extern "C" {

// A compiled frame holds its slots in registers the collector cannot find
// precisely. Underneath a force or a call, which is where a compiled loop can
// spend unbounded time, that no longer stops a collection: the machine stack
// is scanned instead, and whatever it points at stays where it is
// (`Process::jit_stack_hi`, `Heap::set_conservative_roots`). So those two
// vouch, through `ScannedForGc`. Every other helper below that can run the
// machine still pins the heap and says so, because what it reaches holds
// values somewhere a scan of the stack would not see: `arith` on two lists is
// `concat_lists`, and a native is whatever it is.
namespace {

/// Vouch for a nested force under compiled code, when the stack will be
/// scanned -- which it is whenever compiled code is running from a machine
/// loop. Outside one there is nothing to scan up to, and it pins instead.
struct ScannedForGc {
    Process& p;
    const bool scanned;
    const bool saved;
    explicit ScannedForGc(Process& proc)
        : p(proc), scanned(proc.jit_stack_hi != nullptr), saved(proc.force_vouched) {
        if (scanned) p.force_vouched = true; else ++p.force_pins;
    }
    ~ScannedForGc() {
        if (scanned) p.force_vouched = saved; else --p.force_pins;
    }
    ScannedForGc(const ScannedForGc&) = delete;
    ScannedForGc& operator=(const ScannedForGc&) = delete;
};

}  // namespace

int dream_rt_force(Process* p, Value v, Value* out) {
    ScannedForGc scanned(*p);
    Value result;
    if (!force_whnf(*p, v, &result)) {
        *out = p->result;
        return 0;
    }
    *out = result;
    return 1;
}

int dream_rt_switch_head(Process* p, Value v, Value* out) {
    v = resolve(v);
    // As `take_switch`: a list cell's head, or an array's first element.
    if (is_obj(v, ObjType::Cons)) return dream_rt_force(p, static_cast<ConsObj*>(as_obj(v))->head, out);
    if (is_obj(v, ObjType::Array) && static_cast<ArrayObj*>(as_obj(v))->len > 0) {
        return dream_rt_force(p, static_cast<ArrayObj*>(as_obj(v))->items()[0], out);
    }
    *out = UNIT;
    return 1;
}

int dream_rt_arith(Process* p, int32_t op, Value a, Value b, Value* out) {
    PinsTheHeap pinned(*p);
    return jit_arith(*p, Op(op), a, b, out) ? 1 : 0;
}

int dream_rt_compare(Process* p, int32_t op, Value a, Value b, Value* out) {
    PinsTheHeap pinned(*p);
    return jit_compare(*p, Op(op), a, b, out) ? 1 : 0;
}

Value dream_rt_float(Process* p, double d) { return p->heap().make_float(d); }

namespace {
/// The double a number in hand stands for. Only ever asked of a value `arith`
/// has just answered with an operand that was a float, which makes the answer
/// a float box, or a tensor, which is the one thing this declines.
inline bool number_as_double(Value v, double* out) {
    if (is_obj(v, ObjType::Float)) {
        *out = static_cast<FloatObj*>(as_obj(v))->value;
        return true;
    }
    if (bigint::is_integer(v)) {
        *out = bigint::to_double(v);
        return true;
    }
    return false;
}
}  // namespace

int dream_rt_arith_f(Process* p, int32_t op, Value a, Value b, double* out, Value* err) {
    PinsTheHeap pinned(*p);
    Value r;
    if (!jit_arith(*p, Op(op), a, b, &r)) {
        *err = r;
        return 0;
    }
    // An operation with a float on one side of it answers a float or raises
    // -- unless the other side is a tensor, where `2.0 * t` is a tensor. The
    // compiled code holds that answer in a register typed double and cannot
    // take anything else, so it gives the call back to the interpreter (2,
    // which it turns into `JIT_BAIL`). The body is pure, so running it again
    // from its frame is the answer it would have given.
    if (!number_as_double(r, out)) return 2;
    return 1;
}

int dream_rt_to_int(Process* p, Value v, Value* out) {
    if (bigint::is_integer(v)) {
        *out = v;
        return 1;
    }
    if (is_obj(v, ObjType::Float)) {
        if (bigint::from_double(p->heap(), static_cast<FloatObj*>(as_obj(v))->value, out)) return 1;
        *out = raise_error(*p, well_known(p->runtime()).type_error,
                           "to_int of an infinity or a NaN is not an integer");
        return 0;
    }
    *out = raise_error(*p, well_known(p->runtime()).type_error, "to_int needs a number");
    return 0;
}

int dream_rt_floor(Process* p, Value v, Value* out) {
    if (bigint::is_integer(v)) {
        *out = v;
        return 1;
    }
    if (is_obj(v, ObjType::Float)) {
        if (bigint::from_double(p->heap(), std::floor(static_cast<FloatObj*>(as_obj(v))->value),
                                out)) {
            return 1;
        }
        *out = raise_error(*p, well_known(p->runtime()).type_error,
                           "floor of an infinity or a NaN is not an integer");
        return 0;
    }
    *out = raise_error(*p, well_known(p->runtime()).type_error, "floor needs a number");
    return 0;
}

int dream_rt_abs(Process* p, Value v, Value* out) {
    if (is_fixnum(v)) {
        int64_t n = fixnum_value(v);
        *out = make_integer(*p, n < 0 ? -n : n);
        return 1;
    }
    if (bigint::is_big(v)) {
        *out = bigint::abs(p->heap(), v);
        return 1;
    }
    if (is_obj(v, ObjType::Float)) {
        *out = p->heap().make_float(std::fabs(static_cast<FloatObj*>(as_obj(v))->value));
        return 1;
    }
    *out = raise_error(*p, well_known(p->runtime()).type_error, "abs needs a number");
    return 0;
}

int dream_rt_int_of_double(Process* p, double d, Value* out) {
    if (bigint::from_double(p->heap(), d, out)) return 1;
    *out = raise_error(*p, well_known(p->runtime()).type_error,
                       "an infinity or a NaN is not an integer");
    return 0;
}

int dream_rt_number_double(Process* p, Value v, double* out) {
    if (bigint::is_big(v)) {
        *out = bigint::to_double(v);
        return 1;
    }
    return 0;
}

Value dream_rt_type_error(Process* p, const char* message) {
    return raise_error(*p, well_known(p->runtime()).type_error, message);
}

Value dream_rt_type_error_got(Process* p, const char* message, Value v) {
    return raise_error(*p, well_known(p->runtime()).type_error,
                       std::string(message) + ", got " + describe(*p, v));
}

int64_t* dream_rt_reduction_slot(Process* p) { return &p->reductions; }

uint64_t dream_rt_stack_floor(Process* p) {
    return p->compiled_stack_floor();
}

Value* dream_rt_frame_slots(Value frame) {
    return static_cast<FrameObj*>(as_obj(frame))->slots();
}


// ---------------------------------------------------------------------------
// Calling a host native
//
// What licenses this is written where the decision is made, in `native_site` in
// jit.cpp: only a *pure* native of fixed arity is admitted, which is what makes
// the shape below enough. A pure native performs no effect, so nothing here has
// to be ordered against anything; and it cannot park, because the natives that
// park are every one of them impure and the Dream work a pure one runs
// underneath itself is pure too. `Block` is answered all the same, in the one
// way that is honest about not having handled it.
//
// The heap is pinned for the duration. Not for the compiled frame above --
// a scan of the machine stack covers that now, see `ScannedForGc` -- but for
// the native below, which may hold a value where no scan of the stack looks.
// A native that vouches for itself still collects underneath, as it would
// interpreted; one that does not walks a long lazy structure with nothing able
// to collect any of it. See `PinsTheHeap`.
// ---------------------------------------------------------------------------

namespace {

/// Run `fn` as the native `callee`, with `argc` arguments already in hand.
///
/// The arguments are pushed onto the process's value stack rather than passed
/// from an array of this frame's own, because that is where a native looks for
/// them: the interpreter hands one `p.stack.data() + base`, and a native that
/// grows the stack underneath itself -- which several do, to hold a list across
/// a force -- is written against exactly that. Doing it the same way here means
/// no native has to know which tier called it.
int run_native(Process& p, Value callee, NativeFn fn, uint32_t argc, const Value* args,
               Value* out) {
    PinsTheHeap pinned(p);
    const size_t base = p.stack.size();
    for (uint32_t i = 0; i < argc; ++i) p.stack.push_back(args[i]);

    NativeResult r = fn(p, callee, p.stack.data() + base, argc);
    // Not when a force inside the native was suspended: its work is on the
    // stack above the arguments, and `enter_function` retries the call.
    if (!p.park_requested) p.stack.resize(base);

    switch (r.outcome) {
        case NativeOutcome::Value:
            *out = r.value;
            return 1;
        case NativeOutcome::Raise:
            *out = r.value;
            return 0;
        case NativeOutcome::Enter: {
            // The native found something and did not force it. The machine
            // forces such a value as the continuation of the call, so forcing
            // it here is the same point in the program -- what differs is that
            // this is a nested loop on the C++ stack where the machine's is a
            // heap continuation. `Process::force_nest` is what bounds that, and
            // `kMaxForceNestForCompiled` is where the interpreter takes over.
            //
            // Pinned rather than held in a local on principle: nothing can
            // collect underneath `PinsTheHeap`, but a value that crosses a
            // force belongs somewhere the collector rewrites, and the cost of
            // saying so is a push and a pop.
            Pin keep(p, r.value);
            Value w;
            if (!force_whnf(p, keep.get(), &w)) {
                *out = p.result;
                return 0;
            }
            *out = w;
            return 1;
        }
        case NativeOutcome::Block:
            break;
    }

    // The native parked the process, which compiled code has no way to be
    // resumed into: there is no continuation to push, because the body it would
    // resume is machine code half way through a register allocation. It is
    // unreachable for an admitted native -- only an impure one blocks -- and the
    // answer here is the one `dream_rt_force` gives for the same situation,
    // which is to report what the process is holding as a raise rather than to
    // pretend the call produced a value.
    *out = p.result;
    return 0;
}

}  // namespace

int dream_rt_native(Process* p, uint32_t module_index, uint32_t member_index, uint32_t argc,
                    Value a0, Value a1, Value a2, Value a3, Value* out) {
    const ModuleDef& def = p->runtime().modules()[module_index];
    const NativeDef& m = def.members[member_index];

    // The callee value this process hands out for this member, which is the one
    // `Field` would have produced: built once and cached, because an FFI binding
    // reads its own `user` field back out of it and because rebuilding it would
    // allocate a name and an object per call. The cache is indexed by the
    // runtime-wide member id, exactly as `member_value_slow` indexes it.
    const uint32_t id = def.member_base + member_index;
    if (p->native_cache.empty()) {
        p->native_cache.assign(p->runtime().native_member_count(), NIL_SLOT);
    }
    Value callee = id < p->native_cache.size() ? p->native_cache[id] : NIL_SLOT;
    if (callee == NIL_SLOT) {
        Value name = p->heap().make_string(m.name, uint32_t(std::strlen(m.name)));
        callee = make_native(*p, m.fn, name, m.arity, m.strict_mask, m.user);
        if (id < p->native_cache.size()) p->native_cache[id] = callee;
    }

    const Value args[4] = {a0, a1, a2, a3};
    return run_native(*p, callee, m.fn, argc, args, out);
}

int dream_rt_builtin(Process* p, uint32_t builtin_id, uint32_t argc, Value a0, Value a1, Value a2,
                     Value a3, Value* out) {
    // A builtin is an immediate: it names itself, and there is nothing to build
    // or to cache.
    const BuiltinDef& bd = builtin_def(builtin_id);
    const Value args[4] = {a0, a1, a2, a3};
    return run_native(*p, make_builtin(builtin_id), bd.fn, argc, args, out);
}

// ---------------------------------------------------------------------------
// Lists, arrays and maps
//
// Each of these is the matching case of `container_get`, `container_set` or
// `step_eval` in interp.cpp, with the machine's continuation replaced by a
// nested force where the interpreter would have pushed one. The messages are
// the interpreter's, word for word, because the e2e harness compares the two
// tiers' output and an error is output.
//
// Every one pins the heap: what each runs -- a map's lookup, a list's walk --
// holds values in C++ locals across a force, and has not been audited for a
// collection underneath it. `dream_rt_force` and `dream_rt_apply` are the two
// that may collect; see `ScannedForGc`.
// ---------------------------------------------------------------------------

namespace {

inline bool is_sequence_value(Value v) {
    return is_obj(v, ObjType::Array) || is_nil(v) || is_obj(v, ObjType::Cons);
}

inline Value rt_type_error(Process& p, const std::string& message) {
    return raise_error(p, well_known(p.runtime()).type_error, message);
}

inline Value rt_out_of_bounds(Process& p, const std::string& message) {
    return raise_error(p, well_known(p.runtime()).out_of_bounds, message);
}

}  // namespace

int dream_rt_get(Process* p, Value c, Value k, int32_t has_default, Value* out) {
    PinsTheHeap pinned(*p);
    c = resolve(c);
    k = resolve(k);
    if (is_obj(c, ObjType::Map)) {
        Value found;
        if (map_lookup(*p, c, k, &found)) {
            *out = found;
            return 1;
        }
        if (has_default) return 2;
        *out = raise_error(*p, well_known(p->runtime()).no_such_key,
                           "the map has no key " + describe(*p, k));
        return 0;
    }
    if (is_obj(c, ObjType::Tensor) && is_fixnum(k)) {
        switch (tensor_index(*p, c, fixnum_value(k), out)) {
            case 1: return 1;
            case 0: return 0;
            default: break;
        }
        if (has_default) return 2;
        *out = rt_out_of_bounds(*p, "index " + std::to_string(fixnum_value(k)) +
                                        " is outside a tensor of length " +
                                        std::to_string(tensor_len(c)));
        return 0;
    }
    if (is_obj(c, ObjType::Record) && is_fixnum(k)) {
        auto* r = static_cast<RecordObj*>(as_obj(c));
        const int64_t i = fixnum_value(k);
        if (i >= 0 && i < int64_t(r->len)) {
            *out = r->items()[i];
            return 1;
        }
        if (has_default) return 2;
        *out = rt_out_of_bounds(*p, "field " + std::to_string(i) + " is outside " + describe(*p, c));
        return 0;
    }
    if (is_obj(c, ObjType::Trunk) && is_fixnum(k)) {
        const int64_t i = fixnum_value(k);
        if (i >= 0 && i < int64_t(trunk_length(c))) {
            *out = trunk_element(c, uint32_t(i));
            return 1;
        }
        if (has_default) return 2;
        *out = rt_out_of_bounds(*p, "index " + std::to_string(i) + " is outside a trunk of " +
                                        std::to_string(trunk_length(c)));
        return 0;
    }
    if (!is_sequence_value(c)) {
        *out = rt_type_error(*p, "`.[ ]` reads a map, an array, a list or a trunk, not " +
                                     describe(*p, c));
        return 0;
    }
    if (!is_fixnum(k)) {
        *out = rt_type_error(*p, "a position is an integer, not " + describe(*p, k));
        return 0;
    }
    const int64_t i = fixnum_value(k);
    if (is_obj(c, ObjType::Array)) {
        auto* a = static_cast<ArrayObj*>(as_obj(c));
        if (i >= 0 && i < int64_t(a->len)) {
            *out = a->items()[i];
            return 1;
        }
        if (has_default) return 2;
        *out = rt_out_of_bounds(*p, "index " + std::to_string(i) + " is outside an array of " +
                                        std::to_string(a->len));
        return 0;
    }
    if (i < 0 || i > int64_t(0xFFFFFFFFu)) {
        if (has_default) return 2;
        *out = rt_out_of_bounds(*p, "index " + std::to_string(i) + " is not a position in a list");
        return 0;
    }
    // The walk `list_get` makes, forcing each tail it has to step over. Nothing
    // can collect underneath the pin, so a cell pointer taken before a force is
    // still the cell after it.
    const uint32_t index = uint32_t(i);
    Value cur = c;
    for (uint32_t walked = 0;; ++walked) {
        cur = resolve(cur);
        if (!is_obj(cur, ObjType::Cons)) {
            if (has_default) return 2;
            *out = rt_out_of_bounds(*p, "index " + std::to_string(index) +
                                            " is past the end of a list of " +
                                            std::to_string(walked));
            return 0;
        }
        auto* cell = static_cast<ConsObj*>(as_obj(cur));
        if (walked == index) {
            *out = cell->head;
            return 1;
        }
        Value tail = resolve(cell->tail);
        if (!is_whnf(tail) && !force_whnf(*p, tail, &tail)) {
            *out = p->result;
            return 0;
        }
        cur = tail;
    }
}

int dream_rt_set_many(Process* p, Value c, uint32_t count, const Value* pairs, Value* out) {
    PinsTheHeap pinned(*p);
    c = resolve(c);
    const bool record = is_obj(c, ObjType::Record);
    if (record || is_obj(c, ObjType::Array)) {
        const uint32_t len = record ? static_cast<RecordObj*>(as_obj(c))->len
                                    : static_cast<ArrayObj*>(as_obj(c))->len;
        // The messages `container_set` and `dream_rt_set` raise, at the key
        // the chain of `Set`s would have raised them at.
        for (uint32_t i = 0; i < count; ++i) {
            Value k = resolve(pairs[2 * i]);
            if (!is_fixnum(k)) {
                *out = rt_type_error(*p, record ? "a field of a record is changed by position, not " +
                                                      describe(*p, k)
                                                : "a position is an integer, not " + describe(*p, k));
                return 0;
            }
            const int64_t at = fixnum_value(k);
            if (at < 0 || at >= int64_t(len)) {
                *out = rt_out_of_bounds(*p, record ? "field " + std::to_string(at) + " is outside " +
                                                         describe(*p, c)
                                                   : "index " + std::to_string(at) +
                                                         " is outside an array of " + std::to_string(len));
                return 0;
            }
        }
        Value copy = record ? p->heap().make_record(static_cast<RecordObj*>(as_obj(c))->tag, len)
                            : p->heap().make_array(len);
        Value* dst = record ? static_cast<RecordObj*>(as_obj(copy))->items()
                            : static_cast<ArrayObj*>(as_obj(copy))->items();
        const Value* src = record ? static_cast<RecordObj*>(as_obj(c))->items()
                                  : static_cast<ArrayObj*>(as_obj(c))->items();
        Obj* holder = as_obj(copy);
        for (uint32_t i = 0; i < len; ++i) {
            dst[i] = src[i];
            p->heap().remember_if_old(holder, src[i]);
        }
        for (uint32_t i = 0; i < count; ++i) {
            const Value v = pairs[2 * i + 1];
            dst[fixnum_value(resolve(pairs[2 * i]))] = v;
            p->heap().remember_if_old(holder, v);
        }
        *out = copy;
        return 1;
    }
    if (is_obj(c, ObjType::Map)) {
        *out = map_insert_many(*p, c, count, pairs);
        return 1;
    }
    // A list, or something `Set` refuses: a pair at a time, which is `Set`
    // exactly.
    for (uint32_t i = 0; i < count; ++i) {
        if (!dream_rt_set(p, c, pairs[2 * i], pairs[2 * i + 1], &c)) {
            *out = c;
            return 0;
        }
    }
    *out = c;
    return 1;
}

int dream_rt_set(Process* p, Value c, Value k, Value v, Value* out) {
    PinsTheHeap pinned(*p);
    c = resolve(c);
    k = resolve(k);
    if (is_obj(c, ObjType::Map)) {
        *out = map_insert(*p, c, k, v);
        return 1;
    }
    if (is_obj(c, ObjType::Record) && is_fixnum(k)) {
        const uint32_t n = static_cast<RecordObj*>(as_obj(c))->len;
        const int64_t i = fixnum_value(k);
        if (i < 0 || i >= int64_t(n)) {
            *out = rt_out_of_bounds(*p, "field " + std::to_string(i) + " is outside " + describe(*p, c));
            return 0;
        }
        auto* src = static_cast<RecordObj*>(as_obj(c));
        Value copy = p->heap().make_record(src->tag, n);
        auto* dst = static_cast<RecordObj*>(as_obj(copy));
        for (uint32_t j = 0; j < n; ++j) {
            Value item = uint32_t(i) == j ? v : src->items()[j];
            dst->items()[j] = item;
            p->heap().remember_if_old(dst, item);
        }
        *out = copy;
        return 1;
    }
    if (!is_sequence_value(c)) {
        *out = rt_type_error(*p, "`.[ => ]` changes a map, an array or a list, not " +
                                     describe(*p, c));
        return 0;
    }
    if (!is_fixnum(k)) {
        *out = rt_type_error(*p, "a position is an integer, not " + describe(*p, k));
        return 0;
    }
    const int64_t i = fixnum_value(k);
    if (is_obj(c, ObjType::Array)) {
        const uint32_t n = static_cast<ArrayObj*>(as_obj(c))->len;
        if (i < 0 || i >= int64_t(n)) {
            *out = rt_out_of_bounds(*p, "index " + std::to_string(i) + " is outside an array of " +
                                            std::to_string(n));
            return 0;
        }
        Value copy = p->heap().make_array(n);
        auto* src = static_cast<ArrayObj*>(as_obj(c));
        auto* dst = static_cast<ArrayObj*>(as_obj(copy));
        for (uint32_t j = 0; j < n; ++j) {
            Value item = uint32_t(i) == j ? v : src->items()[j];
            dst->items()[j] = item;
            p->heap().remember_if_old(dst, item);
        }
        *out = copy;
        return 1;
    }
    if (i < 0 || i > int64_t(0xFFFFFFFFu)) {
        *out = rt_out_of_bounds(*p, "index " + std::to_string(i) + " is not a position in a list");
        return 0;
    }
    // `list_set`: the cells in front of the one replaced are new, everything
    // behind it is shared. The cells passed wait on the value stack, which is
    // where the interpreter keeps them too.
    const uint32_t index = uint32_t(i);
    const size_t base = p->stack.size();
    Value cur = c;
    for (uint32_t walked = 0;; ++walked) {
        cur = resolve(cur);
        if (!is_obj(cur, ObjType::Cons)) {
            p->stack.resize(base);
            *out = rt_out_of_bounds(*p, "index " + std::to_string(index) +
                                            " is past the end of a list of " +
                                            std::to_string(walked));
            return 0;
        }
        if (walked == index) {
            Value built = p->heap().make_cons(v, static_cast<ConsObj*>(as_obj(cur))->tail);
            while (p->stack.size() > base) {
                Value head = static_cast<ConsObj*>(as_obj(p->stack.back()))->head;
                built = p->heap().make_cons(head, built);
                p->stack.pop_back();
            }
            *out = built;
            return 1;
        }
        p->stack.push_back(cur);
        Value tail = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
        if (!is_whnf(tail) && !force_whnf(*p, tail, &tail)) {
            p->stack.resize(base);
            *out = p->result;
            return 0;
        }
        cur = tail;
    }
}

Value dream_rt_cons(Process* p, Value head, Value tail) { return p->heap().make_cons(head, tail); }

Value dream_rt_cons_after(Process* p, Value last, Value head) {
    Value cell = p->heap().make_cons(head, NIL);
    if (last != 0) dream_rt_set_tail(p, last, cell);
    return cell;
}

void dream_rt_set_tail(Process* p, Value cell, Value tail) {
    auto* c = static_cast<ConsObj*>(as_obj(cell));
    c->tail = tail;
    p->heap().remember_if_old(c, tail);
}

Value dream_rt_make_list(Process* p, uint32_t n, const Value* items) {
    Value list = NIL;
    for (uint32_t i = n; i > 0; --i) list = p->heap().make_cons(items[i - 1], list);
    return list;
}

Value dream_rt_make_array(Process* p, uint32_t n, const Value* items) {
    // An array large enough to be born old is remembered at birth by the
    // allocator, so filling it needs no barrier (see `alloc_bare`).
    Value arr = p->heap().make_array(n);
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    for (uint32_t i = 0; i < n; ++i) a->items()[i] = items[i];
    return arr;
}

Value dream_rt_make_record(Process* p, uint32_t id, uint32_t n, const Value* items) {
    // `id` is the node's `c`: an image atom index, and the variant bit.
    Value rec = p->heap().make_record(
        p->runtime().image_atom(id & RECORD_MAKE_ATOM) | (id & RECORD_MAKE_VARIANT), n);
    auto* r = static_cast<RecordObj*>(as_obj(rec));
    for (uint32_t i = 0; i < n; ++i) r->items()[i] = items[i];
    return rec;
}

/// `%{}`: what `MakeMap` with no entries answers, a fresh empty map.
Value dream_rt_make_map(Process* p) { return p->heap().make_map(0); }

Value dream_rt_literal_str(Process* p, uint32_t index) { return literal_string_value(*p, index); }

Value dream_rt_global(Process* p, uint32_t index) { return global_value(*p, index); }

Value dream_rt_snapshot(Process* p, Value closure, uint32_t nslots, const Value* vals,
                        const uint32_t* binds) {
    Value fr = p->heap().make_frame_filling(closure, nslots, nslots);
    auto* fo = static_cast<FrameObj*>(as_obj(fr));
    for (uint32_t i = 0; i < nslots; ++i) fo->slots()[i] = vals[i];
    // After the plain slots, because a binding's thunk is against this very
    // frame: it reads the parameters out of it when it is forced. A binding
    // compiled code has already computed arrives as its value, and stays.
    for (uint32_t i = 0; i < nslots; ++i) {
        if (binds[i] != NO_NODE && vals[i] == NIL_SLOT) {
            fo->slots()[i] = p->heap().make_thunk(binds[i], fr);
        }
    }
    return fr;
}

Value dream_rt_yield_frame(Process* p, Value frame, uint32_t n, const Value* vals) {
    auto* old = static_cast<FrameObj*>(as_obj(frame));
    const uint32_t nslots = old->nslots;
    const Value closure = old->closure;
    Value fr = p->heap().make_frame_filling(closure, nslots, 0);
    auto* fo = static_cast<FrameObj*>(as_obj(fr));
    for (uint32_t i = 0; i < n && i < nslots; ++i) fo->slots()[i] = vals[i];
    return fr;
}

void dream_rt_settle(Process* p, Value frame, uint32_t slot, Value value) {
    // The binding was suspended in this frame while compiled code was still
    // computing it, and the computation has now finished. The thunk is
    // updated exactly as forcing it would have updated it -- in place, so
    // whatever already holds it sees the answer -- and is never run. One
    // already forced, or being forced, is left alone: either it holds this
    // answer or it is about to.
    Value v = static_cast<FrameObj*>(as_obj(frame))->slots()[slot];
    if (!is_obj(v, ObjType::Thunk)) return;
    Obj* o = as_obj(v);
    p->heap().remember_if_old(o, value);
    std::atomic_ref<Value>(static_cast<IndirectObj*>(o)->target).store(value, std::memory_order_relaxed);
    obj_type_store_release(o, ObjType::Indirect);
}

Value dream_rt_adopt(Process* p, Value frame, uint32_t nslots, const Value* vals,
                     const uint32_t* binds) {
    // The interpreter's frame for this call, taken over as the snapshot. What
    // it lacks is what running the body would have written by now: a thunk in
    // each binding's slot. Only into a slot still empty -- one the body has
    // not written is all there can be before the first back-edge -- and through
    // the barrier, because this is a frame compiled code did not make.
    //
    // A frame on the process's stack dies with the call, and the suspensions
    // made against a snapshot outlive it, so that one is copied to the heap
    // first and the copy is the snapshot. The stack frame stays as the call
    // left it, which is what a bail back into the interpreter runs in.
    if (as_obj(frame)->gc & GC_SHARED) frame = dream::heap_frame(*p, frame);
    auto* fo = static_cast<FrameObj*>(as_obj(frame));
    for (uint32_t i = 0; i < nslots && i < fo->nslots; ++i) {
        if (binds[i] == NO_NODE || fo->slots()[i] != NIL_SLOT) continue;
        // A binding compiled code has already computed goes in as its value.
        Value th = vals[i] != NIL_SLOT ? vals[i] : p->heap().make_thunk(binds[i], frame);
        p->heap().remember_if_old(fo, th);
        value_slot_store(&fo->slots()[i], th);
    }
    return frame;
}

Value dream_rt_thunk(Process* p, uint32_t node, Value frame) {
    return p->heap().make_thunk(node, frame);
}

namespace {

/// Would applying `v` perform an effect? Purity is spelling: a function record
/// carries the `!` of its name as `FN_IMPURE`, and a native carries it in its
/// name. A partial application is its function's.
bool callee_is_impure(Process& p, Value v) {
    v = resolve(v);
    for (int depth = 0; depth < 64; ++depth) {
        if (is_obj(v, ObjType::Pap)) {
            v = resolve(static_cast<PapObj*>(as_obj(v))->fn);
            continue;
        }
        if (is_obj(v, ObjType::Closure)) {
            // A lambda is never impure by spelling -- it has no name to spell
            // it with -- but one written inside an impure function may perform
            // effects all the same, and the compiler marks its body for it.
            // Run from here, such a lambda's effects happen and are then
            // performed again when a park or a bail retries the call from the
            // top: a fold whose function spawned a process spawned it forever.
            const FuncRec& f = p.code->func(static_cast<ClosureObj*>(as_obj(v))->func);
            if (f.flags & FN_IMPURE) return true;
            return f.body != NO_NODE && (p.code->node(f.body).flags & F_IMPURE) != 0;
        }
        if (is_builtin(v)) {
            const char* name = builtin_def(uint32_t(imm_payload(v))).name;
            const size_t n = std::strlen(name);
            return n > 0 && name[n - 1] == '!';
        }
        if (is_obj(v, ObjType::Native)) {
            Value name = static_cast<NativeObj*>(as_obj(v))->name;
            if (!is_obj(name, ObjType::Str)) return true;
            auto* s = static_cast<StrObj*>(as_obj(name));
            return s->len > 0 && s->data()[s->len - 1] == '!';
        }
        return false;
    }
    return true;
}

}  // namespace

int dream_rt_apply(Process* p, Value callee, uint32_t argc, Value* args, Value* out) {
    // Compiled code orders no effects, and a call it makes is retried from the
    // top if something under it parks -- so an impure function must never run
    // from here. A pure caller is handed one only through a parameter, which
    // the purity check cannot see through, and that is rare enough that the
    // answer is to give the whole call back (2) before anything happens: the
    // compiled body has performed no effect, since this is the only way it
    // could have, and the interpreter runs it from its frame.
    if (callee_is_impure(*p, callee)) return 2;
    ScannedForGc scanned(*p);
    // A nested loop that runs the slice out re-arms it rather than stopping
    // (`Process::slice_spent`), which is right for a native and wrong for a
    // compiled loop around this call: it would see a full budget at its next
    // back-edge and carry on under the pin, collecting nothing, for as long as
    // the calls kept re-arming it. So a slice spent in here is left spent, and
    // the loop yields at its next back-edge. Only one spent *here*: under a
    // force that had already spent the slice, doing this would make every
    // iteration yield.
    const bool spent_before = p->slice_spent;
    Value r;
    const bool ok = apply_whnf_taking(*p, callee, args, argc, &r);
    if (!spent_before && p->slice_spent && p->reductions > 0) p->reductions = 0;
    if (!ok) {
        *out = p->result;
        return 0;
    }
    *out = r;
    return 1;
}

}  // extern "C"

bool dream::impure_callee(Process& p, Value v) { return callee_is_impure(p, v); }

extern "C" Value dream_rt_peek(Value c, Value k) {
    // `thunk_for`'s `Get`, and nothing more: an element found without forcing
    // anything, or zero where it would have suspended the read.
    c = resolve(c);
    k = resolve(k);
    if (!is_fixnum(k)) return NIL_SLOT;
    const int64_t i = fixnum_value(k);
    if (i < 0) return NIL_SLOT;
    if (is_obj(c, ObjType::Array)) {
        auto* a = static_cast<ArrayObj*>(as_obj(c));
        return i < int64_t(a->len) ? a->items()[i] : NIL_SLOT;
    }
    if (is_obj(c, ObjType::Record)) {
        auto* r = static_cast<RecordObj*>(as_obj(c));
        return i < int64_t(r->len) ? r->items()[i] : NIL_SLOT;
    }
    if (!is_obj(c, ObjType::Cons) || i > 8) return NIL_SLOT;
    Value cur = c;
    for (int64_t step = 0; step < i; ++step) {
        Value tail = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
        if (!is_obj(tail, ObjType::Cons)) return NIL_SLOT;
        cur = tail;
    }
    return static_cast<ConsObj*>(as_obj(cur))->head;
}
