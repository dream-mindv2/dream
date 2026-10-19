// The runtime interface JIT-compiled code calls back into.
//
// Compiled code handles the fixnum fast paths inline and calls these for
// everything else, so the two tiers agree on semantics by construction: the
// slow paths are literally the interpreter's.

#pragma once

#include <cstdint>

#include "value.hpp"

namespace dream {
class Process;
}

extern "C" {

/// Force to weak head normal form. Returns 1 on success, 0 if it raised (in
/// which case *out holds the error).
int dream_rt_force(dream::Process* p, dream::Value v, dream::Value* out);

/// Force a cons head for switch_head, with the same success/error convention.
/// Non-cons subjects produce UNIT, which dispatches to the default arm.
int dream_rt_switch_head(dream::Process* p, dream::Value v, dream::Value* out);

/// Arithmetic and comparison slow paths: floats, strings, overflow, and the
/// type errors. `op` is the image opcode.
int dream_rt_arith(dream::Process* p, int32_t op, dream::Value a, dream::Value b, dream::Value* out);
int dream_rt_compare(dream::Process* p, int32_t op, dream::Value a, dream::Value b, dream::Value* out);

/// Allocate a float box. Allocation itself never collects -- a collection runs
/// only at a safepoint -- so this is safe to call with the compiled frame's
/// values in machine registers, which is what makes an unboxed loop able to box
/// its answer on the way out.
dream::Value dream_rt_float(dream::Process* p, double d);

/// Arithmetic whose result is known to be a float, answered as a raw double.
///
/// The unboxed path takes this when one of its operands is not statically a
/// float and turns out at runtime not to be a number in hand either. `arith`
/// then either raises -- which is the usual outcome, and the error is the same
/// one the interpreter would have produced -- or answers a number, because an
/// operation with a float box on one side of it and a number on the other is
/// the double path in `arith` and nothing else. So the result unboxes, and the
/// caller stays in doubles rather than having to rejoin two representations.
///
/// Returns 1 with the double in `*out`, 0 with the error in `*err`, or 2
/// when the answer is not a number at all -- a tensor scaled by a float --
/// which the caller cannot hold and hands back to the interpreter.
int dream_rt_arith_f(dream::Process* p, int32_t op, dream::Value a, dream::Value b, double* out,
                     dream::Value* err);

/// The three numeric natives whose answer is an integer or keeps the shape of
/// its argument, for the case compiled code did not specialize: `to_int`,
/// `math.floor` and `math.abs`. Each is the native's own body. Returns 1 with
/// the value in `*out`, or 0 with the error there instead.
int dream_rt_to_int(dream::Process* p, dream::Value v, dream::Value* out);
int dream_rt_floor(dream::Process* p, dream::Value v, dream::Value* out);
int dream_rt_abs(dream::Process* p, dream::Value v, dream::Value* out);

/// The integer a double already in a register truncates to: the tail of
/// `to_int` and of `math.floor` when compiled code has the number unboxed. A
/// bignum past 2^62; an error for an infinity or a NaN, with 0 answered.
int dream_rt_int_of_double(dream::Process* p, double d, dream::Value* out);
/// The double a tagged number stands for, where compiled code met something
/// that was neither a fixnum nor a float box: a bignum, or an error.
int dream_rt_number_double(dream::Process* p, dream::Value v, double* out);

/// Raise a type error with a fixed message, for the shapes compiled code
/// rejects inline (a non-bool condition, say).
dream::Value dream_rt_type_error(dream::Process* p, const char* message);
/// The same with what was found instead, described as the interpreter
/// describes it: `message, got <v>`. The two tiers raise the same words.
dream::Value dream_rt_type_error_got(dream::Process* p, const char* message, dream::Value v);

/// The address of the process's reduction counter, fetched once on entry so
/// the loop back-edge can decrement it without a call.
int64_t* dream_rt_reduction_slot(dream::Process* p);

/// Actual machine-stack budget, including nested interpreter forces.
uint64_t dream_rt_stack_floor(dream::Process* p);

/// The frame a yielding loop resumes in: `frame`'s closure and size, the first
/// `n` slots from `vals`, the rest empty. See `Emitter::emit_yield`.
dream::Value dream_rt_yield_frame(dream::Process* p, dream::Value frame, uint32_t n,
                                  const dream::Value* vals);

/// Slots of the frame, for reading the arguments a compiled body was entered with.
dream::Value* dream_rt_frame_slots(dream::Value frame);


/// Call a host native the way the interpreter calls one.
///
/// The two shapes a native comes in are the two entry points: a *builtin* is an
/// immediate carrying an index into a static table, and a *member* is one
/// function of one host module -- `std.math`'s
/// `sqrt` -- named by where it sits rather than by a pointer, so that the
/// callee value this process already built for it is what gets called.
///
/// Arguments arrive in registers and are moved onto the process's value stack,
/// which is where a native expects to find them and where the collector can see
/// them. Four is the whole of it: a native taking more is not admitted, so the
/// signature is fixed and nothing has to be spilled through memory at the call
/// site. Unused positions are `UNIT` and are never read, because `argc` says
/// how many there are.
///
/// Returns 1 with the value in `*out`, or 0 with the error there instead.
/// Whichever it is, the value is in weak head normal form: a native that hands
/// back something it found without forcing it (`NativeOutcome::Enter`) is
/// forced here, which is exactly where the machine forces one too.
int dream_rt_native(dream::Process* p, uint32_t module_index, uint32_t member_index, uint32_t argc,
                    dream::Value a0, dream::Value a1, dream::Value a2, dream::Value a3,
                    dream::Value* out);
int dream_rt_builtin(dream::Process* p, uint32_t builtin_id, uint32_t argc, dream::Value a0,
                     dream::Value a1, dream::Value a2, dream::Value a3, dream::Value* out);

// ---------------------------------------------------------------------------
// Lists, arrays and maps
//
// What a compiled body needs to read and build the data a program is made of.
// The fast paths -- the head of a cell, an in-range array element, the empty
// test -- are written inline by the emitter; these are everything else, and
// each is the interpreter's own code for the same opcode, so the tiers agree on
// every message and every edge.
// ---------------------------------------------------------------------------

/// `c.[k]` with both in hand. Answers 1 with the element, *unforced*, in
/// `*out` (the emitter forces it, which is where the machine's `enter` forces
/// it); 2 when nothing is there and `has_default` says the node has an `else`;
/// and 0 with the error in `*out` otherwise. A list is walked, forcing its
/// spine as the machine's walk would.
int dream_rt_get(dream::Process* p, dream::Value c, dream::Value k, int32_t has_default,
                 dream::Value* out);

/// `c.[k => v]` with the container and the key in hand. `v` is stored as it
/// stands. 1 with the new container, or 0 with the error.
int dream_rt_set(dream::Process* p, dream::Value c, dream::Value k, dream::Value v,
                 dream::Value* out);

/// `SetMany`: `count` pairs, `pairs[2i]` a key and `pairs[2i + 1]` its value,
/// applied to `c` in order -- a record or an array copied once. 1 with the
/// new container, or 0 with the error the chain of `Set`s would have raised.
int dream_rt_set_many(dream::Process* p, dream::Value c, uint32_t count, const dream::Value* pairs,
                      dream::Value* out);

/// `thunk_for`'s read of `c.[k]`: the element, unforced, when it is already in
/// hand -- an array in range, or within the part of a list already built -- and
/// zero where `thunk_for` would have suspended the read. Forces nothing.
dream::Value dream_rt_peek(dream::Value c, dream::Value k);

/// A list cell, an array, a list literal. None of them can raise.
dream::Value dream_rt_cons(dream::Process* p, dream::Value head, dream::Value tail);
/// A cell whose tail is not known yet, linked in after `last` when there is
/// one (`last` zero means none), and the cell's tail filled in later. What a
/// loop building a list front to back is made of; see "Building ahead" in
/// jit.cpp. Neither can raise, and both keep the write barrier, because
/// `last` is a cell compiled code made but the collector may already own.
dream::Value dream_rt_cons_after(dream::Process* p, dream::Value last, dream::Value head);
void dream_rt_set_tail(dream::Process* p, dream::Value cell, dream::Value tail);
dream::Value dream_rt_make_list(dream::Process* p, uint32_t n, const dream::Value* items);
dream::Value dream_rt_make_array(dream::Process* p, uint32_t n, const dream::Value* items);
dream::Value dream_rt_make_record(dream::Process* p, uint32_t id, uint32_t n, const dream::Value* items);
/// An empty map, as `%{}` makes one. A literal with entries forces its keys,
/// and is left to the interpreter (see "What the interpreter evaluates").
dream::Value dream_rt_make_map(dream::Process* p);

/// The string, or the boxed float, for an image constant: the per-process
/// shared copy the interpreter hands out, so a literal keeps one identity.
dream::Value dream_rt_literal_str(dream::Process* p, uint32_t index);

/// The value of image global `index` -- a module, a function, or a 0-arity
/// pure value -- as the interpreter's `global_value` answers it, unforced.
dream::Value dream_rt_global(dream::Process* p, uint32_t index);

// ---------------------------------------------------------------------------
// Suspending an expression
//
// Compiled code has no heap frame of its own -- its slots are registers -- so
// an expression it must *not* evaluate cannot simply be wrapped in a thunk
// against one. These make a frame on demand: a copy of what the slots hold
// right now, which is exactly the frame the interpreter would have been
// running this iteration in. A thunk against it means what a thunk against
// that frame would have meant. See "Lazy positions" in jit.cpp.
// ---------------------------------------------------------------------------

/// A frame of `nslots` slots under `closure`, filled from `vals`. A slot whose
/// entry in `binds` is not `NO_NODE` is one a `let` bound, whose value compiled
/// code wrote out where it was read; the frame gets what the interpreter would
/// have put there, a thunk of the binding's expression against this frame.
dream::Value dream_rt_snapshot(dream::Process* p, dream::Value closure, uint32_t nslots,
                               const dream::Value* vals, const uint32_t* binds);

/// A remembered binding finished after a snapshot suspended it: settle the
/// snapshot's thunk of it to `value` in place. See "A binding computed once"
/// in jit.cpp.
void dream_rt_settle(dream::Process* p, dream::Value frame, uint32_t slot, dream::Value value);
/// The interpreter's own frame for this call adopted as the snapshot: the
/// slots `binds` names get the thunks `dream_rt_snapshot` would have made.
/// Answers the snapshot, which is a copy on the heap when the frame is on the
/// process's stack.
dream::Value dream_rt_adopt(dream::Process* p, dream::Value frame, uint32_t nslots,
                            const dream::Value* vals, const uint32_t* binds);

/// `node` suspended against `frame`.
dream::Value dream_rt_thunk(dream::Process* p, uint32_t node, dream::Value frame);

/// Apply a closure -- or any value that can be applied -- to `argc` arguments
/// as they stand, and force the answer to weak head normal form. 1 with the
/// answer, 0 with the error, and 2 -- having done nothing -- when the callee is
/// impure and the call has to be the interpreter's. A nested machine loop
/// that may collect, as a force may: the arguments are taken onto the value
/// stack and `args` cleared, so the caller's array does not keep them alive
/// once the callee has let them go (`apply_whnf_taking`).
int dream_rt_apply(dream::Process* p, dream::Value callee, uint32_t argc,
                   dream::Value* args, dream::Value* out);

}  // extern "C"
