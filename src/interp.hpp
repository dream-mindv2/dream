// The reduction machine.
//
// `run_process` advances a process by at most `budget` reductions and then
// returns, whatever the process is in the middle of. All of its state lives in
// the Process, so "the middle of" is a legal place to stop.

#pragma once

#include <string>

#include "process.hpp"
#include "runtime.hpp"
#include "value.hpp"

namespace dream {

/// Advance `p` until its budget runs out, it blocks, or it finishes.
void run_process(Process& p, int64_t budget);

/// Set up a fresh process to apply `callee` to `argc` values already pushed on
/// its value stack, and to halt when that returns.
void prime_apply(Process& p, Value callee, uint32_t argc);
/// Set up a fresh process to run the program's entry, function `func` of
/// `img`: `main!` as a value, or `main! ()` called with a unit for each
/// parameter it was written with.
void prime_entry(Process& p, const Image& img, uint32_t func);
/// Set up a fresh process to force `v` and halt.
void prime_force(Process& p, Value v);

/// Build a value for `node` without evaluating it. Constants and variable
/// references are returned directly -- allocating a thunk to wrap a value that
/// is already in normal form would cost an allocation and lose sharing.
Value thunk_for(Process& p, uint32_t node, Value frame);

/// The value of image global `index`, created on first use in this process.
Value global_value(Process& p, uint32_t index);

/// Force to weak head normal form from inside a native, by running a nested
/// machine loop. Returns false if the evaluation raised; the error is left in
/// `p.result`.
///
/// Whether that loop may *collect* is the caller's to say: see `VouchesForGc`.
bool force_whnf(Process& p, Value v, Value* out);

/// Apply `callee` to `argc` arguments and force the answer to weak head normal
/// form, by the same nested loop `force_whnf` runs. What compiled code calls a
/// closure with: the arguments arrive as they stand, suspended or not, exactly
/// as `do_apply` would have been handed them. Same collector rules as above.
bool apply_whnf(Process& p, Value callee, const Value* args, uint32_t argc, Value* out);
/// The same, taking the arguments: once they are on the value stack, `args`
/// is cleared. For a caller whose array is on the machine stack while a
/// collection may scan it (`Process::jit_stack_hi`): left as it was, it would
/// keep the arguments alive for as long as the call runs, which for a lazy
/// list handed to a loop is the whole list.
bool apply_whnf_taking(Process& p, Value callee, Value* args, uint32_t argc, Value* out);

/// How many continuations a process may have pending: `DREAM_MAX_DEPTH`.
size_t max_depth_limit();

/// A count or a byte size as a person writes one: digits, then optionally a
/// unit -- `k`, `m` or `g`, any case, with or without a trailing `b`, in powers
/// of 1024. False for anything else, and for zero.
bool parse_size(const char* text, size_t* out);

/// The per-process limits, given on the command line (`--max-depth`,
/// `--max-stack`, `--max-heap`). A zero leaves that one to its environment
/// variable or its default. They are read once, when the first process runs,
/// so this has to be called before that.
void override_limits(size_t conts, size_t stack, size_t heap_bytes);

/// The shared string for image string constant `index`, made once per process.
Value literal_string_value(Process& p, uint32_t index);

/// "My locals survive a collection" -- the claim that lets a nested force
/// collect.
///
/// A native that walks a lazy structure runs unbounded Dream work underneath
/// itself, and with the heap pinned that work has nowhere to put its garbage:
/// one `str.concat_all` over the compiler's own image was measured allocating
/// 354 MB of nursery, 87% of it dead, with no safepoint able to reach any of
/// it. Letting the nested loop collect is what fixes that, and the price is
/// this: a frame that holds a `Value` or an `Obj*` across a force must keep it
/// somewhere the collector rewrites -- `p.stack` or `p.pins` -- and re-read it
/// afterwards, never in a C++ local.
///
/// A vouch covers the frame that makes it and nothing below: `force_whnf`
/// consumes the flag and clears it, so a native reached through the forced
/// expression starts out unvouched like any other. A force nobody vouched for
/// raises `force_pins`, and a collection needs it at zero -- which is what
/// makes the guarantee hold for a *chain* of frames rather than just the
/// nearest one. Two unvouched frames deep, the vouched frame between them
/// still cannot collect, and that is the right answer.
///
/// Declare it at the top of the region that forces, not around each call: the
/// flag is restored on destruction, so a native that forces in a loop vouches
/// once.
///
/// One rule about *where* it may go, learnt the hard way. A native called
/// through `resume_native` has a known chain above it -- `run_process`,
/// `step_eval`, `do_apply`, `resume_native` -- and that chain is audited. A
/// helper the *machine* reaches directly does not: `concat_lists` is the list
/// `+`, and the operator path above it holds both operands in C++ locals, as
/// does any JIT-compiled frame that got there through `dream_rt_arith`. So
/// vouching is for natives, and a helper reached from the machine vouches only
/// if every path to it has been walked. See `PinsTheHeap`.
struct VouchesForGc {
    Process& p;
    const bool saved;
    explicit VouchesForGc(Process& proc) : p(proc), saved(proc.force_vouched) {
        p.force_vouched = true;
    }
    ~VouchesForGc() { p.force_vouched = saved; }
    VouchesForGc(const VouchesForGc&) = delete;
    VouchesForGc& operator=(const VouchesForGc&) = delete;
};

/// The opposite claim: "whatever anyone below me says, a collection here would
/// lose something."
///
/// Compiled code is why this exists. A JIT-compiled function keeps its slots in
/// machine registers -- that is what compiling it is *for* -- and the collector
/// cannot rewrite them. Underneath a force or a call (`dream_rt_force`,
/// `dream_rt_apply`) that no longer stops a collection: the machine stack is
/// scanned and what it points at is pinned in place (`Process::jit_stack_hi`).
/// The other runtime helpers still say this, because what they run keeps
/// values where a scan of the stack does not look -- `dream_rt_arith` on two
/// lists is `concat_lists`, which forces a whole spine, and a native may hold
/// one in a `std::vector` -- and `force_pins` makes the answer stick however
/// deep the call goes.
///
/// This is not hypothetical. Vouching for `concat_lists` -- which is written
/// correctly and keeps everything on the value stack -- made `effects_once`
/// hang about one run in twelve, because the compiled frame above it, not the
/// native below it, was what could not survive the collection.
struct PinsTheHeap {
    Process& p;
    explicit PinsTheHeap(Process& proc) : p(proc) { ++p.force_pins; }
    ~PinsTheHeap() { --p.force_pins; }
    PinsTheHeap(const PinsTheHeap&) = delete;
    PinsTheHeap& operator=(const PinsTheHeap&) = delete;
};

/// One `Value` held somewhere the collector can rewrite it, for as long as the
/// C++ frame that owns it lives. The counterpart of the promise `VouchesForGc`
/// makes: what cannot stay in a local goes here and is read back with `get`.
struct Pin {
    Process& p;
    const size_t at;
    Pin(Process& proc, Value v) : p(proc), at(proc.pins.size()) { p.pins.push_back(v); }
    ~Pin() { p.pins.resize(at); }
    Value get() const { return p.pins[at]; }
    Pin(const Pin&) = delete;
    Pin& operator=(const Pin&) = delete;
};

/// Force a value and everything reachable through it. Used by printing.
bool force_deep(Process& p, Value v, Value* out);

/// Render a value the way `console.print!` and `to_string` do. Forces as it
/// goes, so it can raise; returns false when it does.
bool stringify(Process& p, Value v, std::string* out);

/// One place in where an error was raised (see `locate_error` in interp.cpp):
/// a function that was running, or -- `made` -- the function that made a value
/// being forced. `path` is empty and `line` zero when the image does not say.
struct TracePlace {
    bool made;
    std::string name;
    std::string path;
    uint32_t line;
    uint32_t col;
};

/// Where `err` was raised, innermost first; empty for a value that is not an
/// error or has not been raised.
std::vector<TracePlace> error_trace(Process& p, Value err);

/// How a process's failure is reported when nobody handled it: its rendering,
/// except that a failed `match` reads as a diagnostic, `path:line:col: ...`,
/// followed by where it was raised, a line a place.
std::string describe_failure(Process& p, Value err);

/// A record's type as `type_of` names it, and as it is written: `Item`, and
/// `Item.file` for a variant (see RecordObj). `v` must be a record.
std::string record_type_name(Process& p, Value v);
std::string record_shown_name(Process& p, Value v);

/// A short, non-forcing description, for error messages.
std::string describe(Process& p, Value v);
/// The name the image gives function `fi`.
std::string image_function_name(Process& p, uint32_t fi);

Value make_integer(Process& p, int64_t v);

/// Would applying `v` perform an effect? Purity is spelling: a function's `!`,
/// a native's name, a partial application's function. Anything it cannot see
/// through counts as impure.
bool impure_callee(Process& p, Value v);

/// Build a callable host function. `user` is host state the native can read
/// back from its own object, which is how FFI bindings distinguish themselves.
Value make_native(Process& p, NativeFn fn, Value name, uint32_t arity,
                  uint32_t strict_mask, uint64_t user);

/// Arithmetic and comparison, exposed so JIT-compiled code can share exactly
/// the interpreter's slow paths rather than reimplementing them.
bool jit_arith(Process& p, Op op, Value a, Value b, Value* out);
bool jit_compare(Process& p, Op op, Value a, Value b, Value* out);
Value raise_error(Process& p, uint32_t kind_atom, const std::string& message);

}  // namespace dream
