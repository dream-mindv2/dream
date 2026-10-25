// A green process: the unit of concurrency, scheduling, and garbage collection.
//
// A process is a small heap plus two explicit stacks. Nothing about its state
// lives on the C++ stack, which is what lets a scheduler stop it between any
// two reductions, resume it on a different OS thread, collect its heap with
// precise roots, and keep hundreds of thousands of them alive at once. That is
// the same bargain BEAM makes, and it is why the interpreter is written as a
// state machine rather than as a recursive `eval`.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "heap.hpp"
#include "image.hpp"
#include "value.hpp"

// The two compiler builtins the scan of the machine stack rests on, spelled
// once for MSVC too (`collect_under_compiled`, `run_process`). Clang under
// MSVC's driver and MinGW's GCC both have the GCC spellings, so only MSVC's
// own compiler takes the other branch. `_AddressOfReturnAddress` is the slot
// just above a function's frame rather than its base, which serves the same
// purpose: a bound with all of the function's own words on one side.
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define DREAM_NOINLINE __declspec(noinline)
#define DREAM_FRAME_ADDRESS() _AddressOfReturnAddress()
#else
#define DREAM_NOINLINE __attribute__((noinline))
#define DREAM_FRAME_ADDRESS() __builtin_frame_address(0)
#endif

namespace dream {

class Runtime;

// ---------------------------------------------------------------------------
// Continuations
// ---------------------------------------------------------------------------

enum class ContKind : uint8_t {
    Halt,        // nothing left to do; the result is the process's answer
    UpdateThunk, // v1 = the thunk being forced; overwrite it with the result
    ApplyTo,     // a = argc; the arguments are the top `argc` of the value stack
    IfBranch,    // a = then node, b = else node, v1 = frame
    BinRight,    // a = opcode, b = right-hand node, v1 = frame
    BinFinish,   // a = opcode, v1 = the already-forced left-hand value
    LogicRight,  // a = opcode (And/Or), b = right-hand node, v1 = frame
    UnaryFinish, // a = opcode
    BlockNext,   // a = kids offset, b = count, c = next index, v1 = frame
    Catch,       // a = handler node, b = slot, c = value-stack depth, v1 = frame
    Finalize,    // a = cleanup node, c = value-stack depth, v1 = frame
    Finalized,   // a = raised, c = value-stack depth, v1 = saved outcome
    FieldOf,     // a = string index of the member name
    NativeArg,   // a = stack base, b = argc, c = index just forced, v1 = callee
    NativeArgs,  // a = the call node, b = stack base, c = index just evaluated,
                 // v1 = frame; the arguments filled in so far sit at the base
    NativeRetry, // a = stack base, b = argc, v1 = callee; re-invoke after a block
    ClosureArgs, // a = the call node, b = stack base, c = index just evaluated,
                 // v1 = frame; as NativeArgs, for a closure's strict arguments
    MapEntry,    // a = stack base, b = pair count, c = index, v1 = the map
    IndexKey,    // b = the get/set node, v1 = frame; the container is the result
    IndexApply,  // b = the get/set node, v1 = frame; the container is on the value stack
    GetWalk,     // a = cells still to step over, b = the get node, c = the index, v1 = frame
    SetWalk,     // a = cells still to step over, b = the value's node, c = the index,
                 // v1 = frame; the cells stepped over so far are on the value stack
    SwitchOn,    // a = the switch node, v1 = frame; the result is its subject
    SwitchKey,   // a = the switch node, v1 = frame; the result is the key (a head)
    EnterRetry,  // a = function index, b = value-stack height, v1 = frame; re-enter
                 // a compiled call whose nested force was suspended (see
                 // `enter_function`)
    SetManyStep, // b = the set_many node, c = the next pair, v1 = frame; the
                 // result is the container with the pairs before it applied
    BindThen,    // a = kids offset, b = count, c = the bind's index, v1 = frame;
                 // the result goes in the bind's slot and the block goes on
};

struct Cont {
    ContKind kind;
    uint8_t pad0;
    uint16_t pad1;
    uint32_t a;
    uint32_t b;
    uint32_t c;
    Value v1;
};

/// The continuation stack: the machine's "what to do when this part is done".
///
/// This was a `std::vector<Cont>`, and pushing one was **6.5% of a
/// self-compile** -- more than `step_return`, and second only to the dispatch
/// switch itself. Not the growth, which amortizes to nothing against a stack
/// that is pushed and popped rather than grown: the *call*. `step_eval`
/// pushes a continuation from some thirty places, and at that many sites GCC
/// stops inlining `emplace_back` and emits an outlined clone, so every
/// continuation the machine pushed paid a function call and a capacity check
/// that no caller could hoist. A `Cont` is twenty-four bytes of POD; pushing
/// one is four stores, and it should cost four stores.
///
/// So the fast path is written out and nothing else is: `grow` is the only
/// thing out of line, and it is reached once per doubling. The type is
/// deliberately not a general container -- it has exactly the operations the
/// machine performs, which is why `insert_at` exists instead of iterators.
class ContStack {
public:
    ContStack() = default;
    ~ContStack() { std::free(data_); }
    ContStack(const ContStack&) = delete;
    ContStack& operator=(const ContStack&) = delete;

    /// `always_inline` because the heuristics say no and they are wrong here.
    /// `step_eval` is one switch over every opcode and pushes a continuation
    /// from some thirty of its arms, which puts it far past the inliner's
    /// growth ceiling -- so GCC outlines even this, and a push stayed a call
    /// costing 6% of a self-compile after it stopped being a `std::vector`.
    /// The body is a compare and two stores; it is smaller than its own call
    /// sequence.
#if defined(_MSC_VER)
    __forceinline void push_back
#else
    __attribute__((always_inline)) inline void push_back
#endif
    (const Cont& c) {
        if (size_ == cap_) grow();
        data_[size_++] = c;
    }

    /// Splice one in below the work already on the stack. Rare -- only a retry
    /// under a suspended nested force -- and O(n) on purpose, because making
    /// the common push cheaper is what this class is for.
    void insert_at(size_t i, const Cont& c) {
        if (size_ == cap_) grow();
        std::memmove(data_ + i + 1, data_ + i, (size_ - i) * sizeof(Cont));
        data_[i] = c;
        ++size_;
    }

    void pop_back() { --size_; }
    void clear() { size_ = 0; }
    Cont& back() { return data_[size_ - 1]; }
    const Cont& back() const { return data_[size_ - 1]; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void reserve(size_t n) { if (n > cap_) regrow(n); }

    Cont* begin() { return data_; }
    Cont* end() { return data_ + size_; }

private:
    void grow() { regrow(cap_ ? cap_ * 2 : 64); }
    void regrow(size_t want);

    Cont* data_ = nullptr;
    size_t size_ = 0;
    size_t cap_ = 0;
};

/// The interpreter's frames, one stack per process, outside the heap.
///
/// A frame was a heap object, 44% of everything a self-compile allocated, and
/// half of those were kept alive past their call by a suspension made in them
/// -- so promoted, copied and marked. A suspension now copies the values it
/// reads (`flat_thunk` in interp.cpp) and a closure always did, so nothing on
/// the heap points at an interpreter frame and its life is exactly its call's:
/// it is reachable from the current frame and the continuations, and from
/// nothing that outlives them. See "Suspensions that hold values, and frames
/// on a stack" in docs/notes/vm-performance.md.
///
/// A frame records in `pad` the continuation depth it was made at. Every
/// continuation that names a frame was pushed while that frame was current, so
/// it sits at or above that depth: once the stack is below it, nothing can
/// reach the frame. Two places act on that. `step_return` drops frames made
/// deeper than the stack now is, and making a frame drops those made at the
/// depth it is made at or deeper -- a tail call's caller, whose last use was
/// building the callee's arguments. A cont spliced in below (`insert_at`) only
/// makes a recorded depth an underestimate, which keeps a frame a little
/// longer and never frees one early.
///
/// A nested machine loop (`nested_whnf`) raises `floor` to the frames it was
/// started over, because the code that started it is still in the middle of
/// using its frame at a depth the nested loop will reach again.
///
/// The frames carry `GC_SHARED`, which every path of the collector already
/// reads as "not this heap's, do not move or trace": they are roots instead,
/// visited by `Process::visit_roots`, so a young value in one is found and
/// updated without a barrier. Memory comes in chunks that are kept, so a deep
/// recursion's pages are reused rather than returned and asked for again.
class FrameStack {
public:
    FrameStack() = default;
    ~FrameStack();
    FrameStack(const FrameStack&) = delete;
    FrameStack& operator=(const FrameStack&) = delete;

    /// A frame of `nslots` slots made at continuation depth `depth`, the first
    /// `filled` of which the caller writes. Frames at `depth` or deeper are
    /// dropped first (see above).
#if defined(_MSC_VER)
    __forceinline FrameObj* push
#else
    __attribute__((always_inline)) inline FrameObj* push
#endif
    (Value closure, uint32_t nslots, uint32_t filled, uint32_t depth) {
        if (depth <= release_at_) release_from(depth);
        const size_t bytes = sizeof(FrameObj) + size_t(nslots) * sizeof(Value);
        if (size_t(end_ - top_) < bytes) next_chunk(bytes);
        auto* o = reinterpret_cast<FrameObj*>(top_);
        top_ += bytes;
        o->type = ObjType::Frame;
        o->gc = GC_SHARED;
        o->aux = 0;
        o->bytes = uint32_t(bytes);
        o->closure = closure;
        o->nslots = nslots;
        o->pad = depth;
        Value* slots = o->slots();
        for (uint32_t i = filled; i < nslots; ++i) slots[i] = NIL_SLOT;
        if (size_ == cap_) grow();
        frames_[size_++] = o;
        release_at_ = depth;
        return o;
    }

    /// Drop the frames made deeper than `depth`: the continuation stack has
    /// come back below them.
    void returned_to(size_t depth) {
        if (depth < release_at_) release_from(uint32_t(depth) + 1);
    }

    /// The frames from here up are a nested loop's; see `floor`.
    size_t raise_floor() {
        const size_t was = floor_;
        floor_ = size_;
        release_at_ = 0;
        return was;
    }
    void restore_floor(size_t was) {
        floor_ = was;
        release_at_ = size_ > floor_ ? frames_[size_ - 1]->pad : 0;
    }

    template <class F>
    void each(F&& f) {
        for (size_t i = 0; i < size_; ++i) f(frames_[i]);
    }
    size_t size() const { return size_; }
    /// Whether `v` points into this stack's memory. The machine's frame
    /// register, and a frame a nested force saved to put back, may name a
    /// frame that has been dropped and its memory given to another, where a
    /// header is no longer a header: the collector asks this rather than read
    /// one. A frame here never needs forwarding anyway.
    bool holds(Value v) const {
        if (!is_ptr(v)) return false;
        auto* at = reinterpret_cast<const uint8_t*>(as_obj(v));
        for (const Chunk& c : chunks_) {
            if (at >= c.base && at < c.base + c.cap) return true;
        }
        return false;
    }
    /// Everything at once, for a process that is done.
    void clear();

private:
    void release_from(uint32_t depth);
    void next_chunk(size_t bytes);
    void grow();

    struct Chunk {
        uint8_t* base;
        size_t cap;
    };
    FrameObj** frames_ = nullptr;
    size_t size_ = 0;
    size_t cap_ = 0;
    size_t floor_ = 0;
    /// The depth of the top frame when it may be dropped, else 0: a frame
    /// made at depth `d` is dropped by a push at a depth `<= d` or a return
    /// below `d`, and one compare against this is all either costs when there
    /// is nothing to drop.
    uint32_t release_at_ = 0;
    std::vector<Chunk> chunks_;
    size_t chunk_ = 0;
    uint8_t* top_ = nullptr;
    uint8_t* end_ = nullptr;
};

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// A message in flight. It carries its own heap: the sender deep-copies the
/// value into it, and the receiver copies it out into its own heap. Neither
/// side ever touches the other's heap, so a send needs no heap lock -- only
/// the mailbox queue is synchronized.
struct Message {
    std::unique_ptr<Heap> heap;
    Value value = UNIT;
    uint64_t from_pid = 0;
};

class Mailbox {
public:
    void push(std::unique_ptr<Message> m);
    std::unique_ptr<Message> pop();
    /// Take the first message, copied into `dest`; false when there is none.
    ///
    /// The copy is made *before* the lock is let go, and that is not tidiness.
    /// A census walks mailboxes under this lock to find what messages hold,
    /// and a message popped and then copied is, for that moment, in no
    /// mailbox and not yet in any heap -- a census finishing then would free
    /// a region the copy is about to read. Under the lock, either the walk
    /// sees the message or the copy has already told the census it crossed.
    bool pop_into(Heap& dest, Value* out);
    bool empty() const;
    size_t size() const;

    /// Selective receive. Only the owning process ever removes a message, so a
    /// position it has looked at stays the same message until it removes it:
    /// senders only append. That is what lets the scan live in Dream, one
    /// position at a time, rather than in a native that would have to run the
    /// caller's predicate underneath itself.
    ///
    /// `peek_into` copies the message at `i` into `dest` and answers false when
    /// there is no such message. The copy is made under the lock because the
    /// message's heap is about to be read by a thread that does not own it.
    bool peek_into(size_t i, Heap& dest, Value* out) const;
    /// Remove the message at `i`; false when there is none.
    bool remove_at(size_t i);
    /// Every message waiting, under the lock, for a census to walk.
    template <class F>
    void for_each(F&& f) const {
        std::lock_guard<std::mutex> g(mutex_);
        for (const auto& m : queue_) f(m->value);
    }

private:
    mutable std::mutex mutex_;
    std::deque<std::unique_ptr<Message>> queue_;
};

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

enum class Mode : uint8_t {
    Eval,    // reduce `node` in `frame` to weak head normal form
    Return,  // `result` is in WHNF; hand it to the top continuation
    Raise,   // `result` is an error; unwind to the nearest catch
    Halted,
};

enum class ProcStatus : uint8_t {
    Runnable,
    Running,
    Waiting,   // blocked in `recv!`, `join!`, or on a descriptor
    Finished,
    Failed,
};

/// What a parked process is parked on.
///
/// The status alone says a process is stuck but not why, and "waiting for a
/// message that will never come" and "waiting for a socket the kernel has not
/// answered" look identical from outside. `std.vm` reports this so the
/// difference is visible without a debugger.
enum class WaitReason : uint8_t {
    None,
    Message,   // `recv!`
    Join,      // `join!`
    Io,        // a descriptor the poller is watching
    Timer,     // `vm.sleep_until!`, or a message wait with a deadline
};

const char* wait_reason_name(WaitReason r);
const char* mode_name(Mode m);
const char* cont_kind_name(ContKind k);

class Process;

/// Run the destructor of every C resource `p` still owns, newest first, and
/// forget them. Called as a process finishes, from `os.exit!`, and from the
/// process's destructor as a backstop; a second call finds nothing to do.
void release_foreign(Process& p);

/// The lowest address of this OS thread's machine stack, or 0 where that
/// cannot be asked. What compiled calls and nested forces measure their room
/// against.
uintptr_t machine_stack_low();

/// Room left on this thread's machine stack, past `here`, before a nested
/// machine loop would run into its end: what a force needs to be sure of
/// before it starts one rather than crash in the middle (see
/// "stack_thread.hpp" for why the size differs by platform).
inline bool machine_stack_has_room(const void* here) {
    constexpr uintptr_t reserve = 256 * 1024;
    const uintptr_t low = machine_stack_low();
    return low == 0 || reinterpret_cast<uintptr_t>(here) >= low + reserve;
}

class Process final : public RootSource {
public:
    Process(Runtime& rt, uint64_t id);
    ~Process() override;

    uint64_t id() const { return id_; }

    /// The handle of a `net.connect!` that has been issued but not yet
    /// finished, or -1. A blocking native is re-entered from the top when its
    /// process is woken, and `connect` is the one operation where the second
    /// call must ask how the first went rather than start again.
    int64_t io_pending = -1;

    /// The `std.os.exec!` whose child is still running, or -1. Same reason as
    /// `io_pending`: a blocking native is re-entered from the top, and this is
    /// how the second entry knows which child it was waiting for.
    int64_t os_pending = -1;

    /// Set when a nested force (`force_whnf`) was suspended because a blocking
    /// native inside it asked to park. The force's continuations are left in
    /// place to be resumed; `force_resume_at` is the index in `conts` where
    /// the outer operation's retry must be spliced in, *below* them.
    bool force_blocked = false;

    size_t force_resume_at = 0;

    /// `DREAM_JIT_CHECK_EFFECTS`: what each compiled call on this process's
    /// machine stack has done that is an effect, innermost last, and what the
    /// attempts before it did when it is a retry. A compiled call that parks
    /// is run again from the top, which is only sound because compiled bodies
    /// are pure and forcing is memoised; an effect seen twice across attempts
    /// is the proof that one of those has a gap. Empty unless the check is on.
    /// See `effect_check` in interp.cpp.
    struct EffectAttempt {
        uint32_t func;
        std::vector<uint64_t> prior;
        std::vector<uint64_t> seen;
    };
    std::vector<EffectAttempt> effect_attempts;
    /// The attempts of a parked compiled call, carried by its `EnterRetry`
    /// (in the continuation's `c`) until the retry runs.
    std::unordered_map<uint32_t, std::vector<uint64_t>> effect_retries;
    uint32_t effect_retry_next = 1;
    /// Set by `EnterRetry` for the one `enter_function` it makes.
    uint32_t effect_retry_id = 0;

    /// Whether a nested force may collect, and what it costs to say so.
    ///
    /// `force_whnf` runs the machine underneath a native, on the C++ stack, and
    /// for most of the VM's life it could not collect there: a native holding a
    /// raw `Value` across the call would have had its object promoted out from
    /// under it. The cost of that rule was not a bug but a *size* -- a native
    /// that walks a long lazy list runs unbounded Dream work with the heap
    /// pinned, and one `str.concat_all` over the image was measured allocating
    /// 354 MB of nursery that no safepoint could reach. 87% of it was garbage.
    ///
    /// So the rule is now earned rather than assumed. Collection is legal in a
    /// nested force exactly when every C++ frame between it and `run_process`
    /// has said its locals can survive one -- which means keeping them on
    /// `stack` or in `pins` and re-reading, never in a C++ local. A frame says
    /// so with `VouchesForGc` (interp.hpp); `force_pins` counts the frames that
    /// have *not*, and a collection needs it at zero.
    ///
    /// `force_vouched` is how the claim reaches `force_whnf`: the guard sets
    /// it, the force consumes it, and clears it for everything it goes on to
    /// reach -- because a vouch is about the frame that made it and says
    /// nothing about the natives that run below.
    bool force_vouched = false;
    uint32_t force_pins = 0;

    /// Where the machine stack a collection under compiled code must scan
    /// ends, or null when no compiled frame is running.
    ///
    /// Compiled code is the one place the rule above cannot be kept: its
    /// values are in registers and stack slots, where nothing can re-read
    /// them. So instead of vouching it is *scanned*. `loop_stack_hi` is the
    /// frame of the innermost machine loop -- `run_process`, or the nested one
    /// behind `force_whnf` -- which keeps nothing in C++ locals at a
    /// safepoint. `enter_function` copies it to `jit_stack_hi` when it enters
    /// the outermost compiled frame, and a collection while that is set pins
    /// every object a word between the collector and there points into
    /// (`Heap::set_conservative_roots`). The frames in between -- compiled
    /// bodies, the `dream_rt_*` helpers they call, and the interpreter's own
    /// frames that entered them -- are covered without any of them having to
    /// say what they hold. A native among them that holds a value anywhere
    /// but the machine stack still pins the heap as it always has.
    void* loop_stack_hi = nullptr;
    void* jit_stack_hi = nullptr;

    /// Nested `force_whnf` loops currently on the process's machine stack.
    ///

    /// Interpreted code never nests them deeply -- each force is a heap
    /// continuation and the loop returns between links -- so a deep count is a
    /// *compiled* signature: a compiled body forces a slot by calling, and
    /// `load_slot -> force -> dream_rt_force -> force_whnf` runs a whole nested
    /// machine, which re-enters another compiled body, which forces again. One
    /// real C++ frame per link of a lazy chain, against a fixed machine stack.
    /// That was the SIGSEGV under "Fixed: compiled code forcing a long thunk
    /// chain crashed" in docs/notes/vm-performance.md; `enter_function` declines the compiled tier
    /// past `kMaxForceNestForCompiled`, and the interpreter, whose recursion is
    /// heap continuations and so has a limit it can check, finishes the chain.
    uint32_t force_nest = 0;

    /// Values a C++ frame is holding across a call that can collect.
    ///
    /// The interpreter keeps no machine state on the C++ stack, which is the
    /// whole reason the root set is a list rather than a stack walk. The
    /// exceptions are few and all of them are here: the callee a native call
    /// must still name when the native parks, and the spare arguments
    /// `do_apply` sets aside when a native is over-applied. The collector
    /// visits this like any other root and rewrites what it finds, so the
    /// frame reads its value back out rather than trusting the copy it kept.
    std::vector<Value> pins;

    /// Bytes already charged to some function by the allocation profile.
    ///
    /// A step's allocation is measured as the heap's total before and after,
    /// and a step that enters a native which forces runs whole machine loops
    /// underneath itself -- every one of which measures and charges the same
    /// bytes again. So each level subtracts what the levels below it have
    /// already claimed, and this is the running tally they claim into. Only
    /// touched while profiling.
    uint64_t alloc_attributed = 0;
    /// The heap's collection count when the heap limit was last raised in
    /// this process. The limit is judged by the live bytes the last
    /// collection measured, so until another collection has measured them
    /// again every step would raise it afresh -- including the steps of the
    /// `catch` that caught it, which is what made it uncatchable.
    uint64_t heap_limit_raised_at = UINT64_MAX;
    // A kill waits while cleanup runs, including across slices and parking.
    uint32_t cleanup_depth = 0;
    bool kill_unwinding = false;

    Runtime& runtime() { return rt_; }
    Heap& heap() { return heap_; }
    const Heap& heap() const { return heap_; }

    /// The image this process runs, cached at construction.
    ///
    /// Reaching it the honest way -- `p.runtime().image()` -- is two dependent
    /// loads before the one that reads the node, and the machine asks for a
    /// node on every step. An image is loaded before any process exists and is
    /// never replaced, so the indirection buys nothing.
    const Image* code = nullptr;

    // --- machine state ---
    Mode mode = Mode::Halted;
    uint32_t node = NO_NODE;
    Value frame = UNIT;
    Value result = UNIT;

    std::vector<Value> stack;
    ContStack conts;
    FrameStack frames;

    /// A frame for a call about to be entered, on `frames`. See `FrameStack`.
    Value push_frame(Value closure, uint32_t nslots, uint32_t filled = 0) {
        return from_obj(frames.push(closure, nslots, filled, uint32_t(conts.size())));
    }

    // --- scheduling ---
    std::atomic<ProcStatus> status{ProcStatus::Runnable};
    int64_t reductions = 0;
    uint64_t total_reductions = 0;
    /// The budget `run_process` was given, so a nested force can re-arm it.
    int64_t slice = 4000;
    /// The slice ran out while a nested force was running.
    ///
    /// A nested force cannot hand the process back -- the natives above it are
    /// waiting on the C++ stack for a value -- so the budget is not something
    /// it can spend down to zero and stop at. Left alone it goes arbitrarily
    /// negative, and then everything that asks "is there budget left?" is told
    /// "no" for ever. That is not a counter being untidy: a JIT-compiled loop
    /// reached through `strict!` yielded on *every* iteration for it, giving
    /// back the heap frame the compilation exists to remove and re-entering
    /// the interpreter once per element. So the nested loop re-arms the budget
    /// and sets this instead, and `run_process` ends the slice the moment the
    /// force it was under returns.
    bool slice_spent = false;

    Mailbox mailbox;

    /// One cache slot per image global. Globals are per-process, not shared:
    /// a memoized top-level value lives in whichever heap forced it, and heaps
    /// are private. Code is shared; data is not -- as in BEAM.
    std::vector<Value> globals;

    /// One slot per host module member, indexed by `ModuleDef::member_base + i`.
    ///
    /// `native.tail xs` used to build two objects before it could call
    /// anything: a string for the member's name and a `NativeObj` around the
    /// function pointer. Both are decided by which member it is and nothing
    /// else, so the same pair was allocated again at every call -- in a fold
    /// over ten million elements, forty million objects that were all equal.
    /// The value belongs to this heap, so the cache does too, and it is a root.
    std::vector<Value> native_cache;

    /// One slot per image string constant: the `StrObj` this process hands out
    /// for it. A literal is immutable and has no identity a program can
    /// observe, so one copy per process does what a fresh copy per evaluation
    /// did, without the allocation. A root, like the others here.
    ///
    /// Per-process, and sized by the whole program's string table, which is
    /// the same bargain `globals` above makes: a table this process may only
    /// use a corner of, in exchange for the lookup being an index. It is
    /// allocated on the first literal a process reaches, so a process that
    /// evaluates none pays nothing, and the table is small even for a large
    /// program -- the compiler's own image has 1752 strings against 1118
    /// globals. If a workload ever wants hundreds of thousands of processes
    /// each touching a corner of a huge string table, this and `globals` want
    /// bounding together.
    std::vector<Value> string_cache;

    /// The same for float constants, which are boxed and so cost an
    /// allocation each: a loop over `1.0 / x` allocated one per iteration for
    /// a value the image already held.
    std::vector<Value> float_cache;

    /// The C resources this process owns through `std.ffi`: every handle it
    /// was given by a foreign call, with the destructor that lets it go. Null
    /// until the first one, so a process that never touches C pays a pointer.
    /// A process owns its resources the way it owns its heap -- nobody else can
    /// reach them, and when it ends they end, in the reverse of the order they
    /// were made. `release_foreign` is what does that; see dream/src/ffi.cpp.
    struct ForeignTable* foreign = nullptr;

    /// Set once the process stops; `failed` distinguishes a raised error.
    Value exit_value = UNIT;
    bool failed = false;
    std::string failure_text;

    /// Processes blocked in `join!` on this one. The mutex also guards the
    /// transition to a finished status, so a joiner cannot register itself
    /// against a process that has already exited.
    std::mutex waiters_mutex;
    std::vector<uint64_t> waiters;
    /// Processes that asked, with `vm.monitor!`, to be told when this one
    /// ends. Each is sent `[:down, pid, outcome]` by `Scheduler::finish`.
    /// Under `waiters_mutex`, for the reason `waiters` is: registering and
    /// finishing must not interleave.
    std::vector<uint64_t> monitors;

    /// Set by `vm.kill!`. A process is only ever stopped by the worker running
    /// it, so a kill is a request: it takes effect at the start of the next
    /// slice, and `kill!` wakes a parked target so there is one. A slice is
    /// 4000 reductions, which is as prompt as preemption already is.
    std::atomic<bool> kill_requested{false};
    /// Why, copied out of the killer's heap the way a message is. Written once,
    /// under `sched_mutex`, by whichever kill arrived first.
    std::unique_ptr<Message> kill_reason;

    /// The timer a timed wait armed (`Scheduler::arm_timer`), or 0, and the
    /// deadline it was armed for. Touched only by the worker running this
    /// process -- from the native that waits -- so it needs no lock. A blocked
    /// native is re-entered from the top on every wake, and this is how the
    /// second entry knows its timer is already set.
    uint64_t timer_seq = 0;
    int64_t timer_deadline = -1;

    /// A census round this process owes a report for, or 0: set by
    /// `Scheduler::request_census` when the process was running or queued, and
    /// honoured -- a census major of its own heap -- at the start of its next
    /// slice or as it parks, whichever comes first.
    std::atomic<uint64_t> census_due{0};

    /// Guards the parking handshake below. Only the worker currently running a
    /// process may change its machine state, so a blocking builtin does not
    /// park the process itself -- it asks to be parked, and the worker does it
    /// once it has stopped touching the process. Anyone waking a process that
    /// is still running just leaves a note.
    std::mutex sched_mutex;
    /// Set by a blocking builtin: "park me when this slice ends".
    bool park_requested = false;
    /// Set by a waker that found the process still running.
    bool wake_pending = false;
    /// Why this process asked to be parked. Written by the blocking native
    /// alongside `park_requested`, and read by `std.vm` from another thread.
    std::atomic<WaitReason> wait_reason{WaitReason::None};
    /// The descriptor being waited on, when `wait_reason` is `Io`.
    std::atomic<int> wait_fd{-1};

    bool is_done() const {
        ProcStatus s = status.load(std::memory_order_relaxed);
        return s == ProcStatus::Finished || s == ProcStatus::Failed;
    }

    /// Collect if the heap has grown past its threshold. Only legal at a
    /// safepoint, where every live value is reachable from the stacks.
    void maybe_collect();
    /// `maybe_collect` underneath compiled code, with the machine stack as
    /// roots. See `jit_stack_hi`.
    void collect_under_compiled();

    /// Room for another compiled call on this OS thread's machine stack.
    /// A nested force resets the JIT's recursion counter, so count bytes from
    /// the outer compiled entry as well as keeping space above the guard page.
    uintptr_t compiled_stack_floor() const;
    bool compiled_stack_has_room(void* here) const {
        return reinterpret_cast<uintptr_t>(here) >= compiled_stack_floor();
    }

    void visit_roots(Heap& heap) override;

    /// The error a kill ends this process with: kind `:killed`, the reason
    /// `kill!` was given as the payload. Allocates in this heap; call it only
    /// where an allocation is allowed.
    Value kill_error();

    // Convenience for natives and the interpreter.
    Value make_error_value(const char* kind, const char* message);
    [[noreturn]] void unreachable(const char* what);

private:
    Runtime& rt_;
    uint64_t id_;
    Heap heap_;
};

}  // namespace dream
