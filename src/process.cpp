#include "process.hpp"

#include <algorithm>

#ifdef _WIN32
#include "windows.hpp"
#elif defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#include "interp.hpp"
#include "runtime.hpp"

namespace dream {

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Eval: return "eval";
        case Mode::Return: return "return";
        case Mode::Raise: return "raise";
        case Mode::Halted: return "halted";
    }
    return "?";
}

const char* cont_kind_name(ContKind k) {
    switch (k) {
        case ContKind::Halt: return "halt";
        case ContKind::UpdateThunk: return "update_thunk";
        case ContKind::ApplyTo: return "apply_to";
        case ContKind::IfBranch: return "if_branch";
        case ContKind::BinRight: return "bin_right";
        case ContKind::BinFinish: return "bin_finish";
        case ContKind::LogicRight: return "logic_right";
        case ContKind::UnaryFinish: return "unary_finish";
        case ContKind::BlockNext: return "block_next";
        case ContKind::BindThen: return "bind_then";
        case ContKind::Catch: return "catch";
        case ContKind::Finalize: return "finalize";
        case ContKind::Finalized: return "finalized";
        case ContKind::FieldOf: return "field_of";
        case ContKind::NativeArg: return "native_arg";
        case ContKind::NativeArgs: return "native_args";
        case ContKind::NativeRetry: return "native_retry";
        case ContKind::ClosureArgs: return "closure_args";
        case ContKind::MapEntry: return "map_entry";
        case ContKind::IndexKey: return "index_key";
        case ContKind::IndexApply: return "index_apply";
        case ContKind::GetWalk: return "get_walk";
        case ContKind::SetWalk: return "set_walk";
        case ContKind::SwitchOn: return "switch_on";
        case ContKind::SwitchKey: return "switch_key";
        case ContKind::EnterRetry: return "enter_retry";
        case ContKind::SetManyStep: return "set_many_step";
    }
    return "?";
}

const char* wait_reason_name(WaitReason r) {
    switch (r) {
        case WaitReason::Message: return "message";
        case WaitReason::Join: return "join";
        case WaitReason::Io: return "io";
        case WaitReason::Timer: return "timer";
        case WaitReason::None: break;
    }
    return "none";
}

// ---------------------------------------------------------------------------
// Mailbox
// ---------------------------------------------------------------------------

/// Grow the continuation stack. Out of line because it runs once per doubling
/// and the push beside it runs tens of millions of times; `Cont` is trivially
/// copyable, so this is a realloc and nothing more.
void ContStack::regrow(size_t want) {
    Cont* next = static_cast<Cont*>(std::realloc(data_, want * sizeof(Cont)));
    if (!next) {
        std::fprintf(stderr, "dream: out of memory growing the continuation stack\n");
        std::abort();
    }
    data_ = next;
    cap_ = want;
}

FrameStack::~FrameStack() {
    std::free(frames_);
    for (Chunk& c : chunks_) std::free(c.base);
}

void FrameStack::release_from(uint32_t depth) {
    while (size_ > floor_ && frames_[size_ - 1]->pad >= depth) {
        auto* f = reinterpret_cast<uint8_t*>(frames_[--size_]);
        // Frames are taken in order through the chunks, so the one a frame
        // came from is this chunk or one before it.
        while (f < chunks_[chunk_].base || f >= chunks_[chunk_].base + chunks_[chunk_].cap) --chunk_;
        top_ = f;
        end_ = chunks_[chunk_].base + chunks_[chunk_].cap;
    }
    release_at_ = size_ > floor_ ? frames_[size_ - 1]->pad : 0;
}

void FrameStack::next_chunk(size_t bytes) {
    // The space left in this chunk waits until the frames before it are
    // gone; the next chunk is reused when it is large enough, and replaced
    // when it is not.
    size_t at = top_ ? chunk_ + 1 : 0;
    if (at < chunks_.size() && chunks_[at].cap < bytes) {
        for (size_t i = at; i < chunks_.size(); ++i) std::free(chunks_[i].base);
        chunks_.resize(at);
    }
    if (at == chunks_.size()) {
        size_t cap = chunks_.empty() ? size_t(64) << 10 : chunks_.back().cap * 2;
        if (cap > (size_t(16) << 20)) cap = size_t(16) << 20;
        if (cap < bytes) cap = bytes;
        auto* base = static_cast<uint8_t*>(std::malloc(cap));
        if (!base) {
            std::fprintf(stderr, "dream: out of memory growing the frame stack\n");
            std::abort();
        }
        chunks_.push_back(Chunk{base, cap});
    }
    chunk_ = at;
    top_ = chunks_[at].base;
    end_ = top_ + chunks_[at].cap;
}

void FrameStack::grow() {
    const size_t want = cap_ ? cap_ * 2 : 256;
    auto** next = static_cast<FrameObj**>(std::realloc(frames_, want * sizeof(FrameObj*)));
    if (!next) {
        std::fprintf(stderr, "dream: out of memory growing the frame stack\n");
        std::abort();
    }
    frames_ = next;
    cap_ = want;
}

void FrameStack::clear() {
    size_ = 0;
    floor_ = 0;
    release_at_ = 0;
    chunk_ = 0;
    top_ = chunks_.empty() ? nullptr : chunks_[0].base;
    end_ = chunks_.empty() ? nullptr : chunks_[0].base + chunks_[0].cap;
}

void Mailbox::push(std::unique_ptr<Message> m) {
    std::lock_guard<std::mutex> g(mutex_);
    queue_.push_back(std::move(m));
}

std::unique_ptr<Message> Mailbox::pop() {
    std::lock_guard<std::mutex> g(mutex_);
    if (queue_.empty()) return nullptr;
    auto m = std::move(queue_.front());
    queue_.pop_front();
    return m;
}

bool Mailbox::empty() const {
    std::lock_guard<std::mutex> g(mutex_);
    return queue_.empty();
}

size_t Mailbox::size() const {
    std::lock_guard<std::mutex> g(mutex_);
    return queue_.size();
}

Value Process::kill_error() {
    Value reason = UNIT;
    {
        std::lock_guard<std::mutex> g(sched_mutex);
        if (kill_reason) reason = Heap::copy_between(heap_, kill_reason->value);
    }
    // `reason` is held across one allocation, which cannot collect.
    return heap_.make_error(make_atom(well_known(rt_).killed), reason);
}

bool Mailbox::pop_into(Heap& dest, Value* out) {
    std::unique_ptr<Message> m;
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (queue_.empty()) return false;
        *out = Heap::copy_between(dest, queue_.front()->value);
        m = std::move(queue_.front());
        queue_.pop_front();
    }
    // The message's own heap goes after the lock, with `m`.
    return true;
}

bool Mailbox::peek_into(size_t i, Heap& dest, Value* out) const {
    std::lock_guard<std::mutex> g(mutex_);
    if (i >= queue_.size()) return false;
    *out = Heap::copy_between(dest, queue_[i]->value);
    return true;
}

bool Mailbox::remove_at(size_t i) {
    std::lock_guard<std::mutex> g(mutex_);
    if (i >= queue_.size()) return false;
    queue_.erase(queue_.begin() + std::ptrdiff_t(i));
    return true;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

Process::Process(Runtime& rt, uint64_t id) : rt_(rt), id_(id), heap_(64 * 1024) {
    stack.reserve(64);
    conts.reserve(64);
    code = rt.has_image() ? &rt.image() : nullptr;
}

Process::~Process() { release_foreign(*this); }

uintptr_t Process::compiled_stack_floor() const {
    // Thread-local, not process-local: a green process can resume on another
    // scheduler worker, whose stack may have a different address and size.
    static thread_local const uintptr_t low = []() -> uintptr_t {
#if defined(_WIN32)
        // AllocationBase names the reserved stack, including its uncommitted
        // part. VirtualQuery is available on the Windows 7 target as well.
        int anchor;
        MEMORY_BASIC_INFORMATION region{};
        return VirtualQuery(&anchor, &region, sizeof region)
                   ? reinterpret_cast<uintptr_t>(region.AllocationBase) : 0;
#elif defined(__APPLE__)
        const auto top = reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
        return top - pthread_get_stacksize_np(pthread_self());
#elif defined(__linux__)
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) != 0) return 0;
        void* bottom = nullptr;
        size_t size = 0;
        const int result = pthread_attr_getstack(&attr, &bottom, &size);
        pthread_attr_destroy(&attr);
        return result == 0 ? reinterpret_cast<uintptr_t>(bottom) : 0;
#else
        return 0;
#endif
    }();
    // Leave space for the next generated frame, the interpreter and GC. The
    // old slot-count estimate missed temporary arrays and spilled registers:
    // a newly admitted wide dispatcher could use 140 KB for a single call.
    constexpr uintptr_t reserve = 256 * 1024;
    uintptr_t floor = low ? low + reserve : 0;
    if (jit_stack_hi) {
        const auto top = reinterpret_cast<uintptr_t>(jit_stack_hi);
        constexpr uintptr_t budget = 512 * 1024;
        floor = std::max(floor, top > budget ? top - budget : top);
    }
    return floor;
}

void Process::maybe_collect() {
    if (jit_stack_hi) {
        collect_under_compiled();
        return;
    }
    // A mark the helpers are still on is not a state to launch anything from:
    // starting a second collection while the first's helpers were walking old
    // space would be two collectors on one heap. The one legal move is to
    // finalize what the helpers did -- which also folds in whatever minor a
    // full collection would have owed, so the nursery emergency resolves here
    // too. An old space past its threshold *starts* the concurrent mark when
    // it can, and falls back to the whole-world major when the helpers are
    // busy elsewhere or the heap is too small to gain from them. A nursery
    // past its mark just gets the minor.
    if (heap_.marking()) {
        heap_.finalize_concurrent_mark(*this);
    } else if (heap_.major_due()) {
        if (!heap_.start_concurrent_mark(*this)) heap_.major_collect(*this);
    } else {
        heap_.minor_collect(*this);
    }
}

namespace {

/// The collection itself, one frame below the one that pushed the registers:
/// its own frame address is then below every word that matters, the saved
/// registers included. Never inlined, or that would stop being true.
DREAM_NOINLINE void collect_from_here(Process& p, Heap& heap) {
    heap.set_conservative_roots(DREAM_FRAME_ADDRESS(), p.jit_stack_hi);
    // A concurrent mark is not *started* from here. Its snapshot would be
    // read while compiled frames hold values it cannot see, and nothing
    // would pin those at the finalize but a second scan of a stack that has
    // moved on. A whole-world major costs this one collection its overlap.
    if (heap.marking()) {
        heap.finalize_concurrent_mark(p);
    } else if (heap.major_due()) {
        heap.major_collect(p);
    } else {
        heap.minor_collect(p);
    }
    heap.clear_conservative_roots();
}

}  // namespace

DREAM_NOINLINE void Process::collect_under_compiled() {
    // Every callee-saved register onto this frame, where the scan will read
    // it: a compiled caller may be keeping a value in one. (A caller-saved
    // register holding a value across the call that led here was spilled to
    // the caller's frame by the calling convention.) `setjmp` would do it
    // too, but glibc mangles the frame pointer it saves, and a compiled body
    // with no frame pointer uses that register like any other. MSVC has no
    // `__builtin_unwind_init`; `RtlCaptureContext` writes every register into
    // a `CONTEXT` here instead, which is more than is needed and no less.
#if defined(_MSC_VER) && !defined(__clang__)
    CONTEXT registers;
    RtlCaptureContext(&registers);
    collect_from_here(*this, heap_);
    // Reading the capture after the call keeps it, and this frame, alive
    // across it, as the empty `asm` below does for GCC.
    static_cast<void>(*reinterpret_cast<volatile unsigned char*>(&registers));
#else
    __builtin_unwind_init();
    collect_from_here(*this, heap_);
    // Keeps the call above from becoming a tail call, which would pop this
    // frame -- and the registers with it -- before the scan.
    asm volatile("" ::: "memory");
#endif
}

void Process::visit_roots(Heap& heap) {
    // Everything the machine can still reach lives in these few places. That
    // is the whole payoff of keeping process state out of the C++ stack: the
    // root set is a list, not a stack walk. `pins` is the one concession, and
    // it exists so that the handful of C++ frames that genuinely do hold a
    // `Value` across a collection can be written down rather than reasoned
    // about -- see `Process::pins`.
    // The frame register may name a frame on `frames` that has been dropped,
    // and so may a pin -- a nested force saves the frame it will put back --
    // so neither is read as an object until it is known not to be one of
    // those (`FrameStack::holds`).
    if (!frames.holds(frame)) heap.forward(&frame);
    heap.forward(&result);
    heap.forward(&exit_value);
    for (Value& v : stack) heap.forward(&v);
    // Usually empty: a native call pins its callee only when that callee is a
    // host object rather than a builtin immediate, and nothing else pins at all
    // unless a native has been over-applied.
    for (Value& v : pins) {
        if (!frames.holds(v)) heap.forward(&v);
    }
    // A continuation names only frames still live, so its header can be
    // read; one on `frames` is skipped, being visited below as itself.
    for (Cont& c : conts) {
        if (is_obj(c.v1, ObjType::Frame) && (as_obj(c.v1)->gc & GC_SHARED)) continue;
        heap.forward(&c.v1);
    }
    // The interpreter's frames are not on the heap, and are roots: see
    // `FrameStack`. Some may be done with and not yet dropped, which costs a
    // scan and keeps what they hold one collection longer, and nothing else.
    frames.each([&](FrameObj* f) {
        heap.forward(&f->closure);
        Value* slots = f->slots();
        for (uint32_t i = 0; i < f->nslots; ++i) {
            if (slots[i] != NIL_SLOT) heap.forward(&slots[i]);
        }
    });
    for (Value& g : globals) {
        if (g != NIL_SLOT) heap.forward(&g);
    }
    for (Value& v : native_cache) {
        if (v != NIL_SLOT) heap.forward(&v);
    }
    for (Value& v : string_cache) {
        if (v != NIL_SLOT) heap.forward(&v);
    }
    for (Value& v : float_cache) {
        if (v != NIL_SLOT) heap.forward(&v);
    }
}

Value Process::make_error_value(const char* kind, const char* message) {
    uint32_t atom = rt_.intern_atom(kind);
    Value msg = heap_.make_string(message, uint32_t(std::char_traits<char>::length(message)));
    return heap_.make_error(make_atom(atom), msg);
}

void Process::unreachable(const char* what) {
    std::fprintf(stderr, "dream: internal error: %s\n", what);
    std::abort();
}

}  // namespace dream
