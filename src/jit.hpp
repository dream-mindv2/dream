// The JIT tier.
//
// The interpreter is the baseline; the JIT is an optimization that must never
// change observable behaviour. Functions are compiled after they prove hot,
// and a compiled function is entered through the same calling convention the
// interpreter uses, so the two tiers interleave freely.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "value.hpp"

namespace dream {

class Runtime;
class Process;

enum class JitPolicy { Calls, CallsPerSize, CallsTimesSize, WorkPerSize, CallsPerSizeSquared, Adaptive };

inline bool parse_jit_policy(std::string_view name, JitPolicy& policy) {
    if (name == "calls") policy = JitPolicy::Calls;
    else if (name == "density" || name == "calls_per_size") policy = JitPolicy::CallsPerSize;
    else if (name == "calls_times_size") policy = JitPolicy::CallsTimesSize;
    else if (name == "work" || name == "work_per_size") policy = JitPolicy::WorkPerSize;
    else if (name == "calls_per_size_squared") policy = JitPolicy::CallsPerSizeSquared;
    else if (name == "adaptive") policy = JitPolicy::Adaptive;
    else return false;
    return true;
}

inline const char* jit_policy_name(JitPolicy policy) {
    switch (policy) {
        case JitPolicy::Calls: return "calls";
        case JitPolicy::CallsPerSize: return "density";
        case JitPolicy::CallsTimesSize: return "calls_times_size";
        case JitPolicy::WorkPerSize: return "work";
        case JitPolicy::CallsPerSizeSquared: return "calls_per_size_squared";
        case JitPolicy::Adaptive: return "adaptive";
    }
    return "density";
}

/// Signature of a JIT-compiled function body.
///
/// `status` reports what happened: 0 the return value is a result, 1 it is an
/// error to raise, 2 the reduction budget ran out mid-loop and the frame holds
/// the loop-carried state, so the caller should resume the body from the top,
/// 3 compiled self recursion ran out of machine stack and the call has to be
/// interpreted instead, and 4 an entry guard did not hold, so this one call has
/// to be interpreted. Neither 3 nor 4 is a failure -- a compiled body is
/// arithmetic and has written nothing down at either point -- and 2 is what
/// keeps compiled loops preemptible.
using CompiledFn = Value (*)(Process* p, Value frame, int* status);

/// The `status` values above, named. The emitter's copies are in jit.cpp,
/// where they have to be constants the IR can use; these are what reads one.
enum JitStatus {
    JitOk = 0,
    JitRaised = 1,
    JitYielded = 2,
    JitTooDeep = 3,
    JitBailed = 4,
};

class Jit {
public:
    static bool available();

    explicit Jit(Runtime& rt);
    ~Jit();
    Jit(const Jit&) = delete;
    Jit& operator=(const Jit&) = delete;

    /// The tier to enter `func_index` with: a compiled body, or null for the
    /// interpreter.
    ///
    /// This runs on *every function entry*, so it is inline, takes no lock,
    /// and crosses into the compiler only on the one entry that makes a
    /// function hot. What makes that possible is that `cached_` holds three
    /// states rather than two -- a compiled body, the rejected marker, or null
    /// while the function is still cold -- so one atomic load answers the
    /// question. With only "compiled" and "everything else", a function LLVM
    /// had refused took the JIT's global mutex and two hash lookups on every
    /// entry, for ever, to be told again that it could not be compiled; in a
    /// program that enters a hundred and eighty million functions that is not
    /// a slow path, it is the program.
    CompiledFn tier(uint32_t func_index) {
        if (func_index >= cached_.size()) return nullptr;
        CompiledFn fn = cached_[func_index].load(std::memory_order_acquire);
        if (fn) {
            if (reinterpret_cast<uintptr_t>(fn) == kQueuedBits) {
                // Still counted while it waits, so the compile threads can
                // take the hottest function first (`take_hottest`).
                counts_[func_index].store(counts_[func_index].load(std::memory_order_relaxed) + 1,
                                          std::memory_order_relaxed);
                return nullptr;
            }
            return reinterpret_cast<uintptr_t>(fn) == kRejectedBits ? nullptr : fn;
        }
        // Counting is a heuristic, so it is a relaxed load and store rather
        // than a read-modify-write: two racing workers may lose a count
        // between them, and a function compiled one entry late is not an
        // observable difference.
        const uint32_t n = counts_[func_index].load(std::memory_order_relaxed) + 1;
        counts_[func_index].store(n, std::memory_order_relaxed);
        if (n < threshold_.load(std::memory_order_relaxed)) return nullptr;
        return on_enter(func_index);
    }

    /// The entry that made `func_index` hot: hand it to the compile thread, or
    /// -- when compiling synchronously -- compile it now or mark it rejected,
    /// so `tier` never asks again. The slow path, and the only one that locks.
    CompiledFn on_enter(uint32_t func_index);

    /// Cheap samples of interpreted work, so one expensive call can become
    /// hot and a frequently entered wrapper does not crowd out a real loop.
    bool samples_work() const { return work_ranking_; }
    void note_work(uint32_t func_index) {
        if (func_index >= work_.size()) return;
        CompiledFn fn = cached_[func_index].load(std::memory_order_relaxed);
        if (fn && fn != queued()) return;
        const uint32_t n = work_[func_index].load(std::memory_order_relaxed) + 1;
        work_[func_index].store(n, std::memory_order_relaxed);
        if (!fn && n >= threshold_.load(std::memory_order_relaxed)) on_enter(func_index);
    }

    /// Stop offering the compiled body for `func_index`. Called when compiled
    /// self recursion ran out of machine stack: the interpreter, whose
    /// recursion is on the heap, takes that function over for the rest of the
    /// run. One relaxed store on a path taken at most once per function.
    void deoptimize(uint32_t func_index);

    /// A compiled body answered `JitBailed`: a guard on a value it carries as
    /// a double did not hold. One bail is a fact about one call, and the call
    /// is simply run interpreted. A run of them is a fact about the program --
    /// a function whose signature says `:float` and whose callers pass it
    /// integers, which the types allow -- and a peeled loop pays a first
    /// iteration for every one, so after `kMaxBails` in a row the function is
    /// the interpreter's. Anything but a bail ends the run (`note_ran`).
    void note_bail(uint32_t func_index) {
        if (func_index >= bails_.size()) return;
        const uint8_t n = uint8_t(bails_[func_index].load(std::memory_order_relaxed) + 1);
        bails_[func_index].store(n, std::memory_order_relaxed);
        if (n >= kMaxBails) deoptimize(func_index);
    }
    /// The compiled body ran: a load, and a store only when a run of bails is
    /// being ended.
    void note_ran(uint32_t func_index) {
        if (func_index < bails_.size() && bails_[func_index].load(std::memory_order_relaxed)) {
            bails_[func_index].store(0, std::memory_order_relaxed);
        }
    }

    /// Compile now, regardless of temperature. Returns nullptr on failure.
    CompiledFn compile(uint32_t func_index, std::string* error);

    /// What `precompile` did.
    struct Precompiled {
        uint32_t compiled = 0;
        uint32_t refused = 0;
        double ms = 0;
    };
    /// Compile every function the tier can take, on `threads` threads, before
    /// the program starts: `--precompile`. Each answer goes straight into the
    /// tier table, so the first call of a function is already compiled and
    /// nothing is queued while the program runs.
    Precompiled precompile(unsigned threads);

    void set_threshold(uint32_t calls);
    /// How many compile threads to start, before the first function gets
    /// hot. `--jit-threads`; otherwise `DREAM_JIT_THREADS`, or two.
    void set_threads(unsigned n);
    /// Choose before executing the image; a CLI setting overrides the environment.
    void set_policy(JitPolicy policy);
    JitPolicy policy() const;
    uint32_t threshold() const;
    uint64_t compiled_count() const;
    /// Human-readable IR for one function, for `--dump-jit`.
    std::string dump_ir(uint32_t func_index);

private:
    /// What `cached_` holds for a function compilation has been tried on and
    /// failed. Any value that cannot be a function pointer would do; one is
    /// the cheapest to compare against.
    static constexpr uintptr_t kRejectedBits = 1;
    static CompiledFn rejected() { return reinterpret_cast<CompiledFn>(kRejectedBits); }
    /// What `cached_` holds for a function waiting on the compile thread. The
    /// interpreter runs it meanwhile, exactly as it ran it while it was cold;
    /// `tier` answers both of the markers with the same single compare.
    static constexpr uintptr_t kQueuedBits = 2;
    static CompiledFn queued() { return reinterpret_cast<CompiledFn>(kQueuedBits); }

    /// The compile thread's loop. See "Compiling in the background" in jit.cpp.
    void compile_worker();

    /// Dense per-function entry counters and compiled-body cache, indexed by
    /// image function index. They live on `Jit` rather than in `Impl` so
    /// `cached_compiled`, which runs on every function entry, can read the
    /// cache with a single load and no PIMPL indirection. Both are sized once,
    /// in the constructor, after the image has loaded.
    std::vector<std::atomic<uint32_t>> counts_;
    std::vector<std::atomic<uint32_t>> work_;
    bool work_ranking_ = false;
    std::vector<std::atomic<CompiledFn>> cached_;
    /// Consecutive bails per function; see `note_bail`. Relaxed, like
    /// `counts_`, and for the same reason: a lost count moves a heuristic.
    std::vector<std::atomic<uint8_t>> bails_;
    static constexpr uint8_t kMaxBails = 16;
    /// Mirrors the threshold inside `Impl`, so `tier` can read it without
    /// reaching through the PIMPL.
    std::atomic<uint32_t> threshold_{0};

    /// Compile one function, from any thread; see "Several compile threads"
    /// in jit.cpp.
    CompiledFn compile_one(uint32_t func_index, std::string* error);
    /// The queued function with the most entries counted per node of its
    /// body, taken off the queue -- a compile's cost grows with its body, so
    /// it is calls divided by size that repays the budget quickest. Caller
    /// holds the queue's lock.
    uint32_t take_hottest();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dream
