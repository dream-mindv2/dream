// The threads a collection borrows.
//
// A process's collector runs on the thread that owns the process, because that
// is the only thread allowed to touch its heap -- but during a collection the
// mutator is stopped, and while it is stopped the heap has no owner in the
// sense that matters. So the work can be split, and on a machine with cores to
// spare the pause shrinks by roughly the number of threads that join in.
//
// docs/gc.md calls for "idle workers join in". This is that, with the workers
// kept apart from the scheduler's: a scheduler worker is asleep waiting for a
// *process*, and waking it for something that is not a process would put the
// collector inside the run queue's handshake, the deadlock detector's idle
// count, and `wait_for_all`. Separate threads that do nothing but this are the
// same bargain without the entanglement -- and they work in the places that
// have no scheduler at all: the unit tests, and a host embedding the VM.
//
// What the scheduler *does* decide is how many of them may run. `set_idle_hint`
// hands the pool the scheduler's live count of idle workers, and a collection
// recruits no more helpers than there are cores nobody is using. A machine
// whose workers are all busy running processes collects on one thread, which is
// the right answer: the other cores have something better to do.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace dream {

class GcPool {
public:
    /// The one pool. Its threads are started on first use and live until the
    /// process exits, because a thread that only ever sleeps costs a stack and
    /// starting one inside a collection would be paying the price the pool
    /// exists to avoid.
    static GcPool& instance();

    /// Run `body(index, count)` on `count` threads -- the caller is index 0 --
    /// and return only when every one of them has finished.
    ///
    /// Returns false without running anything when the pool cannot help:
    /// another heap is already collecting in it, the scheduler says no worker
    /// is idle, or helpers are switched off. The caller then collects alone,
    /// which is always correct; parallelism here is an optimization and never
    /// a requirement.
    bool run(unsigned want, const std::function<void(unsigned, unsigned)>& body);

    /// The other half of a collection, and how a concurrent mark uses the pool.
    /// `start` launches the same round `run` would, but the caller does not
    /// take part -- it hands the body to the helpers and returns at once, so
    /// the mutator can keep running while they work. `join` is what waits for
    /// them, and is a no-op when no round is in flight. Only the heap that
    /// started the round may join it: `busy_` sees to that, the same as `run`.
    ///
    /// The body must not assume the caller ran it. It is run on helper indexes
    /// 1..count-1 only, and `count` is the whole round including the caller's
    /// slot -- a round of one helper, say, arrives as index 1 of 2.
    bool start(unsigned want, const std::function<void(unsigned, unsigned)>& body);
    /// Wait for the round `start` launched on the helpers to finish. A no-op
    /// when no round is running. This is also what a heap that is going away
    /// mid-mark calls, so the helpers finish on memory that is still theirs.
    void join();

    /// The most threads a collection may use, the caller included. Zero means
    /// helpers are switched off entirely (`DREAM_GC_THREADS=1`).
    unsigned capacity() const { return capacity_; }

    /// Point the pool at the scheduler's idle-worker count. Read, never
    /// written, and only as a hint -- it may go stale between the read and the
    /// collection, which costs at worst a little oversubscription.
    void set_idle_hint(const std::atomic<unsigned>* idle) { idle_hint_ = idle; }

    ~GcPool();

private:
    GcPool();
    GcPool(const GcPool&) = delete;
    GcPool& operator=(const GcPool&) = delete;

    void helper_loop(unsigned index);
    /// Start helper threads until there are at least `n` of them.
    void ensure_threads(unsigned n);
    /// The shared front of `start` and `run`: negotiate how many threads the
    /// round may have and launch its helpers. Returns the number of threads --
    /// the caller's slot included -- or zero when the pool declined, in which
    /// case the caller must do the work on its own thread.
    unsigned launch(unsigned want, const std::function<void(unsigned, unsigned)>& body);

    unsigned capacity_ = 0;
    const std::atomic<unsigned>* idle_hint_ = nullptr;

    /// Held by whichever heap is collecting in parallel right now. A second
    /// one does not wait its turn -- waiting would be a pause, and the pause
    /// is the thing being removed -- it collects by itself instead.
    std::atomic<bool> busy_{false};

    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> threads_;
    const std::function<void(unsigned, unsigned)>* body_ = nullptr;
    /// How many threads this round wants, the caller included.
    unsigned round_size_ = 0;
    /// Bumped once per round; a helper compares it against what it last saw,
    /// so a helper that was slow to wake cannot miss a round or run one twice.
    uint64_t generation_ = 0;
    /// Helpers still running the round's body.
    unsigned pending_ = 0;
    bool stopping_ = false;
};

}  // namespace dream
