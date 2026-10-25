// Scheduling.
//
// Processes are multiplexed onto a small pool of OS threads. Each worker owns
// a run queue and steals from its neighbours when it goes hungry. A process
// runs for a fixed number of reductions and is then put back on the queue,
// whatever it was in the middle of -- the same preemption model as BEAM, and
// the reason a tight loop in one process cannot starve the others.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>

#include "process.hpp"
#include "stack_thread.hpp"
#include "runtime.hpp"

namespace dream {

class Scheduler {
public:
    enum class JoinState { Ready, Failed, Blocked };

    /// The reduction budget one process gets before it is rescheduled.
    static constexpr int64_t REDUCTIONS_PER_SLICE = 4000;

    Scheduler(Runtime& rt, unsigned worker_count);
    ~Scheduler();

    /// How many workers to run when nobody said: one per core this process may
    /// use (`usable_cores()`, which a pinned or containerized process has
    /// fewer of than the machine).
    ///
    /// It was capped at eight, because an idle worker woke every 500us to look
    /// for work and a single-process program paid for every core the machine
    /// had: 2.75 s for a self-compile with four workers, 3.03 s with
    /// twenty-four. Idle workers now back off (`worker_loop`), and a compile is
    /// no longer one process -- it is a few hundred parts, and on a 24-thread
    /// machine the cap was 4.9 s against 4.4 for sixteen workers. `-j` still
    /// says how many.
    static unsigned default_workers();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void start();
    /// Drive the same scheduler from an embedding's event loop. No OS workers
    /// are started. Each call runs at most max_slices and returns to the host.
    /// Call only on the owning thread, before start(), and never mix modes.
    bool pump(unsigned max_slices = 1);
    /// Delay until the next timer, in milliseconds; -1 if none is armed.
    /// Intended for a cooperative host deciding when to call pump again.
    int64_t next_timer_delay_ms();
    /// Ask the workers to stop and join them.
    void stop();

    /// Register a new process. It is not runnable until `enqueue`.
    std::shared_ptr<Process> create_process();
    /// Make `p` runnable on some worker other than `avoid`, when there is one.
    void enqueue(const std::shared_ptr<Process>& p, unsigned avoid = UINT32_MAX);

    /// Write a snapshot of every process and open handle to stderr.
    ///
    /// The same report `vm.dump!` produces, reachable from C++ so that a
    /// program which has stopped making progress can still be asked what it is
    /// doing -- at that point no Dream code can run to ask on its own.
    void dump(const char* why) const;

    /// Block until every process has finished, or until deadlock is detected.
    /// Returns false when nothing could run but processes were still alive.
    bool wait_for_all();

    /// Run a program whose root process is primed: queue it, start the
    /// workers, wait, stop. Answers what `wait_for_all` does.
    ///
    /// One method because the order is the whole of it: the command line and
    /// the embedding API had each been fixed to get it right, and compile-time
    /// evaluation still had it wrong. A process is counted as active by `enqueue`, so a
    /// scheduler started first has a window in which the root is live,
    /// nothing is active and every worker is idle -- which is the definition
    /// the deadlock check tests, and it fires. The flag is sticky, so the run
    /// answers "deadlocked" however it goes on. A worker waits 500us before
    /// it asks, so the window only opens when the starting thread is
    /// descheduled for longer than that: a loaded machine, which is why a
    /// compile-time `comp` failed one run in several under `mind test` and
    /// never under `just test`. Queued before the workers exist, there is no
    /// such instant.
    bool run(const std::shared_ptr<Process>& root);

    // --- process-facing operations ---
    uint64_t spawn_from(Process& parent, Value work);
    bool send(Process& sender, uint64_t target, Value message);
    bool receive(Process& p, Value* out);
    JoinState join(Process& p, uint64_t target, Value* out);

    /// Ask `target` to stop, with `reason`. It ends at the start of its next
    /// slice as a failure of kind `:killed` whose payload is the reason, which
    /// is what its joiners and monitors are told. A kill cannot be caught: the
    /// process is not unwound, it is simply not run again. Answers false for a
    /// pid that does not exist; killing a finished process does nothing.
    bool kill(Process& from, uint64_t target, Value reason);

    /// Send `p` `[:down, target, outcome]` when `target` ends -- at once, if
    /// it already has. `outcome` is `[:ok, value]` or `[:error, e]`, the shape
    /// `proc.outcome!` answers, and a failure delivered this way counts as
    /// handled, as one delivered to a joiner does.
    void monitor(Process& p, uint64_t target);
    /// Undo one `monitor` of `target` by `p`. A `[:down ..]` already sent
    /// stays in the mailbox; the caller takes it out if it cares.
    void demonitor(Process& p, uint64_t target);

    /// Wake `pid` at `deadline_ns` on the monotonic clock `vm.now_ns!` reads.
    /// Answers a sequence number for `cancel_timer`. A pending timer counts as
    /// a wait on something outside the scheduler, as a descriptor does, so a
    /// program whose every process is asleep is not mistaken for a deadlock.
    uint64_t arm_timer(uint64_t pid, int64_t deadline_ns);
    /// Forget a timer that has not fired. Harmless on one that has.
    void cancel_timer(uint64_t seq);

    /// Start a census of the shared area -- or, with one already running, ask
    /// for another when it ends. Every live process reports once: a parked one
    /// is censused here and now, under its lock, and a running or queued one
    /// at its next slice or as it parks. The last report walks mailboxes, kill
    /// reasons and finished processes, and frees what nobody holds. See
    /// "Freeing what was shared" in docs/gc.md.
    void request_census();

    /// Make a parked process runnable again. Public because the IO poller
    /// calls it from its own thread when a descriptor becomes ready, which is
    /// the same handshake a message arriving uses.
    void wake(uint64_t pid);

    /// The number that names this scheduler to threads it does not own.
    ///
    /// The IO poller and a child process's reaper finish on their own threads,
    /// at a moment nothing orders against this scheduler going away: a program
    /// can end, and its `Scheduler` -- a local in `dream_main`, or in a
    /// compile-time evaluation -- be destroyed, while a child it started is
    /// still being reaped. They used to hold a `Scheduler*` and woke into
    /// freed memory then (ThreadSanitizer: `enqueue` reading `workers_` after
    /// the destructor). So they hold this instead, and reach the scheduler
    /// only through `wake_external`, which finds nothing once the scheduler is
    /// gone. A serial rather than the pointer, because the next scheduler is
    /// often built at the same address -- the compile-time ones are locals,
    /// one after another -- and a stale pointer would wake into that.
    uint64_t serial() const { return serial_; }

    /// `wake(pid)` and then `note_io_wait(false)` on the scheduler `serial`
    /// names, in that order (see `run_child` in os.cpp), or nothing if it no
    /// longer exists.
    static void wake_external(uint64_t serial, uint64_t pid);

    /// `note_io_wait(false)`, `count` times, on the scheduler `serial` names,
    /// or nothing if it no longer exists.
    static void release_io_waits(uint64_t serial, size_t count);

    /// A process is waiting on something outside the scheduler -- a descriptor
    /// the poller is watching. Such a process is not deadlocked even though
    /// nothing in the system can run: the kernel still owes it an answer.
    void note_io_wait(bool waiting) {
        if (waiting) {
            io_waiters_.fetch_add(1, std::memory_order_relaxed);
        } else {
            io_waiters_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    size_t io_waiters() const { return io_waiters_.load(std::memory_order_relaxed); }

    // --- introspection, for `std.vm` ---
    size_t live() const { return live_.load(std::memory_order_relaxed); }
    size_t runnable() const { return runnable_.load(std::memory_order_relaxed); }
    unsigned idle_workers() const { return idle_workers_.load(std::memory_order_relaxed); }
    /// How many processes are sitting on run queues right now.
    size_t queued() const;

    unsigned worker_count() const { return unsigned(workers_.size()); }
    /// Errors raised by processes that nobody was waiting on. A failure that
    /// reaches a joiner is that joiner's business, not the runtime's, so it is
    /// dropped from this list when it is delivered.
    std::vector<std::string> take_failures();
    uint64_t total_reductions() const { return total_reductions_.load(); }
    bool deadlocked() const { return deadlocked_.load(); }

private:
    uint64_t serial_ = 0;

    struct Worker {
        std::mutex mutex;
        std::deque<std::shared_ptr<Process>> queue;
        /// Runs Dream, so its stack is the VM's to size (`kWorkerStack`).
        StackThread thread;
    };

    void worker_loop(unsigned index);
    std::shared_ptr<Process> take_local(unsigned index);
    std::shared_ptr<Process> steal(unsigned thief);
    void run_slice(const std::shared_ptr<Process>& p, unsigned index);
    void requeue(unsigned index, const std::shared_ptr<Process>& p);
    void finish(const std::shared_ptr<Process>& p);
    /// The message a monitor is sent when `p` ends. Called with `p` finished.
    std::unique_ptr<Message> down_message(Process& p);
    /// Put a message in `target`'s mailbox and wake it. False for no such pid.
    bool deliver(uint64_t target, std::unique_ptr<Message> msg);
    void timer_loop();

    /// A process has been censused for `round`, or has ended.
    void census_report(uint64_t pid, uint64_t round);
    /// Census `p`'s heap for `round` if it owes one. The caller is the one
    /// thread allowed to touch `p` right now.
    void census_if_due(Process& p);
    /// The round's last report: the walks, and the freeing.
    void census_finish(uint64_t round);
    void note_idle(bool idle);
    /// Tell an idle worker there is work, if one is idle.
    void wake_idle();

    Runtime& rt_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<unsigned> next_worker_{0};
    std::atomic<bool> running_{false};
    bool cooperative_ = false;

    /// Processes that exist and have not finished.
    std::atomic<size_t> live_{0};
    /// Processes sitting on a run queue or currently executing.
    std::atomic<size_t> runnable_{0};
    /// Processes that are queued **or running** -- one counter, deliberately.
    ///
    /// This is what the deadlock check asks about, and it is one atomic rather
    /// than two because the question has to be answered without observing a
    /// half-finished transition. The previous answer was "walk every queue,
    /// then read `runnable_`", and it has a race that predates the counter:
    /// a worker re-enqueueing its process pushes it onto a queue and only then
    /// decrements `runnable_`, so a walk that has already passed that queue can
    /// go on to read `runnable_` after the decrement and conclude that a
    /// perfectly healthy program is deadlocked. It needs all workers idle to
    /// fire, which is why it stayed hidden while the default was one worker per
    /// core and surfaced as soon as that became eight.
    ///
    /// A process joins this count when it is enqueued and leaves it when its
    /// slice ends without re-enqueueing -- so a slice that hands the process
    /// straight back increments before it decrements and the count never dips.
    /// Parking is a decrement, which is correct: a parked process is waiting
    /// for a `send!` from someone still counted here, or for the poller, which
    /// `io_waiters_` covers.
    std::atomic<size_t> active_{0};
    /// Processes sitting on some worker's queue, waiting to be picked up.
    ///
    /// A hint for `steal`, and deliberately nothing more. It lets an idle
    /// worker skip the walk over every other worker's queue -- one mutex
    /// acquisition apiece, paid by every idle worker every 500us, which on a
    /// machine with many cores running a program with one process is
    /// O(workers^2) lock operations per poll, all of it contending with the
    /// single worker actually running the program.
    ///
    /// It carries no correctness: reading it stale costs at most a missed
    /// steal, which the 500us retry and `enqueue`'s notify both cover, and
    /// which the walk itself could already do by passing a victim just before
    /// it was given work. The deadlock check next to it needs a stronger
    /// answer than this and keeps its walk -- see `worker_loop` for why.
    std::atomic<size_t> queued_{0};
    std::atomic<unsigned> idle_workers_{0};
    /// Processes parked on a descriptor rather than on another process.
    std::atomic<size_t> io_waiters_{0};
    std::atomic<uint64_t> total_reductions_{0};
    std::atomic<bool> deadlocked_{false};
    /// The process `run` was handed, and whether it has failed. An error
    /// nothing caught in the root ends the run there, live processes or not:
    /// see `finish`.
    std::atomic<uint64_t> root_id_{0};
    std::atomic<bool> root_failed_{false};

    /// Timed waits, by deadline. One thread sleeps until the earliest and
    /// wakes whoever it belongs to; `timer_index_` is how a cancel finds an
    /// entry without a walk. A wake is only ever a hint -- a woken native
    /// looks again at what it was waiting for -- so a timer that fires just
    /// after the message it was racing costs one spurious wake.
    struct TimerEntry {
        uint64_t seq;
        uint64_t pid;
    };
    std::mutex timer_mutex_;
    std::condition_variable timer_cv_;
    std::multimap<int64_t, TimerEntry> timers_;
    std::unordered_map<uint64_t, std::multimap<int64_t, TimerEntry>::iterator> timer_index_;
    uint64_t next_timer_ = 1;
    bool timers_running_ = false;
    std::thread timer_thread_;

    /// The census round in progress, or 0, and the processes it is waiting to
    /// hear from. Lock order: a process's `sched_mutex`, then this.
    std::mutex census_mutex_;
    uint64_t census_round_ = 0;
    std::unordered_map<uint64_t, bool> census_waiting_;
    bool census_again_ = false;

    std::mutex failures_mutex_;
    std::unordered_map<uint64_t, std::string> failures_;

    std::mutex idle_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    /// Wake `wait_for_all` after changing what it waits on. See the definition.
    void notify_done();
};

/// Write what `DREAM_TIMELINE` recorded, for a run ending now from inside
/// `exiting`'s slice (`os.exit!`), which no scheduler will see stop.
void write_timeline(Process& exiting);
/// Whether `DREAM_TIMELINE` is recording, and its clock in microseconds.
bool timeline_on();
int64_t timeline_now();
/// A native call of `p`'s that ran from `t0` until now without yielding: a
/// long one is a slice no scheduler could have cut.
void timeline_native(Process& p, const std::string& name, int64_t t0);
/// One of a profiled process's steps, in function `fi`, sampled by its time:
/// with `--profile`, every 1024th step a profiled process takes is recorded,
/// so its work can be told apart by when it ran.
void timeline_sample(Process& p, uint32_t fi);

}  // namespace dream
