#include "scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "io.hpp"

#include <chrono>
#include <shared_mutex>

#include "builtins.hpp"
#include "cores.hpp"
#include "gc_pool.hpp"
#include "interp.hpp"

namespace dream {

using namespace std::chrono_literals;

// The schedulers that exist, by serial, for the threads that wake into one
// without owning it -- see `Scheduler::serial`. A wake holds the shared lock
// for as long as it is touching the scheduler, and a destructor takes the
// exclusive one to leave, so a destructor waits out a wake already under way
// and every wake after it finds nothing.
namespace {
std::shared_mutex& live_mutex() {
    static std::shared_mutex m;
    return m;
}
std::unordered_map<uint64_t, Scheduler*>& live_schedulers() {
    static std::unordered_map<uint64_t, Scheduler*> m;
    return m;
}
std::atomic<uint64_t> next_serial{1};
}  // namespace

Scheduler::Scheduler(Runtime& rt, unsigned worker_count) : rt_(rt) {
    if (worker_count == 0) worker_count = 1;
    for (unsigned i = 0; i < worker_count; ++i) {
        workers_.push_back(std::make_unique<Worker>());
    }
    serial_ = next_serial.fetch_add(1);
    {
        std::unique_lock<std::shared_mutex> g(live_mutex());
        live_schedulers()[serial_] = this;
    }
    rt_.set_scheduler(this);
}

Scheduler::~Scheduler() {
    // Unreachable from outside first, then stopped: a wake that arrives in
    // between finds no scheduler, which is right -- nothing will run again.
    {
        std::unique_lock<std::shared_mutex> g(live_mutex());
        live_schedulers().erase(serial_);
    }
    stop();
    // And the runtime forgets it, so that nothing asking the runtime later --
    // its own destructor, closing what the program left open -- is handed a
    // pointer to this.
    if (rt_.scheduler() == this) rt_.set_scheduler(nullptr);
}

void Scheduler::wake_external(uint64_t serial, uint64_t pid) {
    std::shared_lock<std::shared_mutex> g(live_mutex());
    auto it = live_schedulers().find(serial);
    if (it == live_schedulers().end()) return;
    it->second->wake(pid);
    it->second->note_io_wait(false);
}

void Scheduler::release_io_waits(uint64_t serial, size_t count) {
    std::shared_lock<std::shared_mutex> g(live_mutex());
    auto it = live_schedulers().find(serial);
    if (it == live_schedulers().end()) return;
    for (size_t i = 0; i < count; ++i) it->second->note_io_wait(false);
}

unsigned Scheduler::default_workers() {
    return std::min(usable_cores(), 8u);
}

void Scheduler::start() {
    if (running_.exchange(true)) return;
    // A collection may borrow threads, but only as many as there are workers
    // with nothing to run: the cores a process is using are not the
    // collector's to take. See gc_pool.hpp.
    GcPool::instance().set_idle_hint(&idle_workers_);
    for (unsigned i = 0; i < workers_.size(); ++i) {
        workers_[i]->thread = std::thread([this, i] { worker_loop(i); });
    }
    {
        std::lock_guard<std::mutex> g(timer_mutex_);
        timers_running_ = true;
    }
    timer_thread_ = std::thread([this] { timer_loop(); });
}

void Scheduler::stop() {
    if (!running_.exchange(false)) return;
    {
        std::lock_guard<std::mutex> g(timer_mutex_);
        timers_running_ = false;
        // Nothing will run to be woken now, so a timer still pending is
        // dropped along with the count of waits it was holding up.
        for (size_t i = 0; i < timers_.size(); ++i) note_io_wait(false);
        timers_.clear();
        timer_index_.clear();
    }
    timer_cv_.notify_all();
    if (timer_thread_.joinable()) timer_thread_.join();
    GcPool::instance().set_idle_hint(nullptr);
    work_cv_.notify_all();
    for (auto& w : workers_) {
        if (w->thread.joinable()) w->thread.join();
    }
}

std::shared_ptr<Process> Scheduler::create_process() {
    auto p = rt_.spawn_process();
    live_.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void Scheduler::enqueue(const std::shared_ptr<Process>& p) {
    p->status.store(ProcStatus::Runnable, std::memory_order_relaxed);
    // Round-robin placement. A process that blocks and wakes may land on a
    // different worker than it started on, which is fine: nothing in a process
    // is tied to a thread.
    unsigned i = next_worker_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    {
        std::lock_guard<std::mutex> g(workers_[i]->mutex);
        workers_[i]->queue.push_back(p);
        queued_.fetch_add(1, std::memory_order_relaxed);
    }
    // Sequentially consistent, and once per slice rather than per reduction,
    // so the ordering costs nothing worth measuring and buys the deadlock
    // check a count it can trust. See `active_`.
    active_.fetch_add(1);
    work_cv_.notify_one();
}

std::shared_ptr<Process> Scheduler::take_local(unsigned index) {
    Worker& w = *workers_[index];
    std::lock_guard<std::mutex> g(w.mutex);
    if (w.queue.empty()) return nullptr;
    auto p = std::move(w.queue.front());
    w.queue.pop_front();
    queued_.fetch_sub(1, std::memory_order_relaxed);
    // Counted as runnable while still holding the queue lock. A live process
    // must be findable at every instant -- in a queue, in `runnable_`, or
    // parked -- and counting it after the pop instead would leave a moment
    // when it is in none of those, which the deadlock check reads as "nothing
    // left to run".
    runnable_.fetch_add(1, std::memory_order_relaxed);
    return p;
}

std::shared_ptr<Process> Scheduler::steal(unsigned thief) {
    // Nothing is queued anywhere, so there is nobody to steal from and no
    // reason to take a single lock to find that out. This is the common case
    // for a program with one process, where every worker but one is idle and
    // asking. A process enqueued immediately after this load is not missed:
    // `enqueue` notifies the condition variable and the loop comes back round,
    // which is exactly what happened before when the walk passed a victim just
    // before it was given work.
    if (queued_.load(std::memory_order_relaxed) == 0) return nullptr;

    // Take from the back of a victim's queue: the front is the work it is
    // about to run, and leaving that alone keeps its cache warm.
    for (unsigned n = 1; n < workers_.size(); ++n) {
        unsigned victim = (thief + n) % workers_.size();
        Worker& w = *workers_[victim];
        std::lock_guard<std::mutex> g(w.mutex);
        if (w.queue.empty()) continue;
        auto p = std::move(w.queue.back());
        w.queue.pop_back();
        queued_.fetch_sub(1, std::memory_order_relaxed);
        runnable_.fetch_add(1, std::memory_order_relaxed);  // see `take_local`
        return p;
    }
    return nullptr;
}

void Scheduler::note_idle(bool idle) {
    if (idle) {
        idle_workers_.fetch_add(1, std::memory_order_relaxed);
    } else {
        idle_workers_.fetch_sub(1, std::memory_order_relaxed);
    }
}

void Scheduler::worker_loop(unsigned index) {
    while (running_.load(std::memory_order_relaxed)) {
        std::shared_ptr<Process> p = take_local(index);
        if (!p) p = steal(index);

        if (!p) {
            note_idle(true);
            {
                std::unique_lock<std::mutex> lk(idle_mutex_);
                work_cv_.wait_for(lk, 500us);
            }
            note_idle(false);

            // Nothing runnable anywhere, every worker asleep, yet processes
            // remain: they are all parked on a message that will never come.
            if (live_.load() > 0 && idle_workers_.load() + 1 >= workers_.size()) {
                // One load, and the reason it is one is in `active_`: asking
                // this as "walk every queue, then read `runnable_`" cannot be
                // done without observing a process mid-transition between the
                // two, and answers "deadlocked" when it does.
                //
                // A process waiting on a descriptor is not deadlocked: the
                // poller thread still owes it a wake-up, and nothing inside
                // the scheduler can produce that.
                if (active_.load() == 0 && io_waiters_.load() == 0) {
                    deadlocked_.store(true);
                    notify_done();
                }
            }
            continue;
        }

        // Already counted by whichever of `take_local` or `steal` produced it.
        run_slice(p, index);
        runnable_.fetch_sub(1, std::memory_order_relaxed);
        // Leaves the active count only if the slice did not hand the process
        // back: `run_slice` ends in `enqueue`, which has already incremented.
        active_.fetch_sub(1);
    }
}

void Scheduler::run_slice(const std::shared_ptr<Process>& p, unsigned index) {
    // Start unwinding between slices, then let the machine run cleanup.
    if (p->kill_requested.load(std::memory_order_acquire) &&
        !p->kill_unwinding && p->cleanup_depth == 0) {
        if (p->wait_reason.load(std::memory_order_relaxed) == WaitReason::Io) {
            int fd = p->wait_fd.load(std::memory_order_relaxed);
            if (fd >= 0) io_cancel_wait(fd, p->id());
        }
        if (p->timer_seq) {
            cancel_timer(p->timer_seq);
            p->timer_seq = 0;
        }
        p->park_requested = false;
        p->kill_unwinding = true;
        p->result = p->kill_error();
        p->mode = Mode::Raise;
    }
    census_if_due(*p);
    p->status.store(ProcStatus::Running, std::memory_order_relaxed);
    p->wait_reason.store(WaitReason::None, std::memory_order_relaxed);
    p->wait_fd.store(-1, std::memory_order_relaxed);
    uint64_t before = p->total_reductions;

    run_process(*p, REDUCTIONS_PER_SLICE);

    total_reductions_.fetch_add(p->total_reductions - before, std::memory_order_relaxed);

    if (p->mode == Mode::Halted) {
        finish(p);
        return;
    }

    {
        // The parking handshake. This worker has stopped touching the machine
        // state, so it is now safe to publish that the process is parked -- and
        // to notice a wake that arrived while it was still running.
        std::lock_guard<std::mutex> g(p->sched_mutex);
        if (p->park_requested) {
            p->park_requested = false;
            // A census asked of a running process is answered before it parks:
            // parked, nothing would run it to answer. It is not answered *here*
            // -- this lock is held, and the last report takes every process's
            // -- but by not parking: the process goes round once more, as a
            // wake that beat the park sends it, and answers at the start of
            // that slice. The blocking native it was in looks again and asks
            // to park again.
            if (!p->wake_pending && !p->census_due.load(std::memory_order_acquire) &&
                !(p->kill_requested.load(std::memory_order_acquire) &&
                  !p->kill_unwinding && p->cleanup_depth == 0)) {
                p->status.store(ProcStatus::Waiting, std::memory_order_release);
                return;
            }
            // Woken before we could park: fall through and run again. The
            // blocking builtin re-checks its condition, so a spurious wake
            // costs one reduction and nothing else.
            p->wake_pending = false;
        }
    }

    // Budget spent mid-computation, or a wake beat the park. Put it back and
    // let something else run; the whole state is in the Process, so there is
    // nothing to save.
    requeue(index, p);
}

// A process whose slice ran out goes back on the queue of the worker that ran
// it, not round-robin like a new one. `enqueue` spreads processes, which is
// right for one that has just been made or woken; for one that is merely
// yielding it meant a program of one process moved to a different worker --
// and so, usually, a different core -- every 4000 reductions, paying a futex
// wake of a sleeping thread and a cold cache each time. On a self-compile that
// was forty thousand migrations. Kept local, the worker picks it straight back
// up, and nobody is woken unless something else is already waiting behind it,
// in which case an idle worker is told there is work to steal.
void Scheduler::requeue(unsigned index, const std::shared_ptr<Process>& p) {
    p->status.store(ProcStatus::Runnable, std::memory_order_relaxed);
    bool others;
    {
        std::lock_guard<std::mutex> g(workers_[index]->mutex);
        workers_[index]->queue.push_back(p);
        queued_.fetch_add(1, std::memory_order_relaxed);
        others = workers_[index]->queue.size() > 1;
    }
    active_.fetch_add(1);  // see `enqueue`
    if (others) work_cv_.notify_one();
}

size_t Scheduler::queued() const {
    size_t n = 0;
    for (auto& w : workers_) {
        std::lock_guard<std::mutex> g(w->mutex);
        n += w->queue.size();
    }
    return n;
}

std::vector<std::string> Scheduler::take_failures() {
    std::lock_guard<std::mutex> g(failures_mutex_);
    std::vector<std::string> out;
    out.reserve(failures_.size());
    for (auto& [pid, text] : failures_) out.push_back(text);
    failures_.clear();
    return out;
}

void Scheduler::finish(const std::shared_ptr<Process>& p) {
    // Before anyone can be told the process is done: a joiner that goes on to
    // look at what the C side holds must find its resources already let go.
    release_foreign(*p);
    std::vector<uint64_t> waiters;
    std::vector<uint64_t> monitors;
    bool had_waiters = false;
    {
        std::lock_guard<std::mutex> g(p->waiters_mutex);
        p->exit_value = p->result;
        // A monitor is told the outcome as a value, as a joiner is, so a
        // failure it hears about is handled by the same reasoning.
        had_waiters = !p->waiters.empty() || !p->monitors.empty();
        p->status.store(p->failed ? ProcStatus::Failed : ProcStatus::Finished,
                        std::memory_order_release);
        waiters.swap(p->waiters);
        monitors.swap(p->monitors);

        // Record the failure only when nobody is waiting for it. A joiner
        // receives the error as a value and decides what it means; reporting it
        // as well would turn a handled failure into a spurious complaint. The
        // insert stays under the waiters lock, because that is the lock a
        // joiner takes to confess a process done: a joiner that reads the
        // failed status here must also see the record just made. Recording
        // outside the lock let a join that ran in the gap erase nothing -- the
        // done status was set and the failure not yet recorded -- and the
        // handled failure was then reported at shutdown as uncaught, whenever
        // the timer happened to put the join exactly there.
        if (p->failed && !had_waiters) {
            std::string text = describe_failure(*p, p->result);
            std::lock_guard<std::mutex> gf(failures_mutex_);
            failures_.emplace(p->id(), "process " + std::to_string(p->id()) + ": " + text);
        }
    }

    // The root failing is the end of the program, whoever else is still
    // alive. Nothing can catch an error the root did not, and waiting for the
    // rest -- the run's usual end -- reported it only once they had all ended:
    // a program whose `main!` raised while a registry or a listener it started
    // was still up printed nothing and never exited (examples/16_registry,
    // about one run in forty). The deadlock check cannot see that case,
    // because a listener is parked on IO and not on a message.
    if (p->failed && p->id() == root_id_.load()) root_failed_.store(true);

    // Sent before `live_` drops, so a program whose last two processes are a
    // monitor and what it watches cannot be counted finished in between.
    for (uint64_t id : monitors) deliver(id, down_message(*p));
    if (p->timer_seq) {
        cancel_timer(p->timer_seq);
        p->timer_seq = 0;
    }
    // Ended, so it has nothing more to say to a census: what it left is
    // `exit_value`, which the round's last report walks.
    uint64_t round;
    {
        std::lock_guard<std::mutex> g(census_mutex_);
        round = census_round_;
    }
    if (round) census_report(p->id(), round);
    live_.fetch_sub(1, std::memory_order_acq_rel);
    for (uint64_t id : waiters) wake(id);
    notify_done();
}

std::unique_ptr<Message> Scheduler::down_message(Process& p) {
    auto msg = std::make_unique<Message>();
    msg->heap = std::make_unique<Heap>(1024);
    Heap& h = *msg->heap;
    Value value = Heap::copy_between(h, p.exit_value);
    // `Result.ok value` or `Result.error e`, the prelude's union, as a
    // variant record of one field -- what `syntax.union_decl`'s constructor
    // makes, so the joiner's `Result.ok(v)` pattern takes it apart.
    Value outcome = make_result(h, rt_, !p.failed, value);
    Value pid = h.make_pid(p.id());
    msg->value = h.make_cons(make_atom(rt_.intern_atom("down")),
                             h.make_cons(pid, h.make_cons(outcome, NIL)));
    msg->from_pid = p.id();
    return msg;
}

bool Scheduler::deliver(uint64_t target, std::unique_ptr<Message> msg) {
    auto proc = rt_.find_process(target);
    if (!proc) return false;
    proc->mailbox.push(std::move(msg));
    // Push first, then wake: a receiver that parks between those two steps is
    // still found by `wake`, because parking takes the same lock.
    wake(target);
    return true;
}

bool Scheduler::kill(Process& from, uint64_t target, Value reason) {
    auto proc = rt_.find_process(target);
    if (!proc) return false;
    if (proc->is_done()) return true;
    {
        std::lock_guard<std::mutex> g(proc->sched_mutex);
        if (!proc->kill_reason) {
            auto msg = std::make_unique<Message>();
            msg->heap = std::make_unique<Heap>(256);
            msg->value = Heap::copy_between(*msg->heap, reason);
            msg->from_pid = from.id();
            proc->kill_reason = std::move(msg);
        }
    }
    proc->kill_requested.store(true, std::memory_order_release);
    // Parked, it has to run once more to be stopped; running, the note this
    // leaves keeps it from parking before the slice that stops it.
    wake(target);
    return true;
}

void Scheduler::monitor(Process& p, uint64_t target) {
    auto proc = rt_.find_process(target);
    if (!proc) return;
    std::unique_ptr<Message> now;
    {
        std::lock_guard<std::mutex> g(proc->waiters_mutex);
        if (!proc->is_done()) {
            proc->monitors.push_back(p.id());
            return;
        }
        now = down_message(*proc);
        if (proc->failed) {
            std::lock_guard<std::mutex> gf(failures_mutex_);
            failures_.erase(target);
        }
    }
    deliver(p.id(), std::move(now));
}

void Scheduler::demonitor(Process& p, uint64_t target) {
    auto proc = rt_.find_process(target);
    if (!proc) return;
    std::lock_guard<std::mutex> g(proc->waiters_mutex);
    auto& ms = proc->monitors;
    auto it = std::find(ms.begin(), ms.end(), p.id());
    if (it != ms.end()) ms.erase(it);
}

// ---------------------------------------------------------------------------
// The census of the shared area
// ---------------------------------------------------------------------------

void Scheduler::request_census() {
    uint64_t round;
    std::vector<std::shared_ptr<Process>> procs = rt_.all_processes();
    {
        std::lock_guard<std::mutex> g(census_mutex_);
        if (census_round_) {
            census_again_ = true;
            return;
        }
        round = rt_.shared().begin_round();
        census_round_ = round;
        census_waiting_.clear();
        for (auto& p : procs)
            if (!p->is_done()) census_waiting_.emplace(p->id(), true);
        if (census_waiting_.empty()) {
            census_finish(round);
            return;
        }
    }
    for (auto& p : procs) {
        if (p->is_done()) continue;
        std::unique_lock<std::mutex> g(p->sched_mutex);
        if (p->status.load(std::memory_order_acquire) == ProcStatus::Waiting) {
            // Parked: nobody is touching it, and a wake has to take this lock
            // before it can make it runnable. So its heap is censused here.
            p->heap().census_collect(*p, round);
            g.unlock();
            census_report(p->id(), round);
        } else {
            p->census_due.store(round, std::memory_order_release);
        }
    }
}

void Scheduler::census_if_due(Process& p) {
    uint64_t due = p.census_due.exchange(0, std::memory_order_acq_rel);
    if (!due) return;
    p.heap().census_collect(p, due);
    census_report(p.id(), due);
}

void Scheduler::census_report(uint64_t pid, uint64_t round) {
    bool again = false;
    {
        std::lock_guard<std::mutex> g(census_mutex_);
        if (round != census_round_) return;
        if (!census_waiting_.erase(pid)) return;
        if (!census_waiting_.empty()) return;
        census_finish(round);
        again = census_again_;
        census_again_ = false;
    }
    if (again) request_census();
}

void Scheduler::census_finish(uint64_t round) {
    // Called with `census_mutex_` held. What no collector of the round traced:
    // a message waiting in a mailbox, a kill's reason, and what a finished
    // process left for whoever joins it.
    for (auto& p : rt_.all_processes()) {
        p->mailbox.for_each([&](Value v) { shared_census_walk(v, round); });
        // Written once, under the process's lock, before `kill_requested` is
        // published -- so readable here without that lock, which a worker
        // reporting from its own slice may be holding.
        if (p->kill_requested.load(std::memory_order_acquire) && p->kill_reason) {
            shared_census_walk(p->kill_reason->value, round);
        }
        if (p->is_done()) shared_census_walk(p->exit_value, round);
    }
    rt_.shared().finish_round(round);
    census_round_ = 0;
}

// ---------------------------------------------------------------------------
// Timers
// ---------------------------------------------------------------------------

static int64_t monotonic_ns() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

uint64_t Scheduler::arm_timer(uint64_t pid, int64_t deadline_ns) {
    uint64_t seq;
    {
        std::lock_guard<std::mutex> g(timer_mutex_);
        seq = next_timer_++;
        auto it = timers_.emplace(deadline_ns, TimerEntry{seq, pid});
        timer_index_.emplace(seq, it);
        note_io_wait(true);
    }
    timer_cv_.notify_one();
    return seq;
}

void Scheduler::cancel_timer(uint64_t seq) {
    std::lock_guard<std::mutex> g(timer_mutex_);
    auto it = timer_index_.find(seq);
    if (it == timer_index_.end()) return;
    timers_.erase(it->second);
    timer_index_.erase(it);
    note_io_wait(false);
}

void Scheduler::timer_loop() {
    std::unique_lock<std::mutex> lk(timer_mutex_);
    while (timers_running_) {
        if (timers_.empty()) {
            timer_cv_.wait(lk);
            continue;
        }
        auto first = timers_.begin();
        int64_t now = monotonic_ns();
        if (first->first > now) {
            timer_cv_.wait_for(lk, std::chrono::nanoseconds(first->first - now));
            continue;
        }
        TimerEntry e = first->second;
        timer_index_.erase(e.seq);
        timers_.erase(first);
        lk.unlock();
        // Wake before releasing the count, for the reason the poller does:
        // the other order leaves an instant with nothing runnable and nothing
        // owed, which the deadlock check reads as a deadlock.
        wake(e.pid);
        note_io_wait(false);
        lk.lock();
    }
}

void Scheduler::notify_done() {
    // Through the mutex `wait_for_all` tests its condition under, and after the
    // state that condition reads has changed. Notifying without it lost the
    // wake-up whenever it landed between the waiter's test and its going to
    // sleep: `live_` reached zero, the notify found nobody waiting, and the
    // main thread slept for ever on a program that had finished -- every
    // worker idle, nothing left to say so. Rare enough to take a loaded
    // machine and a few hundred runs to see; `just test` is both.
    { std::lock_guard<std::mutex> g(idle_mutex_); }
    done_cv_.notify_all();
}

void Scheduler::wake(uint64_t pid) {
    auto proc = rt_.find_process(pid);
    if (!proc) return;
    std::lock_guard<std::mutex> g(proc->sched_mutex);
    if (proc->status.load(std::memory_order_relaxed) == ProcStatus::Waiting) {
        // It is parked and nobody is touching its machine state, so it is safe
        // to make it runnable. It resumes in Return mode on the NativeRetry
        // continuation the blocking builtin left behind, which calls that
        // builtin again -- this time it finds what it was waiting for.
        enqueue(proc);
        return;
    }
    // Still running, or between running and parking. Leave a note; the worker
    // checks for it before it commits to parking, so the wake cannot be lost.
    proc->wake_pending = true;
}

void Scheduler::dump(const char* why) const {
    std::string out = std::string("--- vm: ") + why + " ---\n";
    out += "scheduler: workers=" + std::to_string(workers_.size())
         + " idle=" + std::to_string(idle_workers_.load())
         + " live=" + std::to_string(live_.load())
         + " runnable=" + std::to_string(runnable_.load())
         + " io_waiters=" + std::to_string(io_waiters_.load())
         + " reductions=" + std::to_string(total_reductions_.load())
         + (deadlocked_.load() ? " DEADLOCKED" : "") + "\n";
    for (auto& proc : rt_.all_processes()) {
        ProcStatus st = proc->status.load(std::memory_order_relaxed);
        const char* name = st == ProcStatus::Runnable   ? "runnable"
                         : st == ProcStatus::Running    ? "running"
                         : st == ProcStatus::Waiting    ? "waiting"
                         : st == ProcStatus::Finished   ? "finished"
                                                        : "failed";
        out += "  process " + std::to_string(proc->id()) + ": " + name;
        if (st == ProcStatus::Waiting) {
            WaitReason r = proc->wait_reason.load(std::memory_order_relaxed);
            out += " on " + std::string(wait_reason_name(r));
            int fd = proc->wait_fd.load(std::memory_order_relaxed);
            if (r == WaitReason::Io && fd >= 0) out += " fd=" + std::to_string(fd);
        }
        // Mode and the top continuation say *where* in the machine a process
        // is, which is the difference between "looping in my program" and
        // "looping in the runtime".
        out += " mode=" + std::string(mode_name(proc->mode))
             + " conts=" + std::to_string(proc->conts.size())
             + (proc->conts.empty() ? std::string()
                                    : " top=" + std::string(cont_kind_name(proc->conts.back().kind)))
             + " reductions=" + std::to_string(proc->total_reductions)
             + " mailbox=" + std::to_string(proc->mailbox.size()) + "\n";
    }
    for (const IoHandleInfo& h : io_snapshot()) {
        const char* kind = h.kind == HandleKind::File     ? "file"
                         : h.kind == HandleKind::Listener ? "listener"
                                                          : "stream";
        out += "  handle " + std::to_string(h.handle) + ": fd=" + std::to_string(h.fd) + " "
             + kind + " busy=" + std::to_string(h.busy);
        if (h.waiting_pid) out += " waiting=" + std::to_string(h.waiting_pid);
        out += "\n";
    }
    std::fwrite(out.data(), 1, out.size(), stderr);
    std::fflush(stderr);
}

bool Scheduler::run(const std::shared_ptr<Process>& root) {
    root_id_.store(root->id());
    enqueue(root);
    start();
    bool clean = wait_for_all();
    stop();
    return clean;
}

bool Scheduler::wait_for_all() {
    // `DREAM_STUCK_SECONDS=n` reports what every process is doing if the whole
    // system goes that long without spending a single reduction. A program
    // parked on IO is not deadlocked -- the kernel may yet answer -- so the
    // runtime cannot simply declare it stuck; but a program that has stopped
    // for good is exactly the case where nothing is left that could ask.
    const char* env = std::getenv("DREAM_STUCK_SECONDS");
    double stuck_after = env ? std::atof(env) : 0.0;

    std::unique_lock<std::mutex> lk(idle_mutex_);
    if (stuck_after <= 0.0) {
        done_cv_.wait(lk, [this] {
            return live_.load() == 0 || deadlocked_.load() || root_failed_.load() || !running_.load();
        });
        return live_.load() == 0;
    }

    uint64_t last = total_reductions_.load();
    bool reported = false;
    for (;;) {
        auto period = std::chrono::duration<double>(stuck_after);
        bool done = done_cv_.wait_for(lk, std::chrono::duration_cast<std::chrono::milliseconds>(period), [this] {
            return live_.load() == 0 || deadlocked_.load() || root_failed_.load() || !running_.load();
        });
        if (done) break;
        uint64_t now = total_reductions_.load();
        if (now == last) {
            dump("no progress");
        }
        (void)reported;
        last = now;
    }
    return live_.load() == 0;
}

// ---------------------------------------------------------------------------
// Process-facing operations
// ---------------------------------------------------------------------------

uint64_t Scheduler::spawn_from(Process& parent, Value work) {
    if (!running_.load()) return 0;
    auto child = create_process();

    // The work crosses a heap boundary, so it is copied. A closure copies with
    // its captured values, which is exactly the isolation we want: the child
    // shares no mutable state with its parent.
    Value copied = Heap::copy_between(child->heap(), work);
    prime_apply(*child, copied, 0);
    enqueue(child);
    return child->id();
}

bool Scheduler::send(Process& sender, uint64_t target, Value message) {
    if (!rt_.find_process(target)) return false;
    auto msg = std::make_unique<Message>();
    msg->heap = std::make_unique<Heap>(1024);
    msg->value = Heap::copy_between(*msg->heap, message);
    msg->from_pid = sender.id();
    return deliver(target, std::move(msg));
}

bool Scheduler::receive(Process& p, Value* out) {
    std::lock_guard<std::mutex> g(p.sched_mutex);
    if (!p.mailbox.pop_into(p.heap(), out)) {
        p.park_requested = true;
        p.wait_reason.store(WaitReason::Message, std::memory_order_relaxed);
        return false;
    }
    return true;
}

Scheduler::JoinState Scheduler::join(Process& p, uint64_t target, Value* out) {
    auto proc = rt_.find_process(target);
    if (!proc) {
        // Unknown pid: either it never existed or it has been reaped.
        *out = UNIT;
        return JoinState::Ready;
    }
    if (proc.get() == &p) {
        *out = raise_error(p, well_known(rt_).error, "a process cannot join itself");
        return JoinState::Failed;
    }

    std::lock_guard<std::mutex> g(proc->waiters_mutex);
    if (proc->is_done()) {
        *out = Heap::copy_between(p.heap(), proc->exit_value);
        if (proc->failed) {
            // Delivered to a joiner, so it is handled; drop it from the list
            // the runtime reports at shutdown.
            std::lock_guard<std::mutex> gf(failures_mutex_);
            failures_.erase(target);
            return JoinState::Failed;
        }
        return JoinState::Ready;
    }
    // Register before parking, under the same lock the exiting process takes,
    // so an exit that happens right now cannot slip past us.
    proc->waiters.push_back(p.id());
    {
        std::lock_guard<std::mutex> g2(p.sched_mutex);
        p.park_requested = true;
        p.wait_reason.store(WaitReason::Join, std::memory_order_relaxed);
    }
    return JoinState::Blocked;
}

}  // namespace dream
