// Input and output. See io.hpp for why this is not a server process.

#include "io.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "builtins.hpp"
#include "interp.hpp"
#include "process.hpp"
#include "scheduler.hpp"
#include "digest.hpp"
#include "crypto.hpp"
#include "tls.hpp"

#if defined(__linux__) && !defined(DREAM_PORTABLE_POLLER)
#define DREAM_HAVE_EPOLL 1
#include <sys/epoll.h>
#include <sys/eventfd.h>
#else
#define DREAM_HAVE_EPOLL 0
#endif

#include "platform_io.hpp"
#include <chrono>
#include <condition_variable>

namespace dream {
namespace {

[[maybe_unused]] bool trace_on() { static bool on = ::getenv("DREAM_IO_TRACE") != nullptr; return on; }
#define IOTRACE(...) do { if (trace_on()) { std::fprintf(stderr, "[io] " __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)


// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

Value io_error(Process& p, const char* kind, const std::string& message) {
    return raise_error(p, p.runtime().intern_atom(kind), message);
}

NativeResult fail(Process& p, const char* kind, const std::string& message) {
    return NativeResult::raise(io_error(p, kind, message));
}

/// An `errno` turned into the kind a Dream program can match on. The message
/// keeps the system's own wording, which is more specific than anything this
/// file could invent.
NativeResult fail_errno(Process& p, const std::string& what, int err) {
    const char* kind = "io_error";
    switch (err) {
        case ENOENT: kind = "not_found"; break;
        case EACCES:
        case EPERM: kind = "permission_denied"; break;
        case EEXIST: kind = "already_exists"; break;
        case EISDIR:
        case ENOTDIR: kind = "wrong_kind"; break;
        case ECONNREFUSED: kind = "connection_refused"; break;
        case ECONNRESET:
        case EPIPE: kind = "connection_lost"; break;
        case EADDRINUSE: kind = "address_in_use"; break;
        case ETIMEDOUT: kind = "timed_out"; break;
        default: break;
    }
    return fail(p, kind, what + ": " + std::strerror(err));
}

std::string string_arg(Value v) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Str)) return std::string();
    auto* s = static_cast<StrObj*>(as_obj(v));
    return std::string(s->data(), s->len);
}

/// Deliberately false for a big string. Everything that asks goes on to call
/// `string_arg`, which materializes a `std::string` -- a path a payload must
/// never take. `write!` is the one place that wants the bytes rather than a
/// string, and it asks `string_bytes` instead.
bool is_string(Value v) { return is_obj(resolve(v), ObjType::Str); }

// ---------------------------------------------------------------------------
// Handles
//
// A handle is a small integer, not a pointer and not a raw descriptor. It
// carries a generation, so an id left over from a closed handle is rejected
// rather than silently addressing whatever descriptor the kernel has since
// handed out for the same number. That is the whole safety argument: no Dream
// program can reach a descriptor it was not given, or one it has closed.
// ---------------------------------------------------------------------------

void forget_waiter(int fd);

/// A connection that speaks TLS: the engine, and what io.cpp keeps on its
/// behalf between calls. See tls.hpp for why the engine never sees the socket.
///
/// It is held by a `shared_ptr` from the handle, so an operation that has
/// resolved a handle keeps its session alive even if another process closes
/// the handle meanwhile -- the same reason `busy` exists for the descriptor.
struct TlsSession {
    /// Held around every use of the engine, which is not thread-safe: two
    /// processes on two workers may hold one handle. Never held across a park.
    std::mutex mutex;
    std::unique_ptr<tls::Engine> engine;
    enum class State { Handshaking, Ready, Failed } state = State::Handshaking;
    /// Whether `connect!`/`accept!` has answered for a handshake that has
    /// finished. A handshake that finishes with records still to send parks
    /// on the socket, and the call is made again when it can be written: this
    /// is how that second call knows it is the same one, not a second
    /// handshake asked of a connection that already has one.
    bool announced = false;
    /// The network has ended; `feed_eof` has been said.
    bool eof = false;
    /// Ciphertext the engine has produced and the socket has not yet taken.
    std::string out;
    /// A `write!` whose records did not all fit in the socket is parked until
    /// they do, and its retry answers `accepted`: the plaintext it handed over
    /// was taken in full the first time and must not be encrypted twice.
    uint64_t writer = 0;
    size_t accepted = 0;
    tls::Failure failure;
};

struct Handle {
    int fd = -1;
    HandleKind kind = HandleKind::File;
    uint32_t generation = 0;
    bool open = false;
    /// Set between issuing a `connect` and learning whether it succeeded.
    bool connecting = false;
    /// Closing a descriptor the VM does not own would be rude, and closing
    /// stdin twice would be worse.
    bool owned = true;
    /// A socket the VM opened or accepted, written with `sys::send` so that a
    /// peer that has gone is an EPIPE and not a SIGPIPE. Pipes and terminals
    /// are `Stream`s too, and `send` would refuse them.
    bool socket = false;
    /// How many operations are in flight on this handle right now.
    ///
    /// Operations release the table lock before their syscall -- holding it
    /// would serialize every process's IO behind whichever one is slowest --
    /// so `close!` cannot simply close the descriptor: another process may be
    /// inside `read(2)` on it, and the number would be handed straight back
    /// out by the next `open`. Instead `close!` marks the handle closed and
    /// the last operation to finish does the closing.
    int busy = 0;
    /// Set by `tls.connect!` or `tls.accept!`, after which reads and writes
    /// go through it.
    std::shared_ptr<TlsSession> tls;
};

constexpr uint64_t HANDLE_INDEX_BITS = 32;
constexpr uint64_t HANDLE_INDEX_MASK = (uint64_t(1) << HANDLE_INDEX_BITS) - 1;

class HandleTable {
public:
    static HandleTable& get() {
        static HandleTable t;
        return t;
    }

    /// Take ownership of `fd` and return the id that addresses it.
    int64_t add(int fd, HandleKind kind, bool owned = true, bool socket = false) {
        std::lock_guard<std::mutex> g(mutex_);
        size_t i = 0;
        for (; i < slots_.size(); ++i) {
            if (!slots_[i].open && slots_[i].busy == 0) break;
        }
        if (i == slots_.size()) slots_.push_back(Handle{});
        Handle& h = slots_[i];
        h.fd = fd;
        h.kind = kind;
        h.open = true;
        h.connecting = false;
        h.owned = owned;
        h.socket = socket;
        h.tls.reset();
        ++h.generation;
        return int64_t((uint64_t(h.generation) << HANDLE_INDEX_BITS) | uint64_t(i));
    }

    /// The handle behind `id`, or null when the id is stale, closed or was
    /// never issued.
    Handle* find(int64_t id) {
        if (id < 0) return nullptr;
        uint64_t raw = uint64_t(id);
        size_t index = size_t(raw & HANDLE_INDEX_MASK);
        uint32_t generation = uint32_t(raw >> HANDLE_INDEX_BITS);
        if (index >= slots_.size()) return nullptr;
        Handle& h = slots_[index];
        if (!h.open || h.generation != generation) return nullptr;
        return &h;
    }

    std::mutex& mutex() { return mutex_; }

    /// Resolve `id` and mark an operation in flight. Returns false when the id
    /// is stale, closed, or was never issued.
    bool acquire(int64_t id, int* fd, HandleKind* kind, bool* socket, std::shared_ptr<TlsSession>* tls) {
        std::lock_guard<std::mutex> g(mutex_);
        Handle* h = find(id);
        if (!h) return false;
        ++h->busy;
        *fd = h->fd;
        *kind = h->kind;
        *socket = h->socket;
        *tls = h->tls;
        return true;
    }

    /// Make the handle speak TLS from now on. False when it has closed.
    bool attach_tls(int64_t id, std::shared_ptr<TlsSession> session) {
        std::lock_guard<std::mutex> g(mutex_);
        Handle* h = find(id);
        if (!h) return false;
        h->tls = std::move(session);
        return true;
    }

    /// Finish an operation, closing the descriptor if `close!` was waiting on
    /// this being the last one.
    void release(int64_t id) {
        int to_close = -1;
        {
            std::lock_guard<std::mutex> g(mutex_);
            size_t index = size_t(uint64_t(id) & HANDLE_INDEX_MASK);
            if (index >= slots_.size()) return;
            Handle& h = slots_[index];
            if (--h.busy == 0 && !h.open && h.owned && h.fd >= 0) {
                to_close = h.fd;
                h.fd = -1;
            }
        }
        if (to_close >= 0) {
            // The close `close!` put off until this operation finished. That
            // operation may have gone on to park its process on the
            // descriptor -- `accept!` holds the handle until it has armed the
            // poller -- and `close!`'s own `forget` ran before there was a
            // waiter to release. Closing silently here left it parked for
            // ever: a closed descriptor leaves epoll without an event. A test
            // that stopped a listener while its accept loop was between the
            // two hung at exit about one run in ten.
            forget_waiter(to_close);
            sys::close(to_close);
        }
    }

    /// Every open handle, for `std.vm`.
    std::vector<IoHandleInfo> snapshot() {
        std::vector<IoHandleInfo> out;
        std::lock_guard<std::mutex> g(mutex_);
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Handle& h = slots_[i];
            if (!h.open) continue;
            IoHandleInfo info;
            info.handle = int64_t((uint64_t(h.generation) << HANDLE_INDEX_BITS) | uint64_t(i));
            info.fd = h.fd;
            info.kind = h.kind;
            info.busy = h.busy;
            out.push_back(info);
        }
        return out;
    }

    /// Close every handle still open, for shutdown.
    void close_all() {
        std::lock_guard<std::mutex> g(mutex_);
        for (Handle& h : slots_) {
            if (h.open && h.owned && h.fd >= 0) sys::close(h.fd);
            h.open = false;
            h.tls.reset();
        }
    }

private:
    std::mutex mutex_;
    /// A deque, not a vector: `add` may grow the table while a caller holds a
    /// `Handle*` from `find`, and a deque does not move what is already there.
    std::deque<Handle> slots_;
};

/// Holds an operation open on a handle for as long as it is in scope.
class Held {
public:
    Held() = default;
    ~Held() { if (ok_) HandleTable::get().release(id_); }
    Held(const Held&) = delete;
    Held& operator=(const Held&) = delete;

    bool open(Value v) {
        Value h = resolve(v);
        if (!is_fixnum(h)) return false;
        id_ = fixnum_value(h);
        ok_ = HandleTable::get().acquire(id_, &fd_, &kind_, &socket_, &tls_);
        return ok_;
    }

    /// The handle's TLS session, or null for one that speaks plain bytes.
    const std::shared_ptr<TlsSession>& tls() const { return tls_; }

    int fd() const { return fd_; }
    HandleKind kind() const { return kind_; }
    bool socket() const { return socket_; }
    int64_t id() const { return id_; }
    /// A regular file is always "ready" to `epoll`, so waiting on one is
    /// meaningless and its calls are issued directly.
    bool pollable() const { return kind_ != HandleKind::File; }

private:
    int64_t id_ = -1;
    int fd_ = -1;
    HandleKind kind_ = HandleKind::File;
    bool socket_ = false;
    bool ok_ = false;
    std::shared_ptr<TlsSession> tls_;
};


// ---------------------------------------------------------------------------
// The poller
//
// One thread, one `epoll` set, and a table of which process is waiting on
// which descriptor. A native that cannot complete registers interest and parks
// its process; this thread wakes the process when the kernel says the
// descriptor is ready.
//
// Interest is registered EPOLLONESHOT: it fires once and is disarmed, so a
// process that is woken but then loses a race for the data simply arms again.
// Level-triggered would spin, and edge-triggered without oneshot would wake
// every waiter on a shared descriptor.
// ---------------------------------------------------------------------------

class Poller {
public:
    static Poller& get() {
        static Poller p;
        return p;
    }

    bool available() const { return running_.load(); }

    void start() {
#if DREAM_HAVE_EPOLL
        if (running_.exchange(true)) return;
        epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) {
            running_ = false;
            return;
        }
        // A descriptor whose only job is to make `epoll_wait` return, so the
        // thread can notice it has been asked to stop.
        wake_fd_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (wake_fd_ >= 0) {
            epoll_event ev{};
            ev.events = EPOLLIN;
            ev.data.fd = wake_fd_;
            ::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wake_fd_, &ev);
        }
        thread_ = std::thread([this] { loop(); });
#else
        if (running_.exchange(true)) return;
        thread_ = std::thread([this] { loop_portable(); });
#endif
    }

    void stop() {
#if DREAM_HAVE_EPOLL
        if (!running_.exchange(false)) return;
        if (wake_fd_ >= 0) {
            uint64_t one = 1;
            [[maybe_unused]] sys::Count n = sys::write(wake_fd_, &one, sizeof one);
        }
        if (thread_.joinable()) thread_.join();
        if (wake_fd_ >= 0) sys::close(wake_fd_);
        if (epoll_fd_ >= 0) sys::close(epoll_fd_);
        wake_fd_ = -1;
        epoll_fd_ = -1;
#else
        if (!running_.exchange(false)) return;
        changed_.notify_all();
        if (thread_.joinable()) thread_.join();
#endif
    }

    /// Wait for `fd` to become readable (or writable) on behalf of `pid`.
    ///
    /// A descriptor has waiters in each direction, because a socket is
    /// full-duplex -- one process may be parked reading it while another is
    /// parked writing to it, and each must be woken by its own readiness --
    /// and several in one direction, because several processes may accept on
    /// one listener. Readiness wakes everyone waiting in that direction; the
    /// ones that lose the race for it find the operation would block again
    /// and arm again. It used to be one waiter per descriptor, and a second
    /// replaced the first, which was then never woken: a hang that needed two
    /// processes parked at once, so a large loopback buffer on Linux hid it
    /// and a small one on macOS did not.
    ///
    /// Returns false when this build cannot wait, so the caller can fall back
    /// to a blocking call rather than parking a process nothing will wake.
    bool arm(int fd, bool writable, uint64_t pid, Scheduler* sched) {
#if DREAM_HAVE_EPOLL
        if (epoll_fd_ < 0) return false;
        uint32_t events;
        bool added;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto [it, fresh] = waiters_.try_emplace(fd);
            added = fresh;
            std::vector<Waiter>& side = writable ? it->second.writers : it->second.readers;
            side.push_back(Waiter{pid, sched->serial()});
            events = interest(it->second);
            // Made under the lock, so that two processes arming one descriptor
            // in opposite directions cannot leave it registered for only one.
            epoll_event ev{};
            ev.events = events;
            ev.data.fd = fd;
            // MOD first: the descriptor is usually already in the set from an
            // earlier wait, and ADD would fail with EEXIST.
            if (::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) < 0) {
                if (errno != ENOENT || ::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
                    side.pop_back();
                    if (added) waiters_.erase(it);
                    return false;
                }
            }
        }
        sched->note_io_wait(true);
        IOTRACE("arm fd=%d %s pid=%llu", fd, writable ? "w" : "r", (unsigned long long)pid);
        return true;
#else
        std::lock_guard<std::mutex> g(mutex_);
        if (!running_) return false;
        sched->note_io_wait(true);
        (writable ? waiters_[fd].writers : waiters_[fd].readers).push_back(Waiter{pid, sched->serial()});
        changed_.notify_all();
        return true;
#endif
    }

    /// A process parked on `fd`, or 0 if none is.
    uint64_t waiter_of(int fd) {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = waiters_.find(fd);
        if (it == waiters_.end()) return 0;
        if (!it->second.readers.empty()) return it->second.readers.front().pid;
        return it->second.writers.empty() ? 0 : it->second.writers.front().pid;
    }

    /// Stop watching `fd`, because it is being closed.
    void forget(int fd) {
        Waiters w{};
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = waiters_.find(fd);
            if (it != waiters_.end()) {
                w = it->second;
                waiters_.erase(it);
            }
#if DREAM_HAVE_EPOLL
            if (epoll_fd_ >= 0) ::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
#endif
        }
        // Whoever was waiting must be released, or closing a socket out from
        // under a reader would park that reader for ever. It wakes, retries,
        // and gets the "closed" error the retry produces.
        release(w.readers);
        release(w.writers);
    }

    /// Stop waiting on `fd` for one process, which is being killed. Its count
    /// as an IO waiter goes with it; the descriptor stays watched for anyone
    /// else waiting on it.
    void cancel(int fd, uint64_t pid) {
        size_t dropped = 0;
        uint64_t sched = 0;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = waiters_.find(fd);
            if (it == waiters_.end()) return;
            for (std::vector<Waiter>* side : {&it->second.readers, &it->second.writers}) {
                for (size_t i = 0; i < side->size();) {
                    if ((*side)[i].pid == pid) {
                        sched = (*side)[i].sched;
                        side->erase(side->begin() + std::ptrdiff_t(i));
                        ++dropped;
                    } else {
                        ++i;
                    }
                }
            }
            if (!dropped) return;
#if DREAM_HAVE_EPOLL
            if (it->second.readers.empty() && it->second.writers.empty()) {
                waiters_.erase(it);
                if (epoll_fd_ >= 0) ::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
            } else if (epoll_fd_ >= 0) {
                epoll_event ev{};
                ev.events = interest(it->second);
                ev.data.fd = fd;
                ::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev);
            }
#else
            if (it->second.readers.empty() && it->second.writers.empty()) waiters_.erase(it);
#endif
        }
        Scheduler::release_io_waits(sched, dropped);
    }

private:
    /// The scheduler is named by its serial, not held: this thread outlives
    /// it. See `Scheduler::serial`.
    struct Waiter {
        uint64_t pid = 0;
        uint64_t sched = 0;
    };
    /// Who is parked on one descriptor, in each direction.
    struct Waiters {
        std::vector<Waiter> readers;
        std::vector<Waiter> writers;
    };

    /// Wake waiters and stop counting them. Wake before releasing the
    /// IO-waiter count; see `run_child` in os.cpp for why the other order
    /// invents a deadlock.
    static void release(const std::vector<Waiter>& ws) {
        for (const Waiter& w : ws) Scheduler::wake_external(w.sched, w.pid);
    }

#if DREAM_HAVE_EPOLL
    /// The events to ask for: one-shot, in the directions someone waits in.
    static uint32_t interest(const Waiters& w) {
        uint32_t e = EPOLLONESHOT | EPOLLERR | EPOLLHUP;
        if (!w.readers.empty()) e |= EPOLLIN;
        if (!w.writers.empty()) e |= EPOLLOUT;
        return e;
    }

    void loop() {
        std::vector<epoll_event> events(64);
        while (running_.load(std::memory_order_relaxed)) {
            int n = ::epoll_wait(epoll_fd_, events.data(), int(events.size()), 100);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            for (int i = 0; i < n; ++i) {
                int fd = events[i].data.fd;
                if (fd == wake_fd_) {
                    uint64_t drain = 0;
                    [[maybe_unused]] sys::Count r = sys::read(wake_fd_, &drain, sizeof drain);
                    continue;
                }
                uint32_t got = events[i].events;
                bool failed = (got & (EPOLLERR | EPOLLHUP)) != 0;
                std::vector<Waiter> readers, writers;
                {
                    std::lock_guard<std::mutex> g(mutex_);
                    auto it = waiters_.find(fd);
                    if (it == waiters_.end()) continue;
                    if (failed || (got & EPOLLIN)) readers.swap(it->second.readers);
                    if (failed || (got & EPOLLOUT)) writers.swap(it->second.writers);
                    if (it->second.readers.empty() && it->second.writers.empty()) {
                        waiters_.erase(it);
                    } else {
                        // One-shot disarmed the descriptor; whoever is still
                        // waiting in the other direction needs it armed again.
                        epoll_event ev{};
                        ev.events = interest(it->second);
                        ev.data.fd = fd;
                        ::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev);
                    }
                }
                IOTRACE("ready fd=%d -> wake %zu readers, %zu writers", fd, readers.size(), writers.size());
                release(readers);
                release(writers);
            }
        }
    }
#endif

#if !DREAM_HAVE_EPOLL
    void loop_portable() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (running_) {
            for (auto it = waiters_.begin(); it != waiters_.end();) {
                Waiters& w = it->second;
                if (!w.readers.empty() && sys::ready(it->first, false)) {
                    release(w.readers);
                    w.readers.clear();
                }
                if (!w.writers.empty() && sys::ready(it->first, true)) {
                    release(w.writers);
                    w.writers.clear();
                }
                if (w.readers.empty() && w.writers.empty()) it = waiters_.erase(it);
                else ++it;
            }
            changed_.wait_for(lock, std::chrono::milliseconds(5));
        }
    }
    std::condition_variable changed_;
#endif
    std::atomic<bool> running_{false};
    int epoll_fd_ = -1;
    int wake_fd_ = -1;
    std::thread thread_;
    std::mutex mutex_;
    std::unordered_map<int, Waiters> waiters_;
};

}  // namespace

void io_cancel_wait(int fd, uint64_t pid) { Poller::get().cancel(fd, pid); }

namespace {

/// Park `p` until `fd` is ready. The caller returns the result unchanged.
///
/// Returning `block()` on its own only says "my result is not ready"; asking to
/// be parked is what stops the worker from running this process again straight
/// away. The request is made before the descriptor is armed, so a readiness
/// notification that arrives in between is not lost: `wake` finds the process
/// still running and leaves a note the worker checks before it commits.
NativeResult wait_for(Process& p, int fd, bool writable) {
    Scheduler* sched = p.runtime().scheduler();
    if (!sched) {
        return fail(p, "io_error", "no scheduler is running, so nothing could wake this process");
    }
    {
        std::lock_guard<std::mutex> g(p.sched_mutex);
        p.park_requested = true;
        p.wait_reason.store(WaitReason::Io, std::memory_order_relaxed);
        p.wait_fd.store(fd, std::memory_order_relaxed);
    }
    if (!Poller::get().arm(fd, writable, p.id(), sched)) {
        std::lock_guard<std::mutex> g(p.sched_mutex);
        p.park_requested = false;
        return fail(p, "io_error",
                    "this build cannot wait on a descriptor without blocking a worker");
    }
    return NativeResult::block();
}

bool set_nonblocking(int fd) {
    return sys::nonblocking(fd);
}

// ---------------------------------------------------------------------------
// TLS on a handle
//
// The loop every TLS operation runs: write out whatever records the engine has
// made, ask the engine for what the caller wants, and when it needs more of the
// network than it has, read some and feed it. Each step that would block parks
// the process on the socket instead (`wait_for`), and the native is entered
// again from the top when it can go on -- which is why everything a call has
// done so far lives in the session and not on the C++ stack. The session's
// mutex is held for the steps and released by returning, never across a park.
// ---------------------------------------------------------------------------

enum class Pump { Done, Again, Eof, Failed };

/// Write the session's pending records. `Again` when the socket is full.
Pump flush_tls(TlsSession& s, int fd, int* err) {
    while (!s.out.empty()) {
        sys::Count n = sys::send(fd, s.out.data(), s.out.size());
        if (n > 0) {
            s.out.erase(0, size_t(n));
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return Pump::Again;
        *err = n < 0 ? errno : EPIPE;
        return Pump::Failed;
    }
    return Pump::Done;
}

/// Read what ciphertext has arrived and feed it to the engine.
Pump pull_tls(TlsSession& s, int fd, int* err) {
    char buf[16384];
    for (;;) {
        sys::Count n = sys::read(fd, buf, sizeof buf);
        if (n > 0) {
            s.engine->feed(buf, size_t(n));
            return Pump::Done;
        }
        if (n == 0) {
            s.eof = true;
            s.engine->feed_eof();
            return Pump::Eof;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return Pump::Again;
        *err = errno;
        return Pump::Failed;
    }
}

/// Raise what the engine said went wrong, and remember it: a session that has
/// failed refuses everything after, rather than handing on whatever state the
/// library was left in.
///
/// The peer is told twice over: the alert the engine made, if it made one, and
/// then the end of this side's stream. A TLS session that has failed sends
/// nothing more, and a peer that is not told so -- a server whose client
/// refused its certificate after a handshake the server saw succeed -- waits
/// for ever on a read. Only the sending half is shut, so whatever the peer
/// says back can still be read, and the handle stays the program's to close.
NativeResult tls_failed(Process& p, TlsSession& s, int fd) {
    s.state = TlsSession::State::Failed;
    s.failure = s.engine->failure();
    int ignored = 0;
    s.out += s.engine->take_output();  // the alert, if there is one
    flush_tls(s, fd, &ignored);
    sys::shutdown(fd, SHUT_WR);
    return fail(p, s.failure.kind.c_str(), s.failure.message);
}

NativeResult tls_refused(Process& p, TlsSession& s) {
    if (s.state == TlsSession::State::Handshaking) {
        return fail(p, "tls_error", "this connection's TLS handshake has not finished");
    }
    return fail(p, s.failure.kind.empty() ? "tls_error" : s.failure.kind.c_str(),
                "TLS on this connection already failed: " + s.failure.message);
}

/// `read!` on a TLS handle: plaintext, "" at the end of the stream.
NativeResult tls_read(Process& p, Held& h, size_t want) {
    TlsSession& s = *h.tls();
    std::lock_guard<std::mutex> g(s.mutex);
    if (s.state != TlsSession::State::Ready) return tls_refused(p, s);
    std::vector<char> buf(want);
    for (;;) {
        int err = 0;
        switch (flush_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), true);
            case Pump::Failed: return fail_errno(p, "write", err);
            default: break;
        }
        size_t got = 0;
        tls::Status st = s.engine->read(buf.data(), want, &got);
        s.out += s.engine->take_output();
        switch (st) {
            case tls::Status::Ok:
                return NativeResult::ok(p.heap().make_string(buf.data(), uint32_t(got)));
            case tls::Status::Closed:
                return NativeResult::ok(p.heap().make_string("", 0));
            case tls::Status::Error:
                return tls_failed(p, s, h.fd());
            case tls::Status::WantRead:
                break;
        }
        // An engine that wants more after it was told the network ended has
        // nothing more coming, and a stream that ended is "".
        if (s.eof) return NativeResult::ok(p.heap().make_string("", 0));
        switch (flush_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), true);
            case Pump::Failed: return fail_errno(p, "write", err);
            default: break;
        }
        switch (pull_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), false);
            case Pump::Failed: return fail_errno(p, "read", err);
            default: break;
        }
    }
}

/// `write!` on a TLS handle. Encrypts up to 64 KiB of `data` and answers how
/// much, once the records it made are all on the wire.
NativeResult tls_write(Process& p, Held& h, const Bytes& data) {
    TlsSession& s = *h.tls();
    std::lock_guard<std::mutex> g(s.mutex);
    if (s.state != TlsSession::State::Ready) return tls_refused(p, s);
    int err = 0;
    if (s.writer == p.id() && s.accepted > 0) {
        // The retry of a write that parked: its plaintext is already records.
        switch (flush_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), true);
            case Pump::Failed: s.writer = 0; s.accepted = 0; return fail_errno(p, "write", err);
            default: break;
        }
        size_t n = s.accepted;
        s.writer = 0;
        s.accepted = 0;
        return NativeResult::ok(make_fixnum(int64_t(n)));
    }
    // Records someone else left behind go first, so the stream stays in order.
    switch (flush_tls(s, h.fd(), &err)) {
        case Pump::Again: return wait_for(p, h.fd(), true);
        case Pump::Failed: return fail_errno(p, "write", err);
        default: break;
    }
    size_t take = size_t(std::min<uint64_t>(data.len, 65536));
    if (s.engine->write(data.data, take) != tls::Status::Ok) return tls_failed(p, s, h.fd());
    s.out += s.engine->take_output();
    switch (flush_tls(s, h.fd(), &err)) {
        case Pump::Again:
            s.writer = p.id();
            s.accepted = take;
            return wait_for(p, h.fd(), true);
        case Pump::Failed: return fail_errno(p, "write", err);
        default: break;
    }
    return NativeResult::ok(make_fixnum(int64_t(take)));
}

/// Say goodbye on a TLS handle, as far as the socket will take it without
/// waiting: a close_notify is a courtesy, and a close that parked for one
/// would make `close!` something that can hang.
void tls_goodbye(TlsSession& s, int fd) {
    std::lock_guard<std::mutex> g(s.mutex);
    if (s.state != TlsSession::State::Ready) return;
    s.engine->close();
    s.out += s.engine->take_output();
    int ignored = 0;
    flush_tls(s, fd, &ignored);
    s.state = TlsSession::State::Failed;
    s.failure = {"tls_error", "the connection was closed"};
}

// ---------------------------------------------------------------------------
// std.io
// ---------------------------------------------------------------------------

/// Read at most `want` bytes. Returns "" at end of file, which is what makes
/// a read loop terminate without a separate `eof` call racing the read.
NativeResult io_read(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) {
        return fail(p, "io_closed", "this handle is closed, or was never opened");
    }
    Value n = resolve(args[1]);
    if (!is_fixnum(n) || fixnum_value(n) < 0) {
        return fail(p, "type_error", "read! needs a byte count");
    }
    size_t want = size_t(fixnum_value(n));
    if (want == 0) return NativeResult::ok(p.heap().make_string("", 0));
    // One call cannot be asked to fill the heap; a reader loops anyway.
    if (want > (1u << 24)) want = 1u << 24;
    if (h.tls()) return tls_read(p, h, want);

    std::vector<char> buf(want);
    for (;;) {
        sys::Count got = sys::read(h.fd(), buf.data(), want);
        if (got >= 0) return NativeResult::ok(p.heap().make_string(buf.data(), uint32_t(got)));
        if (errno == EINTR) continue;
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && h.pollable()) {
            return wait_for(p, h.fd(), false);
        }
        return fail_errno(p, "read", errno);
    }
}

/// Write as much of `data` as the descriptor will take, and answer how much
/// that was. A short write is not an error -- `write_all!` is the loop.
NativeResult io_write(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) {
        return fail(p, "io_closed", "this handle is closed, or was never opened");
    }
    Bytes str;
    if (!string_bytes(args[1], &str)) return fail(p, "type_error", "write! needs a string");
    if (str.len == 0) return NativeResult::ok(make_fixnum(0));
    if (h.tls()) return tls_write(p, h, str);

    for (;;) {
        // A big string is written from where it lies in the mapped image, with
        // no intermediate buffer -- the kernel reads the pages, so a payload
        // larger than memory costs no memory to write. `write` takes a size_t
        // and answers how much it took, and a short write is the caller's loop
        // either way, so nothing here has to know which kind it was handed.
        sys::Count put = h.socket() ? sys::send(h.fd(), str.data, size_t(str.len))
                                    : sys::write(h.fd(), str.data, size_t(str.len));
        if (put >= 0) return NativeResult::ok(make_fixnum(int64_t(put)));
        if (errno == EINTR) continue;
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && h.pollable()) {
            return wait_for(p, h.fd(), true);
        }
        return fail_errno(p, "write", errno);
    }
}

NativeResult io_open(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "open! needs a path");
    std::string path = string_arg(args[0]);

    Value m = resolve(args[1]);
    if (!is_atom(m)) {
        return fail(p, "type_error", "open! needs a mode: :read, :write, :append or :update");
    }
    std::string mode = p.runtime().atom_name(uint32_t(imm_payload(m)));
    int flags;
    if (mode == "read") flags = O_RDONLY;
    else if (mode == "write") flags = O_WRONLY | O_CREAT | O_TRUNC;
    else if (mode == "append") flags = O_WRONLY | O_CREAT | O_APPEND;
    else if (mode == "update") flags = O_RDWR | O_CREAT;
    else return fail(p, "bad_argument", "`:" + mode + "` is not a mode; use :read, :write, :append or :update");

    int fd = sys::open(path.c_str(), flags | O_CLOEXEC, 0644);
    if (fd < 0) return fail_errno(p, "open " + path, errno);

    // What it actually is decides how it can be waited on: opening a fifo by
    // name gives a descriptor that must be polled, not read straight through.
    sys::FileStat st{};
    HandleKind kind = HandleKind::File;
    if (sys::fstat(fd, &st) == 0 && !S_ISREG(st.st_mode)) {
        kind = HandleKind::Stream;
        set_nonblocking(fd);
    }
    return NativeResult::ok(make_fixnum(HandleTable::get().add(fd, kind)));
}

std::filesystem::path fs_path(const std::string& utf8);
std::string fs_text(const std::filesystem::path& path);

/// Atomically reserve a fresh name: a read-write file, or a private directory.
/// Randomness makes collisions unlikely; O_EXCL/mkdir make them harmless.
NativeResult io_temp(Process& p, Value callee, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) {
        return fail(p, "type_error", "temp_file!/temp_dir! need a directory and a name prefix");
    }
    std::string dir = string_arg(args[0]);
    std::string prefix = string_arg(args[1]);
    if (dir.empty() || dir.find('\0') != std::string::npos ||
        prefix.find_first_of("/\\:") != std::string::npos || prefix.find('\0') != std::string::npos) {
        return fail(p, "bad_argument", "temporary entries need a directory and a prefix without separators or a drive");
    }
    const bool directory = static_cast<NativeObj*>(as_obj(callee))->user != 0;
    const char* hex = "0123456789abcdef";
    for (unsigned attempt = 0; attempt < 128; ++attempt) {
        unsigned char random[16];
        std::string error;
        if (!secure_random_bytes(random, sizeof random, &error)) return fail(p, "io_error", error);
        std::string name = prefix;
        for (unsigned char byte : random) {
            name += hex[byte >> 4];
            name += hex[byte & 15];
        }
        std::string path = fs_text(fs_path(dir) / fs_path(name));
        if (directory) {
            if (sys::mkdir(path.c_str(), 0700) == 0) {
                return NativeResult::ok(p.heap().make_string(path.data(), uint32_t(path.size())));
            }
        } else {
            int fd = sys::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd >= 0) {
                Value handle = make_fixnum(HandleTable::get().add(fd, HandleKind::File));
                Value text = p.heap().make_string(path.data(), uint32_t(path.size()));
                return NativeResult::ok(p.heap().make_cons(text, p.heap().make_cons(handle, NIL)));
            }
        }
        if (errno == EINTR) continue;
        if (errno != EEXIST) return fail_errno(p, "create temporary entry in " + dir, errno);
    }
    return fail(p, "already_exists", "could not reserve a temporary name in " + dir);
}

NativeResult io_close(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return fail(p, "type_error", "close! needs an io handle");
    {
        Held held;
        if (held.open(v) && held.tls()) tls_goodbye(*held.tls(), held.fd());
    }
    int fd = -1;
    bool close_now = false;
    {
        std::lock_guard<std::mutex> g(HandleTable::get().mutex());
        Handle* h = HandleTable::get().find(fixnum_value(v));
        if (!h) return fail(p, "io_closed", "this handle is already closed");
        fd = h->fd;
        h->open = false;
        h->tls.reset();
        // Nobody is mid-syscall on it, so it can go now. Otherwise the last
        // operation to finish closes it -- see `HandleTable::release`.
        if (h->busy == 0 && h->owned && fd >= 0) {
            h->fd = -1;
            close_now = true;
        }
    }
    // Release any waiter first: closing a socket out from under a reader would
    // otherwise park that reader for ever. It wakes, retries, and gets the
    // "closed" error.
    Poller::get().forget(fd);
    if (close_now) sys::close(fd);
    return NativeResult::ok(UNIT);
}

NativeResult io_flush(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    // Only a regular file has anything buffered in the kernel worth forcing
    // out; asking a socket to fsync is an error, not a no-op.
    if (h.kind() == HandleKind::File && sys::fsync(h.fd()) < 0) {
        return fail_errno(p, "flush", errno);
    }
    return NativeResult::ok(UNIT);
}

NativeResult io_seek(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    Value off = resolve(args[1]);
    if (!is_fixnum(off)) return fail(p, "type_error", "seek! needs a byte offset");
    sys::Offset where = sys::lseek(h.fd(), sys::Offset(fixnum_value(off)), SEEK_SET);
    if (where < 0) return fail_errno(p, "seek", errno);
    return NativeResult::ok(make_fixnum(int64_t(where)));
}

NativeResult io_size(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    sys::FileStat st{};
    if (sys::fstat(h.fd(), &st) < 0) return fail_errno(p, "size", errno);
    return NativeResult::ok(make_fixnum(int64_t(st.st_size)));
}

NativeResult io_kind(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    const char* name = h.kind() == HandleKind::File     ? "file"
                     : h.kind() == HandleKind::Listener ? "listener"
                                                        : "stream";
    return NativeResult::ok(make_atom(p.runtime().intern_atom(name)));
}

NativeResult io_is_terminal(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    return NativeResult::ok(make_bool(sys::is_terminal(h.fd())));
}

NativeResult io_enable_ansi(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    return NativeResult::ok(make_bool(sys::enable_ansi(h.fd())));
}

NativeResult io_is_open(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return NativeResult::ok(make_bool(false));
    std::lock_guard<std::mutex> g(HandleTable::get().mutex());
    return NativeResult::ok(make_bool(HandleTable::get().find(fixnum_value(v)) != nullptr));
}

/// Whether waiting on a descriptor parks the process rather than the worker.
NativeResult io_async(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_bool(io_async_available()));
}

// --- the standard streams ---------------------------------------------------
//
// Handed out as handles the program does not own: closing one would close the
// process's real stdin, which is not this VM's to do.

int64_t std_handle(int fd) {
    static std::mutex mutex;
    static int64_t ids[3] = {-1, -1, -1};
    std::lock_guard<std::mutex> g(mutex);
    if (ids[fd] < 0) {
        sys::FileStat st{};
        bool regular = sys::fstat(fd, &st) == 0 && S_ISREG(st.st_mode);
        ids[fd] = HandleTable::get().add(fd, regular ? HandleKind::File : HandleKind::Stream,
                                         /*owned=*/false);
    }
    return ids[fd];
}

NativeResult io_stdin(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(std_handle(0)));
}
NativeResult io_stdout(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(std_handle(1)));
}
NativeResult io_stderr(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(std_handle(2)));
}

// --- paths ------------------------------------------------------------------

NativeResult io_exists(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "exists! needs a path");
    sys::FileStat st{};
    return NativeResult::ok(make_bool(sys::stat(string_arg(args[0]).c_str(), &st) == 0));
}

NativeResult io_is_dir(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "is_dir! needs a path");
    sys::FileStat st{};
    if (sys::stat(string_arg(args[0]).c_str(), &st) != 0) return NativeResult::ok(make_bool(false));
    return NativeResult::ok(make_bool(S_ISDIR(st.st_mode)));
}

// --- stamps and digests -------------------------------------------------------
//
// What a build cache is made of. A file's stamp -- its size and when it was
// last written -- is kept beside the digest of its contents, and the contents
// are read again only when the stamp has moved: git's index trick, and what
// lets a build where nothing changed cost a `stat` per input. Digesting is
// done here rather than in Dream because it is the one part of a no-op build
// that would otherwise read every byte of every input.

/// When a file was last written, in nanoseconds since the epoch. Windows keeps
/// seconds in `_stat64`, which is coarser but still a stamp: a file rewritten
/// within the second at the same size is read again only if something else
/// moved, the same risk git's index accepts on filesystems with coarse times.
int64_t modified_ns(const sys::FileStat& st) {
#if defined(_WIN32)
    return int64_t(st.st_mtime) * 1000000000;
#elif defined(__APPLE__)
    return int64_t(st.st_mtimespec.tv_sec) * 1000000000 + st.st_mtimespec.tv_nsec;
#else
    return int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
#endif
}

/// `[size, modified, kind]`, or `()` when nothing is there.
///
/// Nothing being there is an answer rather than a raise, because it is the
/// question a stamp check most often asks: an input that has been deleted has
/// moved as surely as one that has been rewritten. A list rather than a map
/// because it is a stamp, compared whole far more often than it is read.
NativeResult io_stat(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "stat! needs a path");
    std::string path = string_arg(args[0]);
    sys::FileStat st{};
    if (sys::stat(path.c_str(), &st) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) return NativeResult::ok(UNIT);
        return fail_errno(p, "stat " + path, errno);
    }
    const char* kind = S_ISDIR(st.st_mode) ? "dir" : S_ISREG(st.st_mode) ? "file" : "other";
    Value k = make_atom(p.runtime().intern_atom(kind));
    Value list = p.heap().make_cons(k, NIL);
    list = p.heap().make_cons(make_integer(p, modified_ns(st)), list);
    list = p.heap().make_cons(make_integer(p, int64_t(st.st_size)), list);
    return NativeResult::ok(list);
}

Value hex_string(Process& p, const std::string& hex) {
    return p.heap().make_string(hex.data(), uint32_t(hex.size()));
}

/// The SHA-256 of a file's contents, as 64 hex digits. It reads the file here
/// rather than handing it to Dream a chunk at a time, which is the point: a
/// megabyte of input is one call and no allocation on the Dream heap.
NativeResult io_digest_file(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "digest! needs a path");
    std::string path = string_arg(args[0]);
    int fd = sys::open(path.c_str(), O_RDONLY | O_CLOEXEC, 0);
    if (fd < 0) return fail_errno(p, "digest " + path, errno);
    Sha256 sha;
    std::vector<char> buf(1 << 16);
    for (;;) {
        sys::Count got = sys::read(fd, buf.data(), buf.size());
        if (got == 0) break;
        if (got < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            sys::close(fd);
            return fail_errno(p, "digest " + path, e);
        }
        sha.update(buf.data(), size_t(got));
    }
    sys::close(fd);
    return NativeResult::ok(hex_string(p, sha.hex()));
}

/// The SHA-256 of a string, the same digest `digest!` gives the file holding
/// it. Pure, and in `std.io` rather than among the builtins because it is the
/// same question as `digest!` asked of bytes already in hand, and a builtin
/// would cost an opcode for something no inner loop calls.
NativeResult io_digest(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return fail(p, "type_error", "digest needs a string");
    Sha256 sha;
    sha.update(b.data, size_t(b.len));
    return NativeResult::ok(hex_string(p, sha.hex()));
}

// --- what a build does to files ---------------------------------------------
//
// Links, copies, trees and permissions: the rest of what a build asks of the
// file system, beside the stamps and digests above. `std.file` is written on
// these. Each is one call where the same walk written in Dream would be a
// recursion over `list_dir!`, a read and a write per file.

std::filesystem::path fs_path(const std::string& utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string fs_text(const std::filesystem::path& path) {
    auto bytes = path.generic_u8string();
    return std::string(bytes.begin(), bytes.end());
}

NativeResult fail_fs(Process& p, const std::string& what, const std::error_code& ec) {
    const char* kind = "io_error";
    if (ec == std::errc::no_such_file_or_directory) kind = "not_found";
    else if (ec == std::errc::permission_denied || ec == std::errc::operation_not_permitted) kind = "permission_denied";
    else if (ec == std::errc::file_exists) kind = "already_exists";
    else if (ec == std::errc::not_a_directory || ec == std::errc::is_a_directory) kind = "wrong_kind";
    return fail(p, kind, what + ": " + ec.message());
}

/// `link! from to` -- `to` becomes another name for `from`: a hard link, or a
/// copy where one cannot be made (another volume, a file system without
/// them). Either way `to` is new and must not exist yet. A step's outputs are
/// never written to again once made, so a link is as good as a copy and costs
/// nothing; placing a result is a link beside it and a `rename!` over it.
NativeResult io_link(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) return fail(p, "type_error", "link! needs two paths");
    auto from = fs_path(string_arg(args[0]));
    auto to = fs_path(string_arg(args[1]));
    std::error_code ec;
    std::filesystem::create_hard_link(from, to, ec);
    if (!ec) return NativeResult::ok(make_atom(p.runtime().intern_atom("linked")));
    if (ec == std::errc::file_exists || ec == std::errc::no_such_file_or_directory) {
        return fail_fs(p, "link " + fs_text(from) + " to " + fs_text(to), ec);
    }
    ec.clear();
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::none, ec);
    if (ec) return fail_fs(p, "copy " + fs_text(from) + " to " + fs_text(to), ec);
    return NativeResult::ok(make_atom(p.runtime().intern_atom("copied")));
}

/// `copy! from to` -- `to` becomes a copy of `from`, replacing it if it is
/// there, with `from`'s permissions. Where `link!` shares a file, this makes
/// one of `to`'s own: what a build places outside its cache is copied, so that
/// anything which later edits it in place -- `strip`, `patchelf`, a linker
/// writing into its old output -- cannot reach back into the cache.
NativeResult io_copy(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) return fail(p, "type_error", "copy! needs two paths");
    auto from = fs_path(string_arg(args[0]));
    auto to = fs_path(string_arg(args[1]));
    std::error_code ec;
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) return fail_fs(p, "copy " + fs_text(from) + " to " + fs_text(to), ec);
    return NativeResult::ok(UNIT);
}

/// `chmod! path mode` -- set the permission bits, `0o755` being `493`. On
/// Windows only the owner's write bit means anything (it is the read-only
/// attribute), and the rest are accepted and ignored, so a build can mark its
/// outputs executable without asking what it is running on.
NativeResult io_chmod(Process& p, Value, Value* args, uint32_t) {
    Value mode = resolve(args[1]);
    if (!is_string(args[0]) || !is_fixnum(mode) || fixnum_value(mode) < 0 || fixnum_value(mode) > 07777) {
        return fail(p, "type_error", "chmod! needs a path and a mode");
    }
    std::error_code ec;
    std::filesystem::permissions(fs_path(string_arg(args[0])),
                                 std::filesystem::perms(fixnum_value(mode)), ec);
    if (ec) return fail_fs(p, "chmod " + string_arg(args[0]), ec);
    return NativeResult::ok(UNIT);
}

Value stamp_list(Process& p, int64_t size, int64_t modified, const char* kind) {
    Value list = p.heap().make_cons(make_atom(p.runtime().intern_atom(kind)), NIL);
    list = p.heap().make_cons(make_integer(p, modified), list);
    return p.heap().make_cons(make_integer(p, size), list);
}

/// `lstat! path` -- `stat!` of the name itself rather than what it leads to:
/// a symbolic link answers `:link`, its size being the length of the target
/// it holds and its time when the link was made, and a dangling one is still
/// there. Anything that is not a link answers exactly what `stat!` does.
///
/// A separate call rather than a fourth kind from `stat!`, because a stamp
/// wants the file a build will actually read, and that is the one a link
/// leads to.
NativeResult io_lstat(Process& p, Value self, Value* args, uint32_t argc) {
    if (!is_string(args[0])) return fail(p, "type_error", "lstat! needs a path");
    std::string path = string_arg(args[0]);
#if defined(_WIN32)
    // `_wstat64` follows a reparse point and there is no `_wlstat64`; the
    // attributes call does not follow, which is the half wanted here.
    std::error_code ec;
    auto st = std::filesystem::symlink_status(fs_path(path), ec);
    if (ec || st.type() == std::filesystem::file_type::not_found) return NativeResult::ok(UNIT);
    if (st.type() != std::filesystem::file_type::symlink) return io_stat(p, self, args, argc);
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(windows::wide(path.c_str()).c_str(), GetFileExInfoStandard, &data)) {
        return fail(p, "io_error", "lstat " + path);
    }
    // FILETIME counts 100ns ticks from 1601; the epoch is 11644473600s later.
    int64_t ticks = (int64_t(data.ftLastWriteTime.dwHighDateTime) << 32) | data.ftLastWriteTime.dwLowDateTime;
    auto target = std::filesystem::read_symlink(fs_path(path), ec);
    return NativeResult::ok(stamp_list(p, ec ? 0 : int64_t(fs_text(target).size()),
                                       (ticks - 116444736000000000) * 100, "link"));
#else
    (void)self; (void)argc;
    sys::FileStat st{};
    if (::lstat(path.c_str(), &st) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) return NativeResult::ok(UNIT);
        return fail_errno(p, "lstat " + path, errno);
    }
    const char* kind = S_ISLNK(st.st_mode) ? "link" : S_ISDIR(st.st_mode) ? "dir" : S_ISREG(st.st_mode) ? "file" : "other";
    return NativeResult::ok(stamp_list(p, int64_t(st.st_size), modified_ns(st), kind));
#endif
}

/// `mode! path` -- the permission bits `chmod!` sets, `0o755` being `493`, of
/// what the path leads to; `()` when nothing is there, as `stat!` answers.
/// Windows has only the read-only attribute to report, which its C library
/// spreads across the owner, group and other bits: `0o666` or `0o444`, with
/// the execute bits for a directory or a name it runs.
NativeResult io_mode(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "mode! needs a path");
    std::string path = string_arg(args[0]);
    sys::FileStat st{};
    if (sys::stat(path.c_str(), &st) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) return NativeResult::ok(UNIT);
        return fail_errno(p, "mode " + path, errno);
    }
    return NativeResult::ok(make_fixnum(int64_t(st.st_mode) & 07777));
}

/// `is_link! path` -- whether the name itself is a symbolic link, dangling or
/// not; `is_dir!`'s question of the name rather than of where it leads.
NativeResult io_is_link(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "is_link! needs a path");
    std::error_code ec;
    return NativeResult::ok(make_bool(std::filesystem::is_symlink(fs_path(string_arg(args[0])), ec)));
}

/// `symlink! target to` -- `to` becomes a symbolic link holding `target`, in
/// the order `ln -s` and `link!` take them. The target is stored as written:
/// a relative one is read from the link's directory, not from where the
/// program is running, and it need not exist yet. `to` must not.
///
/// Windows makes a link to a directory differently from one to a file, so
/// the target is looked at, from the link's directory, to choose; on POSIX
/// the two calls are the same one. Windows also refuses either without
/// developer mode or the privilege, which surfaces as `:permission_denied`.
NativeResult io_symlink(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) return fail(p, "type_error", "symlink! needs two paths");
    auto target = fs_path(string_arg(args[0]));
    auto to = fs_path(string_arg(args[1]));
    std::error_code ec;
    auto seen = target.is_absolute() ? target : to.parent_path() / target;
    if (std::filesystem::is_directory(seen, ec)) std::filesystem::create_directory_symlink(target, to, ec);
    else std::filesystem::create_symlink(target, to, ec);
    if (ec) return fail_fs(p, "symlink " + fs_text(to) + " to " + fs_text(target), ec);
    return NativeResult::ok(UNIT);
}

/// `read_link! path` -- the target a symbolic link holds, as it was written
/// rather than resolved. Not a link is `:wrong_kind`.
NativeResult io_read_link(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "read_link! needs a path");
    auto path = fs_path(string_arg(args[0]));
    std::error_code ec;
    auto target = std::filesystem::read_symlink(path, ec);
    if (ec == std::errc::invalid_argument) return fail(p, "wrong_kind", "read_link " + fs_text(path) + ": not a symbolic link");
    if (ec) return fail_fs(p, "read_link " + fs_text(path), ec);
    std::string text = fs_text(target);
    return NativeResult::ok(p.heap().make_string(text.data(), uint32_t(text.size())));
}

/// `walk! dir` -- every file under `dir`, as paths relative to it with `/`
/// between the parts, sorted. Directories are walked and not listed, and a
/// link to a directory is not followed, so a tree that links back into itself
/// is still finite. Picking files by pattern is left to the caller.
NativeResult io_walk(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "walk! needs a directory");
    auto root = fs_path(string_arg(args[0]));
    std::error_code ec;
    std::filesystem::recursive_directory_iterator it(
        root, std::filesystem::directory_options::skip_permission_denied, ec), end;
    if (ec) return fail_fs(p, "walk " + fs_text(root), ec);
    std::vector<std::string> files;
    for (; it != end; it.increment(ec)) {
        if (ec) return fail_fs(p, "walk " + fs_text(root), ec);
        std::error_code kind_ec;
        if (it->is_regular_file(kind_ec)) files.push_back(fs_text(it->path().lexically_relative(root)));
    }
    std::sort(files.begin(), files.end());
    Value list = NIL;
    for (size_t i = files.size(); i-- > 0;) {
        list = p.heap().make_cons(p.heap().make_string(files[i].data(), uint32_t(files[i].size())), list);
    }
    return NativeResult::ok(list);
}

/// `mkdir_all! path` -- the directory and any parents it needs. Already there
/// is not an error.
NativeResult io_mkdir_all(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "mkdir_all! needs a path");
    std::error_code ec;
    std::filesystem::create_directories(fs_path(string_arg(args[0])), ec);
    if (ec) return fail_fs(p, "mkdir " + string_arg(args[0]), ec);
    return NativeResult::ok(UNIT);
}

/// `remove_all! path` -- the path and everything under it, answering how many
/// entries went. Nothing there is not an error and answers 0: a scratch
/// directory is cleared whether or not the last run got as far as making it.
NativeResult io_remove_all(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "remove_all! needs a path");
    std::error_code ec;
    auto removed = std::filesystem::remove_all(fs_path(string_arg(args[0])), ec);
    if (ec) return fail_fs(p, "remove " + string_arg(args[0]), ec);
    return NativeResult::ok(make_integer(p, int64_t(removed)));
}

NativeResult io_remove(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "remove! needs a path");
    std::string path = string_arg(args[0]);
    if (sys::remove(path.c_str()) != 0) return fail_errno(p, "remove " + path, errno);
    return NativeResult::ok(UNIT);
}

NativeResult io_rename(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) {
        return fail(p, "type_error", "rename! needs two paths");
    }
    std::string from = string_arg(args[0]);
    if (sys::rename(from.c_str(), string_arg(args[1]).c_str()) != 0) {
        return fail_errno(p, "rename " + from, errno);
    }
    return NativeResult::ok(UNIT);
}

NativeResult io_mkdir(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "mkdir! needs a path");
    std::string path = string_arg(args[0]);
    if (sys::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return fail_errno(p, "mkdir " + path, errno);
    }
    return NativeResult::ok(UNIT);
}

// ---------------------------------------------------------------------------
// std.net
//
// Sockets are handles like any other, so `io.read!` and `io.write!` work on
// them unchanged -- that is what "net on top of io" means here. Only the
// operations that have no file equivalent live in this module: listening,
// accepting, and connecting.
// ---------------------------------------------------------------------------

/// Bind and listen on `port` at `addr`, whose port is filled in here.
static NativeResult listen_at(Process& p, sockaddr_in addr, int64_t port) {
    int fd = sys::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return fail_errno(p, "socket", errno);

    // Without this, a listener that has just exited leaves the port unusable
    // for a minute or two, which makes every restart of a server a coin toss.
    int on = 1;
    sys::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (sys::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
        int e = errno;
        sys::close(fd);
        return fail_errno(p, "bind", e);
    }
    if (sys::listen(fd, 128) < 0) {
        int e = errno;
        sys::close(fd);
        return fail_errno(p, "listen", e);
    }
    set_nonblocking(fd);
    return NativeResult::ok(make_fixnum(HandleTable::get().add(fd, HandleKind::Listener)));
}

/// Bind and listen on `port`, on every interface.
NativeResult net_listen(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v) || fixnum_value(v) < 0 || fixnum_value(v) > 65535) {
        return fail(p, "bad_argument", "listen! needs a port between 0 and 65535");
    }
    sockaddr_in addr{};
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    return listen_at(p, addr, fixnum_value(v));
}

/// Bind and listen on `port` at one address: `"127.0.0.1"` or `"localhost"`
/// for this machine alone, `"0.0.0.0"` for every interface, or a name that
/// resolves to one of this machine's addresses. A server that is only meant
/// to be reached from here should say so, rather than be reachable by the
/// whole network because `listen!` cannot say anything else.
NativeResult net_listen_on(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "listen_on! needs a host to bind to");
    Value v = resolve(args[1]);
    if (!is_fixnum(v) || fixnum_value(v) < 0 || fixnum_value(v) > 65535) {
        return fail(p, "bad_argument", "listen_on! needs a port between 0 and 65535");
    }
    std::string host = string_arg(args[0]);
    // IPv4, as `connect!` is. `getaddrinfo` blocks, but a listener is made
    // once, and a numeric host never touches the resolver.
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* found = nullptr;
    int gai = ::getaddrinfo(host.c_str(), nullptr, &hints, &found);
    if (gai != 0 || !found) {
        return fail(p, "not_found", "cannot resolve " + host + ": " + ::gai_strerror(gai));
    }
    sockaddr_in addr{};
    std::memcpy(&addr, found->ai_addr, sizeof addr);
    ::freeaddrinfo(found);
    return listen_at(p, addr, fixnum_value(v));
}

/// Take the next incoming connection, parking this process until one arrives.
NativeResult net_accept(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this listener is closed");
    if (h.kind() != HandleKind::Listener) {
        return fail(p, "wrong_kind", "accept! needs a listening socket");
    }

    for (;;) {
        int client = sys::accept(h.fd(), nullptr, nullptr);
        if (client >= 0) {
            set_nonblocking(client);
            int on = 1;
            // Small writes should go out now, not wait for more to accumulate.
            sys::setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
            return NativeResult::ok(
                make_fixnum(HandleTable::get().add(client, HandleKind::Stream, true, true)));
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return wait_for(p, h.fd(), false);
        return fail_errno(p, "accept", errno);
    }
}

/// Open a connection to `host` on `port`.
///
/// This is the one operation that cannot simply be retried on wake-up: the
/// second `connect` on the same socket does not redo the work, it reports what
/// happened. So the handle remembers that a connection is in flight, and the
/// retry asks the socket how it went.
NativeResult net_connect_finish(Process& p);

NativeResult net_connect(Process& p, Value, Value* args, uint32_t) {
    // Woken from the wait below: the attempt is over, and the socket knows
    // whether it worked.
    if (p.io_pending >= 0) return net_connect_finish(p);

    if (!is_string(args[0])) return fail(p, "type_error", "connect! needs a host name");
    Value pv = resolve(args[1]);
    if (!is_fixnum(pv) || fixnum_value(pv) < 0 || fixnum_value(pv) > 65535) {
        return fail(p, "bad_argument", "connect! needs a port between 0 and 65535");
    }
    std::string host = string_arg(args[0]);
    std::string port = std::to_string(fixnum_value(pv));

    // `getaddrinfo` blocks, and there is no portable non-blocking resolver.
    // It is called once per connection, before the socket exists, so it stalls
    // one worker briefly rather than for the life of the connection.
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    int gai = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &found);
    if (gai != 0 || !found) {
        return fail(p, "not_found", "cannot resolve " + host + ": " + ::gai_strerror(gai));
    }

    int fd = sys::socket(found->ai_family, found->ai_socktype, found->ai_protocol);
    if (fd < 0) {
        int e = errno;
        ::freeaddrinfo(found);
        return fail_errno(p, "socket", e);
    }
    set_nonblocking(fd);
    int rc = sys::connect(fd, found->ai_addr, static_cast<sys::SockLen>(found->ai_addrlen));
    int e = errno;
    ::freeaddrinfo(found);

    if (rc == 0) {
        int on = 1;
        sys::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
        return NativeResult::ok(make_fixnum(HandleTable::get().add(fd, HandleKind::Stream, true, true)));
    }
    if (e != EINPROGRESS && e != EINTR) {
        sys::close(fd);
        return fail_errno(p, "connect to " + host, e);
    }

    // In flight. Hand back a handle so the retry has somewhere to keep state,
    // and wait for the socket to become writable, which is how the kernel
    // reports that the attempt has finished one way or the other.
    int64_t id = HandleTable::get().add(fd, HandleKind::Stream, true, true);
    {
        std::lock_guard<std::mutex> g(HandleTable::get().mutex());
        if (Handle* h = HandleTable::get().find(id)) h->connecting = true;
    }
    p.io_pending = id;
    return wait_for(p, fd, true);
}

/// The second half of `connect!`, reached when the socket became writable.
NativeResult net_connect_finish(Process& p) {
    int64_t id = p.io_pending;
    p.io_pending = -1;
    std::unique_lock<std::mutex> lock(HandleTable::get().mutex());
    Handle* h = HandleTable::get().find(id);
    if (!h) return fail(p, "io_closed", "the connection was closed while it was being made");
    h->connecting = false;

    int err = 0;
    sys::SockLen len = sizeof err;
    if (sys::getsockopt(h->fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0) err = errno;
    if (err != 0) {
        int fd = h->fd;
        h->open = false;
        h->fd = -1;
        lock.unlock();
        sys::close(fd);
        return fail_errno(p, "connect", err);
    }
    int on = 1;
    sys::setsockopt(h->fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
    lock.unlock();
    return NativeResult::ok(make_fixnum(id));
}

/// The address of the other end of a connection, as "host:port".
NativeResult net_peer(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this socket is closed");
    sockaddr_in addr{};
    sys::SockLen len = sizeof addr;
    if (sys::getpeername(h.fd(), reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
        return fail_errno(p, "peer", errno);
    }
    char text[INET_ADDRSTRLEN] = {0};
    ::inet_ntop(AF_INET, &addr.sin_addr, text, sizeof text);
    std::string out = std::string(text) + ":" + std::to_string(ntohs(addr.sin_port));
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// The port a listener actually got, which is the only way to learn it after
/// asking for port 0.
NativeResult net_port(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this socket is closed");
    sockaddr_in addr{};
    sys::SockLen len = sizeof addr;
    if (sys::getsockname(h.fd(), reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
        return fail_errno(p, "port", errno);
    }
    return NativeResult::ok(make_fixnum(int64_t(ntohs(addr.sin_port))));
}

/// Stop sending, so the other end sees end of file while this end can still
/// read what is already on the way.
NativeResult net_shutdown(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this socket is closed");
    // Over TLS the end of what this side sends is a close_notify, and only
    // then the end of the stream.
    if (h.tls()) tls_goodbye(*h.tls(), h.fd());
    sys::shutdown(h.fd(), SHUT_WR);
    return NativeResult::ok(UNIT);
}

// ---------------------------------------------------------------------------
// std.tls
//
// A connection becomes TLS in place: `connect!` and `accept!` take a handle
// that is already a connected stream and answer the same handle, which from
// then on reads and writes plaintext through the session. In place, rather
// than a new handle, because the protocols that matter most upgrade a
// connection they have already spoken on -- PostgreSQL's SSLRequest, SMTP's
// STARTTLS -- and because every function that takes a handle then takes a TLS
// one unchanged: `std.streams`, `std.remote`, a program's own reader.
// ---------------------------------------------------------------------------

/// The value at atom `key` in an options map, forced, or false when absent.
bool option(Process& p, Value map, const char* key, Value* out) {
    Value found;
    if (!map_lookup(p, map, make_atom(p.runtime().intern_atom(key)), &found)) return false;
    return force_whnf(p, found, out);
}

/// A string option. False with `*bad` set when it is there and not a string.
bool string_option(Process& p, Value map, const char* key, std::string* out, bool* bad) {
    Value v;
    if (!option(p, map, key, &v) || is_unit(v)) return false;
    Bytes b;
    if (!string_bytes(v, &b)) {
        *bad = true;
        return false;
    }
    out->assign(b.data, size_t(b.len));
    return true;
}

/// The options map as a configuration, or a message saying which one is wrong.
std::string read_config(Process& p, Value options, tls::Config* c) {
    Value map;
    if (!force_whnf(p, options, &map)) return "the options could not be evaluated";
    if (is_unit(map)) return "";
    if (!is_obj(map, ObjType::Map)) return "the options are a map, such as %{ :host => \"example.com\" }";
    bool bad = false;
    string_option(p, map, "host", &c->host, &bad);
    if (bad) return "`:host` is a string";
    string_option(p, map, "ca", &c->ca_pem, &bad);
    if (bad) return "`:ca` is a string of PEM certificates";
    string_option(p, map, "crl", &c->crl_pem, &bad);
    if (bad) return "`:crl` is a string of PEM revocation lists";
    string_option(p, map, "identity", &c->identity_p12, &bad);
    if (bad) return "`:identity` is a string holding a PKCS#12 bundle";
    string_option(p, map, "password", &c->identity_password, &bad);
    if (bad) return "`:password` is a string";
    Value v;
    if (option(p, map, "verify", &v)) {
        if (!is_bool(v)) return "`:verify` is true or false";
        c->verify = truthy(v);
    }
    if (option(p, map, "check_name", &v)) {
        if (!is_bool(v)) return "`:check_name` is true or false";
        c->check_name = truthy(v);
    }
    if (option(p, map, "alpn", &v)) {
        Value cur = v;
        for (;;) {
            Value cell;
            if (!force_whnf(p, cur, &cell)) return "`:alpn` could not be evaluated";
            if (is_nil(cell)) break;
            if (!is_obj(cell, ObjType::Cons)) return "`:alpn` is a list of protocol names";
            auto* cons = static_cast<ConsObj*>(as_obj(cell));
            Value name;
            Bytes b;
            if (!force_whnf(p, cons->head, &name) || !string_bytes(name, &b)) {
                return "`:alpn` is a list of protocol names";
            }
            c->alpn.emplace_back(b.data, size_t(b.len));
            cur = cons->tail;
        }
    }
    return "";
}

/// The handshake, for either side. Entered again after every park until it
/// finishes; the session remembers how far it got.
NativeResult tls_handshake(Process& p, Value* args, bool server) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    if (h.kind() != HandleKind::Stream) {
        return fail(p, "wrong_kind", server ? "tls.accept! needs an accepted connection"
                                            : "tls.connect! needs a connected socket");
    }
    std::shared_ptr<TlsSession> session = h.tls();
    if (!session) {
        tls::Config config;
        config.server = server;
        config.verify = !server;
        std::string bad = read_config(p, args[1], &config);
        if (!bad.empty()) return fail(p, "bad_argument", bad);
        if (!server && config.verify && config.check_name && config.host.empty()) {
            return fail(p, "bad_argument",
                        "a client that verifies needs `:host`, the name the certificate must carry; "
                        "`:check_name => false` checks the chain alone, and `:verify => false` nothing");
        }
        if (server && config.identity_p12.empty()) {
            return fail(p, "bad_argument", "a server needs `:identity`, a PKCS#12 bundle of its certificate and key");
        }
        tls::Failure why;
        std::unique_ptr<tls::Engine> engine = tls::make_engine(config, &why);
        if (!engine) return fail(p, why.kind.c_str(), why.message);
        session = std::make_shared<TlsSession>();
        session->engine = std::move(engine);
        if (!HandleTable::get().attach_tls(h.id(), session)) {
            return fail(p, "io_closed", "this handle is closed");
        }
    }

    TlsSession& s = *session;
    std::lock_guard<std::mutex> g(s.mutex);
    if (s.state == TlsSession::State::Failed) return tls_refused(p, s);
    if (s.state == TlsSession::State::Ready && s.announced) {
        return fail(p, "tls_error", "this connection already speaks TLS");
    }
    for (;;) {
        int err = 0;
        switch (flush_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), true);
            case Pump::Failed:
                s.state = TlsSession::State::Failed;
                s.failure = {"handshake_failed", std::string("the connection failed: ") + std::strerror(err)};
                return fail(p, "handshake_failed", s.failure.message);
            default: break;
        }
        if (s.state == TlsSession::State::Ready) {
            s.announced = true;
            return NativeResult::ok(make_fixnum(h.id()));
        }
        tls::Status st = s.engine->handshake();
        s.out += s.engine->take_output();
        switch (st) {
            case tls::Status::Ok:
                s.state = TlsSession::State::Ready;
                continue;
            case tls::Status::Error:
                return tls_failed(p, s, h.fd());
            case tls::Status::Closed:
                s.state = TlsSession::State::Failed;
                s.failure = {"handshake_failed", "the peer closed the connection during the handshake"};
                return fail(p, "handshake_failed", s.failure.message);
            case tls::Status::WantRead:
                break;
        }
        if (s.eof) {
            s.state = TlsSession::State::Failed;
            s.failure = {"handshake_failed", "the peer closed the connection during the handshake"};
            return fail(p, "handshake_failed", s.failure.message);
        }
        switch (flush_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), true);
            case Pump::Failed: return fail_errno(p, "write", err);
            default: break;
        }
        switch (pull_tls(s, h.fd(), &err)) {
            case Pump::Again: return wait_for(p, h.fd(), false);
            case Pump::Failed: return fail_errno(p, "read", err);
            default: break;
        }
    }
}

NativeResult tls_connect(Process& p, Value, Value* args, uint32_t) { return tls_handshake(p, args, false); }
NativeResult tls_accept(Process& p, Value, Value* args, uint32_t) { return tls_handshake(p, args, true); }

Value text_or_unit(Process& p, const std::string& s) {
    return s.empty() ? UNIT : p.heap().make_string(s.data(), uint32_t(s.size()));
}

/// `%{ :version, :cipher, :alpn, :peer }` for a TLS handle, `()` for a plain one.
NativeResult tls_info(Process& p, Value, Value* args, uint32_t) {
    Held h;
    if (!h.open(args[0])) return fail(p, "io_closed", "this handle is closed");
    if (!h.tls()) return NativeResult::ok(UNIT);
    tls::Info info;
    {
        std::lock_guard<std::mutex> g(h.tls()->mutex);
        if (h.tls()->state != TlsSession::State::Ready) return tls_refused(p, *h.tls());
        info = h.tls()->engine->info();
    }
    Value m = p.heap().make_map(4);
    auto put = [&](const char* key, const std::string& value) {
        m = map_insert(p, m, make_atom(p.runtime().intern_atom(key)), text_or_unit(p, value));
    };
    put("version", info.version);
    put("cipher", info.cipher);
    put("alpn", info.alpn);
    put("peer", info.peer);
    return NativeResult::ok(m);
}

NativeResult tls_backend(Process& p, Value, Value*, uint32_t) {
    std::string name = tls::backend();
    return NativeResult::ok(p.heap().make_string(name.data(), uint32_t(name.size())));
}

void forget_waiter(int fd) { Poller::get().forget(fd); }

}  // namespace

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

void io_init() { if (sys::init()) Poller::get().start(); }

void io_shutdown() {
    Poller::get().stop();
    HandleTable::get().close_all();
}

bool io_async_available() { return Poller::get().available(); }

std::vector<IoHandleInfo> io_snapshot() {
    // The waiter is asked for outside the table lock: the two are separate
    // locks, and taking them together here would invent an ordering that
    // nothing else obeys.
    std::vector<IoHandleInfo> out = HandleTable::get().snapshot();
    for (IoHandleInfo& info : out) info.waiting_pid = Poller::get().waiter_of(info.fd);
    return out;
}

ModuleDef make_io_module() {
    return ModuleDef{"std.io",
                     {
                         {"open!", 2, 0b11, io_open},
                         {"temp_file!", 2, 0b11, io_temp, 0},
                         {"temp_dir!", 2, 0b11, io_temp, 1},
                         {"close!", 1, 0b1, io_close},
                         {"read!", 2, 0b11, io_read},
                         {"write!", 2, 0b11, io_write},
                         {"flush!", 1, 0b1, io_flush},
                         {"seek!", 2, 0b11, io_seek},
                         {"size!", 1, 0b1, io_size},
                         {"kind!", 1, 0b1, io_kind},
                         {"is_open!", 1, 0b1, io_is_open},
                         {"is_terminal!", 1, 0b1, io_is_terminal},
                         {"enable_ansi!", 1, 0b1, io_enable_ansi},
                         {"stdin!", 1, 0b1, io_stdin},
                         {"stdout!", 1, 0b1, io_stdout},
                         {"stderr!", 1, 0b1, io_stderr},
                         {"exists!", 1, 0b1, io_exists},
                         {"is_dir!", 1, 0b1, io_is_dir},
                         {"remove!", 1, 0b1, io_remove},
                         {"rename!", 2, 0b11, io_rename},
                         {"mkdir!", 1, 0b1, io_mkdir},
                         {"stat!", 1, 0b1, io_stat},
                         {"digest!", 1, 0b1, io_digest_file},
                         {"digest", 1, 0b1, io_digest},
                         {"mkdir_all!", 1, 0b1, io_mkdir_all},
                         {"remove_all!", 1, 0b1, io_remove_all},
                         {"link!", 2, 0b11, io_link},
                         {"copy!", 2, 0b11, io_copy},
                         {"chmod!", 2, 0b11, io_chmod},
                         {"mode!", 1, 0b1, io_mode},
                         {"lstat!", 1, 0b1, io_lstat},
                         {"is_link!", 1, 0b1, io_is_link},
                         {"symlink!", 2, 0b11, io_symlink},
                         {"read_link!", 1, 0b1, io_read_link},
                         {"walk!", 1, 0b1, io_walk},
                         {"async", 1, 0b1, io_async},
                     }};
}

ModuleDef make_tls_module() {
    return ModuleDef{"std.tls",
                     {
                         {"connect!", 2, 0b01, tls_connect},
                         {"accept!", 2, 0b01, tls_accept},
                         {"info!", 1, 0b1, tls_info},
                         {"backend", 1, 0b0, tls_backend},
                     }};
}

ModuleDef make_net_module() {
    return ModuleDef{"std.net",
                     {
                         {"listen!", 1, 0b1, net_listen},
                         {"listen_on!", 2, 0b11, net_listen_on},
                         {"accept!", 1, 0b1, net_accept},
                         {"connect!", 2, 0b11, net_connect},
                         {"peer!", 1, 0b1, net_peer},
                         {"port!", 1, 0b1, net_port},
                         {"shutdown!", 1, 0b1, net_shutdown},
                     }};
}

}  // namespace dream
