// The operating system: arguments, environment, directories, subprocesses.
//
// The interesting one is `exec!`. Running a child to completion means waiting
// for it, and a Dream process must never wait on a worker thread -- that would
// block every other process queued behind it. So `exec!` hands the whole job
// (spawn, read both pipes, reap) to a helper thread and parks the calling
// process through the same handshake `recv!` and the IO natives use. The
// helper wakes it when the child has exited.
//
// A helper thread rather than the epoll poller because a child process is not
// one descriptor but three things at once -- two pipes and an exit status --
// and driving that from a native that is re-entered from the top on every
// wake-up would mean keeping the partially-read output somewhere. One thread
// per running child is the honest cost, and a build tool runs a handful.

#include "os.hpp"

#include <algorithm>
#include <filesystem>
#include "windows.hpp"
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <atomic>
#include <condition_variable>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <spawn.h>
#if defined(DREAM_TERMUX)
// Termux supplies its own spawn.h, which omits this API 34 libc function.
// We target API 34 and link Bionic directly, not libandroid-spawn's older ABI.
extern "C" int posix_spawn_file_actions_addchdir_np(posix_spawn_file_actions_t*, const char*);
#endif
#include <sys/stat.h>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "builtins.hpp"
#include "interp.hpp"
#include "process.hpp"
#include "scheduler.hpp"

#ifndef _WIN32
extern char** environ;
#endif

namespace dream {
namespace {

Value os_error(Process& p, const char* kind, const std::string& message) {
    return raise_error(p, p.runtime().intern_atom(kind), message);
}

NativeResult fail(Process& p, const char* kind, const std::string& message) {
    return NativeResult::raise(os_error(p, kind, message));
}

std::string string_arg(Value v) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Str)) return std::string();
    auto* s = static_cast<StrObj*>(as_obj(v));
    return std::string(s->data(), s->len);
}

bool is_string(Value v) { return is_obj(resolve(v), ObjType::Str); }

Value text(Process& p, const std::string& s) {
    return p.heap().make_string(s.data(), uint32_t(s.size()));
}

/// A list of strings, built back to front so it comes out in order.
Value string_list(Process& p, const std::vector<std::string>& items) {
    Value list = NIL;
    for (size_t i = items.size(); i-- > 0;) {
        list = p.heap().make_cons(text(p, items[i]), list);
    }
    return list;
}

/// Read a Dream list of strings into a vector. Returns false if any element is
/// not a string, which is the only way `exec!` can be misused.
bool read_string_list(Process& p, Value list, std::vector<std::string>* out) {
    Value cur = list;
    for (;;) {
        Value w;
        if (!force_whnf(p, cur, &w)) return false;
        if (is_nil(w)) return true;
        if (!is_obj(w, ObjType::Cons)) return false;
        auto* c = static_cast<ConsObj*>(as_obj(w));
        Value head;
        if (!force_whnf(p, c->head, &head)) return false;
        if (!is_obj(head, ObjType::Str)) return false;
        auto* s = static_cast<StrObj*>(as_obj(head));
        out->emplace_back(s->data(), s->len);
        cur = c->tail;
    }
}

// ---------------------------------------------------------------------------
// Subprocesses
// ---------------------------------------------------------------------------

/// One `exec!` in flight. The Dream process holds only the id; the job itself
/// outlives the native call, because the native returns as soon as it has
/// parked and is entered again from the top when it is woken.
struct Job {
    std::atomic<bool> done{false};
    int code = -1;
    bool started = false;
    bool timed_out = false;
    std::string error;   // non-empty when the child could not be started
    std::string out;
    std::string err;
    std::thread worker;
    /// Where the child starts, or empty for this VM's own working directory.
    /// A child's own, because `chdir!` is the whole VM's: two build steps
    /// running at once could not each change it without breaking the other.
    std::string dir;
    /// `NAME=value` for each variable the child gets in addition to -- or in
    /// place of -- this VM's own. A child's own, like `dir`: `set_env!` is
    /// the whole VM's, and two checks running at once each want theirs.
    std::vector<std::string> env;
    /// The child's errors into the same pipe as its output, so that `out`
    /// holds both in the order they were written -- what a person running it
    /// in a terminal sees, and what a check compares against a recorded file.
    bool joined = false;

    /// 0 means "wait as long as it takes". Anything else is a deadline after
    /// which the child is killed -- worth having because a build tool spends
    /// its life running other people's programs, and `git clone` against an
    /// unreachable host will otherwise wait for ever.
    int64_t timeout_ms = 0;
    std::mutex finish_mutex;
    std::condition_variable finished;
    bool child_done = false;
};

class Jobs {
public:
    static Jobs& get() {
        static Jobs j;
        return j;
    }

    int64_t add(std::shared_ptr<Job> job) {
        std::lock_guard<std::mutex> g(mutex_);
        int64_t id = next_++;
        jobs_[id] = std::move(job);
        return id;
    }

    std::shared_ptr<Job> find(int64_t id) {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    void erase(int64_t id) {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = jobs_.find(id);
            if (it == jobs_.end()) return;
            job = it->second;
            jobs_.erase(it);
        }
        if (job->worker.joinable()) job->worker.join();
    }

    /// Wait for every child still running, at shutdown.
    void drain() {
        std::vector<std::shared_ptr<Job>> all;
        {
            std::lock_guard<std::mutex> g(mutex_);
            for (auto& [id, job] : jobs_) all.push_back(job);
            jobs_.clear();
        }
        for (auto& job : all) {
            if (job->worker.joinable()) job->worker.join();
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<int64_t, std::shared_ptr<Job>> jobs_;
    int64_t next_ = 1;
};

/// The variables that switch on a VM's debugging -- tracing, synchronous or
/// refused compiles, heap verification -- and are not passed to a child
/// `exec!` starts. Each is about the VM it was set for; inherited, it made a
/// whole tree of VMs slow and noisy where one was being looked at. Tuning
/// (`DREAM_CORES`, `DREAM_NURSERY_MAX`, ..) and finding things
/// (`DREAM_LIBRARY_PATH`, `MINDV2_PATH`) are still inherited.
bool debug_variable(const std::string& name) {
    static const char* const names[] = {
        "DREAM_JIT_SYNC", "DREAM_JIT_WHY", "DREAM_JIT_TRACE", "DREAM_JIT_DENY",
        "DREAM_JIT_AHEAD", "DREAM_JIT_CHECK_EFFECTS", "DREAM_PROBE_THUNK",
        "DREAM_VERIFY_HEAP", "DREAM_GC_TRACE", "DREAM_IO_TRACE", "DREAM_STUCK_SECONDS",
    };
    for (const char* n : names) if (name == n) return true;
    return false;
}

bool inherit_debug_variables() {
    const char* v = std::getenv("DREAM_DEBUG_INHERIT");
    return v && *v && std::strcmp(v, "0") != 0;
}

#ifdef _WIN32
// Pipe reads live on helper threads, just as on POSIX. Only the explicitly
// listed standard handles are inherited by the child.
std::string slurp(HANDLE pipe) {
    std::string out;
    char buf[4096];
    DWORD n;
    while (ReadFile(pipe, buf, sizeof buf, &n, nullptr) && n) out.append(buf, n);
    CloseHandle(pipe);
    return out;
}

void run_child(std::shared_ptr<Job> job, std::vector<std::string> argv,
               uint64_t pid, uint64_t sched) {
    auto finish = [&] {
        job->done.store(true, std::memory_order_release);
        Scheduler::wake_external(sched, pid);
    };
    HANDLE out_read = nullptr, out_write = nullptr, err_read = nullptr, err_write = nullptr;
    HANDLE input = INVALID_HANDLE_VALUE;
    auto cleanup = [&] {
        for (HANDLE h : {out_read, out_write, err_read, err_write, input})
            if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h);
    };
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    if (!CreatePipe(&out_read, &out_write, &security, 0) ||
        !CreatePipe(&err_read, &err_write, &security, 0) ||
        !SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0)) {
        job->error = "cannot create pipes: " + windows::error();
        cleanup(); finish(); return;
    }
    HANDLE current_input = GetStdHandle(STD_INPUT_HANDLE);
    if (!current_input || current_input == INVALID_HANDLE_VALUE ||
        !DuplicateHandle(GetCurrentProcess(), current_input, GetCurrentProcess(),
                         &input, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
        input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &security, OPEN_EXISTING, 0, nullptr);
    }
    STARTUPINFOEXW start{};
    start.StartupInfo.cb = sizeof(start);
    start.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    start.StartupInfo.hStdInput = input;
    start.StartupInfo.hStdOutput = out_write;
    // Joined, the error stream is the output's pipe, and the handle is
    // listed once below: a handle listed twice is refused.
    start.StartupInfo.hStdError = job->joined ? out_write : err_write;
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> attributes(bytes);
    start.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(start.lpAttributeList, 1, 0, &bytes)) {
        job->error = windows::error(); cleanup(); finish(); return;
    }
    HANDLE handles[] = {input, out_write, err_write};
    bool ready = UpdateProcThreadAttribute(start.lpAttributeList, 0,
        PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles,
        (job->joined ? 2 : 3) * sizeof(HANDLE), nullptr, nullptr);
    std::wstring command;
    for (const auto& arg : argv) {
        if (!command.empty()) command += L' ';
        command += windows::quote(arg);
    }
    // The same overlay as on POSIX, as the block `CreateProcessW` takes: every
    // `NAME=value` NUL-terminated, and one more NUL at the end. Names compare
    // without case, as Windows compares them.
    std::wstring block;
    if (!job->env.empty() || !inherit_debug_variables()) {
        auto upper_name = [](const std::wstring& kv) {
            std::wstring n = kv.substr(0, kv.find(L'=', 1));
            for (auto& c : n) c = towupper(c);
            return n;
        };
        std::vector<std::wstring> ours;
        for (const std::string& kv : job->env) ours.push_back(windows::wide(kv));
        wchar_t* inherited = GetEnvironmentStringsW();
        for (wchar_t* e = inherited; e && *e; e += wcslen(e) + 1) {
            std::wstring entry(e);
            // The name as UTF-8, converted once: building it from two calls'
            // iterators paired the begin of one temporary with the end of
            // another, and narrowed each UTF-16 unit to a byte on the way.
            bool replaced = !inherit_debug_variables() &&
                            debug_variable(windows::utf8(upper_name(entry)));
            for (const std::wstring& kv : ours) {
                if (upper_name(kv) == upper_name(entry)) { replaced = true; break; }
            }
            if (!replaced) { block += entry; block.push_back(L'\0'); }
        }
        if (inherited) FreeEnvironmentStringsW(inherited);
        for (const std::wstring& kv : ours) { block += kv; block.push_back(L'\0'); }
        block.push_back(L'\0');
    }
    PROCESS_INFORMATION child{};
    bool created = ready && CreateProcessW(nullptr, command.data(), nullptr, nullptr,
        TRUE, EXTENDED_STARTUPINFO_PRESENT | (block.empty() ? 0 : CREATE_UNICODE_ENVIRONMENT),
        block.empty() ? nullptr : block.data(),
        job->dir.empty() ? nullptr : windows::wide(job->dir).c_str(), &start.StartupInfo, &child);
    DWORD error = GetLastError();
    DeleteProcThreadAttributeList(start.lpAttributeList);
    CloseHandle(out_write); out_write = nullptr;
    CloseHandle(err_write); err_write = nullptr;
    if (!created) {
        job->error = "cannot run `" + argv[0] + "`: " + windows::error(error);
        cleanup(); finish(); return;
    }
    CloseHandle(input); input = nullptr;
    CloseHandle(child.hThread);
    job->started = true;
    std::thread out_reader([&] { job->out = slurp(out_read); });
    std::thread err_reader([&] { job->err = slurp(err_read); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(job->timeout_ms);
    while (WaitForSingleObject(child.hProcess, 50) == WAIT_TIMEOUT) {
        if (job->timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
            job->timed_out = true;
            TerminateProcess(child.hProcess, 124);
            WaitForSingleObject(child.hProcess, INFINITE);
            break;
        }
    }
    DWORD code = 0;
    GetExitCodeProcess(child.hProcess, &code);
    job->code = int(code);
    CloseHandle(child.hProcess);
    out_reader.join(); err_reader.join();
    finish();
}
#else
/// Read everything from `fd` until end of file, then close it.
std::string slurp(int fd) {
    std::string out;
    char buf[4096];
    for (;;) {
        ssize_t n = ::read(fd, buf, sizeof buf);
        if (n > 0) { out.append(buf, size_t(n)); continue; }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
    ::close(fd);
    return out;
}

/// Spawn `argv`, collect both streams, reap, and wake `pid`.
///
/// The scheduler is named by its serial rather than held, because this thread
/// can outlive it: a program may end while a child it started is still being
/// reaped. See `Scheduler::serial`.
void run_child(std::shared_ptr<Job> job, std::vector<std::string> argv,
               uint64_t pid, uint64_t sched) {
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    auto finish = [&] {
        job->done.store(true, std::memory_order_release);
        if (sched) {
            // Wake first, then stop counting this process as an IO waiter.
            //
            // The order matters. `io_waiters` is what tells the deadlock check
            // that a parked process is owed a wake-up from outside the
            // scheduler. Clearing it first opens a window in which the process
            // is neither counted as waiting nor yet runnable, and an idle
            // worker looking in that window sees no runnable process, an empty
            // queue and no IO waiters -- and declares a deadlock that is not
            // one. With one worker the check runs every 500us and hit that
            // window every time.
            //
            // This way round the count is merely released a moment late, which
            // can only delay a real deadlock report, never invent one.
            Scheduler::wake_external(sched, pid);
        }
    };

    // Close-on-exec, so that only *this* child gets these pipes: the spawn's
    // `dup2` onto 1 and 2 clears the flag on the copies it means to hand over.
    // Without it every child started while this one runs inherits our write
    // ends, and our reader sees no end of file until the last of those exits
    // -- an example that printed and quit at once was reported as timing out
    // because an unrelated, endless one was spawned beside it. macOS has no
    // `pipe2`; the window between `pipe` and `fcntl` there is a race this
    // narrows rather than closes.
    auto cloexec_pipe = [](int fds[2]) {
#if defined(__APPLE__)
        if (::pipe(fds) != 0) return false;
        ::fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        ::fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        return true;
#else
        return ::pipe2(fds, O_CLOEXEC) == 0;
#endif
    };
    if (!cloexec_pipe(out_pipe) || !cloexec_pipe(err_pipe)) {
        job->error = std::string("cannot create a pipe: ") + std::strerror(errno);
        for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]}) {
            if (fd >= 0) ::close(fd);
        }
        finish();
        return;
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], 1);
    posix_spawn_file_actions_adddup2(&actions, job->joined ? out_pipe[1] : err_pipe[1], 2);
    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
    // Joined, the child must not hold the error pipe open either, or the
    // reader of it would wait for the child rather than see its end at once.
    if (job->joined) posix_spawn_file_actions_addclose(&actions, err_pipe[1]);
    if (!job->dir.empty()) {
        // Before the program is looked up, so a relative program name is
        // relative to `dir`, as it would be to a shell that had `cd`'d there.
        // Every libc this VM is built against has it: glibc since 2.29, and
        // macOS since 10.15, Android since API 34. Nothing here falls back to `chdir!`, which would
        // move every other process's working directory with it.
        posix_spawn_file_actions_addchdir_np(&actions, job->dir.c_str());
    }

    std::vector<char*> args;
    args.reserve(argv.size() + 1);
    for (std::string& a : argv) args.push_back(a.data());
    args.push_back(nullptr);

    pid_t child = 0;
    // `posix_spawnp`, so a bare program name is found on PATH -- `git` should
    // mean what it means in a shell.
    // This VM's environment with the job's variables laid over it: a name the
    // job sets is dropped from what is inherited, and the job's value added.
    //
    // And without the VM's debugging switches (`debug_variable`), which are
    // about *this* VM: `DREAM_JIT_SYNC=1 mind docs` made every compiler mind
    // started compile synchronously, and its examples reported timeouts that
    // were not there. A job that names one passes it on, and
    // `DREAM_DEBUG_INHERIT=1` passes them all.
    std::vector<char*> envp;
    auto name_of = [](const std::string& kv) { return kv.substr(0, kv.find('=')); };
    const bool inherit_debug = inherit_debug_variables();
    for (char** e = environ; *e; ++e) {
        std::string entry(*e);
        bool replaced = !inherit_debug && debug_variable(name_of(entry));
        for (const std::string& kv : job->env) {
            if (name_of(kv) == name_of(entry)) { replaced = true; break; }
        }
        if (!replaced) envp.push_back(*e);
    }
    for (std::string& kv : job->env) envp.push_back(kv.data());
    envp.push_back(nullptr);
    int rc = ::posix_spawnp(&child, args[0], &actions, nullptr, args.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    if (rc != 0) {
        job->error = std::string("cannot run `") + argv[0] + "`: " + std::strerror(rc);
        ::close(out_pipe[0]);
        ::close(err_pipe[0]);
        finish();
        return;
    }
    job->started = true;

    // A watchdog, when a deadline was asked for. It waits to be told the child
    // finished and kills it if that never comes; the reads below then see end
    // of file and `waitpid` reaps as usual, so the normal path needs no
    // special case for having been killed.
    std::thread watchdog;
    if (job->timeout_ms > 0) {
        watchdog = std::thread([job, child] {
            std::unique_lock<std::mutex> lk(job->finish_mutex);
            if (!job->finished.wait_for(lk, std::chrono::milliseconds(job->timeout_ms),
                                        [&] { return job->child_done; })) {
                job->timed_out = true;
                ::kill(child, SIGKILL);
            }
        });
    }

    // stderr on its own thread: a child that fills one pipe while nothing
    // drains the other would deadlock against a reader that takes them in
    // turn, and "it hung when the output got long" is a miserable bug.
    std::string err_text;
    std::thread err_reader([&] { err_text = slurp(err_pipe[0]); });
    job->out = slurp(out_pipe[0]);
    err_reader.join();
    job->err = std::move(err_text);

    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    job->code = WIFEXITED(status) ? WEXITSTATUS(status)
                                  : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1);

    if (watchdog.joinable()) {
        {
            std::lock_guard<std::mutex> lk(job->finish_mutex);
            job->child_done = true;
        }
        job->finished.notify_all();
        watchdog.join();
    }
    finish();
}

#endif

NativeResult os_exec_with(Process& p, Value* args, int64_t timeout_ms, Value dir,
                          std::vector<std::string> env = {}, bool joined = false);

NativeResult os_exec(Process& p, Value, Value* args, uint32_t) {
    return os_exec_with(p, args, 0, UNIT);
}

/// `exec_for! program args milliseconds` -- the same, with a deadline. A child
/// that outlives it is killed, and the result says `timed_out` so the caller
/// can tell that from an ordinary non-zero exit.
NativeResult os_exec_for(Process& p, Value, Value* args, uint32_t) {
    Value ms = resolve(args[2]);
    if (!is_fixnum(ms) || fixnum_value(ms) < 0) {
        return fail(p, "type_error", "exec_for! needs a timeout in milliseconds");
    }
    return os_exec_with(p, args, fixnum_value(ms), UNIT);
}

/// `exec_in! dir program args milliseconds` -- the same again, started in
/// `dir`, with 0 for no deadline. What a build step runs things with: it
/// owns a directory and must not have to share the VM's.
NativeResult os_exec_in(Process& p, Value, Value* args, uint32_t) {
    if (p.os_pending >= 0) return os_exec_with(p, args + 1, 0, UNIT);
    if (!is_string(args[0])) return fail(p, "type_error", "exec_in! needs a directory");
    Value ms = resolve(args[3]);
    if (!is_fixnum(ms) || fixnum_value(ms) < 0) {
        return fail(p, "type_error", "exec_in! needs a timeout in milliseconds, or 0 for none");
    }
    return os_exec_with(p, args + 1, fixnum_value(ms), args[0]);
}

NativeResult os_exec_env(Process& p, Value* args, bool joined);

/// `exec_with! dir env program args milliseconds` -- `exec_in!` with variables
/// of its own: `env` is a list of `NAME=value`, laid over this VM's
/// environment for the child alone. What a check runs a test suite with.
NativeResult os_exec_with_env(Process& p, Value, Value* args, uint32_t) {
    return os_exec_env(p, args, false);
}

/// `exec_joined! dir env program args milliseconds` -- `exec_with!` with the
/// child's errors written into its output, interleaved as they happened: `out`
/// is everything it printed and `err` is empty. A check that compares what a
/// program printed with a recorded file wants it as a terminal shows it, which
/// two streams read apart cannot give back.
NativeResult os_exec_joined(Process& p, Value, Value* args, uint32_t) {
    return os_exec_env(p, args, true);
}

NativeResult os_exec_env(Process& p, Value* args, bool joined) {
    if (p.os_pending >= 0) return os_exec_with(p, args + 2, 0, UNIT);
    if (!is_string(args[0])) return fail(p, "type_error", "exec_with! needs a directory");
    std::vector<std::string> env;
    if (!read_string_list(p, args[1], &env)) {
        if (p.park_requested) return NativeResult::block();
        return fail(p, "type_error", "exec_with! needs a list of NAME=value strings");
    }
    for (const auto& kv : env) {
        if (kv.find('=') == std::string::npos || kv.front() == '=' || kv.find('\0') != std::string::npos) {
            return fail(p, "bad_argument", "exec_with!: `" + kv + "` is not NAME=value");
        }
    }
    Value ms = resolve(args[4]);
    if (!is_fixnum(ms) || fixnum_value(ms) < 0) {
        return fail(p, "type_error", "exec_with! needs a timeout in milliseconds, or 0 for none");
    }
    return os_exec_with(p, args + 2, fixnum_value(ms), args[0], std::move(env), joined);
}

NativeResult os_exec_with(Process& p, Value* args, int64_t timeout_ms, Value dir,
                          std::vector<std::string> env, bool joined) {
    // Entered again after the helper thread woke us: the child has exited.
    if (p.os_pending >= 0) {
        int64_t id = p.os_pending;
        p.os_pending = -1;
        auto job = Jobs::get().find(id);
        if (!job) return fail(p, "os_error", "the subprocess result was lost");

        NativeResult result = [&]() -> NativeResult {
            if (!job->error.empty()) return fail(p, "not_found", job->error);
            Value m = p.heap().make_map(8);
            m = map_insert(p, m, make_atom(p.runtime().intern_atom("code")),
                       make_integer(p, int64_t(job->code)));
            m = resolve(m);
            m = map_insert(p, m, make_atom(p.runtime().intern_atom("out")),
                       text(p, job->out));
            m = resolve(m);
            m = map_insert(p, m, make_atom(p.runtime().intern_atom("err")),
                       text(p, job->err));
            m = resolve(m);
            m = map_insert(p, m, make_atom(p.runtime().intern_atom("timed_out")),
                       make_bool(job->timed_out));
            return NativeResult::ok(resolve(m));
        }();
        Jobs::get().erase(id);
        return result;
    }

    if (!is_string(args[0])) return fail(p, "type_error", "exec! needs a program name");
    std::vector<std::string> argv{string_arg(args[0])};
    if (!read_string_list(p, args[1], &argv)) {
        if (p.park_requested) return NativeResult::block();
        return fail(p, "type_error", "exec! needs a list of string arguments");
    }

    for (const auto& arg : argv) {
        if (arg.find('\0') != std::string::npos)
            return fail(p, "bad_argument", "subprocess arguments cannot contain NUL bytes");
#ifdef _WIN32
        try { (void)windows::wide(arg); }
        catch (const std::system_error& error) { return fail(p, "bad_argument", error.what()); }
#endif
    }
    if (argv[0].empty()) return fail(p, "bad_argument", "exec! needs a non-empty program name");

    Scheduler* sched = p.runtime().scheduler();
    if (!sched) return fail(p, "os_error", "no scheduler is running to wake this process");

    auto job = std::make_shared<Job>();
    job->timeout_ms = timeout_ms;
    job->env = std::move(env);
    job->joined = joined;
    if (dir != UNIT) {
        job->dir = string_arg(dir);
        if (job->dir.empty() || job->dir.find('\0') != std::string::npos) {
            return fail(p, "bad_argument", "exec_in! needs a directory");
        }
    }
    int64_t id = Jobs::get().add(job);
    p.os_pending = id;

    // Ask to be parked before the child can finish, so a wake-up that arrives
    // immediately is not lost: `wake` finds the process still running and
    // leaves a note the worker checks before it commits to parking.
    {
        std::lock_guard<std::mutex> g(p.sched_mutex);
        p.park_requested = true;
        p.wait_reason.store(WaitReason::Io, std::memory_order_relaxed);
    }
    sched->note_io_wait(true);
    job->worker = std::thread(run_child, job, std::move(argv), p.id(), sched->serial());
    return NativeResult::block();
}

// ---------------------------------------------------------------------------
// Everything else
// ---------------------------------------------------------------------------

NativeResult os_args(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(string_list(p, p.runtime().program_args()));
}

NativeResult os_env(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "env! needs a name");
#ifdef _WIN32
    const wchar_t* v = _wgetenv(windows::wide(string_arg(args[0])).c_str());
    return NativeResult::ok(v ? text(p, windows::utf8(v)) : UNIT);
#else
    const char* v = ::getenv(string_arg(args[0]).c_str());
    return NativeResult::ok(v ? text(p, v) : UNIT);
#endif
}

NativeResult os_set_env(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0]) || !is_string(args[1])) {
        return fail(p, "type_error", "set_env! needs a name and a value");
    }
#ifdef _WIN32
    _wputenv_s(windows::wide(string_arg(args[0])).c_str(), windows::wide(string_arg(args[1])).c_str());
#else
    ::setenv(string_arg(args[0]).c_str(), string_arg(args[1]).c_str(), 1);
#endif
    return NativeResult::ok(UNIT);
}

NativeResult os_cwd(Process& p, Value, Value*, uint32_t) {
    std::error_code ec;
    auto path = std::filesystem::current_path(ec);
    if (ec) return fail(p, "os_error", ec.message());
    auto bytes = path.generic_u8string();
    return NativeResult::ok(text(p, std::string(bytes.begin(), bytes.end())));
}

NativeResult os_chdir(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "chdir! needs a path");
    std::string path_bytes = string_arg(args[0]);
    std::error_code ec;
    std::filesystem::current_path(std::filesystem::path(std::u8string(path_bytes.begin(), path_bytes.end())), ec);
    if (ec) return fail(p, ec == std::errc::no_such_file_or_directory ? "not_found" : "os_error", ec.message());
    return NativeResult::ok(UNIT);
}

NativeResult os_list_dir(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "list_dir! needs a path");
    std::string path_bytes = string_arg(args[0]);
    std::error_code ec;
    std::filesystem::directory_iterator it(std::filesystem::path(std::u8string(path_bytes.begin(), path_bytes.end())), ec), end;
    std::vector<std::string> names;
    while (!ec && it != end) {
        auto bytes = it->path().filename().u8string();
        names.emplace_back(bytes.begin(), bytes.end());
        it.increment(ec);
    }
    if (ec) return fail(p, ec == std::errc::no_such_file_or_directory ? "not_found" : "os_error", ec.message());
    std::sort(names.begin(), names.end());
    return NativeResult::ok(string_list(p, names));
}

NativeResult os_pid(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(
#ifdef _WIN32
        ::_getpid()
#else
        ::getpid()
#endif
    )));
}

/// The machine's architecture, in the names the image header's target bits
/// use. What the VM was built for rather than what the kernel says, which is
/// the same thing for every question a program can ask of it: this is the
/// architecture the program's native code has to match.
NativeResult os_arch(Process& p, Value, Value*, uint32_t) {
#if defined(__x86_64__) || defined(_M_X64)
    const char* name = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    const char* name = "aarch64";
#else
    const char* name = "unknown";
#endif
    return NativeResult::ok(make_atom(p.runtime().intern_atom(name)));
}

NativeResult os_platform(Process& p, Value, Value*, uint32_t) {
#if defined(__ANDROID__)
    const char* name = "android";
#elif defined(__linux__)
    const char* name = "linux";
#elif defined(__APPLE__)
    const char* name = "macos";
#elif defined(_WIN32)
    const char* name = "windows";
#else
    const char* name = "unknown";
#endif
    return NativeResult::ok(make_atom(p.runtime().intern_atom(name)));
}

/// Stop the whole program. Every other process is abandoned where it stands,
/// which is why this is worth reaching for only from the top of a program.
/// `monotonic!` -- milliseconds from some fixed point in this process's life.
///
/// A steady clock, so it never jumps: the number it answers only ever grows,
/// whatever happens to the wall clock. That is what makes a *difference*
/// between two readings a duration, which is the only thing this is for --
/// where the zero is has no meaning at all.
///
/// Timing anything in a lazy language means forcing it first. A phase that has
/// not been forced has not run, so `let t = monotonic! ()` around an unforced
/// value times the building of a thunk.
NativeResult os_monotonic(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return NativeResult::ok(
        make_integer(p, int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(now).count())));
}

/// `now!` -- milliseconds since the Unix epoch, from the wall clock.
///
/// This one can jump backwards, because the wall clock can: it is what to
/// stamp a file or a log line with, and never what to measure a duration with.
NativeResult os_now(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return NativeResult::ok(
        make_integer(p, int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(now).count())));
}

#ifdef DREAM_PROFILE_GENERATE
extern "C" void __gcov_dump();
#endif

/// End the process now, running nothing on the way out -- POSIX `_Exit`.
///
/// On Windows `_Exit` is not that. It is `ExitProcess`, which kills every other
/// thread and then runs each DLL's static destructors, the VM's among them,
/// against a heap and a collector whose threads vanished mid-step. So Windows
/// ends with `TerminateProcess` on itself, which is what `_Exit` means
/// everywhere else. Callers flush first; nothing after this runs.
[[noreturn]] static void exit_now(int code) {
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
    std::_Exit(code);
}

// Its training-only flush changes this cold function's control-flow graph;
// exclude it from GCC's profile rather than applying mismatched counters.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((no_profile_instrument_function))
#endif
NativeResult os_exit(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return fail(p, "type_error", "exit! needs an exit code");
    // Nothing runs after this, so anything the run owes its caller has to be
    // said here. A measurement that only printed on the way out of `main`
    // would never print for a program that ends by exiting, which is most
    // tools -- `dreams` among them, so `--stats` on a compile said nothing at
    // all until this line existed.
    p.runtime().print_profile();
    p.runtime().print_stats();
    // `_Exit` runs no destructors, and a C library's are often where it
    // writes out what it was holding. The exiting process's are the ones it
    // can run safely: every other process is still running on its own worker.
    release_foreign(p);
#ifdef DREAM_PROFILE_GENERATE
    // _Exit bypasses GCC's normal profile writer. Compiler training runs
    // finish through this native, so flush explicitly in instrumented builds.
    __gcov_dump();
#endif
    std::fflush(nullptr);
    exit_now(int(fixnum_value(v)));
}

/// `replace! program args` -- become another program.
///
/// This is `execvp`: the image, the heap and every thread of this VM are gone,
/// and the program named here takes over the process. Nothing comes back, so
/// there is no result type and no `!`-suffixed thing to do with one -- the only
/// way this returns is by failing.
///
/// `exec!` is the wrong tool for handing over to something *interactive*: it
/// gives the child pipes and reads them to the end, so a program that prompts
/// has nobody typing at it. A tool that runs an editor, a shell or a REPL wants
/// this instead -- the child inherits the terminal, because it inherits
/// everything.
NativeResult os_replace(Process& p, Value, Value* args, uint32_t) {
    if (!is_string(args[0])) return fail(p, "type_error", "replace! needs a program name");
    std::vector<std::string> argv{string_arg(args[0])};
    if (!read_string_list(p, args[1], &argv)) {
        if (p.park_requested) return NativeResult::block();
        return fail(p, "type_error", "replace! needs a list of string arguments");
    }

    std::vector<char*> raw;
    raw.reserve(argv.size() + 1);
    for (std::string& a : argv) raw.push_back(a.data());
    raw.push_back(nullptr);

    // Anything still sitting in a stdio buffer would be lost with the address
    // space, so it goes out first.
    std::fflush(nullptr);
#ifdef _WIN32
    std::wstring command;
    for (const auto& arg : argv) {
        if (!command.empty()) command += L' ';
        command += windows::quote(arg);
    }
    STARTUPINFOW start{};
    start.cb = sizeof(start);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0,
                        nullptr, nullptr, &start, &child))
        return fail(p, "not_found", windows::error());
    CloseHandle(child.hThread);
    WaitForSingleObject(child.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(child.hProcess, &code);
    CloseHandle(child.hProcess);
    std::fflush(nullptr);
    exit_now(int(code));
#else
    ::execvp(raw[0], raw.data());
#endif
    return fail(p, "not_found",
                std::string("cannot run `") + argv[0] + "`: " + std::strerror(errno));
}

/// `install_dirs! ()`: every directory the installation keeps things in, as
/// the VM itself would search them for `dream NAME` -- `$MINDV2_PATH` split,
/// or `~/.mindv2`, or nothing. See "the installation" in os.hpp.
NativeResult os_install_dirs(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(string_list(p, install_dirs()));
}

}  // namespace

void os_shutdown() { Jobs::get().drain(); }

std::string home_dir() {
#if defined(_WIN32)
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    // `getenv` answers null for a variable that is not set, and constructing a
    // `std::string` from null is undefined rather than empty.
    return home ? std::string(home) : std::string();
}

/// The shell expands a tilde it can see, and `export MINDV2_PATH="~/.mindv2"`
/// hides it inside quotes, so what arrives is a literal `~`. Nothing on disk is
/// called that, so every lookup under it silently found nothing, which reads as
/// "the image is not installed" when it is sitting right there. A path is not
/// text to this program, so expanding it is this program's job.
std::string expand_home(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    if (path.size() > 1 && path[1] != '/' && path[1] != '\\') return path;  // `~other`, a user
    const std::string home = home_dir();
    if (home.empty()) return path;
    return home + path.substr(1);
}

/// A list rather than one directory: the toolchain wrapper sets
/// `$MINDV2_PATH` to what an installation ships in whatever store paths those
/// live in, so the compiler image and the standard library can sit apart. A
/// plain `~/.mindv2` is a list of one. Each element gets the `~` treatment
/// here too, because a tilde inside a joined list is hidden from the shell.
std::vector<std::string> split_dir_list(const std::string& list) {
    std::vector<std::string> dirs;
    const char sep =
#if defined(_WIN32)
        ';';
#else
        ':';
#endif
    std::size_t start = 0;
    while (start <= list.size()) {
        std::size_t end = list.find(sep, start);
        if (end == std::string::npos) end = list.size();
        if (end > start) dirs.push_back(expand_home(list.substr(start, end - start)));
        start = end + 1;
    }
    return dirs;
}

/// Where the running binary is, with every link on the way resolved, or an
/// empty path when the system will not say. `/usr/bin/dream` may be a link
/// into `/usr/lib/dream/bin`, and it is the file's own place that says which
/// installation it came with, not the name it was run by.
std::filesystem::path own_executable() {
    std::error_code ec;
    std::filesystem::path exe;
#if defined(_WIN32)
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    exe = std::filesystem::path(buf);
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    exe = std::filesystem::path(buf.c_str());
#else
    exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
#endif
    std::filesystem::path real = std::filesystem::canonical(exe, ec);
    return ec ? exe : real;
}

/// The installation this VM was installed with: `lib/dream` beside the
/// directory the binary is in, so `/usr/bin/dream` has `/usr/lib/dream` and a
/// Homebrew keg or a tarball unpacked into `~/.local` has its own. That is
/// the layout `ship` packages Dream in (the `[ship]` sections of the
/// repository's mind.toml). A VM in a build tree has no such directory and
/// answers `""`.
std::string own_installation() {
    const std::filesystem::path exe = own_executable();
    if (exe.empty()) return "";
    const std::filesystem::path dir = exe.parent_path().parent_path() / "lib" / "dream";
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return "";
    const std::u8string text = dir.u8string();
    return std::string(text.begin(), text.end());
}

std::string install_path() {
    if (const char* from_env = std::getenv("MINDV2_PATH")) return std::string(from_env);
    // Before `~/.mindv2`: a packaged VM is one with its standard library and
    // compiler beside it, and a `~/.mindv2` left from building by hand is
    // older or newer than both -- the mismatch `just install-artifacts` warns
    // about, where the installed `std` cannot build the installed compiler.
    const std::string own = own_installation();
    if (!own.empty()) return own;
    const std::string home = home_dir();
    if (home.empty()) return "";
    std::string candidate = home + "/.mindv2";
    std::error_code ec;
    if (std::filesystem::is_directory(std::filesystem::path(std::u8string(candidate.begin(), candidate.end())), ec)) {
        return candidate;
    }
    return "";
}

std::vector<std::string> install_dirs() { return split_dir_list(install_path()); }

bool is_file(const std::string& path) {
    std::error_code ec;
    auto st = std::filesystem::status(std::filesystem::path(std::u8string(path.begin(), path.end())), ec);
    return !ec && std::filesystem::exists(st) && !std::filesystem::is_directory(st);
}

std::string installed_image(const std::string& name, const std::vector<std::string>& dirs) {
    for (const std::string& dir : dirs) {
        for (const std::string& candidate : {dir + "/" + name, dir + "/" + name + ".dream"}) {
            if (is_file(candidate)) return candidate;
        }
    }
    return "";
}

ModuleDef make_os_module() {
    return ModuleDef{"std.os",
                     {
                         {"args!", 1, 0b1, os_args},
                         {"env!", 1, 0b1, os_env},
                         {"set_env!", 2, 0b11, os_set_env},
                         {"cwd!", 1, 0b1, os_cwd},
                         {"chdir!", 1, 0b1, os_chdir},
                         {"list_dir!", 1, 0b1, os_list_dir},
                         {"exec!", 2, 0b01, os_exec},
                         {"exec_for!", 3, 0b101, os_exec_for},
                         {"exec_in!", 4, 0b1011, os_exec_in},
                         {"exec_with!", 5, 0b10101, os_exec_with_env},
                         {"exec_joined!", 5, 0b10101, os_exec_joined},
                         {"replace!", 2, 0b01, os_replace},
                         {"monotonic!", 1, 0b1, os_monotonic},
                         {"now!", 1, 0b1, os_now},
                         {"pid!", 1, 0b1, os_pid},
                         {"platform", 1, 0b1, os_platform},
                         {"install_dirs!", 1, 0b1, os_install_dirs},
                         {"arch", 1, 0b1, os_arch},
                         {"exit!", 1, 0b1, os_exit},
                     }};
}

}  // namespace dream
