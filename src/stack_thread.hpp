#pragma once

// A thread whose machine stack is a size the VM chooses.
//
// `std::thread` takes whatever the platform gives a new thread, and the
// platforms do not agree: glibc gives `ulimit -s`, usually 8 MB; Android's
// Bionic and Windows give 1 MB. A Dream process forces suspensions by nesting
// machine loops on that stack -- a deep `lazy.settled`, a long chain of
// suspensions -- so how deep a program may go was decided by the libc it ran
// on. The compiler compiling itself needs about 2 MB; on a phone, under
// Termux, it segfaulted near the end of the build. The scheduler's workers,
// which run every Dream process, are started with `kWorkerStack` instead,
// whatever the platform. A stack is address space reserved and committed only
// as it is touched, so the size costs nothing a shallow program does not use.

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <process.h>

#include "windows.hpp"
#else
#include <pthread.h>
#endif

namespace dream {

/// The machine stack every scheduler worker runs Dream on.
constexpr size_t kWorkerStack = size_t(16) << 20;

class StackThread {
public:
    StackThread() = default;
    StackThread(const StackThread&) = delete;
    StackThread& operator=(const StackThread&) = delete;
    StackThread(StackThread&& o) noexcept { *this = std::move(o); }
    StackThread& operator=(StackThread&& o) noexcept {
        std::swap(started_, o.started_);
#if defined(_WIN32)
        std::swap(handle_, o.handle_);
#else
        std::swap(thread_, o.thread_);
#endif
        return *this;
    }
    ~StackThread() = default;

    /// Run `f` on a new thread with a stack of `stack_bytes`. Should the
    /// system refuse that size, the thread gets its default stack instead:
    /// a smaller stack is a limit, not a reason not to run.
    void start(size_t stack_bytes, std::function<void()> f) {
        auto* fn = new std::function<void()>(std::move(f));
#if defined(_WIN32)
        uintptr_t h = _beginthreadex(nullptr, unsigned(stack_bytes), &StackThread::trampoline, fn,
                                     STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (!h) h = _beginthreadex(nullptr, 0, &StackThread::trampoline, fn, 0, nullptr);
        if (!h) {
            delete fn;
            throw std::runtime_error("cannot start a thread");
        }
        handle_ = reinterpret_cast<HANDLE>(h);
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        int rc = pthread_attr_setstacksize(&attr, stack_bytes);
        if (rc == 0) rc = pthread_create(&thread_, &attr, &StackThread::trampoline, fn);
        pthread_attr_destroy(&attr);
        if (rc != 0) rc = pthread_create(&thread_, nullptr, &StackThread::trampoline, fn);
        if (rc != 0) {
            delete fn;
            throw std::runtime_error("cannot start a thread");
        }
#endif
        started_ = true;
    }

    bool joinable() const { return started_; }

    void join() {
        if (!started_) return;
#if defined(_WIN32)
        WaitForSingleObject(handle_, INFINITE);
        CloseHandle(handle_);
#else
        pthread_join(thread_, nullptr);
#endif
        started_ = false;
    }

private:
#if defined(_WIN32)
    static unsigned __stdcall trampoline(void* arg) {
#else
    static void* trampoline(void* arg) {
#endif
        std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(arg));
        (*fn)();
        return 0;
    }

    bool started_ = false;
#if defined(_WIN32)
    HANDLE handle_ = nullptr;
#else
    pthread_t thread_{};
#endif
};

}  // namespace dream
