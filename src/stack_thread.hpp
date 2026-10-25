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

namespace dream {

/// The machine stack every scheduler worker runs Dream on.
constexpr size_t kWorkerStack = size_t(16) << 20;

/// A thread with a stack of a chosen size. The platform's part is in
/// stack_thread.cpp, so that no header pulls in <windows.h>: its macros
/// (`KEY_READ`, `min`, ..) break whatever source includes this after it.
class StackThread {
public:
    StackThread() = default;
    StackThread(const StackThread&) = delete;
    StackThread& operator=(const StackThread&) = delete;
    ~StackThread() = default;

    /// Run `f` on a new thread with a stack of `stack_bytes`. Should the
    /// system refuse that size, the thread gets its default stack instead:
    /// a smaller stack is a limit, not a reason not to run.
    void start(size_t stack_bytes, std::function<void()> f);
    bool joinable() const { return started_; }
    void join();

private:
    bool started_ = false;
    /// The pthread_t, or the Windows HANDLE, as the platform has it.
    void* handle_ = nullptr;
};

}  // namespace dream
