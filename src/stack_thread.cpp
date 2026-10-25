// The platform's part of StackThread (stack_thread.hpp): a thread started
// with a stack of the size asked for, by pthreads or by the Windows API.
#include "stack_thread.hpp"

#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include <process.h>

#include "windows.hpp"
#else
#include <pthread.h>
#endif

namespace dream {
namespace {

#if defined(_WIN32)
unsigned __stdcall trampoline(void* arg) {
#else
void* trampoline(void* arg) {
#endif
    std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(arg));
    (*fn)();
    return 0;
}

}  // namespace

void StackThread::start(size_t stack_bytes, std::function<void()> f) {
    auto* fn = new std::function<void()>(std::move(f));
#if defined(_WIN32)
    uintptr_t h = _beginthreadex(nullptr, unsigned(stack_bytes), &trampoline, fn,
                                 STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!h) h = _beginthreadex(nullptr, 0, &trampoline, fn, 0, nullptr);
    if (!h) {
        delete fn;
        throw std::runtime_error("cannot start a thread");
    }
    handle_ = reinterpret_cast<void*>(h);
#else
    auto* thread = new pthread_t();
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int rc = pthread_attr_setstacksize(&attr, stack_bytes);
    if (rc == 0) rc = pthread_create(thread, &attr, &trampoline, fn);
    pthread_attr_destroy(&attr);
    if (rc != 0) rc = pthread_create(thread, nullptr, &trampoline, fn);
    if (rc != 0) {
        delete fn;
        delete thread;
        throw std::runtime_error("cannot start a thread");
    }
    handle_ = thread;
#endif
    started_ = true;
}

void StackThread::join() {
    if (!started_) return;
#if defined(_WIN32)
    WaitForSingleObject(static_cast<HANDLE>(handle_), INFINITE);
    CloseHandle(static_cast<HANDLE>(handle_));
#else
    auto* thread = static_cast<pthread_t*>(handle_);
    pthread_join(*thread, nullptr);
    delete thread;
#endif
    handle_ = nullptr;
    started_ = false;
}

}  // namespace dream
