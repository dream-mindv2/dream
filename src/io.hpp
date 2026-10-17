// Input and output.
//
// The problem this file solves is that a Dream process is not an OS thread.
// Hundreds of thousands of processes share a handful of workers, so a native
// that calls a blocking `read(2)` does not block one process -- it blocks a
// worker, and every process queued behind it.
//
// So IO never blocks a worker. A native tries the operation without blocking;
// if the file descriptor is not ready it registers interest with a poller and
// returns `NativeResult::block()`, which parks the *process* through the same
// handshake `recv!` uses. A readiness poller (epoll on Linux) wakes the
// process when the descriptor is ready, and the native is called again. From
// Dream the call looks like an ordinary synchronous one.
//
// The data never crosses a heap boundary: the native reads straight into the
// calling process's heap. That is the reason IO is not a server process here.
// A server would have to deep-copy every buffer into the caller on the way
// back, which doubles the cost of every read and makes one process a
// bottleneck for all IO in the system.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime.hpp"

namespace dream {

/// What a descriptor is, which decides how it can be waited on.
enum class HandleKind : uint8_t {
    /// A regular file. `epoll` reports these as always ready, so waiting on
    /// one is meaningless; reads and writes are issued directly.
    File,
    /// A socket, pipe or terminal: pollable, and driven by the poller.
    Stream,
    /// A listening socket, waited on for an incoming connection.
    Listener,
};

/// Set up the poller and the standard streams. Called once, when the runtime
/// is created.
void io_init();
/// Stop the poller thread and close every handle the program left open.
void io_shutdown();

/// Whether this build can wait on a descriptor without blocking a worker.
/// False if initialization failed; reported by `io.async ()`.
bool io_async_available();

/// One open handle, as `std.vm` reports it.
struct IoHandleInfo {
    int64_t handle = -1;
    int fd = -1;
    HandleKind kind = HandleKind::File;
    int busy = 0;
    /// The process parked on this descriptor, or 0 if none is.
    uint64_t waiting_pid = 0;
};

/// Every handle currently open, and who is waiting on it.
std::vector<IoHandleInfo> io_snapshot();

/// Withdraw process `pid` from waiting on `fd`: it is being killed, and a
/// waiter left behind would be counted as owed a wake-up for ever.
void io_cancel_wait(int fd, uint64_t pid);

ModuleDef make_io_module();
ModuleDef make_net_module();
ModuleDef make_tls_module();

}  // namespace dream
