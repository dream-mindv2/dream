// The browser has no ambient native handles, subprocesses, or FFI modules.
// DOM and backend access is provided only by the explicit graphics bridge.
#include "io.hpp"
#include "os.hpp"
#include "cores.hpp"
namespace dream {
unsigned usable_cores() { return 1; }
void io_init() {}
void io_shutdown() {}
void os_shutdown() {}
bool io_async_available() { return false; }
std::vector<IoHandleInfo> io_snapshot() { return {}; }
void io_cancel_wait(int, uint64_t) {}
std::vector<std::string> install_dirs() { return {}; }
bool is_file(const std::string&) { return false; }
std::string installed_image(const std::string&, const std::vector<std::string>&) { return {}; }
void release_foreign(Process&) {}
}
