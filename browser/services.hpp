#pragma once
// The browser's versions of the modules every program reaches. services.cpp
// says what each member answers there.
#include "runtime.hpp"

namespace dream {
ModuleDef make_browser_io_module();
ModuleDef make_browser_os_module();
/// Whether the program called `os.exit!`, and with what.
bool browser_exit_requested(int* code);
void browser_reset_exit();
}
