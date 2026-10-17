// `std.regex`'s matcher, as a native of `std.vm`; regex.cpp says why.
#pragma once

#include "runtime.hpp"

namespace dream {

/// `vm.regex_run code subject from mode`: run a program `std.regex` compiled.
NativeResult vm_regex_run(Process& p, Value callee, Value* args, uint32_t argc);

}  // namespace dream
