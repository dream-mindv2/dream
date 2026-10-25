#pragma once
#include <string>
#include <utility>
#include <vector>

namespace dream {
/// What this VM was built with, as `dream --version` lists it: each line a
/// part -- the JIT, TLS, the C interface, the GPU, windows -- and what it is.
std::vector<std::pair<std::string, std::string>> build_features();
}
