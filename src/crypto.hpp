// `std.crypto`: hashes, MACs, key derivation, signatures and randomness, as
// natives.
#pragma once

#include <cstddef>
#include <string>

#include "runtime.hpp"

namespace dream {

ModuleDef make_crypto_module();

/// Fill `size` bytes from the operating system's secure generator.
/// On failure, leave a diagnostic in `error` and return false.
bool secure_random_bytes(unsigned char* data, size_t size, std::string* error);

}  // namespace dream
