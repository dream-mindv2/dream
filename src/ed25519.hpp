// Ed25519 signatures (RFC 8032), written out.
//
// Written out for the reason the hashes in digest.hpp are: the TLS library is
// OpenSSL on two platforms and SChannel on the third, and Windows' CNG has no
// Ed25519, so a signature taken from libcrypto would be one Windows could not
// check. What a package manager signs has to verify the same everywhere.
//
// The arithmetic is TweetNaCl's -- sixteen 16-bit limbs in 64-bit words, a
// ladder with no branch or index that depends on a secret -- because it is
// small enough to read whole and its timing does not depend on the key. It is
// slow next to a tuned implementation, a few hundred microseconds a
// signature, which is nothing beside the fetch or the file being signed.
//
// Verification is stricter than TweetNaCl's, as RFC 8032 asks: an `S` that is
// not reduced below the group order, a public key whose `y` is not below the
// field prime, and a key encoding `x = 0` with the sign bit set are all
// refused. Without the first, anyone holding a signature could make a second,
// different one for the same message, which matters to anything that keys on
// a signature's bytes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace dream {

inline constexpr size_t kEd25519Seed = 32;
inline constexpr size_t kEd25519Public = 32;
inline constexpr size_t kEd25519Signature = 64;

/// The public key of a 32-byte secret seed.
void ed25519_public(uint8_t out[kEd25519Public], const uint8_t seed[kEd25519Seed]);

/// The signature of `message` under the key `seed` makes. Deterministic: the
/// same key and message always sign to the same bytes, so signing needs no
/// randomness and cannot leak the key through a bad source of it.
void ed25519_sign(uint8_t out[kEd25519Signature], const uint8_t seed[kEd25519Seed],
                  const uint8_t* message, size_t len);

/// Whether `signature` is `public_key`'s signature of `message`.
bool ed25519_verify(const uint8_t signature[kEd25519Signature], const uint8_t public_key[kEd25519Public],
                    const uint8_t* message, size_t len);

}  // namespace dream
