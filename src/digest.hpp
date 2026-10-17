// The hash functions `std.crypto` offers, written out.
//
// Written out rather than taken from a library. OpenSSL is the TLS backend on
// two platforms of three, and on Windows it is SChannel, so a hash that came
// from libcrypto would be a hash Windows does not have. These are a few
// hundred lines, need nothing, and are the same bytes everywhere. SHA-256 was
// here first, for `io.digest`: the same digest pins a fetched source
// (`std.build.fetch`), where the number has to be the one the source's
// publisher printed -- which is always this one.
//
// Every hash is the same machine -- a block buffer, a running length, a
// compression function and a padding rule -- and `BlockHash` is that machine
// once. A hash supplies its block size, its state and `compress`; the padding
// differs only in how wide the length is and which end it is written from,
// which are two constants. `Hasher` is the same thing again behind a switch,
// for code (HMAC, PBKDF2, HKDF) that is handed the algorithm at run time and
// should be written once rather than once per hash.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <string>

namespace dream {

/// The buffering and padding every Merkle-Damgard hash shares. `D` provides
/// `compress(const uint8_t*)` and `write_state(uint8_t*)`.
template <class D, size_t Block, size_t LengthBytes, bool BigEndian>
class BlockHash {
public:
    static constexpr size_t block_size = Block;

    void update(const void* data, size_t len) {
        auto* p = static_cast<const uint8_t*>(data);
        total_ += len;
        if (used_ > 0) {
            size_t take = len < Block - used_ ? len : Block - used_;
            std::memcpy(buf_ + used_, p, take);
            used_ += take;
            p += take;
            len -= take;
            if (used_ < Block) return;
            self().compress(buf_);
            used_ = 0;
        }
        for (; len >= Block; p += Block, len -= Block) self().compress(p);
        std::memcpy(buf_, p, len);
        used_ = len;
    }

    /// Pads, compresses the last block and writes the digest. Spent afterwards.
    void finish(uint8_t* out) {
        const uint64_t bits = total_ * 8;
        buf_[used_++] = 0x80;
        if (used_ > Block - LengthBytes) {
            std::memset(buf_ + used_, 0, Block - used_);
            self().compress(buf_);
            used_ = 0;
        }
        std::memset(buf_ + used_, 0, Block - used_);
        // A length wider than 64 bits is zero above them: no message here is
        // two exabytes long.
        for (size_t i = 0; i < 8; ++i) {
            uint8_t b = uint8_t(bits >> (8 * i));
            if constexpr (BigEndian) buf_[Block - 1 - i] = b;
            else buf_[Block - LengthBytes + i] = b;
        }
        self().compress(buf_);
        self().write_state(out);
    }

private:
    D& self() { return static_cast<D&>(*this); }
    uint8_t buf_[Block];
    size_t used_ = 0;
    uint64_t total_ = 0;
};

class Md5 : public BlockHash<Md5, 64, 8, false> {
public:
    static constexpr size_t digest_size = 16;
    void compress(const uint8_t* p);
    void write_state(uint8_t* out) const;

private:
    uint32_t h_[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
};

class Sha256 : public BlockHash<Sha256, 64, 8, true> {
public:
    static constexpr size_t digest_size = 32;
    void compress(const uint8_t* p);
    void write_state(uint8_t* out) const;
    /// The digest as 64 lowercase hex digits, as `io.digest` spells it. Spent
    /// afterwards.
    std::string hex();

private:
    uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
};

class Sha1 : public BlockHash<Sha1, 64, 8, true> {
public:
    static constexpr size_t digest_size = 20;
    void compress(const uint8_t* p);
    void write_state(uint8_t* out) const;

private:
    uint32_t h_[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
};

/// SHA-512, and SHA-384 as the same compression from a different start,
/// written out to six words instead of eight.
class Sha512 : public BlockHash<Sha512, 128, 16, true> {
public:
    explicit Sha512(bool truncated_384 = false);
    size_t digest_size() const { return words_ * 8; }
    void compress(const uint8_t* p);
    void write_state(uint8_t* out) const;

private:
    uint64_t h_[8];
    int words_;
};

enum class HashAlg : uint8_t { Md5, Sha1, Sha256, Sha384, Sha512 };

/// One hash, chosen at run time. Copyable, which is what HMAC wants: the key's
/// two padded blocks are hashed once and the states copied for every message.
class Hasher {
public:
    explicit Hasher(HashAlg alg);
    void update(const void* data, size_t len);
    /// Writes `digest_size()` bytes. Spent afterwards.
    void finish(uint8_t* out);
    size_t digest_size() const;
    size_t block_size() const;

private:
    HashAlg alg_;
    union {
        Md5 md5_;
        Sha1 sha1_;
        Sha256 sha256_;
        Sha512 sha512_;
    };
};

/// The largest digest any `HashAlg` writes.
inline constexpr size_t kMaxDigest = 64;

/// HMAC (RFC 2104) with the key already folded into its two padded states, so
/// a loop that signs under one key thousands of times -- PBKDF2 -- pays for the
/// key once.
class Hmac {
public:
    Hmac(HashAlg alg, const uint8_t* key, size_t key_len);
    /// The MAC of `msg`, `digest_size()` bytes.
    void mac(const uint8_t* msg, size_t len, uint8_t* out) const;
    size_t digest_size() const { return inner_.digest_size(); }

private:
    Hasher inner_;
    Hasher outer_;
};

}  // namespace dream
