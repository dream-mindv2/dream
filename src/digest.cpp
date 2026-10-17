#include "digest.hpp"

#include <new>

namespace dream {

namespace {

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
inline uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

inline uint32_t load_be32(const uint8_t* p) {
    return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]);
}
inline uint32_t load_le32(const uint8_t* p) {
    return uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | uint32_t(p[0]);
}
inline uint64_t load_be64(const uint8_t* p) {
    return uint64_t(load_be32(p)) << 32 | load_be32(p + 4);
}
inline void store_be32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v);
}
inline void store_le32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
inline void store_be64(uint8_t* p, uint64_t v) {
    store_be32(p, uint32_t(v >> 32));
    store_be32(p + 4, uint32_t(v));
}

}  // namespace

// --- MD5 (RFC 1321) -------------------------------------------------------------

namespace {

constexpr uint32_t kMd5Sines[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

constexpr int kMd5Shifts[16] = {7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};

}  // namespace

void Md5::compress(const uint8_t* p) {
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) w[i] = load_le32(p + 4 * i);
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    for (int i = 0; i < 64; ++i) {
        uint32_t f;
        int g;
        switch (i >> 4) {
            case 0: f = (b & c) | (~b & d); g = i; break;
            case 1: f = (d & b) | (~d & c); g = (5 * i + 1) & 15; break;
            case 2: f = b ^ c ^ d; g = (3 * i + 5) & 15; break;
            default: f = c ^ (b | ~d); g = (7 * i) & 15; break;
        }
        uint32_t next = b + rotl32(a + f + kMd5Sines[i] + w[g], kMd5Shifts[(i >> 4) * 4 + (i & 3)]);
        a = d; d = c; c = b; b = next;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
}

void Md5::write_state(uint8_t* out) const {
    for (int i = 0; i < 4; ++i) store_le32(out + 4 * i, h_[i]);
}

// --- SHA-1 (FIPS 180-4) ------------------------------------------------------------

void Sha1::compress(const uint8_t* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = load_be32(p + 4 * i);
    for (int i = 16; i < 80; ++i) w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
        else { f = b ^ c ^ d; k = 0xca62c1d6; }
        uint32_t t = rotl32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl32(b, 30); b = a; a = t;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e;
}

void Sha1::write_state(uint8_t* out) const {
    for (int i = 0; i < 5; ++i) store_be32(out + 4 * i, h_[i]);
}

// --- SHA-256 (FIPS 180-4) ----------------------------------------------------------

namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

}  // namespace

void Sha256::compress(const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = load_be32(p + 4 * i);
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) + ((e & f) ^ (~e & g)) +
                      kSha256K[i] + w[i];
        uint32_t t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::write_state(uint8_t* out) const {
    for (int i = 0; i < 8; ++i) store_be32(out + 4 * i, h_[i]);
}

std::string Sha256::hex() {
    uint8_t d[32];
    finish(d);
    static const char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; ++i) {
        out[2 * i] = digits[d[i] >> 4];
        out[2 * i + 1] = digits[d[i] & 15];
    }
    return out;
}

// --- SHA-512 and SHA-384 (FIPS 180-4) ------------------------------------------------

namespace {

constexpr uint64_t kSha512K[80] = {
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
};

constexpr uint64_t kSha512Init[8] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

constexpr uint64_t kSha384Init[8] = {
    0xcbbb9d5dc1059ed8, 0x629a292a367cd507, 0x9159015a3070dd17, 0x152fecd8f70e5939,
    0x67332667ffc00b31, 0x8eb44a8768581511, 0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4,
};

}  // namespace

Sha512::Sha512(bool truncated_384) : words_(truncated_384 ? 6 : 8) {
    std::memcpy(h_, truncated_384 ? kSha384Init : kSha512Init, sizeof h_);
}

void Sha512::compress(const uint8_t* p) {
    uint64_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = load_be64(p + 8 * i);
    for (int i = 16; i < 80; ++i) {
        uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint64_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 80; ++i) {
        uint64_t t1 = h + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) + ((e & f) ^ (~e & g)) +
                      kSha512K[i] + w[i];
        uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha512::write_state(uint8_t* out) const {
    for (int i = 0; i < words_; ++i) store_be64(out + 8 * i, h_[i]);
}

// --- one hash, chosen at run time ------------------------------------------------------
//
// A union of the five rather than a virtual base: every one of them is a plain
// value with no destructor, so copying a `Hasher` is copying bytes, and HMAC
// copies one per message.

Hasher::Hasher(HashAlg alg) : alg_(alg) {
    switch (alg) {
        case HashAlg::Md5: new (&md5_) Md5(); break;
        case HashAlg::Sha1: new (&sha1_) Sha1(); break;
        case HashAlg::Sha256: new (&sha256_) Sha256(); break;
        case HashAlg::Sha384: new (&sha512_) Sha512(true); break;
        case HashAlg::Sha512: new (&sha512_) Sha512(false); break;
    }
}

void Hasher::update(const void* data, size_t len) {
    switch (alg_) {
        case HashAlg::Md5: md5_.update(data, len); break;
        case HashAlg::Sha1: sha1_.update(data, len); break;
        case HashAlg::Sha256: sha256_.update(data, len); break;
        case HashAlg::Sha384:
        case HashAlg::Sha512: sha512_.update(data, len); break;
    }
}

void Hasher::finish(uint8_t* out) {
    switch (alg_) {
        case HashAlg::Md5: md5_.finish(out); break;
        case HashAlg::Sha1: sha1_.finish(out); break;
        case HashAlg::Sha256: sha256_.finish(out); break;
        case HashAlg::Sha384:
        case HashAlg::Sha512: sha512_.finish(out); break;
    }
}

size_t Hasher::digest_size() const {
    switch (alg_) {
        case HashAlg::Md5: return Md5::digest_size;
        case HashAlg::Sha1: return Sha1::digest_size;
        case HashAlg::Sha256: return Sha256::digest_size;
        case HashAlg::Sha384: return 48;
        case HashAlg::Sha512: return 64;
    }
    return 0;
}

size_t Hasher::block_size() const {
    return alg_ == HashAlg::Sha384 || alg_ == HashAlg::Sha512 ? 128 : 64;
}

// --- HMAC (RFC 2104) -------------------------------------------------------------------

Hmac::Hmac(HashAlg alg, const uint8_t* key, size_t key_len) : inner_(alg), outer_(alg) {
    const size_t block = inner_.block_size();
    uint8_t k[128] = {0};
    if (key_len > block) {
        Hasher h(alg);
        h.update(key, key_len);
        h.finish(k);
    } else if (key_len) {
        std::memcpy(k, key, key_len);
    }
    uint8_t pad[128];
    for (size_t i = 0; i < block; ++i) pad[i] = uint8_t(k[i] ^ 0x36);
    inner_.update(pad, block);
    for (size_t i = 0; i < block; ++i) pad[i] = uint8_t(k[i] ^ 0x5c);
    outer_.update(pad, block);
}

void Hmac::mac(const uint8_t* msg, size_t len, uint8_t* out) const {
    uint8_t inner_digest[kMaxDigest];
    Hasher in = inner_;
    in.update(msg, len);
    in.finish(inner_digest);
    Hasher outer = outer_;
    outer.update(inner_digest, digest_size());
    outer.finish(out);
}

}  // namespace dream
