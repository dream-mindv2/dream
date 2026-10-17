#include "ed25519.hpp"

#include <cstring>
#include <initializer_list>
#include <utility>

#include "digest.hpp"

namespace dream {

namespace {

// --- the field, integers mod 2^255 - 19 -------------------------------------------
//
// An element is sixteen limbs of 16 bits, least significant first, held in
// 64-bit words so that a product's column sums fit before they are carried.
// Limbs go negative in between; `carry` brings them back into range, and only
// `pack` reduces to the one canonical representative. The shifts of negative
// limbs are arithmetic, which C++20 defines.

using Fe = int64_t[16];

constexpr Fe kZero = {};
constexpr Fe kOne = {1};
// -121665/121666, the curve's d; twice it; the base point's x and y; and
// sqrt(-1). Computed from their definitions, not copied.
constexpr Fe kD = {0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070,
                   0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203};
constexpr Fe kD2 = {0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0,
                    0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406};
constexpr Fe kBaseX = {0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c,
                       0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169};
constexpr Fe kBaseY = {0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
                       0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666};
constexpr Fe kSqrtM1 = {0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43,
                        0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83};

void fe_copy(Fe out, const Fe a) {
    for (int i = 0; i < 16; ++i) out[i] = a[i];
}

/// Each limb back to 16 bits, its excess carried up; what leaves the top limb
/// re-enters the bottom times 38, since 2^256 = 38 (mod p).
void fe_carry(Fe o) {
    for (int i = 0; i < 16; ++i) {
        o[i] += int64_t(1) << 16;
        int64_t c = o[i] >> 16;
        if (i < 15) o[i + 1] += c - 1;
        else o[0] += 38 * (c - 1);
        o[i] -= c << 16;
    }
}

/// Swap `p` and `q` when `b` is 1, by masks rather than a branch.
void fe_swap(Fe p, Fe q, int64_t b) {
    int64_t mask = ~(b - 1);
    for (int i = 0; i < 16; ++i) {
        int64_t t = mask & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

/// The canonical 32 bytes: fully carried, then p subtracted while that does
/// not go negative (twice is always enough).
void fe_pack(uint8_t out[32], const Fe n) {
    Fe t, m;
    fe_copy(t, n);
    fe_carry(t);
    fe_carry(t);
    fe_carry(t);
    for (int pass = 0; pass < 2; ++pass) {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; ++i) {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        int64_t borrow = (m[15] >> 16) & 1;
        m[14] &= 0xffff;
        fe_swap(t, m, 1 - borrow);
    }
    for (int i = 0; i < 16; ++i) {
        out[2 * i] = uint8_t(t[i] & 0xff);
        out[2 * i + 1] = uint8_t(t[i] >> 8);
    }
}

/// The top bit is a point encoding's sign of x, not part of y.
void fe_unpack(Fe o, const uint8_t in[32]) {
    for (int i = 0; i < 16; ++i) o[i] = in[2 * i] + (int64_t(in[2 * i + 1]) << 8);
    o[15] &= 0x7fff;
}

bool fe_equal(const Fe a, const Fe b) {
    uint8_t x[32], y[32];
    fe_pack(x, a);
    fe_pack(y, b);
    return std::memcmp(x, y, 32) == 0;
}

/// The low bit of the canonical value: the "sign" of x.
int fe_parity(const Fe a) {
    uint8_t d[32];
    fe_pack(d, a);
    return d[0] & 1;
}

void fe_add(Fe o, const Fe a, const Fe b) {
    for (int i = 0; i < 16; ++i) o[i] = a[i] + b[i];
}

void fe_sub(Fe o, const Fe a, const Fe b) {
    for (int i = 0; i < 16; ++i) o[i] = a[i] - b[i];
}

void fe_mul(Fe o, const Fe a, const Fe b) {
    int64_t t[31] = {};
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) t[i + j] += a[i] * b[j];
    for (int i = 0; i < 15; ++i) t[i] += 38 * t[i + 16];
    for (int i = 0; i < 16; ++i) o[i] = t[i];
    fe_carry(o);
    fe_carry(o);
}

void fe_square(Fe o, const Fe a) { fe_mul(o, a, a); }

/// a^(p-2), which is 1/a.
void fe_invert(Fe o, const Fe a) {
    Fe c;
    fe_copy(c, a);
    for (int i = 253; i >= 0; --i) {
        fe_square(c, c);
        if (i != 2 && i != 4) fe_mul(c, c, a);
    }
    fe_copy(o, c);
}

/// a^((p-5)/8), the step a square root mod p is built from.
void fe_pow2523(Fe o, const Fe a) {
    Fe c;
    fe_copy(c, a);
    for (int i = 250; i >= 0; --i) {
        fe_square(c, c);
        if (i != 1) fe_mul(c, c, a);
    }
    fe_copy(o, c);
}

// --- the curve, in extended coordinates (X : Y : Z : T), x = X/Z, y = Y/Z, T = XY/Z

struct Point {
    Fe x, y, z, t;
};

/// p += q. The unified formula, so doubling is the same call and the ladder
/// below never branches on which it is doing.
void point_add(Point& p, const Point& q) {
    Fe a, b, c, d, t, e, f, g, h;
    fe_sub(a, p.y, p.x);
    fe_sub(t, q.y, q.x);
    fe_mul(a, a, t);
    fe_add(b, p.x, p.y);
    fe_add(t, q.x, q.y);
    fe_mul(b, b, t);
    fe_mul(c, p.t, q.t);
    fe_mul(c, c, kD2);
    fe_mul(d, p.z, q.z);
    fe_add(d, d, d);
    fe_sub(e, b, a);
    fe_sub(f, d, c);
    fe_add(g, d, c);
    fe_add(h, b, a);
    fe_mul(p.x, e, f);
    fe_mul(p.y, h, g);
    fe_mul(p.z, g, f);
    fe_mul(p.t, e, h);
}

void point_swap(Point& p, Point& q, int64_t b) {
    fe_swap(p.x, q.x, b);
    fe_swap(p.y, q.y, b);
    fe_swap(p.z, q.z, b);
    fe_swap(p.t, q.t, b);
}

/// y, with the sign of x in the top bit.
void point_pack(uint8_t out[32], const Point& p) {
    Fe zi, x, y;
    fe_invert(zi, p.z);
    fe_mul(x, p.x, zi);
    fe_mul(y, p.y, zi);
    fe_pack(out, y);
    out[31] ^= uint8_t(fe_parity(x) << 7);
}

/// p = s * q, a Montgomery ladder over all 256 bits whatever `s` is: the same
/// additions in the same order for every scalar.
void scalar_mult(Point& p, Point q, const uint8_t s[32]) {
    fe_copy(p.x, kZero);
    fe_copy(p.y, kOne);
    fe_copy(p.z, kOne);
    fe_copy(p.t, kZero);
    for (int i = 255; i >= 0; --i) {
        int64_t bit = (s[i / 8] >> (i & 7)) & 1;
        point_swap(p, q, bit);
        point_add(q, p);
        point_add(p, p);
        point_swap(p, q, bit);
    }
}

void scalar_base(Point& p, const uint8_t s[32]) {
    Point base;
    fe_copy(base.x, kBaseX);
    fe_copy(base.y, kBaseY);
    fe_copy(base.z, kOne);
    fe_mul(base.t, kBaseX, kBaseY);
    scalar_mult(p, base, s);
}

/// The point a public key encodes, negated -- verification wants -A -- or
/// false for 32 bytes that encode no point, or encode one non-canonically.
bool point_unpack_negated(Point& r, const uint8_t in[32]) {
    // y must be below p: the top byte's low seven bits all set, every byte
    // between all set, and the lowest at least 0xed is p or more.
    bool high = (in[31] & 0x7f) == 0x7f && in[0] >= 0xed;
    for (int i = 1; high && i < 31; ++i) high = in[i] == 0xff;
    if (high) return false;

    Fe num, den, den2, den4, den6, t, check;
    fe_copy(r.z, kOne);
    fe_unpack(r.y, in);
    // x^2 = (y^2 - 1) / (d y^2 + 1), and its root computed as RFC 8032 does.
    fe_square(num, r.y);
    fe_mul(den, num, kD);
    fe_sub(num, num, r.z);
    fe_add(den, r.z, den);
    fe_square(den2, den);
    fe_square(den4, den2);
    fe_mul(den6, den4, den2);
    fe_mul(t, den6, num);
    fe_mul(t, t, den);
    fe_pow2523(t, t);
    fe_mul(t, t, num);
    fe_mul(t, t, den);
    fe_mul(t, t, den);
    fe_mul(r.x, t, den);
    fe_square(check, r.x);
    fe_mul(check, check, den);
    if (!fe_equal(check, num)) fe_mul(r.x, r.x, kSqrtM1);
    fe_square(check, r.x);
    fe_mul(check, check, den);
    if (!fe_equal(check, num)) return false;
    // x = 0 has no negative, so a sign bit on it is a second spelling.
    if (fe_equal(r.x, kZero) && (in[31] >> 7)) return false;
    if (fe_parity(r.x) == (in[31] >> 7)) fe_sub(r.x, kZero, r.x);
    fe_mul(r.t, r.x, r.y);
    return true;
}

// --- scalars, integers mod the group order L ---------------------------------------

constexpr uint8_t kOrder[32] = {0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7,
                                0xa2, 0xde, 0xf9, 0xde, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10};

/// out = x mod L, for x up to 64 little-endian byte-sized limbs. `x` is
/// spent.
void mod_order(uint8_t out[32], int64_t x[64]) {
    int64_t carry;
    for (int i = 63; i >= 32; --i) {
        carry = 0;
        int j;
        for (j = i - 32; j < i - 12; ++j) {
            x[j] += carry - 16 * x[i] * kOrder[j - (i - 32)];
            carry = (x[j] + 128) >> 8;
            x[j] -= carry * 256;
        }
        x[j] += carry;
        x[i] = 0;
    }
    carry = 0;
    for (int j = 0; j < 32; ++j) {
        x[j] += carry - (x[31] >> 4) * kOrder[j];
        carry = x[j] >> 8;
        x[j] &= 255;
    }
    for (int j = 0; j < 32; ++j) x[j] -= carry * kOrder[j];
    for (int i = 0; i < 32; ++i) {
        x[i + 1] += x[i] >> 8;
        out[i] = uint8_t(x[i] & 255);
    }
}

/// A 64-byte hash, read as a little-endian integer, mod L.
void reduce_hash(uint8_t out[32], const uint8_t h[64]) {
    int64_t x[64];
    for (int i = 0; i < 64; ++i) x[i] = h[i];
    mod_order(out, x);
}

/// Whether a signature's S is below L: the check that makes a signature the
/// only one, since S + L would otherwise verify as well.
bool below_order(const uint8_t s[32]) {
    for (int i = 31; i >= 0; --i) {
        if (s[i] < kOrder[i]) return true;
        if (s[i] > kOrder[i]) return false;
    }
    return false;
}

void sha512(uint8_t out[64], std::initializer_list<std::pair<const uint8_t*, size_t>> parts) {
    Hasher h(HashAlg::Sha512);
    for (auto [data, len] : parts) h.update(data, len);
    h.finish(out);
}

/// The secret scalar and the nonce prefix a seed expands to.
void expand(uint8_t scalar[32], uint8_t prefix[32], const uint8_t seed[32]) {
    uint8_t d[64];
    sha512(d, {{seed, kEd25519Seed}});
    d[0] &= 248;
    d[31] &= 127;
    d[31] |= 64;
    std::memcpy(scalar, d, 32);
    std::memcpy(prefix, d + 32, 32);
}

}  // namespace

void ed25519_public(uint8_t out[kEd25519Public], const uint8_t seed[kEd25519Seed]) {
    uint8_t a[32], prefix[32];
    expand(a, prefix, seed);
    Point p;
    scalar_base(p, a);
    point_pack(out, p);
}

void ed25519_sign(uint8_t out[kEd25519Signature], const uint8_t seed[kEd25519Seed],
                  const uint8_t* message, size_t len) {
    uint8_t a[32], prefix[32], pub[32], h[64], r[32], k[32];
    expand(a, prefix, seed);
    Point p;
    scalar_base(p, a);
    point_pack(pub, p);

    // r = H(prefix || M), R = rB.
    sha512(h, {{prefix, 32}, {message, len}});
    reduce_hash(r, h);
    scalar_base(p, r);
    point_pack(out, p);

    // S = r + H(R || A || M) a  (mod L).
    sha512(h, {{out, 32}, {pub, 32}, {message, len}});
    reduce_hash(k, h);
    int64_t x[64] = {};
    for (int i = 0; i < 32; ++i) x[i] = r[i];
    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 32; ++j) x[i + j] += int64_t(k[i]) * a[j];
    mod_order(out + 32, x);
}

bool ed25519_verify(const uint8_t signature[kEd25519Signature], const uint8_t public_key[kEd25519Public],
                    const uint8_t* message, size_t len) {
    if (!below_order(signature + 32)) return false;
    Point minus_a;
    if (!point_unpack_negated(minus_a, public_key)) return false;

    // Accept when SB - H(R || A || M) A encodes to R.
    uint8_t h[64], k[32], got[32];
    sha512(h, {{signature, 32}, {public_key, 32}, {message, len}});
    reduce_hash(k, h);
    Point p, sb;
    scalar_mult(p, minus_a, k);
    scalar_base(sb, signature + 32);
    point_add(p, sb);
    point_pack(got, p);
    return std::memcmp(got, signature, 32) == 0;
}

}  // namespace dream
