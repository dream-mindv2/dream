#include "bigint.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#include "heap.hpp"

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <intrin.h>
#endif

namespace dream::bigint {

namespace {

using Limb = uint64_t;

// --- one word at a time ----------------------------------------------------------
//
// The three operations every algorithm here is built from: a 64x64 product to
// 128 bits, and a 128-by-64 quotient whose high word is below the divisor.
// GCC and Clang have a 128-bit integer; MSVC has the x64 intrinsics; anything
// else gets the four-halves versions, which are correct and slower.

#if defined(__SIZEOF_INT128__)
using U128 = unsigned __int128;

inline Limb mul_wide(Limb a, Limb b, Limb* hi) {
    U128 p = U128(a) * b;
    *hi = Limb(p >> 64);
    return Limb(p);
}

inline Limb div_wide(Limb hi, Limb lo, Limb d, Limb* r) {
#if defined(__x86_64__)
    // `divq` is exactly this operation. Written as `U128 / d`, GCC and Clang
    // call `__udivti3`, a general 128-by-128 division, several times slower.
    Limb q;
    __asm__("divq %4" : "=a"(q), "=d"(*r) : "a"(lo), "d"(hi), "rm"(d));
    return q;
#else
    U128 n = (U128(hi) << 64) | lo;
    *r = Limb(n % d);
    return Limb(n / d);
#endif
}

#elif defined(_MSC_VER) && defined(_M_X64)

inline Limb mul_wide(Limb a, Limb b, Limb* hi) { return _umul128(a, b, hi); }
inline Limb div_wide(Limb hi, Limb lo, Limb d, Limb* r) { return _udiv128(hi, lo, d, r); }

#else

inline Limb mul_wide(Limb a, Limb b, Limb* hi) {
    const Limb a0 = uint32_t(a), a1 = a >> 32, b0 = uint32_t(b), b1 = b >> 32;
    const Limb p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const Limb mid = (p00 >> 32) + uint32_t(p01) + uint32_t(p10);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (mid << 32) | uint32_t(p00);
}

/// Bit by bit: the portable fallback, reached only on a compiler with neither
/// a 128-bit integer nor the x64 intrinsics.
inline Limb div_wide(Limb hi, Limb lo, Limb d, Limb* r) {
    Limb q = 0;
    for (int i = 63; i >= 0; --i) {
        const bool carry = (hi >> 63) != 0;
        hi = (hi << 1) | (lo >> 63);
        lo <<= 1;
        q <<= 1;
        if (carry || hi >= d) {
            hi -= d;
            q |= 1;
        }
    }
    *r = hi;
    return q;
}

#endif

inline Limb add_carry(Limb a, Limb b, Limb* carry) {
    Limb s = a + *carry;
    Limb c = s < a;
    s += b;
    *carry = c + (s < b);
    return s;
}

inline Limb sub_borrow(Limb a, Limb b, Limb* borrow) {
    Limb d = a - b;
    Limb br = a < b;
    Limb e = d - *borrow;
    *borrow = br + (d < *borrow);
    return e;
}

inline size_t trimmed(const Limb* d, size_t n) {
    while (n > 0 && d[n - 1] == 0) --n;
    return n;
}

// --- magnitudes ---------------------------------------------------------------------
//
// Unsigned limb arrays, least significant first. Lengths are trimmed on input
// unless a function says otherwise; outputs are written at full width and
// trimmed by whoever reads them.

int cmp_mag(const Limb* a, size_t an, const Limb* b, size_t bn) {
    if (an != bn) return an < bn ? -1 : 1;
    for (size_t i = an; i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

/// out = a + b, an >= bn. Writes an + 1 limbs.
void add_mag(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out) {
    Limb carry = 0;
    size_t i = 0;
    for (; i < bn; ++i) out[i] = add_carry(a[i], b[i], &carry);
    for (; i < an; ++i) {
        out[i] = a[i] + carry;
        carry = out[i] < carry;
    }
    out[an] = carry;
}

/// out = a - b, a >= b. Writes an limbs. `out` may be `a`.
void sub_mag(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out) {
    Limb borrow = 0;
    size_t i = 0;
    for (; i < bn; ++i) out[i] = sub_borrow(a[i], b[i], &borrow);
    for (; i < an; ++i) {
        out[i] = a[i] - borrow;
        borrow = a[i] < borrow;
    }
}

/// a += b in place, where a has room for the carry to run off the end of b:
/// `an` is a's full width and the carry out of it is returned.
Limb add_into(Limb* a, size_t an, const Limb* b, size_t bn) {
    Limb carry = 0;
    size_t i = 0;
    for (; i < bn; ++i) a[i] = add_carry(a[i], b[i], &carry);
    for (; carry && i < an; ++i) {
        a[i] += carry;
        carry = a[i] == 0;
    }
    return carry;
}

/// a -= b in place, a >= b.
void sub_into(Limb* a, size_t an, const Limb* b, size_t bn) {
    Limb borrow = 0;
    size_t i = 0;
    for (; i < bn; ++i) a[i] = sub_borrow(a[i], b[i], &borrow);
    for (; borrow && i < an; ++i) {
        borrow = a[i] == 0;
        a[i] -= 1;
    }
}

/// out[0..an] = a * m + add. Returns nothing; out[an] is the high limb.
void mul_1(const Limb* a, size_t an, Limb m, Limb add, Limb* out) {
    Limb carry = add;
    for (size_t i = 0; i < an; ++i) {
        Limb hi;
        Limb lo = mul_wide(a[i], m, &hi);
        lo += carry;
        hi += lo < carry;
        out[i] = lo;
        carry = hi;
    }
    out[an] = carry;
}

/// out[0..an] += a * m, returning the carry out of out[an - 1].
Limb addmul_1(Limb* out, const Limb* a, size_t an, Limb m) {
    Limb carry = 0;
    for (size_t i = 0; i < an; ++i) {
        Limb hi;
        Limb lo = mul_wide(a[i], m, &hi);
        lo += carry;
        hi += lo < carry;
        lo += out[i];
        hi += lo < out[i];
        out[i] = lo;
        carry = hi;
    }
    return carry;
}

/// out = a * b, schoolbook. Writes an + bn limbs; out must not overlap.
void mul_school(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out) {
    if (an < bn) {
        std::swap(a, b);
        std::swap(an, bn);
    }
    if (bn == 0) {
        std::fill(out, out + an, Limb(0));
        return;
    }
    mul_1(a, an, b[0], 0, out);
    for (size_t j = 1; j < bn; ++j) out[an + j] = addmul_1(out + j, a, an, b[j]);
}

/// Below this many limbs in the smaller operand, schoolbook wins: Karatsuba's
/// three half-size products save a quarter of the multiplies and pay for it in
/// additions and scratch. 32 is where the two cross on x86-64; it is a broad
/// minimum, and anything from 24 to 48 measures the same.
constexpr size_t kKaratsubaThreshold = 32;

void mul_mag(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out);

/// Karatsuba for an >= bn > an / 2: split both at m = ceil(an / 2),
///   a*b = z2 B^2m + ((a0 + a1)(b0 + b1) - z0 - z2) B^m + z0.
void mul_karatsuba(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out) {
    const size_t m = (an + 1) / 2;
    const Limb *a0 = a, *a1 = a + m, *b0 = b, *b1 = b + m;
    const size_t a0n = trimmed(a0, m), a1n = an - m;
    const size_t b0n = trimmed(b0, m), b1n = bn - m;

    // z0 into the low 2m limbs of out, z2 into the rest. They do not overlap,
    // and together they are exactly out's an + bn limbs.
    std::fill(out, out + an + bn, Limb(0));
    mul_mag(a0, a0n, b0, b0n, out);
    mul_mag(a1, a1n, b1, b1n, out + 2 * m);

    std::vector<Limb> scratch(4 * (m + 1) + 2);
    Limb* sa = scratch.data();          // a0 + a1, m + 1 limbs
    Limb* sb = sa + (m + 1);            // b0 + b1, m + 1 limbs
    Limb* z1 = sb + (m + 1);            // their product, 2m + 2 limbs
    std::fill(sa, sa + 2 * (m + 1), Limb(0));
    if (a0n >= a1n) add_mag(a0, a0n, a1, a1n, sa); else add_mag(a1, a1n, a0, a0n, sa);
    if (b0n >= b1n) add_mag(b0, b0n, b1, b1n, sb); else add_mag(b1, b1n, b0, b0n, sb);
    const size_t san = trimmed(sa, m + 1), sbn = trimmed(sb, m + 1);
    std::fill(z1, z1 + 2 * m + 2, Limb(0));
    mul_mag(sa, san, sb, sbn, z1);

    // z1 -= z0 and z1 -= z2, read back out of out where they were written.
    size_t z1n = trimmed(z1, 2 * m + 2);
    sub_into(z1, z1n, out, trimmed(out, 2 * m));
    sub_into(z1, z1n, out + 2 * m, trimmed(out + 2 * m, an + bn - 2 * m));
    z1n = trimmed(z1, z1n);
    add_into(out + m, an + bn - m, z1, z1n);
}

/// out = a * b. Writes an + bn limbs; out must not overlap either operand.
void mul_mag(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* out) {
    if (an < bn) {
        std::swap(a, b);
        std::swap(an, bn);
    }
    if (bn < kKaratsubaThreshold) {
        mul_school(a, an, b, bn, out);
        return;
    }
    if (2 * bn > an) {
        mul_karatsuba(a, an, b, bn, out);
        return;
    }
    // Unbalanced: a in pieces of b's length, each a balanced product, added
    // into place. Without this, splitting a long a at its middle leaves b's
    // upper half empty and the recursion degenerates to schoolbook.
    std::fill(out, out + an + bn, Limb(0));
    std::vector<Limb> piece(2 * bn);
    for (size_t at = 0; at < an; at += bn) {
        const size_t len = std::min(bn, an - at);
        const size_t pn = trimmed(a + at, len);
        std::fill(piece.begin(), piece.end(), Limb(0));
        mul_mag(a + at, pn, b, bn, piece.data());
        add_into(out + at, an + bn - at, piece.data(), trimmed(piece.data(), pn + bn));
    }
}

/// q = a / d and the remainder returned, for a single-limb d. q has an limbs;
/// it may be a.
Limb divmod_1(const Limb* a, size_t an, Limb d, Limb* q) {
    Limb r = 0;
    for (size_t i = an; i-- > 0;) q[i] = div_wide(r, a[i], d, &r);
    return r;
}

/// Knuth's algorithm D (TAOCP 4.3.1). a = q * b + r, bn >= 2, an >= bn.
/// q gets an - bn + 1 limbs and r bn limbs; either may be null.
void divmod_knuth(const Limb* a, size_t an, const Limb* b, size_t bn, Limb* q, Limb* r) {
    // Normalise: shift both until b's top bit is set, so each quotient digit
    // guessed from the top two limbs is at most two too large.
    const int s = std::countl_zero(b[bn - 1]);
    std::vector<Limb> buf(an + 1 + bn);
    Limb* u = buf.data();
    Limb* v = u + an + 1;
    if (s) {
        for (size_t i = bn - 1; i > 0; --i) v[i] = (b[i] << s) | (b[i - 1] >> (64 - s));
        v[0] = b[0] << s;
        u[an] = a[an - 1] >> (64 - s);
        for (size_t i = an - 1; i > 0; --i) u[i] = (a[i] << s) | (a[i - 1] >> (64 - s));
        u[0] = a[0] << s;
    } else {
        std::memcpy(v, b, bn * 8);
        std::memcpy(u, a, an * 8);
        u[an] = 0;
    }

    const Limb vtop = v[bn - 1], vnext = v[bn - 2];
    for (size_t j = an - bn + 1; j-- > 0;) {
        // Guess q̂ from the top two limbs of the remainder over the top limb of
        // the divisor, and correct it with the next limb down.
        Limb qhat, rhat;
        bool rhat_overflow = false;
        if (u[j + bn] >= vtop) {
            qhat = ~Limb(0);
            rhat = u[j + bn - 1] + vtop;
            rhat_overflow = rhat < vtop;
        } else {
            qhat = div_wide(u[j + bn], u[j + bn - 1], vtop, &rhat);
        }
        while (!rhat_overflow) {
            Limb phi;
            Limb plo = mul_wide(qhat, vnext, &phi);
            if (phi < rhat || (phi == rhat && plo <= u[j + bn - 2])) break;
            --qhat;
            rhat += vtop;
            rhat_overflow = rhat < vtop;
        }

        // u[j..j+bn] -= q̂ * v. If that went negative, q̂ was one too large:
        // add v back and take one off.
        Limb borrow = 0, carry = 0;
        for (size_t i = 0; i < bn; ++i) {
            Limb hi;
            Limb lo = mul_wide(qhat, v[i], &hi);
            lo += carry;
            hi += lo < carry;
            carry = hi;
            u[j + i] = sub_borrow(u[j + i], lo, &borrow);
        }
        Limb top = u[j + bn];
        u[j + bn] = top - carry - borrow;
        const bool negative = top < carry || top - carry < borrow;
        if (negative) {
            --qhat;
            Limb c = 0;
            for (size_t i = 0; i < bn; ++i) u[j + i] = add_carry(u[j + i], v[i], &c);
            u[j + bn] += c;
        }
        if (q) q[j] = qhat;
    }

    if (r) {
        if (s) {
            for (size_t i = 0; i < bn - 1; ++i) r[i] = (u[i] >> s) | (u[i + 1] << (64 - s));
            r[bn - 1] = u[bn - 1] >> s;
        } else {
            std::memcpy(r, u, bn * 8);
        }
    }
}

// --- operands --------------------------------------------------------------------------
//
// Every value-level operation reads its operands through a `Mag`: the limbs,
// how many, and the sign. A fixnum's magnitude is one limb kept in the `Mag`
// itself -- its absolute value always fits, since a fixnum is 63 bits -- so a
// bignum and a fixnum are read by the same code.

struct Mag {
    const Limb* d;
    size_t n;
    bool neg;
    Limb one;
};

inline void read(Value v, Mag* m) {
    if (is_fixnum(v)) {
        int64_t x = fixnum_value(v);
        m->neg = x < 0;
        m->one = m->neg ? Limb(0) - Limb(x) : Limb(x);
        m->d = &m->one;
        m->n = m->one ? 1 : 0;
    } else {
        auto* b = static_cast<const BigIntObj*>(as_obj(v));
        m->d = b->limbs();
        m->n = b->len;
        m->neg = b->neg != 0;
    }
}

/// The canonical value of `o`'s first `n` limbs with sign `neg`: a fixnum when
/// it fits, `o` trimmed otherwise. Every operation ends here.
Value finish(BigIntObj* o, size_t n, bool neg) {
    const Limb* d = o->limbs();
    n = trimmed(d, n);
    if (n == 0) return make_fixnum(0);
    if (n == 1) {
        // A fixnum holds [-2^62, 2^62).
        constexpr Limb kLimit = Limb(1) << 62;
        if (d[0] < kLimit) return make_fixnum(neg ? -int64_t(d[0]) : int64_t(d[0]));
        if (neg && d[0] == kLimit) return make_fixnum(-int64_t(kLimit));
    }
    o->len = uint32_t(n);
    o->neg = neg ? 1 : 0;
    return from_obj(o);
}

/// |a| + |b| or |a| - |b| with the result's sign, as `add` and `sub` both are
/// once the signs have been looked at.
Value add_signed(Heap& h, const Mag& a, const Mag& b, bool b_neg) {
    if (a.neg == b_neg) {
        const Mag& x = a.n >= b.n ? a : b;
        const Mag& y = a.n >= b.n ? b : a;
        BigIntObj* o = h.alloc_bigint(uint32_t(x.n + 1));
        add_mag(x.d, x.n, y.d, y.n, o->limbs());
        return finish(o, x.n + 1, a.neg);
    }
    int c = cmp_mag(a.d, a.n, b.d, b.n);
    if (c == 0) return make_fixnum(0);
    const Mag& x = c > 0 ? a : b;
    const Mag& y = c > 0 ? b : a;
    BigIntObj* o = h.alloc_bigint(uint32_t(x.n));
    sub_mag(x.d, x.n, y.d, y.n, o->limbs());
    return finish(o, x.n, c > 0 ? a.neg : b_neg);
}

/// Quotient and remainder of magnitudes, either output optional, into fresh
/// bignums.
void divmod(Heap& h, const Mag& a, const Mag& b, Value* q, Value* r) {
    const bool qneg = a.neg != b.neg;
    if (cmp_mag(a.d, a.n, b.d, b.n) < 0) {
        if (q) *q = make_fixnum(0);
        if (r) {
            BigIntObj* o = h.alloc_bigint(uint32_t(a.n));
            std::memcpy(o->limbs(), a.d, a.n * 8);
            *r = finish(o, a.n, a.neg);
        }
        return;
    }
    if (b.n == 1) {
        BigIntObj* qo = h.alloc_bigint(uint32_t(a.n));
        Limb rem = divmod_1(a.d, a.n, b.d[0], qo->limbs());
        if (q) *q = finish(qo, a.n, qneg);
        if (r) {
            BigIntObj* ro = h.alloc_bigint(1);
            ro->limbs()[0] = rem;
            *r = finish(ro, 1, a.neg);
        }
        return;
    }
    BigIntObj* qo = q ? h.alloc_bigint(uint32_t(a.n - b.n + 1)) : nullptr;
    BigIntObj* ro = r ? h.alloc_bigint(uint32_t(b.n)) : nullptr;
    divmod_knuth(a.d, a.n, b.d, b.n, qo ? qo->limbs() : nullptr, ro ? ro->limbs() : nullptr);
    if (q) *q = finish(qo, a.n - b.n + 1, qneg);
    if (r) *r = finish(ro, b.n, a.neg);
}

/// 10^19, the largest power of ten in a limb: decimal is read and written
/// nineteen digits at a time.
constexpr Limb kTen19 = 10000000000000000000ull;

}  // namespace

// --- arithmetic --------------------------------------------------------------------------

Value add(Heap& h, Value a, Value b) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    return add_signed(h, x, y, y.neg);
}

Value sub(Heap& h, Value a, Value b) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    return add_signed(h, x, y, y.n ? !y.neg : false);
}

Value mul(Heap& h, Value a, Value b) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    if (x.n == 0 || y.n == 0) return make_fixnum(0);
    BigIntObj* o = h.alloc_bigint(uint32_t(x.n + y.n));
    if (y.n == 1) mul_1(x.d, x.n, y.d[0], 0, o->limbs());
    else if (x.n == 1) mul_1(y.d, y.n, x.d[0], 0, o->limbs());
    else mul_mag(x.d, x.n, y.d, y.n, o->limbs());
    return finish(o, x.n + y.n, x.neg != y.neg);
}

Value quot(Heap& h, Value a, Value b) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    Value q = make_fixnum(0);
    divmod(h, x, y, &q, nullptr);
    return q;
}

Value rem(Heap& h, Value a, Value b) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    Value r = make_fixnum(0);
    divmod(h, x, y, nullptr, &r);
    return r;
}

Value negate(Heap& h, Value a) {
    Mag x;
    read(a, &x);
    BigIntObj* o = h.alloc_bigint(uint32_t(x.n ? x.n : 1));
    std::memcpy(o->limbs(), x.d, x.n * 8);
    return finish(o, x.n, !x.neg);
}

Value abs(Heap& h, Value a) {
    Mag x;
    read(a, &x);
    if (!x.neg) return a;
    BigIntObj* o = h.alloc_bigint(uint32_t(x.n));
    std::memcpy(o->limbs(), x.d, x.n * 8);
    return finish(o, x.n, false);
}

// --- bitwise -----------------------------------------------------------------------------
//
// Sign-magnitude has no bits to combine, so `&`, `|` and `^` go through two's
// complement and back, as CPython's do: each operand is written out in one
// limb more than the wider of them -- so the top limb is all sign extension,
// zeros or ones -- the limbs are combined, and the top limb of the answer
// says its sign. A negative answer's magnitude is its two's complement
// negated again. Shifts need none of that: they move a magnitude, and only
// `>>` of a negative number looks at the bits it drops.

namespace {

/// `m` as `n` limbs of two's complement; `n` exceeds `m.n`.
void to_twos(const Mag& m, size_t n, Limb* out) {
    std::memset(out, 0, n * 8);
    std::memcpy(out, m.d, m.n * 8);
    if (m.neg) {
        Limb carry = 1;
        for (size_t i = 0; i < n; ++i) {
            Limb v = ~out[i];
            out[i] = v + carry;
            carry = carry && out[i] == 0 ? 1 : 0;
        }
    }
}

enum class BitOp { And, Or, Xor };

Value bitwise(Heap& h, Value a, Value b, BitOp op) {
    Mag x, y;
    read(a, &x);
    read(b, &y);
    const size_t n = std::max(x.n, y.n) + 1;
    std::vector<Limb> xa(n), ya(n);
    to_twos(x, n, xa.data());
    to_twos(y, n, ya.data());
    BigIntObj* o = h.alloc_bigint(uint32_t(n));
    Limb* r = o->limbs();
    for (size_t i = 0; i < n; ++i) {
        switch (op) {
            case BitOp::And: r[i] = xa[i] & ya[i]; break;
            case BitOp::Or: r[i] = xa[i] | ya[i]; break;
            case BitOp::Xor: r[i] = xa[i] ^ ya[i]; break;
        }
    }
    const bool neg = (r[n - 1] >> 63) != 0;
    if (neg) {
        // Back to a magnitude: invert and add one.
        Limb carry = 1;
        for (size_t i = 0; i < n; ++i) {
            Limb v = ~r[i];
            r[i] = v + carry;
            carry = carry && r[i] == 0 ? 1 : 0;
        }
    }
    return finish(o, n, neg);
}

}  // namespace

Value bit_and(Heap& h, Value a, Value b) { return bitwise(h, a, b, BitOp::And); }
Value bit_or(Heap& h, Value a, Value b) { return bitwise(h, a, b, BitOp::Or); }
Value bit_xor(Heap& h, Value a, Value b) { return bitwise(h, a, b, BitOp::Xor); }

Value bit_not(Heap& h, Value a) {
    // ~a is -(a + 1).
    return negate(h, add(h, a, make_fixnum(1)));
}

Value shift_left(Heap& h, Value a, uint64_t k) {
    Mag x;
    read(a, &x);
    if (x.n == 0) return make_fixnum(0);
    const size_t limbs = size_t(k / 64);
    const unsigned bits = unsigned(k % 64);
    const size_t n = x.n + limbs + 1;
    BigIntObj* o = h.alloc_bigint(uint32_t(n));
    Limb* r = o->limbs();
    std::memset(r, 0, n * 8);
    for (size_t i = 0; i < x.n; ++i) {
        r[i + limbs] |= x.d[i] << bits;
        if (bits) r[i + limbs + 1] |= x.d[i] >> (64 - bits);
    }
    return finish(o, n, x.neg);
}

Value shift_right(Heap& h, Value a, uint64_t k) {
    Mag x;
    read(a, &x);
    const uint64_t limbs = k / 64;
    const unsigned bits = unsigned(k % 64);
    if (limbs >= x.n) return make_fixnum(x.neg ? -1 : 0);
    // Whether any 1 bit falls off the bottom: a negative number then rounds
    // down, one further from zero, as floor division by 2^k does.
    bool lost = false;
    for (uint64_t i = 0; i < limbs && !lost; ++i) lost = x.d[i] != 0;
    if (!lost && bits) lost = (x.d[limbs] & ((Limb(1) << bits) - 1)) != 0;
    const size_t n = size_t(x.n - limbs);
    BigIntObj* o = h.alloc_bigint(uint32_t(n + 1));
    Limb* r = o->limbs();
    for (size_t i = 0; i < n; ++i) {
        Limb lo = x.d[i + limbs] >> bits;
        Limb hi = bits && i + limbs + 1 < x.n ? x.d[i + limbs + 1] << (64 - bits) : 0;
        r[i] = lo | hi;
    }
    r[n] = 0;
    if (x.neg && lost) {
        Limb carry = 1;
        for (size_t i = 0; i <= n && carry; ++i) {
            r[i] += 1;
            carry = r[i] == 0 ? 1 : 0;
        }
    }
    return finish(o, n + 1, x.neg);
}

// --- comparison --------------------------------------------------------------------------

int compare(Value a, Value b) {
    if (is_fixnum(a) && is_fixnum(b)) {
        int64_t x = fixnum_value(a), y = fixnum_value(b);
        return x < y ? -1 : (x > y ? 1 : 0);
    }
    Mag x, y;
    read(a, &x);
    read(b, &y);
    if (x.neg != y.neg) return x.neg ? -1 : 1;
    int c = cmp_mag(x.d, x.n, y.d, y.n);
    return x.neg ? -c : c;
}

bool equal(const BigIntObj* a, const BigIntObj* b) {
    return a->len == b->len && a->neg == b->neg &&
           std::memcmp(a->limbs(), b->limbs(), size_t(a->len) * 8) == 0;
}

uint64_t hash(const BigIntObj* a) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ a->neg;
    for (uint32_t i = 0; i < a->len; ++i) {
        h ^= a->limbs()[i];
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    return h;
}

// --- conversion ----------------------------------------------------------------------------

double to_double(Value v) {
    if (is_fixnum(v)) return double(fixnum_value(v));
    Mag m;
    read(v, &m);
    // The top 64 significant bits, and whether anything below them is set.
    // Rounding those to 53 by hand is the one way to round once: converting
    // limb by limb and adding rounds at every step.
    const Limb* d = m.d;
    const size_t n = m.n;
    const int lead = std::countl_zero(d[n - 1]);
    const size_t bits = n * 64 - size_t(lead);
    Limb top = d[n - 1];
    bool sticky = false;
    size_t below = n - 1;  // limbs wholly beneath `top`
    if (lead && n >= 2) {
        top = (top << lead) | (d[n - 2] >> (64 - lead));
        sticky = (d[n - 2] << lead) != 0;
        below = n - 2;
    } else if (lead) {
        top <<= lead;
    }
    for (size_t i = 0; !sticky && i < below; ++i) sticky = d[i] != 0;
    Limb mant = top >> 11;
    const Limb rest = top & 0x7ff;
    if (rest > 0x400 || (rest == 0x400 && (sticky || (mant & 1)))) ++mant;
    // `mant` is now at most 2^53, which a double holds exactly; ldexp scales
    // it, and past the exponent range answers infinity.
    double r = std::ldexp(double(mant), int(bits) - 53);
    return m.neg ? -r : r;
}

bool from_double(Heap& h, double d, Value* out) {
    if (!std::isfinite(d)) return false;
    d = std::trunc(d);
    if (d >= -4611686018427387904.0 && d < 4611686018427387904.0) {
        *out = make_fixnum(int64_t(d));
        return true;
    }
    int exp;
    double frac = std::frexp(std::fabs(d), &exp);  // |d| = frac * 2^exp, frac in [0.5, 1)
    Limb mant = Limb(std::ldexp(frac, 53));           // 53 significant bits
    const int shift = exp - 53;                        // >= 9 here: |d| >= 2^62
    const size_t n = size_t(shift / 64) + 2;
    BigIntObj* o = h.alloc_bigint(uint32_t(n));
    Limb* l = o->limbs();
    std::fill(l, l + n, Limb(0));
    const int word = shift / 64, bit = shift % 64;
    l[word] = mant << bit;
    if (bit) l[word + 1] = mant >> (64 - bit);
    *out = finish(o, n, d < 0);
    return true;
}

Value from_int64(Heap& h, int64_t v) {
    if (fixnum_fits(v)) return make_fixnum(v);
    BigIntObj* o = h.alloc_bigint(1);
    o->limbs()[0] = v < 0 ? Limb(0) - Limb(v) : Limb(v);
    return finish(o, 1, v < 0);
}

Value from_uint64(Heap& h, uint64_t v) {
    if (v < (uint64_t(1) << 62)) return make_fixnum(int64_t(v));
    BigIntObj* o = h.alloc_bigint(1);
    o->limbs()[0] = v;
    return finish(o, 1, false);
}

bool to_int64(Value v, int64_t* out) {
    if (is_fixnum(v)) {
        *out = fixnum_value(v);
        return true;
    }
    Mag m;
    read(v, &m);
    if (m.n != 1) return false;
    if (m.neg) {
        if (m.d[0] > (Limb(1) << 63)) return false;
        *out = int64_t(Limb(0) - m.d[0]);
    } else {
        if (m.d[0] >= (Limb(1) << 63)) return false;
        *out = int64_t(m.d[0]);
    }
    return true;
}

bool to_uint64(Value v, uint64_t* out) {
    if (is_fixnum(v)) {
        if (fixnum_value(v) < 0) return false;
        *out = uint64_t(fixnum_value(v));
        return true;
    }
    Mag m;
    read(v, &m);
    if (m.neg || m.n != 1) return false;
    *out = m.d[0];
    return true;
}

void to_le_bytes(Value v, size_t width, uint8_t* out) {
    Mag m;
    read(v, &m);
    // The magnitude's bytes, then -- for a negative number -- inverted with
    // one added, which is the two's complement at this width.
    for (size_t i = 0; i < width; ++i) {
        const size_t limb = i / 8;
        out[i] = limb < m.n ? uint8_t(m.d[limb] >> (8 * (i % 8))) : uint8_t(0);
    }
    if (m.neg) {
        unsigned carry = 1;
        for (size_t i = 0; i < width; ++i) {
            unsigned b = unsigned(uint8_t(~out[i])) + carry;
            out[i] = uint8_t(b);
            carry = b >> 8;
        }
    }
}

size_t le_width(Value v) {
    Mag m;
    read(v, &m);
    if (m.n == 0) return 1;
    const size_t bits = m.n * 64 - size_t(std::countl_zero(m.d[m.n - 1]));
    // A non-negative number needs a clear sign bit above its `bits`; a
    // negative one of magnitude exactly 2^(bits-1) is the one value that fits
    // with the sign bit as its top bit.
    bool power_of_two = m.neg && std::popcount(m.d[m.n - 1]) == 1;
    for (size_t i = 0; power_of_two && i + 1 < m.n; ++i) power_of_two = m.d[i] == 0;
    return power_of_two ? (bits + 7) / 8 : bits / 8 + 1;
}

Value from_le_bytes(Heap& h, const uint8_t* bytes, size_t n) {
    if (n == 0) return make_fixnum(0);
    const bool neg = (bytes[n - 1] & 0x80) != 0;
    const size_t limbs = (n + 7) / 8;
    BigIntObj* o = h.alloc_bigint(uint32_t(limbs));
    Limb* l = o->limbs();
    std::fill(l, l + limbs, Limb(0));
    // The magnitude of a negative number is the complement plus one.
    unsigned carry = 1;
    for (size_t i = 0; i < n; ++i) {
        uint8_t b = bytes[i];
        if (neg) {
            unsigned x = unsigned(uint8_t(~b)) + carry;
            b = uint8_t(x);
            carry = x >> 8;
        }
        l[i / 8] |= Limb(b) << (8 * (i % 8));
    }
    return finish(o, limbs, neg);
}

void append_decimal(Value v, std::string* out) {
    if (is_fixnum(v)) {
        out->append(std::to_string(fixnum_value(v)));
        return;
    }
    Mag m;
    read(v, &m);
    // Peel nineteen digits at a time off a copy, least significant first.
    std::vector<Limb> work(m.d, m.d + m.n);
    std::vector<Limb> chunks;
    chunks.reserve(m.n * 20 / 19 + 1);
    size_t n = m.n;
    while (n > 0) {
        chunks.push_back(divmod_1(work.data(), n, kTen19, work.data()));
        n = trimmed(work.data(), n);
    }
    if (m.neg) out->push_back('-');
    out->append(std::to_string(chunks.back()));
    char buf[20];
    for (size_t i = chunks.size() - 1; i-- > 0;) {
        Limb c = chunks[i];
        for (int k = 18; k >= 0; --k) {
            buf[k] = char('0' + c % 10);
            c /= 10;
        }
        out->append(buf, 19);
    }
}

std::string to_string(Value v) {
    std::string s;
    append_decimal(v, &s);
    return s;
}

bool parse(Heap& h, const char* s, size_t n, Value* out) {
    size_t i = 0;
    bool neg = false;
    if (i < n && (s[i] == '-' || s[i] == '+')) {
        neg = s[i] == '-';
        ++i;
    }
    if (i == n) return false;
    for (size_t k = i; k < n; ++k) {
        if (s[k] < '0' || s[k] > '9') return false;
    }
    while (i + 1 < n && s[i] == '0') ++i;
    const size_t digits = n - i;
    // Eighteen digits are under 2^62: a fixnum, and no allocation.
    if (digits <= 18) {
        int64_t v = 0;
        for (; i < n; ++i) v = v * 10 + (s[i] - '0');
        *out = make_fixnum(neg ? -v : v);
        return true;
    }
    // A limb holds 19.26 digits, so this is enough and at most one more.
    const size_t cap = digits / 19 + 2;
    BigIntObj* o = h.alloc_bigint(uint32_t(cap));
    Limb* l = o->limbs();
    size_t len = 0;
    // The first chunk takes whatever is left over, so every one after it is
    // exactly nineteen digits and multiplies by exactly 10^19.
    size_t first = digits % 19 ? digits % 19 : 19;
    while (i < n) {
        Limb chunk = 0, scale = 1;
        for (size_t k = 0; k < first; ++k) {
            chunk = chunk * 10 + Limb(s[i + k] - '0');
            scale *= 10;
        }
        i += first;
        first = 19;
        mul_1(l, len, scale, chunk, l);
        len = trimmed(l, len + 1);
    }
    *out = finish(o, len, neg);
    return true;
}

}  // namespace dream::bigint
