// The CPU kernels behind tensors, written once and compiled twice.
//
// This file is included by tensor_kernels.cpp, which compiles it for the
// baseline the rest of the VM targets, and by tensor_kernels_avx2.cpp, which
// compiles it again with AVX2 and FMA switched on. `tensor_kernels()` in
// tensor.cpp asks the CPU once which of the two it can run and hands out that
// table. Each copy lives in its own namespace, named by `DREAM_KERNEL_NS`.
//
// Why two translation units rather than `target_clones` on each function: the
// matrix product's inner kernel only reaches its speed when it is inlined into
// the loops around it, and GCC will not inline a function compiled for one
// instruction set into a clone compiled for another. The clone it produced
// spilled its accumulators to memory and ran at 1 GFLOP/s where the same
// source built for the machine ran at 33 -- see "The matrix product" in
// docs/notes/tensors.md. Compiling the whole file per target sidesteps the
// question: everything in it agrees about the instruction set.
//
// Nothing here knows about Dream values. Everything is a pointer, a count and
// an operation code; tensor.cpp has already checked shapes and sizes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "tensor_kernels.hpp"

namespace dream {
namespace DREAM_KERNEL_NS {
namespace {

using Z = size_t;

// Each operation is its own loop, with the switch outside it, so that every
// loop is a plain `out[i] = x[i] op y[i]` the vectorizer can take: `each`
// is instantiated once per lambda, and each lambda is its own type.
template <class F>
inline void each(const double* __restrict x, const double* __restrict y, double* __restrict out,
                 Z n, F f) {
    for (Z i = 0; i < n; ++i) out[i] = f(x[i], y[i]);
}
template <class F>
inline void each_scalar(const double* __restrict x, double s, double* __restrict out, Z n, F f) {
    for (Z i = 0; i < n; ++i) out[i] = f(x[i], s);
}
template <class F>
inline void each_unary(const double* __restrict x, double* __restrict out, Z n, F f) {
    for (Z i = 0; i < n; ++i) out[i] = f(x[i]);
}

// One case per `KernelOp`, written once and expanded into each of the three
// shapes of loop below. A comparison answers 1 or 0, so a mask multiplies.
#define DREAM_KERNEL_BINARY_CASES(LOOP)                                         \
    case KOP_ADD: LOOP([](double a, double b) { return a + b; }); break;        \
    case KOP_SUB: LOOP([](double a, double b) { return a - b; }); break;        \
    case KOP_MUL: LOOP([](double a, double b) { return a * b; }); break;        \
    case KOP_DIV: LOOP([](double a, double b) { return a / b; }); break;        \
    case KOP_MOD: LOOP([](double a, double b) { return std::fmod(a, b); }); break; \
    case KOP_MAX: LOOP([](double a, double b) { return std::fmax(a, b); }); break; \
    case KOP_MIN: LOOP([](double a, double b) { return std::fmin(a, b); }); break; \
    case KOP_POW: LOOP([](double a, double b) { return std::pow(a, b); }); break;  \
    case KOP_LT: LOOP([](double a, double b) { return a < b ? 1.0 : 0.0; }); break; \
    case KOP_LE: LOOP([](double a, double b) { return a <= b ? 1.0 : 0.0; }); break; \
    case KOP_GT: LOOP([](double a, double b) { return a > b ? 1.0 : 0.0; }); break; \
    case KOP_GE: LOOP([](double a, double b) { return a >= b ? 1.0 : 0.0; }); break; \
    case KOP_EQ: LOOP([](double a, double b) { return a == b ? 1.0 : 0.0; }); break; \
    default: LOOP([](double a, double b) { return a != b ? 1.0 : 0.0; }); break;

void binary(int op, const double* __restrict x, const double* __restrict y,
            double* __restrict out, Z n) {
#define DREAM_LOOP(f) each(x, y, out, n, f)
    switch (op) { DREAM_KERNEL_BINARY_CASES(DREAM_LOOP) }
#undef DREAM_LOOP
}

void scalar(int op, const double* __restrict x, double s, bool scalar_left,
            double* __restrict out, Z n) {
    if (scalar_left) {
        // `s op x[i]`: the same cases with the operands swapped.
#define DREAM_LOOP(f) each_scalar(x, s, out, n, [](double a, double b) { return f(b, a); })
        switch (op) { DREAM_KERNEL_BINARY_CASES(DREAM_LOOP) }
#undef DREAM_LOOP
        return;
    }
#define DREAM_LOOP(f) each_scalar(x, s, out, n, f)
    switch (op) { DREAM_KERNEL_BINARY_CASES(DREAM_LOOP) }
#undef DREAM_LOOP
}

#undef DREAM_KERNEL_BINARY_CASES

void unary(int fn, const double* __restrict x, double* __restrict out, Z n) {
    switch (fn) {
        case KFN_NEG: each_unary(x, out, n, [](double a) { return -a; }); break;
        case KFN_SQRT: each_unary(x, out, n, [](double a) { return std::sqrt(a); }); break;
        case KFN_EXP: each_unary(x, out, n, [](double a) { return std::exp(a); }); break;
        case KFN_LOG: each_unary(x, out, n, [](double a) { return std::log(a); }); break;
        case KFN_ABS: each_unary(x, out, n, [](double a) { return std::fabs(a); }); break;
        case KFN_TANH: each_unary(x, out, n, [](double a) { return std::tanh(a); }); break;
        case KFN_SIN: each_unary(x, out, n, [](double a) { return std::sin(a); }); break;
        case KFN_COS: each_unary(x, out, n, [](double a) { return std::cos(a); }); break;
        case KFN_RELU: each_unary(x, out, n, [](double a) { return a > 0.0 ? a : 0.0; }); break;
        case KFN_SIGMOID:
            each_unary(x, out, n, [](double a) { return 1.0 / (1.0 + std::exp(-a)); });
            break;
        case KFN_FLOOR: each_unary(x, out, n, [](double a) { return std::floor(a); }); break;
        case KFN_CEIL: each_unary(x, out, n, [](double a) { return std::ceil(a); }); break;
        // Half away from zero, which is C's `round` and OpenCL's: the two
        // devices agree.
        case KFN_ROUND: each_unary(x, out, n, [](double a) { return std::round(a); }); break;
        case KFN_SIGN:
            each_unary(x, out, n, [](double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); });
            break;
        default: each_unary(x, out, n, [](double a) { return std::erf(a); }); break;
    }
}

// Eight running sums rather than one. A single accumulator is a chain of
// dependent additions, one every four cycles, and the compiler may not split
// it -- floating-point addition is not associative, so splitting changes the
// answer. Writing the split out is what licenses the vector unit, at the price
// the comment just named: the rounding is that of eight interleaved sums, not
// of one left-to-right sum. Every BLAS makes the same trade.
double dot(const double* __restrict a, const double* __restrict b, Z n) {
    double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    Z i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; ++l) acc[l] += a[i + l] * b[i + l];
    double tail = 0;
    for (; i < n; ++i) tail += a[i] * b[i];
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) +
           tail;
}

double reduce(int op, const double* __restrict x, Z n) {
    if (op == KRED_SUM) {
        double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        Z i = 0;
        for (; i + 8 <= n; i += 8)
            for (int l = 0; l < 8; ++l) acc[l] += x[i + l];
        double tail = 0;
        for (; i < n; ++i) tail += x[i];
        return ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
               ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
    }
    double r = x[0];
    if (op == KRED_MIN) {
        for (Z i = 1; i < n; ++i) r = x[i] < r ? x[i] : r;
    } else {
        for (Z i = 1; i < n; ++i) r = x[i] > r ? x[i] : r;
    }
    return r;
}

// --- the matrix product ------------------------------------------------------
//
// The shape every fast GEMM has (Goto and van de Geijn): block the product so
// that a panel of B stays in L2 and a sliver of A in L1, copy both into the
// order the inner kernel reads them ("packing"), and make the inner kernel a
// fixed MR x NR tile of C held entirely in registers while it walks k. With
// four-wide vectors a 6 x 8 tile is twelve accumulators, two loads of B and a
// broadcast of A per step -- fifteen of AVX2's sixteen registers.
//
// The vector type is GCC's and Clang's portable one, as wide as the target's
// registers: four doubles with AVX, two on the baseline (SSE2 on x86-64, NEON
// on ARM). The tile is the same 6 x 8 either way; only the number of vectors
// per row of it changes. With AVX2 `c += a * b` contracts to an FMA.

#if defined(__AVX__)
constexpr size_t VW = 4;
#else
constexpr size_t VW = 2;
#endif
#if defined(__GNUC__) || defined(__clang__)
typedef double vec __attribute__((vector_size(VW * sizeof(double)), aligned(8)));
#else
// MSVC has no vector extension. A struct of the same doubles with the four
// operations `micro` uses gives the same layout -- the tile is copied out with
// `memcpy` -- and leaves vectorizing to the optimizer.
struct vec {
    double x[VW];
};
inline vec operator+(vec a, double s) {
    for (size_t i = 0; i < VW; ++i) a.x[i] += s;
    return a;
}
inline vec operator+(vec a, vec b) {
    for (size_t i = 0; i < VW; ++i) a.x[i] += b.x[i];
    return a;
}
inline vec operator*(vec a, vec b) {
    for (size_t i = 0; i < VW; ++i) a.x[i] *= b.x[i];
    return a;
}
inline vec& operator+=(vec& a, vec b) {
    for (size_t i = 0; i < VW; ++i) a.x[i] += b.x[i];
    return a;
}
#endif

constexpr Z MR = 6, NR = 8, NV = NR / VW, KC = 256, MC = 96, NC = 2048;

inline vec loadv(const double* p) {
    vec v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline void storev(double* p, vec v) { std::memcpy(p, &v, sizeof v); }

/// Row `i` of `op` from column `j`, `len` long: a pointer into the operand's
/// own numbers when they lie in a row, otherwise gathered or computed into
/// `tmp`.
inline const double* row_of(const GemmOperand& op, Z i, Z j, Z len, double* tmp) {
    if (op.segment) {
        op.segment(op.ctx, i + op.row0, j, len, tmp);
        return tmp;
    }
    const double* at = op.data + (i + op.row0) * op.rs + j * op.cs;
    if (op.cs == 1) return at;
    for (Z l = 0; l < len; ++l) tmp[l] = at[l * op.cs];
    return tmp;
}

/// Whether `op`'s columns lie contiguously: a transposed operand, which is
/// read down its columns rather than gathered across its rows.
inline bool by_column(const GemmOperand& op) { return !op.segment && op.rs == 1 && op.cs != 1; }

/// B's `kc x nc` block from `(pc, jc)` as panels of NR columns, each
/// `kc x NR` and contiguous, the last padded with zeros.
void pack_b(const GemmOperand& B, Z pc, Z jc, Z kc, Z nc, double* Bp, double* tmp) {
    if (by_column(B)) {
        for (Z j = 0; j < nc; ++j) {
            const double* col = B.data + (pc + B.row0) + (jc + j) * B.cs;
            double* d = Bp + (j / NR) * kc * NR + j % NR;
            for (Z k = 0; k < kc; ++k) d[k * NR] = col[k];
        }
    } else {
        for (Z k = 0; k < kc; ++k) {
            const double* src = row_of(B, pc + k, jc, nc, tmp);
            for (Z j = 0; j < nc; j += NR) {
                const Z w = std::min(NR, nc - j);
                double* d = Bp + (j / NR) * kc * NR + k * NR;
                for (Z l = 0; l < w; ++l) d[l] = src[j + l];
            }
        }
    }
    if (const Z w = nc % NR) {
        double* d = Bp + (nc / NR) * kc * NR;
        for (Z k = 0; k < kc; ++k)
            for (Z l = w; l < NR; ++l) d[k * NR + l] = 0;
    }
}

/// A's `mc x kc` block from `(ic, pc)` as slivers of MR rows, each stored
/// column by column, the last padded with zeros.
void pack_a(const GemmOperand& A, Z ic, Z pc, Z mc, Z kc, double* Ap, double* tmp) {
    if (by_column(A)) {
        for (Z k = 0; k < kc; ++k) {
            const double* col = A.data + (ic + A.row0) + (pc + k) * A.cs;
            for (Z i = 0; i < mc; ++i) Ap[(i / MR) * kc * MR + k * MR + i % MR] = col[i];
        }
    } else {
        for (Z i = 0; i < mc; ++i) {
            const double* src = row_of(A, ic + i, pc, kc, tmp);
            double* d = Ap + (i / MR) * kc * MR + i % MR;
            for (Z k = 0; k < kc; ++k) d[k * MR] = src[k];
        }
    }
    if (const Z h = mc % MR) {
        double* d = Ap + (mc / MR) * kc * MR;
        for (Z k = 0; k < kc; ++k)
            for (Z r = h; r < MR; ++r) d[k * MR + r] = 0;
    }
}

/// C's `h x w` tile (at most MR x NR) plus the product of an A sliver and a B
/// panel. A full tile is written straight from the registers; an edge tile
/// goes through a scratch copy so that nothing past C's edge is touched.
// GCC's unroll hint, which Clang reads too. MSVC has no such pragma and warns
// about one it does not know (C4068), so there it is nothing.
#ifndef DREAM_UNROLL
#if defined(__GNUC__) || defined(__clang__)
#define DREAM_PRAGMA(x) _Pragma(#x)
#define DREAM_UNROLL(n) DREAM_PRAGMA(GCC unroll n)
#else
#define DREAM_UNROLL(n)
#endif
#endif

inline void micro(Z kc, const double* __restrict a, const double* __restrict b, double* C,
                  Z ldc, Z h, Z w) {
    vec c[MR][NV] = {};
    for (Z k = 0; k < kc; ++k, a += MR, b += NR) {
        vec bv[NV];
DREAM_UNROLL(4)
        for (Z v = 0; v < NV; ++v) bv[v] = loadv(b + v * VW);
DREAM_UNROLL(8)
        for (Z r = 0; r < MR; ++r) {
            const vec ar = vec{} + a[r];
DREAM_UNROLL(4)
            for (Z v = 0; v < NV; ++v) c[r][v] += ar * bv[v];
        }
    }
    if (h == MR && w == NR) {
        for (Z r = 0; r < MR; ++r)
            for (Z v = 0; v < NV; ++v) {
                double* at = C + r * ldc + v * VW;
                storev(at, loadv(at) + c[r][v]);
            }
        return;
    }
    double t[MR][NR];
    std::memcpy(t, c, sizeof t);
    for (Z r = 0; r < h; ++r)
        for (Z l = 0; l < w; ++l) C[r * ldc + l] += t[r][l];
}

/// `C = A x B` for row-major `A` (M x K) and `B` (K x N), `C` (M x N)
/// overwritten. Thin products go to the loops that suit them: packing a panel
/// to multiply it by one column is all overhead.
///
/// `epi` sees each block of C the moment it is final. In the blocked loop that
/// is the last panel of K: once `pc` has reached the end, the `mc x nc` block
/// just computed will not be touched again, and it is as hot in cache as it
/// will ever be -- which is the whole of what fusing into a product is for. A
/// thin product hands over everything at the end, having touched it once.
void gemm(const GemmOperand& A, const GemmOperand& B, double* C, Z M, Z K, Z N,
          const GemmEpilogue* epi) {
    if (N == 1) {
        // B is one column: read it once into a vector, then a dot product
        // per row of A.
        std::vector<double> col(K), row(K);
        for (Z k = 0; k < K; ++k) col[k] = *row_of(B, k, 0, 1, &row[0]);
        for (Z i = 0; i < M; ++i) C[i] = dot(row_of(A, i, 0, K, row.data()), col.data(), K);
        if (epi) epi->fn(epi->ctx, epi->row_base, epi->row_base + M, 0, 1);
        return;
    }
    std::memset(C, 0, M * N * sizeof(double));
    if (M == 1) {
        // A row vector times a matrix: a sum of B's rows, each scaled, which
        // reads B once and in order.
        std::vector<double> arow(K), brow(N);
        const double* a = row_of(A, 0, 0, K, arow.data());
        for (Z k = 0; k < K; ++k) {
            const double ak = a[k];
            const double* __restrict b = row_of(B, k, 0, N, brow.data());
            double* __restrict c = C;
            for (Z j = 0; j < N; ++j) c[j] += ak * b[j];
        }
        if (epi) epi->fn(epi->ctx, epi->row_base, epi->row_base + 1, 0, N);
        return;
    }
    std::vector<double> Bp(KC * (NC + NR)), Ap((MC + MR) * KC), tmp(std::max(NC, KC));
    for (Z jc = 0; jc < N; jc += NC) {
        const Z nc = std::min(NC, N - jc);
        for (Z pc = 0; pc < K; pc += KC) {
            const Z kc = std::min(KC, K - pc);
            pack_b(B, pc, jc, kc, nc, Bp.data(), tmp.data());
            for (Z ic = 0; ic < M; ic += MC) {
                const Z mc = std::min(MC, M - ic);
                pack_a(A, ic, pc, mc, kc, Ap.data(), tmp.data());
                for (Z jr = 0; jr < nc; jr += NR)
                    for (Z ir = 0; ir < mc; ir += MR)
                        micro(kc, Ap.data() + (ir / MR) * kc * MR, Bp.data() + (jr / NR) * kc * NR,
                              C + (ic + ir) * N + jc + jr, N, std::min(MR, mc - ir),
                              std::min(NR, nc - jr));
                if (epi && pc + kc == K)
                    epi->fn(epi->ctx, epi->row_base + ic, epi->row_base + ic + mc, jc, jc + nc);
            }
        }
    }
}

/// `out` (C x R) is `in` (R x C) transposed, in 32 x 32 blocks so that both
/// sides are read and written a cache line at a time.
void transpose(const double* __restrict in, double* __restrict out, Z R, Z C) {
    constexpr Z B = 32;
    for (Z i0 = 0; i0 < R; i0 += B)
        for (Z j0 = 0; j0 < C; j0 += B) {
            const Z i1 = std::min(R, i0 + B), j1 = std::min(C, j0 + B);
            for (Z i = i0; i < i1; ++i)
                for (Z j = j0; j < j1; ++j) out[j * R + i] = in[i * C + j];
        }
}

}  // namespace

const TensorKernels kernels = {
    binary, scalar, unary, dot, reduce, gemm, transpose,
};

}  // namespace DREAM_KERNEL_NS
}  // namespace dream
