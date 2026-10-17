// The table of CPU kernels a tensor operation runs, and the codes they take.
//
// Two copies of the table exist, one per instruction set (see the head of
// tensor_kernels.inc); `tensor_kernels()` picks between them once.

#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>

namespace dream {

/// Binary operations, `+ - * / %` first -- shared with the GPU's kernels,
/// which switch on the same numbers. The rest are `std.tensor` functions
/// rather than operators: the larger and smaller of two, a power, and the
/// comparisons, which answer 1 where they hold and 0 where they do not, so
/// that a mask is a tensor like any other and multiplies into a chain.
enum KernelOp : int {
    KOP_ADD = 0, KOP_SUB, KOP_MUL, KOP_DIV, KOP_MOD,
    KOP_MAX, KOP_MIN, KOP_POW, KOP_LT, KOP_LE, KOP_GT, KOP_GE, KOP_EQ, KOP_NE,
};

/// Elementwise functions of one argument.
enum KernelFn : int {
    KFN_NEG = 0, KFN_SQRT, KFN_EXP, KFN_LOG, KFN_ABS, KFN_TANH, KFN_SIN, KFN_COS,
    KFN_RELU, KFN_SIGMOID, KFN_FLOOR, KFN_CEIL, KFN_ROUND, KFN_SIGN, KFN_ERF,
};

/// Reductions to one number, and along an axis. The two `ARG` forms are only
/// along an axis: where along it the least or greatest value first is.
enum KernelReduce : int { KRED_SUM = 0, KRED_MIN, KRED_MAX, KRED_ARGMIN, KRED_ARGMAX };

/// A fused elementwise program (see `TensorExpr` in value.hpp): two bytes an
/// instruction, an opcode and its argument, run on a stack.
///
///   FUSE_LOAD k   push input k, read at `i % count` -- equal shapes and every
///                 broadcast a tensor allows, since an input's shape is always
///                 a trailing part of the answer's
///   FUSE_CONST k  push constant k
///   FUSE_BIN op   pop y, pop x, push `x op y` (`KOP_*`)
///   FUSE_UN fn    pop x, push `fn x` (`KFN_*`)
///   FUSE_LOADT k  push input k -- an R x C matrix -- read as its C x R
///                 transpose, at `i % (R * C)`: what `tensor.transpose`
///                 defers to, so that a transpose is read where it lies
///   FUSE_PRODUCT  push the matrix product the program starts from, at `i`
///                 (see `TensorExpr::product_a`); at most one per program,
///                 and only where the answer is exactly as long as the product
///   FUSE_LOADR a  push input `a & 7`, each of its numbers repeated `r`
///                 times, where `r` is constant `a >> 3`: read at
///                 `(i / r) % count`. What `tensor.repeat` defers to, so that
///                 a value per row meets every number of its row -- a row's
///                 maximum subtracted from it -- inside the chain, with
///                 nothing stretched in memory
///
/// The limits keep a program small enough that the CPU's evaluator keeps its
/// whole stack in L1, and that a GPU kernel built from it has a short argument
/// list. A chain that would pass one is computed and starts a new program.
enum FuseCode : uint8_t {
    FUSE_LOAD = 0, FUSE_CONST, FUSE_BIN, FUSE_UN, FUSE_PRODUCT, FUSE_LOADT, FUSE_LOADR,
};
constexpr unsigned kFuseMaxInputs = 8, kFuseMaxConsts = 16, kFuseMaxCode = 64,
                   kFuseMaxDepth = 12;

struct FuseProgram {
    const uint8_t* code;
    unsigned ncode, ninputs, nconsts, depth;
};

/// What the matrix product calls as each tile of C becomes final -- once its
/// last panel of K has been added in, while the tile is still in cache. The
/// rows and columns are of the whole C: `row_base` is added to the rows of the
/// band this call computes, which is how a band on its own thread reports
/// where it is.
struct GemmEpilogue {
    void (*fn)(void* ctx, size_t row0, size_t row1, size_t col0, size_t col1);
    void* ctx;
    size_t row_base;
};

/// One operand of the matrix product, as the product reads it. Element
/// `(i, j)` is `data[(i + row0) * rs + j * cs]` -- a matrix as it is stored
/// (`rs` its width, `cs` 1), or one read as its transpose (`rs` 1, `cs` the
/// stored width), with nothing copied either way. When `segment` is set it is
/// instead whatever `segment` writes for `len` elements of row `i + row0`
/// from column `j`: a deferred tensor computed straight into the panel the
/// product was about to copy it into. `row0` is how a band of rows on its
/// own thread is told where it starts.
struct GemmOperand {
    const double* data;
    size_t rs, cs;
    void (*segment)(void* ctx, size_t row, size_t col, size_t len, double* out);
    void* ctx;
    size_t row0;
};

struct TensorKernels {
    /// `out[i] = x[i] op y[i]` over `n` elements.
    void (*binary)(int op, const double* x, const double* y, double* out, size_t n);
    /// `out[i] = x[i] op s`, or `s op x[i]` when `scalar_left`.
    void (*scalar)(int op, const double* x, double s, bool scalar_left, double* out, size_t n);
    void (*unary)(int fn, const double* x, double* out, size_t n);
    double (*dot)(const double* a, const double* b, size_t n);
    /// `n` must be at least 1.
    double (*reduce)(int op, const double* x, size_t n);
    /// `C = A x B`, `C` row-major and overwritten, and `epi` (when there is
    /// one) told as each tile of C is finished.
    void (*gemm)(const GemmOperand& A, const GemmOperand& B, double* C, size_t M, size_t K,
                 size_t N, const GemmEpilogue* epi);
    void (*transpose)(const double* in, double* out, size_t rows, size_t cols);
};

namespace kernels_base { extern const TensorKernels kernels; }
#if defined(DREAM_TENSOR_AVX2)
namespace kernels_avx2 { extern const TensorKernels kernels; }
#endif

/// The fastest table this CPU can run, chosen on the first call.
const TensorKernels& tensor_kernels();
/// Every table this CPU can run, baseline first: what the VM's own tests run
/// each kernel through. A function rather than the tables themselves because a
/// Windows DLL exports functions by default and data only when told to.
std::vector<const TensorKernels*> tensor_kernel_tables();
/// Its name, for `tensor.backend`: "avx2" or "baseline".
const char* tensor_kernels_name();

}  // namespace dream
