// The GPU behind `tensor.gpu`: device memory and the kernels that run on it.
//
// One device per runtime, found on first use and kept for the life of the VM.
// It is reached through OpenCL, loaded with `dlopen` rather than linked: the VM
// gains no build dependency, a machine with no OpenCL runs every program that
// does not ask for a GPU, and one that asks gets an error naming what is
// missing. OpenCL supports compatible NVIDIA, AMD, Intel and Apple runtimes,
// and a CPU implementation of it
// (POCL) exists, which is how the tests run this code on a machine without a
// GPU. Linux can also load registered vendor ICDs without a system loader.
// A CUDA or Metal backend would sit behind the same functions.
//
// Everything is asynchronous except `download`. Operations are enqueued in
// order on one queue and return at once, so a chain of products and sums runs
// on the device back to back without the host waiting on any of them; only
// reading an answer back stops to wait. That is the whole of the speed model,
// and docs/notes/tensors.md has the rest.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tensor_kernels.hpp"

namespace dream::gpu {

/// A block of device memory, reference counted. A GPU tensor holds one
/// reference; a reshape shares the buffer and takes another. The last release
/// frees the memory, which OpenCL defers until every queued command reading it
/// has run.
struct Buffer;

/// Element types a device buffer can hold. A float on a GPU is the fast case
/// -- most consumer GPUs run doubles at a thirty-second of the rate, or not at
/// all -- so `tensor.gpu` makes `F32` and `tensor.gpu64` asks for `F64`.
enum Dtype : int { F64 = 0, F32 = 1 };

/// Whether a device is usable, finding one on the first call. When it is not,
/// `why` says why: no OpenCL library, no device, a kernel that would not build.
bool available(std::string* why);
/// The device's name, or "" when there is none.
std::string device_name();
/// Whether the device can hold `F64` buffers.
bool has_f64();
/// The device's memory in bytes, or 0 when no device has been set up.
/// Asks nothing of OpenCL that is not already loaded: a program that never
/// touched the GPU answers 0 without looking for one.
size_t memory_bytes();

size_t dtype_size(int dtype);

Buffer* alloc(size_t bytes, std::string* err);
void retain(Buffer* b);
void release(Buffer* b);

/// A deferred operand of a product on the device: its program, with the
/// inputs and constants it reads, computed per element as the product loads
/// it. Only `fused_matmul` takes one.
struct OperandProgram {
    FuseProgram prog;
    Buffer* const* ins;
    const size_t* counts;
    const size_t* rows;
    const double* consts;
};

/// One operand of a product on the device: element `(i, j)` is
/// `buf[i * rs + j * cs]`, so that a transpose is read by strides, not copied
/// -- or, when `program` is set, that program's value at `i * width + j`.
struct Operand {
    Buffer* buf;
    size_t rs, cs;
    const OperandProgram* program = nullptr;
};

/// Host doubles into a device buffer of `dtype`, converting on the way.
bool upload(Buffer* dst, int dtype, const double* src, size_t n, std::string* err);
/// A device buffer of `dtype` back into host doubles. Waits for every queued
/// operation that writes it.
bool download(Buffer* src, int dtype, double* dst, size_t n, std::string* err);
/// `n` elements from `src` at element `offset` into `dst` at element
/// `dst_offset`.
bool copy(Buffer* src, int dtype, size_t offset, Buffer* dst, size_t n, std::string* err,
          size_t dst_offset = 0);
/// Seeded uniforms generated on the device, with no host staging buffer.
bool random_uniform(int dtype, Buffer* out, size_t n, uint64_t seed, std::string* err);
/// Rows of `src`, each `width` elements, gathered into `dst` in the order
/// `rows` names them: row `r` of `dst` is row `rows[r]` of `src`.
bool gather(int dtype, Buffer* src, const uint32_t* rows, size_t nrows, size_t width, Buffer* dst,
            std::string* err);

/// A fused program's values folded along an axis by `op` (a `KRED_*`): the
/// program's answer is `outer x len x inner`, and `out[o * inner + i]` folds
/// position `(o * len + a) * inner + i` over every `a`, in order. The `ARG`
/// forms store the position `a` the fold settled on.
bool fused_axis(int op, int dtype, const FuseProgram& prog, Buffer* const* ins, const size_t* counts,
                const size_t* rows, const double* consts, size_t outer, size_t len, size_t inner,
                Buffer* out, std::string* err);
/// A fused elementwise program (`FuseProgram`, tensor_kernels.hpp) computed
/// into `out`: one kernel, generated from the program the first time it is
/// met and cached. `ins[k]` holds `counts[k]` elements, read at `i % count`.
///
/// `rows[k]` is input k's rows when it is a matrix, which a `FUSE_LOADT`
/// needs to read it as its transpose.
bool fused(int dtype, const FuseProgram& prog, Buffer* const* ins, const size_t* counts,
           const size_t* rows, const double* consts, Buffer* out, size_t n, std::string* err);
/// The same program's sum, minimum or maximum (`KRED_*`) over `n >= 1`
/// elements, folded on the device and finished on the host in double
/// precision, with the elements never stored.
bool fused_reduce(int op, int dtype, const FuseProgram& prog, Buffer* const* ins,
                  const size_t* counts, const size_t* rows, const double* consts, size_t n,
                  double* out, std::string* err);
/// `C = A x B` with a fused program applied to each element of C as it is
/// stored: the program starts from `FUSE_PRODUCT`, which is that element. One
/// kernel for `relu (w @ x + b)`, generated and cached as `fused`'s are.
bool fused_matmul(int dtype, const FuseProgram& prog, Operand a, Operand b, size_t M, size_t K,
                  size_t N, Buffer* const* ins, const size_t* counts, const size_t* rows,
                  const double* consts, Buffer* out, std::string* err);
/// Whether `matmul` cuts a product of these sizes along K ("split-K" at
/// `gpu::matmul`), which a product with a chain after it must then run as
/// the product and the chain separately.
bool splits_k(size_t M, size_t K, size_t N);
/// `C = A x B`, M x K by K x N, C row-major.
bool matmul(int dtype, Operand a, Operand b, Buffer* c, size_t M, size_t K, size_t N,
            std::string* err);
bool transpose(int dtype, Buffer* in, Buffer* out, size_t rows, size_t cols, std::string* err);

/// `batch` products at once: `C[b] = A[b] x B[b]`, each `M x K` by `K x N`,
/// the batches one after another in each buffer.
bool bmm(int dtype, Buffer* a, Buffer* b, Buffer* c, size_t batch, size_t M, size_t K, size_t N,
         std::string* err);
/// The numbers of `in` with their axes reordered: `out` has `rank` axes of
/// lengths `dims`, and its element at coordinates `i` is `in`'s at offset
/// `sum i[k] * strides[k]` -- the input's stride for whichever of its axes
/// became axis `k`.
bool permute(int dtype, Buffer* in, Buffer* out, size_t rank, const size_t* dims,
             const size_t* strides, size_t count, std::string* err);

/// The geometry of a sliding window over images `[n, h, w, c]`, channels
/// last: a `kh x kw` window moved `stride` at a time over the image padded
/// with `pad` zeros on every side, giving an `oh x ow` grid of windows.
struct Window {
    size_t n, h, w, c, kh, kw, stride, pad, oh, ow;
};
/// Every window's numbers as a row: `out` is `[n * oh * ow, kh * kw * c]`, a
/// row per window in image, row, column order, and in each row the window's
/// numbers in row, column, channel order. Padding reads as 0.
bool im2col(int dtype, const Window& g, Buffer* in, Buffer* out, std::string* err);
/// The reverse: each number of the images `[n, h, w, c]` is the sum of every
/// place in `cols` that `im2col` would have copied it to.
bool col2im(int dtype, const Window& g, Buffer* cols, Buffer* out, std::string* err);

}  // namespace dream::gpu
