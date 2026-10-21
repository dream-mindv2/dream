// Tensors: the operators, `@`, and `std.tensor`.
//
// The arrangement is a thin layer of checking over two backends. Every
// operation here reads its operands' shapes, decides the shape of the answer
// (or raises `:shape_error` naming both), allocates it, and hands raw pointers
// to the CPU kernels (tensor_kernels.inc) or buffer handles to the GPU's
// (gpu.cpp). Which backend runs is decided by where the operands are, never by
// the operation: a program moves a tensor to the GPU with `tensor.gpu` and
// everything it then does to it -- `+`, `@`, `tensor.sum` -- runs there, and
// `tensor.host` brings an answer back. Every operation has a GPU form; what
// reads numbers back to the host is only what answers in host terms -- a
// number, a list, a printed string, `==`.
//
// Elementwise work, products, transposes and reductions do not compute where
// they are called: they record a program and compute it, fused, when
// something reads the numbers. "fusion" below is the mechanism, and
// docs/notes/tensors.md the measurements.
//
// Nothing here forces anything but the arguments of `std.tensor` functions
// that take lists. The operators reach this from `arith`, which holds its
// operands in C++ locals (and JIT-compiled code, in registers): nothing below
// may collect, and nothing does -- `alloc` never collects, and the one walk
// that forces, `from_nested`, is a native's, which the machine never collects
// underneath.

#include "tensor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "bigint.hpp"
#include "builtins.hpp"
#include "gpu.hpp"
#include "heap.hpp"
#include "interp.hpp"
#include "process.hpp"
#include "tensor_kernels.hpp"

namespace dream {

// --- the CPU kernel table --------------------------------------------------------

const TensorKernels& tensor_kernels() {
    static const TensorKernels* chosen = [] {
#if defined(DREAM_TENSOR_AVX2)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            return &kernels_avx2::kernels;
#endif
        return &kernels_base::kernels;
    }();
    return *chosen;
}

std::vector<const TensorKernels*> tensor_kernel_tables() {
    std::vector<const TensorKernels*> tables = {&kernels_base::kernels};
#if defined(DREAM_TENSOR_AVX2)
    if (&tensor_kernels() == &kernels_avx2::kernels) tables.push_back(&kernels_avx2::kernels);
#endif
    return tables;
}

const char* tensor_kernels_name() {
#if defined(DREAM_TENSOR_AVX2)
    if (&tensor_kernels() == &kernels_avx2::kernels) return "avx2";
#endif
    return "baseline";
}

// --- what the heap needs ---------------------------------------------------------

void* tensor_handle(const TensorObj* t) {
    void* h;
    std::memcpy(&h, t->data(), sizeof h);
    return h;
}

void retain_external(Obj* o) {
    gpu::retain(static_cast<gpu::Buffer*>(tensor_handle(static_cast<TensorObj*>(o))));
}

void release_external(Obj* o) {
    gpu::release(static_cast<gpu::Buffer*>(tensor_handle(static_cast<TensorObj*>(o))));
}

size_t external_budget() {
    const size_t m = gpu::memory_bytes();
    return m ? m / 4 : std::numeric_limits<size_t>::max();
}

namespace {

constexpr uint8_t TENSOR_GPU = 1;

/// The most elements one tensor can hold: its whole size must fit the
/// header's 32-bit byte count, which is 4 GiB of doubles less the header.
constexpr uint64_t kMaxElements =
    (uint64_t(std::numeric_limits<uint32_t>::max()) - sizeof(TensorObj) - 64) / sizeof(double);

struct Shape {
    uint32_t rank = 0;
    uint32_t dims[TENSOR_MAX_RANK] = {};
    uint64_t count = 0;
};

Shape shape_of(const TensorObj* t) {
    Shape s;
    s.rank = t->rank;
    for (uint32_t i = 0; i < t->rank; ++i) s.dims[i] = t->dims[i];
    s.count = t->count;
    return s;
}

Shape shape2(uint32_t a, uint32_t b) {
    Shape s;
    s.rank = 2;
    s.dims[0] = a;
    s.dims[1] = b;
    s.count = uint64_t(a) * b;
    return s;
}

Shape shape1(uint32_t a) {
    Shape s;
    s.rank = 1;
    s.dims[0] = a;
    s.count = a;
    return s;
}

std::string shape_text(const Shape& s) {
    std::string out = "[";
    for (uint32_t i = 0; i < s.rank; ++i) {
        if (i) out += ", ";
        out += std::to_string(s.dims[i]);
    }
    return out + "]";
}

bool same_shape(const Shape& a, const Shape& b) {
    if (a.rank != b.rank) return false;
    for (uint32_t i = 0; i < a.rank; ++i)
        if (a.dims[i] != b.dims[i]) return false;
    return true;
}

TensorObj* tensor_of(Value v) {
    v = resolve(v);
    return is_tensor(v) ? static_cast<TensorObj*>(as_obj(v)) : nullptr;
}

bool on_gpu(const TensorObj* t) { return t->device == TENSOR_GPU; }
gpu::Buffer* buffer_of(const TensorObj* t) {
    return static_cast<gpu::Buffer*>(tensor_handle(t));
}

bool is_number(Value v) {
    return is_fixnum(v) || is_obj(v, ObjType::Float) || is_obj(v, ObjType::BigInt);
}

/// A deferred tensor computed, or a computed one as it is. See "fusion" below.
bool settle(Process& p, TensorObj** t, Value* err);
double number_of(Value v) {
    if (is_obj(v, ObjType::Float)) return static_cast<FloatObj*>(as_obj(v))->value;
    return bigint::to_double(v);
}

// --- raising -----------------------------------------------------------------------
//
// Every helper below that can fail answers false and leaves the error value
// in `*err`, which is the shape `arith` already has. The natives turn that into
// `NativeResult::raise`.

Value error_of(Process& p, const char* kind, const std::string& message) {
    return raise_error(p, p.runtime().intern_atom(kind), message);
}

bool fail(Process& p, Value* err, const char* kind, const std::string& message) {
    *err = error_of(p, kind, message);
    return false;
}

bool type_fail(Process& p, Value* err, const std::string& message) {
    *err = raise_error(p, well_known(p.runtime()).type_error, message);
    return false;
}

bool gpu_fail(Process& p, Value* err, const std::string& message) {
    return fail(p, err, "gpu_error", message);
}

// --- making tensors ----------------------------------------------------------------

/// A host tensor of shape `s`, its numbers for the caller to write.
bool new_host(Process& p, const Shape& s, Value* out, double** data) {
    if (s.count > kMaxElements)
        return fail(p, out, "out_of_memory",
                    "a tensor of shape " + shape_text(s) + " is larger than 4 GiB");
    Value v = p.heap().make_tensor(s.rank, s.dims, s.count);
    *data = static_cast<TensorObj*>(as_obj(v))->data();
    *out = v;
    return true;
}

/// A GPU tensor of shape `s` around a buffer the caller has already made and
/// filled (or enqueued the filling of); the tensor takes the reference.
Value wrap_gpu(Process& p, const Shape& s, int dtype, gpu::Buffer* b) {
    return p.heap().make_external_tensor(s.rank, s.dims, s.count, uint8_t(dtype), TENSOR_GPU, b);
}

/// A fresh, unfilled device buffer for `count` elements of `dtype`.
gpu::Buffer* gpu_alloc(Process& p, uint64_t count, int dtype, Value* err) {
    std::string why;
    gpu::Buffer* b = gpu::alloc(size_t(count) * gpu::dtype_size(dtype), &why);
    if (!b) gpu_fail(p, err, why);
    return b;
}

/// A GPU tensor's numbers, read back into `out`.
bool download(Process& p, const TensorObj* t, std::vector<double>& out, Value* err) {
    out.resize(size_t(t->count));
    std::string why;
    if (!gpu::download(buffer_of(t), t->dtype, out.data(), out.size(), &why))
        return gpu_fail(p, err, why);
    return true;
}

/// Any tensor's numbers, as a pointer to host memory: its own for a host
/// tensor, `scratch` filled from the device for a GPU one.
bool host_view(Process& p, TensorObj* t, std::vector<double>& scratch, const double** data,
               Value* err) {
    if (!settle(p, &t, err)) return false;
    if (!on_gpu(t)) {
        *data = t->data();
        return true;
    }
    if (!download(p, t, scratch, err)) return false;
    *data = scratch.data();
    return true;
}

/// Host numbers onto the GPU as a tensor of `dtype`.
bool upload(Process& p, const Shape& s, const double* data, int dtype, Value* out) {
    gpu::Buffer* b = gpu_alloc(p, s.count, dtype, out);
    if (!b) return false;
    std::string why;
    if (!gpu::upload(b, dtype, data, size_t(s.count), &why)) {
        gpu::release(b);
        return gpu_fail(p, out, why);
    }
    *out = wrap_gpu(p, s, dtype, b);
    return true;
}

// --- reading Dream data into a tensor ----------------------------------------------

/// The elements of a list or an array, forcing a list's spine. False with
/// `*is_seq` false when `v` is neither; false with an error when a force raised.
bool items_of(Process& p, Value v, std::vector<Value>& items, bool* is_seq, Value* err) {
    items.clear();
    *is_seq = true;
    v = resolve(v);
    if (is_obj(v, ObjType::Array)) {
        auto* a = static_cast<ArrayObj*>(as_obj(v));
        items.assign(a->items(), a->items() + a->len);
        return true;
    }
    if (!is_nil(v) && !is_obj(v, ObjType::Cons)) {
        *is_seq = false;
        return false;
    }
    Value cur = v;
    for (;;) {
        Value w;
        if (!force_whnf(p, cur, &w)) {
            *err = p.result;
            return false;
        }
        if (!is_obj(w, ObjType::Cons)) break;
        auto* c = static_cast<ConsObj*>(as_obj(w));
        items.push_back(c->head);
        cur = c->tail;
    }
    return true;
}

/// Fill `out` from nested lists or arrays of numbers whose shape is `s`,
/// checking that every level has the length the first element at that depth
/// had. Forcing allocates but never collects here (this is a native's work),
/// so the values gathered into `items` stay where they are.
bool fill_nested(Process& p, Value v, const Shape& s, uint32_t depth, double*& out, Value* err) {
    Value w;
    if (!force_whnf(p, v, &w)) {
        *err = p.result;
        return false;
    }
    if (depth == s.rank) {
        if (!is_number(w))
            return type_fail(p, err, "a tensor holds numbers, not " + describe(p, w));
        *out++ = number_of(w);
        return true;
    }
    std::vector<Value> items;
    bool is_seq;
    if (!items_of(p, w, items, &is_seq, err)) {
        if (is_seq) return false;
        return fail(p, err, "shape_error",
                    "the rows of a tensor must all have the same shape " + shape_text(s) +
                        ", but one has " + describe(p, w) + " where a list was expected");
    }
    if (items.size() != s.dims[depth])
        return fail(p, err, "shape_error",
                    "the rows of a tensor must all have the same length: expected " +
                        std::to_string(s.dims[depth]) + " at depth " + std::to_string(depth) +
                        ", found " + std::to_string(items.size()));
    for (Value item : items)
        if (!fill_nested(p, item, s, depth + 1, out, err)) return false;
    return true;
}

/// A host tensor from nested lists or arrays of numbers. The shape is read off
/// the first element at each depth and every other element is held to it.
bool from_nested(Process& p, Value v, Value* out) {
    Shape s;
    Value cur = v;
    std::vector<Value> items;
    for (;;) {
        Value w;
        if (!force_whnf(p, cur, &w)) {
            *out = p.result;
            return false;
        }
        if (is_number(w)) break;
        bool is_seq;
        if (!items_of(p, w, items, &is_seq, out)) {
            if (is_seq) return false;
            return type_fail(p, out, "a tensor is made from lists or arrays of numbers, not " +
                                         describe(p, w));
        }
        if (items.empty())
            return fail(p, out, "shape_error", "a tensor cannot be empty");
        if (s.rank == TENSOR_MAX_RANK)
            return fail(p, out, "shape_error",
                        "a tensor has at most " + std::to_string(TENSOR_MAX_RANK) + " axes");
        if (items.size() > std::numeric_limits<uint32_t>::max())
            return fail(p, out, "shape_error", "an axis this long does not fit a tensor");
        s.dims[s.rank++] = uint32_t(items.size());
        cur = items[0];
    }
    if (s.rank == 0)
        return type_fail(p, out, "a tensor is made from a list or an array of numbers, not a "
                                 "single number");
    s.count = 1;
    for (uint32_t i = 0; i < s.rank; ++i) {
        s.count *= s.dims[i];
        if (s.count > kMaxElements)
            return fail(p, out, "out_of_memory", "a tensor this large is larger than 4 GiB");
    }
    double* data;
    Value t;
    if (!new_host(p, s, &t, &data)) {
        *out = t;
        return false;
    }
    if (!fill_nested(p, v, s, 0, data, out)) return false;
    *out = t;
    return true;
}

/// A tensor, or lists and arrays that make one. What every `std.tensor`
/// function and `@` accept, so that `tensor.dot [1, 2] [3, 4]` needs no
/// conversion written out. The operators do *not* take lists -- `+` already
/// means something for two of them.
bool coerce(Process& p, Value v, TensorObj** t, Value* err) {
    Value w;
    if (!force_whnf(p, v, &w)) {
        *err = p.result;
        return false;
    }
    if (is_tensor(w)) {
        *t = static_cast<TensorObj*>(as_obj(w));
        return true;
    }
    Value made;
    if (!from_nested(p, w, &made)) {
        *err = made;
        return false;
    }
    *t = static_cast<TensorObj*>(as_obj(made));
    return true;
}

/// A shape written as a list or an array of positive integers.
bool read_shape(Process& p, Value v, Shape* s, Value* err) {
    std::vector<Value> items;
    bool is_seq;
    if (!items_of(p, v, items, &is_seq, err)) {
        if (is_seq) return false;
        return type_fail(p, err, "a shape is a list of integers, not " + describe(p, v));
    }
    if (items.empty() || items.size() > TENSOR_MAX_RANK)
        return fail(p, err, "shape_error",
                    "a shape has between 1 and " + std::to_string(TENSOR_MAX_RANK) + " axes");
    s->rank = uint32_t(items.size());
    s->count = 1;
    for (uint32_t i = 0; i < s->rank; ++i) {
        Value w;
        if (!force_whnf(p, items[i], &w)) {
            *err = p.result;
            return false;
        }
        if (!is_fixnum(w) || fixnum_value(w) < 1 ||
            fixnum_value(w) > int64_t(std::numeric_limits<uint32_t>::max()))
            return fail(p, err, "shape_error",
                        "an axis's length is a positive integer, not " + describe(p, w));
        s->dims[i] = uint32_t(fixnum_value(w));
        s->count *= s->dims[i];
        if (s->count > kMaxElements)
            return fail(p, err, "out_of_memory", "a tensor of that shape is larger than 4 GiB");
    }
    return true;
}

// --- the CPU's matrix product, in parallel when it is worth it -------------------
//
// Rows of C are independent, so a large product is split into bands of rows,
// one thread each, every thread running the whole blocked kernel on its band
// (and packing B for itself -- K x N against a band's M x K x N, which is
// nothing once a band is a few dozen rows). The threads are made for the call
// and joined before it returns: the product is a pure function of its inputs,
// and nothing the scheduler does can see it was parallel.
//
// Below about thirty million operations the threads cost more than they save.
// `DREAM_TENSOR_THREADS` caps them; 1 turns this off.

unsigned tensor_threads() {
#ifdef DREAM_BROWSER
    return 1;
#else
    static const unsigned n = [] {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (const char* e = std::getenv("DREAM_TENSOR_THREADS")) {
            long v = std::strtol(e, nullptr, 10);
            if (v >= 1) return std::min<unsigned>(hw, unsigned(v));
        }
        return std::min<unsigned>(hw, 16);
    }();
    return n;
#endif
}

/// A computed matrix as a product reads it: `cols` wide, stored row by row.
GemmOperand plain_operand(const double* data, size_t cols) {
    return GemmOperand{data, cols, 1, nullptr, nullptr, 0};
}

/// `C = A x B`, and `epi` (with `row_base` ignored) told of each finished
/// tile of C in whole-matrix rows. The tiles of different bands are disjoint,
/// so an epilogue that writes only its own tile may run on every band's
/// thread at once.
void host_gemm(const GemmOperand& A, const GemmOperand& B, double* C, size_t M, size_t K,
               size_t N, const GemmEpilogue* epi = nullptr) {
    const TensorKernels& k = tensor_kernels();
    const double work = 2.0 * double(M) * double(K) * double(N);
    size_t threads = work < 3e7 ? 1 : std::min<size_t>(tensor_threads(), M / 32);
    if (threads <= 1) {
        k.gemm(A, B, C, M, K, N, epi);
        return;
    }
    // Bands in multiples of the kernel's six rows, so no band ends in a
    // partial tile that another band would have filled.
    size_t band = (M + threads - 1) / threads;
    band = (band + 5) / 6 * 6;
    auto run = [&k, &A, &B, C, K, N, epi](size_t m0, size_t rows) {
        GemmEpilogue here;
        if (epi) here = GemmEpilogue{epi->fn, epi->ctx, m0};
        GemmOperand band = A;
        band.row0 += m0;
        k.gemm(band, B, C + m0 * N, rows, K, N, epi ? &here : nullptr);
    };
    std::vector<std::thread> pool;
    for (size_t m0 = band; m0 < M; m0 += band) pool.emplace_back(run, m0, std::min(band, M - m0));
    run(0, std::min(band, M));
    for (auto& t : pool) t.join();
}

// --- elementwise ---------------------------------------------------------------------

int kernel_op(Op op) {
    switch (op) {
        case Op::Add: return KOP_ADD;
        case Op::Sub: return KOP_SUB;
        case Op::Mul: return KOP_MUL;
        case Op::Div: return KOP_DIV;
        default: return KOP_MOD;
    }
}

/// The shape of `a op b`. Equal shapes, or one whose shape is the trailing
/// part of the other's -- a row added to every row of a matrix -- which is
/// numpy's broadcasting without its stretching of length-one axes.
bool broadcast(Process& p, const char* what, const Shape& a, const Shape& b, Shape* out,
               Value* err) {
    if (same_shape(a, b)) {
        *out = a;
        return true;
    }
    const Shape& big = a.rank >= b.rank ? a : b;
    const Shape& small = a.rank >= b.rank ? b : a;
    bool suffix = small.rank < big.rank;
    for (uint32_t i = 0; suffix && i < small.rank; ++i)
        suffix = small.dims[i] == big.dims[big.rank - small.rank + i];
    if (!suffix)
        return fail(p, err, "shape_error",
                    std::string("cannot apply `") + what + "` to tensors of shape " +
                        shape_text(a) + " and " + shape_text(b));
    *out = big;
    return true;
}

bool same_device(Process& p, const TensorObj* a, const TensorObj* b, const char* what,
                 Value* err) {
    if (a->device != b->device)
        return fail(p, err, "device_error",
                    std::string(what) + " needs both tensors in one place, but one is on the GPU "
                    "and the other is not: move one with `tensor.gpu` or `tensor.host`");
    if (on_gpu(a) && a->dtype != b->dtype)
        return fail(p, err, "device_error",
                    std::string(what) + " needs both GPU tensors to hold the same type: one is "
                    "float32 (`tensor.gpu`) and the other float64 (`tensor.gpu64`)");
    return true;
}

// --- fusion ------------------------------------------------------------------------
//
// The elementwise operators and functions do not compute. They record a
// program (`TensorExpr`, value.hpp) and answer a deferred tensor of the right
// shape; whatever needs the numbers computes the whole chain at once. Shapes
// and devices are checked as the program is built, so every error that
// depends on them is raised where it always was. What is deferred is only the
// arithmetic, and arithmetic cannot fail.
//
// A tensor smaller than `kFuseMin` is computed on the spot: fusing saves
// passes over memory, and a few thousand doubles are already in cache, where
// the object that records the program would cost more than the pass it saves.
// On the GPU everything is deferred, because there the saving is kernel
// launches, and a launch costs the same however small the tensor.

constexpr uint64_t kFuseMin = 4096;

/// A program being built: a `TensorExpr` before it is stored or run. Its
/// inputs are computed tensors, held by pointer -- nothing can move them while
/// a native or an operator is running, because nothing collects.
struct Program {
    TensorObj* inputs[kFuseMaxInputs];
    double consts[kFuseMaxConsts];
    uint8_t code[kFuseMaxCode * 2];
    unsigned ninputs = 0, nconsts = 0, ncode = 0;
    /// The matrix product the program starts from (`FUSE_PRODUCT`), or none.
    TensorObj* product_a = nullptr;
    TensorObj* product_b = nullptr;
    /// Past one of the limits. The caller computes an operand and starts over.
    bool overflow = false;
    /// Two different products met in one program. Only one can be the
    /// product the program is fused into; the caller computes the other.
    bool second_product = false;

    void emit(uint8_t op, unsigned arg) {
        if (ncode == kFuseMaxCode) {
            overflow = true;
            return;
        }
        code[2 * ncode] = op;
        code[2 * ncode + 1] = uint8_t(arg);
        ++ncode;
    }
    unsigned input(TensorObj* t) {
        for (unsigned i = 0; i < ninputs; ++i)
            if (inputs[i] == t) return i;
        if (ninputs == kFuseMaxInputs) {
            overflow = true;
            return 0;
        }
        inputs[ninputs] = t;
        return ninputs++;
    }
    unsigned constant(double c) {
        for (unsigned i = 0; i < nconsts; ++i)
            if (std::memcmp(&consts[i], &c, sizeof c) == 0) return i;
        if (nconsts == kFuseMaxConsts) {
            overflow = true;
            return 0;
        }
        consts[nconsts] = c;
        return nconsts++;
    }
    void product(TensorObj* a, TensorObj* b) {
        if (product_a && (product_a != a || product_b != b)) second_product = true;
        product_a = a;
        product_b = b;
        emit(FUSE_PRODUCT, 0);
    }
    unsigned depth() const {
        unsigned sp = 0, most = 0;
        for (unsigned i = 0; i < ncode; ++i) {
            switch (code[2 * i]) {
                case FUSE_LOAD: case FUSE_LOADT: case FUSE_LOADR: case FUSE_CONST:
                case FUSE_PRODUCT:
                    most = std::max(most, ++sp);
                    break;
                case FUSE_BIN: --sp; break;
                default: break;
            }
        }
        return most;
    }
    FuseProgram view() const { return FuseProgram{code, ncode, ninputs, nconsts, depth()}; }
};

TensorObj* computed_of(TensorObj* t) {
    if (!tensor_deferred(t)) return t;
    Value r = tensor_expr(t)->result;
    return r == NIL_SLOT ? nullptr : static_cast<TensorObj*>(as_obj(r));
}

/// Append what `v` computes: a computed tensor is a load, a number a
/// constant, and a deferred tensor its whole program, so that a chain is one
/// flat program however it was built.
void append(Program& prog, Value v) {
    v = resolve(v);
    if (is_number(v)) {
        prog.emit(FUSE_CONST, prog.constant(number_of(v)));
        return;
    }
    auto* t = static_cast<TensorObj*>(as_obj(v));
    if (TensorObj* done = computed_of(t)) {
        prog.emit(FUSE_LOAD, prog.input(done));
        return;
    }
    TensorExpr* e = tensor_expr(t);
    const uint8_t* c = e->code();
    for (unsigned i = 0; i < e->ncode; ++i) {
        unsigned arg = c[2 * i + 1];
        switch (c[2 * i]) {
            case FUSE_LOAD:
            case FUSE_LOADT:
                arg = prog.input(static_cast<TensorObj*>(as_obj(e->inputs()[arg])));
                break;
            case FUSE_CONST: arg = prog.constant(e->consts()[arg]); break;
            case FUSE_LOADR:
                arg = prog.input(static_cast<TensorObj*>(as_obj(e->inputs()[arg & 7]))) |
                      prog.constant(e->consts()[arg >> 3]) << 3;
                break;
            case FUSE_PRODUCT:
                prog.product(static_cast<TensorObj*>(as_obj(e->product_a)),
                             static_cast<TensorObj*>(as_obj(e->product_b)));
                continue;
            default: break;
        }
        prog.emit(c[2 * i], arg);
    }
}

/// The shape of `a @ b` as matrices: a vector on the left is a row, on the
/// right a column.
struct Gemm {
    size_t M, K, N;
};
Gemm gemm_of(const TensorObj* a, const TensorObj* b) {
    return Gemm{a->rank == 2 ? a->dims[0] : 1u, a->rank == 2 ? a->dims[1] : a->dims[0],
                b->rank == 2 ? b->dims[1] : 1u};
}

/// The program's value at the numbers `[base, base + len)` of an answer of
/// `n`, a block at a time small enough that every stack slot stays in L1.
///
/// The stack holds pointers, not copies: a load of an input as long as the
/// answer points straight into it, a constant is carried as a scalar and meets
/// a vector through the scalar kernel, and only a broadcast load or an
/// operation's result fills a buffer. Each stack position has two buffers so
/// that an operation never writes the buffer it is reading -- the kernels take
/// `__restrict` pointers, and an aliased one would be undefined. The last
/// instruction writes the caller's memory directly when it can.
constexpr size_t kFuseBlock = 256;

struct Slot {
    const double* p;
    double s;
    bool scalar;
};

/// Scratch for one thread of evaluation: two blocks per stack position.
struct Scratch {
    std::vector<double> mem;
    explicit Scratch(unsigned depth) : mem(size_t(depth) * 2 * kFuseBlock) {}
    void ensure(unsigned depth) {
        if (mem.size() < size_t(depth) * 2 * kFuseBlock) mem.resize(size_t(depth) * 2 * kFuseBlock);
    }
    double* at(unsigned pos, unsigned which) { return mem.data() + (pos * 2 + which) * kFuseBlock; }
};

/// One block. Answers the pointer the block's numbers are at: `out` when the
/// last instruction could write there, otherwise a scratch buffer.
///
/// `product` is the whole computed product a `FUSE_PRODUCT` reads, or null.
/// An epilogue computes in place -- `out` is the product's own block -- and
/// then the last instruction must not write `out` directly, since its operands
/// may still point into it; the caller copies the answer over afterwards.
const double* run_block(const FuseProgram& prog, TensorObj* const* inputs, const double* consts,
                        const double* product, size_t base, size_t len, Scratch& scratch,
                        double* out) {
    const TensorKernels& k = tensor_kernels();
    const bool direct = !(product && out == product + base);
    Slot stack[kFuseMaxDepth];
    unsigned sp = 0;
    for (unsigned i = 0; i < prog.ncode; ++i) {
        const unsigned op = prog.code[2 * i], arg = prog.code[2 * i + 1];
        const bool last = direct && i + 1 == prog.ncode;
        switch (op) {
            case FUSE_PRODUCT: stack[sp++] = Slot{product + base, 0, false}; break;
            case FUSE_LOADT: {
                // An R x C input read as its C x R transpose: answer position
                // `o` is row `o / R`, column `o % R` of the transpose, which
                // is `in[(o % R) * C + o / R]`. Walked with counters rather
                // than a division per element.
                const TensorObj* in = inputs[arg];
                const size_t R = in->dims[0], C = in->dims[1], m = R * C;
                const double* src = in->data();
                double* buf = scratch.at(sp, 0);
                size_t o = base % m, r = o % R, c = o / R;
                for (size_t j = 0; j < len; ++j) {
                    buf[j] = src[r * C + c];
                    if (++r == R) {
                        r = 0;
                        if (++c == C) c = 0;
                    }
                }
                stack[sp++] = Slot{buf, 0, false};
                break;
            }
            case FUSE_LOADR: {
                // Each number of the input `r` times over, the whole input
                // repeating every `r * m`: position `o` reads `(o / r) % m`.
                const TensorObj* in = inputs[arg & 7];
                const size_t m = size_t(in->count), r = size_t(consts[arg >> 3]);
                const double* src = in->data();
                double* buf = scratch.at(sp, 0);
                size_t at = (base / r) % m, rem = base % r;
                for (size_t j = 0; j < len; ++j) {
                    buf[j] = src[at];
                    if (++rem == r) {
                        rem = 0;
                        if (++at == m) at = 0;
                    }
                }
                stack[sp++] = Slot{buf, 0, false};
                break;
            }
            case FUSE_LOAD: {
                const TensorObj* in = inputs[arg];
                const size_t m = size_t(in->count);
                // Within the input's first `m` numbers -- all of it for an
                // input as long as the answer -- index `i` is `i % m`.
                if (base + len <= m) {
                    stack[sp++] = Slot{in->data() + base, 0, false};
                    break;
                }
                // A broadcast input: its numbers repeat every `m`, so copy
                // the block in runs that wrap at the end of the input.
                double* buf = scratch.at(sp, 0);
                size_t at = base % m;
                for (size_t done = 0; done < len;) {
                    size_t run = std::min(len - done, m - at);
                    std::memcpy(buf + done, in->data() + at, run * sizeof(double));
                    done += run;
                    at = 0;
                }
                stack[sp++] = Slot{buf, 0, false};
                break;
            }
            case FUSE_CONST: stack[sp++] = Slot{nullptr, consts[arg], true}; break;
            case FUSE_BIN: {
                Slot y = stack[--sp];
                Slot x = stack[sp - 1];
                if (x.scalar && y.scalar) {
                    double a = x.s, b = y.s, r;
                    k.scalar(int(arg), &a, b, false, &r, 1);
                    stack[sp - 1] = Slot{nullptr, r, true};
                    break;
                }
                double* dst = last ? out : scratch.at(sp - 1, scratch.at(sp - 1, 0) == x.p ? 1 : 0);
                if (x.scalar) k.scalar(int(arg), y.p, x.s, true, dst, len);
                else if (y.scalar) k.scalar(int(arg), x.p, y.s, false, dst, len);
                else k.binary(int(arg), x.p, y.p, dst, len);
                stack[sp - 1] = Slot{dst, 0, false};
                break;
            }
            default: {  // FUSE_UN
                Slot x = stack[sp - 1];
                if (x.scalar) {
                    double r;
                    k.unary(int(arg), &x.s, &r, 1);
                    stack[sp - 1] = Slot{nullptr, r, true};
                    break;
                }
                double* dst = last ? out : scratch.at(sp - 1, scratch.at(sp - 1, 0) == x.p ? 1 : 0);
                k.unary(int(arg), x.p, dst, len);
                stack[sp - 1] = Slot{dst, 0, false};
                break;
            }
        }
    }
    // A program whose answer is a scalar or a bare load cannot have written
    // `out`; a constant fills it.
    if (stack[0].scalar) {
        std::fill(out, out + len, stack[0].s);
        return out;
    }
    return stack[0].p;
}

/// Run `body(begin, end)` over `[0, n)`, split across threads when `n` is
/// large enough that the threads pay for themselves. Elementwise work is
/// bound by memory, not arithmetic, so the line is higher than the product's.
///
/// `work` is how many numbers the whole range touches, when that is not `n`
/// -- a sum along an axis does `n` outputs' worth of `len` each -- and
/// `align` the granule a chunk is a multiple of.
template <class Body>
void parallel_range(size_t n, Body body, size_t work = 0, size_t align = kFuseBlock) {
    if (work == 0) work = n;
    const size_t threads = work < (size_t(1) << 18)
                               ? 1
                               : std::min<size_t>({size_t(tensor_threads()), work >> 16,
                                                   (n + align - 1) / align});
    if (threads <= 1) {
        body(size_t(0), n, size_t(0));
        return;
    }
    size_t chunk = (n + threads - 1) / threads;
    chunk = (chunk + align - 1) / align * align;
    std::vector<std::thread> pool;
    size_t index = 1;
    for (size_t b = chunk; b < n; b += chunk, ++index)
        pool.emplace_back([&body, b, chunk, n, index] { body(b, std::min(n, b + chunk), index); });
    body(size_t(0), std::min(n, chunk), size_t(0));
    for (auto& t : pool) t.join();
}

void run_host(const FuseProgram& prog, TensorObj* const* inputs, const double* consts,
              double* out, size_t n) {
    parallel_range(n, [&](size_t begin, size_t end, size_t) {
        Scratch scratch(std::max(prog.depth, 1u));
        for (size_t base = begin; base < end; base += kFuseBlock) {
            const size_t len = std::min(kFuseBlock, end - base);
            const double* r = run_block(prog, inputs, consts, nullptr, base, len, scratch, out + base);
            if (r != out + base) std::memcpy(out + base, r, len * sizeof(double));
        }
    });
}

/// What the product's epilogue needs: the program, and the whole C it rewrites
/// in place, tile by tile, as the product finishes them.
struct EpilogueRun {
    const FuseProgram* prog;
    TensorObj* const* inputs;
    const double* consts;
    double* C;
    size_t N;
};

/// Run the program over one finished tile of C, a row at a time -- a tile is
/// not contiguous, but each of its rows is, and the program indexes by the
/// position in the whole answer, so a row is just a range.
void epilogue_tile(void* ctx, size_t row0, size_t row1, size_t col0, size_t col1) {
    auto* e = static_cast<EpilogueRun*>(ctx);
    Scratch scratch(std::max(e->prog->depth, 1u));
    for (size_t r = row0; r < row1; ++r) {
        const size_t end = r * e->N + col1;
        for (size_t base = r * e->N + col0; base < end; base += kFuseBlock) {
            const size_t len = std::min(kFuseBlock, end - base);
            double* out = e->C + base;
            const double* got = run_block(*e->prog, e->inputs, e->consts, e->C, base, len, scratch, out);
            if (got != out) std::memcpy(out, got, len * sizeof(double));
        }
    }
}

/// The program's sum, minimum or maximum, block by block, with nothing stored.
double reduce_host(int op, const FuseProgram& prog, TensorObj* const* inputs,
                   const double* consts, size_t n) {
    const TensorKernels& k = tensor_kernels();
    auto combine = [op](double a, double b) {
        switch (op) {
            case KRED_SUM: return a + b;
            case KRED_MIN: return b < a ? b : a;
            default: return b > a ? b : a;
        }
    };
    // One partial per chunk `parallel_range` makes, each from a non-empty
    // range; `seen` marks the ones that ran.
    std::vector<double> partial(tensor_threads() + 1);
    std::vector<char> seen(partial.size(), 0);
    parallel_range(n, [&](size_t begin, size_t end, size_t index) {
        Scratch scratch(std::max(prog.depth, 1u));
        std::vector<double> block(kFuseBlock);
        double acc = 0;
        for (size_t base = begin; base < end; base += kFuseBlock) {
            const size_t len = std::min(kFuseBlock, end - base);
            const double* r = run_block(prog, inputs, consts, nullptr, base, len, scratch, block.data());
            const double part = k.reduce(op, r, len);
            acc = base == begin ? part : combine(acc, part);
        }
        partial[index] = acc;
        seen[index] = 1;
    });
    double r = partial[0];
    for (size_t i = 1; i < partial.size(); ++i)
        if (seen[i]) r = combine(r, partial[i]);
    return r;
}

/// The device buffers a program reads, with each one's length and, for a
/// matrix, its rows (which a transposed load needs).
void buffers_of(const Program& prog, gpu::Buffer** out, size_t* counts, size_t* rows) {
    for (unsigned i = 0; i < prog.ninputs; ++i) {
        const TensorObj* t = prog.inputs[i];
        out[i] = buffer_of(t);
        counts[i] = size_t(t->count);
        rows[i] = t->rank == 2 ? t->dims[0] : size_t(t->count);
    }
}

/// How a product reads one of its operands, worked out once per product and
/// kept alive for the length of it. A computed tensor is read where it lies; a
/// transpose (`[LOADT k]`) is read where its source lies, by strides; any
/// other deferred tensor is computed a row segment at a time, straight into
/// the panel the product packs (`operand_segment`), and never stored.
struct OperandPlan {
    GemmOperand host{};
    gpu::Operand device{};
    Program prog;
    FuseProgram view{};
    size_t cols = 0;
    // A deferred operand on the GPU: its program and buffers, for the
    // generated product kernel to compute as it loads.
    gpu::Buffer* ins[kFuseMaxInputs];
    size_t counts[kFuseMaxInputs], rows[kFuseMaxInputs];
    gpu::OperandProgram on_device{};
};

/// `len` numbers of a deferred operand's row `row` from column `col`. Called
/// by the product's threads at once, so the scratch is each thread's own.
void operand_segment(void* ctx, size_t row, size_t col, size_t len, double* out) {
    auto* plan = static_cast<OperandPlan*>(ctx);
    thread_local Scratch scratch(1);
    scratch.ensure(std::max(plan->view.depth, 1u));
    const size_t base = row * plan->cols + col;
    for (size_t done = 0; done < len; done += kFuseBlock) {
        const size_t n = std::min(kFuseBlock, len - done);
        const double* r = run_block(plan->view, plan->prog.inputs, plan->prog.consts, nullptr,
                                    base + done, n, scratch, out + done);
        if (r != out + done) std::memcpy(out + done, r, n * sizeof(double));
    }
}

bool is_transpose(const Program& prog) {
    return !prog.product_a && prog.ncode == 1 && prog.code[0] == FUSE_LOADT;
}

/// The plan for operand `t`, which the product sees as `cols` wide.
void plan_operand(TensorObj* t, size_t cols, OperandPlan& plan) {
    if (TensorObj* done = computed_of(t)) {
        if (on_gpu(done)) plan.device = gpu::Operand{buffer_of(done), cols, 1};
        else plan.host = plain_operand(done->data(), cols);
        return;
    }
    append(plan.prog, from_obj(t));
    if (!plan.prog.product_a && plan.prog.ncode == 1 && plan.prog.code[0] == FUSE_LOAD) {
        // A reshape of a computed tensor: its numbers are the operand's, in
        // the operand's order.
        TensorObj* src = plan.prog.inputs[0];
        if (on_gpu(src)) plan.device = gpu::Operand{buffer_of(src), cols, 1};
        else plan.host = plain_operand(src->data(), cols);
        return;
    }
    if (is_transpose(plan.prog)) {
        // Element (i, j) of the transpose of an R x C source is the source's
        // (j, i): one step along a row of the transpose is a whole row of the
        // source.
        TensorObj* src = plan.prog.inputs[0];
        const size_t C = src->dims[1];
        if (on_gpu(src)) plan.device = gpu::Operand{buffer_of(src), 1, C};
        else plan.host = GemmOperand{src->data(), 1, C, nullptr, nullptr, 0};
        return;
    }
    plan.view = plan.prog.view();
    plan.cols = cols;
    if (t->device == TENSOR_GPU) {
        buffers_of(plan.prog, plan.ins, plan.counts, plan.rows);
        plan.on_device = gpu::OperandProgram{plan.view, plan.ins, plan.counts, plan.rows,
                                             plan.prog.consts};
        plan.device = gpu::Operand{nullptr, 0, 0, &plan.on_device};
        return;
    }
    plan.host = GemmOperand{nullptr, 0, 0, operand_segment, &plan, 0};
}

bool run_program(Process& p, const Program& prog, const Shape& s, uint8_t device, uint8_t dtype,
                 Value* out);

/// A GPU product cut along K (see "split-K" at `gpu::matmul`), with the
/// chain after it, if any, run as a second pass. The generated product
/// kernel applies a chain to each element of C as it stores it, which a
/// split product cannot: no slice holds a finished element. So the product
/// is computed on its own -- its operands computed first if they are
/// programs, since the split kernel reads buffers -- and the chain then runs
/// as an ordinary fused program that loads it. One extra pass over C, on
/// products whose C is small by construction.
bool run_split_product(Process& p, const Program& prog, const Shape& s, uint8_t dtype, Value* out) {
    TensorObj* a = prog.product_a;
    TensorObj* b = prog.product_b;
    const Gemm g = gemm_of(a, b);
    OperandPlan pa, pb;
    plan_operand(a, g.K, pa);
    if (pa.device.program) {
        if (!settle(p, &a, out)) return false;
        pa = OperandPlan{};
        plan_operand(a, g.K, pa);
    }
    plan_operand(b, g.N, pb);
    if (pb.device.program) {
        if (!settle(p, &b, out)) return false;
        pb = OperandPlan{};
        plan_operand(b, g.N, pb);
    }
    const Shape ps = g.N == 1 ? shape1(uint32_t(g.M)) : shape2(uint32_t(g.M), uint32_t(g.N));
    gpu::Buffer* c = gpu_alloc(p, uint64_t(g.M) * g.N, dtype, out);
    if (!c) return false;
    std::string why;
    if (!gpu::matmul(dtype, pa.device, pb.device, c, g.M, g.K, g.N, &why)) {
        gpu::release(c);
        return gpu_fail(p, out, why);
    }
    Value product = wrap_gpu(p, ps, dtype, c);
    if (prog.ncode == 1) {
        // A bare product: the answer is the product, in the answer's shape.
        if (same_shape(ps, s)) {
            *out = product;
            return true;
        }
        gpu::retain(c);
        *out = wrap_gpu(p, s, dtype, c);
        return true;
    }
    Program rest = prog;
    rest.product_a = rest.product_b = nullptr;
    const unsigned k = rest.input(tensor_of(product));
    if (rest.overflow) return fail(p, out, "shape_error", "a tensor expression too large to compute");
    for (unsigned i = 0; i < rest.ncode; ++i)
        if (rest.code[2 * i] == FUSE_PRODUCT) {
            rest.code[2 * i] = FUSE_LOAD;
            rest.code[2 * i + 1] = uint8_t(k);
        }
    return run_program(p, rest, s, TENSOR_GPU, dtype, out);
}

/// Compute a program into a new tensor of shape `s` where its inputs are.
///
/// A program that starts from a product is the product with an epilogue: the
/// blocked kernel computes C, and the rest of the program rewrites each tile
/// of it as the kernel finishes the tile (`epilogue_tile`), or, on a GPU, as
/// the generated product kernel stores each element. A program that is the
/// product and nothing more has no epilogue to run.
bool run_program(Process& p, const Program& prog, const Shape& s, uint8_t device, uint8_t dtype,
                 Value* out) {
    const FuseProgram view = prog.view();
    const bool bare_product = prog.product_a && prog.ncode == 1;
    // A bare transpose is the blocked transpose kernel, which moves memory a
    // cache line at a time both ways rather than gathering a column at a time.
    const bool bare_transpose = is_transpose(prog);
    OperandPlan pa, pb;
    Gemm g{};
    if (prog.product_a) {
        g = gemm_of(prog.product_a, prog.product_b);
        plan_operand(prog.product_a, g.K, pa);
        plan_operand(prog.product_b, g.N, pb);
    }
    if (device == TENSOR_GPU && prog.product_a && gpu::splits_k(g.M, g.K, g.N))
        return run_split_product(p, prog, s, dtype, out);
    if (device == TENSOR_GPU) {
        gpu::Buffer* b = gpu_alloc(p, s.count, dtype, out);
        if (!b) return false;
        gpu::Buffer* ins[kFuseMaxInputs];
        size_t counts[kFuseMaxInputs], rows[kFuseMaxInputs];
        buffers_of(prog, ins, counts, rows);
        std::string why;
        bool ok;
        if (prog.product_a) {
            // The fixed product kernel reads buffers; an operand that is a
            // program needs a generated one, even with nothing after it.
            const bool computed_operand = pa.device.program || pb.device.program;
            ok = bare_product && !computed_operand
                     ? gpu::matmul(dtype, pa.device, pb.device, b, g.M, g.K, g.N, &why)
                     : gpu::fused_matmul(dtype, view, pa.device, pb.device, g.M, g.K, g.N, ins,
                                         counts, rows, prog.consts, b, &why);
        } else if (bare_transpose) {
            ok = gpu::transpose(dtype, ins[0], b, prog.inputs[0]->dims[0], prog.inputs[0]->dims[1],
                                &why);
        } else {
            ok = gpu::fused(dtype, view, ins, counts, rows, prog.consts, b, size_t(s.count), &why);
        }
        if (!ok) {
            gpu::release(b);
            return gpu_fail(p, out, why);
        }
        *out = wrap_gpu(p, s, dtype, b);
        return true;
    }
    double* data;
    Value t;
    if (!new_host(p, s, &t, &data)) {
        *out = t;
        return false;
    }
    if (prog.product_a) {
        EpilogueRun run{&view, prog.inputs, prog.consts, data, g.N};
        GemmEpilogue epi{epilogue_tile, &run, 0};
        host_gemm(pa.host, pb.host, data, g.M, g.K, g.N, bare_product ? nullptr : &epi);
    } else if (bare_transpose) {
        const TensorObj* src = prog.inputs[0];
        tensor_kernels().transpose(src->data(), data, src->dims[0], src->dims[1]);
    } else {
        run_host(view, prog.inputs, prog.consts, data, size_t(s.count));
    }
    *out = t;
    return true;
}

/// Store a program as a deferred tensor of shape `s`.
Value defer(Process& p, const Program& prog, const Shape& s, uint8_t device, uint8_t dtype) {
    Value v = p.heap().make_deferred_tensor(s.rank, s.dims, s.count, dtype, device, prog.ninputs,
                                            prog.nconsts, prog.ncode);
    TensorExpr* e = tensor_expr(static_cast<TensorObj*>(as_obj(v)));
    e->depth = uint16_t(prog.depth());
    for (unsigned i = 0; i < prog.ninputs; ++i) e->inputs()[i] = from_obj(prog.inputs[i]);
    if (prog.product_a) {
        e->product_a = from_obj(prog.product_a);
        e->product_b = from_obj(prog.product_b);
    }
    std::memcpy(e->consts(), prog.consts, prog.nconsts * sizeof(double));
    std::memcpy(e->code(), prog.code, prog.ncode * 2);
    return v;
}

/// The tensor with its numbers in hand: itself, or the answer a deferred one
/// stands for -- computed now, kept in its `result`, and its inputs let go,
/// so that the next read finds the answer and the inputs can die.
bool settle(Process& p, TensorObj** t, Value* err) {
    TensorObj* x = *t;
    if (TensorObj* done = computed_of(x)) {
        *t = done;
        return true;
    }
    Program prog;
    append(prog, from_obj(x));
    Value r;
    if (!run_program(p, prog, shape_of(x), x->device, x->dtype, &r)) {
        *err = r;
        return false;
    }
    TensorExpr* e = tensor_expr(x);
    value_slot_store(&e->result, r);
    p.heap().remember_if_old(x, r);
    for (unsigned i = 0; i < e->ninputs; ++i) value_slot_store(&e->inputs()[i], UNIT);
    if (e->product_a != NIL_SLOT) {
        value_slot_store(&e->product_a, UNIT);
        value_slot_store(&e->product_b, UNIT);
    }
    *t = static_cast<TensorObj*>(as_obj(r));
    return true;
}

/// Whether `t` is a deferred tensor not yet computed whose program starts
/// from a product.
bool pending_product(TensorObj* t) {
    return t && !computed_of(t) && tensor_expr(t)->product_a != NIL_SLOT;
}

/// The program for `instr` applied to `operands` (one for a function, two for
/// an operator), each a tensor or a number, for an answer of `count` numbers.
bool build(Process& p, const Value* operands, unsigned n, uint8_t instr, uint8_t arg,
           uint64_t count, Program* prog, Value* err) {
    Value ops[2] = {operands[0], n > 1 ? operands[1] : UNIT};
    // When the program will not do, one operand is computed and the program
    // built again from its answer:
    //
    // - Past a limit, the chain is long enough that one more pass over memory
    //   is nothing beside it, so the longer operand.
    // - With two different products, or a product shorter than the answer
    //   (`v + w @ x` where `v` is a matrix and the product a row, repeated
    //   down it), the operand holding the product. A product is fused by
    //   running the program as each tile of it is finished, which needs the
    //   answer and the product to be the same numbers in the same places.
    //
    // A few rounds at most: with both operands computed the program is two
    // loads and an operation.
    for (int round = 0;; ++round) {
        *prog = Program{};
        for (unsigned i = 0; i < n; ++i) append(*prog, ops[i]);
        prog->emit(instr, arg);
        bool product_fits = true;
        if (prog->product_a) {
            const Gemm g = gemm_of(prog->product_a, prog->product_b);
            product_fits = !prog->second_product && uint64_t(g.M) * g.N == count;
        }
        if (!prog->overflow && product_fits && prog->depth() <= kFuseMaxDepth) return true;
        unsigned pick = n;
        if (!product_fits) {
            for (unsigned i = 0; i < n; ++i)
                if (pending_product(tensor_of(ops[i]))) pick = i;
        } else {
            uint16_t longest = 0;
            for (unsigned i = 0; i < n; ++i) {
                TensorObj* t = tensor_of(ops[i]);
                if (t && !computed_of(t) && tensor_expr(t)->ncode >= longest) {
                    longest = tensor_expr(t)->ncode;
                    pick = i;
                }
            }
        }
        if (pick == n || round > 3)
            return fail(p, err, "shape_error", "a tensor expression too large to compute");
        TensorObj* t = tensor_of(ops[pick]);
        if (!settle(p, &t, err)) return false;
        ops[pick] = from_obj(t);
    }
}

/// `instr arg` over `operands`, to an answer of shape `s` where `like` is:
/// deferred, or computed now when it is small and on the host.
bool elementwise(Process& p, const Value* operands, unsigned n, uint8_t instr, uint8_t arg,
                 const Shape& s, const TensorObj* like, Value* out) {
    Program prog;
    if (!build(p, operands, n, instr, arg, s.count, &prog, out)) return false;
    if (like->device == TENSOR_HOST && s.count < kFuseMin)
        return run_program(p, prog, s, like->device, like->dtype, out);
    *out = defer(p, prog, s, like->device, like->dtype);
    return true;
}

bool map_unary(Process& p, int fn, TensorObj* x, Value* out) {
    Value v = from_obj(x);
    return elementwise(p, &v, 1, FUSE_UN, uint8_t(fn), shape_of(x), x, out);
}

/// The sum, minimum or maximum of every number `x` stands for. A deferred `x`
/// is reduced as it is computed, block by block, and never stored.
bool reduce(Process& p, int op, TensorObj* x, double* out, Value* err) {
    // A product is computed, epilogue and all, and kept, and then folded.
    // Folding each tile as the product finishes it was measured and not kept:
    // see "Finishing fusion" in docs/notes/tensors.md.
    if (pending_product(x) && !settle(p, &x, err)) return false;
    Program prog;
    append(prog, from_obj(x));
    const FuseProgram view = prog.view();
    const size_t n = size_t(x->count);
    if (x->device == TENSOR_GPU) {
        gpu::Buffer* ins[kFuseMaxInputs];
        size_t counts[kFuseMaxInputs], rows[kFuseMaxInputs];
        buffers_of(prog, ins, counts, rows);
        std::string why;
        if (!gpu::fused_reduce(op, x->dtype, view, ins, counts, rows, prog.consts, n, out, &why))
            return gpu_fail(p, err, why);
        return true;
    }
    if (TensorObj* done = computed_of(x)) {
        *out = tensor_kernels().reduce(op, done->data(), n);
        return true;
    }
    *out = reduce_host(op, view, prog.inputs, prog.consts, n);
    return true;
}

/// How a reduction along an axis folds one more value in, for `KRED_SUM`,
/// `KRED_MIN` and `KRED_MAX` and their `ARG` forms. `acc` is the running
/// answer and `best` the value it stands for (the same slot for the plain
/// forms; a separate one for the `ARG` forms, whose answer is a position).
/// `a` is the position along the axis; the first one starts the fold, so a
/// minimum or maximum never begins from a made-up identity. Ties keep the
/// first position, which is what every argmax does.
struct AxisFold {
    int op;
    void start(double* acc, double* best, double v) const {
        if (op >= KRED_ARGMIN) {
            *best = v;
            *acc = 0;
        } else {
            *acc = v;
        }
    }
    void add(double* acc, double* best, double v, size_t a) const {
        switch (op) {
            case KRED_SUM: *acc += v; break;
            case KRED_MIN: *acc = v < *acc ? v : *acc; break;
            case KRED_MAX: *acc = v > *acc ? v : *acc; break;
            case KRED_ARGMIN:
                if (v < *best) {
                    *best = v;
                    *acc = double(a);
                }
                break;
            default:
                if (v > *best) {
                    *best = v;
                    *acc = double(a);
                }
                break;
        }
    }
};

/// A fold along an axis on the host, the operand's program computed as it is
/// folded and never stored. `out` is `outer x inner`; source position
/// `(o * len + a) * inner + i` folds into `out[o * inner + i]`, in the order of
/// `a`, which is what makes the answer the same at any thread count.
///
/// Two ways through, by the width of what follows the axis. A wide one folds
/// a segment of a row per step along the axis, the threads dividing the
/// outputs. A narrow one -- the last axis, `inner` 1 -- scans each output's
/// whole contiguous range once, a block at a time, the threads dividing the
/// rows; folding segments of one number would be a block evaluation per
/// number.
void axis_host(int op, const FuseProgram& prog, TensorObj* const* inputs, const double* consts,
               double* out, size_t outer, size_t len, size_t inner) {
    const size_t work = outer * len * inner;
    const AxisFold fold{op};
    const bool arg = op >= KRED_ARGMIN;
    if (inner >= kFuseBlock) {
        parallel_range(outer * inner, [&](size_t begin, size_t end, size_t) {
            Scratch scratch(std::max(prog.depth, 1u));
            std::vector<double> block(kFuseBlock), best(arg ? end - begin : 0);
            for (size_t u = begin; u < end;) {
                const size_t o = u / inner, i0 = u % inner;
                const size_t piece = std::min(end - u, inner - i0);
                double* acc = out + u;
                double* b = arg ? best.data() + (u - begin) : acc;
                for (size_t a = 0; a < len; ++a) {
                    const size_t base = (o * len + a) * inner + i0;
                    for (size_t done = 0; done < piece; done += kFuseBlock) {
                        const size_t l = std::min(kFuseBlock, piece - done);
                        const double* r = run_block(prog, inputs, consts, nullptr, base + done, l,
                                                    scratch, block.data());
                        if (op == KRED_SUM) {
                            for (size_t j = 0; j < l; ++j) acc[done + j] += r[j];
                        } else if (a == 0) {
                            for (size_t j = 0; j < l; ++j)
                                fold.start(acc + done + j, b + done + j, r[j]);
                        } else {
                            for (size_t j = 0; j < l; ++j)
                                fold.add(acc + done + j, b + done + j, r[j], a);
                        }
                    }
                }
                u += piece;
            }
        }, work);
        return;
    }
    parallel_range(outer, [&](size_t begin, size_t end, size_t) {
        Scratch scratch(std::max(prog.depth, 1u));
        std::vector<double> block(kFuseBlock), best(arg ? inner : 0);
        for (size_t o = begin; o < end; ++o) {
            double* acc = out + o * inner;
            double* b = arg ? best.data() : acc;
            const size_t start = o * len * inner, stop = start + len * inner;
            for (size_t base = start; base < stop; base += kFuseBlock) {
                const size_t l = std::min(kFuseBlock, stop - base);
                const double* r = run_block(prog, inputs, consts, nullptr, base, l, scratch,
                                            block.data());
                if (op == KRED_SUM) {
                    for (size_t j = 0; j < l; ++j) acc[(base - start + j) % inner] += r[j];
                    continue;
                }
                for (size_t j = 0; j < l; ++j) {
                    const size_t at = base - start + j, i = at % inner, a = at / inner;
                    if (a == 0) fold.start(acc + i, b + i, r[j]);
                    else fold.add(acc + i, b + i, r[j], a);
                }
            }
        }
    }, work, 1);
}

/// `t` folded along `axis` (of a tensor of two or more axes) by `op`, a
/// `KRED_*`; the axis drops out of the shape. Fused: the program of a
/// deferred `t` runs as it is folded, on the host or in a generated GPU
/// kernel, and nothing but the answers is stored. A product is computed
/// first, as for any reduction.
bool reduce_axis(Process& p, int op, TensorObj* t, uint32_t axis, Value* out) {
    if (pending_product(t) && !settle(p, &t, out)) return false;
    Program prog;
    append(prog, from_obj(t));
    const FuseProgram view = prog.view();
    Shape s;
    size_t outer = 1, inner = 1;
    for (uint32_t i = 0; i < t->rank; ++i) {
        if (i < axis) outer *= t->dims[i];
        if (i > axis) inner *= t->dims[i];
        if (i != axis) s.dims[s.rank++] = t->dims[i];
    }
    s.count = uint64_t(outer) * inner;
    const size_t len = t->dims[axis];
    if (on_gpu(t)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, t->dtype, out);
        if (!b) return false;
        gpu::Buffer* ins[kFuseMaxInputs];
        size_t counts[kFuseMaxInputs], rows[kFuseMaxInputs];
        buffers_of(prog, ins, counts, rows);
        std::string why;
        if (!gpu::fused_axis(op, t->dtype, view, ins, counts, rows, prog.consts, outer, len, inner,
                             b, &why)) {
            gpu::release(b);
            return gpu_fail(p, out, why);
        }
        *out = wrap_gpu(p, s, t->dtype, b);
        return true;
    }
    double* data;
    if (!new_host(p, s, out, &data)) return false;
    std::fill(data, data + s.count, 0.0);
    axis_host(op, view, prog.inputs, prog.consts, data, outer, len, inner);
    return true;
}

// --- products ----------------------------------------------------------------------

/// An operand of a deferred product, as the product will read it: computed
/// if it is a product itself (a product's own operand is computed, not fused).
/// Anything else deferred stays deferred, and is read where it lies or
/// computed as the product loads it -- into the host product's panels as they
/// are packed, or into the GPU product kernel's tiles.
bool product_operand(Process& p, TensorObj** t, Value* err) {
    TensorObj* x = *t;
    if (TensorObj* done = computed_of(x)) {
        *t = done;
        return true;
    }
    if (pending_product(x)) return settle(p, t, err);
    return true;
}

/// `a @ b`. Two vectors give their dot product, a number; a matrix and a
/// vector, in either order, a vector; two matrices a matrix.
bool batched_matmul(Process& p, TensorObj* a, TensorObj* b, Value* out);
bool reshape(Process& p, TensorObj* t, const Shape& s, Value* out);

bool matmul(Process& p, TensorObj* a, TensorObj* b, Value* out) {
    if (a->rank == 3 && (b->rank == 3 || b->rank == 2)) return batched_matmul(p, a, b, out);
    if (a->rank > 2 || b->rank > 2)
        return fail(p, out, "shape_error",
                    "`@` multiplies vectors, matrices and batches of matrices (`[b, m, k] @ [b, k, n]` "
                    "or `[b, m, k] @ [k, n]`), not tensors of shape " +
                        shape_text(shape_of(a)) + " and " + shape_text(shape_of(b)));
    if (!same_device(p, a, b, "`@`", out)) return false;
    // As matrices: a vector on the left is a row, on the right a column.
    const size_t M = a->rank == 2 ? a->dims[0] : 1;
    const size_t K = a->rank == 2 ? a->dims[1] : a->dims[0];
    const size_t Kb = b->dims[0];
    const size_t N = b->rank == 2 ? b->dims[1] : 1;
    if (K != Kb)
        return fail(p, out, "shape_error",
                    "`@` needs the inner lengths to agree, but they are " + std::to_string(K) +
                        " and " + std::to_string(Kb) + " (shapes " + shape_text(shape_of(a)) +
                        " and " + shape_text(shape_of(b)) + ")");
    Shape s;
    if (a->rank == 2 && b->rank == 2) s = shape2(uint32_t(M), uint32_t(N));
    else if (a->rank == 2) s = shape1(uint32_t(M));
    else if (b->rank == 2) s = shape1(uint32_t(N));
    else s = shape1(1);  // two vectors: answered as a number below
    const bool scalar = a->rank == 1 && b->rank == 1;

    // A product large enough to be worth it is deferred like elementwise work,
    // so that what is done to it next can be fused into it -- and its
    // operands are kept as they are wherever the product can read them in
    // place (`plan_operand`), so that what was done to them before is fused
    // into it too. The dot product of two vectors answers a number, which
    // nothing fuses with.
    if (!scalar && (on_gpu(a) || s.count >= kFuseMin)) {
        if (!product_operand(p, &a, out) || !product_operand(p, &b, out)) return false;
        Program prog;
        prog.product(a, b);
        *out = defer(p, prog, s, a->device, a->dtype);
        return true;
    }

    // The dot product is the sum of the elementwise product: deferred and
    // folded as it is computed, so that a chain on either side is never
    // stored, and on the GPU one fused reduction rather than a product kernel
    // with a single column of work. Two computed vectors on the host keep the
    // dot kernel, which allocates nothing at all.
    if (scalar && (on_gpu(a) || !computed_of(a) || !computed_of(b))) {
        Value prod;
        if (!tensor_arith(p, Op::Mul, from_obj(a), from_obj(b), &prod)) {
            *out = prod;
            return false;
        }
        double r;
        if (!reduce(p, KRED_SUM, tensor_of(prod), &r, out)) return false;
        *out = p.heap().make_float(r);
        return true;
    }

    if (!settle(p, &a, out) || !settle(p, &b, out)) return false;
    if (on_gpu(a)) {
        gpu::Buffer* c = gpu_alloc(p, s.count, a->dtype, out);
        if (!c) return false;
        std::string why;
        if (!gpu::matmul(a->dtype, gpu::Operand{buffer_of(a), K, 1}, gpu::Operand{buffer_of(b), N, 1},
                         c, M, K, N, &why)) {
            gpu::release(c);
            return gpu_fail(p, out, why);
        }
        if (scalar) {
            double d = 0;
            bool ok = gpu::download(c, a->dtype, &d, 1, &why);
            gpu::release(c);
            if (!ok) return gpu_fail(p, out, why);
            *out = p.heap().make_float(d);
            return true;
        }
        *out = wrap_gpu(p, s, a->dtype, c);
        return true;
    }
    if (scalar) {
        *out = p.heap().make_float(tensor_kernels().dot(a->data(), b->data(), K));
        return true;
    }
    double* data;
    Value t;
    if (!new_host(p, s, &t, &data)) {
        *out = t;
        return false;
    }
    host_gemm(plain_operand(a->data(), K), plain_operand(b->data(), N), data, M, K, N);
    *out = t;
    return true;
}

/// `a @ b` for a batch of matrices `a` (`[B, M, K]`): by one matrix `b`
/// (`[K, N]`), which is one ordinary product of `a`'s rows -- `[B * M, K]`
/// -- and so defers and fuses like any other; or by a batch `b` (`[B, K,
/// N]`), each pair multiplied, `[B, M, N]`. A batch of products is what
/// attention is made of: many small products, so the host runs one per
/// thread rather than splitting each, and the GPU one launch for them all.
bool batched_matmul(Process& p, TensorObj* a, TensorObj* b, Value* out) {
    if (!same_device(p, a, b, "`@`", out)) return false;
    const uint32_t B = a->dims[0], M = a->dims[1], K = a->dims[2];
    if (b->rank == 2) {
        if (b->dims[0] != K)
            return fail(p, out, "shape_error",
                        "`@` needs the inner lengths to agree, but they are " + std::to_string(K) +
                            " and " + std::to_string(b->dims[0]) + " (shapes " +
                            shape_text(shape_of(a)) + " and " + shape_text(shape_of(b)) + ")");
        Value rows, prod;
        if (!reshape(p, a, shape2(B * M, K), &rows)) {
            *out = rows;
            return false;
        }
        if (!matmul(p, tensor_of(rows), b, &prod)) {
            *out = prod;
            return false;
        }
        Shape s;
        s.rank = 3;
        s.dims[0] = B;
        s.dims[1] = M;
        s.dims[2] = b->dims[1];
        s.count = uint64_t(B) * M * b->dims[1];
        return reshape(p, tensor_of(prod), s, out);
    }
    const uint32_t N = b->dims[2];
    if (b->dims[0] != B || b->dims[1] != K)
        return fail(p, out, "shape_error",
                    "`@` of two batches needs as many matrices in each and the inner lengths to "
                    "agree, not shapes " + shape_text(shape_of(a)) + " and " + shape_text(shape_of(b)));
    if (!settle(p, &a, out) || !settle(p, &b, out)) return false;
    Shape s;
    s.rank = 3;
    s.dims[0] = B;
    s.dims[1] = M;
    s.dims[2] = N;
    s.count = uint64_t(B) * M * N;
    if (s.count > kMaxElements)
        return fail(p, out, "out_of_memory", "a batch of products this large is larger than 4 GiB");
    if (on_gpu(a)) {
        gpu::Buffer* c = gpu_alloc(p, s.count, a->dtype, out);
        if (!c) return false;
        std::string why;
        if (!gpu::bmm(a->dtype, buffer_of(a), buffer_of(b), c, B, M, K, N, &why)) {
            gpu::release(c);
            return gpu_fail(p, out, why);
        }
        *out = wrap_gpu(p, s, a->dtype, c);
        return true;
    }
    double* data;
    Value t;
    if (!new_host(p, s, &t, &data)) {
        *out = t;
        return false;
    }
    const double* da = a->data();
    const double* db = b->data();
    const TensorKernels& k = tensor_kernels();
    parallel_range(B, [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i)
            k.gemm(plain_operand(da + i * M * K, K), plain_operand(db + i * K * N, N),
                   data + i * M * N, M, K, N, nullptr);
    }, size_t(s.count) * K, 1);
    *out = t;
    return true;
}

/// `t` with its axes reordered: axis `k` of the answer is axis `axes[k]` of
/// `t`. A matrix's transpose is `[1, 0]`; attention's heads are moved
/// between the batch and the sequence with `[0, 2, 1, 3]`.
bool permute_axes(Process& p, TensorObj* t, const std::vector<uint32_t>& axes, Value* out) {
    if (!settle(p, &t, out)) return false;
    const uint32_t r = t->rank;
    size_t in_stride[TENSOR_MAX_RANK];
    size_t step = 1;
    for (uint32_t k = r; k-- > 0;) {
        in_stride[k] = step;
        step *= t->dims[k];
    }
    Shape s;
    s.rank = r;
    s.count = t->count;
    size_t dims[TENSOR_MAX_RANK], strides[TENSOR_MAX_RANK];
    for (uint32_t k = 0; k < r; ++k) {
        s.dims[k] = t->dims[axes[k]];
        dims[k] = s.dims[k];
        strides[k] = in_stride[axes[k]];
    }
    if (on_gpu(t)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, t->dtype, out);
        if (!b) return false;
        std::string why;
        if (!gpu::permute(t->dtype, buffer_of(t), b, r, dims, strides, size_t(s.count), &why)) {
            gpu::release(b);
            return gpu_fail(p, out, why);
        }
        *out = wrap_gpu(p, s, t->dtype, b);
        return true;
    }
    double* data;
    if (!new_host(p, s, out, &data)) return false;
    const double* in = t->data();
    // Each thread walks its range of the answer with a counter per axis,
    // starting from the coordinates of its first element.
    parallel_range(size_t(s.count), [&](size_t begin, size_t end, size_t) {
        size_t coord[TENSOR_MAX_RANK] = {};
        size_t rest = begin, at = 0;
        for (uint32_t k = r; k-- > 0;) {
            coord[k] = rest % dims[k];
            rest /= dims[k];
            at += coord[k] * strides[k];
        }
        for (size_t i = begin; i < end; ++i) {
            data[i] = in[at];
            for (uint32_t k = r; k-- > 0;) {
                at += strides[k];
                if (++coord[k] < dims[k]) break;
                at -= coord[k] * strides[k];
                coord[k] = 0;
            }
        }
    });
    return true;
}

/// A matrix transposed. Deferred where anything is: a `[LOADT k]` program,
/// which an elementwise chain reads in place, a product reads by strides, and
/// which on its own is the blocked transpose kernel (`run_program`).
bool transpose(Process& p, TensorObj* x, Value* out) {
    if (x->rank == 1) {
        *out = from_obj(x);
        return true;
    }
    if (x->rank != 2)
        return fail(p, out, "shape_error",
                    "`transpose` takes a vector or a matrix, not shape " +
                        shape_text(shape_of(x)));
    if (!computed_of(x)) {
        // The transpose of a transpose is what it was taken of.
        Program prog;
        append(prog, from_obj(x));
        if (is_transpose(prog)) {
            *out = from_obj(prog.inputs[0]);
            return true;
        }
        if (!settle(p, &x, out)) return false;
    }
    x = computed_of(x);
    const size_t R = x->dims[0], C = x->dims[1];
    const Shape s = shape2(uint32_t(C), uint32_t(R));
    if (on_gpu(x) || s.count >= kFuseMin) {
        Program prog;
        prog.emit(FUSE_LOADT, prog.input(x));
        *out = defer(p, prog, s, x->device, x->dtype);
        return true;
    }
    double* data;
    Value t;
    if (!new_host(p, s, &t, &data)) {
        *out = t;
        return false;
    }
    tensor_kernels().transpose(x->data(), data, R, C);
    *out = t;
    return true;
}

/// The same numbers in shape `s`, of the same size. Nothing is copied that
/// does not have to be:
///
/// - A deferred tensor whose inputs are all as long as it is reads each of
///   them at exactly the position it writes, so its program means the same
///   under any shape of the same size, and it stays deferred, reshaped. One
///   with a broadcast input does not -- a row repeated down a matrix is not a
///   row repeated down a vector -- and is computed first.
/// - On the GPU the buffer is shared, which is free: neither tensor can
///   change it.
/// - A large computed tensor on the host answers a deferred `[LOAD k]` in the
///   new shape: a chain reads it where it lies, and a product reads it by
///   strides (`plan_operand`). Only a small one is copied.
bool reshape(Process& p, TensorObj* t, const Shape& s, Value* out) {
    if (!computed_of(t)) {
        Program prog;
        append(prog, from_obj(t));
        bool flat = true;
        for (unsigned i = 0; i < prog.ninputs; ++i) flat = flat && prog.inputs[i]->count == t->count;
        if (flat) {
            *out = defer(p, prog, s, t->device, t->dtype);
            return true;
        }
    }
    if (!settle(p, &t, out)) return false;
    if (on_gpu(t)) {
        gpu::retain(buffer_of(t));
        *out = wrap_gpu(p, s, t->dtype, buffer_of(t));
        return true;
    }
    if (s.count >= kFuseMin) {
        Program prog;
        prog.emit(FUSE_LOAD, prog.input(t));
        *out = defer(p, prog, s, t->device, t->dtype);
        return true;
    }
    double* data;
    if (!new_host(p, s, out, &data)) return false;
    std::memcpy(data, t->data(), size_t(t->count) * sizeof(double));
    return true;
}

// --- rendering -----------------------------------------------------------------------

/// A number as a float prints (`append_shortest`), at the precision the
/// tensor holds.
void render_number(double d, bool f32, std::string* out) { append_shortest(d, f32, out); }

void render_axis(const double*& data, const Shape& s, uint32_t depth, bool f32, std::string* out) {
    out->push_back('[');
    for (uint32_t i = 0; i < s.dims[depth]; ++i) {
        if (i) out->append(", ");
        if (depth + 1 == s.rank) render_number(*data++, f32, out);
        else render_axis(data, s, depth + 1, f32, out);
    }
    out->push_back(']');
}

// --- the natives -------------------------------------------------------------------

NativeResult raised(Value err) { return NativeResult::raise(err); }

/// The tensor argument `i`, made from lists if it is not one already, as it
/// stands: deferred if it is. For what reads only the shape, and for what
/// fuses with the program a deferred tensor carries.
#define DREAM_SHAPE_ARG(name, i)                                       \
    TensorObj* name = nullptr;                                         \
    do {                                                               \
        Value dream_err_;                                              \
        if (!coerce(p, args[i], &name, &dream_err_)) return raised(dream_err_); \
    } while (0)

/// The same, with its numbers computed: what everything else takes.
#define DREAM_ARG(name, i)                                             \
    DREAM_SHAPE_ARG(name, i);                                          \
    do {                                                               \
        Value dream_err_;                                              \
        if (!settle(p, &name, &dream_err_)) return raised(dream_err_); \
    } while (0)

NativeResult t_of_list(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 0);
    return NativeResult::ok(from_obj(t));
}

/// Nested lists of floats, the inverse of `of_list`.
NativeResult t_to_list(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 0);
    std::vector<double> scratch;
    const double* data;
    Value err;
    if (!host_view(p, t, scratch, &data, &err)) return raised(err);
    const Shape s = shape_of(t);
    // Built innermost first and back to front, one level of lists at a time:
    // `level` holds the lists of the current depth in order.
    std::vector<Value> level(size_t(s.count));
    for (size_t i = 0; i < level.size(); ++i) level[i] = p.heap().make_float(data[i]);
    for (uint32_t d = s.rank; d-- > 0;) {
        const size_t width = s.dims[d];
        std::vector<Value> up(level.size() / width);
        for (size_t g = 0; g < up.size(); ++g) {
            Value list = NIL;
            for (size_t i = width; i-- > 0;) list = p.heap().make_cons(level[g * width + i], list);
            up[g] = list;
        }
        level.swap(up);
    }
    return NativeResult::ok(level[0]);
}

NativeResult filled(Process& p, Value shape, double x) {
    Shape s;
    Value t;
    if (!read_shape(p, shape, &s, &t)) return raised(t);
    double* data;
    if (!new_host(p, s, &t, &data)) return raised(t);
    std::fill(data, data + s.count, x);
    return NativeResult::ok(t);
}

NativeResult t_zeros(Process& p, Value, Value* args, uint32_t) { return filled(p, args[0], 0.0); }
NativeResult t_ones(Process& p, Value, Value* args, uint32_t) { return filled(p, args[0], 1.0); }

NativeResult t_fill(Process& p, Value, Value* args, uint32_t) {
    Value x = resolve(args[1]);
    if (!is_number(x)) {
        Value err;
        type_fail(p, &err, "`fill` fills with a number, not " + describe(p, x));
        return raised(err);
    }
    return filled(p, args[0], number_of(x));
}

/// A positive integer argument, for `identity` and `range`.
bool length_arg(Process& p, Value v, const char* what, uint32_t* n, Value* err) {
    v = resolve(v);
    if (!is_fixnum(v) || fixnum_value(v) < 1 || uint64_t(fixnum_value(v)) > kMaxElements)
        return fail(p, err, "shape_error",
                    std::string("`") + what + "` takes a positive length, not " + describe(p, v));
    *n = uint32_t(fixnum_value(v));
    return true;
}

NativeResult t_identity(Process& p, Value, Value* args, uint32_t) {
    uint32_t n;
    Value t;
    if (!length_arg(p, args[0], "identity", &n, &t)) return raised(t);
    if (uint64_t(n) * n > kMaxElements) {
        fail(p, &t, "out_of_memory", "an identity matrix this large is larger than 4 GiB");
        return raised(t);
    }
    double* data;
    if (!new_host(p, shape2(n, n), &t, &data)) return raised(t);
    std::fill(data, data + size_t(n) * n, 0.0);
    for (size_t i = 0; i < n; ++i) data[i * n + i] = 1.0;
    return NativeResult::ok(t);
}

NativeResult t_range(Process& p, Value, Value* args, uint32_t) {
    uint32_t n;
    Value t;
    if (!length_arg(p, args[0], "range", &n, &t)) return raised(t);
    double* data;
    if (!new_host(p, shape1(n), &t, &data)) return raised(t);
    for (size_t i = 0; i < n; ++i) data[i] = double(i);
    return NativeResult::ok(t);
}

/// Uniform numbers in [0, 1), the same ones for the same seed on every
/// machine: splitmix64, whose top 53 bits make the double. A pure function of
/// its arguments, as everything in the language that is not spelled with a `!`.
NativeResult t_random(Process& p, Value, Value* args, uint32_t) {
    Value seed = resolve(args[0]);
    if (!is_fixnum(seed)) {
        Value err;
        type_fail(p, &err, "`random` takes an integer seed, not " + describe(p, seed));
        return raised(err);
    }
    Shape s;
    Value t;
    if (!read_shape(p, args[1], &s, &t)) return raised(t);
    double* data;
    if (!new_host(p, s, &t, &data)) return raised(t);
    uint64_t x = uint64_t(fixnum_value(seed));
    for (uint64_t i = 0; i < s.count; ++i) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        data[i] = double(z >> 11) * (1.0 / 9007199254740992.0);
    }
    return NativeResult::ok(t);
}

/// Uniforms with the shape, device and dtype of a tensor. In particular a
/// dropout mask never needs to cross from the host to the device.
NativeResult t_random_like(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(like, 1);
    Value seed = resolve(args[0]), out;
    if (!is_fixnum(seed)) {
        type_fail(p, &out, "`random_like` takes an integer seed");
        return raised(out);
    }
    const Shape s = shape_of(like);
    const uint64_t start = uint64_t(fixnum_value(seed));
    if (on_gpu(like)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, like->dtype, &out);
        if (!b) return raised(out);
        std::string why;
        if (!gpu::random_uniform(like->dtype, b, s.count, start, &why)) {
            gpu::release(b);
            gpu_fail(p, &out, why);
            return raised(out);
        }
        return NativeResult::ok(wrap_gpu(p, s, like->dtype, b));
    }
    double* data;
    if (!new_host(p, s, &out, &data)) return raised(out);
    uint64_t x = start;
    for (uint64_t i = 0; i < s.count; ++i) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        data[i] = double(z >> 11) * (1.0 / 9007199254740992.0);
    }
    return NativeResult::ok(out);
}

NativeResult t_shape(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    Value list = NIL;
    for (uint32_t i = t->rank; i-- > 0;) list = p.heap().make_cons(make_fixnum(t->dims[i]), list);
    return NativeResult::ok(list);
}

NativeResult t_rank(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    return NativeResult::ok(make_fixnum(t->rank));
}

NativeResult t_size(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    return NativeResult::ok(make_integer(p, int64_t(t->count)));
}

NativeResult t_device(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    return NativeResult::ok(make_atom(p.runtime().intern_atom(on_gpu(t) ? "gpu" : "host")));
}

NativeResult t_dtype(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    return NativeResult::ok(
        make_atom(p.runtime().intern_atom(on_gpu(t) && t->dtype == gpu::F32 ? "f32" : "f64")));
}

/// The same numbers in a new shape with the same count. On the GPU the buffer
/// itself is shared, which is free: neither tensor can change it.
NativeResult t_reshape(Process& p, Value, Value* args, uint32_t) {
    Shape s;
    Value err;
    if (!read_shape(p, args[0], &s, &err)) return raised(err);
    DREAM_SHAPE_ARG(t, 1);
    if (s.count != t->count) {
        fail(p, &err, "shape_error",
             "cannot reshape " + std::to_string(t->count) + " numbers (shape " +
                 shape_text(shape_of(t)) + ") to shape " + shape_text(s));
        return raised(err);
    }
    Value out;
    if (!reshape(p, t, s, &out)) return raised(out);
    return NativeResult::ok(out);
}

NativeResult t_transpose(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    Value out;
    if (!transpose(p, t, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// One element, by a list of indices, one per axis.
NativeResult t_at(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 0);
    std::vector<Value> items;
    bool is_seq;
    Value err;
    if (!items_of(p, args[1], items, &is_seq, &err)) {
        if (!is_seq) type_fail(p, &err, "`at` takes a list of indices");
        return raised(err);
    }
    if (items.size() != t->rank) {
        fail(p, &err, "shape_error",
             "`at` needs one index per axis: " + std::to_string(t->rank) + ", not " +
                 std::to_string(items.size()));
        return raised(err);
    }
    uint64_t offset = 0;
    for (uint32_t i = 0; i < t->rank; ++i) {
        Value w;
        if (!force_whnf(p, items[i], &w)) return raised(p.result);
        if (!is_fixnum(w) || fixnum_value(w) < 0 || fixnum_value(w) >= int64_t(t->dims[i])) {
            err = raise_error(p, well_known(p.runtime()).out_of_bounds,
                              "index " + describe(p, w) + " is outside axis " + std::to_string(i) +
                                  " of length " + std::to_string(t->dims[i]));
            return raised(err);
        }
        offset = offset * t->dims[i] + uint64_t(fixnum_value(w));
    }
    if (!on_gpu(t)) return NativeResult::ok(p.heap().make_float(t->data()[offset]));
    std::string why;
    gpu::Buffer* one = gpu::alloc(gpu::dtype_size(t->dtype), &why);
    double d = 0;
    bool ok = one && gpu::copy(buffer_of(t), t->dtype, size_t(offset), one, 1, &why) &&
              gpu::download(one, t->dtype, &d, 1, &why);
    if (one) gpu::release(one);
    if (!ok) {
        gpu_fail(p, &err, why);
        return raised(err);
    }
    return NativeResult::ok(p.heap().make_float(d));
}

NativeResult t_matmul(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(a, 0);
    DREAM_SHAPE_ARG(b, 1);
    Value out;
    if (!matmul(p, a, b, &out)) return raised(out);
    return NativeResult::ok(out);
}

NativeResult t_dot(Process& p, Value callee, Value* args, uint32_t argc) {
    DREAM_SHAPE_ARG(a, 0);
    DREAM_SHAPE_ARG(b, 1);
    if (a->rank != 1 || b->rank != 1) {
        Value err;
        fail(p, &err, "shape_error",
             "`dot` takes two vectors, not shapes " + shape_text(shape_of(a)) + " and " +
                 shape_text(shape_of(b)) + "; `@` multiplies matrices");
        return raised(err);
    }
    Value out;
    if (!matmul(p, a, b, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// `outer a b`: every product `a[i] * b[j]`, as a matrix. A product with an
/// inner length of one, which is what lets the GPU do it with the same kernel.
NativeResult t_outer(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(a, 0);
    DREAM_SHAPE_ARG(b, 1);
    Value err;
    if (a->rank != 1 || b->rank != 1) {
        fail(p, &err, "shape_error", "`outer` takes two vectors");
        return raised(err);
    }
    if (!same_device(p, a, b, "`outer`", &err)) return raised(err);
    const uint32_t M = a->dims[0], N = b->dims[0];
    if (uint64_t(M) * N > kMaxElements) {
        fail(p, &err, "out_of_memory", "an outer product this large is larger than 4 GiB");
        return raised(err);
    }
    // A column times a row: an ordinary product with an inner length of one,
    // so that it defers, and what is done to it next fuses into it, as after
    // any other `@`.
    Value col, row, out;
    if (!reshape(p, a, shape2(M, 1), &col)) return raised(col);
    if (!reshape(p, b, shape2(1, N), &row)) return raised(row);
    if (!matmul(p, tensor_of(col), tensor_of(row), &out)) return raised(out);
    return NativeResult::ok(out);
}

/// `sum`, `minimum` and `maximum`, by the native's `user` field.
NativeResult t_reduce(Process& p, Value callee, Value* args, uint32_t) {
    const int op = int(static_cast<NativeObj*>(as_obj(callee))->user);
    DREAM_SHAPE_ARG(t, 0);
    double r;
    Value err;
    if (!reduce(p, op, t, &r, &err)) return raised(err);
    return NativeResult::ok(p.heap().make_float(r));
}

/// Keep the reduction on its device as a one-element tensor. Reshaping to
/// a column lets the existing parallel axis reduction do the whole fold.
NativeResult t_sum_tensor(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    Value column, out;
    if (!reshape(p, t, shape2(uint32_t(t->count), 1), &column)) return raised(column);
    if (!reduce_axis(p, KRED_SUM, tensor_of(column), 0, &out)) return raised(out);
    return NativeResult::ok(out);
}

NativeResult t_mean(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    double r;
    Value err;
    if (!reduce(p, KRED_SUM, t, &r, &err)) return raised(err);
    return NativeResult::ok(p.heap().make_float(r / double(t->count)));
}

/// The Euclidean length of all the numbers: the square root of the sum of
/// their squares.
NativeResult t_norm(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    // `t * t` deferred and summed as it is computed: one pass, nothing stored,
    // and the same on either device.
    Value sq, err;
    if (!tensor_arith(p, Op::Mul, from_obj(t), from_obj(t), &sq)) return raised(sq);
    double r;
    if (!reduce(p, KRED_SUM, static_cast<TensorObj*>(as_obj(sq)), &r, &err)) return raised(err);
    return NativeResult::ok(p.heap().make_float(std::sqrt(r)));
}

/// A fold along one axis, which drops out of the shape, by the native's
/// `user` field (a `KRED_*`): `sum_axis 0` of a matrix is the sum of its rows,
/// `max_axis 1` the largest of each row, `argmax_axis 1` where in each row it
/// is. The `ARG` forms answer positions as numbers, which is what lets them
/// stay on the GPU. A vector folds to one number -- an integer position for
/// the `ARG` forms.
NativeResult t_axis(Process& p, Value callee, Value* args, uint32_t) {
    const int op = int(static_cast<NativeObj*>(as_obj(callee))->user);
    static const char* const names[] = {"sum_axis", "min_axis", "max_axis", "argmin_axis",
                                        "argmax_axis"};
    Value ax = resolve(args[0]);
    DREAM_SHAPE_ARG(t, 1);
    Value err;
    if (!is_fixnum(ax) || fixnum_value(ax) < 0 || fixnum_value(ax) >= int64_t(t->rank)) {
        fail(p, &err, "shape_error",
             std::string("`") + names[op] + "` takes an axis of shape " +
                 shape_text(shape_of(t)) + ", not " + describe(p, ax));
        return raised(err);
    }
    if (t->rank == 1 && op < KRED_ARGMIN) {
        double r;
        if (!reduce(p, op, t, &r, &err)) return raised(err);
        return NativeResult::ok(p.heap().make_float(r));
    }
    if (t->rank == 1) {
        // A vector as a one-row matrix, folded along its row.
        Value row, at;
        if (!reshape(p, t, shape2(1, t->dims[0]), &row)) return raised(row);
        if (!reduce_axis(p, op, tensor_of(row), 1, &at)) return raised(at);
        TensorObj* one = tensor_of(at);
        std::vector<double> scratch;
        const double* data;
        if (!host_view(p, one, scratch, &data, &err)) return raised(err);
        return NativeResult::ok(make_fixnum(int64_t(data[0])));
    }
    Value out;
    if (!reduce_axis(p, op, t, uint32_t(fixnum_value(ax)), &out)) return raised(out);
    return NativeResult::ok(out);
}

/// `sqrt`, `exp`, `relu` and the rest, by the native's `user` field.
NativeResult t_unary(Process& p, Value callee, Value* args, uint32_t) {
    const int fn = int(static_cast<NativeObj*>(as_obj(callee))->user);
    DREAM_SHAPE_ARG(t, 0);
    Value out;
    if (!map_unary(p, fn, t, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// The name a binary `std.tensor` function is called by, for its errors.
const char* binary_name(int op) {
    static const char* const names[] = {"+", "-", "*", "/", "%", "max", "min", "pow",
                                        "lt", "le", "gt", "ge", "eq", "ne"};
    return names[op];
}

/// An operand of a binary function: a number as it is, or a tensor, made
/// from lists if need be.
bool binary_operand(Process& p, Value v, Value* out, Value* err) {
    Value w;
    if (!force_whnf(p, v, &w)) {
        *err = p.result;
        return false;
    }
    if (is_number(w)) {
        *out = w;
        return true;
    }
    TensorObj* t;
    if (!coerce(p, w, &t, err)) return false;
    *out = from_obj(t);
    return true;
}

/// `max`, `min`, `pow` and the comparisons, by the native's `user` field (a
/// `KOP_*`): elementwise exactly as the operators are -- a tensor or a number
/// on either side, broadcast the same way, fused the same way.
NativeResult t_binary(Process& p, Value callee, Value* args, uint32_t) {
    const int op = int(static_cast<NativeObj*>(as_obj(callee))->user);
    Value a, b, out;
    if (!binary_operand(p, args[0], &a, &out) || !binary_operand(p, args[1], &b, &out))
        return raised(out);
    if (!tensor_binary(p, op, a, b, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// A tensor of `t`'s shape, and where `t` is, every number `x`. Deferred
/// like any elementwise result -- a program of one constant -- so on the GPU
/// it costs no upload and fuses into whatever reads it.
NativeResult t_fill_like(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 0);
    Value x = resolve(args[1]), out;
    if (!is_number(x)) {
        type_fail(p, &out, "`fill_like` fills with a number, not " + describe(p, x));
        return raised(out);
    }
    Program prog;
    prog.emit(FUSE_CONST, prog.constant(number_of(x)));
    const Shape s = shape_of(t);
    if (!on_gpu(t) && s.count < kFuseMin) {
        if (!run_program(p, prog, s, t->device, t->dtype, &out)) return raised(out);
        return NativeResult::ok(out);
    }
    return NativeResult::ok(defer(p, prog, s, t->device, t->dtype));
}

/// `t` with a new last axis of length `n`, each of its numbers repeated `n`
/// times along it: a value per row, stretched across the row. Nothing is
/// stretched in memory -- it is a `[LOADR k]` program, which a chain reads in
/// place, so `x - repeat k (max_axis 1 x)` is one pass over `x`.
NativeResult t_repeat(Process& p, Value, Value* args, uint32_t) {
    uint32_t n;
    Value out;
    if (!length_arg(p, args[0], "repeat", &n, &out)) return raised(out);
    DREAM_ARG(t, 1);
    Shape s = shape_of(t);
    if (s.rank == TENSOR_MAX_RANK) {
        fail(p, &out, "shape_error",
             "`repeat` adds an axis, and a tensor has at most " +
                 std::to_string(TENSOR_MAX_RANK) + " axes");
        return raised(out);
    }
    s.dims[s.rank++] = n;
    s.count *= n;
    if (s.count > kMaxElements) {
        fail(p, &out, "out_of_memory", "a repeated tensor this large is larger than 4 GiB");
        return raised(out);
    }
    Program prog;
    prog.emit(FUSE_LOADR, prog.input(t) | prog.constant(double(n)) << 3);
    if (!on_gpu(t) && s.count < kFuseMin) {
        if (!run_program(p, prog, s, t->device, t->dtype, &out)) return raised(out);
        return NativeResult::ok(out);
    }
    return NativeResult::ok(defer(p, prog, s, t->device, t->dtype));
}

/// The shape of `t` with its first axis `rows` long: what a slice, a gather
/// or a concatenation answers.
Shape with_rows(const TensorObj* t, uint32_t rows) {
    Shape s = shape_of(t);
    s.dims[0] = rows;
    s.count = s.count / t->dims[0] * rows;
    return s;
}

/// A tensor made of whole rows -- the first axis -- of others, row `r` of the
/// answer copied from `rows[r]` of `src`. What `slice`, `take` and `concat`
/// come down to; on the GPU it is copies or one gather, and nothing comes
/// back to the host.
bool gather_rows(Process& p, TensorObj* src, const std::vector<uint32_t>& rows, Value* out) {
    const Shape s = with_rows(src, uint32_t(rows.size()));
    const size_t width = size_t(src->count / src->dims[0]);
    if (on_gpu(src)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, src->dtype, out);
        if (!b) return false;
        std::string why;
        // A run of consecutive rows is one copy; anything else one gather.
        bool consecutive = true;
        for (size_t r = 1; r < rows.size() && consecutive; ++r)
            consecutive = rows[r] == rows[r - 1] + 1;
        const bool ok = consecutive
                            ? gpu::copy(buffer_of(src), src->dtype, size_t(rows[0]) * width, b,
                                        rows.size() * width, &why)
                            : gpu::gather(src->dtype, buffer_of(src), rows.data(), rows.size(),
                                          width, b, &why);
        if (!ok) {
            gpu::release(b);
            return gpu_fail(p, out, why);
        }
        *out = wrap_gpu(p, s, src->dtype, b);
        return true;
    }
    double* data;
    if (!new_host(p, s, out, &data)) return false;
    const double* from = src->data();
    for (size_t r = 0; r < rows.size(); ++r)
        std::memcpy(data + r * width, from + size_t(rows[r]) * width, width * sizeof(double));
    return true;
}

/// A non-negative integer argument below `limit`.
bool index_arg(Process& p, Value v, const char* what, uint64_t limit, uint32_t* n, Value* err) {
    Value w;
    if (!force_whnf(p, v, &w)) {
        *err = p.result;
        return false;
    }
    if (!is_fixnum(w) || fixnum_value(w) < 0 || uint64_t(fixnum_value(w)) >= limit) {
        *err = raise_error(p, well_known(p.runtime()).out_of_bounds,
                           std::string("`") + what + "` takes a row from 0 to " +
                               std::to_string(limit) + ", exclusive, not " + describe(p, w));
        return false;
    }
    *n = uint32_t(fixnum_value(w));
    return true;
}

/// `slice start count t`: rows `start` to `start + count` of `t`, along its
/// first axis. A batch of a dataset.
NativeResult t_slice(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 2);
    uint32_t start, count;
    Value err;
    if (!index_arg(p, args[0], "slice", t->dims[0], &start, &err)) return raised(err);
    if (!length_arg(p, args[1], "slice", &count, &err)) return raised(err);
    if (uint64_t(start) + count > t->dims[0]) {
        err = raise_error(p, well_known(p.runtime()).out_of_bounds,
                          "`slice` of " + std::to_string(count) + " rows from row " +
                              std::to_string(start) + " runs past the end of " +
                              std::to_string(t->dims[0]));
        return raised(err);
    }
    std::vector<uint32_t> rows(count);
    for (uint32_t r = 0; r < count; ++r) rows[r] = start + r;
    Value out;
    if (!gather_rows(p, t, rows, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// Row numbers from a list, an array, or a tensor of whole numbers, each
/// below `limit`.
bool row_indices(Process& p, Value v, const char* what, uint64_t limit,
                 std::vector<uint32_t>& rows, Value* err) {
    Value w;
    if (!force_whnf(p, v, &w)) {
        *err = p.result;
        return false;
    }
    if (TensorObj* t = tensor_of(w)) {
        std::vector<double> scratch;
        const double* data;
        if (!host_view(p, t, scratch, &data, err)) return false;
        rows.resize(size_t(t->count));
        for (size_t i = 0; i < rows.size(); ++i) {
            const double d = data[i];
            if (!(d >= 0) || d >= double(limit) || d != std::floor(d)) {
                *err = raise_error(p, well_known(p.runtime()).out_of_bounds,
                                   std::string("`") + what + "` was given row " +
                                       std::to_string(d) + " of " + std::to_string(limit));
                return false;
            }
            rows[i] = uint32_t(d);
        }
        return true;
    }
    std::vector<Value> items;
    bool is_seq;
    if (!items_of(p, w, items, &is_seq, err)) {
        if (!is_seq)
            type_fail(p, err, std::string("`") + what + "` takes a list of row numbers, not " +
                                  describe(p, w));
        return false;
    }
    rows.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i)
        if (!index_arg(p, items[i], what, limit, &rows[i], err)) return false;
    return true;
}

/// `take indices t`: the rows of `t` named by `indices`, in that order, any
/// row any number of times. Shuffling a dataset, or looking up embeddings.
NativeResult t_take(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 1);
    std::vector<uint32_t> rows;
    Value out;
    if (!row_indices(p, args[0], "take", t->dims[0], rows, &out)) return raised(out);
    if (rows.empty()) {
        fail(p, &out, "shape_error", "`take` needs at least one row: a tensor cannot be empty");
        return raised(out);
    }
    if (!gather_rows(p, t, rows, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// `concat ts`: tensors joined along their first axis. Every one must have the
/// same shape after it, and all be in one place.
NativeResult t_concat(Process& p, Value, Value* args, uint32_t) {
    std::vector<Value> items;
    bool is_seq;
    Value err;
    if (!items_of(p, args[0], items, &is_seq, &err)) {
        if (!is_seq) type_fail(p, &err, "`concat` takes a list of tensors");
        return raised(err);
    }
    if (items.empty()) {
        fail(p, &err, "shape_error", "`concat` needs at least one tensor");
        return raised(err);
    }
    // Every piece computed first, and kept: nothing collects in a native.
    std::vector<TensorObj*> parts(items.size());
    uint64_t rows = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!coerce(p, items[i], &parts[i], &err) || !settle(p, &parts[i], &err))
            return raised(err);
        const TensorObj* t = parts[i];
        const TensorObj* first = parts[0];
        bool same = t->rank == first->rank;
        for (uint32_t d = 1; same && d < t->rank; ++d) same = t->dims[d] == first->dims[d];
        if (!same) {
            fail(p, &err, "shape_error",
                 "`concat` joins tensors whose shapes agree after the first axis, not " +
                     shape_text(shape_of(first)) + " and " + shape_text(shape_of(t)));
            return raised(err);
        }
        if (!same_device(p, first, t, "`concat`", &err)) return raised(err);
        rows += t->dims[0];
    }
    if (rows > std::numeric_limits<uint32_t>::max()) {
        fail(p, &err, "shape_error", "`concat` would make an axis too long for a tensor");
        return raised(err);
    }
    const Shape s = with_rows(parts[0], uint32_t(rows));
    if (s.count > kMaxElements) {
        fail(p, &err, "out_of_memory", "a concatenation this large is larger than 4 GiB");
        return raised(err);
    }
    Value out;
    if (on_gpu(parts[0])) {
        gpu::Buffer* b = gpu_alloc(p, s.count, parts[0]->dtype, &out);
        if (!b) return raised(out);
        std::string why;
        size_t at = 0;
        for (TensorObj* t : parts) {
            if (!gpu::copy(buffer_of(t), t->dtype, 0, b, size_t(t->count), &why, at)) {
                gpu::release(b);
                gpu_fail(p, &out, why);
                return raised(out);
            }
            at += size_t(t->count);
        }
        return NativeResult::ok(wrap_gpu(p, s, parts[0]->dtype, b));
    }
    double* data;
    if (!new_host(p, s, &out, &data)) return raised(out);
    for (TensorObj* t : parts) {
        std::memcpy(data, t->data(), size_t(t->count) * sizeof(double));
        data += t->count;
    }
    return NativeResult::ok(out);
}

/// `one_hot n indices`: a row per index, `n` wide, 1 at the index and 0
/// everywhere else. Class labels as a classifier's target.
NativeResult t_one_hot(Process& p, Value, Value* args, uint32_t) {
    uint32_t n;
    Value out;
    if (!length_arg(p, args[0], "one_hot", &n, &out)) return raised(out);
    std::vector<uint32_t> rows;
    if (!row_indices(p, args[1], "one_hot", n, rows, &out)) return raised(out);
    if (rows.empty()) {
        fail(p, &out, "shape_error", "`one_hot` needs at least one index");
        return raised(out);
    }
    if (uint64_t(rows.size()) * n > kMaxElements) {
        fail(p, &out, "out_of_memory", "a one-hot matrix this large is larger than 4 GiB");
        return raised(out);
    }
    double* data;
    if (!new_host(p, shape2(uint32_t(rows.size()), n), &out, &data)) return raised(out);
    std::fill(data, data + rows.size() * n, 0.0);
    for (size_t r = 0; r < rows.size(); ++r) data[r * n + rows[r]] = 1.0;
    return NativeResult::ok(out);
}

/// Standard normal numbers, the same for the same seed on every machine:
/// `random`'s splitmix64 stream, two uniforms at a time through Box and
/// Muller. The first uniform is taken from (0, 1] so its logarithm is finite.
NativeResult t_normal(Process& p, Value, Value* args, uint32_t) {
    Value seed = resolve(args[0]);
    if (!is_fixnum(seed)) {
        Value err;
        type_fail(p, &err, "`normal` takes an integer seed, not " + describe(p, seed));
        return raised(err);
    }
    Shape s;
    Value t;
    if (!read_shape(p, args[1], &s, &t)) return raised(t);
    double* data;
    if (!new_host(p, s, &t, &data)) return raised(t);
    uint64_t x = uint64_t(fixnum_value(seed)) ^ 0x6A09E667F3BCC909ull;
    auto uniform = [&x] {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return double(z >> 11) * (1.0 / 9007199254740992.0);
    };
    const double two_pi = 6.283185307179586476925286766559;
    for (uint64_t i = 0; i < s.count; i += 2) {
        const double u1 = 1.0 - uniform(), u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        data[i] = r * std::cos(two_pi * u2);
        if (i + 1 < s.count) data[i + 1] = r * std::sin(two_pi * u2);
    }
    return NativeResult::ok(t);
}

/// The numbers as a string of bytes: each a little-endian IEEE double, in
/// order. What saving a model writes; `of_bytes` reads it back exactly.
NativeResult t_to_bytes(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 0);
    std::vector<double> scratch;
    const double* data;
    Value err;
    if (!host_view(p, t, scratch, &data, &err)) return raised(err);
    const uint64_t bytes = t->count * sizeof(double);
    if (bytes > std::numeric_limits<uint32_t>::max()) {
        fail(p, &err, "out_of_memory", "a tensor this large does not fit a string");
        return raised(err);
    }
    std::string out(size_t(bytes), '\0');
    for (uint64_t i = 0; i < t->count; ++i) {
        uint64_t bits;
        std::memcpy(&bits, &data[i], sizeof bits);
        for (int b = 0; b < 8; ++b) out[size_t(i * 8 + b)] = char(bits >> (8 * b));
    }
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// A host tensor of `shape` from the bytes `to_bytes` wrote.
NativeResult t_of_bytes(Process& p, Value, Value* args, uint32_t) {
    Shape s;
    Value out;
    if (!read_shape(p, args[0], &s, &out)) return raised(out);
    Value b = resolve(args[1]);
    if (!is_obj(b, ObjType::Str)) {
        type_fail(p, &out, "`of_bytes` reads a string of bytes, not " + describe(p, b));
        return raised(out);
    }
    const auto* str = static_cast<const StrObj*>(as_obj(b));
    if (uint64_t(str->len) != s.count * sizeof(double)) {
        fail(p, &out, "shape_error",
             "`of_bytes` needs " + std::to_string(s.count * sizeof(double)) +
                 " bytes for shape " + shape_text(s) + ", not " + std::to_string(str->len));
        return raised(out);
    }
    double* data;
    if (!new_host(p, s, &out, &data)) return raised(out);
    const auto* bytes = reinterpret_cast<const unsigned char*>(str->data());
    for (uint64_t i = 0; i < s.count; ++i) {
        uint64_t bits = 0;
        for (int k = 0; k < 8; ++k) bits |= uint64_t(bytes[i * 8 + k]) << (8 * k);
        std::memcpy(&data[i], &bits, sizeof bits);
    }
    return NativeResult::ok(out);
}

/// A whole number argument at least `least`, for a window's geometry.
bool count_arg(Process& p, Value v, const char* what, int64_t least, size_t* n, Value* err) {
    v = resolve(v);
    if (!is_fixnum(v) || fixnum_value(v) < least || fixnum_value(v) > (int64_t(1) << 31))
        return fail(p, err, "shape_error",
                    std::string("`") + what + "` takes a whole number of at least " +
                        std::to_string(least) + ", not " + describe(p, v));
    *n = size_t(fixnum_value(v));
    return true;
}

/// The window `kh x kw`, `stride`, `pad` over images of `shape` (`[n, h, w,
/// c]`), with the grid of windows it makes; false when it makes none.
bool window_of(Process& p, const char* what, const uint32_t* dims, Value* args, gpu::Window* g,
               Value* err) {
    if (!count_arg(p, args[0], what, 1, &g->kh, err) || !count_arg(p, args[1], what, 1, &g->kw, err) ||
        !count_arg(p, args[2], what, 1, &g->stride, err) || !count_arg(p, args[3], what, 0, &g->pad, err))
        return false;
    g->n = dims[0];
    g->h = dims[1];
    g->w = dims[2];
    g->c = dims[3];
    if (g->h + 2 * g->pad < g->kh || g->w + 2 * g->pad < g->kw)
        return fail(p, err, "shape_error",
                    std::string("`") + what + "`: a " + std::to_string(g->kh) + " x " +
                        std::to_string(g->kw) + " window does not fit an image of " +
                        std::to_string(g->h) + " x " + std::to_string(g->w) + " padded by " +
                        std::to_string(g->pad));
    g->oh = (g->h + 2 * g->pad - g->kh) / g->stride + 1;
    g->ow = (g->w + 2 * g->pad - g->kw) / g->stride + 1;
    return true;
}

/// `im2col kh kw stride pad x`: every `kh x kw` window of the images `x`
/// (`[n, h, w, c]`, channels last), moved `stride` at a time over the images
/// padded by `pad` zeros, as a row: `[n * oh * ow, kh * kw * c]`. A
/// convolution is then one product of these rows with the kernel as a
/// matrix, and a pooling one fold along each row.
NativeResult t_im2col(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(x, 4);
    Value err;
    if (x->rank != 4) {
        fail(p, &err, "shape_error",
             "`im2col` takes images `[n, h, w, c]`, not shape " + shape_text(shape_of(x)));
        return raised(err);
    }
    gpu::Window g;
    if (!window_of(p, "im2col", x->dims, args, &g, &err)) return raised(err);
    const uint64_t rows = uint64_t(g.n) * g.oh * g.ow, K = uint64_t(g.kh) * g.kw * g.c;
    if (rows > std::numeric_limits<uint32_t>::max() || K > std::numeric_limits<uint32_t>::max() ||
        rows * K > kMaxElements) {
        fail(p, &err, "out_of_memory", "the windows of these images are larger than 4 GiB");
        return raised(err);
    }
    const Shape s = shape2(uint32_t(rows), uint32_t(K));
    Value out;
    if (on_gpu(x)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, x->dtype, &out);
        if (!b) return raised(out);
        std::string why;
        if (!gpu::im2col(x->dtype, g, buffer_of(x), b, &why)) {
            gpu::release(b);
            gpu_fail(p, &out, why);
            return raised(out);
        }
        return NativeResult::ok(wrap_gpu(p, s, x->dtype, b));
    }
    double* data;
    if (!new_host(p, s, &out, &data)) return raised(out);
    const double* in = x->data();
    parallel_range(size_t(rows), [&](size_t begin, size_t end, size_t) {
        for (size_t row = begin; row < end; ++row) {
            const size_t ox = row % g.ow, oy = (row / g.ow) % g.oh, b = row / (g.ow * g.oh);
            double* dst = data + row * K;
            for (size_t ky = 0; ky < g.kh; ++ky) {
                const int64_t iy = int64_t(oy * g.stride + ky) - int64_t(g.pad);
                for (size_t kx = 0; kx < g.kw; ++kx, dst += g.c) {
                    const int64_t ix = int64_t(ox * g.stride + kx) - int64_t(g.pad);
                    if (iy < 0 || iy >= int64_t(g.h) || ix < 0 || ix >= int64_t(g.w)) {
                        std::fill(dst, dst + g.c, 0.0);
                    } else {
                        std::memcpy(dst, in + ((b * g.h + size_t(iy)) * g.w + size_t(ix)) * g.c,
                                    g.c * sizeof(double));
                    }
                }
            }
        }
    }, size_t(rows * K), 1);
    return NativeResult::ok(out);
}

/// `col2im shape kh kw stride pad cols`: the reverse of `im2col` for images
/// of `shape` -- each number the sum of every place in `cols` that `im2col`
/// would have copied it to. What a convolution's gradient flows back through.
NativeResult t_col2im(Process& p, Value, Value* args, uint32_t) {
    Shape is;
    Value err;
    if (!read_shape(p, args[0], &is, &err)) return raised(err);
    DREAM_ARG(cols, 5);
    if (is.rank != 4) {
        fail(p, &err, "shape_error", "`col2im` makes images `[n, h, w, c]`, not shape " + shape_text(is));
        return raised(err);
    }
    gpu::Window g;
    if (!window_of(p, "col2im", is.dims, args + 1, &g, &err)) return raised(err);
    const uint64_t rows = uint64_t(g.n) * g.oh * g.ow, K = uint64_t(g.kh) * g.kw * g.c;
    if (cols->rank != 2 || cols->dims[0] != rows || cols->dims[1] != K) {
        fail(p, &err, "shape_error",
             "`col2im` needs the rows `im2col` makes for images of shape " + shape_text(is) +
                 ", which are [" + std::to_string(rows) + ", " + std::to_string(K) + "], not " +
                 shape_text(shape_of(cols)));
        return raised(err);
    }
    Value out;
    if (on_gpu(cols)) {
        gpu::Buffer* b = gpu_alloc(p, is.count, cols->dtype, &out);
        if (!b) return raised(out);
        std::string why;
        if (!gpu::col2im(cols->dtype, g, buffer_of(cols), b, &why)) {
            gpu::release(b);
            gpu_fail(p, &out, why);
            return raised(out);
        }
        return NativeResult::ok(wrap_gpu(p, is, cols->dtype, b));
    }
    double* data;
    if (!new_host(p, is, &out, &data)) return raised(out);
    std::fill(data, data + is.count, 0.0);
    const double* src = cols->data();
    // Each image is its own: the threads divide the images, and within one
    // the windows are added in order, so the sums are the same at any thread
    // count.
    const size_t per_image = g.oh * g.ow;
    parallel_range(g.n, [&](size_t begin, size_t end, size_t) {
        for (size_t b = begin; b < end; ++b) {
            for (size_t r = 0; r < per_image; ++r) {
                const size_t ox = r % g.ow, oy = r / g.ow;
                const double* row = src + (b * per_image + r) * K;
                for (size_t ky = 0; ky < g.kh; ++ky) {
                    const int64_t iy = int64_t(oy * g.stride + ky) - int64_t(g.pad);
                    for (size_t kx = 0; kx < g.kw; ++kx, row += g.c) {
                        const int64_t ix = int64_t(ox * g.stride + kx) - int64_t(g.pad);
                        if (iy < 0 || iy >= int64_t(g.h) || ix < 0 || ix >= int64_t(g.w)) continue;
                        double* dst = data + ((b * g.h + size_t(iy)) * g.w + size_t(ix)) * g.c;
                        for (size_t ci = 0; ci < g.c; ++ci) dst[ci] += row[ci];
                    }
                }
            }
        }
    }, size_t(rows * K), 1);
    return NativeResult::ok(out);
}

/// `permute axes t`: the axes of `t` reordered, axis `k` of the answer being
/// axis `axes[k]` of `t`.
NativeResult t_permute(Process& p, Value, Value* args, uint32_t) {
    DREAM_SHAPE_ARG(t, 1);
    std::vector<Value> items;
    bool is_seq;
    Value err;
    if (!items_of(p, args[0], items, &is_seq, &err)) {
        if (!is_seq) type_fail(p, &err, "`permute` takes a list of axes");
        return raised(err);
    }
    std::vector<uint32_t> axes(items.size());
    std::vector<bool> seen(t->rank, false);
    bool ok = items.size() == t->rank;
    for (size_t k = 0; ok && k < items.size(); ++k) {
        Value w;
        if (!force_whnf(p, items[k], &w)) return raised(p.result);
        ok = is_fixnum(w) && fixnum_value(w) >= 0 && fixnum_value(w) < int64_t(t->rank) &&
             !seen[size_t(fixnum_value(w))];
        if (ok) {
            axes[k] = uint32_t(fixnum_value(w));
            seen[axes[k]] = true;
        }
    }
    if (!ok) {
        fail(p, &err, "shape_error",
             "`permute` takes each axis of shape " + shape_text(shape_of(t)) +
                 " once, in its new order");
        return raised(err);
    }
    Value out;
    if (!permute_axes(p, t, axes, &out)) return raised(out);
    return NativeResult::ok(out);
}

/// Onto the GPU, as float32 (`user` 1) or float64 (`user` 0). A tensor that
/// is already there in that type is answered as it is.
NativeResult t_gpu(Process& p, Value callee, Value* args, uint32_t) {
    const int dtype = int(static_cast<NativeObj*>(as_obj(callee))->user);
    DREAM_ARG(t, 0);
    Value err;
    std::string why;
    if (!gpu::available(&why)) {
        fail(p, &err, "no_gpu", "no GPU is available: " + why);
        return raised(err);
    }
    if (dtype == gpu::F64 && !gpu::has_f64()) {
        fail(p, &err, "no_gpu",
             "this GPU (" + gpu::device_name() + ") has no double precision; `tensor.gpu` "
             "moves a tensor as float32");
        return raised(err);
    }
    if (on_gpu(t) && t->dtype == dtype) return NativeResult::ok(from_obj(t));
    std::vector<double> scratch;
    const double* data;
    if (!host_view(p, t, scratch, &data, &err)) return raised(err);
    Value out;
    if (!upload(p, shape_of(t), data, dtype, &out)) return raised(out);
    return NativeResult::ok(out);
}

NativeResult t_host(Process& p, Value, Value* args, uint32_t) {
    DREAM_ARG(t, 0);
    if (!on_gpu(t)) return NativeResult::ok(from_obj(t));
    double* data;
    Value out;
    if (!new_host(p, shape_of(t), &out, &data)) return raised(out);
    std::string why;
    if (!gpu::download(buffer_of(t), t->dtype, data, size_t(t->count), &why)) {
        gpu_fail(p, &out, why);
        return raised(out);
    }
    return NativeResult::ok(out);
}

NativeResult t_gpu_available(Process& p, Value, Value*, uint32_t) {
    std::string why;
    return NativeResult::ok(make_bool(gpu::available(&why)));
}

/// The GPU's name, or `()` when there is none.
NativeResult t_gpu_name(Process& p, Value, Value*, uint32_t) {
    std::string why;
    if (!gpu::available(&why)) return NativeResult::ok(UNIT);
    std::string name = gpu::device_name();
    return NativeResult::ok(p.heap().make_string(name.data(), uint32_t(name.size())));
}

/// Which CPU kernels this machine runs: "avx2" or "baseline".
NativeResult t_cpu_kernels(Process& p, Value, Value*, uint32_t) {
    const char* name = tensor_kernels_name();
    return NativeResult::ok(p.heap().make_string(name, uint32_t(std::strlen(name))));
}

}  // namespace

// --- what the rest of the VM calls -----------------------------------------------

bool tensor_arith(Process& p, Op op, Value a, Value b, Value* out) {
    return tensor_binary(p, kernel_op(op), a, b, out);
}

bool tensor_binary(Process& p, int kop, Value a, Value b, Value* out) {
    a = resolve(a);
    b = resolve(b);
    TensorObj* x = tensor_of(a);
    TensorObj* y = tensor_of(b);
    const char* what = binary_name(kop);
    Shape s;
    const TensorObj* like;
    if (x && y) {
        if (!broadcast(p, what, shape_of(x), shape_of(y), &s, out)) return false;
        if (!same_device(p, x, y, (std::string("`") + what + "`").c_str(), out)) return false;
        like = x;
    } else if (x && is_number(b)) {
        s = shape_of(x);
        like = x;
    } else if (y && is_number(a)) {
        s = shape_of(y);
        like = y;
    } else {
        return type_fail(p, out, std::string("cannot apply `") + what + "` to " +
                                     describe(p, a) + " and " + describe(p, b) +
                                     ": a tensor combines with a tensor or a number");
    }
    const Value operands[2] = {a, b};
    return elementwise(p, operands, 2, FUSE_BIN, uint8_t(kop), s, like, out);
}

bool tensor_force(Process& p, Value v, Value* err) {
    TensorObj* t = tensor_of(v);
    return settle(p, &t, err);
}

bool tensor_equal(Process& p, Value a, Value b, bool* raised) {
    TensorObj* x = tensor_of(a);
    TensorObj* y = tensor_of(b);
    if (!same_shape(shape_of(x), shape_of(y))) return false;
    std::vector<double> sx, sy;
    const double* dx;
    const double* dy;
    Value err;
    if (!host_view(p, x, sx, &dx, &err) || !host_view(p, y, sy, &dy, &err)) {
        p.result = err;
        *raised = true;
        return false;
    }
    for (uint64_t i = 0; i < x->count; ++i)
        if (dx[i] != dy[i]) return false;
    return true;
}

bool tensor_compare(Process& p, Value a, Value b, int* out) {
    TensorObj* x = tensor_of(a);
    TensorObj* y = tensor_of(b);
    auto order = [](auto u, auto v) { return u < v ? -1 : (u > v ? 1 : 0); };
    if (x->rank != y->rank) {
        *out = order(x->rank, y->rank);
        return true;
    }
    for (uint32_t i = 0; i < x->rank; ++i)
        if (x->dims[i] != y->dims[i]) {
            *out = order(x->dims[i], y->dims[i]);
            return true;
        }
    std::vector<double> sx, sy;
    const double* dx;
    const double* dy;
    Value err;
    if (!host_view(p, x, sx, &dx, &err) || !host_view(p, y, sy, &dy, &err)) {
        p.result = err;
        return false;
    }
    for (uint64_t i = 0; i < x->count; ++i)
        if (dx[i] != dy[i]) {
            *out = order(dx[i], dy[i]);
            return true;
        }
    *out = 0;
    return true;
}

bool tensor_render(Process& p, Value v, std::string* out) {
    TensorObj* t = tensor_of(v);
    const Shape s = shape_of(t);
    const bool f32 = on_gpu(t) && t->dtype == gpu::F32;
    out->append(on_gpu(t) ? (f32 ? "gpu tensor" : "gpu64 tensor") : "tensor");
    // A printed tensor is for reading. Past a thousand numbers nobody reads
    // them, and rendering a large one would build a string of megabytes to
    // say so.
    if (s.count > 1000) {
        out->append(" of shape ");
        out->append(shape_text(s));
        return true;
    }
    std::vector<double> scratch;
    const double* data;
    Value err;
    if (!host_view(p, t, scratch, &data, &err)) {
        p.result = err;
        return false;
    }
    render_axis(data, s, 0, f32, out);
    return true;
}

int tensor_index(Process& p, Value v, int64_t i, Value* out) {
    TensorObj* t = tensor_of(v);
    if (i < 0 || i >= int64_t(t->dims[0])) return 2;
    if (!settle(p, &t, out)) return 0;
    if (t->rank == 1) {
        if (!on_gpu(t)) {
            *out = p.heap().make_float(t->data()[i]);
            return 1;
        }
        std::string why;
        gpu::Buffer* one = gpu::alloc(gpu::dtype_size(t->dtype), &why);
        double d = 0;
        bool ok = one && gpu::copy(buffer_of(t), t->dtype, size_t(i), one, 1, &why) &&
                  gpu::download(one, t->dtype, &d, 1, &why);
        if (one) gpu::release(one);
        if (!ok) {
            gpu_fail(p, out, why);
            return 0;
        }
        *out = p.heap().make_float(d);
        return 1;
    }
    // A row: everything after the first axis, copied out.
    Shape s;
    s.rank = t->rank - 1;
    s.count = 1;
    for (uint32_t d = 1; d < t->rank; ++d) {
        s.dims[d - 1] = t->dims[d];
        s.count *= t->dims[d];
    }
    if (on_gpu(t)) {
        gpu::Buffer* b = gpu_alloc(p, s.count, t->dtype, out);
        if (!b) return 0;
        std::string why;
        if (!gpu::copy(buffer_of(t), t->dtype, size_t(uint64_t(i) * s.count), b, size_t(s.count),
                       &why)) {
            gpu::release(b);
            gpu_fail(p, out, why);
            return 0;
        }
        *out = wrap_gpu(p, s, t->dtype, b);
        return 1;
    }
    double* data;
    if (!new_host(p, s, out, &data)) return 0;
    std::memcpy(data, t->data() + uint64_t(i) * s.count, size_t(s.count) * sizeof(double));
    return 1;
}

NativeResult tensor_matmul_builtin(Process& p, Value, Value* args, uint32_t) {
    return t_matmul(p, UNIT, args, 2);
}

bool tensor_host_copy(Process& p, Value v, Value* out) {
    TensorObj* t = tensor_of(v);
    std::vector<double> scratch;
    const double* data;
    if (!host_view(p, t, scratch, &data, out)) return false;
    double* dest;
    if (!new_host(p, shape_of(t), out, &dest)) return false;
    std::memcpy(dest, data, size_t(t->count) * sizeof(double));
    return true;
}

NativeResult tensor_of_bytes_builtin(Process& p, Value callee, Value* args, uint32_t argc) {
    return t_of_bytes(p, callee, args, argc);
}

ModuleDef make_tensor_module() {
    // Arguments are forced to weak head normal form before each call (bit i of
    // the mask is argument i); a list argument's elements are forced as they
    // are read.
    return ModuleDef{
        "std.tensor",
        {
            {"of_list", 1, 0b1, t_of_list},
            {"to_list", 1, 0b1, t_to_list},
            {"zeros", 1, 0b1, t_zeros},
            {"ones", 1, 0b1, t_ones},
            {"fill", 2, 0b11, t_fill},
            {"identity", 1, 0b1, t_identity},
            {"range", 1, 0b1, t_range},
            {"random", 2, 0b11, t_random},
            {"random_like", 2, 0b11, t_random_like},
            {"shape", 1, 0b1, t_shape},
            {"rank", 1, 0b1, t_rank},
            {"size", 1, 0b1, t_size},
            {"device", 1, 0b1, t_device},
            {"dtype", 1, 0b1, t_dtype},
            {"reshape", 2, 0b11, t_reshape},
            {"transpose", 1, 0b1, t_transpose},
            {"at", 2, 0b11, t_at},
            {"matmul", 2, 0b11, t_matmul},
            {"dot", 2, 0b11, t_dot},
            {"outer", 2, 0b11, t_outer},
            {"sum", 1, 0b1, t_reduce, KRED_SUM},
            {"sum_tensor", 1, 0b1, t_sum_tensor},
            {"minimum", 1, 0b1, t_reduce, KRED_MIN},
            {"maximum", 1, 0b1, t_reduce, KRED_MAX},
            {"mean", 1, 0b1, t_mean},
            {"norm", 1, 0b1, t_norm},
            {"sum_axis", 2, 0b11, t_axis, KRED_SUM},
            {"min_axis", 2, 0b11, t_axis, KRED_MIN},
            {"max_axis", 2, 0b11, t_axis, KRED_MAX},
            {"argmin_axis", 2, 0b11, t_axis, KRED_ARGMIN},
            {"argmax_axis", 2, 0b11, t_axis, KRED_ARGMAX},
            {"sqrt", 1, 0b1, t_unary, KFN_SQRT},
            {"exp", 1, 0b1, t_unary, KFN_EXP},
            {"log", 1, 0b1, t_unary, KFN_LOG},
            {"abs", 1, 0b1, t_unary, KFN_ABS},
            {"tanh", 1, 0b1, t_unary, KFN_TANH},
            {"sin", 1, 0b1, t_unary, KFN_SIN},
            {"cos", 1, 0b1, t_unary, KFN_COS},
            {"relu", 1, 0b1, t_unary, KFN_RELU},
            {"sigmoid", 1, 0b1, t_unary, KFN_SIGMOID},
            {"floor", 1, 0b1, t_unary, KFN_FLOOR},
            {"ceil", 1, 0b1, t_unary, KFN_CEIL},
            {"round", 1, 0b1, t_unary, KFN_ROUND},
            {"sign", 1, 0b1, t_unary, KFN_SIGN},
            {"erf", 1, 0b1, t_unary, KFN_ERF},
            {"max", 2, 0b11, t_binary, KOP_MAX},
            {"min", 2, 0b11, t_binary, KOP_MIN},
            {"pow", 2, 0b11, t_binary, KOP_POW},
            {"lt", 2, 0b11, t_binary, KOP_LT},
            {"le", 2, 0b11, t_binary, KOP_LE},
            {"gt", 2, 0b11, t_binary, KOP_GT},
            {"ge", 2, 0b11, t_binary, KOP_GE},
            {"eq", 2, 0b11, t_binary, KOP_EQ},
            {"ne", 2, 0b11, t_binary, KOP_NE},
            {"fill_like", 2, 0b11, t_fill_like},
            {"repeat", 2, 0b11, t_repeat},
            {"slice", 3, 0b111, t_slice},
            {"take", 2, 0b11, t_take},
            {"concat", 1, 0b1, t_concat},
            {"one_hot", 2, 0b11, t_one_hot},
            {"normal", 2, 0b11, t_normal},
            {"to_bytes", 1, 0b1, t_to_bytes},
            {"im2col", 5, 0b11111, t_im2col},
            {"permute", 2, 0b11, t_permute},
            {"col2im", 6, 0b111111, t_col2im},
            {"of_bytes", 2, 0b11, t_of_bytes},
            {"gpu", 1, 0b1, t_gpu, gpu::F32},
            {"gpu64", 1, 0b1, t_gpu, gpu::F64},
            {"host", 1, 0b1, t_host},
            {"gpu_available", 1, 0b1, t_gpu_available},
            {"gpu_name", 1, 0b1, t_gpu_name},
            {"cpu_kernels", 1, 0b1, t_cpu_kernels},
        }};
}

}  // namespace dream
