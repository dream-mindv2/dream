// The GPU backend: OpenCL, loaded at run time. gpu.hpp says why it is shaped
// this way; this file is the plumbing.
//
// The handful of OpenCL types and constants used here are declared below
// rather than taken from <CL/cl.h>, so that building the VM needs no OpenCL
// headers either. They are fixed by the specification and by every ICD loader's
// ABI, which is what a dlopen'd library has to agree with in any case.

#include "gpu.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

#include "tensor_kernels.hpp"

#if defined(_WIN32)
#include "windows.hpp"
#else
#include <dlfcn.h>
#endif

namespace dream::gpu {
namespace {

// --- the slice of OpenCL 1.2 this uses -----------------------------------------

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bitfield = cl_ulong;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bitfield CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_bitfield CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;
constexpr cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_EXTENSIONS = 0x1030;
constexpr cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_uint CL_TRUE = 1;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

#if defined(_WIN32)
#define DREAM_CL_API __stdcall
#else
#define DREAM_CL_API
#endif

struct Api {
    cl_int(DREAM_CL_API* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int(DREAM_CL_API* GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
    cl_int(DREAM_CL_API* GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context(DREAM_CL_API* CreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void*,
                                            void*, cl_int*);
    cl_command_queue(DREAM_CL_API* CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int*);
    cl_mem(DREAM_CL_API* CreateBuffer)(cl_context, cl_bitfield, size_t, void*, cl_int*);
    cl_int(DREAM_CL_API* ReleaseMemObject)(cl_mem);
    cl_int(DREAM_CL_API* EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
                                             const void*, cl_uint, const cl_event*, cl_event*);
    cl_int(DREAM_CL_API* EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t,
                                            void*, cl_uint, const cl_event*, cl_event*);
    cl_int(DREAM_CL_API* EnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                                            cl_uint, const cl_event*, cl_event*);
    cl_program(DREAM_CL_API* CreateProgramWithSource)(cl_context, cl_uint, const char**,
                                                      const size_t*, cl_int*);
    cl_int(DREAM_CL_API* BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                                       void*, void*);
    cl_int(DREAM_CL_API* GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*,
                                              size_t*);
    cl_kernel(DREAM_CL_API* CreateKernel)(cl_program, const char*, cl_int*);
    cl_int(DREAM_CL_API* SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int(DREAM_CL_API* EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                               const size_t*, const size_t*, cl_uint,
                                               const cl_event*, cl_event*);
    cl_int(DREAM_CL_API* Finish)(cl_command_queue);
    cl_int(DREAM_CL_API* Flush)(cl_command_queue);
    cl_int(DREAM_CL_API* EnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event*,
                                                    cl_event*);
    cl_int(DREAM_CL_API* WaitForEvents)(cl_uint, const cl_event*);
    cl_int(DREAM_CL_API* ReleaseEvent)(cl_event);
};

// --- the kernels ---------------------------------------------------------------
//
// The fixed kernels -- the products and transpose -- are one source, built
// once per element type with `T` defined to it. Elementwise work and
// reductions are not here: each fused program gets kernels generated for it
// (`fused_source`), whose operation codes are tensor_kernels.hpp's, so the
// host and the device agree on what `op == 3` means. Every constant is written
// `(T)n`: a bare `1.0` is a double, and a device without doubles refuses the
// whole program over it.

const char* const kSource = R"CL(
#ifdef FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// The partial products of a split-K product added up, in slice order.
__kernel void sum_parts(__global const T* P, __global T* C, ulong n, int parts) {
    ulong i = get_global_id(0);
    if (i >= n) return;
    T acc = (T)0;
    for (int s = 0; s < parts; ++s) acc += P[(ulong)s * n + i];
    C[i] = acc;
}

// A product with one column: one work-item per row, a dot product each. The
// tiled kernel would run sixteen work-items per row and throw fifteen away.
__kernel void matvec(__global const T* A, ulong ars, ulong acs, __global const T* x, ulong xrs,
                     __global T* y, int M, int K) {
    int i = get_global_id(0);
    if (i >= M) return;
    T acc = (T)0;
    for (int k = 0; k < K; ++k) acc += A[i * ars + k * acs] * x[k * xrs];
    y[i] = acc;
}

__kernel void transpose(__global const T* in, __global T* out, int R, int C) {
    int j = get_global_id(0), i = get_global_id(1);
    if (i < R && j < C) out[(ulong)j * R + i] = in[(ulong)i * C + j];
}

// A sliding window's numbers as rows (see `gpu::im2col`): one work-item per
// number of the answer, reading its image or 0 where the window hangs over
// the padding.
__kernel void im2col(__global const T* in, __global T* out, ulong n, ulong h, ulong w, ulong c,
                     ulong kh, ulong kw, ulong stride, ulong pad, ulong oh, ulong ow) {
    ulong g = get_global_id(0);
    ulong K = kh * kw * c;
    if (g >= n * oh * ow * K) return;
    ulong row = g / K, col = g % K;
    ulong ci = col % c, kx = (col / c) % kw, ky = col / (c * kw);
    ulong ox = row % ow, oy = (row / ow) % oh, b = row / (ow * oh);
    long iy = (long)(oy * stride + ky) - (long)pad, ix = (long)(ox * stride + kx) - (long)pad;
    out[g] = (iy >= 0 && iy < (long)h && ix >= 0 && ix < (long)w)
                 ? in[((b * h + (ulong)iy) * w + (ulong)ix) * c + ci] : (T)0;
}

// The reverse, as a gather rather than a scatter: one work-item per number
// of the image, adding up the places every window that covers it put it.
// Nothing is written twice, so nothing needs an atomic.
__kernel void col2im(__global const T* cols, __global T* out, ulong n, ulong h, ulong w, ulong c,
                     ulong kh, ulong kw, ulong stride, ulong pad, ulong oh, ulong ow) {
    ulong g = get_global_id(0);
    if (g >= n * h * w * c) return;
    ulong ci = g % c, ix = (g / c) % w, iy = (g / (c * w)) % h, b = g / (c * w * h);
    ulong K = kh * kw * c;
    T acc = (T)0;
    for (ulong ky = 0; ky < kh; ++ky) {
        long ty = (long)(iy + pad) - (long)ky;
        if (ty < 0 || ty % (long)stride != 0) continue;
        ulong oy = (ulong)ty / stride;
        if (oy >= oh) continue;
        for (ulong kx = 0; kx < kw; ++kx) {
            long tx = (long)(ix + pad) - (long)kx;
            if (tx < 0 || tx % (long)stride != 0) continue;
            ulong ox = (ulong)tx / stride;
            if (ox >= ow) continue;
            acc += cols[((b * oh + oy) * ow + ox) * K + (ky * kw + kx) * c + ci];
        }
    }
    out[g] = acc;
}

// Axes reordered (see `gpu::permute`): one work-item per number of the
// answer, its coordinates peeled off the last axis first.
__kernel void permute(__global const T* in, __global T* out, ulong count, int rank,
                      ulong d0, ulong d1, ulong d2, ulong d3, ulong d4, ulong d5,
                      ulong s0, ulong s1, ulong s2, ulong s3, ulong s4, ulong s5) {
    ulong g = get_global_id(0);
    if (g >= count) return;
    ulong dims[6] = {d0, d1, d2, d3, d4, d5};
    ulong strides[6] = {s0, s1, s2, s3, s4, s5};
    ulong rest = g, at = 0;
    for (int k = rank - 1; k >= 0; --k) {
        at += (rest % dims[k]) * strides[k];
        rest /= dims[k];
    }
    out[g] = in[at];
}

// Whole rows by number: one work-item per element of the answer.
__kernel void gather(__global const T* in, __global const uint* rows, __global T* out,
                     ulong width, ulong n) {
    ulong g = get_global_id(0);
    if (g < n) out[g] = in[(ulong)rows[g / width] * width + g % width];
}

// Counter form of the host's splitmix64 stream: each work-item can draw
// its own element without a shared generator or a host upload.
__kernel void random_uniform(__global T* out, ulong n, ulong seed) {
    ulong i = get_global_id(0);
    if (i >= n) return;
    ulong z = seed + (i + 1) * (ulong)0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * (ulong)0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * (ulong)0x94D049BB133111EBUL;
    z ^= z >> 31;
    out[i] = min((T)(z >> 11) * (T)(1.0 / 9007199254740992.0),
                 nextafter((T)1, (T)0));
}

)CL";

// What every generated elementwise kernel starts with: the operations a
// program's instructions name, as functions, so that a nested `relu` is a call
// and not its operand written out twice at every level.
const char* const kFusedPrelude = R"CL(
#ifdef FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
inline T relu_(T a) { return a > (T)0 ? a : (T)0; }
inline T sigmoid_(T a) { return (T)1 / ((T)1 + exp(-a)); }
inline T combine_(int op, T a, T b) {
    return op == 0 ? a + b : (op == 1 ? fmin(a, b) : fmax(a, b));
}
inline T sign_(T a) { return a > (T)0 ? (T)1 : (a < (T)0 ? (T)-1 : (T)0); }
)CL";

/// The blocks of C a work-item of a product may compute, `w x w` (see
/// `product_kernel`), smallest first. A large block does the most arithmetic
/// per read of local memory, and makes the fewest work-groups: a 4096-square
/// product runs at 5.9 TFLOP/s on an RTX 3070 with 8 x 8 against 4.1 with 4
/// x 4, while a 256 x 64 one is a single work-group at 8 x 8 and leaves the
/// device idle. Each launch takes the largest that still fills the device
/// (`pick_wpt`).
constexpr size_t kWpts[3] = {2, 4, 8};

/// The source of a matrix product kernel. Every product the device runs --
/// the plain one, a slice of a split-K one, a batch, and one generated with a
/// chain applied as it stores -- is this, differing only in how an element
/// of each operand is loaded, how an element of C is stored, and which part
/// of K is walked.
///
/// **Register tiling.** The first kernel was the textbook one: a work-item
/// per element of C, a `TS x TS` tile through local memory. Each multiply-add
/// then costs two reads of local memory, and that, not arithmetic, bounded
/// it -- 1.2 TFLOP/s for 4096 x 4096 on an RTX 3070 that does twenty. Here
/// each work-item computes a `WPT x WPT` block of C, so a work-group covers
/// a `TS * WPT` square: per step along K a work-item reads `2 * WPT` values
/// from local memory and does `WPT * WPT` multiply-adds with them, held in
/// registers. A work-item's elements are `TS` apart rather than adjacent, so
/// that neighbouring work-items read neighbouring local memory and store to
/// neighbouring addresses.
///
/// In the strings: `load_a` reads `A` at row `r`, column `k`; `load_b` reads
/// `B` at row `k`, column `c`; `store` writes the element `v` at row `r`,
/// column `c` (and may use `i`, its index in a row-major C); `k0` and `k1`
/// bound the walk; `prologue` runs first, with the group's position known.
std::string product_kernel(const std::string& name, size_t wpt, const std::string& params,
                           const std::string& prologue, const std::string& load_a,
                           const std::string& load_b, const std::string& store,
                           const std::string& k0, const std::string& k1) {
    // The block size is written into the source rather than left to a macro,
    // so that one program holds the product at several block sizes.
    const std::string W = std::to_string(wpt);
    std::string body = "__kernel void " + name + "(" + params + ") {\n"
           "    __local T As[TS][TS * WPT];\n"
           "    __local T Bs[TS][TS * WPT];\n"
           "    const int tx = get_local_id(0), ty = get_local_id(1);\n"
           "    const int row0 = get_group_id(1) * TS * WPT, col0 = get_group_id(0) * TS * WPT;\n" +
           prologue +
           "    const int kfrom = " + k0 + ", kto = " + k1 + ";\n"
           "    T acc[WPT][WPT];\n"
           "    for (int i = 0; i < WPT; ++i) for (int j = 0; j < WPT; ++j) acc[i][j] = (T)0;\n"
           "    for (int t = kfrom; t < kto; t += TS) {\n"
           "        for (int l = 0; l < WPT; ++l) {\n"
           "            const int idx = ty * TS + tx + l * TS * TS;\n"
           "            {\n"
           "                const int rr = idx / TS, kk = idx % TS;\n"
           "                const int r = row0 + rr, k = t + kk;\n"
           "                As[kk][rr] = (r < M && k < kto) ? (" + load_a + ") : (T)0;\n"
           "            }\n"
           "            {\n"
           "                const int kk = idx / (TS * WPT), cc = idx % (TS * WPT);\n"
           "                const int k = t + kk, c = col0 + cc;\n"
           "                Bs[kk][cc] = (k < kto && c < N) ? (" + load_b + ") : (T)0;\n"
           "            }\n"
           "        }\n"
           "        barrier(CLK_LOCAL_MEM_FENCE);\n"
           "        for (int k = 0; k < TS; ++k) {\n"
           "            T a[WPT], b[WPT];\n"
           "            for (int i = 0; i < WPT; ++i) a[i] = As[k][ty + TS * i];\n"
           "            for (int j = 0; j < WPT; ++j) b[j] = Bs[k][tx + TS * j];\n"
           "            for (int i = 0; i < WPT; ++i)\n"
           "                for (int j = 0; j < WPT; ++j) acc[i][j] = mad(a[i], b[j], acc[i][j]);\n"
           "        }\n"
           "        barrier(CLK_LOCAL_MEM_FENCE);\n"
           "    }\n"
           "    for (int wi = 0; wi < WPT; ++wi)\n"
           "        for (int wj = 0; wj < WPT; ++wj) {\n"
           "            const int r = row0 + ty + TS * wi, c = col0 + tx + TS * wj;\n"
           "            if (r < M && c < N) {\n"
           "                const ulong i = (ulong)r * N + c;\n"
           "                const T v = acc[wi][wj];\n"
           "                " + store + ";\n"
           "            }\n"
           "        }\n"
           "}\n";
    std::string out;
    for (size_t at = 0;;) {
        const size_t hit = body.find("WPT", at);
        if (hit == std::string::npos) {
            out += body.substr(at);
            break;
        }
        out += body.substr(at, hit - at) + W;
        at = hit + 3;
    }
    return out;
}

/// The fixed kernels' whole source: `kSource`, and the products made by
/// `product_kernel`.
std::string fixed_source() {
    const std::string strided_a = "A[r * ars + k * acs]", strided_b = "B[k * brs + c * bcs]";
    std::string src = kSource;
    // The product, each operand read through strides: a matrix as it is
    // stored, or one read as its transpose with nothing copied.
    for (size_t w : kWpts)
        src += product_kernel("matmul" + std::to_string(w), w,
                          "__global const T* A, ulong ars, ulong acs, __global const T* B, ulong brs, "
                          "ulong bcs, __global T* C, int M, int K, int N",
                          "", strided_a, strided_b, "C[i] = v", "0", "K");
    // One slice of K (see "split-K" at `gpu::matmul`), the slice the third
    // dimension of the launch, into partial products `P[slice]`.
    src += product_kernel("matmul_part", 4,
                          "__global const T* A, ulong ars, ulong acs, __global const T* B, ulong brs, "
                          "ulong bcs, __global T* P, int M, int K, int N, int chunk",
                          "    const int kz = get_global_id(2);\n", strided_a, strided_b,
                          "P[(ulong)kz * M * N + i] = v", "kz * chunk", "min(K, kz * chunk + chunk)");
    // `batch` products at once, the batch the third dimension of the launch.
    src += product_kernel("bmm", 4,
                          "__global const T* A0, __global const T* B0, __global T* C0, int M, int K, int N",
                          "    const ulong bz = get_global_id(2);\n"
                          "    __global const T* A = A0 + bz * (ulong)M * K;\n"
                          "    __global const T* B = B0 + bz * (ulong)K * N;\n"
                          "    __global T* C = C0 + bz * (ulong)M * N;\n",
                          "A[(ulong)r * K + k]", "B[(ulong)k * N + c]", "C[i] = v", "0", "K");
    return src;
}

/// The fixed kernels, built for one element type.
struct Program {
    cl_program program = nullptr;
    cl_kernel matmul[3] = {nullptr, nullptr, nullptr};  // by `wpt_index`
    cl_kernel matvec = nullptr, transpose = nullptr, gather = nullptr,
              matmul_part = nullptr, sum_parts = nullptr,
              im2col = nullptr, col2im = nullptr, bmm = nullptr, permute = nullptr, random_uniform = nullptr;
};

/// The two kernels generated for one fused program: the elementwise one, and
/// the reduction that computes the same values and folds them instead of
/// storing them.
///
/// A program that starts from a product gets the product kernels instead,
/// with the program applied to each element as it is stored: `mm` for a
/// product with more than one column, `mv` for one with exactly one.
struct Fused {
    cl_program program = nullptr;
    cl_kernel map = nullptr, reduce = nullptr, axis = nullptr, argaxis = nullptr, mv = nullptr;
    cl_kernel mm[3] = {nullptr, nullptr, nullptr};  // by `wpt_index`
};

struct Device {
    Api cl{};
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    Program programs[2];  // by Dtype
    /// Fused kernels by their source, which names the dtype. A program is
    /// built the first time it is met, and a loop meets the same one every
    /// iteration -- its constants are arguments, not part of the source.
    std::unordered_map<std::string, Fused> fused;
    bool f64 = false;
    size_t tile = 16;
    std::atomic<size_t> memory{0};  // CL_DEVICE_GLOBAL_MEM_SIZE, once ready
    size_t group = 64;    // the reduction's work-group size, a power of two
    std::string name;
    std::string why;      // empty when usable
    /// How far ahead of the device the host may run. See `Launch::run`.
    unsigned launches = 0;
    cl_event previous_marker = nullptr;
    /// Device memory allocated since the last marker. Launches alone did not
    /// bound what was in flight: a convolution's launches each touch a
    /// hundred megabytes, so two windows of them held gigabytes of released
    /// buffers that OpenCL could not free yet, and an 8 GB card ran out.
    std::atomic<size_t> since_marker{0};
    /// clSetKernelArg is not safe on one kernel from two threads, and every
    /// process that computes on the GPU shares these kernels. One lock around
    /// set-and-enqueue is all the ordering the queue needs.
    std::mutex lock;
};

Device& dev() {
    static Device d;
    return d;
}

std::string cl_failure(const char* what, cl_int code) {
    return std::string("OpenCL failed in ") + what + " (error " + std::to_string(code) + ")";
}

void* open_library(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(void* lib) {
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

/// Prefer the system ICD loader. On Linux, also discover registered vendor
/// libraries: some installations ship these without libOpenCL (notably NixOS).
std::vector<std::string> library_names() {
    if (const char* forced = std::getenv("DREAM_OPENCL_LIB")) return {forced};
#if defined(_WIN32)
    return {"OpenCL.dll"};
#elif defined(__APPLE__)
    return {"/System/Library/Frameworks/OpenCL.framework/OpenCL"};
#else
    std::vector<std::string> names = {"libOpenCL.so.1", "libOpenCL.so",
                                    "/run/opengl-driver/lib/libOpenCL.so.1"};
    std::vector<std::string> dirs = {"/etc/OpenCL/vendors", "/run/opengl-driver/etc/OpenCL/vendors"};
    if (const char* override_dir = std::getenv("OCL_ICD_VENDORS")) dirs = {override_dir};
    for (const auto& dir : dirs) {
        std::error_code ec;
        std::vector<std::filesystem::path> files;
        for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".icd") files.push_back(it->path());
        std::sort(files.begin(), files.end());
        for (const auto& path : files) {
            std::ifstream file(path);
            std::string name;
            std::getline(file, name);
            const auto first = name.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || name[first] == '#') continue;
            name = name.substr(first, name.find_last_not_of(" \t\r\n") - first + 1);
            if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
            if (name.find('/') == std::string::npos)
                names.push_back(std::string("/run/opengl-driver/lib/") + name);
        }
    }
    return names;
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

// cl_khr_icd's dispatch table is an append-only ABI. Use only the entries
// through OpenCL 1.2, after checking the platform version. A vendor need not
// export the core functions (NVIDIA does not); each platform carries its table.
// Slots: https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_icd.h
bool load_api(Api& api, void* lib, void* const* dispatch, std::string* why) {
#define DREAM_CL_LOAD(field, name, slot)                                   \
    api.field = reinterpret_cast<decltype(api.field)>(                     \
        dispatch ? dispatch[slot] : symbol(lib, name));                    \
    if (!api.field) {                                                      \
        *why = std::string("the OpenCL library has no ") + name;           \
        return false;                                                     \
    }
    DREAM_CL_LOAD(GetDeviceIDs, "clGetDeviceIDs", 2)
    DREAM_CL_LOAD(GetDeviceInfo, "clGetDeviceInfo", 3)
    DREAM_CL_LOAD(CreateContext, "clCreateContext", 4)
    DREAM_CL_LOAD(CreateCommandQueue, "clCreateCommandQueue", 9)
    DREAM_CL_LOAD(CreateBuffer, "clCreateBuffer", 14)
    DREAM_CL_LOAD(ReleaseMemObject, "clReleaseMemObject", 18)
    DREAM_CL_LOAD(EnqueueWriteBuffer, "clEnqueueWriteBuffer", 49)
    DREAM_CL_LOAD(EnqueueReadBuffer, "clEnqueueReadBuffer", 48)
    DREAM_CL_LOAD(EnqueueCopyBuffer, "clEnqueueCopyBuffer", 50)
    DREAM_CL_LOAD(CreateProgramWithSource, "clCreateProgramWithSource", 26)
    DREAM_CL_LOAD(BuildProgram, "clBuildProgram", 30)
    DREAM_CL_LOAD(GetProgramBuildInfo, "clGetProgramBuildInfo", 33)
    DREAM_CL_LOAD(CreateKernel, "clCreateKernel", 34)
    DREAM_CL_LOAD(SetKernelArg, "clSetKernelArg", 38)
    DREAM_CL_LOAD(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel", 59)
    DREAM_CL_LOAD(Finish, "clFinish", 47)
    DREAM_CL_LOAD(Flush, "clFlush", 46)
    DREAM_CL_LOAD(EnqueueMarkerWithWaitList, "clEnqueueMarkerWithWaitList", 105)
    DREAM_CL_LOAD(WaitForEvents, "clWaitForEvents", 41)
    DREAM_CL_LOAD(ReleaseEvent, "clReleaseEvent", 44)
#undef DREAM_CL_LOAD
    return true;
}

/// Keep each platform's functions together: direct vendor libraries can have
/// different dispatch tables. GPUs across all candidates precede CPU devices.
bool pick_device(Device& d, std::string* why) {
    struct Candidate { Api api; cl_platform_id platform; };
    std::vector<Candidate> candidates;
    *why = "no usable OpenCL device was found (install your GPU's OpenCL runtime, "
           "or set DREAM_OPENCL_LIB to its loader or vendor library)";
    for (const auto& name : library_names()) {
        void* lib = open_library(name.c_str());
        if (!lib) continue;
        auto platforms = reinterpret_cast<decltype(Api::GetPlatformIDs)>(symbol(lib, "clGetPlatformIDs"));
        const bool vendor = !platforms;
        if (vendor) {
            using Extension = void*(DREAM_CL_API*)(const char*);
            auto extension = reinterpret_cast<Extension>(symbol(lib, "clGetExtensionFunctionAddress"));
            if (extension)
                platforms = reinterpret_cast<decltype(platforms)>(extension("clIcdGetPlatformIDsKHR"));
        }
        cl_uint count = 0;
        if (!platforms || platforms(0, nullptr, &count) != CL_SUCCESS || !count) {
            *why = name + ": OpenCL reports no platforms";
            close_library(lib);
            continue;
        }
        std::vector<cl_platform_id> ids(count);
        if (platforms(count, ids.data(), nullptr) != CL_SUCCESS) {
            *why = name + ": OpenCL platform enumeration failed";
            close_library(lib);
            continue;
        }
        bool used = false;
        for (auto id : ids) {
            if (!id) continue;
            void* const* dispatch = nullptr;
            if (vendor) {
                dispatch = *reinterpret_cast<void* const* const*>(id);
                using Info = cl_int(DREAM_CL_API*)(cl_platform_id, cl_uint, size_t, void*, size_t*);
                if (!dispatch) continue;
                auto info = reinterpret_cast<Info>(dispatch[1]);
                char version[128] = {};
                unsigned major = 0, minor = 0;
                if (!info || info(id, 0x0901 /* CL_PLATFORM_VERSION */, sizeof version, version, nullptr) != CL_SUCCESS ||
                    std::sscanf(version, "OpenCL %u.%u", &major, &minor) != 2 ||
                    major < 1 || (major == 1 && minor < 2)) {
                    *why = name + ": OpenCL 1.2 or later is required";
                    continue;
                }
            }
            Api api{};
            if (!load_api(api, lib, dispatch, why)) continue;
            api.GetPlatformIDs = platforms;
            candidates.push_back({api, id});
            used = true;
        }
        // Keep usable libraries loaded for the runtime's lifetime.
        if (!used) close_library(lib);
        // The system loader already enumerates its registered vendors. Only
        // consult direct vendors if it has no usable GPU.
        for (const auto& candidate : candidates) {
            cl_device_id id = nullptr;
            if (candidate.api.GetDeviceIDs(candidate.platform, CL_DEVICE_TYPE_GPU, 1, &id, nullptr) == CL_SUCCESS && id) {
                d.cl = candidate.api;
                d.device = id;
                d.platform = candidate.platform;
                return true;
            }
        }
    }
    for (const auto& candidate : candidates) {
        cl_device_id id = nullptr;
        if (candidate.api.GetDeviceIDs(candidate.platform, CL_DEVICE_TYPE_ALL, 1, &id, nullptr) == CL_SUCCESS && id) {
            d.cl = candidate.api;
            d.device = id;
            d.platform = candidate.platform;
            return true;
        }
    }
    return false;
}

/// Compile `source` with `options`, saying what went wrong in `why`.
cl_program compile(Device& d, const std::string& source, const std::string& options,
                   std::string* why) {
    cl_int err = 0;
    const char* src = source.c_str();
    cl_program program = d.cl.CreateProgramWithSource(d.context, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS) {
        *why = cl_failure("clCreateProgramWithSource", err);
        return nullptr;
    }
    err = d.cl.BuildProgram(program, 1, &d.device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t len = 0;
        d.cl.GetProgramBuildInfo(program, d.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        d.cl.GetProgramBuildInfo(program, d.device, CL_PROGRAM_BUILD_LOG, len, log.data(),
                                 nullptr);
        *why = "the GPU kernels did not build: " + log;
        return nullptr;
    }
    return program;
}

bool kernel(Device& d, cl_program program, const char* name, cl_kernel* out, std::string* why) {
    cl_int err = 0;
    *out = d.cl.CreateKernel(program, name, &err);
    if (err != CL_SUCCESS) *why = cl_failure(name, err);
    return err == CL_SUCCESS;
}

std::string options_for(const Device& d, int dtype) {
    return std::string(dtype == F32 ? "-DT=float" : "-DT=double -DFP64") +
           " -DTS=" + std::to_string(d.tile);
}

bool build(Device& d, Program& prog, int dtype, std::string* why) {
    prog.program = compile(d, fixed_source(), options_for(d, dtype), why);
    return prog.program && kernel(d, prog.program, "matmul2", &prog.matmul[0], why) &&
           kernel(d, prog.program, "matmul4", &prog.matmul[1], why) &&
           kernel(d, prog.program, "matmul8", &prog.matmul[2], why) &&
           kernel(d, prog.program, "matvec", &prog.matvec, why) &&
           kernel(d, prog.program, "matmul_part", &prog.matmul_part, why) &&
           kernel(d, prog.program, "sum_parts", &prog.sum_parts, why) &&
           kernel(d, prog.program, "transpose", &prog.transpose, why) &&
           kernel(d, prog.program, "gather", &prog.gather, why) &&
           kernel(d, prog.program, "random_uniform", &prog.random_uniform, why) &&
           kernel(d, prog.program, "im2col", &prog.im2col, why) &&
           kernel(d, prog.program, "col2im", &prog.col2im, why) &&
           kernel(d, prog.program, "bmm", &prog.bmm, why) &&
           kernel(d, prog.program, "permute", &prog.permute, why);
}

void init(Device& d) {
    std::string why;
    if (!pick_device(d, &why)) {
        d.why = why;
        return;
    }
    char name[256] = {0};
    d.cl.GetDeviceInfo(d.device, CL_DEVICE_NAME, sizeof name - 1, name, nullptr);
    d.name = name;
    size_t ext_len = 0;
    d.cl.GetDeviceInfo(d.device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
    std::string ext(ext_len, '\0');
    d.cl.GetDeviceInfo(d.device, CL_DEVICE_EXTENSIONS, ext_len, ext.data(), nullptr);
    d.f64 = ext.find("cl_khr_fp64") != std::string::npos;
    cl_ulong mem = 0;
    d.cl.GetDeviceInfo(d.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, nullptr);
    size_t max_group = 0;
    d.cl.GetDeviceInfo(d.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof max_group, &max_group,
                       nullptr);
    // A 16 x 16 tile wants 256 work-items in a group, which every desktop GPU
    // allows; a smaller device gets 8 x 8.
    d.tile = max_group >= 256 ? 16 : 8;
    d.group = 64;
    while (d.group > max_group && d.group > 1) d.group /= 2;

    cl_int err = 0;
    const intptr_t properties[] = {0x1084 /* CL_CONTEXT_PLATFORM */,
                                   reinterpret_cast<intptr_t>(d.platform), 0};
    d.context = d.cl.CreateContext(properties, 1, &d.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        d.why = cl_failure("clCreateContext", err);
        return;
    }
    d.queue = d.cl.CreateCommandQueue(d.context, d.device, 0, &err);
    if (err != CL_SUCCESS) {
        d.why = cl_failure("clCreateCommandQueue", err);
        return;
    }
    if (!build(d, d.programs[F32], F32, &why)) {
        d.why = why;
        return;
    }
    if (d.f64 && !build(d, d.programs[F64], F64, &why)) d.f64 = false;
    d.memory.store(size_t(mem), std::memory_order_release);
}

/// Wait for everything queued. Registered with `atexit` once a device is
/// set up: the queue is asynchronous, so a program can end with work still
/// in flight, and `exit` then runs the OpenCL library's destructors while
/// its own threads are still running that work -- POCL was caught compiling
/// a kernel with its LLVM state destroyed underneath it, about one exit in
/// thirty. An `atexit` handler runs before the libraries are torn down.
void drain_at_exit() {
    Device& d = dev();
    if (d.queue) d.cl.Finish(d.queue);
}

Device* ready(std::string* err) {
    static std::once_flag once;
    Device& d = dev();
    std::call_once(once, [&] {
        init(d);
        if (d.queue) std::atexit(drain_at_exit);
    });
    if (!d.why.empty()) {
        *err = d.why;
        return nullptr;
    }
    return &d;
}

Program* program_for(Device& d, int dtype, std::string* err) {
    if (dtype == F64 && !d.f64) {
        *err = "this GPU (" + d.name + ") has no double precision; use tensor.gpu for float32";
        return nullptr;
    }
    return &d.programs[dtype];
}

size_t round_up(size_t n, size_t m) { return (n + m - 1) / m * m; }

/// The block a product of `M x N` is computed in: the largest of `kWpts`
/// that still makes enough work-groups to occupy the device, which is the
/// smallest when none does.
size_t pick_wpt(const Device& d, size_t M, size_t N) {
    for (size_t k = 3; k-- > 1;) {
        const size_t span = d.tile * kWpts[k];
        if (((M + span - 1) / span) * ((N + span - 1) / span) >= 96) return kWpts[k];
    }
    return kWpts[0];
}

/// The index in `kWpts` of a block size.
size_t wpt_index(size_t w) { return w == 2 ? 0 : (w == 4 ? 1 : 2); }

/// The launch of a product over an `M x N` answer in `w x w` blocks: a
/// work-group of `tile x tile` work-items per `tile * w` square of C.
void product_grid(const Device& d, size_t M, size_t N, size_t w, size_t* global, size_t* group) {
    const size_t span = d.tile * w;
    global[0] = (N + span - 1) / span * d.tile;
    global[1] = (M + span - 1) / span * d.tile;
    group[0] = group[1] = d.tile;
}

}  // namespace

struct Buffer {
    cl_mem mem;
    size_t bytes;
    std::atomic<int> refs;
};

bool available(std::string* why) { return ready(why) != nullptr; }

std::string device_name() {
    std::string ignored;
    Device* d = ready(&ignored);
    return d ? d->name : std::string();
}

size_t memory_bytes() { return dev().memory.load(std::memory_order_acquire); }

bool has_f64() {
    std::string ignored;
    Device* d = ready(&ignored);
    return d && d->f64;
}

size_t dtype_size(int dtype) { return dtype == F32 ? sizeof(float) : sizeof(double); }

Buffer* alloc(size_t bytes, std::string* err) {
    Device* d = ready(err);
    if (!d) return nullptr;
    cl_int code = 0;
    // OpenCL refuses a zero-sized buffer, and a tensor is never empty, but a
    // partial result can be: one byte is a buffer nothing reads.
    cl_mem mem = d->cl.CreateBuffer(d->context, CL_MEM_READ_WRITE, bytes ? bytes : 1, nullptr, &code);
    if (code != CL_SUCCESS) {
        *err = cl_failure("clCreateBuffer", code) + " allocating " + std::to_string(bytes) +
               " bytes of device memory";
        return nullptr;
    }
    auto* b = new Buffer{mem, bytes, {1}};
    d->since_marker.fetch_add(bytes, std::memory_order_relaxed);
    return b;
}

void retain(Buffer* b) { b->refs.fetch_add(1, std::memory_order_relaxed); }

void release(Buffer* b) {
    if (b->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    dev().cl.ReleaseMemObject(b->mem);
    delete b;
}

bool upload(Buffer* dst, int dtype, const double* src, size_t n, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    std::lock_guard<std::mutex> g(d->lock);
    cl_int code;
    if (dtype == F32) {
        std::vector<float> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = float(src[i]);
        code = d->cl.EnqueueWriteBuffer(d->queue, dst->mem, CL_TRUE, 0, n * sizeof(float),
                                        tmp.data(), 0, nullptr, nullptr);
    } else {
        code = d->cl.EnqueueWriteBuffer(d->queue, dst->mem, CL_TRUE, 0, n * sizeof(double), src, 0,
                                        nullptr, nullptr);
    }
    if (code != CL_SUCCESS) *err = cl_failure("clEnqueueWriteBuffer", code);
    return code == CL_SUCCESS;
}

bool download(Buffer* src, int dtype, double* dst, size_t n, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    std::lock_guard<std::mutex> g(d->lock);
    cl_int code;
    if (dtype == F32) {
        std::vector<float> tmp(n);
        code = d->cl.EnqueueReadBuffer(d->queue, src->mem, CL_TRUE, 0, n * sizeof(float),
                                       tmp.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < n; ++i) dst[i] = tmp[i];
    } else {
        code = d->cl.EnqueueReadBuffer(d->queue, src->mem, CL_TRUE, 0, n * sizeof(double), dst, 0,
                                       nullptr, nullptr);
    }
    if (code != CL_SUCCESS) *err = cl_failure("clEnqueueReadBuffer", code);
    return code == CL_SUCCESS;
}

bool copy(Buffer* src, int dtype, size_t offset, Buffer* dst, size_t n, std::string* err,
          size_t dst_offset) {
    Device* d = ready(err);
    if (!d) return false;
    std::lock_guard<std::mutex> g(d->lock);
    const size_t w = dtype_size(dtype);
    cl_int code = d->cl.EnqueueCopyBuffer(d->queue, src->mem, dst->mem, offset * w,
                                          dst_offset * w, n * w, 0, nullptr, nullptr);
    if (code != CL_SUCCESS) *err = cl_failure("clEnqueueCopyBuffer", code);
    return code == CL_SUCCESS;
}

namespace {

/// Arguments in order, then the launch. Everything under the device's lock.
struct Launch {
    Device& d;
    cl_kernel k;
    cl_uint at = 0;
    cl_int code = CL_SUCCESS;
    template <class X>
    Launch& arg(const X& x) {
        if (code == CL_SUCCESS) code = d.cl.SetKernelArg(k, at, sizeof(X), &x);
        ++at;
        return *this;
    }
    Launch& local(size_t bytes) {
        if (code == CL_SUCCESS) code = d.cl.SetKernelArg(k, at, bytes, nullptr);
        ++at;
        return *this;
    }
    /// The element type's scalar: a float for an F32 program, a double for F64.
    Launch& scalar(int dtype, double s) {
        if (dtype == F32) return arg(float(s));
        return arg(s);
    }
    bool run(cl_uint dims, const size_t* global, const size_t* group, std::string* err) {
        if (code == CL_SUCCESS)
            code = d.cl.EnqueueNDRangeKernel(d.queue, k, dims, nullptr, global, group, 0, nullptr,
                                             nullptr);
        if (code != CL_SUCCESS) {
            *err = cl_failure("a GPU kernel launch", code);
            return false;
        }
        throttle();
        return true;
    }

    /// Keep at most two windows of launches in flight.
    ///
    /// A released buffer is freed only once every queued command that reads it
    /// has run, and the host enqueues far faster than any device computes: a
    /// loop of a thousand products queued a thousand results' worth of memory
    /// before the first was freed, and measured eight gigabytes for a program
    /// whose live set was eight megabytes. So every `kWindow` launches drop a
    /// marker, and wait for the marker before it. The device is never idle on
    /// our account -- a whole window is always queued behind the one waited
    /// for -- and what is in flight, with the memory it pins, is bounded.
    ///
    /// A window also closes once it has allocated an eighth of the device's
    /// memory (at least 256 MB), so what two windows pin is bounded in bytes
    /// as well as in launches.
    void throttle() {
        constexpr unsigned kWindow = 32;
        const size_t byte_window =
            std::max<size_t>(size_t(256) << 20, d.memory.load(std::memory_order_relaxed) / 8);
        if (++d.launches % kWindow != 0 &&
            d.since_marker.load(std::memory_order_relaxed) < byte_window)
            return;
        d.since_marker.store(0, std::memory_order_relaxed);
        cl_event marker = nullptr;
        if (d.cl.EnqueueMarkerWithWaitList(d.queue, 0, nullptr, &marker) != CL_SUCCESS) return;
        d.cl.Flush(d.queue);
        if (d.previous_marker) {
            d.cl.WaitForEvents(1, &d.previous_marker);
            d.cl.ReleaseEvent(d.previous_marker);
        }
        d.previous_marker = marker;
    }
};

/// How many slices of K a product should be cut into: 1 for an ordinary
/// one. See `matmul`.
size_t split_k(const Device& d, size_t M, size_t K, size_t N) {
    // Counted in the work-groups `matmul_part` makes, 4 x 4 blocks each.
    const size_t span = d.tile * 4;
    const size_t tiles = ((M + span - 1) / span) * ((N + span - 1) / span);
    if (K < 1024 || tiles >= 64) return 1;
    // Enough slices to give the device a few hundred work-groups, each slice
    // at least 256 deep so that a slice is still mostly arithmetic.
    const size_t want = (256 + tiles - 1) / tiles;
    return std::max<size_t>(1, std::min(want, K / 256));
}

}  // namespace

bool splits_k(size_t M, size_t K, size_t N) {
    std::string ignored;
    Device* d = ready(&ignored);
    return d && N > 1 && split_k(*d, M, K, N) > 1;
}

namespace {

/// The product cut along K: each slice into its own partial product, then
/// the partials added. Called under the device's lock.
bool matmul_split(Device& d, Program& p, int dtype, const Operand& a, const Operand& b, Buffer* c,
                  size_t M, size_t K, size_t N, size_t parts, std::string* err) {
    const size_t w = dtype_size(dtype);
    cl_int code = 0;
    cl_mem partial = d.cl.CreateBuffer(d.context, CL_MEM_READ_WRITE, parts * M * N * w, nullptr, &code);
    d.since_marker.fetch_add(parts * M * N * w, std::memory_order_relaxed);
    if (code != CL_SUCCESS) {
        *err = cl_failure("clCreateBuffer", code) + " for a split product";
        return false;
    }
    const size_t chunk = ((K + parts - 1) / parts + d.tile - 1) / d.tile * d.tile;
    size_t global[3], group[3];
    product_grid(d, M, N, 4, global, group);
    global[2] = parts;
    group[2] = 1;
    bool ok = Launch{d, p.matmul_part}
                  .arg(a.buf->mem).arg(cl_ulong(a.rs)).arg(cl_ulong(a.cs))
                  .arg(b.buf->mem).arg(cl_ulong(b.rs)).arg(cl_ulong(b.cs))
                  .arg(partial).arg(cl_int(M)).arg(cl_int(K)).arg(cl_int(N)).arg(cl_int(chunk))
                  .run(3, global, group, err);
    if (ok) {
        const size_t n = M * N, g1 = round_up(n, 64);
        ok = Launch{d, p.sum_parts}.arg(partial).arg(c->mem).arg(cl_ulong(n)).arg(cl_int(parts))
                 .run(1, &g1, nullptr, err);
    }
    // Released now, freed by OpenCL once the kernels reading it have run.
    d.cl.ReleaseMemObject(partial);
    return ok;
}

}  // namespace

/// `C = A x B`.
///
/// **Split-K.** The tiled kernel gives each tile of C one work-group, which
/// walks the whole of K. A product with a small C and a long K -- a
/// convolution's kernel gradient is `[kh * kw * c, rows] x [rows,
/// filters]`, with `rows` a hundred thousand -- is then a handful of
/// work-groups each doing a hundred thousand steps in turn, on a device
/// built to run thousands at once: 46 ms a step for a 3 x 3 convolution of
/// 128 small images. When C has too few tiles to fill the device, K is cut
/// into slices computed side by side into partial products, which a second
/// kernel adds in order.
bool matmul(int dtype, Operand a, Operand b, Buffer* c, size_t M, size_t K, size_t N,
            std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> g(d->lock);
    if (N > 1) {
        const size_t parts = split_k(*d, M, K, N);
        if (parts > 1) return matmul_split(*d, *p, dtype, a, b, c, M, K, N, parts, err);
    }
    if (N == 1) {
        size_t global = round_up(M, 64);
        return Launch{*d, p->matvec}
            .arg(a.buf->mem).arg(cl_ulong(a.rs)).arg(cl_ulong(a.cs))
            .arg(b.buf->mem).arg(cl_ulong(b.rs))
            .arg(c->mem).arg(cl_int(M)).arg(cl_int(K))
            .run(1, &global, nullptr, err);
    }
    size_t global[2], group[2];
    const size_t w = pick_wpt(*d, M, N);
    product_grid(*d, M, N, w, global, group);
    return Launch{*d, p->matmul[wpt_index(w)]}
        .arg(a.buf->mem).arg(cl_ulong(a.rs)).arg(cl_ulong(a.cs))
        .arg(b.buf->mem).arg(cl_ulong(b.rs)).arg(cl_ulong(b.cs))
        .arg(c->mem).arg(cl_int(M)).arg(cl_int(K)).arg(cl_int(N))
        .run(2, global, group, err);
}

bool transpose(int dtype, Buffer* in, Buffer* out, size_t rows, size_t cols, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> g(d->lock);
    size_t global[2] = {round_up(cols, 16), round_up(rows, 16)};
    return Launch{*d, p->transpose}
        .arg(in->mem).arg(out->mem).arg(cl_int(rows)).arg(cl_int(cols))
        .run(2, global, nullptr, err);
}

bool random_uniform(int dtype, Buffer* out, size_t n, uint64_t seed, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> g(d->lock);
    const size_t global = round_up(n, 64);
    return Launch{*d, p->random_uniform}
        .arg(out->mem).arg(cl_ulong(n)).arg(cl_ulong(seed))
        .run(1, &global, nullptr, err);
}

bool gather(int dtype, Buffer* src, const uint32_t* rows, size_t nrows, size_t width, Buffer* dst,
            std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    // The row numbers go up as a buffer of their own, written before the
    // kernel is queued and released after: OpenCL keeps it until the
    // kernel that reads it has run.
    Buffer* idx = alloc(nrows * sizeof(uint32_t), err);
    if (!idx) return false;
    bool ok;
    {
        std::lock_guard<std::mutex> g(d->lock);
        cl_int code = d->cl.EnqueueWriteBuffer(d->queue, idx->mem, CL_TRUE, 0,
                                               nrows * sizeof(uint32_t), rows, 0, nullptr, nullptr);
        ok = code == CL_SUCCESS;
        if (!ok) *err = cl_failure("clEnqueueWriteBuffer", code);
        if (ok) {
            const size_t n = nrows * width, global = round_up(n, 64);
            ok = Launch{*d, p->gather}
                     .arg(src->mem).arg(idx->mem).arg(dst->mem).arg(cl_ulong(width)).arg(cl_ulong(n))
                     .run(1, &global, nullptr, err);
        }
    }
    release(idx);
    return ok;
}

namespace {

/// `im2col` or `col2im`, which take the same arguments and differ in what
/// one work-item is: a number of the rows, or a number of the image.
bool window_kernel(int dtype, bool forward, const Window& g, Buffer* in, Buffer* out,
                   std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> lock(d->lock);
    const size_t items = forward ? g.n * g.oh * g.ow * g.kh * g.kw * g.c : g.n * g.h * g.w * g.c;
    const size_t global = round_up(items, 64);
    return Launch{*d, forward ? p->im2col : p->col2im}
        .arg(in->mem).arg(out->mem).arg(cl_ulong(g.n)).arg(cl_ulong(g.h)).arg(cl_ulong(g.w))
        .arg(cl_ulong(g.c)).arg(cl_ulong(g.kh)).arg(cl_ulong(g.kw)).arg(cl_ulong(g.stride))
        .arg(cl_ulong(g.pad)).arg(cl_ulong(g.oh)).arg(cl_ulong(g.ow))
        .run(1, &global, nullptr, err);
}

}  // namespace

bool bmm(int dtype, Buffer* a, Buffer* b, Buffer* c, size_t batch, size_t M, size_t K, size_t N,
         std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> lock(d->lock);
    size_t global[3], group[3];
    product_grid(*d, M, N, 4, global, group);
    global[2] = batch;
    group[2] = 1;
    return Launch{*d, p->bmm}
        .arg(a->mem).arg(b->mem).arg(c->mem).arg(cl_int(M)).arg(cl_int(K)).arg(cl_int(N))
        .run(3, global, group, err);
}

bool permute(int dtype, Buffer* in, Buffer* out, size_t rank, const size_t* dims,
             const size_t* strides, size_t count, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    Program* p = program_for(*d, dtype, err);
    if (!p) return false;
    std::lock_guard<std::mutex> lock(d->lock);
    const size_t global = round_up(count, 64);
    Launch l{*d, p->permute};
    l.arg(in->mem).arg(out->mem).arg(cl_ulong(count)).arg(cl_int(rank));
    for (size_t k = 0; k < 6; ++k) l.arg(cl_ulong(k < rank ? dims[k] : 1));
    for (size_t k = 0; k < 6; ++k) l.arg(cl_ulong(k < rank ? strides[k] : 0));
    return l.run(1, &global, nullptr, err);
}

bool im2col(int dtype, const Window& g, Buffer* in, Buffer* out, std::string* err) {
    return window_kernel(dtype, true, g, in, out, err);
}

bool col2im(int dtype, const Window& g, Buffer* cols, Buffer* out, std::string* err) {
    return window_kernel(dtype, false, g, cols, out, err);
}

namespace {

/// One value function of a generated kernel: its source, the parameters it
/// adds to a kernel's list, and the arguments that pass them on.
struct ValueFn {
    std::string text, params, args;
};

/// `name(i, n, ..)`: the value of `prog` at position `i` of an answer of `n`,
/// with its inputs and constants named with `pre` so that several programs
/// can share one kernel's parameter list. A program that starts from a product
/// takes the product's element as `p`.
ValueFn value_function(const FuseProgram& prog, const std::string& name, const std::string& pre,
                       bool product) {
    ValueFn f;
    for (unsigned k = 0; k < prog.ninputs; ++k) {
        const std::string i = pre + std::to_string(k);
        f.params += ", __global const T* in" + i + ", ulong n" + i + ", ulong r" + i;
        f.args += ", in" + i + ", n" + i + ", r" + i;
    }
    for (unsigned k = 0; k < prog.nconsts; ++k) {
        f.params += ", T c" + pre + std::to_string(k);
        f.args += ", c" + pre + std::to_string(k);
    }
    // Each input is loaded the way the program reads it -- as it lies, as
    // its transpose, or both -- once, at the top.
    std::string loads;
    std::vector<bool> plain(prog.ninputs, false), turned(prog.ninputs, false);
    for (unsigned i = 0; i < prog.ncode; ++i) {
        if (prog.code[2 * i] == FUSE_LOAD) plain[prog.code[2 * i + 1]] = true;
        if (prog.code[2 * i] == FUSE_LOADT) turned[prog.code[2 * i + 1]] = true;
    }
    for (unsigned k = 0; k < prog.ninputs; ++k) {
        const std::string i = std::to_string(k), q = pre + i;
        if (plain[k])
            loads += "    T x" + i + " = in" + q + "[n" + q + " == n ? i : i % n" + q + "];\n";
        if (turned[k])
            loads += "    ulong o" + i + " = i % n" + q + ";\n"
                     "    T xt" + i + " = in" + q + "[(o" + i + " % r" + q + ") * (n" + q + " / r" + q +
                     ") + o" + i + " / r" + q + "];\n";
    }
    std::vector<std::string> stack;
    // By `KernelOp`: an infix operator, or a function of two (`%` and on), or
    // a comparison, which answers 1 or 0 as the host's kernels do.
    static const char* const binops[] = {" + ", " - ", " * ", " / ", "fmod", "fmax", "fmin",
                                         "pow", " < ", " <= ", " > ", " >= ", " == ", " != "};
    static const char* const fns[] = {"-", "sqrt", "exp", "log", "fabs", "tanh", "sin", "cos",
                                      "relu_", "sigmoid_", "floor", "ceil", "round", "sign_",
                                      "erf"};
    for (unsigned i = 0; i < prog.ncode; ++i) {
        const unsigned op = prog.code[2 * i], arg = prog.code[2 * i + 1];
        switch (op) {
            case FUSE_LOAD: stack.push_back("x" + std::to_string(arg)); break;
            case FUSE_CONST: stack.push_back("c" + pre + std::to_string(arg)); break;
            case FUSE_PRODUCT: stack.push_back("p"); break;
            case FUSE_LOADT: stack.push_back("xt" + std::to_string(arg)); break;
            case FUSE_LOADR: {
                // Each number repeated `r` times: `(i / r) % count`.
                const std::string q = pre + std::to_string(arg & 7);
                stack.push_back("in" + q + "[(i / (ulong)c" + pre + std::to_string(arg >> 3) +
                                ") % n" + q + "]");
                break;
            }
            case FUSE_BIN: {
                std::string y = stack.back();
                stack.pop_back();
                std::string& x = stack.back();
                if (arg <= KOP_DIV) x = "(" + x + binops[arg] + y + ")";
                else if (arg <= KOP_POW) x = std::string(binops[arg]) + "(" + x + ", " + y + ")";
                else x = "((" + x + binops[arg] + y + ") ? (T)1 : (T)0)";
                break;
            }
            default: stack.back() = std::string(fns[arg]) + "(" + stack.back() + ")"; break;
        }
    }
    f.text = "inline T " + name + "(ulong i, ulong n" + (product ? ", T p" : "") + f.params +
             ") {\n" + loads + "    return " + stack.back() + ";\n}\n";
    return f;
}

/// The OpenCL source for a fused program: a `value` function computing one
/// element from the inputs and constants, and the kernels around it. The
/// constants are arguments, so that `x * 0.5` and `x * 0.25` are one kernel
/// and a loop that changes a scalar every iteration builds nothing new.
///
/// A program that starts from a product gets the product kernels instead,
/// and an operand of that product that is itself a deferred program (`pa`,
/// `pb`) is computed by its own value function as the product loads it --
/// into the tile of local memory the kernel was going to copy it into, so a
/// chain feeding the product is never stored.
std::string fused_source(int dtype, const FuseProgram& prog, const FuseProgram* pa,
                         const FuseProgram* pb) {
    bool product = false;
    for (unsigned i = 0; i < prog.ncode; ++i) product = product || prog.code[2 * i] == FUSE_PRODUCT;
    const ValueFn v = value_function(prog, "value", "", product);
    std::string src = std::string("// ") + (dtype == F32 ? "f32" : "f64") + "\n" + kFusedPrelude +
                      v.text;
    if (product) {
        std::string params = v.params, args = v.args;
        std::string load_a = "A[r * ars + k * acs]";
        std::string load_b = "B[k * brs + c * bcs]";
        std::string load_ma = "A[i * ars + k * acs]", load_x = "x[k * xrs]";
        if (pa) {
            const ValueFn f = value_function(*pa, "va", "a", false);
            src += f.text;
            params += f.params;
            args += f.args;
            load_a = "va((ulong)r * K + k, (ulong)M * K" + f.args + ")";
            load_ma = "va((ulong)i * K + k, (ulong)M * K" + f.args + ")";
        }
        if (pb) {
            const ValueFn f = value_function(*pb, "vb", "b", false);
            src += f.text;
            params += f.params;
            args += f.args;
            load_b = "vb((ulong)k * N + c, (ulong)K * N" + f.args + ")";
            load_x = "vb((ulong)k, (ulong)K" + f.args + ")";
        }
        // The fixed product (`product_kernel`), with the loads of a computed
        // operand replaced by its value function and the store replaced by
        // the program applied to what would have been stored.
        for (size_t w : kWpts)
            src += product_kernel("mm" + std::to_string(w), w,
                              "__global const T* A, ulong ars, ulong acs, __global const T* B, ulong brs, "
                              "ulong bcs, __global T* C, int M, int K, int N, ulong n" + params,
                              "", load_a, load_b, "C[i] = value(i, n, v" + v.args + ")", "0", "K");
        src += "__kernel void mv(__global const T* A, ulong ars, ulong acs, __global const T* x,\n"
               "                 ulong xrs, __global T* y, int M, int K, ulong n" + params + ") {\n"
               "    int i = get_global_id(0);\n"
               "    if (i >= M) return;\n"
               "    T acc = (T)0;\n"
               "    for (int k = 0; k < K; ++k) acc += " + load_ma + " * " + load_x + ";\n"
               "    y[i] = value((ulong)i, n, acc" + v.args + ");\n}\n";
        return src;
    }
    const std::string& params = v.params;
    const std::string& args = v.args;
    src += "__kernel void map(__global T* out, ulong n" + params + ") {\n"
           "    ulong i = get_global_id(0);\n"
           "    if (i < n) out[i] = value(i, n" + args + ");\n}\n";
    src += "__kernel void reduce(__global T* partial, __local T* scratch, int op, ulong n" +
           params + ") {\n"
           "    ulong gid = get_global_id(0), gsz = get_global_size(0);\n"
           "    uint lid = get_local_id(0), lsz = get_local_size(0);\n"
           "    T acc = op == 0 ? (T)0 : value(0, n" + args + ");\n"
           "    for (ulong i = gid; i < n; i += gsz) acc = combine_(op, acc, value(i, n" + args + "));\n"
           "    scratch[lid] = acc;\n"
           "    barrier(CLK_LOCAL_MEM_FENCE);\n"
           "    for (uint s = lsz / 2; s > 0; s >>= 1) {\n"
           "        if (lid < s) scratch[lid] = combine_(op, scratch[lid], scratch[lid + s]);\n"
           "        barrier(CLK_LOCAL_MEM_FENCE);\n"
           "    }\n"
           "    if (lid == 0) partial[get_group_id(0)] = scratch[0];\n}\n";
    // A fold along an axis: one work-item per answer and slice of the axis,
    // folding the slice's values in order into `out[slice][answer]`. With one
    // slice as long as the axis that is the whole fold; with several, the
    // partials are folded again (see `fused_axis`). `argaxis` keeps where the
    // least (`op` 3) or greatest value first is, as a number, in one slice.
    src += "__kernel void axis(__global T* out, int op, ulong outer, ulong len, ulong inner, ulong chunk, ulong n" +
           params + ") {\n"
           "    ulong g = get_global_id(0), c = get_global_id(1);\n"
           "    if (g >= outer * inner) return;\n"
           "    ulong o = g / inner, i = g % inner;\n"
           "    ulong a0 = c * chunk, a1 = min(len, a0 + chunk);\n"
           "    T acc = value((o * len + a0) * inner + i, n" + args + ");\n"
           "    for (ulong a = a0 + 1; a < a1; ++a)\n"
           "        acc = combine_(op, acc, value((o * len + a) * inner + i, n" + args + "));\n"
           "    out[c * outer * inner + g] = acc;\n}\n";
    src += "__kernel void argaxis(__global T* out, int op, ulong outer, ulong len, ulong inner, ulong n" +
           params + ") {\n"
           "    ulong g = get_global_id(0);\n"
           "    if (g >= outer * inner) return;\n"
           "    ulong o = g / inner, i = g % inner, at = 0;\n"
           "    T best = value(o * len * inner + i, n" + args + ");\n"
           "    for (ulong a = 1; a < len; ++a) {\n"
           "        T v = value((o * len + a) * inner + i, n" + args + ");\n"
           "        if (op == 3 ? v < best : v > best) { best = v; at = a; }\n"
           "    }\n"
           "    out[g] = (T)at;\n}\n";
    return src;
}

/// The kernels for a program, built the first time it is asked for. Called
/// under the device's lock, which is what guards the cache.
Fused* fused_kernels(Device& d, int dtype, const FuseProgram& prog, std::string* why,
                     const FuseProgram* pa = nullptr, const FuseProgram* pb = nullptr) {
    std::string src = fused_source(dtype, prog, pa, pb);
    auto it = d.fused.find(src);
    if (it != d.fused.end()) return &it->second;
    Fused f;
    f.program = compile(d, src, options_for(d, dtype), why);
    if (!f.program) return nullptr;
    const bool product = src.find("__kernel void mm2(") != std::string::npos;
    if (product ? !kernel(d, f.program, "mm2", &f.mm[0], why) ||
                      !kernel(d, f.program, "mm4", &f.mm[1], why) ||
                      !kernel(d, f.program, "mm8", &f.mm[2], why) || !kernel(d, f.program, "mv", &f.mv, why)
                : !kernel(d, f.program, "map", &f.map, why) ||
                      !kernel(d, f.program, "reduce", &f.reduce, why) ||
                      !kernel(d, f.program, "axis", &f.axis, why) ||
                      !kernel(d, f.program, "argaxis", &f.argaxis, why))
        return nullptr;
    return &d.fused.emplace(std::move(src), f).first->second;
}

/// The inputs and constants every fused kernel takes after its own arguments.
void bind_program(Launch& l, int dtype, const FuseProgram& prog, Buffer* const* ins,
                  const size_t* counts, const size_t* rows, const double* consts) {
    for (unsigned k = 0; k < prog.ninputs; ++k)
        l.arg(ins[k]->mem).arg(cl_ulong(counts[k])).arg(cl_ulong(rows[k]));
    for (unsigned k = 0; k < prog.nconsts; ++k) l.scalar(dtype, consts[k]);
}

}  // namespace

bool fused(int dtype, const FuseProgram& prog, Buffer* const* ins, const size_t* counts,
           const size_t* rows, const double* consts, Buffer* out, size_t n, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    if (!program_for(*d, dtype, err)) return false;
    std::lock_guard<std::mutex> g(d->lock);
    Fused* f = fused_kernels(*d, dtype, prog, err);
    if (!f) return false;
    size_t global = round_up(n, 64);
    Launch l{*d, f->map};
    l.arg(out->mem).arg(cl_ulong(n));
    bind_program(l, dtype, prog, ins, counts, rows, consts);
    return l.run(1, &global, nullptr, err);
}

bool fused_matmul(int dtype, const FuseProgram& prog, Operand a, Operand b, size_t M, size_t K,
                  size_t N, Buffer* const* ins, const size_t* counts, const size_t* rows,
                  const double* consts, Buffer* out, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    if (!program_for(*d, dtype, err)) return false;
    std::lock_guard<std::mutex> g(d->lock);
    Fused* f = fused_kernels(*d, dtype, prog, err, a.program ? &a.program->prog : nullptr,
                             b.program ? &b.program->prog : nullptr);
    if (!f) return false;
    const cl_ulong n = cl_ulong(M) * N;
    // An operand computed by its own program has no buffer; the kernel's
    // parameter for one is still there and never read, so any buffer fills it.
    cl_mem amem = a.buf ? a.buf->mem : out->mem;
    cl_mem bmem = b.buf ? b.buf->mem : out->mem;
    auto bind_all = [&](Launch& l) {
        bind_program(l, dtype, prog, ins, counts, rows, consts);
        for (const OperandProgram* op : {a.program, b.program})
            if (op) bind_program(l, dtype, op->prog, op->ins, op->counts, op->rows, op->consts);
    };
    if (N == 1) {
        size_t global = round_up(M, 64);
        Launch l{*d, f->mv};
        l.arg(amem).arg(cl_ulong(a.rs)).arg(cl_ulong(a.cs)).arg(bmem).arg(cl_ulong(b.rs))
            .arg(out->mem).arg(cl_int(M)).arg(cl_int(K)).arg(n);
        bind_all(l);
        return l.run(1, &global, nullptr, err);
    }
    size_t global[2], group[2];
    const size_t w = pick_wpt(*d, M, N);
    product_grid(*d, M, N, w, global, group);
    Launch l{*d, f->mm[wpt_index(w)]};
    l.arg(amem).arg(cl_ulong(a.rs)).arg(cl_ulong(a.cs)).arg(bmem).arg(cl_ulong(b.rs))
        .arg(cl_ulong(b.cs)).arg(out->mem).arg(cl_int(M)).arg(cl_int(K)).arg(cl_int(N)).arg(n);
    bind_all(l);
    return l.run(2, global, group, err);
}

bool fused_axis(int op, int dtype, const FuseProgram& prog, Buffer* const* ins, const size_t* counts,
                const size_t* rows, const double* consts, size_t outer, size_t len, size_t inner,
                Buffer* out, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    if (!program_for(*d, dtype, err)) return false;
    std::lock_guard<std::mutex> g(d->lock);
    Fused* f = fused_kernels(*d, dtype, prog, err);
    if (!f) return false;
    const size_t answers = outer * inner;
    if (op >= KRED_ARGMIN) {
        size_t global = round_up(answers, 64);
        Launch l{*d, f->argaxis};
        l.arg(out->mem).arg(cl_int(op)).arg(cl_ulong(outer)).arg(cl_ulong(len)).arg(cl_ulong(inner))
            .arg(cl_ulong(outer * len * inner));
        bind_program(l, dtype, prog, ins, counts, rows, consts);
        return l.run(1, &global, nullptr, err);
    }
    // A long axis with few answers -- a bias's gradient is a hundred thousand
    // rows summed into thirty-two numbers -- is a few work-items each walking
    // the whole axis: 32 ms of a 34 ms training step. Such an axis is cut
    // into slices folded side by side, and the partials folded again, as the
    // product is cut along K. The partials are folded in slice order, so the
    // answer does not depend on how the device schedules them.
    size_t slices = 1;
    if (answers < 4096 && len >= 1024)
        slices = std::min<size_t>(len / 256, (8192 + answers - 1) / answers);
    const size_t chunk = (len + slices - 1) / slices;
    slices = (len + chunk - 1) / chunk;
    const size_t w = dtype_size(dtype);
    cl_mem target = out->mem;
    cl_int code = 0;
    if (slices > 1) {
        target = d->cl.CreateBuffer(d->context, CL_MEM_READ_WRITE, slices * answers * w, nullptr, &code);
        if (code != CL_SUCCESS) {
            *err = cl_failure("clCreateBuffer", code) + " for a split fold";
            return false;
        }
        d->since_marker.fetch_add(slices * answers * w, std::memory_order_relaxed);
    }
    size_t global[2] = {round_up(answers, 64), slices};
    Launch l{*d, f->axis};
    l.arg(target).arg(cl_int(op)).arg(cl_ulong(outer)).arg(cl_ulong(len)).arg(cl_ulong(inner))
        .arg(cl_ulong(chunk)).arg(cl_ulong(outer * len * inner));
    bind_program(l, dtype, prog, ins, counts, rows, consts);
    bool ok = l.run(2, global, nullptr, err);
    if (slices > 1) {
        if (ok) {
            // The partials as `[slices, answers]`, folded down their first
            // axis by the same kernel over a program that only loads them.
            static const uint8_t load[2] = {FUSE_LOAD, 0};
            const FuseProgram loads{load, 1, 1, 0, 1};
            Fused* pf = fused_kernels(*d, dtype, loads, err);
            ok = pf != nullptr;
            if (ok) {
                size_t g2[2] = {round_up(answers, 64), 1};
                ok = Launch{*d, pf->axis}
                         .arg(out->mem).arg(cl_int(op)).arg(cl_ulong(1)).arg(cl_ulong(slices))
                         .arg(cl_ulong(answers)).arg(cl_ulong(slices)).arg(cl_ulong(slices * answers))
                         .arg(target).arg(cl_ulong(slices * answers)).arg(cl_ulong(slices * answers))
                         .run(2, g2, nullptr, err);
            }
        }
        d->cl.ReleaseMemObject(target);
    }
    return ok;
}

bool fused_reduce(int op, int dtype, const FuseProgram& prog, Buffer* const* ins,
                  const size_t* counts, const size_t* rows, const double* consts, size_t n,
                  double* out, std::string* err) {
    Device* d = ready(err);
    if (!d) return false;
    if (!program_for(*d, dtype, err)) return false;
    // Enough groups to fill a GPU, few enough that the host's share is nothing.
    const size_t groups = std::min<size_t>(256, (n + d->group - 1) / d->group);
    const size_t w = dtype_size(dtype);
    Buffer* partial = alloc(groups * w, err);
    if (!partial) return false;
    bool ok;
    {
        std::lock_guard<std::mutex> g(d->lock);
        Fused* f = fused_kernels(*d, dtype, prog, err);
        ok = f != nullptr;
        if (ok) {
            size_t global = groups * d->group, group = d->group;
            Launch l{*d, f->reduce};
            l.arg(partial->mem).local(group * w).arg(cl_int(op)).arg(cl_ulong(n));
            bind_program(l, dtype, prog, ins, counts, rows, consts);
            ok = l.run(1, &global, &group, err);
        }
    }
    std::vector<double> parts(groups);
    if (ok) ok = download(partial, dtype, parts.data(), groups, err);
    release(partial);
    if (!ok) return false;
    double r = parts[0];
    for (size_t i = 1; i < groups; ++i) {
        switch (op) {
            case KRED_SUM: r += parts[i]; break;
            case KRED_MIN: r = parts[i] < r ? parts[i] : r; break;
            default: r = parts[i] > r ? parts[i] : r; break;
        }
    }
    *out = r;
    return true;
}

}  // namespace dream::gpu
