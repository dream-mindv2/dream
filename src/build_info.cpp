// What `dream --version` says this VM was built with.
//
// Asked here, in libdream, because the switches that decide it are the
// library's own (DREAM_HAVE_LLVM, DREAM_TENSOR_AVX2, the TLS source chosen) and
// the command line is compiled without them. Each part says what it actually
// is -- the LLVM the JIT was built against, the TLS library as it reports
// itself -- and a part that is found only when it is first used (OpenCL,
// WebKitGTK) is said to be, since "built with" would claim more than the
// binary holds.
#include "build_info.hpp"
#include "gpu.hpp"
#include "graphics.hpp"
#include "os.hpp"
#include "tls.hpp"
#ifdef DREAM_HAVE_LLVM
#include "llvm/Config/llvm-config.h"
#endif

namespace dream {

namespace {
std::string compiler() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}
}  // namespace

std::vector<std::pair<std::string, std::string>> build_features() {
    std::vector<std::pair<std::string, std::string>> out;
#ifdef DREAM_HAVE_LLVM
    out.push_back({"jit", "LLVM " LLVM_VERSION_STRING});
#else
    out.push_back({"jit", "none: interpreted only"});
#endif
    out.push_back({"tls", tls::backend()});
#if defined(FFI_STATIC_BUILD) || defined(DREAM_SELF_CONTAINED)
    out.push_back({"ffi", "libffi, linked in"});
#else
    out.push_back({"ffi", "libffi"});
#endif
    // What is loaded when first used is looked for now, so the line says
    // whether this machine has it, and which file it would be.
    const std::string opencl = gpu::library_found();
    out.push_back({"gpu", "OpenCL, loaded when a tensor first goes to the GPU: " +
                              (opencl.empty() ? std::string("no OpenCL library found here") : opencl)});
#if defined(__linux__) && !defined(__ANDROID__)
    const std::string webkit = graphics_library();
    out.push_back({"windows", "GTK 3 and WebKitGTK 4.1 or 4.0, loaded when an application first opens one: " +
                                  (webkit.empty() ? std::string("no WebKitGTK found here") : webkit)});
#else
    out.push_back({"windows", "none on this platform"});
#endif
#ifdef DREAM_TENSOR_AVX2
    out.push_back({"tensors", "baseline and AVX2 kernels, chosen at start"});
#else
    out.push_back({"tensors", "baseline kernels"});
#endif
    std::string build = compiler();
#ifdef DREAM_SELF_CONTAINED
    build += ", self-contained (only the C library is shared)";
#endif
#ifdef DREAM_PGO_USE
    build += ", profile-guided";
#endif
#ifdef DREAM_PROFILE_GENERATE
    build += ", instrumented for a profile";
#endif
#ifdef DREAM_TERMUX
    build += ", Termux";
#endif
#ifdef NDEBUG
    build += ", optimized";
#else
    build += ", with assertions";
#endif
    out.push_back({"build", build});
    // Where `dream -x`, the standard library and installed programs are found,
    // in the order searched: what `mind doctor` reads to say what a build
    // would compile with.
    std::string dirs;
    for (const std::string& d : install_dirs()) dirs += (dirs.empty() ? "" : ", ") + d;
    out.push_back({"install", dirs.empty() ? std::string("none") : dirs});
    return out;
}

}  // namespace dream
