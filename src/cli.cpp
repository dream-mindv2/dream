#include "windows.hpp"
#ifdef _WIN32
#include <shellapi.h>
#endif
// The `dream` command: load a bytecode image and run it.

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <filesystem>

#include "builtins.hpp"
#include "image.hpp"
#include <thread>
#include "interp.hpp"
#include "jit.hpp"
#include "os.hpp"
#include "process.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

using namespace dream;

// Which VM this is: a release number, and the commit it was built from when
// the build knew it (CMake asks git at configure time). A reproduced build
// names the VM it ran on, since the compiler's output is the compiler's and
// the VM's together.
#ifndef DREAM_VERSION
#define DREAM_VERSION "0.1.0"
#endif
#ifndef DREAM_COMMIT
#define DREAM_COMMIT "unknown"
#endif

namespace {

const char* USAGE =
    "dream -- the Dream virtual machine\n"
    "\n"
    "usage: dream [options] <image> [arguments for the program]\n"
    "\n"
    "<image> is a file, that file with `.dream` added, or either of those in\n"
    "the installation -- so `dream mind` runs ./mind.dream, or the installed one.\n"
    "The installation is $MINDV2_PATH; or lib/dream beside the directory this\n"
    "binary is in, as a package installs it; or ~/.mindv2.\n"
    "\n"
    "options:\n"
    "      --version        print which VM this is and exit\n"
    "  -x, --exec <name>    run <name>.dream from the installation, not from here\n"
    "  -e, --entry <name>   run this global instead of `main!`\n"
    "  -j, --workers <n>    scheduler threads (default: cores, capped at 8)\n"
    "      --dump           disassemble the image and exit\n"
    "      --stats          print reduction and heap statistics\n"
    "      --profile [n]    count reductions per function and print the hottest\n"
    "      --no-jit         stay in the interpreter\n"
    "      --any-target     run an image built for another platform anyway\n"
    "      --jit-threshold <n>  calls before a function is compiled\n"
    "      --precompile         compile every function the JIT can take, on every\n"
    "                           core, before running anything\n"
    "      --jit-threads <n>    threads compiling in the background (default 2;\n"
    "                           also $DREAM_JIT_THREADS)\n"
    "      --jit-policy <name>   density (default), calls, calls_times_size, work,\n"
    "                           calls_per_size_squared, or adaptive; overrides\n"
    "                           $DREAM_JIT_RANK\n"
    "      --max-heap <size>    heap one process may grow to before it fails with\n"
    "                           :out_of_memory: bytes, or 512M, 2G (default 1G;\n"
    "                           also $DREAM_MAX_HEAP)\n"
    "      --max-depth <n>      continuations one process may have pending\n"
    "                           (default 4M; also $DREAM_MAX_DEPTH)\n"
    "      --max-stack <n>      values one process may hold on its stack\n"
    "                           (default 4M; also $DREAM_MAX_STACK)\n"
    "      --nursery-max <size> the most a process's nursery grows to, in bytes or\n"
    "                           64M (default 32M; also $DREAM_NURSERY_MAX, then\n"
    "                           the program's own `[vm] nursery_max`)\n"
    "      --dump-jit <fn>  print the LLVM IR generated for a function, named or\n"
    "                       given as #N, its index -- generated functions share names\n"
    "      --mindv2         mindv2 path override\n"
    "      --mindv2-path    print the installation's directories and exit\n"
    "  -h, --help           show this message\n";
// --- the image's target ---------------------------------------------------------
//
// An image records the systems and architectures it may run on, as two sets of
// bits in its header (dreams/src/target.dr decides them). One built for another
// machine is refused here, before anything runs, because the alternative is a
// program that fails at its first `dlopen` with an error about a library it
// never mentioned. The check is the command's and not the loader's: the
// compiler runs images of the program it is compiling for `comp`, and a
// cross-build's target is not the machine doing the compiling.

/// This VM's bit in each set, fixed by the format.
constexpr uint8_t HOST_OS =
#if defined(__ANDROID__)
    0x08;
#elif defined(__linux__)
    0x01;
#elif defined(__APPLE__)
    0x02;
#elif defined(_WIN32)
    0x04;
#else
    0x00;
#endif
constexpr uint8_t HOST_ARCH =
#if defined(__x86_64__) || defined(_M_X64)
    0x01;
#elif defined(__aarch64__) || defined(_M_ARM64)
    0x02;
#else
    0x00;
#endif

/// `linux, windows`, for a set; the names are the ones `--target` takes.
std::string target_names(uint8_t bits, const char* const names[], size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        if (!(bits & (1u << i))) continue;
        if (!out.empty()) out += ", ";
        out += names[i];
    }
    return out;
}

const char* const OS_NAMES[] = {"linux", "macos", "windows", "android"};
const char* const ARCH_NAMES[] = {"x86_64", "aarch64"};

/// `linux (x86_64)`: a set as a person reads it, with an empty side left out.
std::string describe_target(uint8_t os, uint8_t arch) {
    if (!os) return arch ? target_names(arch, ARCH_NAMES, 2) : "any platform";
    std::string out = target_names(os, OS_NAMES, 4);
    if (arch) out += " (" + target_names(arch, ARCH_NAMES, 2) + ")";
    return out;
}

/// Empty when the image may run here, and otherwise why not. A set is either
/// empty, meaning any, or has to contain this machine's bit.
std::string target_mismatch(const Image& img, const std::string& name) {
    const uint8_t os = img.target_os(), arch = img.target_arch();
    const bool os_ok = os == 0 || (os & HOST_OS) != 0;
    const bool arch_ok = arch == 0 || (arch & HOST_ARCH) != 0;
    if (os_ok && arch_ok) return {};
    return name + " was built for " + describe_target(os, arch) + "; this is " +
           describe_target(HOST_OS, HOST_ARCH) +
           ".\nRebuild it for this platform, or run it anyway with --any-target.";
}

void dump_node(const Image& img, uint32_t idx, int depth, std::string& out);

void indent(std::string& out, int depth) { out.append(size_t(depth) * 2, ' '); }

void dump_kids(const Image& img, uint32_t off, uint32_t count, int depth, std::string& out) {
    for (uint32_t i = 0; i < count; ++i) dump_node(img, img.kid(off + i), depth, out);
}

void dump_node(const Image& img, uint32_t idx, int depth, std::string& out) {
    if (idx == NO_NODE) {
        indent(out, depth);
        out += "<none>\n";
        return;
    }
    const Node& n = img.node(idx);
    indent(out, depth);
    char buf[128];
    std::snprintf(buf, sizeof buf, "%%%u %s", idx, op_name(Op(n.op)));
    out += buf;

    switch (Op(n.op)) {
        case Op::ConstInt:
            std::snprintf(buf, sizeof buf, " %" PRId64, img.integer(n.a));
            out += buf;
            break;
        case Op::ConstFloat:
            std::snprintf(buf, sizeof buf, " %g", img.real(n.a));
            out += buf;
            break;
        case Op::ConstStr: out += " \"" + img.str(n.a).str() + "\""; break;
        case Op::ConstAtom: out += " :" + img.atom_name(n.a).str(); break;
        case Op::ConstBool: out += n.a ? " true" : " false"; break;
        case Op::ConstChar:
            std::snprintf(buf, sizeof buf, " U+%04X", n.a);
            out += buf;
            break;
        case Op::Local:
        case Op::Capture:
            std::snprintf(buf, sizeof buf, " %u", n.a);
            out += buf;
            break;
        case Op::Global:
            out += " @" + img.str(img.global(n.a).name).str();
            break;
        case Op::Builtin: out += std::string(" ") + builtin_def(n.a).name; break;
        case Op::Field: out += " ." + img.str(n.b).str(); break;
        case Op::MakeClosure:
        case Op::MakeThunk:
            out += " fn#" + std::to_string(n.a) + " " + img.str(img.func(n.a).name).str();
            break;
        case Op::Apply:
            std::snprintf(buf, sizeof buf, "/%u", n.c);
            out += buf;
            break;
        case Op::SwitchHead:
        case Op::SwitchAtom:
            std::snprintf(buf, sizeof buf, "/%u", n.c / 2);
            out += buf;
            break;
        case Op::Block:
        case Op::MakeList:
        case Op::MakeArray:
        case Op::MakeMap:
        case Op::RecordMake:
            std::snprintf(buf, sizeof buf, "/%u", n.b);
            out += buf;
            break;
        case Op::SetMany:
            std::snprintf(buf, sizeof buf, "/%u", n.c);
            out += buf;
            break;
        case Op::Bind:
            std::snprintf(buf, sizeof buf, " $%u", n.a);
            out += buf;
            break;
        default: break;
    }

    std::string flags;
    if (n.flags & F_STRICT) flags += "strict,";
    if (n.flags & F_IMPURE) flags += "impure,";
    if (n.flags & F_TAIL) flags += "tail,";
    if (!flags.empty()) {
        flags.pop_back();
        out += "  ; " + flags;
    }
    out += "\n";

    switch (Op(n.op)) {
        case Op::Field:
        case Op::Force:
        case Op::Neg:
        case Op::Not:
        case Op::BitNot:
        case Op::TypeIs:
            dump_node(img, n.a, depth + 1, out);
            break;
        case Op::Apply:
        case Op::SwitchHead:
        case Op::SwitchAtom:
            dump_node(img, n.a, depth + 1, out);
            dump_kids(img, n.b, n.c, depth + 1, out);
            break;
        case Op::If:
            dump_node(img, n.a, depth + 1, out);
            dump_node(img, n.b, depth + 1, out);
            dump_node(img, n.c, depth + 1, out);
            break;
        case Op::Block:
        case Op::MakeList:
        case Op::MakeArray:
        case Op::RecordMake:
            dump_kids(img, n.a, n.b, depth + 1, out);
            break;
        case Op::MakeMap:
            dump_kids(img, n.a, n.b * 2, depth + 1, out);
            break;
        case Op::Bind:
            dump_node(img, n.b, depth + 1, out);
            break;
        case Op::Try: case Op::Ensure:
            dump_node(img, n.a, depth + 1, out);
            dump_node(img, n.b, depth + 1, out);
            break;
        case Op::Get: case Op::Set:
            dump_node(img, n.a, depth + 1, out);
            dump_node(img, n.b, depth + 1, out);
            if (n.c != NO_NODE) dump_node(img, n.c, depth + 1, out);
            break;
        case Op::SetMany:
            dump_node(img, n.a, depth + 1, out);
            dump_kids(img, n.b, n.c * 2, depth + 1, out);
            break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or:
            dump_node(img, n.a, depth + 1, out);
            dump_node(img, n.b, depth + 1, out);
            break;
        default: break;
    }
}

void dump_image(const Image& img) {
    std::printf("; module \"%s\"  source \"%s\"  version %u.%u%s\n",
                img.module_name().str().c_str(), img.source_name().str().c_str(),
                img.version_major(), img.version_minor(),
                img.has_debug_info() ? "  (debug)" : "");
    if (img.entry() != NO_NODE) {
        std::printf("; entry fn#%u `%s`\n", img.entry(),
                    img.str(img.func(img.entry()).name).str().c_str());
    }
    for (uint32_t i = 0; i < img.import_count(); ++i) {
        const ImportRec& ir = img.import(i);
        std::printf("import %s as %s\n", img.str(ir.path).str().c_str(),
                    img.str(ir.alias).str().c_str());
    }
    for (uint32_t i = 0; i < img.func_count(); ++i) {
        const FuncRec& f = img.func(i);
        std::printf("\nfn#%u %s  arity %u  slots %u  captures %u  [%s%s%s%s]\n", i,
                    img.str(f.name).str().c_str(), f.arity, f.slots, f.n_captures,
                    (f.flags & FN_IMPURE) ? "impure" : "pure",
                    (f.flags & FN_REC) ? " rec" : "",
                    (f.flags & FN_THUNK) ? " thunk" : "",
                    (f.flags & FN_GLOBAL_VALUE) ? " memoized" : "");
        std::string out;
        dump_node(img, f.body, 1, out);
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
}

}  // namespace


/// Resolve the image to run.
///
/// `dream mind` should mean what `mind` means when it is typed on its own, so a
/// name is tried four ways, nearest first:
///
///   1. as written, which is what a path is;
///   2. with `.dream` added, so `dream mind` runs `./mind.dream` -- an image is
///      a program, and naming its extension every time is noise;
///   3. and 4., both of those under `$MINDV2_PATH`, where an installation keeps
///      what it ships, the way a shell finds a binary on PATH.
///
/// The working directory comes before the installation, because a project's own
/// build is what someone standing in it means. The installed lookup is for a
/// bare name only: a path with a separator in it is a place, and answering it
/// with a file from somewhere else would be a surprise. `dream -x NAME` is the
/// other half of this -- the installation and nothing else.
std::string skip_file(std::string& path, const std::string& MINDV2_PATH, bool* file_ok) {
    *file_ok = true;

    const bool is_bare_name =
        path.find('/') == std::string::npos && path.find('\\') == std::string::npos;

    for (const std::string& candidate : {path, path + ".dream"}) {
        if (is_file(candidate)) return candidate;
    }
    if (is_bare_name) {
        std::string found = installed_image(path, split_dir_list(MINDV2_PATH));
        if (!found.empty()) return found;
    }

    *file_ok = false;
    return "";
}

/// `-x name`: the installed image called `name`, and nothing else.
///
/// This is the same lookup `skip_file` falls back to, without the fallback. A
/// bare `dream lucid` prefers a file called `lucid` in the working directory,
/// which is what a path should mean; `-x lucid` says the installation is the
/// only place to look, so a stray file next to the caller cannot shadow an
/// installed program.
std::string installed_only(const std::string& name, const std::string& MINDV2_PATH,
                           bool* file_ok) {
    *file_ok = true;
    if (MINDV2_PATH.empty()) {
        std::fprintf(stderr,
                     "dream: -x needs $MINDV2_PATH (or ~/.mindv2) to look in, to find `%s`\n",
                     name.c_str());
        *file_ok = false;
        return "";
    }
    std::string found = installed_image(name, split_dir_list(MINDV2_PATH));
    if (found.empty()) {
        std::fprintf(stderr, "dream: no installed image `%s` in %s\n", name.c_str(),
                     MINDV2_PATH.c_str());
        *file_ok = false;
    }
    return found;
}

int dream_main(int argc, char** argv) {
    std::string path, entry;
    std::string dump_jit_fn;
    unsigned workers = 0;
    bool dump = false, stats = false, use_jit = true, any_target = false;
    size_t profile_top = 0;
    uint32_t jit_threshold = 0;
    unsigned jit_threads = 0;
    JitPolicy jit_policy = JitPolicy::CallsPerSize;
    bool jit_policy_set = false;
    bool precompile = false;
    size_t max_heap = 0, max_depth = 0, max_stack = 0, nursery_max = 0;

    std::vector<std::string> program_args;
    bool past_image = false;
    // `-x` has already resolved the image, so the working directory must not be
    // consulted for it again.
    bool installed = false;
    std::string MINDV2_PATH = install_path();
    
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        // Once the image is named, stop interpreting anything as a VM option.
        if (past_image) {
            program_args.push_back(a);
            continue;
        }
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "dream: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            std::fputs(USAGE, stdout);
            return 0;
        } else if (a == "--version") {
            std::printf("dream %s (%s)\n", DREAM_VERSION, DREAM_COMMIT);
            return 0;
        } else if (a == "-x" || a == "--exec") {
            bool ok = true;
            path = installed_only(next("--exec"), MINDV2_PATH, &ok);
            if (!ok) return 1;
            installed = true;
            past_image = true;
        } else if (a == "-e" || a == "--entry") {
            entry = next("--entry");
        } else if (a == "-j" || a == "--workers") {
            workers = unsigned(std::stoul(next("--workers")));
        } else if (a == "--dump") {
            dump = true;
        } else if (a == "--stats") {
            stats = true;
        } else if (a == "--profile") {
            // The count is optional: `--profile` alone shows a screenful.
            profile_top = 25;
            if (i + 1 < argc && argv[i + 1][0] != '-' && std::isdigit(argv[i + 1][0])) {
                profile_top = size_t(std::stoul(argv[++i]));
            }
        } else if (a == "--no-jit") {
            use_jit = false;
        } else if (a == "--any-target") {
            any_target = true;
        }else if (a == "-m" || a == "--mindv2") {
            MINDV2_PATH = expand_home(next("--mindv2"));
            // Said for the program too, and for what it runs: an image that
            // opens another by name (`std.image`'s `installed`) or finds the
            // standard library looks where the VM that ran it looked.
#ifdef _WIN32
            _putenv_s("MINDV2_PATH", MINDV2_PATH.c_str());
#else
            ::setenv("MINDV2_PATH", MINDV2_PATH.c_str(), 1);
#endif
        } else if (a == "--mindv2-path") {
            // The toolchain wrapper sets `$MINDV2_PATH` inside its own process,
            // so a caretaker (an editor extension, a shell script) cannot read
            // it from its own environment. Printing what the VM would resolve
            // names and packages under is how it is asked: the VM is the one
            // place the installed path is guaranteed to be right.
            std::printf("%s\n", MINDV2_PATH.c_str());
            return 0;
        } else if (a == "--jit-policy") {
            const std::string name = next("--jit-policy");
            if (!parse_jit_policy(name, jit_policy)) {
                std::fprintf(stderr, "dream: unknown JIT policy '%s'; choose density, calls, "
                                     "calls_times_size, work, calls_per_size_squared, or adaptive\n",
                             name.c_str());
                return 2;
            }
            jit_policy_set = true;
        } else if (a == "--jit-threshold") {
            jit_threshold = uint32_t(std::stoul(next("--jit-threshold")));
        } else if (a == "--precompile") {
            precompile = true;
        } else if (a == "--jit-threads") {
            jit_threads = unsigned(std::stoul(next("--jit-threads")));
        } else if (a == "--nursery-max") {
            const std::string text = next("--nursery-max");
            if (!parse_size(text.c_str(), &nursery_max)) {
                std::fprintf(stderr, "dream: --nursery-max takes a size such as 65536 or 64M, not `%s`\n",
                             text.c_str());
                return 2;
            }
        } else if (a == "--max-heap" || a == "--max-depth" || a == "--max-stack") {
            const std::string text = next(a.c_str());
            size_t n = 0;
            if (!parse_size(text.c_str(), &n)) {
                std::fprintf(stderr, "dream: %s takes a size such as 4096, 512M or 2G, not `%s`\n",
                             a.c_str(), text.c_str());
                return 2;
            }
            (a == "--max-heap" ? max_heap : a == "--max-depth" ? max_depth : max_stack) = n;
        } else if (a == "--dump-jit") {
            dump_jit_fn = next("--dump-jit");
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "dream: unknown option `%s`\n", a.c_str());
            return 2;
        } else if (path.empty()) {
            path = a;
            past_image = true;
        } else {
            // Everything after the image belongs to the program, not to the
            // VM -- including anything that looks like an option, which is why
            // this branch is reached before the `-` check can complain about
            // it. `std.os.args!` is where it arrives.
            program_args.push_back(a);
        }
    }

    // Before anything runs: the limits are read once, by the first process.
    override_limits(max_depth, max_stack, max_heap);

    if (path.empty()) {
        std::fputs(USAGE, stderr);
        return 2;
    }
    bool file_ok = true;
    // The name as it was typed, because that is the one worth reporting: what
    // the lookup answers on a miss is nothing at all.
    const std::string asked = path;
    if (!installed) path = skip_file(path, MINDV2_PATH, &file_ok);
    if (!file_ok) {
        std::fprintf(stderr, "dream: no such image `%s`\n", asked.c_str());
        return 1;
    }

    Runtime rt;
    // The runtime reports the run's cost, not this function: a program that
    // ends with `os.exit!` never comes back here, and every tool does.
    if (stats) rt.enable_stats();
    rt.set_program_args(std::move(program_args));
    std::string error;
    if (!rt.load_image_file(path, error)) {
        std::fprintf(stderr, "dream: %s\n", error.c_str());
        return 1;
    }
    // The nursery's cap, in the order the person running the program can
    // overrule the program: this command line, then the environment (which
    // `nursery_max_bytes` reads itself), then what the image carries from its
    // manifest. Before any process exists, so every heap is made with it.
    if (nursery_max) {
        set_nursery_max(nursery_max);
    } else if (!std::getenv("DREAM_NURSERY_MAX")) {
        if (const uint64_t asked = rt.image().vm_option(VMOP_NURSERY_MAX)) set_nursery_max(size_t(asked));
    }

    if (profile_top) rt.enable_profile(profile_top);

    // `--dump` only reads the image, so it is allowed whatever it was built
    // for; running it is what the target is about.
    if (!dump && !any_target) {
        const std::string why = target_mismatch(rt.image(), asked);
        if (!why.empty()) {
            std::fprintf(stderr, "dream: %s\n", why.c_str());
            return 1;
        }
    }

    if (dump) {
        dump_image(rt.image());
        return 0;
    }

    // Constructing a Jit registers it with the runtime, so `--no-jit` simply
    // never builds one and every call stays in the interpreter.
    std::unique_ptr<Jit> jit_owner;
    if (use_jit && Jit::available()) jit_owner = std::make_unique<Jit>(rt);
    if (jit_threshold && jit_owner) jit_owner->set_threshold(jit_threshold);
    if (jit_threads && jit_owner) jit_owner->set_threads(jit_threads);
    if (jit_policy_set && jit_owner) jit_owner->set_policy(jit_policy);

    if (!dump_jit_fn.empty()) {
        if (!Jit::available()) {
            std::fprintf(stderr, "dream: this build has no JIT\n");
            return 1;
        }
        if (!jit_owner) jit_owner = std::make_unique<Jit>(rt);
        if (dump_jit_fn.size() > 1 && dump_jit_fn[0] == '#') {
            const unsigned long i = std::strtoul(dump_jit_fn.c_str() + 1, nullptr, 10);
            if (i < rt.image().func_count()) {
                std::fputs(jit_owner->dump_ir(uint32_t(i)).c_str(), stdout);
                return 0;
            }
        }
        // The root module's global of that name first: it is the program's own
        // function. A bare name is shared with every module the program links
        // in -- `std` has its own `atom`, `rev` and `integer` -- so the first
        // function or global of that name in the image is whichever module
        // happened to be laid out earlier.
        const Image& img = rt.image();
        const std::string root = img.module_name().str();
        for (uint32_t m = 0; m < img.module_count(); ++m) {
            const ModuleRec& mr = img.module(m);
            if (!img.str(mr.name).equals(root.c_str())) continue;
            for (uint32_t k = 0; k < mr.globals_count; ++k) {
                const GlobalRec& gr = img.global(mr.globals_start + k);
                if (gr.kind == GLOBAL_FUNCTION && img.str(gr.name).equals(dump_jit_fn.c_str())) {
                    std::fputs(jit_owner->dump_ir(gr.target).c_str(), stdout);
                    return 0;
                }
            }
        }
        for (uint32_t i = 0; i < rt.image().func_count(); ++i) {
            if (rt.image().str(rt.image().func(i).name).equals(dump_jit_fn.c_str())) {
                std::fputs(jit_owner->dump_ir(i).c_str(), stdout);
                return 0;
            }
        }
        std::fprintf(stderr, "dream: no function named `%s`\n", dump_jit_fn.c_str());
        return 1;
    }

    if (precompile && jit_owner) {
        // Every core: nothing else is running yet. `--jit-threads` still
        // says otherwise when given.
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const Jit::Precompiled pc = jit_owner->precompile(jit_threads ? jit_threads : hw);
        if (stats) {
            std::fprintf(stderr, "; precompiled %u functions (%u refused) in %.0f ms on %u threads\n",
                         pc.compiled, pc.refused, pc.ms, jit_threads ? jit_threads : hw);
        }
    }

    uint32_t func = NO_NODE;
    if (!entry.empty()) {
        int g = rt.image().find_global(entry.c_str());
        if (g < 0) {
            std::fprintf(stderr, "dream: no global named `%s`\n", entry.c_str());
            return 1;
        }
        const GlobalRec& gr = rt.image().global(uint32_t(g));
        if (gr.kind != GLOBAL_FUNCTION) {
            std::fprintf(stderr, "dream: `%s` is not a function\n", entry.c_str());
            return 1;
        }
        func = gr.target;
    } else {
        func = rt.image().entry();
        if (func == NO_NODE) {
            std::fprintf(stderr,
                         "dream: this image has no `main!`; pass --entry to name one\n");
            return 1;
        }
    }

    if (workers == 0) workers = Scheduler::default_workers();

    Scheduler sched(rt, workers);
    rt.mark_run_start();
    auto root = sched.create_process();
    Value cl = root->heap().make_closure(func, 0);
    prime_apply(*root, cl, 0);

    bool clean = sched.run(root);

    // The root's failure first: a run ends when the root fails, with the
    // processes it started possibly still alive (`Scheduler::finish`), and
    // that is not a deadlock.
    int status = 0;
    if (root->failed) {
        std::string text = describe_failure(*root, root->exit_value);
        std::fprintf(stderr, "dream: uncaught error: %s\n", text.c_str());
        status = 1;
    } else if (!clean) {
        std::fprintf(stderr,
                     "dream: deadlock -- every process is waiting for a message that "
                     "cannot arrive\n");
        // And who is waiting for what. A deadlock with no names is a report
        // that a program is stuck, which the person running it already knew;
        // the useful part is which process is parked on which thing.
        sched.dump("deadlock");
        status = 1;
    }

    for (const std::string& f : sched.take_failures()) {
        if (f.rfind("process " + std::to_string(root->id()) + ":", 0) == 0) continue;
        std::fprintf(stderr, "dream: uncaught error in %s\n", f.c_str());
        status = 1;
    }

    rt.print_profile();
    rt.print_stats();
    if (stats) {
        // What the runtime cannot know: how this particular invocation was set
        // up.
        // How many functions the JIT took is `print_stats`'s, so that a program
        // ending in `os.exit!` still reports it.
        std::fprintf(stderr, "; %u workers, jit %s\n", workers, jit_owner ? "on" : "off");
    }
    return status;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // The CRT's narrow argv follows the active ANSI code page, not UTF-8.
    int count = 0;
    wchar_t** wide = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!wide) return 1;
    std::vector<std::string> strings;
    for (int i = 0; i < count; ++i) strings.push_back(dream::windows::utf8(wide[i]));
    LocalFree(wide);
    std::vector<char*> pointers;
    for (auto& text : strings) pointers.push_back(text.data());
    pointers.push_back(nullptr);
    return dream_main(count, pointers.data());
#else
    return dream_main(argc, argv);
#endif
}
