// `std.io` and `std.os` as a browser has them.
//
// An ordinary program reaches these two modules almost at once: `console`
// writes through `std.io`, `time` and `log` read the clock through `std.os`,
// `test` does both. A browser VM that left them out could run nothing but
// code written for it, which is the opposite of the point. So each member is
// here, under the name and arity the native VM gives it, and answers in one
// of two ways:
//
//   - where a browser has the thing, the same answer the native VM gives: the
//     three standard streams (stdout and stderr reach the page's console, line
//     by line, through Emscripten's `print`/`printErr`; stdin is at its end),
//     both clocks, the program's arguments (none) and environment (empty);
//   - where it does not -- files, directories, subprocesses -- the error
//     `:unsupported`, naming the member, raised when it is *called*. Importing
//     `std.file` costs nothing until a file is opened, which is what lets a
//     frontend share modules with its backend.
//
// `std.crypto` needs no version of its own: it is self-contained C++, and
// `getrandom` is the browser's `crypto.getRandomValues` under Emscripten.
// `std.net`, `std.tls` and `std.ffi` are not registered at all, and the VM
// already says "module ... is not provided by this runtime" at the first call
// into one. The page's own services -- the DOM, fetch, storage -- are not
// ambient either: they go through the graphics bridge, as data.
#include <chrono>
#include <cstdio>
#include <string>
#include <tuple>
#include "builtins.hpp"
#include "crypto.hpp"
#include "digest.hpp"
#include "interp.hpp"
#include "services.hpp"

namespace dream {
namespace {
bool exit_requested = false;
int exit_code = 0;

NativeResult fail(Process& p, const char* kind, const std::string& message) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom(kind), message));
}
Value text(Process& p, const std::string& s) { return p.heap().make_string(s.data(), uint32_t(s.size())); }
Value atom(Process& p, const char* name) { return make_atom(p.runtime().intern_atom(name)); }

/// The member's own name is what a person reading the error needs: the call
/// that failed is usually several modules away from the program's own code.
NativeResult unsupported(Process& p, Value callee, Value*, uint32_t) {
    const auto* member = reinterpret_cast<const char*>(static_cast<NativeObj*>(as_obj(callee))->user);
    return fail(p, "unsupported", std::string(member) + " is not available in the browser");
}

/// Handles 0, 1 and 2 are the standard streams, as the native VM numbers its
/// first three; nothing else is ever opened here.
bool standard(Value v, int64_t* fd) {
    v = resolve(v);
    if (!is_fixnum(v) || fixnum_value(v) < 0 || fixnum_value(v) > 2) return false;
    *fd = fixnum_value(v);
    return true;
}

NativeResult io_stream(Process&, Value callee, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(int64_t(static_cast<NativeObj*>(as_obj(callee))->user)));
}
NativeResult io_write(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd) || fd == 0) return fail(p, "io_closed", "this handle is not open for writing");
    Bytes s;
    if (!string_bytes(args[1], &s)) return fail(p, "type_error", "write! needs a string");
    // Emscripten hands a stream to `print` a line at a time, so a write that
    // ends without a newline waits for the next one, as a terminal shows it.
    std::fwrite(s.data, 1, size_t(s.len), fd == 1 ? stdout : stderr);
    return NativeResult::ok(make_fixnum(int64_t(s.len)));
}
NativeResult io_read(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd) || fd != 0) return fail(p, "io_closed", "this handle is not open for reading");
    return NativeResult::ok(text(p, ""));
}
NativeResult io_flush(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd)) return fail(p, "io_closed", "this handle is closed");
    std::fflush(fd == 2 ? stderr : stdout);
    return NativeResult::ok(UNIT);
}
NativeResult io_close(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd)) return fail(p, "io_closed", "this handle is already closed");
    return NativeResult::ok(UNIT);
}
NativeResult io_is_open(Process&, Value, Value* args, uint32_t) {
    int64_t fd;
    return NativeResult::ok(make_bool(standard(args[0], &fd)));
}
NativeResult io_false(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd)) return fail(p, "io_closed", "this handle is closed");
    return NativeResult::ok(make_bool(false));
}
NativeResult io_kind(Process& p, Value, Value* args, uint32_t) {
    int64_t fd;
    if (!standard(args[0], &fd)) return fail(p, "io_closed", "this handle is closed");
    return NativeResult::ok(atom(p, "stream"));
}
NativeResult io_digest(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return fail(p, "type_error", "digest needs a string");
    Sha256 sha;
    sha.update(b.data, size_t(b.len));
    return NativeResult::ok(text(p, sha.hex()));
}
NativeResult io_async(Process&, Value, Value*, uint32_t) { return NativeResult::ok(make_bool(false)); }

NativeResult os_args(Process&, Value, Value*, uint32_t) { return NativeResult::ok(NIL); }
NativeResult os_env(Process& p, Value, Value* args, uint32_t) {
    if (!is_stringish(args[0])) return fail(p, "type_error", "env! needs a name");
    return NativeResult::ok(UNIT);
}
NativeResult os_monotonic(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return NativeResult::ok(make_integer(p, std::chrono::duration_cast<std::chrono::milliseconds>(now).count()));
}
NativeResult os_now(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return NativeResult::ok(make_integer(p, std::chrono::duration_cast<std::chrono::milliseconds>(now).count()));
}
NativeResult os_pid(Process& p, Value, Value*, uint32_t) { return NativeResult::ok(make_integer(p, 1)); }
NativeResult os_cwd(Process& p, Value, Value*, uint32_t) { return NativeResult::ok(text(p, "/")); }
NativeResult os_platform(Process& p, Value, Value*, uint32_t) { return NativeResult::ok(atom(p, "browser")); }
NativeResult os_arch(Process& p, Value, Value*, uint32_t) { return NativeResult::ok(atom(p, "wasm32")); }
NativeResult os_install_dirs(Process&, Value, Value*, uint32_t) { return NativeResult::ok(NIL); }
/// The page is not the VM's to close, so ending the program is what exit
/// means here: the process parks for good and the host sees the run finished
/// at its next slice, with the code (`browser_exit_code`).
NativeResult os_exit(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return fail(p, "type_error", "exit! needs an exit code");
    std::fflush(nullptr);
    exit_requested = true;
    exit_code = int(fixnum_value(v));
    p.park_requested = true;
    p.wait_reason.store(WaitReason::Message);
    return NativeResult::block();
}

NativeDef missing(const char* name, uint32_t arity, const char* qualified) {
    return NativeDef{name, arity, 0, unsupported, uint64_t(reinterpret_cast<uintptr_t>(qualified))};
}
}

ModuleDef make_browser_io_module() {
    ModuleDef def{"std.io", {
        {"stdin!", 1, 0b1, io_stream, 0},
        {"stdout!", 1, 0b1, io_stream, 1},
        {"stderr!", 1, 0b1, io_stream, 2},
        {"read!", 2, 0b11, io_read},
        {"write!", 2, 0b11, io_write},
        {"flush!", 1, 0b1, io_flush},
        {"close!", 1, 0b1, io_close},
        {"is_open!", 1, 0b1, io_is_open},
        {"is_terminal!", 1, 0b1, io_false},
        {"enable_ansi!", 1, 0b1, io_false},
        {"kind!", 1, 0b1, io_kind},
        {"digest", 1, 0b1, io_digest},
        {"async", 1, 0b1, io_async},
    }};
    for (auto [name, arity, qualified] : {
             std::tuple{"open!", 2u, "io.open!"}, {"temp_file!", 2u, "io.temp_file!"},
             {"temp_dir!", 2u, "io.temp_dir!"}, {"seek!", 2u, "io.seek!"}, {"size!", 1u, "io.size!"},
             {"exists!", 1u, "io.exists!"}, {"is_dir!", 1u, "io.is_dir!"}, {"remove!", 1u, "io.remove!"},
             {"rename!", 2u, "io.rename!"}, {"mkdir!", 1u, "io.mkdir!"}, {"stat!", 1u, "io.stat!"},
             {"digest!", 1u, "io.digest!"}, {"mkdir_all!", 1u, "io.mkdir_all!"},
             {"remove_all!", 1u, "io.remove_all!"}, {"link!", 2u, "io.link!"}, {"copy!", 2u, "io.copy!"},
             {"chmod!", 2u, "io.chmod!"}, {"mode!", 1u, "io.mode!"}, {"lstat!", 1u, "io.lstat!"},
             {"is_link!", 1u, "io.is_link!"}, {"symlink!", 2u, "io.symlink!"},
             {"read_link!", 1u, "io.read_link!"}, {"walk!", 1u, "io.walk!"}})
        def.members.push_back(missing(name, arity, qualified));
    return def;
}

ModuleDef make_browser_os_module() {
    ModuleDef def{"std.os", {
        {"args!", 1, 0b1, os_args},
        {"env!", 1, 0b1, os_env},
        {"cwd!", 1, 0b1, os_cwd},
        {"monotonic!", 1, 0b1, os_monotonic},
        {"now!", 1, 0b1, os_now},
        {"pid!", 1, 0b1, os_pid},
        {"platform", 1, 0b1, os_platform},
        {"arch", 1, 0b1, os_arch},
        {"install_dirs!", 1, 0b1, os_install_dirs},
        {"exit!", 1, 0b1, os_exit},
    }};
    for (auto [name, arity, qualified] : {
             std::tuple{"set_env!", 2u, "os.set_env!"}, {"chdir!", 1u, "os.chdir!"},
             {"list_dir!", 1u, "os.list_dir!"}, {"exec!", 2u, "os.exec!"}, {"exec_for!", 3u, "os.exec_for!"},
             {"exec_in!", 4u, "os.exec_in!"}, {"exec_with!", 5u, "os.exec_with!"},
             {"exec_joined!", 5u, "os.exec_joined!"}, {"replace!", 2u, "os.replace!"}})
        def.members.push_back(missing(name, arity, qualified));
    return def;
}

bool browser_exit_requested(int* code) {
    if (code) *code = exit_code;
    return exit_requested;
}
void browser_reset_exit() { exit_requested = false; exit_code = 0; }
}
