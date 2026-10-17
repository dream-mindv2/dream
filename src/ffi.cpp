// std.ffi -- calling C from Dream.
//
// The design adds no syntax. A foreign function is made by calling a function
// with its C signature as data, and what comes back is an ordinary Dream
// function:
//
//     let libm = "libm.so.6";
//     let sqrt  = ffi.pure_function libm "sqrt" [:f64] :f64;
//     let puts! = ffi.function "" "puts" [:cstr] :int;
//
// Whether calling it is an effect is the name it is bound to, as it is for
// everything else in the language: `pure_function` makes a value the purity
// check will let a pure function call, and `function` one whose own name ends
// in `!`, which is what the runtime's guards read (`impure_callee`). Nothing
// here can check that a C function is pure. A binding that says so and is not
// is a lie the program told, exactly as a wrong signature is.
//
// **What this can make safe, and what it cannot.** A signature that does not
// match the C function corrupts the call, as it would in C, and nothing below
// can see it. Everything *else* is checked: that the arguments are the kinds
// the signature says, that a handle is live, owned by this process and of the
// tag the parameter names, that a buffer read or write is inside the buffer.
// Pointers are the reason. A raw address carried as an integer can be forged,
// kept after it is freed, freed twice and passed where another kind of pointer
// was wanted, and every one of those is undefined behaviour that surfaces a
// long way from the mistake. So a pointer a C function hands back is *owned*:
//
//     [:own, "sqlite3", "sqlite3_close"]     -- a return (or out) type
//
// makes the result a handle, `[:foreign, "sqlite3", 17]`, and the pointer
// itself stays here, in a table the process owns (`ForeignTable`). A handle is
// only a name for an entry. Using one looks the entry up, so a released handle
// is an error rather than a use-after-free, a handle from another process names
// nothing here, and `[:handle, "sqlite3"]` refuses a `"sqlite3_stmt"`.
//
// **When the destructor runs.** When the program says `release!`; or, for
// everything it did not release, when the process that owns it ends -- by
// returning, by raising, by `os.exit!` -- newest first, which is always
// children before parents. A resource made by a call that took handles
// records them as its parents, and releasing a parent releases its children
// first: closing a database finalizes its statements rather than leaving them
// pointing at freed memory. Dream has no finalizers and does not need them;
// the process is the scope, which is the same idea a process already is for a
// heap. That is also why a C library wrapped as a *server process*
// (`std.foreign`) is the natural shape: whatever it made is let go when it
// stops, crash or not.
//
// **Callbacks** run Dream while C is on the stack, so they have rules of their
// own; see `Callback` below.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bigint.hpp"
#include "builtins.hpp"
#include "interp.hpp"
#include "os.hpp"
#include "process.hpp"
#include "runtime.hpp"

#include "windows.hpp"
#ifndef _WIN32
#include <dlfcn.h>
#ifdef __linux__
#include <link.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#endif
#endif
#include <ffi.h>

namespace dream {

/// What a process owns through the FFI. See the head of this file.
///
/// Keyed by id, and ids come from one counter for the whole runtime, so the
/// map's order is the order of creation -- which is the order to destroy in,
/// reversed -- and an id can never name another process's resource by
/// coincidence.
struct ForeignTable {
    struct Resource {
        std::string tag;
        void* ptr = nullptr;
        /// Null for a borrowed pointer: something else frees it.
        void (*dtor)(void*) = nullptr;
        /// Bytes, for a buffer; 0 when the size is not known, which is what
        /// refuses `peek!` and friends on a pointer nobody measured.
        uint64_t size = 0;
        /// The handles the call that made this one was given. Releasing one of
        /// them releases this first.
        std::vector<uint64_t> parents;
    };
    std::map<uint64_t, Resource> live;
};

namespace {

std::atomic<uint64_t> g_next_resource{1};

/// Run one resource's destructor. The pointer is the only argument every C
/// destructor takes, and whatever it answers -- `sqlite3_close` answers a
/// status -- is ignored: there is nobody left to tell.
void destroy(ForeignTable::Resource& r) {
    if (r.dtor && r.ptr) r.dtor(r.ptr);
    r.ptr = nullptr;
}

}  // namespace

void release_foreign(Process& p) {
    ForeignTable* t = p.foreign;
    if (!t) return;
    p.foreign = nullptr;
    for (auto it = t->live.rbegin(); it != t->live.rend(); ++it) destroy(it->second);
    delete t;
}

namespace {

enum Kind : uint8_t {
    K_VOID, K_BOOL,
    K_I8, K_I16, K_I32, K_I64,
    K_U8, K_U16, K_U32, K_U64,
    K_F32, K_F64,
    K_PTR, K_CSTR, K_BYTES,
    // Compound: written as a list, and carrying more than a kind.
    K_HANDLE,     // [:handle, tag]          an argument: a live handle of that tag
    K_OWN,        // [:own, tag, dtor]       a result: a pointer this process now owns
    K_BORROW,     // [:borrow, tag]          a result: a pointer something else frees
    K_TAKE_CSTR,  // [:cstr, dtor]           a result: a string to copy, then free
    K_OUT,        // [:out, t]               an argument C writes and Dream reads back
    K_CALLBACK,   // [:callback, [a], r]     an argument: a Dream function C may call
    K_INVALID,
};

struct KindName {
    const char* name;
    Kind kind;
    size_t size;
    size_t align;
};

constexpr Kind K_LONG = sizeof(long) == 8 ? K_I64 : K_I32;
constexpr Kind K_ULONG = sizeof(long) == 8 ? K_U64 : K_U32;
constexpr Kind K_SIZE = sizeof(size_t) == 8 ? K_U64 : K_U32;
constexpr Kind K_SSIZE = sizeof(size_t) == 8 ? K_I64 : K_I32;

/// The type vocabulary. The first block is the canonical names; the second is
/// C's own spellings, which differ between platforms (`long` is 32 bits on
/// Windows) and are resolved to a canonical kind here, once, so that a binding
/// written against `<stdio.h>` is right on every platform it compiles for.
///
/// There is no `:float` alias for C's `float`, on purpose: in Dream `:float` is
/// a double, and a signature is exactly where the two readings would be
/// confused. `:f32` says which is meant.
constexpr KindName KINDS[] = {
    {"void", K_VOID, 0, 1},
    {"bool", K_BOOL, sizeof(bool), alignof(bool)},
    {"i8", K_I8, 1, 1},
    {"i16", K_I16, 2, alignof(int16_t)},
    {"i32", K_I32, 4, alignof(int32_t)},
    {"i64", K_I64, 8, alignof(int64_t)},
    {"u8", K_U8, 1, 1},
    {"u16", K_U16, 2, alignof(uint16_t)},
    {"u32", K_U32, 4, alignof(uint32_t)},
    {"u64", K_U64, 8, alignof(uint64_t)},
    {"f32", K_F32, 4, alignof(float)},
    {"f64", K_F64, 8, alignof(double)},
    {"ptr", K_PTR, sizeof(void*), alignof(void*)},
    {"cstr", K_CSTR, sizeof(void*), alignof(void*)},
    {"bytes", K_BYTES, sizeof(void*), alignof(void*)},

    {"char", K_I8, 1, 1},
    {"uchar", K_U8, 1, 1},
    {"short", K_I16, sizeof(short), alignof(short)},
    {"ushort", K_U16, sizeof(short), alignof(short)},
    {"int", K_I32, sizeof(int), alignof(int)},
    {"uint", K_U32, sizeof(int), alignof(int)},
    {"long", K_LONG, sizeof(long), alignof(long)},
    {"ulong", K_ULONG, sizeof(long), alignof(long)},
    {"longlong", K_I64, sizeof(long long), alignof(long long)},
    {"ulonglong", K_U64, sizeof(long long), alignof(long long)},
    {"size", K_SIZE, sizeof(size_t), alignof(size_t)},
    {"ssize", K_SSIZE, sizeof(size_t), alignof(size_t)},
    {"intptr", sizeof(void*) == 8 ? K_I64 : K_I32, sizeof(void*), alignof(void*)},
    {"uintptr", sizeof(void*) == 8 ? K_U64 : K_U32, sizeof(void*), alignof(void*)},
    {"double", K_F64, 8, alignof(double)},
};

const KindName* scalar_named(const std::string& name) {
    for (const auto& k : KINDS) {
        if (name == k.name) return &k;
    }
    return nullptr;
}

const char* kind_name(Kind k) {
    for (const auto& e : KINDS) {
        if (e.kind == k) return e.name;
    }
    return "?";
}

bool is_scalar(Kind k) { return k >= K_BOOL && k <= K_F64; }

constexpr uint32_t MAX_ARGS = 16;

NativeResult fail(Process& p, const std::string& msg) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom("ffi_error"), msg));
}

/// A failure while reading what a program passed: either a sentence to raise
/// as an `:ffi_error`, or an error a forced argument raised itself, which is
/// handed on as it is rather than reworded.
struct Err {
    std::string text;
    bool raised = false;
    Value value = UNIT;
};

NativeResult raise_err(Process& p, const Err& e, const std::string& prefix = "") {
    if (e.raised) return NativeResult::raise(e.value);
    return fail(p, prefix + e.text);
}

bool force(Process& p, Value v, Value* out, Err* e) {
    if (force_whnf(p, v, out)) return true;
    e->raised = true;
    e->value = p.result;
    return false;
}

bool say(Err* e, const std::string& text) {
    e->text = text;
    return false;
}

/// A Dream list, spine and elements forced to weak head normal form.
bool list_items(Process& p, Value v, std::vector<Value>* out, Err* e, const char* what) {
    Value cur;
    if (!force(p, v, &cur, e)) return false;
    for (;;) {
        if (is_nil(cur)) return true;
        if (!is_obj(cur, ObjType::Cons)) return say(e, std::string(what) + " must be a list");
        auto* c = static_cast<ConsObj*>(as_obj(cur));
        Value head, tail = c->tail;
        if (!force(p, c->head, &head, e)) return false;
        out->push_back(head);
        if (!force(p, tail, &cur, e)) return false;
    }
}

bool str_value(Value v, std::string* out) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Str)) return false;
    auto* s = static_cast<StrObj*>(as_obj(v));
    out->assign(s->data(), s->len);
    return true;
}

std::string atom_text(Process& p, Value v) {
    return p.runtime().atom_name(uint32_t(imm_payload(v)));
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

struct Sig;

struct Type {
    Kind kind = K_INVALID;
    std::string tag;
    void (*dtor)(void*) = nullptr;
    std::shared_ptr<Type> inner;  // K_OUT
    std::shared_ptr<Sig> sig;     // K_CALLBACK
};

struct Sig {
    std::vector<Type> args;
    Type ret;
    std::vector<ffi_type*> ffi_args;
    ffi_cif cif{};
    /// How many arguments the Dream caller passes: every C argument that is
    /// not an `out`.
    uint32_t visible = 0;
    bool has_out = false;
};

ffi_type* ffi_type_for(Kind k) {
    switch (k) {
        case K_VOID: return &ffi_type_void;
        case K_BOOL: return &ffi_type_uint8;
        case K_I8: return &ffi_type_sint8;
        case K_I16: return &ffi_type_sint16;
        case K_I32: return &ffi_type_sint32;
        case K_I64: return &ffi_type_sint64;
        case K_U8: return &ffi_type_uint8;
        case K_U16: return &ffi_type_uint16;
        case K_U32: return &ffi_type_uint32;
        case K_U64: return &ffi_type_uint64;
        case K_F32: return &ffi_type_float;
        case K_F64: return &ffi_type_double;
        default: return &ffi_type_pointer;
    }
}

// ---------------------------------------------------------------------------
// Libraries
// ---------------------------------------------------------------------------

struct Library {
    std::string label;  // for messages
    std::string where;  // the file it came from, as well as that is known
    void* handle = nullptr;
};

struct Binding {
    std::string name;
    void* fn = nullptr;
    Sig sig;
};

/// Libraries and bindings live for the life of the VM and are shared by every
/// process, so a function value carries a small id rather than a pointer: an
/// id the registry did not issue is rejected instead of jumped to. A binding is
/// made again each time a process forces the global it is bound to, so the
/// same signature against the same symbol is found by key rather than
/// registered twice.
struct Registry {
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<Library>> libs;
    /// `open!`'s handles, which are numbers a program holds.
    std::vector<Library*> numbered;
    std::vector<std::unique_ptr<Binding>> bindings;
    std::map<std::string, uint64_t> binding_ids;

    static Registry& get() {
        static Registry r;
        return r;
    }
};

std::string last_dl_error() {
#ifdef _WIN32
    return windows::error();
#else
    const char* e = ::dlerror();
    return e ? e : "unknown error";
#endif
}

void* dl_open(const std::string& path) {
#ifdef _WIN32
    return path.empty() ? static_cast<void*>(GetModuleHandleW(nullptr))
                        : static_cast<void*>(LoadLibraryW(windows::wide(path).c_str()));
#else
    return ::dlopen(path.empty() ? nullptr : path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* dl_sym(void* lib, const std::string& name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name.c_str()));
#else
    ::dlerror();
    void* f = ::dlsym(lib, name.c_str());
    return ::dlerror() ? nullptr : f;
#endif
}

/// Load a shared library from bytes -- the payload of the image running now.
///
/// A dynamic loader wants a file. On Linux that file need not be on any disk:
/// `memfd_create` makes an anonymous one and `/proc/self/fd` names it, so an
/// embedded library leaves nothing behind. Elsewhere it is written to the
/// temporary directory, loaded, and -- where the platform allows unlinking a
/// loaded library, which is everywhere but Windows -- removed again at once.
void* dl_open_bytes(const char* data, uint64_t len, const std::string& name, std::string* err) {
#if defined(__linux__)
    int fd = ::memfd_create(("dream-ffi-" + name).c_str(), MFD_CLOEXEC);
    if (fd >= 0) {
        uint64_t done = 0;
        while (done < len) {
            ssize_t n = ::write(fd, data + done, size_t(std::min<uint64_t>(len - done, 1 << 20)));
            if (n <= 0) break;
            done += uint64_t(n);
        }
        if (done == len) {
            std::string path = "/proc/self/fd/" + std::to_string(fd);
            void* h = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) *err = last_dl_error();
            ::close(fd);
            return h;
        }
        ::close(fd);
    }
#endif
    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) {
        *err = "no temporary directory to load it from";
        return nullptr;
    }
    static std::atomic<uint64_t> serial{0};
    std::string file = "dream-ffi-" + std::to_string(uint64_t(std::hash<std::thread::id>{}(
                           std::this_thread::get_id()))) +
                       "-" + std::to_string(serial.fetch_add(1)) + "-" + name;
#ifdef _WIN32
    file += ".dll";
#endif
    auto path = dir / file;
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) {
        *err = "cannot write " + path.string();
        return nullptr;
    }
    bool ok = std::fwrite(data, 1, size_t(len), f) == size_t(len);
    ok = std::fclose(f) == 0 && ok;
    if (!ok) {
        *err = "cannot write " + path.string();
        std::filesystem::remove(path, ec);
        return nullptr;
    }
    void* h = dl_open(path.string());
    if (!h) *err = last_dl_error();
#ifndef _WIN32
    std::filesystem::remove(path, ec);
#endif
    return h;
}

const char* const DESCRIPTORS =
    "a library is a path, `()` for this program, `[:payload, name]`, `[:search, name, files, dirs]` "
    "or `[:any, libraries]`";

// ---------------------------------------------------------------------------
// Finding a library
// ---------------------------------------------------------------------------
//
// A path handed to the platform's loader is how a library used to be named,
// and it is the wrong question for a library installed on the system: the
// file is `libsqlite3.so.0` on one machine, `libsqlite3.so` on another that
// only has the development package, `libsqlite3.0.dylib` under Homebrew's
// prefix -- which macOS's loader never searches -- and on NixOS in no
// directory any loader knows about. A program that guessed one name failed on
// the others with a message that said nothing about where it had looked.
//
// So a library may be a *search*, `[:search, name, files, dirs]`: the names
// its file goes by on this platform, best first, and the directories worth
// trying when the loader does not know it. Both are worked out in Dream
// (`foreign.system`), where the knowledge about platforms can be read and
// tested; what is done here is the looking, because looking is an effect and
// a library descriptor is a value. The order:
//
//   1. `$DREAM_LIB_<NAME>`, the file itself. A person who says exactly which
//      file is obeyed exactly: if it cannot be opened that is the error, and
//      nothing else is tried behind their back.
//   2. every directory of `$DREAM_LIBRARY_PATH`, then each installation
//      directory's `lib` -- what someone set up on purpose.
//   3. the platform's loader, by each name: rpath, `ld.so.cache`,
//      `LD_LIBRARY_PATH` and the rest of what the system already knows.
//   4. the directories the search names, in order.
//
// A name may end in a version wildcard, `libsqlite3.so.*`, which in a
// directory means the highest version there. The loader cannot be asked that,
// so it is only tried with the names that are spelled out.
//
// When nothing is found the error says every place that was tried and what
// to set, because "cannot open libsqlite3.so.0" was the whole of what a person
// used to get, and it is no help at all to someone whose copy is in
// /opt/homebrew/lib.

/// `DREAM_LIB_` and the name, upper-cased, anything not a letter or a digit
/// made `_`: `DREAM_LIB_SQLITE3`, `DREAM_LIB_GTK_4`.
std::string override_variable(const std::string& name) {
    std::string out = "DREAM_LIB_";
    for (char c : name) {
        out += std::isalnum(static_cast<unsigned char>(c))
                   ? char(std::toupper(static_cast<unsigned char>(c)))
                   : '_';
    }
    return out;
}

/// Version strings compared as dotted numbers, so `1.10` is newer than `1.9`.
bool newer_version(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        auto part = [](const std::string& s, size_t* at) {
            uint64_t n = 0;
            bool any = false;
            while (*at < s.size() && std::isdigit(static_cast<unsigned char>(s[*at]))) {
                n = n * 10 + uint64_t(s[*at] - '0');
                ++*at;
                any = true;
            }
            while (*at < s.size() && !std::isdigit(static_cast<unsigned char>(s[*at]))) ++*at;
            return any ? n : 0;
        };
        uint64_t x = part(a, &i), y = part(b, &j);
        if (x != y) return x > y;
    }
    return false;
}

/// The files in `dir` that `pattern` -- a name with one `*` in it -- matches,
/// newest version first.
std::vector<std::string> matching(const std::string& dir, const std::string& pattern) {
    std::vector<std::pair<std::string, std::string>> found;  // version, path
    size_t star = pattern.find('*');
    std::string before = pattern.substr(0, star), after = pattern.substr(star + 1);
    std::error_code ec;
    std::filesystem::directory_iterator it(std::filesystem::path(std::u8string(dir.begin(), dir.end())), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        auto u8 = it->path().filename().u8string();
        std::string file(u8.begin(), u8.end());
        if (file.size() < before.size() + after.size()) continue;
        if (file.compare(0, before.size(), before) != 0) continue;
        if (file.compare(file.size() - after.size(), after.size(), after) != 0) continue;
        std::string version = file.substr(before.size(), file.size() - before.size() - after.size());
        if (version.empty() || !std::isdigit(static_cast<unsigned char>(version[0]))) continue;
        found.emplace_back(version, dir + "/" + file);
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return newer_version(a.first, b.first); });
    std::vector<std::string> out;
    for (auto& f : found) out.push_back(f.second);
    return out;
}

/// Where the loader actually found a library it was given by name. With glibc
/// the link map says; elsewhere the name is the honest answer.
std::string loaded_from(void* handle, const std::string& name) {
#if defined(__GLIBC__)
    struct link_map* map = nullptr;
    if (::dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map && map->l_name && *map->l_name) {
        return map->l_name;
    }
#endif
    (void)handle;
    return name;
}

struct Search {
    std::string name;
    std::vector<std::string> files;
    std::vector<std::string> dirs;
};

/// Run a search: the handle and the file it came from, or the reason nothing
/// was found, which names every place that was tried.
void* run_search(const Search& s, std::string* where, std::string* why) {
    const std::string variable = override_variable(s.name);
    if (const char* exact = std::getenv(variable.c_str()); exact && *exact) {
        std::string path = expand_home(exact);
        void* h = dl_open(path);
        if (h) {
            *where = path;
            return h;
        }
        *why = "$" + variable + " names `" + path + "`, which cannot be opened: " + last_dl_error();
        return nullptr;
    }

    std::vector<std::string> failures;  // found, but would not open
    std::vector<std::string> looked;    // directories with nothing in them
    auto in_dir = [&](const std::string& dir) -> void* {
        for (const std::string& file : s.files) {
            std::vector<std::string> paths;
            if (file.find('*') != std::string::npos) {
                paths = matching(dir, file);
            } else if (is_file(dir + "/" + file)) {
                paths.push_back(dir + "/" + file);
            }
            for (const std::string& path : paths) {
                if (void* h = dl_open(path)) {
                    *where = path;
                    return h;
                }
                failures.push_back(path + ": " + last_dl_error());
            }
        }
        looked.push_back(dir);
        return nullptr;
    };

    std::vector<std::string> chosen;
    if (const char* extra = std::getenv("DREAM_LIBRARY_PATH")) {
        for (const std::string& d : split_dir_list(extra)) chosen.push_back(d);
    }
    for (const std::string& d : install_dirs()) chosen.push_back(d + "/lib");
    for (const std::string& dir : chosen) {
        if (void* h = in_dir(dir)) return h;
    }

    std::string loader_said;
    for (const std::string& file : s.files) {
        if (file.find('*') != std::string::npos) continue;
        if (void* h = dl_open(file)) {
            *where = loaded_from(h, file);
            return h;
        }
        if (loader_said.empty()) loader_said = last_dl_error();
    }

    for (const std::string& d : s.dirs) {
        if (void* h = in_dir(expand_home(d))) return h;
    }

    std::string names;
    for (const std::string& f : s.files) names += (names.empty() ? "" : ", ") + f;
    std::string text = "cannot find the C library `" + s.name + "` (as " + names + ")";
    text += "\n  the system's loader: " + (loader_said.empty() ? std::string("not asked") : loader_said);
    if (!looked.empty()) {
        std::string dirs;
        for (const std::string& d : looked) dirs += (dirs.empty() ? "" : ", ") + d;
        text += "\n  nothing in: " + dirs;
    }
    for (const std::string& f : failures) text += "\n  found but could not open " + f;
    text += "\nSet $" + variable + " to the library's file, or add its directory to $DREAM_LIBRARY_PATH.";
    *why = text;
    return nullptr;
}

/// A list of strings, forced, or false with `what` said.
bool string_items(Process& p, Value v, std::vector<std::string>* out, Err* e, const char* what) {
    std::vector<Value> items;
    if (!list_items(p, v, &items, e, what)) return false;
    for (Value item : items) {
        std::string s;
        if (!str_value(item, &s)) return say(e, std::string(what) + " must be a list of strings");
        out->push_back(std::move(s));
    }
    return true;
}

/// Open what one descriptor names, or the cached library it already opened.
Library* open_library(Process& p, Value v, Err* e) {
    std::string key, label, path;
    const char* bytes = nullptr;
    uint64_t length = 0;
    Search search;
    bool searching = false;
    if (is_unit(v)) {
        key = "self:";
        label = "<this program>";
    } else if (is_obj(v, ObjType::Str)) {
        str_value(v, &path);
        key = path.empty() ? "self:" : "path:" + path;
        label = path.empty() ? "<this program>" : path;
    } else if (is_obj(v, ObjType::Cons)) {
        std::vector<Value> items;
        if (!list_items(p, v, &items, e, "a library")) return nullptr;
        std::string tag = !items.empty() && is_atom(items[0]) ? atom_text(p, items[0]) : "";
        if (tag == "search" && items.size() == 4) {
            if (!str_value(items[1], &search.name)) {
                say(e, "a library search needs the library's name");
                return nullptr;
            }
            if (!string_items(p, items[2], &search.files, e, "a search's file names")) return nullptr;
            if (!string_items(p, items[3], &search.dirs, e, "a search's directories")) return nullptr;
            searching = true;
            key = "search:" + search.name;
            for (const std::string& f : search.files) key += "\n" + f;
            key += "\n";
            for (const std::string& d : search.dirs) key += "\n" + d;
            label = "the C library `" + search.name + "`";
        } else if (tag == "payload" && items.size() == 2) {
            const Image& img = p.runtime().image();
            int64_t index = -1;
            std::string name;
            if (is_fixnum(items[1])) {
                index = fixnum_value(items[1]);
                name = std::to_string(index);
            } else if (str_value(items[1], &name)) {
                index = img.data_index(name);
                if (index < 0) {
                    say(e, "this image carries no payload called `" + name +
                               "`; build it with `--payload " + name + "=FILE`");
                    return nullptr;
                }
            } else {
                say(e, "a payload library is named by a string or an index");
                return nullptr;
            }
            if (index < 0 || uint64_t(index) >= img.data_count()) {
                say(e, "this image carries no payload " + std::to_string(index));
                return nullptr;
            }
            bytes = img.data_bytes(uint32_t(index));
            length = img.data_length(uint32_t(index));
            // The address of the bytes, not the index: two images in one VM (an
            // embedder, a compile-time session) have two payload 0s.
            key = "payload:" + std::to_string(reinterpret_cast<uintptr_t>(bytes));
            label = "payload `" + name + "`";
            path = name;
        } else {
            say(e, DESCRIPTORS);
            return nullptr;
        }
    } else {
        say(e, DESCRIPTORS);
        return nullptr;
    }

    auto& reg = Registry::get();
    std::lock_guard<std::mutex> g(reg.mutex);
    auto it = reg.libs.find(key);
    if (it != reg.libs.end()) return it->second.get();
    std::string why, where = path;
    void* h = nullptr;
    if (searching) {
        h = run_search(search, &where, &why);
    } else if (bytes) {
        h = dl_open_bytes(bytes, length, path, &why);
    } else {
        h = dl_open(path);
        if (h && !path.empty()) where = loaded_from(h, path);
        if (!h) why = "cannot open " + label + ": " + last_dl_error();
    }
    if (!h) {
        say(e, bytes ? "cannot open " + label + ": " + why : why);
        return nullptr;
    }
    auto lib = std::make_unique<Library>();
    lib->label = label;
    lib->where = path.empty() && !searching ? "<this program>" : bytes ? label : where;
    lib->handle = h;
    Library* out = lib.get();
    reg.libs.emplace(key, std::move(lib));
    return out;
}

/// The library a descriptor names, opened once per VM.
///
///   `()` or `""`               this program: the VM and everything linked
///                              into it, which is the C library on any
///                              normal system
///   `"libm.so.6"`              a shared library, found the way the
///                              platform's loader finds one
///   `[:payload, "name"]`       a library carried in this image's payload, by
///   `[:payload, 0]`            the name `--payload NAME=FILE` gave it or by
///                              index
///   `[:search, n, files, dirs]`  a library installed on the system; see
///                              "Finding a library" above
///   `[:any, [lib, ..]]`        the first of these that opens, so a program
///                              can carry its own copy and prefer the
///                              system's, or the other way round
Library* library_of(Process& p, Value desc, Err* e) {
    Value v;
    if (!force(p, desc, &v, e)) return nullptr;
    if (is_obj(v, ObjType::Cons)) {
        std::vector<Value> items;
        if (!list_items(p, v, &items, e, "a library")) return nullptr;
        if (items.size() == 2 && is_atom(items[0]) && atom_text(p, items[0]) == "any") {
            std::vector<Value> choices;
            if (!list_items(p, items[1], &choices, e, "`any`'s libraries")) return nullptr;
            std::string why;
            for (Value choice : choices) {
                Err one;
                if (Library* lib = library_of(p, choice, &one)) return lib;
                if (one.raised) {
                    *e = one;
                    return nullptr;
                }
                why += "\n" + one.text;
            }
            say(e, choices.empty() ? std::string("`any` was given no libraries to choose from")
                                   : "none of the libraries could be opened:" + why);
            return nullptr;
        }
    }
    return open_library(p, v, e);
}

/// A destructor, looked up in the library first and then in the process: a
/// library's `free` is usually libc's, which its handle reaches through its
/// dependencies on ELF and does not on Windows.
void (*destructor_of(Library* lib, const std::string& name, Err* e))(void*) {
    void* f = dl_sym(lib->handle, name);
    if (!f) {
        void* self = dl_open("");
        if (self) f = dl_sym(self, name);
    }
    if (!f && name == "free") f = reinterpret_cast<void*>(&std::free);
    if (!f) {
        say(e, "cannot find the destructor `" + name + "` in " + lib->label);
        return nullptr;
    }
    return reinterpret_cast<void (*)(void*)>(f);
}

// ---------------------------------------------------------------------------
// Reading a signature
// ---------------------------------------------------------------------------

/// Where a type is written, which decides what it may be.
enum class Pos { Arg, Ret, OutInner, CbArg, CbRet };

bool parse_type(Process& p, Value v, Pos pos, Library* lib, Type* t, std::string* key, Err* e);

bool prep(Sig* s, Err* e) {
    s->ffi_args.clear();
    for (const Type& a : s->args) s->ffi_args.push_back(ffi_type_for(a.kind));
    if (ffi_prep_cif(&s->cif, FFI_DEFAULT_ABI, unsigned(s->args.size()), ffi_type_for(s->ret.kind),
                     s->ffi_args.data()) != FFI_OK) {
        return say(e, "libffi cannot prepare a call with this signature");
    }
    return true;
}

bool parse_sig(Process& p, Value args, Value ret, bool callback, Library* lib, Sig* s,
               std::string* key, Err* e) {
    std::vector<Value> items;
    if (!list_items(p, args, &items, e, "the argument types")) return false;
    if (items.size() > MAX_ARGS) {
        return say(e, "more than " + std::to_string(MAX_ARGS) + " arguments");
    }
    *key += "(";
    for (Value item : items) {
        Type t;
        if (!parse_type(p, item, callback ? Pos::CbArg : Pos::Arg, lib, &t, key, e)) return false;
        *key += ",";
        if (t.kind == K_OUT) {
            s->has_out = true;
        } else {
            ++s->visible;
        }
        s->args.push_back(std::move(t));
    }
    *key += ")->";
    Value r;
    if (!force(p, ret, &r, e)) return false;
    return parse_type(p, r, callback ? Pos::CbRet : Pos::Ret, lib, &s->ret, key, e) && prep(s, e);
}

bool parse_type(Process& p, Value v, Pos pos, Library* lib, Type* t, std::string* key, Err* e) {
    Value w;
    if (!force(p, v, &w, e)) return false;
    const bool result_side = pos == Pos::Ret || pos == Pos::OutInner;
    if (is_atom(w)) {
        std::string name = atom_text(p, w);
        if (name == "buffer" && pos == Pos::Arg) {
            t->kind = K_HANDLE;
            t->tag = "buffer";
            *key += "handle:buffer";
            return true;
        }
        const KindName* k = scalar_named(name);
        if (!k) return say(e, "`:" + name + "` is not an FFI type");
        t->kind = k->kind;
        *key += kind_name(k->kind);
        switch (k->kind) {
            case K_VOID:
                if (pos == Pos::Ret || pos == Pos::CbRet) return true;
                return say(e, "`:void` is only a return type");
            case K_BYTES:
                if (pos == Pos::Arg) return true;
                return say(e, "`:bytes` is only an argument type: C cannot say how long a result is");
            case K_CSTR:
                // A callback answering a string would hand C a pointer into a
                // heap that may move the moment the callback returns.
                if (pos == Pos::CbRet) return say(e, "a callback cannot answer a `:cstr`");
                return true;
            default:
                return true;
        }
    }
    if (!is_obj(w, ObjType::Cons)) return say(e, "a type is an atom such as `:i32`, or a list such as `[:out, :i32]`");
    std::vector<Value> items;
    if (!list_items(p, w, &items, e, "a type")) return false;
    if (items.empty() || !is_atom(items[0])) return say(e, "a compound type starts with an atom");
    std::string head = atom_text(p, items[0]);
    auto want = [&](size_t n, const char* shape) {
        if (items.size() == n) return true;
        return say(e, std::string("expected ") + shape);
    };
    auto tag_at = [&](size_t i, std::string* out, const char* what) {
        if (str_value(items[i], out) && !out->empty()) return true;
        return say(e, std::string(what) + " must be a non-empty string");
    };

    if (head == "handle") {
        if (pos != Pos::Arg) return say(e, "`[:handle, tag]` is an argument type; a result is `:own` or `:borrow`");
        if (!want(2, "`[:handle, tag]`") || !tag_at(1, &t->tag, "a handle's tag")) return false;
        t->kind = K_HANDLE;
        *key += "handle:" + t->tag;
        return true;
    }
    if (head == "own") {
        if (!result_side) return say(e, "`[:own, tag, destructor]` is a result type; an argument is `[:handle, tag]`");
        std::string dtor;
        if (!want(3, "`[:own, tag, destructor]`") || !tag_at(1, &t->tag, "a handle's tag") ||
            !tag_at(2, &dtor, "a destructor's name")) {
            return false;
        }
        t->kind = K_OWN;
        t->dtor = destructor_of(lib, dtor, e);
        *key += "own:" + t->tag + ":" + dtor;
        return t->dtor != nullptr;
    }
    if (head == "borrow") {
        if (!result_side) return say(e, "`[:borrow, tag]` is a result type; an argument is `[:handle, tag]`");
        if (!want(2, "`[:borrow, tag]`") || !tag_at(1, &t->tag, "a handle's tag")) return false;
        t->kind = K_BORROW;
        *key += "borrow:" + t->tag;
        return true;
    }
    if (head == "cstr") {
        if (!result_side) return say(e, "`[:cstr, destructor]` is a result type; an argument is `:cstr`");
        std::string dtor;
        if (!want(2, "`[:cstr, destructor]`") || !tag_at(1, &dtor, "a destructor's name")) return false;
        t->kind = K_TAKE_CSTR;
        t->dtor = destructor_of(lib, dtor, e);
        *key += "cstr:" + dtor;
        return t->dtor != nullptr;
    }
    if (head == "out") {
        if (pos != Pos::Arg) return say(e, "`[:out, t]` is an argument type");
        if (!want(2, "`[:out, t]`")) return false;
        auto inner = std::make_shared<Type>();
        *key += "out:";
        if (!parse_type(p, items[1], Pos::OutInner, lib, inner.get(), key, e)) return false;
        if (inner->kind == K_BYTES || inner->kind == K_VOID || inner->kind == K_CALLBACK) {
            return say(e, "`[:out, t]` holds a number, a pointer, a string or a handle");
        }
        t->kind = K_OUT;
        t->inner = std::move(inner);
        return true;
    }
    if (head == "callback") {
        if (pos != Pos::Arg) return say(e, "`[:callback, args, result]` is an argument type");
        if (!want(3, "`[:callback, [argument types], result type]`")) return false;
        auto sig = std::make_shared<Sig>();
        *key += "callback";
        if (!parse_sig(p, items[1], items[2], true, lib, sig.get(), key, e)) return false;
        for (const Type& a : sig->args) {
            if (!is_scalar(a.kind) && a.kind != K_PTR && a.kind != K_CSTR) {
                return say(e, "a callback's arguments are numbers, `:ptr` or `:cstr`");
            }
        }
        if (!is_scalar(sig->ret.kind) && sig->ret.kind != K_VOID && sig->ret.kind != K_PTR) {
            return say(e, "a callback answers a number, `:ptr` or `:void`");
        }
        t->kind = K_CALLBACK;
        t->sig = std::move(sig);
        return true;
    }
    return say(e, "`:" + head + "` does not start a compound type; the compound types are "
                             "`:handle`, `:own`, `:borrow`, `:cstr`, `:out` and `:callback`");
}

// ---------------------------------------------------------------------------
// Values across the boundary
// ---------------------------------------------------------------------------

/// Storage for one scalar, big enough for any of them and for what libffi
/// writes back for a return, which is at least a full `ffi_arg`.
union ArgSlot {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
    void* ptr;
    ffi_arg widened;
    ffi_sarg swidened;
};

ForeignTable& table_of(Process& p) {
    if (!p.foreign) p.foreign = new ForeignTable();
    return *p.foreign;
}

Value make_handle(Process& p, const std::string& tag, uint64_t id) {
    Value id_v = make_fixnum(int64_t(id));
    Value tag_v = p.heap().make_string(tag.data(), uint32_t(tag.size()));
    Value tail = p.heap().make_cons(id_v, NIL);
    tail = p.heap().make_cons(tag_v, tail);
    return p.heap().make_cons(make_atom(p.runtime().intern_atom("foreign")), tail);
}

/// Take ownership of `ptr`, and answer the handle that names it.
Value adopt(Process& p, const std::string& tag, void* ptr, void (*dtor)(void*), uint64_t size,
            const std::vector<uint64_t>& parents) {
    uint64_t id = g_next_resource.fetch_add(1);
    ForeignTable::Resource r;
    r.tag = tag;
    r.ptr = ptr;
    r.dtor = dtor;
    r.size = size;
    r.parents = parents;
    table_of(p).live.emplace(id, std::move(r));
    return make_handle(p, tag, id);
}

/// Read `[:foreign, tag, id]` and find the resource it names.
ForeignTable::Resource* resource_of(Process& p, Value v, uint64_t* id_out, Err* e) {
    std::vector<Value> items;
    if (!list_items(p, v, &items, e, "a handle")) return nullptr;
    std::string tag;
    if (items.size() != 3 || !is_atom(items[0]) || atom_text(p, items[0]) != "foreign" ||
        !str_value(items[1], &tag) || !is_fixnum(items[2])) {
        say(e, "expected a handle, `[:foreign, tag, id]`");
        return nullptr;
    }
    uint64_t id = uint64_t(fixnum_value(items[2]));
    ForeignTable::Resource* r = nullptr;
    if (p.foreign) {
        auto it = p.foreign->live.find(id);
        if (it != p.foreign->live.end()) r = &it->second;
    }
    if (!r || r->tag != tag) {
        say(e, "the `" + tag + "` handle " + std::to_string(id) +
                   " has been released, or belongs to another process");
        return nullptr;
    }
    if (id_out) *id_out = id;
    return r;
}

/// Release one resource, and everything made from it first.
void release_one(ForeignTable& t, uint64_t id) {
    // Children are newer than their parents, so walking down from the newest
    // finds a grandchild before the child it was made from.
    std::vector<uint64_t> children;
    for (auto it = t.live.rbegin(); it != t.live.rend(); ++it) {
        for (uint64_t parent : it->second.parents) {
            if (parent == id) {
                children.push_back(it->first);
                break;
            }
        }
    }
    for (uint64_t c : children) {
        if (t.live.count(c)) release_one(t, c);
    }
    auto it = t.live.find(id);
    if (it == t.live.end()) return;
    destroy(it->second);
    t.live.erase(it);
}

bool need_int(Value v, int64_t* out, Err* e) {
    if (is_fixnum(v)) {
        *out = fixnum_value(v);
        return true;
    }
    // A bignum is in range when it is a 64-bit integer of either signedness:
    // an unsigned one past 2^63 crosses as its bit pattern, which is what a
    // `uint64_t` parameter reads back.
    if (bigint::is_big(v)) {
        uint64_t u;
        if (bigint::to_int64(v, out)) return true;
        if (bigint::to_uint64(v, &u)) {
            *out = int64_t(u);
            return true;
        }
        return say(e, "an integer wider than 64 bits");
    }
    if (is_obj(v, ObjType::Float)) {
        *out = int64_t(static_cast<FloatObj*>(as_obj(v))->value);
        return true;
    }
    return say(e, "expected a number");
}

bool need_float(Value v, double* out, Err* e) {
    if (is_obj(v, ObjType::Float)) {
        *out = static_cast<FloatObj*>(as_obj(v))->value;
        return true;
    }
    if (bigint::is_integer(v)) {
        *out = bigint::to_double(v);
        return true;
    }
    return say(e, "expected a number");
}

/// A number into its C representation, at `where` -- which is exactly the
/// kind's own size, not widened.
bool store_scalar(Kind k, Value v, void* where, Err* e) {
    int64_t i = 0;
    double d = 0;
    switch (k) {
        case K_BOOL: { uint8_t b = truthy(v) ? 1 : 0; std::memcpy(where, &b, 1); return true; }
        case K_I8: { if (!need_int(v, &i, e)) return false; int8_t x = int8_t(i); std::memcpy(where, &x, 1); return true; }
        case K_U8: { if (!need_int(v, &i, e)) return false; uint8_t x = uint8_t(i); std::memcpy(where, &x, 1); return true; }
        case K_I16: { if (!need_int(v, &i, e)) return false; int16_t x = int16_t(i); std::memcpy(where, &x, 2); return true; }
        case K_U16: { if (!need_int(v, &i, e)) return false; uint16_t x = uint16_t(i); std::memcpy(where, &x, 2); return true; }
        case K_I32: { if (!need_int(v, &i, e)) return false; int32_t x = int32_t(i); std::memcpy(where, &x, 4); return true; }
        case K_U32: { if (!need_int(v, &i, e)) return false; uint32_t x = uint32_t(i); std::memcpy(where, &x, 4); return true; }
        case K_I64: case K_U64: { if (!need_int(v, &i, e)) return false; std::memcpy(where, &i, 8); return true; }
        case K_F32: { if (!need_float(v, &d, e)) return false; float x = float(d); std::memcpy(where, &x, 4); return true; }
        case K_F64: { if (!need_float(v, &d, e)) return false; std::memcpy(where, &d, 8); return true; }
        case K_PTR: {
            if (is_unit(v)) { void* n = nullptr; std::memcpy(where, &n, sizeof n); return true; }
            if (!is_fixnum(v)) return say(e, "expected an address, which is carried as an integer");
            void* q = reinterpret_cast<void*>(uintptr_t(fixnum_value(v)));
            std::memcpy(where, &q, sizeof q);
            return true;
        }
        default:
            return say(e, "not a scalar type");
    }
}

/// The C value at `where`, of exactly the kind's size, as a Dream value.
Value load_scalar(Process& p, Kind k, const void* where) {
    switch (k) {
        case K_VOID: return UNIT;
        case K_BOOL: { uint8_t x; std::memcpy(&x, where, 1); return make_bool(x != 0); }
        case K_I8: { int8_t x; std::memcpy(&x, where, 1); return make_fixnum(x); }
        case K_U8: { uint8_t x; std::memcpy(&x, where, 1); return make_fixnum(x); }
        case K_I16: { int16_t x; std::memcpy(&x, where, 2); return make_fixnum(x); }
        case K_U16: { uint16_t x; std::memcpy(&x, where, 2); return make_fixnum(x); }
        case K_I32: { int32_t x; std::memcpy(&x, where, 4); return make_fixnum(x); }
        case K_U32: { uint32_t x; std::memcpy(&x, where, 4); return make_fixnum(x); }
        case K_I64: { int64_t x; std::memcpy(&x, where, 8); return make_integer(p, x); }
        case K_U64: { uint64_t x; std::memcpy(&x, where, 8); return bigint::from_uint64(p.heap(), x); }
        case K_F32: { float x; std::memcpy(&x, where, 4); return p.heap().make_float(x); }
        case K_F64: { double x; std::memcpy(&x, where, 8); return p.heap().make_float(x); }
        case K_PTR: {
            void* q;
            std::memcpy(&q, where, sizeof q);
            return q ? make_integer(p, int64_t(uintptr_t(q))) : UNIT;
        }
        case K_CSTR: {
            const char* s;
            std::memcpy(&s, where, sizeof s);
            return s ? p.heap().make_string(s, uint32_t(std::strlen(s))) : UNIT;
        }
        default: return UNIT;
    }
}

/// A returned scalar. libffi widens an integer narrower than a register to a
/// whole `ffi_arg`, so reading the narrow member would be right on a
/// little-endian machine and wrong on a big-endian one; read the wide one.
Value load_return(Process& p, Kind k, const ArgSlot& slot) {
    switch (k) {
        case K_BOOL: return make_bool(slot.widened != 0);
        case K_I8: return make_fixnum(int8_t(slot.swidened));
        case K_I16: return make_fixnum(int16_t(slot.swidened));
        case K_I32: return make_fixnum(int32_t(slot.swidened));
        case K_U8: return make_fixnum(uint8_t(slot.widened));
        case K_U16: return make_fixnum(uint16_t(slot.widened));
        case K_U32: return make_fixnum(uint32_t(slot.widened));
        default: return load_scalar(p, k, &slot);
    }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

/// A Dream function handed to C for the length of one call.
///
/// libffi makes a small piece of code with the C signature, and calling it runs
/// the Dream function on the process that made the call, through the same
/// nested loop a native uses to force a value (`apply_whnf`). Four rules keep
/// that sound, and each one is checked rather than trusted:
///
/// - **The function must be pure.** Nothing a callback does can be ordered
///   against the C code around it, and a callback that parked -- a `recv!`, a
///   `join!` -- would have to be resumed with C's frames still on the stack,
///   which cannot be done. A pure function cannot park.
/// - **It lives for one call.** The closure is freed when the C function
///   returns. A library that keeps a callback and calls it later (a signal
///   handler, a completion callback) cannot be given one: that call would jump
///   into freed memory. Nothing here can detect it being kept.
/// - **It runs on the calling thread.** A call from another thread is refused
///   without running anything -- a process is not a thing two threads may run
///   at once -- and answered with zero.
/// - **An error does not unwind through C.** The callback answers zero, every
///   later call of it answers zero without running, and the error is raised
///   once the C function has returned.
///
/// Nothing collects while the call is in progress: the native does not vouch
/// for the collector (`VouchesForGc`), so every nested force pins the heap, and
/// the function and the strings C was handed stay where they are.
struct Callback {
    Process* p = nullptr;
    const Sig* sig = nullptr;
    Value fn = UNIT;
    std::thread::id thread;
    ffi_closure* closure = nullptr;
    void* code = nullptr;
    bool failed = false;
    bool raised = false;
    Value error = UNIT;
    std::string text;

    ~Callback() {
        if (closure) ffi_closure_free(closure);
    }
};

void callback_entry(ffi_cif*, void* ret, void** cargs, void* user) {
    auto* cb = static_cast<Callback*>(user);
    std::memset(ret, 0, sizeof(ffi_arg) > 8 ? sizeof(ffi_arg) : 8);
    if (cb->failed) return;
    if (std::this_thread::get_id() != cb->thread) {
        cb->failed = true;
        cb->text = "a callback was called from a thread other than the one that passed it";
        return;
    }
    Process& p = *cb->p;
    const Sig& s = *cb->sig;
    Value vals[MAX_ARGS];
    uint32_t n = uint32_t(s.args.size());
    for (uint32_t i = 0; i < n; ++i) vals[i] = load_scalar(p, s.args[i].kind, cargs[i]);
    if (n == 0) {
        vals[0] = UNIT;
        n = 1;
    }
    Value out;
    if (!apply_whnf(p, cb->fn, vals, n, &out)) {
        cb->failed = true;
        cb->raised = true;
        cb->error = out;
        return;
    }
    if (s.ret.kind == K_VOID) return;
    // Written the width libffi reads a return back at: a whole `ffi_arg` for an
    // integer, the value's own size for a float.
    ArgSlot slot{};
    Err e;
    if (!store_scalar(s.ret.kind, out, &slot, &e)) {
        cb->failed = true;
        cb->text = "a callback answered the wrong kind of value: " + e.text;
        return;
    }
    switch (s.ret.kind) {
        case K_BOOL: case K_U8: { ffi_arg w = slot.u8; std::memcpy(ret, &w, sizeof w); break; }
        case K_U16: { ffi_arg w = slot.u16; std::memcpy(ret, &w, sizeof w); break; }
        case K_U32: { ffi_arg w = slot.u32; std::memcpy(ret, &w, sizeof w); break; }
        case K_I8: { ffi_sarg w = slot.i8; std::memcpy(ret, &w, sizeof w); break; }
        case K_I16: { ffi_sarg w = slot.i16; std::memcpy(ret, &w, sizeof w); break; }
        case K_I32: { ffi_sarg w = slot.i32; std::memcpy(ret, &w, sizeof w); break; }
        case K_F32: std::memcpy(ret, &slot.f32, 4); break;
        default: std::memcpy(ret, &slot, 8); break;
    }
}

// ---------------------------------------------------------------------------
// A call
// ---------------------------------------------------------------------------

/// Everything one call collects on the way in: the handles it was given, which
/// become the parents of whatever it makes, and the callbacks it must free.
struct CallState {
    std::vector<uint64_t> parents;
    std::vector<std::unique_ptr<Callback>> callbacks;
};

bool marshal_arg(Process& p, const Type& t, Value v, ArgSlot* slot, CallState* cs, Err* e) {
    Value w;
    if (!force(p, v, &w, e)) return false;
    switch (t.kind) {
        case K_HANDLE: {
            uint64_t id;
            ForeignTable::Resource* r = resource_of(p, w, &id, e);
            if (!r) return false;
            if (r->tag != t.tag) return say(e, "expected a `" + t.tag + "` handle, not a `" + r->tag + "` one");
            slot->ptr = r->ptr;
            cs->parents.push_back(id);
            return true;
        }
        case K_PTR:
            // A handle is accepted wherever a raw pointer is: it is the safe way
            // to have one, and `address!` would only be the unsafe way round.
            if (is_obj(w, ObjType::Cons)) {
                uint64_t id;
                ForeignTable::Resource* r = resource_of(p, w, &id, e);
                if (!r) return false;
                slot->ptr = r->ptr;
                cs->parents.push_back(id);
                return true;
            }
            return store_scalar(K_PTR, w, slot, e);
        case K_CSTR:
            if (is_unit(w) || is_nil(w)) {
                slot->ptr = nullptr;
                return true;
            }
            // A `bigstr` is a view into the image and has no terminator.
            if (!is_obj(w, ObjType::Str)) return say(e, "expected a string");
            slot->ptr = static_cast<StrObj*>(as_obj(w))->data();  // stored NUL-terminated
            return true;
        case K_BYTES:
            if (is_unit(w)) {
                slot->ptr = nullptr;
                return true;
            }
            if (is_obj(w, ObjType::Str)) {
                slot->ptr = static_cast<StrObj*>(as_obj(w))->data();
                return true;
            }
            if (is_obj(w, ObjType::BigStr)) {
                slot->ptr = const_cast<char*>(static_cast<BigStrObj*>(as_obj(w))->data);
                return true;
            }
            return say(e, "expected a string or a payload view");
        case K_CALLBACK: {
            if (impure_callee(p, w)) {
                return say(e, "a callback must be a pure function: C cannot wait while it performs an effect");
            }
            auto cb = std::make_unique<Callback>();
            cb->p = &p;
            cb->sig = t.sig.get();
            cb->fn = w;
            cb->thread = std::this_thread::get_id();
            cb->closure = static_cast<ffi_closure*>(ffi_closure_alloc(sizeof(ffi_closure), &cb->code));
            if (!cb->closure) return say(e, "cannot allocate a callback");
            if (ffi_prep_closure_loc(cb->closure, &t.sig->cif, callback_entry, cb.get(), cb->code) != FFI_OK) {
                return say(e, "cannot prepare a callback");
            }
            slot->ptr = cb->code;
            cs->callbacks.push_back(std::move(cb));
            return true;
        }
        default:
            return store_scalar(t.kind, w, slot, e);
    }
}

/// A pointer or scalar C handed back, by return or through an `out`, as Dream.
Value marshal_result(Process& p, const Type& t, const ArgSlot& slot, bool is_return, CallState& cs) {
    switch (t.kind) {
        case K_OWN:
        case K_BORROW:
            if (!slot.ptr) return UNIT;
            return adopt(p, t.tag, slot.ptr, t.kind == K_OWN ? t.dtor : nullptr, 0, cs.parents);
        case K_TAKE_CSTR: {
            if (!slot.ptr) return UNIT;
            auto* s = static_cast<const char*>(slot.ptr);
            Value out = p.heap().make_string(s, uint32_t(std::strlen(s)));
            t.dtor(slot.ptr);
            return out;
        }
        default:
            return is_return ? load_return(p, t.kind, slot) : load_scalar(p, t.kind, &slot);
    }
}

Binding* binding_of(uint64_t id) {
    auto& reg = Registry::get();
    std::lock_guard<std::mutex> g(reg.mutex);
    if (id < 1 || id > reg.bindings.size()) return nullptr;
    return reg.bindings[id - 1].get();
}

NativeResult ffi_invoke(Process& p, Value callee, Value* args, uint32_t argc) {
    auto* nat = static_cast<NativeObj*>(as_obj(callee));
    Binding* b = binding_of(nat->user);
    if (!b) return fail(p, "this foreign function is no longer bound");
    const Sig& s = b->sig;
    if (argc < s.visible) {
        return fail(p, "`" + b->name + "` needs " + std::to_string(s.visible) + " arguments");
    }

    const uint32_t n = uint32_t(s.args.size());
    ArgSlot slots[MAX_ARGS];
    ArgSlot outs[MAX_ARGS];
    void* pointers[MAX_ARGS];
    CallState cs;
    uint32_t next = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const Type& t = s.args[i];
        if (t.kind == K_OUT) {
            outs[i].u64 = 0;
            slots[i].ptr = &outs[i];
        } else {
            Err e;
            if (!marshal_arg(p, t, args[next++], &slots[i], &cs, &e)) {
                return raise_err(p, e, "argument " + std::to_string(next) + " of `" + b->name + "`: ");
            }
        }
        pointers[i] = &slots[i];
    }

    ArgSlot ret{};
    ffi_call(const_cast<ffi_cif*>(&s.cif), FFI_FN(b->fn), &ret, pointers);

    // Whatever the call made is owned before anything can raise, so that an
    // error from a callback does not also leak it: an owned handle nobody
    // holds is still destroyed when the process ends.
    Value result = marshal_result(p, s.ret, ret, true, cs);
    if (s.has_out) {
        Pin kept(p, result);
        std::vector<Value> got;
        for (uint32_t i = 0; i < n; ++i) {
            if (s.args[i].kind == K_OUT) got.push_back(marshal_result(p, *s.args[i].inner, outs[i], false, cs));
        }
        Value list = NIL;
        for (size_t i = got.size(); i-- > 0;) list = p.heap().make_cons(got[i], list);
        result = p.heap().make_cons(kept.get(), list);
    }
    for (auto& cb : cs.callbacks) {
        if (!cb->failed) continue;
        if (cb->raised) return NativeResult::raise(cb->error);
        return fail(p, "in a callback passed to `" + b->name + "`: " + cb->text);
    }
    return NativeResult::ok(result);
}

/// Bind `symbol` in `lib` to a signature, and answer the function.
NativeResult bind_in(Process& p, Library* lib, Value symbol_v, Value args_v, Value ret_v, bool pure) {
    std::string symbol;
    Value sv = resolve(symbol_v);
    if (!str_value(sv, &symbol) || symbol.empty()) return fail(p, "a foreign function needs a symbol name");
    Err e;
    auto binding = std::make_unique<Binding>();
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(lib)) + "\n" + symbol + "\n";
    if (!parse_sig(p, args_v, ret_v, false, lib, &binding->sig, &key, &e)) {
        return raise_err(p, e, "the signature of `" + symbol + "`: ");
    }
    uint64_t id = 0;
    {
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> g(reg.mutex);
        auto it = reg.binding_ids.find(key);
        if (it != reg.binding_ids.end()) id = it->second;
    }
    if (id == 0) {
        void* fn = dl_sym(lib->handle, symbol);
        if (!fn) return fail(p, "cannot find `" + symbol + "` in " + lib->label);
        binding->fn = fn;
        binding->name = symbol;
        auto& reg = Registry::get();
        std::lock_guard<std::mutex> g(reg.mutex);
        auto it = reg.binding_ids.find(key);
        if (it != reg.binding_ids.end()) {
            id = it->second;
        } else {
            reg.bindings.push_back(std::move(binding));
            id = reg.bindings.size();
            reg.binding_ids.emplace(key, id);
        }
    }
    const Sig& s = binding_of(id)->sig;
    // A C function of no (visible) arguments still has to be callable, and
    // Dream applies a nullary function to unit, so it takes one it ignores.
    uint32_t arity = s.visible == 0 ? 1 : s.visible;
    std::string name = pure ? symbol : symbol + "!";
    Value name_v = p.heap().make_string(name.data(), uint32_t(name.size()));
    return NativeResult::ok(make_native(p, ffi_invoke, name_v, arity, 0xFFFFFFFFu, id));
}

// ---------------------------------------------------------------------------
// Members
// ---------------------------------------------------------------------------

/// `function lib symbol args result` and `pure_function ..`: the same, but for
/// the name the function carries, which is the runtime's reading of its purity.
/// Which one is decided by the member's `user`, so one body serves both.
NativeResult ffi_function(Process& p, Value callee, Value* args, uint32_t) {
    const bool pure = static_cast<NativeObj*>(as_obj(callee))->user == 1;
    Err e;
    Library* lib = library_of(p, args[0], &e);
    if (!lib) return raise_err(p, e);
    return bind_in(p, lib, args[1], args[2], args[3], pure);
}

/// `locate! lib`: where a library is, by opening it -- `Result.ok file`, or
/// `Result.error why` with the same account of what was tried that a binding
/// would raise. Opening is the only honest answer: a file that is there but
/// built for another machine is not a library this program can use.
NativeResult ffi_locate(Process& p, Value, Value* args, uint32_t) {
    Err e;
    Library* lib = library_of(p, args[0], &e);
    if (!lib && e.raised) return NativeResult::raise(e.value);
    std::string said = lib ? lib->where : e.text;
    Value text = p.heap().make_string(said.data(), uint32_t(said.size()));
    return NativeResult::ok(make_result(p.heap(), p.runtime(), lib != nullptr, text));
}

/// `call! f args`: a foreign function applied to a list. What a library server
/// does with a request, and what lets a function value and its arguments
/// travel in one message.
NativeResult ffi_call(Process& p, Value, Value* args, uint32_t) {
    Value f = resolve(args[0]);
    if (!is_obj(f, ObjType::Native) ||
        static_cast<NativeObj*>(as_obj(f))->fn != reinterpret_cast<void*>(&ffi_invoke)) {
        return fail(p, "call! needs a foreign function");
    }
    std::vector<Value> items;
    Err e;
    if (!list_items(p, args[1], &items, &e, "call!'s arguments")) return raise_err(p, e);
    if (items.empty()) items.push_back(UNIT);
    if (items.size() > MAX_ARGS) return fail(p, "too many arguments");
    // The native reads its arguments from here, and nothing can collect while
    // it does: it forces with the heap pinned.
    return ffi_invoke(p, f, items.data(), uint32_t(items.size()));
}

NativeResult ffi_release(Process& p, Value, Value* args, uint32_t) {
    Err e;
    uint64_t id;
    if (!resource_of(p, args[0], &id, &e)) return raise_err(p, e, "release!: ");
    release_one(*p.foreign, id);
    return NativeResult::ok(UNIT);
}

NativeResult ffi_alive(Process& p, Value, Value* args, uint32_t) {
    Err e;
    ForeignTable::Resource* r = resource_of(p, args[0], nullptr, &e);
    if (!r && e.raised) return NativeResult::raise(e.value);
    return NativeResult::ok(make_bool(r != nullptr));
}

NativeResult ffi_owned(Process& p, Value, Value*, uint32_t) {
    Value list = NIL;
    if (p.foreign) {
        std::vector<std::pair<std::string, uint64_t>> all;
        for (auto& [id, r] : p.foreign->live) all.emplace_back(r.tag, id);
        for (size_t i = all.size(); i-- > 0;) {
            Pin rest(p, list);
            Value h = make_handle(p, all[i].first, all[i].second);
            list = p.heap().make_cons(h, rest.get());
        }
    }
    return NativeResult::ok(list);
}

NativeResult ffi_buffer(Process& p, Value, Value* args, uint32_t) {
    Value n = resolve(args[0]);
    if (!is_fixnum(n) || fixnum_value(n) < 0) return fail(p, "buffer! needs a size in bytes");
    uint64_t size = uint64_t(fixnum_value(n));
    void* mem = std::calloc(size ? size_t(size) : 1, 1);
    if (!mem) return fail(p, "out of memory");
    return NativeResult::ok(adopt(p, "buffer", mem, &std::free, size, {}));
}

NativeResult ffi_size(Process& p, Value, Value* args, uint32_t) {
    Err e;
    ForeignTable::Resource* r = resource_of(p, args[0], nullptr, &e);
    if (!r) return raise_err(p, e, "size!: ");
    return NativeResult::ok(r->size ? make_integer(p, int64_t(r->size)) : UNIT);
}

NativeResult ffi_address(Process& p, Value, Value* args, uint32_t) {
    Err e;
    ForeignTable::Resource* r = resource_of(p, args[0], nullptr, &e);
    if (!r) return raise_err(p, e, "address!: ");
    return NativeResult::ok(make_integer(p, int64_t(uintptr_t(r->ptr))));
}

/// The bytes `[offset, offset + n)` of a sized resource, or an error. Every
/// read and write of foreign memory goes through here, which is what makes
/// them safe: a buffer knows how long it is.
char* span_of(Process& p, Value h, Value off_v, uint64_t n, const char* who, Err* e) {
    ForeignTable::Resource* r = resource_of(p, h, nullptr, e);
    if (!r) return nullptr;
    if (r->size == 0) {
        say(e, std::string(who) + ": a `" + r->tag + "` handle has no known size; only a buffer can be read and written");
        return nullptr;
    }
    Value o = resolve(off_v);
    if (!is_fixnum(o) || fixnum_value(o) < 0) {
        say(e, std::string(who) + " needs an offset");
        return nullptr;
    }
    uint64_t off = uint64_t(fixnum_value(o));
    if (off > r->size || n > r->size - off) {
        say(e, std::string(who) + ": " + std::to_string(n) + " bytes at " + std::to_string(off) +
                   " is outside a buffer of " + std::to_string(r->size));
        return nullptr;
    }
    return static_cast<char*>(r->ptr) + off;
}

const KindName* peekable(Process& p, Value t, Err* e) {
    Value v = resolve(t);
    const KindName* k = is_atom(v) ? scalar_named(atom_text(p, v)) : nullptr;
    if (!k || !(is_scalar(k->kind) || k->kind == K_PTR)) {
        say(e, "a buffer holds numbers, `:bool` and `:ptr`");
        return nullptr;
    }
    return k;
}

NativeResult ffi_peek(Process& p, Value, Value* args, uint32_t) {
    Err e;
    const KindName* k = peekable(p, args[2], &e);
    char* at = k ? span_of(p, args[0], args[1], k->size, "peek!", &e) : nullptr;
    if (!at) return raise_err(p, e);
    return NativeResult::ok(load_scalar(p, k->kind, at));
}

NativeResult ffi_poke(Process& p, Value, Value* args, uint32_t) {
    Err e;
    const KindName* k = peekable(p, args[2], &e);
    char* at = k ? span_of(p, args[0], args[1], k->size, "poke!", &e) : nullptr;
    if (!at || !store_scalar(k->kind, resolve(args[3]), at, &e)) return raise_err(p, e, "");
    return NativeResult::ok(UNIT);
}

NativeResult ffi_read(Process& p, Value, Value* args, uint32_t) {
    Value n = resolve(args[2]);
    if (!is_fixnum(n) || fixnum_value(n) < 0 || fixnum_value(n) > int64_t(UINT32_MAX)) {
        return fail(p, "read! needs a length");
    }
    Err e;
    char* at = span_of(p, args[0], args[1], uint64_t(fixnum_value(n)), "read!", &e);
    if (!at) return raise_err(p, e);
    return NativeResult::ok(p.heap().make_string(at, uint32_t(fixnum_value(n))));
}

NativeResult ffi_read_cstr_at(Process& p, Value, Value* args, uint32_t) {
    Err e;
    char* at = span_of(p, args[0], args[1], 0, "read_cstr!", &e);
    if (!at) return raise_err(p, e);
    // Bounded by the buffer, not by a terminator that may not be there.
    ForeignTable::Resource* r = resource_of(p, args[0], nullptr, &e);
    size_t room = size_t(r->size - uint64_t(at - static_cast<char*>(r->ptr)));
    size_t len = strnlen(at, room);
    return NativeResult::ok(p.heap().make_string(at, uint32_t(len)));
}

NativeResult ffi_write(Process& p, Value, Value* args, uint32_t) {
    Value s = resolve(args[2]);
    const char* data;
    uint64_t len;
    if (is_obj(s, ObjType::Str)) {
        data = static_cast<StrObj*>(as_obj(s))->data();
        len = static_cast<StrObj*>(as_obj(s))->len;
    } else if (is_obj(s, ObjType::BigStr)) {
        data = static_cast<BigStrObj*>(as_obj(s))->data;
        len = static_cast<BigStrObj*>(as_obj(s))->len;
    } else {
        return fail(p, "write! needs a string");
    }
    Err e;
    char* at = span_of(p, args[0], args[1], len, "write!", &e);
    if (!at) return raise_err(p, e);
    std::memcpy(at, data, size_t(len));
    return NativeResult::ok(UNIT);
}

NativeResult ffi_sizeof(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    const KindName* k = is_atom(v) ? scalar_named(atom_text(p, v)) : nullptr;
    if (!k) return fail(p, "sizeof needs a type atom such as `:i32`");
    return NativeResult::ok(make_fixnum(int64_t(k->size)));
}

NativeResult ffi_alignof(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    const KindName* k = is_atom(v) ? scalar_named(atom_text(p, v)) : nullptr;
    if (!k) return fail(p, "alignof needs a type atom such as `:i32`");
    return NativeResult::ok(make_fixnum(int64_t(k->align)));
}

NativeResult ffi_payload_index(Process& p, Value, Value* args, uint32_t) {
    std::string name;
    if (!str_value(args[0], &name)) return fail(p, "payload_index needs a name");
    int64_t i = p.runtime().has_image() ? p.runtime().image().data_index(name) : -1;
    return NativeResult::ok(i < 0 ? UNIT : make_fixnum(i));
}

// --- the first interface, kept -----------------------------------------------
//
// `open!`/`bind!` and raw addresses as integers. A binding made this way goes
// through the same signature reader as `function`, so it takes every type the
// new one does; what it lacks is only a library that knows its own purity.

NativeResult ffi_open(Process& p, Value, Value* args, uint32_t) {
    Err e;
    Library* lib = library_of(p, args[0], &e);
    if (!lib) return raise_err(p, e);
    auto& reg = Registry::get();
    std::lock_guard<std::mutex> g(reg.mutex);
    reg.numbered.push_back(lib);
    return NativeResult::ok(make_fixnum(int64_t(reg.numbered.size())));
}

Library* numbered_lib(Value h) {
    h = resolve(h);
    if (!is_fixnum(h)) return nullptr;
    auto& reg = Registry::get();
    std::lock_guard<std::mutex> g(reg.mutex);
    int64_t i = fixnum_value(h);
    if (i < 1 || size_t(i) > reg.numbered.size()) return nullptr;
    return reg.numbered[size_t(i) - 1];
}

NativeResult ffi_bind(Process& p, Value, Value* args, uint32_t) {
    Library* lib = numbered_lib(args[0]);
    if (!lib) return fail(p, "bind! was given a handle this runtime did not issue");
    return bind_in(p, lib, args[1], args[2], args[3], true);
}

NativeResult ffi_close(Process& p, Value, Value* args, uint32_t) {
    Value h = resolve(args[0]);
    auto& reg = Registry::get();
    std::lock_guard<std::mutex> g(reg.mutex);
    int64_t i = is_fixnum(h) ? fixnum_value(h) : 0;
    if (i < 1 || size_t(i) > reg.numbered.size() || !reg.numbered[size_t(i) - 1]) {
        return fail(p, "close! was given a handle this runtime did not issue");
    }
    // Bindings hold raw pointers into the library, so it stays mapped and only
    // the number is forgotten: unmapping under a live binding would be a crash
    // waiting to happen, and a program that closed a library is not asking for
    // that.
    reg.numbered[size_t(i) - 1] = nullptr;
    return NativeResult::ok(UNIT);
}

NativeResult ffi_load(Process& p, Value, Value* args, uint32_t) {
    Err e;
    Library* lib = library_of(p, args[0], &e);
    if (!lib) return raise_err(p, e);
    return bind_in(p, lib, args[1], args[2], args[3], true);
}

NativeResult ffi_alloc(Process& p, Value, Value* args, uint32_t) {
    Value n = resolve(args[0]);
    if (!is_fixnum(n) || fixnum_value(n) < 0) return fail(p, "alloc! needs a size");
    void* mem = std::calloc(size_t(fixnum_value(n)), 1);
    if (!mem) return fail(p, "out of memory");
    return NativeResult::ok(make_integer(p, int64_t(uintptr_t(mem))));
}

NativeResult ffi_free(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return fail(p, "free! needs an address");
    std::free(reinterpret_cast<void*>(uintptr_t(fixnum_value(v))));
    return NativeResult::ok(UNIT);
}

NativeResult ffi_read_cstr(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return fail(p, "read_cstr! needs an address");
    auto* s = reinterpret_cast<const char*>(uintptr_t(fixnum_value(v)));
    if (!s) return NativeResult::ok(UNIT);
    return NativeResult::ok(p.heap().make_string(s, uint32_t(std::strlen(s))));
}

NativeResult ffi_read_u8(Process& p, Value, Value* args, uint32_t) {
    Value base = resolve(args[0]);
    Value off = resolve(args[1]);
    if (!is_fixnum(base) || !is_fixnum(off)) return fail(p, "read_u8! needs an address and an offset");
    auto* q = reinterpret_cast<uint8_t*>(uintptr_t(fixnum_value(base))) + fixnum_value(off);
    return NativeResult::ok(make_fixnum(*q));
}

NativeResult ffi_write_u8(Process& p, Value, Value* args, uint32_t) {
    Value base = resolve(args[0]);
    Value off = resolve(args[1]);
    Value val = resolve(args[2]);
    if (!is_fixnum(base) || !is_fixnum(off) || !is_fixnum(val)) {
        return fail(p, "write_u8! needs an address, an offset and a byte");
    }
    auto* q = reinterpret_cast<uint8_t*>(uintptr_t(fixnum_value(base))) + fixnum_value(off);
    *q = uint8_t(fixnum_value(val));
    return NativeResult::ok(UNIT);
}

}  // namespace

ModuleDef make_ffi_module() {
    // Strict masks: a signature is read by forcing it here, element by
    // element, so only the positions that are read whole are forced first.
    return ModuleDef{"std.ffi",
                     {
                         {"function", 4, 0b0010, ffi_function, 0},
                         {"pure_function", 4, 0b0010, ffi_function, 1},
                         {"call!", 2, 0b01, ffi_call},
                         {"locate!", 1, 0b0, ffi_locate},
                         {"release!", 1, 0b0, ffi_release},
                         {"alive!", 1, 0b0, ffi_alive},
                         {"owned!", 1, 0b0, ffi_owned},
                         {"buffer!", 1, 0b1, ffi_buffer},
                         {"size!", 1, 0b0, ffi_size},
                         {"address!", 1, 0b0, ffi_address},
                         {"peek!", 3, 0b110, ffi_peek},
                         {"poke!", 4, 0b1110, ffi_poke},
                         {"read!", 3, 0b110, ffi_read},
                         {"read_string!", 2, 0b10, ffi_read_cstr_at},
                         {"write!", 3, 0b110, ffi_write},
                         {"sizeof", 1, 0b1, ffi_sizeof},
                         {"alignof", 1, 0b1, ffi_alignof},
                         {"payload_index", 1, 0b1, ffi_payload_index},
                         {"open!", 1, 0b1, ffi_open},
                         {"close!", 1, 0b1, ffi_close},
                         {"bind!", 4, 0b0011, ffi_bind},
                         {"load!", 4, 0b0010, ffi_load},
                         {"alloc!", 1, 0b1, ffi_alloc},
                         {"free!", 1, 0b1, ffi_free},
                         {"read_cstr!", 1, 0b1, ffi_read_cstr},
                         {"read_u8!", 2, 0b11, ffi_read_u8},
                         {"write_u8!", 3, 0b111, ffi_write_u8},
                     }};
}

}  // namespace dream
