#include "image.hpp"

#include "builtins.hpp"

#include "windows.hpp"
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstring>
#include <bit>
#include <limits>
#include <vector>

namespace dream {
static_assert(std::endian::native == std::endian::little,
              "The mapped Dream image reader requires a little-endian host");
static_assert(sizeof(void*) == 8, "The Dream VM requires a 64-bit host");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Dream images store IEEE 754 binary64 floats");

namespace {

constexpr uint8_t MAGIC[8] = {'D', 'A', 'G', 'N', 'C', 'A', 'A', 'F'};
constexpr uint16_t SUPPORTED_MAJOR = 0;
constexpr size_t HEADER_SIZE = 32;
constexpr size_t SECTION_ENTRY_SIZE = 16;
// Deliberately not a constant here: the builtin table lives in builtins.cpp,
// and a second copy of its size is a copy that drifts. It did -- a builtin was
// appended and every image using it was rejected as malformed.
inline uint32_t builtin_limit() { return builtin_count(); }

constexpr uint32_t tag(const char s[5]) {
    return uint32_t(uint8_t(s[0])) | (uint32_t(uint8_t(s[1])) << 8) |
           (uint32_t(uint8_t(s[2])) << 16) | (uint32_t(uint8_t(s[3])) << 24);
}

inline uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t rd64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

struct SectionInfo {
    uint32_t kind, offset, length, count;
};

/// The node indices a node points at. Shared by the validator's two walks so
/// they cannot disagree about what counts as a child.
void node_children(const Node& n, const uint32_t* kids, std::vector<uint32_t>& out) {
    auto push_kids = [&](uint32_t off, uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) out.push_back(kids[off + i]);
    };
    switch (static_cast<Op>(n.op)) {
        case Op::Field: case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs: case Op::ListTail: case Op::ListIsEmpty:
            out.push_back(n.a);
            break;
        DREAM_PRIMITIVE_CASES
            push_kids(n.b, n.c);
            break;
        case Op::Apply:
            out.push_back(n.a);
            push_kids(n.b, n.c);
            break;
        case Op::If:
            out.push_back(n.a);
            out.push_back(n.b);
            out.push_back(n.c);
            break;
        case Op::Block: case Op::MakeList: case Op::MakeArray: case Op::RecordMake:
            push_kids(n.a, n.b);
            break;
        case Op::SetMany:
            out.push_back(n.a);
            push_kids(n.b, uint64_t(n.c) * 2);
            break;
        case Op::MakeMap:
            push_kids(n.a, uint64_t(n.b) * 2);
            break;
        case Op::SwitchHead: case Op::SwitchAtom:
            out.push_back(n.a);
            push_kids(n.b, n.c);
            break;
        case Op::Bind:
            out.push_back(n.b);
            break;
        case Op::Try: case Op::Ensure:
            out.push_back(n.a);
            out.push_back(n.b);
            break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
        case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
        case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
        case Op::And: case Op::Or: case Op::Cons:
            out.push_back(n.a);
            out.push_back(n.b);
            break;
        case Op::Get: case Op::Set:
            // A `get`'s fallback is optional, and absent is `NO_NODE`, which
            // the walks skip as they skip an `if` with no `else`.
            out.push_back(n.a);
            out.push_back(n.b);
            out.push_back(n.c);
            break;
        default:
            break;
    }
}

}  // namespace

const char* op_name(Op op) {
    switch (op) {
        case Op::Nop: return "nop";
        case Op::ConstInt: return "int";
        case Op::ConstFloat: return "float";
        case Op::ConstStr: return "str";
        case Op::ConstChar: return "char";
        case Op::ConstBool: return "bool";
        case Op::ConstAtom: return "atom";
        case Op::Unit: return "unit";
        case Op::Local: return "local";
        case Op::Capture: return "capture";
        case Op::Global: return "global";
        case Op::Builtin: return "builtin";
        case Op::Field: return "field";
        case Op::Apply: return "apply";
        case Op::If: return "if";
        case Op::Block: return "block";
        case Op::Bind: return "bind";
        case Op::MakeClosure: return "closure";
        case Op::MakeThunk: return "thunk";
        case Op::Try: return "try";
        case Op::Ensure: return "ensure";
        case Op::Force: return "force";
        case Op::TypeIs: return "type_is";
        case Op::StrChars: return "str_chars";
        case Op::ErrorNew: return "error_new";
        case Op::ErrorKind: return "error_kind";
        case Op::ErrorPayload: return "error_payload";
        case Op::StrSlice: return "str_slice";
        case Op::StrFind: return "str_find";
        case Op::StrByte: return "str_byte";
        case Op::StrLe: return "str_le";
        case Op::StrSpan: return "str_span";
        case Op::StrUpto: return "str_upto";
        case Op::CharCode: return "char_code";
        case Op::CharOfCode: return "char_of_code";
        case Op::ToFloat: return "to_float";
        case Op::FloatBytes: return "float_bytes";
        case Op::FloatOfBytes: return "float_of_bytes";
        case Op::ToInt: return "to_int";
        case Op::ParseInt: return "parse_int";
        case Op::ParseFloat: return "parse_float";
        case Op::ToExistingAtom: return "to_existing_atom";
        case Op::ArrayNew: return "array_new";
        case Op::ArrayToList: return "array_to_list";
        case Op::MapHas: return "map_has";
        case Op::MapRemove: return "map_remove";
        case Op::MapPairs: return "map_pairs";
        case Op::DataCount: return "data_count";
        case Op::DataAt: return "data_at";
        case Op::Compare: return "compare";
        case Op::StrInterp: return "str_interp";

        case Op::Cons: return "cons";
        case Op::ListTail: return "list_tail";
        case Op::ListIsEmpty: return "list_is_empty";
        case Op::Add: return "add";
        case Op::Sub: return "sub";
        case Op::Mul: return "mul";
        case Op::Div: return "div";
        case Op::Mod: return "mod";
        case Op::Eq: return "eq";
        case Op::Ne: return "ne";
        case Op::Lt: return "lt";
        case Op::Le: return "le";
        case Op::Gt: return "gt";
        case Op::Ge: return "ge";
        case Op::And: return "and";
        case Op::Or: return "or";
        case Op::Neg: return "neg";
        case Op::BitAnd: return "band";
        case Op::BitOr: return "bor";
        case Op::BitXor: return "bxor";
        case Op::BitNot: return "bnot";
        case Op::Shl: return "shl";
        case Op::Shr: return "shr";
        case Op::Not: return "not";
        case Op::MakeList: return "list";
        case Op::MakeArray: return "array";
        case Op::RecordMake: return "record";
        case Op::MakeMap: return "map";
        case Op::Get: return "get";
        case Op::Set: return "set";
        case Op::SetMany: return "set_many";
        case Op::SwitchHead: return "switch_head";
        case Op::SwitchAtom: return "switch_atom";
        default: return "<bad op>";
    }
}

bool StringRef::equals(const char* s) const {
    size_t n = std::strlen(s);
    return n == len && std::memcmp(data, s, n) == 0;
}

namespace {
void unmap_image(void* address, size_t size) {
#ifdef _WIN32
    UnmapViewOfFile(address);
#else
    ::munmap(address, size);
#endif
}
}

Image::~Image() {
    if (owns_mapping_ && mapping_) {
        unmap_image(mapping_, mapping_size_);
    }
}

bool Image::load_file(const std::string& path, std::string& error) {
#ifdef _WIN32
    HANDLE file = CreateFileW(windows::wide(path).c_str(), GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "cannot open " + path + ": " + windows::error();
        return false;
    }
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(file, &length) || length.QuadPart < int64_t(HEADER_SIZE) ||
        uint64_t(length.QuadPart) > SIZE_MAX) {
        error = path + " has an invalid bytecode image size";
        CloseHandle(file);
        return false;
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    void* p = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (mapping) CloseHandle(mapping);
    if (!p) { error = "cannot map " + path + ": " + windows::error(); return false; }
    size_t file_size = size_t(length.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        error = "cannot stat " + path;
        ::close(fd);
        return false;
    }
    if (st.st_size < static_cast<off_t>(HEADER_SIZE)) {
        error = path + " is too small to be a bytecode image";
        ::close(fd);
        return false;
    }
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        error = "cannot map " + path;
        return false;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
#endif
    // The mapping is what gets unmapped, whatever the image turns out to start
    // at: skipping a shebang below moves `data_` off the page boundary, and
    // munmap only accepts the address it handed out.
    mapping_ = p;
    mapping_size_ = file_size;

    // A shebang line, so that an image can be marked executable and run
    // directly. The compiler's `--shebang` writes one. Anything before the first
    // newline is skipped; an image is binary and never starts with '#'
    // otherwise, since the magic number begins with 'D'.
    const uint8_t* start = static_cast<const uint8_t*>(p);
    size_t size = file_size;
    if (size > 0 && *start == '#') {
        const void* nl = std::memchr(start, '\n', size);
        if (nl == nullptr) {
            error = path + " begins with `#` and has no newline, so it is not an image";
            unmap_image(mapping_, mapping_size_);
            mapping_ = nullptr;
            return false;
        }
        size_t offset = static_cast<const uint8_t*>(nl) - start + 1;
        start += offset;
        size -= offset;
    }
    data_ = start;
    size_ = size;
    owns_mapping_ = true;
    if (!parse(error)) {
        error = path + ": " + error;
        return false;
    }
    return true;
}

bool Image::load_bytes(const uint8_t* data, size_t size, std::string& error) {
    data_ = data;
    size_ = size;
    owns_mapping_ = false;
    return parse(error);
}

bool Image::parse(std::string& error) {
    if (size_ < HEADER_SIZE) {
        error = "truncated header";
        return false;
    }
    if (std::memcmp(data_, MAGIC, 8) != 0) {
        error = "bad magic number (expected DAGNCAAF)";
        return false;
    }
    version_major_ = rd16(data_ + 8);
    version_minor_ = rd16(data_ + 10);
    if (version_major_ != SUPPORTED_MAJOR) {
        error = "unsupported bytecode version " + std::to_string(version_major_) + "." +
                std::to_string(version_minor_);
        return false;
    }
    flags_ = rd32(data_ + 12);
    module_name_ = rd32(data_ + 16);
    source_name_ = rd32(data_ + 20);
    entry_ = rd32(data_ + 24);
    uint32_t n_sections = rd32(data_ + 28);

    if (n_sections > 4096) {
        error = "implausible section count";
        return false;
    }
    if (HEADER_SIZE + size_t(n_sections) * SECTION_ENTRY_SIZE > size_) {
        error = "section table extends past end of file";
        return false;
    }

    for (uint32_t i = 0; i < n_sections; ++i) {
        const uint8_t* e = data_ + HEADER_SIZE + size_t(i) * SECTION_ENTRY_SIZE;
        SectionInfo s{rd32(e), rd32(e + 4), rd32(e + 8), rd32(e + 12)};
        if (size_t(s.offset) + s.length > size_) {
            error = "section extends past end of file";
            return false;
        }
        const uint8_t* base = data_ + s.offset;

        // A record array must actually hold `count` records.
        auto fits = [&](size_t record_size) {
            return size_t(s.count) * record_size <= s.length;
        };
        if (s.kind == tag("KINT")) {
            if (!fits(8)) { error = "KINT section is short"; return false; }
            ints_ = reinterpret_cast<const int64_t*>(base); n_ints_ = s.count;
        } else if (s.kind == tag("KFLT")) {
            if (!fits(8)) { error = "KFLT section is short"; return false; }
            floats_ = reinterpret_cast<const double*>(base); n_floats_ = s.count;
        } else if (s.kind == tag("KSTR")) {
            if (!fits(8)) { error = "KSTR section is short"; return false; }
            strs_ = reinterpret_cast<const StrRec*>(base); n_strs_ = s.count;
        } else if (s.kind == tag("SBLB")) {
            blob_ = reinterpret_cast<const char*>(base); n_blob_ = s.length;
        } else if (s.kind == tag("KATM")) {
            if (!fits(4)) { error = "KATM section is short"; return false; }
            atoms_ = reinterpret_cast<const uint32_t*>(base); n_atoms_ = s.count;
        } else if (s.kind == tag("NODE")) {
            if (!fits(sizeof(Node))) { error = "NODE section is short"; return false; }
            nodes_ = reinterpret_cast<const Node*>(base); n_nodes_ = s.count;
        } else if (s.kind == tag("KIDS")) {
            if (!fits(4)) { error = "KIDS section is short"; return false; }
            kids_ = reinterpret_cast<const uint32_t*>(base); n_kids_ = s.count;
        } else if (s.kind == tag("FUNC")) {
            if (!fits(sizeof(FuncRec))) { error = "FUNC section is short"; return false; }
            funcs_ = reinterpret_cast<const FuncRec*>(base); n_funcs_ = s.count;
        } else if (s.kind == tag("GLBL")) {
            if (!fits(sizeof(GlobalRec))) { error = "GLBL section is short"; return false; }
            globals_ = reinterpret_cast<const GlobalRec*>(base); n_globals_ = s.count;
        } else if (s.kind == tag("IMPT")) {
            if (!fits(sizeof(ImportRec))) { error = "IMPT section is short"; return false; }
            imports_ = reinterpret_cast<const ImportRec*>(base); n_imports_ = s.count;
        } else if (s.kind == tag("MODS")) {
            if (!fits(sizeof(ModuleRec))) { error = "MODS section is short"; return false; }
            modules_ = reinterpret_cast<const ModuleRec*>(base); n_modules_ = s.count;
        } else if (s.kind == tag("LDAT")) {
            if (!fits(sizeof(DataRec))) { error = "LDAT section is short"; return false; }
            large_ = reinterpret_cast<const DataRec*>(base); n_large_ = s.count;
        } else if (s.kind == tag("LNAM")) {
            // Variable-length records, so there is no stride to check here;
            // `validate` walks them.
            names_ = base; names_len_ = s.length; n_names_ = s.count;
        } else if (s.kind == tag("PAYL")) {
            // The section table entry describes only the 8-byte header -- the
            // honest size of what the layout reserved. The payload itself
            // follows and may be larger than a `u32` can count, so its length
            // is that header and nowhere else.
            if (s.length < 8) { error = "PAYL section is short"; return false; }
            payload_len_ = rd64(base);
            payload_off_ = s.offset;
            payload_ = reinterpret_cast<const char*>(base + 8);
            has_payload_ = true;
        } else if (s.kind == tag("ITYP")) {
            if (!fits(sizeof(TypeRec))) { error = "ITYP section is short"; return false; }
            integer_types_ = reinterpret_cast<const TypeRec*>(base); n_integer_types_ = s.count;
        } else if (s.kind == tag("TYPE")) {
            if (!fits(sizeof(TypeRec))) { error = "TYPE section is short"; return false; }
            types_ = reinterpret_cast<const TypeRec*>(base); n_types_ = s.count;
        } else if (s.kind == tag("VMOP")) {
            if (!fits(sizeof(VmOptionRec))) { error = "VMOP section is short"; return false; }
            vm_options_ = reinterpret_cast<const VmOptionRec*>(base); n_vm_options_ = s.count;
        } else if (s.kind == tag("SPAN")) {
            if (!fits(sizeof(SpanRec))) { error = "SPAN section is short"; return false; }
            spans_ = reinterpret_cast<const SpanRec*>(base); n_spans_ = s.count;
        }
        // Unknown sections are skipped, as the format requires.
    }
    if (!validate(error)) return false;
    if (n_types_) {
        float_params_.assign(n_funcs_, 0);
        for (uint32_t i = 0; i < n_types_; ++i) float_params_[types_[i].func] = types_[i].params;
    }
    if (n_integer_types_) {
        integer_params_.assign(n_funcs_, 0);
        for (uint32_t i = 0; i < n_integer_types_; ++i)
            integer_params_[integer_types_[i].func] = integer_types_[i].params;
    }
    // Decode trivial record readers once, without changing the image format.
    accessors_.assign(n_funcs_, Accessor{});
    for (uint32_t i = 0; i < n_funcs_; ++i) {
        const auto& f = funcs_[i];
        const auto& body = node(f.body);
        if (Op(body.op) != Op::Get) continue;
        const auto& container = node(body.a);
        const auto& key = node(body.b);
        if (Op(container.op) != Op::Local || container.a >= f.arity) continue;
        if (Op(key.op) == Op::ConstInt) {
            accessors_[i] = Accessor{container.a, NO_NODE, integer(key.a)};
        } else if (Op(key.op) == Op::Local && key.a < f.arity) {
            accessors_[i] = Accessor{container.a, key.a, 0};
        }
    }
    // The parameters a function forces before it does anything else: the run
    // of strict `local` reads at the head of its body that `lower_body` emits
    // for `!acc`. Only that prefix counts, because it is what makes evaluating
    // the argument in the caller indistinguishable from forcing it on entry --
    // nothing of the callee's has run yet either way, and the strict
    // parameters are forced in the same order. See "Strict arguments are
    // evaluated where they stand" in docs/notes/vm-performance.md.
    strict_args_.assign(n_funcs_, 0);
    for (uint32_t i = 0; i < n_funcs_; ++i) {
        const auto& f = funcs_[i];
        const auto& body = node(f.body);
        if (Op(body.op) != Op::Block) continue;
        uint32_t mask = 0;
        for (uint32_t s = 0; s + 1 < body.b; ++s) {
            const auto& stmt = node(kid(body.a + s));
            if (Op(stmt.op) != Op::Local || !(stmt.flags & F_STRICT) ||
                stmt.a >= f.arity || stmt.a >= 32)
                break;
            mask |= 1u << stmt.a;
        }
        strict_args_[i] = mask;
    }
    return true;
}

int Image::module_of_global(uint32_t global_index) const {
    for (uint32_t i = 0; i < n_modules_; ++i) {
        const ModuleRec& m = modules_[i];
        if (global_index >= m.globals_start && global_index < m.globals_start + m.globals_count) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

SpanRec Image::span(uint32_t node_index) const {
    if (!spans_ || node_index >= n_spans_) return SpanRec{0, 0};
    return spans_[node_index];
}

int Image::find_global(const char* name) const {
    for (uint32_t i = 0; i < n_globals_; ++i) {
        if (str(globals_[i].name).equals(name)) return static_cast<int>(i);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Validation
//
// The interpreter indexes these arrays with no further bounds checks, so every
// index in the image is proven in range here, once. Anything that fails leaves
// the image unusable rather than trusting it partially.
// ---------------------------------------------------------------------------

bool Image::validate(std::string& error) {
    auto fail = [&](const std::string& what) {
        error = what;
        return false;
    };

    if (module_name_ >= n_strs_ || source_name_ >= n_strs_) return fail("bad module or source name index");
    if (entry_ != NO_NODE && entry_ >= n_funcs_) return fail("entry point is not a valid function");

    for (uint32_t i = 0; i < n_strs_; ++i) {
        if (size_t(strs_[i].offset) + strs_[i].length > n_blob_) return fail("string extends past the string blob");
    }
    for (uint32_t i = 0; i < n_atoms_; ++i) {
        if (atoms_[i] >= n_strs_) return fail("atom names an out-of-range string");
    }
    for (uint32_t i = 0; i < n_imports_; ++i) {
        if (imports_[i].path >= n_strs_ || imports_[i].alias >= n_strs_) return fail("bad import string index");
    }
    for (uint32_t i = 0; i < n_globals_; ++i) {
        const GlobalRec& g = globals_[i];
        if (g.name >= n_strs_) return fail("bad global name index");
        if (g.kind == GLOBAL_FUNCTION) {
            if (g.target >= n_funcs_) return fail("global names an out-of-range function");
        } else if (g.kind == GLOBAL_MODULE) {
            if (g.target >= n_imports_) return fail("global names an out-of-range import");
        } else {
            return fail("unknown global kind");
        }
    }
    for (uint32_t i = 0; i < n_modules_; ++i) {
        const ModuleRec& m = modules_[i];
        if (m.name >= n_strs_ || m.source >= n_strs_) return fail("bad module name index");
        if (size_t(m.globals_start) + m.globals_count > n_globals_) {
            return fail("module's global range is out of bounds");
        }
        if (m.derives != NO_NODE && m.derives >= n_modules_) {
            return fail("module derives an out-of-range module");
        }
    }
    // Large data. The two sections are one feature and each is meaningless
    // alone: a table of descriptors with nothing to describe, or a region of
    // bytes nothing can name. An image carrying one and not the other was
    // built wrong, and saying so here is cheaper than a null deref later.
    if ((large_ != nullptr) != has_payload_) {
        return fail(large_ ? "LDAT section without a PAYL section"
                           : "PAYL section without an LDAT section");
    }
    if (has_payload_) {
        // The one sum in the container that can leave a `u32` behind, so it is
        // the one done in 64 bits and written as a subtraction -- adding the
        // length to the offset is exactly what would wrap.
        const size_t header_end = payload_off_ + 8;
        if (header_end > size_ || payload_len_ > uint64_t(size_ - header_end)) {
            return fail("the payload extends past end of file");
        }
        for (uint32_t i = 0; i < n_large_; ++i) {
            const DataRec& d = large_[i];
            if (d.offset > payload_len_ || d.length > payload_len_ - d.offset) {
                return fail("large datum " + std::to_string(i) + " extends past the payload");
            }
        }
    }
    // Payload names: `u32 length` then the bytes, once per datum, unpadded.
    // A name table is only meaningful beside the data it names, and naming a
    // datum that does not exist is a build error rather than something to
    // tolerate.
    data_names_.clear();
    if (names_) {
        if (!has_payload_) return fail("LNAM section without a payload");
        if (n_names_ > n_large_) return fail("LNAM names more data than LDAT describes");
        uint32_t at = 0;
        for (uint32_t i = 0; i < n_names_; ++i) {
            if (names_len_ - at < 4) return fail("LNAM section is short");
            uint32_t n = uint32_t(names_[at]) | uint32_t(names_[at + 1]) << 8 |
                         uint32_t(names_[at + 2]) << 16 | uint32_t(names_[at + 3]) << 24;
            at += 4;
            if (n > names_len_ - at) return fail("LNAM name extends past its section");
            data_names_.emplace_back(reinterpret_cast<const char*>(names_ + at), n);
            at += n;
        }
    }

    for (uint32_t i = 0; i < n_integer_types_; ++i) {
        if (integer_types_[i].func >= n_funcs_) return fail("integer type record names an out-of-range function");
    }
    for (uint32_t i = 0; i < n_types_; ++i) {
        if (types_[i].func >= n_funcs_) return fail("type record names an out-of-range function");
    }
    for (uint32_t i = 0; i < n_funcs_; ++i) {
        const FuncRec& f = funcs_[i];
        if (f.name >= n_strs_) return fail("bad function name index");
        if (f.arity > f.slots) return fail("function arguments exceed its frame slots");
        if (f.body >= n_nodes_) {
            // Which function, because a compiler that produced this needs to
            // know where to look, and the number is the only handle it has.
            return fail("function " + std::to_string(i) + " (`" +
                        (f.name < n_strs_ ? str(f.name).str() : std::string("?")) +
                        "`) has body node " + std::to_string(f.body) + ", but the image has " +
                        std::to_string(n_nodes_) + " nodes");
        }
        if (size_t(f.captures_off) + f.n_captures > n_kids_) return fail("capture list extends past the kids pool");
    }
    for (uint32_t i = 0; i < n_kids_; ++i) {
        (void)i;  // Kid meaning depends on the referring node; checked below.
    }

    // Structural pass: every operand of every node must be in range.
    auto node_ok = [&](uint32_t n) { return n < n_nodes_; };
    auto kids_ok = [&](uint32_t off, uint64_t count) {
        if (uint64_t(off) + count > n_kids_) return false;
        for (uint64_t k = 0; k < count; ++k) {
            if (!node_ok(kids_[off + k])) return false;
        }
        return true;
    };

    for (uint32_t i = 0; i < n_nodes_; ++i) {
        const Node& n = nodes_[i];
        if (n.op >= static_cast<uint8_t>(Op::Count)) return fail("unknown opcode");
        switch (static_cast<Op>(n.op)) {
            case Op::ConstInt: if (n.a >= n_ints_) return fail("bad integer constant index"); break;
            case Op::ConstFloat: if (n.a >= n_floats_) return fail("bad float constant index"); break;
            case Op::ConstStr: if (n.a >= n_strs_) return fail("bad string constant index"); break;
            case Op::ConstAtom: if (n.a >= n_atoms_) return fail("bad atom index"); break;
            case Op::ConstChar: if (n.a > 0x10FFFF || (n.a >= 0xD800 && n.a <= 0xDFFF)) return fail("char constant is not a Unicode scalar value"); break;
            case Op::ConstBool: if (n.a > 1) return fail("bool constant is neither 0 nor 1"); break;
            case Op::Global: if (n.a >= n_globals_) return fail("bad global index"); break;
            case Op::Builtin: if (n.a >= builtin_limit()) return fail("unknown builtin id"); break;
            case Op::MakeClosure:
            case Op::MakeThunk: if (n.a >= n_funcs_) return fail("bad function index"); break;
            case Op::Field:
                if (!node_ok(n.a)) return fail("bad field object node");
                if (n.b >= n_strs_) return fail("bad field name index");
                break;
            DREAM_PRIMITIVE_CASES
                if (n.a != primitive_builtin(Op(n.op)) || n.c != primitive_arity(Op(n.op))
                    || !kids_ok(n.b, n.c)) return fail("bad primitive operands");
                break;
            case Op::Apply:
                if (!node_ok(n.a)) return fail("bad callee node");
                if (!kids_ok(n.b, n.c)) return fail("bad argument list");
                break;
            case Op::If:
                if (!node_ok(n.a) || !node_ok(n.b)) return fail("bad if branch");
                if (n.c != NO_NODE && !node_ok(n.c)) return fail("bad else branch");
                break;
            case Op::Block:
            case Op::MakeList:
            case Op::MakeArray:
                if (!kids_ok(n.a, n.b)) return fail("bad child list");
                break;
            case Op::MakeMap:
                if (!kids_ok(n.a, uint64_t(n.b) * 2)) return fail("bad map entry list");
                break;
            case Op::RecordMake:
                if (!kids_ok(n.a, n.b)) return fail("bad record field list");
                if ((n.c & RECORD_MAKE_ATOM) >= n_atoms_) return fail("bad record id");
                break;
            case Op::SwitchHead: case Op::SwitchAtom:
                if (!node_ok(n.a)) return fail("bad switch subject");
                if (n.c % 2 != 1 || !kids_ok(n.b, n.c)) return fail("bad switch table");
                // The interpreter reads each key as an atom without asking.
                for (uint32_t k = 0; k + 1 < n.c; k += 2) {
                    if (Op(nodes_[kids_[n.b + k]].op) != Op::ConstAtom) {
                        return fail("switch key is not an atom constant");
                    }
                }
                break;
            case Op::Bind: if (!node_ok(n.b)) return fail("bad bind value"); break;
            case Op::Try:
                if (!node_ok(n.a) || !node_ok(n.b)) return fail("bad try body or handler");
                break;
            case Op::Ensure:
                if (!node_ok(n.a) || !node_ok(n.b)) return fail("bad ensure body or cleanup");
                break;
            case Op::Force: case Op::Neg: case Op::Not: case Op::BitNot: case Op::TypeIs: case Op::ListTail: case Op::ListIsEmpty:
                if (!node_ok(n.a)) return fail("bad unary operand");
                if (Op(n.op) == Op::TypeIs && (n.b > DREAM_TYPE_MAP || n.c > 1))
                    return fail("bad type test");
                break;
            case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Mod:
            case Op::BitAnd: case Op::BitOr: case Op::BitXor: case Op::Shl: case Op::Shr:
            case Op::Eq: case Op::Ne: case Op::Lt: case Op::Le: case Op::Gt: case Op::Ge:
            case Op::And: case Op::Or: case Op::Cons:
                if (!node_ok(n.a) || !node_ok(n.b)) return fail("bad binary operand");
                break;
            case Op::Get:
                if (!node_ok(n.a) || !node_ok(n.b)) return fail("bad get operand");
                if (n.c != NO_NODE && !node_ok(n.c)) return fail("bad get fallback");
                break;
            case Op::Set:
                if (!node_ok(n.a) || !node_ok(n.b) || !node_ok(n.c)) return fail("bad set operand");
                break;
            case Op::SetMany:
                if (!node_ok(n.a)) return fail("bad set_many container");
                if (n.c == 0 || !kids_ok(n.b, uint64_t(n.c) * 2)) return fail("bad set_many pairs");
                // The interpreter reads each key without forcing it.
                for (uint32_t k = 0; k < n.c; ++k) {
                    Op key = Op(nodes_[kids_[n.b + 2 * k]].op);
                    if (key != Op::ConstInt && key != Op::ConstAtom) {
                        return fail("set_many key is not an int or atom constant");
                    }
                }
                break;
            default: break;
        }
    }

    // Acyclicity. An execution tree that points back at itself is malformed,
    // and nothing above would notice: every index is in range. The interpreter
    // would merely spin, but the disassembler walks the tree recursively, so a
    // cycle in an untrusted image would overflow the stack.
    {
        enum Mark : uint8_t { White = 0, Grey = 1, Black = 2 };
        std::vector<uint8_t> mark(n_nodes_, White);
        std::vector<uint32_t> kids_of;
        for (uint32_t root = 0; root < n_nodes_; ++root) {
            if (mark[root] != White) continue;
            // Enter/leave events enumerate each node's edges only once. Rebuilding
            // the child vector for every edge made wide lists quadratic to load.
            std::vector<std::pair<uint32_t, bool>> stack;
            stack.push_back({root, false});
            while (!stack.empty()) {
                auto [ni, leaving] = stack.back();
                stack.pop_back();
                if (ni == NO_NODE) continue;
                if (leaving) { mark[ni] = Black; continue; }
                if (mark[ni] == Grey) return fail("the execution tree contains a cycle");
                if (mark[ni] == Black) continue;
                mark[ni] = Grey;
                stack.push_back({ni, true});
                kids_of.clear();
                node_children(nodes_[ni], kids_, kids_of);
                for (auto it = kids_of.rbegin(); it != kids_of.rend(); ++it) {
                    stack.push_back({*it, false});
                }
            }
        }
    }

    // Slot pass: `local` and `capture` indices are only meaningful against the
    // function whose body reaches them, so walk each body. A generation stamp
    // avoids reclearing the visited map, and the explicit worklist means a
    // maliciously deep or cyclic image cannot blow the C++ stack.
    std::vector<uint32_t> seen(n_nodes_, 0);
    std::vector<uint32_t> work;
    for (uint32_t fi = 0; fi < n_funcs_; ++fi) {
        const FuncRec& f = funcs_[fi];
        const uint32_t gen = fi + 1;
        work.clear();
        work.push_back(f.body);
        while (!work.empty()) {
            uint32_t ni = work.back();
            work.pop_back();
            if (ni == NO_NODE || ni >= n_nodes_ || seen[ni] == gen) continue;
            seen[ni] = gen;
            const Node& n = nodes_[ni];
            switch (static_cast<Op>(n.op)) {
                case Op::Local:
                    if (n.a >= f.slots) return fail("local slot is outside the function frame");
                    break;
                case Op::Capture:
                    if (n.a >= f.n_captures) return fail("capture index is outside the capture list");
                    break;
                case Op::Bind:
                    if (n.a >= f.slots) return fail("bind slot is outside the function frame");
                    break;
                case Op::Try:
                    if (n.c >= f.slots) return fail("catch slot is outside the function frame");
                    break;
                default:
                    break;
            }
            // Use the same edges as the structural walk, including Get/Set
            // operands and Get's fallback: all can contain frame accesses.
            node_children(n, kids_, work);
        }
        // Capture descriptors are read against the *parent* frame at closure
        // creation, so they are bounds-checked there rather than here.
    }
    return true;
}

}  // namespace dream
