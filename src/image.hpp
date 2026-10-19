// Loader for `.dream` bytecode images.
//
// The container was designed so that loading is a slice, not a parse: every
// section is an array of fixed-size little-endian records and every edge is an
// index. So we mmap the file and point at it. The only work done up front is
// validation -- an image may come from anywhere, and the interpreter indexes
// these arrays without further checks, so every index has to be proven in
// range exactly once, here.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include "primitive_ops.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace dream {

// Must match the compiler's `ir::Op` in dreams/src/ir.dr.
// See docs/bytecode-format.md for the container.
enum class Op : uint8_t {
    Nop = 0,
    ConstInt = 1,
    ConstFloat = 2,
    ConstStr = 3,
    ConstChar = 4,
    ConstBool = 5,
    ConstAtom = 6,
    Unit = 7,
    Local = 8,
    Capture = 9,
    Global = 10,
    Builtin = 11,
    Field = 12,
    Apply = 13,
    If = 14,
    Block = 15,
    Bind = 16,
    MakeClosure = 17,
    MakeThunk = 18,
    Try = 19,
    Force = 20,
    Add = 21, Sub = 22, Mul = 23, Div = 24, Mod = 25,
    Eq = 26, Ne = 27, Lt = 28, Le = 29, Gt = 30, Ge = 31,
    And = 32, Or = 33,
    Neg = 34, Not = 35,
    MakeList = 36, MakeArray = 37, MakeMap = 38,
    // Reading and changing a container: a map by key, an array or a list by
    // position. Opcodes rather than natives so the element handed back is
    // forced by the machine, not underneath a native call.
    Get = 39, Set = 40,
    // A `match` jumping straight to the arms that can match, by an atom: the
    // head of a list cell (`[:tag, ..]`), or the value itself (`:tag`). `a` is
    // the subject, `b` a kids offset and `c` a kids count: pairs of an atom
    // constant node and a target, then the default target. Every target is a
    // suffix of the ordinary chain of arm tests, so the tests still run -- the
    // switch only skips the arms that cannot match. See `lower_match` in
    // dreams/src/lower.dr.
    SwitchHead = 41, SwitchAtom = 42,
    // Force a, test its surface type b (DREAM_TYPE_INTEGER..DREAM_TYPE_MAP).
    // c is 0 for equality, 1 for inequality. No atom or native call is needed.
    TypeIs = 43,
    Cons = 44, ListTail = 45, ListIsEmpty = 46,
#define DREAM_PRIMITIVE_ENUM(name, code, builtin, arity) name = code,
    DREAM_PRIMITIVES(DREAM_PRIMITIVE_ENUM)
#undef DREAM_PRIMITIVE_ENUM
    // The bitwise operators, on integers: `a & b`, `a | b`, `a ^ b`, `~a`,
    // `a << b` and `a >> b`. Binary but for `BitNot`, whose operand is `a`.
    BitAnd = 75, BitOr = 76, BitXor = 77, BitNot = 78, Shl = 79, Shr = 80,
    // A value of a declared record or union (`RecordObj`): `a` is a kids
    // offset and `b` a count, the fields, each in a lazy position as an array
    // literal's items are; `c` is the atom naming what it is, with
    // `RECORD_MAKE_VARIANT` set when that is a union's variant.
    RecordMake = 81,
    // Several changes to one container at once: `a` is the container, `b` a
    // kids offset and `c` a count of pairs, the kids `k0, v0, k1, v1, ..`.
    // Every key is an int or atom constant, and each value is stored unforced,
    // as `Set` stores one. It answers what the chain of `Set`s it replaces
    // would -- a later key winning, the same error at the same key -- and
    // copies a record or an array once rather than once per key: what a
    // pipeline of setters (`p |> P.set_a 1 |> P.set_b 2`) is lowered to.
    SetMany = 82,
    Ensure = 83,  // a = body, b = cleanup; run cleanup on every exit
    Count
};

// Primitive nodes carry their builtin ID in a, and their argument run in b/c.
// Unlike Apply, they do not evaluate or resolve a callee.
inline constexpr uint32_t primitive_builtin(Op op) {
    switch (op) {
#define DREAM_PRIMITIVE_ID(name, code, builtin, arity) case Op::name: return builtin;
        DREAM_PRIMITIVES(DREAM_PRIMITIVE_ID)
#undef DREAM_PRIMITIVE_ID
        default: return 0xFFFFFFFFu;
    }
}
inline constexpr uint32_t primitive_arity(Op op) {
    switch (op) {
#define DREAM_PRIMITIVE_ARITY(name, code, builtin, arity) case Op::name: return arity;
        DREAM_PRIMITIVES(DREAM_PRIMITIVE_ARITY)
#undef DREAM_PRIMITIVE_ARITY
        default: return 0;
    }
}
inline constexpr bool is_primitive(Op op) { return primitive_builtin(op) != 0xFFFFFFFFu; }

const char* op_name(Op op);

inline constexpr uint32_t NO_NODE = 0xFFFFFFFFu;

/// `RecordMake`'s `c`: the top bit says the id names a variant, the rest is
/// the atom's index.
inline constexpr uint32_t RECORD_MAKE_VARIANT = 0x80000000u;
inline constexpr uint32_t RECORD_MAKE_ATOM = 0x7FFFFFFFu;

// Node flags, set by the compiler.
inline constexpr uint8_t F_STRICT = 1 << 0;
inline constexpr uint8_t F_IMPURE = 1 << 1;
inline constexpr uint8_t F_TAIL = 1 << 2;

// Function record flags.
inline constexpr uint16_t FN_IMPURE = 1 << 0;
inline constexpr uint16_t FN_REC = 1 << 1;
inline constexpr uint16_t FN_THUNK = 1 << 2;
inline constexpr uint16_t FN_GLOBAL_VALUE = 1 << 3;

// Global record flags.
inline constexpr uint32_t G_EXPORTED = 1 << 0;
inline constexpr uint32_t G_IMPURE = 1 << 1;

// Capture descriptors.
inline constexpr uint32_t CAP_FROM_CAPTURE = 0x80000000u;

#pragma pack(push, 1)
struct Node {
    uint8_t op;
    uint8_t flags;
    uint16_t aux;
    uint32_t a;
    uint32_t b;
    uint32_t c;
};
static_assert(sizeof(Node) == 16, "node records are 16 bytes on disk");

struct FuncRec {
    uint32_t name;
    uint32_t body;
    uint16_t arity;
    uint16_t flags;
    uint16_t slots;
    uint16_t n_captures;
    uint32_t captures_off;
    /// Where the function was written: a 1-based line and column, and the
    /// file as a `KSTR` index plus one, so that zero -- what an image written
    /// before these fields existed holds -- means "not known". Read only to
    /// say where an error was raised (`error_trace` in interp.cpp).
    uint32_t line;
    uint32_t col;
    uint32_t source;
};
static_assert(sizeof(FuncRec) == 32, "function records are 32 bytes on disk");

struct GlobalRec {
    uint32_t name;
    uint32_t kind;  // 0 = function, 1 = module
    uint32_t target;
    uint32_t flags;
};
static_assert(sizeof(GlobalRec) == 16, "global records are 16 bytes on disk");

struct ImportRec {
    uint32_t path;
    uint32_t alias;
};

struct StrRec {
    uint32_t offset;
    uint32_t length;
};

struct SpanRec {
    uint32_t start;
    uint32_t end;
};

/// One large datum: where it starts in the payload, and how long it is.
///
/// Both are 64-bit, which is the whole point. Everything else in the container
/// is addressed by `u32`, so the file and every string in it stop at 4 GiB;
/// the payload is the one region allowed past that line, and this is the only
/// record that can name a position inside it.
struct DataRec {
    uint64_t offset;
    uint64_t length;
};
static_assert(sizeof(DataRec) == 16, "large-data records are 16 bytes on disk");

/// What a function's signature declared, for the JIT: which parameters are
/// `:float` (TYPE) or `:integer` (ITYP). A hint and never a promise -- the types are gradual, so a caller
/// with no signature may pass such a parameter anything, and whoever acts on
/// this guards the value first. See "Representation" in jit.cpp.
struct TypeRec {
    uint32_t func;
    uint32_t reserved;
    uint64_t params;
};
static_assert(sizeof(TypeRec) == 16, "type records are 16 bytes on disk");

/// A `VMOP` record: one setting the program's root package asked of the VM
/// in its manifest's `[vm]` table. A key this VM does not know is ignored, so
/// a newer compiler's settings do not stop an older VM from running the image.
struct VmOptionRec {
    uint32_t key;
    uint32_t reserved;
    uint64_t value;
};
static_assert(sizeof(VmOptionRec) == 16, "VM option records are 16 bytes on disk");

/// `VMOP` keys. See `emit.vm_nursery_max` in dreams/src/emit.dr.
inline constexpr uint32_t VMOP_NURSERY_MAX = 1;
inline constexpr uint32_t VMOP_MAX_HEAP = 2;

/// One module of a whole-program image. Compilation is whole-program, so an
/// image carries every module it needs and records which globals belong to
/// which -- for diagnostics, tooling, and reflection.
struct ModuleRec {
    uint32_t name;
    uint32_t source;
    uint32_t globals_start;
    uint32_t globals_count;
    uint32_t flags;
    uint32_t derives;
};
static_assert(sizeof(ModuleRec) == 24, "module records are 24 bytes on disk");
#pragma pack(pop)

/// Provided by the host rather than compiled from Dream source.
inline constexpr uint32_t M_NATIVE = 1 << 0;
/// Declares at least one virtual, so it is meant to be derived.
inline constexpr uint32_t M_ABSTRACT = 1 << 1;

enum GlobalKind : uint32_t { GLOBAL_FUNCTION = 0, GLOBAL_MODULE = 1 };

struct StringRef {
    const char* data;
    uint32_t len;
    std::string str() const { return std::string(data, len); }
    bool equals(const char* s) const;
};

/// The node indices a node points at: what every walk of an arena agrees
/// counts as a child.
void node_children(const Node& n, const uint32_t* kids, std::vector<uint32_t>& out);

class Image {
public:
    struct Accessor {
        uint32_t container_slot = NO_NODE;
        uint32_t key_slot = NO_NODE;
        int64_t key = 0;
    };
    const Accessor& accessor(uint32_t function) const { return accessors_[function]; }
    /// Bit `i` set when the function forces parameter `i` before anything
    /// else it does (`strict_args_` in image.cpp), so a caller may evaluate
    /// that argument instead of suspending it.
    uint32_t strict_args(uint32_t function) const { return strict_args_[function]; }
    /// What a flat suspension of `node` copies out of its frame (see
    /// `ThunkObj`): a count, then that many slots, each the index of a slot the
    /// expression reads, with `FLAT_BOUND` set when the expression also binds
    /// it -- a slot reused by a name inside it, which is not a binding the
    /// suspension waits on. Worked out the first time a suspension of the node
    /// is made, by whichever process gets there, and published with a
    /// compare-and-swap so that every later one is a load.
    const uint32_t* flat_shape(uint32_t node) const {
        const uint32_t* s = flat_shapes_[node].load(std::memory_order_acquire);
        return s ? s : compute_flat_shape(node);
    }
    static constexpr uint32_t FLAT_BOUND = 1u << 31;
    /// The value the image's `VMOP` section gives `key`, or 0 when it gives
    /// none -- every setting there is a size or a count, and 0 is neither.
    uint64_t vm_option(uint32_t key) const {
        for (uint32_t i = 0; i < n_vm_options_; ++i)
            if (vm_options_[i].key == key) return vm_options_[i].value;
        return 0;
    }
    Image() = default;
    ~Image();
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    /// Map a file and validate it. On failure returns false and fills `error`.
    bool load_file(const std::string& path, std::string& error);
    /// Validate an in-memory image. The bytes must outlive the Image.
    bool load_bytes(const uint8_t* data, size_t size, std::string& error);

    uint16_t version_major() const { return version_major_; }
    uint16_t version_minor() const { return version_minor_; }
    bool has_debug_info() const { return (flags_ & 1) != 0; }
    /// Where the image may run: bits 8-15 of the header's flags are a set of
    /// systems, 16-23 a set of architectures, and an empty set is any. See
    /// "The image's target" in docs/bytecode-format.md.
    uint8_t target_os() const { return uint8_t(flags_ >> 8); }
    uint8_t target_arch() const { return uint8_t(flags_ >> 16); }
    uint32_t entry() const { return entry_; }
    /// The image's own bytes, from its header on, payload included: what
    /// `vm.image_digest` reads.
    const uint8_t* bytes() const { return data_; }
    size_t byte_count() const { return size_; }

    StringRef module_name() const { return str(module_name_); }
    StringRef source_name() const { return str(source_name_); }

    int64_t integer(uint32_t i) const { return ints_[i]; }
    double real(uint32_t i) const { return floats_[i]; }
    StringRef str(uint32_t i) const {
        return StringRef{blob_ + strs_[i].offset, strs_[i].length};
    }
    /// Atoms are interned separately from strings so they compare by index.
    StringRef atom_name(uint32_t i) const { return str(atoms_[i]); }
    uint32_t atom_string_index(uint32_t i) const { return atoms_[i]; }

    const Node& node(uint32_t i) const { return nodes_[i]; }
    uint32_t kid(uint32_t i) const { return kids_[i]; }
    const uint32_t* kids_at(uint32_t off) const { return kids_ + off; }
    const FuncRec& func(uint32_t i) const { return funcs_[i]; }
    const GlobalRec& global(uint32_t i) const { return globals_[i]; }
    const ImportRec& import(uint32_t i) const { return imports_[i]; }
    const ModuleRec& module(uint32_t i) const { return modules_[i]; }
    uint32_t module_count() const { return n_modules_; }

    /// Large data: the payload region, addressed by 64-bit numbers.
    ///
    /// A datum is handed out as a pointer into the mapping rather than as
    /// bytes. Copying is what the whole feature exists to avoid: the payload
    /// may be larger than the address space the rest of the format can name,
    /// and the mapping outlives every process that looks at it.
    uint32_t data_count() const { return n_large_; }
    const char* data_bytes(uint32_t i) const { return payload_ + large_[i].offset; }
    uint64_t data_length(uint32_t i) const { return large_[i].length; }
    /// The name `--payload NAME=FILE` gave datum `i`, or empty when the image
    /// has no `LNAM` section. Names are optional and additive: an image built
    /// before them, or by a tool that does not write them, is reached by index.
    std::string_view data_name(uint32_t i) const {
        return i < data_names_.size() ? data_names_[i] : std::string_view();
    }
    /// The index of the datum called `name`, or -1.
    int64_t data_index(std::string_view name) const {
        for (size_t i = 0; i < data_names_.size(); ++i) {
            if (data_names_[i] == name) return int64_t(i);
        }
        return -1;
    }
    /// One bit per parameter the function's signature declares `:float`, or
    /// zero when it declares none or the image has no `TYPE` section.
    uint64_t float_params(uint32_t func) const {
        return func < float_params_.size() ? float_params_[func] : 0;
    }
    /// Integer signature hints from the optional ITYP section.
    uint64_t integer_params(uint32_t func) const {
        return func < integer_params_.size() ? integer_params_[func] : 0;
    }
    /// The module a global belongs to, or -1.
    int module_of_global(uint32_t global_index) const;
    /// Source span for a node, or {0,0} when the image carries no debug info.
    SpanRec span(uint32_t node_index) const;

    uint32_t int_count() const { return n_ints_; }
    uint32_t float_count() const { return n_floats_; }
    uint32_t string_count() const { return n_strs_; }
    uint32_t atom_count() const { return n_atoms_; }
    uint32_t node_count() const { return n_nodes_; }
    uint32_t kid_count() const { return n_kids_; }
    uint32_t func_count() const { return n_funcs_; }
    uint32_t global_count() const { return n_globals_; }
    uint32_t import_count() const { return n_imports_; }

    /// Index of the global with this name, or -1.
    int find_global(const char* name) const;

private:
    const uint32_t* compute_flat_shape(uint32_t node) const;
    mutable std::unique_ptr<std::atomic<const uint32_t*>[]> flat_shapes_;
    std::vector<Accessor> accessors_;
    std::vector<uint32_t> strict_args_;
    /// `TYPE`, spread out by function so that asking costs an index. Empty
    /// when the image has no such section, which is every image built from a
    /// program without a float in a signature.
    std::vector<uint64_t> float_params_;
    std::vector<uint64_t> integer_params_;
    bool parse(std::string& error);
    bool validate(std::string& error);

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool owns_mapping_ = false;
    /// What to hand back to munmap. A shebang line is skipped by advancing
    /// `data_`, which then no longer points at the start of the mapping -- and
    /// munmap takes the address it was given, page-aligned, or nothing.
    void* mapping_ = nullptr;
    size_t mapping_size_ = 0;

    uint16_t version_major_ = 0, version_minor_ = 0;
    uint32_t flags_ = 0, module_name_ = 0, source_name_ = 0, entry_ = NO_NODE;

    const int64_t* ints_ = nullptr;          uint32_t n_ints_ = 0;
    const double* floats_ = nullptr;         uint32_t n_floats_ = 0;
    const StrRec* strs_ = nullptr;           uint32_t n_strs_ = 0;
    const char* blob_ = nullptr;             uint32_t n_blob_ = 0;
    const uint32_t* atoms_ = nullptr;        uint32_t n_atoms_ = 0;
    const Node* nodes_ = nullptr;            uint32_t n_nodes_ = 0;
    const uint32_t* kids_ = nullptr;         uint32_t n_kids_ = 0;
    const FuncRec* funcs_ = nullptr;         uint32_t n_funcs_ = 0;
    const GlobalRec* globals_ = nullptr;     uint32_t n_globals_ = 0;
    const ImportRec* imports_ = nullptr;     uint32_t n_imports_ = 0;
    const SpanRec* spans_ = nullptr;         uint32_t n_spans_ = 0;
    const ModuleRec* modules_ = nullptr;     uint32_t n_modules_ = 0;
    const DataRec* large_ = nullptr;         uint32_t n_large_ = 0;
    const TypeRec* integer_types_ = nullptr; uint32_t n_integer_types_ = 0;
    const TypeRec* types_ = nullptr;         uint32_t n_types_ = 0;
    const VmOptionRec* vm_options_ = nullptr; uint32_t n_vm_options_ = 0;

    /// The payload bytes, and where its `u64` length header sits. The offset
    /// is kept because validation has to prove the payload ends inside the
    /// file, and that is the only sum in the container that can overflow a
    /// `u32`.
    const char* payload_ = nullptr;
    uint64_t payload_len_ = 0;
    size_t payload_off_ = 0;
    bool has_payload_ = false;
    /// `LNAM`, read at load: one name per datum, pointing into the mapping.
    const uint8_t* names_ = nullptr;  uint32_t names_len_ = 0; uint32_t n_names_ = 0;
    std::vector<std::string_view> data_names_;
};

}  // namespace dream
