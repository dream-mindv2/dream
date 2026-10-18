// Value representation.
//
// A Value is one 64-bit word, tagged in its low bits:
//
//   ....1   fixnum          63-bit signed integer, value = (int64)v >> 1
//   ...000  heap pointer    8-byte aligned; 0 is the "no value" sentinel
//   ...010  immediate       subtype in bits 3..7, payload in bits 8..63
//
// Fixnums get the one-bit tag because integer arithmetic is the hot path:
// testing and untagging cost a shift each. Everything else is either an
// immediate (unit, bool, char, atom, nil) or a pointer into the owning
// process's heap.
//
// Heap objects are only ever reachable from the process that allocated them.
// Nothing here is shared between processes -- messages are copied -- so no
// object needs a lock or an atomic field.

#pragma once
#include <bit>

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>

#include "dream/core.h"

namespace dream {

using Value = uint64_t;

// ---------------------------------------------------------------------------
// Tagging
// ---------------------------------------------------------------------------

inline constexpr uint64_t TAG_MASK = 7;
inline constexpr uint64_t TAG_PTR = 0;
inline constexpr uint64_t TAG_IMM = 2;

enum ImmKind : uint64_t {
    IMM_UNIT = 0,
    IMM_BOOL = 1,
    IMM_CHAR = 2,
    IMM_ATOM = 3,
    IMM_NIL = 4,   // the empty list
    IMM_BUILTIN = 5,  // payload is the builtin id from the image
};

/// The sentinel for "no value here". Never a legal Dream value.
inline constexpr Value NIL_SLOT = 0;

inline constexpr bool is_fixnum(Value v) { return (v & 1) != 0; }
inline constexpr int64_t fixnum_value(Value v) { return static_cast<int64_t>(v) >> 1; }
inline constexpr Value make_fixnum(int64_t n) {
    return (static_cast<uint64_t>(n) << 1) | 1;
}
/// Integers outside 63 bits must be boxed; the compiler emits i64 constants.
inline constexpr bool fixnum_fits(int64_t n) {
    return n >= -(int64_t(1) << 62) && n < (int64_t(1) << 62);
}

inline constexpr bool is_ptr(Value v) { return v != 0 && (v & TAG_MASK) == TAG_PTR; }
inline constexpr bool is_imm(Value v) { return (v & TAG_MASK) == TAG_IMM; }
inline constexpr uint64_t imm_kind(Value v) { return (v >> 3) & 0x1F; }
inline constexpr uint64_t imm_payload(Value v) { return v >> 8; }
inline constexpr Value make_imm(ImmKind kind, uint64_t payload) {
    return (payload << 8) | (static_cast<uint64_t>(kind) << 3) | TAG_IMM;
}

inline constexpr Value UNIT = make_imm(IMM_UNIT, 0);
inline constexpr Value NIL = make_imm(IMM_NIL, 0);
inline constexpr Value TRUE_V = make_imm(IMM_BOOL, 1);
inline constexpr Value FALSE_V = make_imm(IMM_BOOL, 0);

inline constexpr Value make_bool(bool b) { return b ? TRUE_V : FALSE_V; }
inline constexpr Value make_char(uint32_t cp) { return make_imm(IMM_CHAR, cp); }
inline constexpr Value make_atom(uint32_t idx) { return make_imm(IMM_ATOM, idx); }
inline constexpr Value make_builtin(uint32_t id) { return make_imm(IMM_BUILTIN, id); }

inline constexpr bool is_imm_kind(Value v, ImmKind k) {
    return is_imm(v) && imm_kind(v) == static_cast<uint64_t>(k);
}
inline constexpr bool is_bool(Value v) { return is_imm_kind(v, IMM_BOOL); }
inline constexpr bool is_unit(Value v) { return is_imm_kind(v, IMM_UNIT); }
inline constexpr bool is_char(Value v) { return is_imm_kind(v, IMM_CHAR); }
inline constexpr bool is_atom(Value v) { return is_imm_kind(v, IMM_ATOM); }
inline constexpr bool is_nil(Value v) { return is_imm_kind(v, IMM_NIL); }
inline constexpr bool is_builtin(Value v) { return is_imm_kind(v, IMM_BUILTIN); }
inline constexpr bool truthy(Value v) { return v == TRUE_V; }

// ---------------------------------------------------------------------------
// Heap objects
// ---------------------------------------------------------------------------

enum class ObjType : uint8_t {
    Float = 0,
    Str,
    BigStr,    // a view into the image's payload; see BigStrObj
    Cons,      // lazy list cell
    Array,
    Map,       // a branch of the hash trie; every map value is one of these
    MapLeaf,   // one entry of a map, or a chain of them sharing a hash
    Closure,
    Thunk,     // an unevaluated (node, frame) pair
    Blackhole, // a thunk currently being forced
    Indirect,  // an updated thunk, pointing at its result
    Pap,       // partial application: too few arguments so far
    Frame,     // an activation: slots for parameters and block bindings
    Module,
    ErrorBox,
    Pid,
    Native,    // a host function registered through the C API
    Tensor,    // packed numbers with a shape; see TensorObj
    BigInt,    // an integer too large for a fixnum; see BigIntObj
    Record,    // a value of a declared record or union; see RecordObj
    Trunk,      // a version of a trunk; see TrunkObj
    TrunkStore, // the slots and indexes a trunk's versions share
    Count
};

const char* obj_type_name(ObjType t);

/// Every heap object starts with this. `bytes` is the whole allocation,
/// which is what lets the collector walk and copy without a type switch.
struct Obj {
    ObjType type;
    uint8_t gc;   // scratch for the collector: mark bit / forwarded flag
    uint16_t aux;  // flags that survive collection; see AUX_DEEP_FORCED
    uint32_t bytes;
};

/// Everything under this object has been forced.
///
/// A value never changes once it is built, so this is true forever once it is
/// true at all -- which is what lets a deep force skip a structure it has
/// already walked. Without it, forcing walks *paths* rather than data, and a
/// value shared n levels deep is walked 2^n times: the compiler's own tables,
/// which share a syntax tree between every table that mentions it, took seconds
/// to walk and no time at all to build.
///
/// It rides in `aux`, which the collector copies with the rest of the header,
/// so it survives a collection. A host that reaches in and replaces a field of
/// an object it did not just make must clear it.
constexpr uint16_t AUX_DEEP_FORCED = 1;

/// The rest of `aux`: an object's identity hash, or zero when nothing has asked
/// for one yet.
///
/// A map compares a list, an array, a closure -- anything but the flat kinds --
/// by identity, and hashed it by its address. An address is not an identity:
/// promotion copies every young object somewhere else, so a list used as a key
/// was found before a minor collection and missing after it, and the copy a
/// message makes had the same problem across processes. So the identity is
/// kept in the header instead, which travels with every copy the collector
/// makes and which `copy_between` carries across. Fifteen bits is not unique,
/// and need not be: equality is still by identity, and a collision costs a
/// longer leaf chain, not a wrong answer. See `identity_hash`.
constexpr uint16_t AUX_IDENTITY = 0xfffe;
static_assert(sizeof(Obj) == 8, "object headers must stay one word");

inline Obj* as_obj(Value v) { return reinterpret_cast<Obj*>(v); }
inline Value from_obj(const Obj* o) { return reinterpret_cast<Value>(o); }
inline bool is_obj(Value v, ObjType t) { return is_ptr(v) && as_obj(v)->type == t; }

struct FloatObj : Obj {
    double value;
};

struct StrObj : Obj {
    uint32_t len;
    uint32_t hash;
    char* data() { return reinterpret_cast<char*>(this + 1); }
    const char* data() const { return reinterpret_cast<const char*>(this + 1); }
};

/// A window onto the program image's payload region.
///
/// A `StrObj` counts its bytes in a `uint32_t` and carries them in its own
/// allocation, which caps a string at 4 GiB and costs a copy at every
/// materialization, every promotion and every `spawn!`. Neither is acceptable
/// for the datum a program is *about* -- a corpus, a model, an asset -- so a
/// big string is not a string at all. It is a length and a pointer into the
/// mapped image, and building one is O(1).
///
/// Sharing the pointer is safe precisely because it does not point into a
/// heap: the bytes belong to the `Runtime`'s image, which is immutable and
/// outlives every process in it. A copy between heaps therefore copies the
/// view, not the bytes. What it may *not* cross is a runtime boundary
/// (`std.vm`), where the pointer would name someone else's mapping.
///
/// `hash` is 0 until something asks, because a map key is the only thing that
/// needs it and walking a gigabyte to answer a question nobody asked is the
/// cost this type exists to refuse.
struct BigStrObj : Obj {
    uint64_t len;
    uint64_t hash;
    const char* data;
};

/// A cons cell. Both fields may be thunks: `[expensive_a, expensive_b]`
/// allocates two thunks and forces neither.
struct ConsObj : Obj {
    Value head;
    Value tail;
};

struct ArrayObj : Obj {
    uint32_t len;
    uint32_t pad;
    Value* items() { return reinterpret_cast<Value*>(this + 1); }
};

/// A value of a declared `group`, `struct` or `union`: its fields, lazy as an
/// array's items are, and the one thing an array does not have -- what it is.
///
/// What it is, is an atom naming the declaration, qualified by the module it
/// was declared in so that two packages' `union Item` are two types:
/// `ship.item.Item` for a record, and `ship.item.Item.file` for a union's
/// variant. Everything a program can ask of a record's kind -- `type_of`, how
/// it prints, what a constructor pattern tests -- is read from that one atom.
/// Two records are equal when their ids are and their fields are; a record and
/// the list or array it used to be are never equal, which is the point of
/// having it.
///
/// The atom is kept as its index, not as a `Value`, in the word beside the
/// length, with `RECORD_VARIANT` in the top bit -- the same packing a
/// `RecordMake` node's `c` operand uses. Holding the whole `Value` cost a word
/// per record: a two-field record was forty bytes against an array's thirty-two,
/// and a record-heavy loop allocated a fifth more for it. Packed, the header is
/// an array's, sixteen bytes, and the fields start where an array's items do.
///
/// Fields are read and changed by position through `get` and `set`, the same
/// opcodes an array's are, so a record's generated accessors are exactly what
/// they were when a record was an array. See "Records are values of their own"
/// in dreams/docs/notes/static-types.md.
struct RecordObj : Obj {
    uint32_t len;
    uint32_t tag;
    Value id() const { return make_atom(tag & ~(1u << 31)); }
    bool variant() const { return (tag >> 31) != 0; }
    Value* items() { return reinterpret_cast<Value*>(this + 1); }
};

static_assert(sizeof(ArrayObj) == 16 && sizeof(RecordObj) == 16,
              "compiled code reads an array's items and a record's fields 16 bytes "
              "in (`Emitter::get` in jit.cpp), each after a length at 8");

/// `RecordObj::tag`'s top bit: the id names a union's variant, `Type.tag`,
/// rather than a record type. Equal to `RECORD_MAKE_VARIANT`, so a node's
/// operand is a header word as it stands.
inline constexpr uint32_t RECORD_VARIANT = 1u << 31;

/// The header word for the atom `id`, a union's variant when `variant`.
inline uint32_t record_tag(Value id, bool variant) {
    return uint32_t(imm_payload(id)) | (variant ? RECORD_VARIANT : 0);
}

/// A trunk: `std.trunk`'s one container, whose meaning is always an
/// immutable sequence and whose representation the VM chooses from what the
/// program does with it. See "Trunks" in docs/notes/trunks.md.
///
/// Every version of a trunk made from another shares one store of slots
/// with it, addressed by positions that never change -- they start at
/// `TRUNK_ORIGIN` and grow both ways, so a push and a prepend each write the
/// next free slot. Exactly one version per store is its root and reads the
/// slots directly. Any other was the root once: when the root writes a slot,
/// or a newer version is made from it, it becomes a version "like `link`,
/// except that slot `pos` holds `value`", which is Baker's version tree, the
/// scheme behind diff arrays. So linear use -- each version used once to make
/// the next, which is most functional code -- writes in place, and an old
/// version still reads exactly what it held, through its chain. A write from a
/// version that is not the root copies, as a persistent structure must.
struct TrunkObj : Obj {
    uint32_t start;
    uint32_t end;
    /// The store, for the root; the next newer version, for any other.
    Value link;
    /// A version that is not the root: the one slot it holds a value of its
    /// own for, or `TRUNK_NO_POS`.
    uint32_t pos;
    uint32_t pad;
    Value value;
};

/// What `TrunkObj::pos` is when a version differs from the next only in its
/// bounds.
inline constexpr uint32_t TRUNK_NO_POS = 0xffffffffu;
/// Where the first element of a new trunk goes: the middle of the
/// position space, so there is as much room to prepend as to push.
inline constexpr uint32_t TRUNK_ORIGIN = 1u << 31;

/// The slots a trunk's versions share, and what is known about them.
///
/// Slot `i` is `slots()[i - base]`; `[lo, hi)` is every slot ever written.
/// `index` maps each flat element to the positions holding it, newest first,
/// for `contains`, `index_of` and `count`; `keyed` maps the flat first item
/// of each element that is a list -- an association list's key -- to its
/// positions, for `lookup` and `put`. Each covers only `[ilo, ihi)` (and
/// `[klo, khi)`), positions whose elements a scan had already forced, so a
/// search answered from an index forces nothing the scan as written would
/// not have. They are made once scanning has cost as much as the trunk
/// is long (`work`), and are the root's: they describe the slots as they are.
struct TrunkStoreObj : Obj {
    uint32_t base;
    uint32_t cap;
    uint32_t lo;
    uint32_t hi;
    uint32_t ilo;
    uint32_t ihi;
    uint32_t klo;
    uint32_t khi;
    uint64_t work;
    /// The collection count it was last put in its heap's remembered set
    /// after (`Heap::remember_once`).
    uint64_t remembered;
    Value index;
    Value keyed;
    Value* slots() { return reinterpret_cast<Value*>(this + 1); }
};

/// The element at position `i` of version `o`: the first value its chain
/// holds for `i` on the way to the root, else the root's slot. Changes
/// nothing, so it is safe on an object of another heap or of the shared area.
inline Value trunk_read(Obj* o, uint32_t i) {
    auto* c = static_cast<TrunkObj*>(o);
    for (;;) {
        Obj* link = reinterpret_cast<Obj*>(c->link);
        if (link->type == ObjType::TrunkStore) {
            auto* st = static_cast<TrunkStoreObj*>(link);
            return st->slots()[i - st->base];
        }
        if (c->pos == i) return c->value;
        c = static_cast<TrunkObj*>(link);
    }
}

/// A persistent map: a hash array mapped trie. Keys are forced (we have to hash
/// them); values stay lazy.
///
/// This was an open-addressed table, which is the right shape for a language
/// that mutates and the wrong one for a language that does not. Every `map_put`
/// had to copy the whole table to leave the original standing, so building a
/// map an entry at a time -- the only way to build one here -- cost O(n^2) time
/// and allocated O(n^2) bytes. A program accumulating ten thousand entries did
/// fifty million copies to do it.
///
/// A trie shares structure instead. Putting rebuilds only the path from the
/// root to the leaf that changed, about log32(n) nodes, and every other node is
/// shared with the map it came from. Both maps stay valid, which is what a
/// persistent value has to promise, and neither pays for the other.
///
/// A branch holds up to 32 children chosen by five bits of the key's hash.
/// `bitmap` says which of the 32 are present and the children are packed
/// against each other, so a sparse node costs only what it actually holds.
/// The root of a map is always a branch, so that every map value has the same
/// type however few entries it has.
struct MapObj : Obj {
    /// Entries in this whole subtree. The root's is the map's size, which is
    /// what makes `len` on a map constant time.
    uint32_t count;
    uint32_t bitmap;
    Value* slots() { return reinterpret_cast<Value*>(this + 1); }
};

/// One entry, or a chain of them when two keys hash to the same 64 bits.
///
/// Keys with different hashes always separate somewhere in the trie, so a chain
/// is only ever a real collision rather than a probe sequence -- which is why
/// there is no load factor here and nothing to rehash.
struct MapLeafObj : Obj {
    uint64_t hash;
    Value key;
    Value value;
    Value next;  // NIL, or another leaf with the same hash
};

/// Five bits of hash per level: 32 children, and a 64-bit hash runs out after
/// thirteen of them, by which point any two distinct hashes have separated.
constexpr uint32_t MAP_BITS = 5;
constexpr uint32_t MAP_WIDTH = 1u << MAP_BITS;
constexpr uint32_t MAP_MASK = MAP_WIDTH - 1;

inline uint32_t map_bit_count(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return uint32_t(std::popcount(x));
#else
    uint32_t n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
#endif
}

/// Which child a hash selects at this depth, and where that child sits in the
/// packed array.
inline uint32_t map_index(uint64_t hash, uint32_t shift) {
    return uint32_t(hash >> shift) & MAP_MASK;
}
inline uint32_t map_slot_of(uint32_t bitmap, uint32_t bit) {
    return map_bit_count(bitmap & (bit - 1));
}

struct ClosureObj : Obj {
    uint32_t func;
    uint32_t ncaps;
    Value* caps() { return reinterpret_cast<Value*>(this + 1); }
};

/// A suspension. Forcing overwrites it in place with an Indirect, so every
/// sharer of the pointer sees the result -- this is what makes laziness
/// share work rather than repeat it.
struct ThunkObj : Obj {
    uint32_t node;
    uint32_t pad;
    Value frame;
};

struct IndirectObj : Obj {
    Value target;
};
static_assert(sizeof(IndirectObj) <= sizeof(ThunkObj),
              "a forced thunk is overwritten in place, so it must not grow");

struct PapObj : Obj {
    Value fn;
    uint32_t nargs;
    uint32_t pad;
    Value* args() { return reinterpret_cast<Value*>(this + 1); }
};

struct FrameObj : Obj {
    Value closure;
    uint32_t nslots;
    uint32_t pad;
    Value* slots() { return reinterpret_cast<Value*>(this + 1); }
};

struct ModuleObj : Obj {
    uint32_t import_index;
    uint32_t pad;
    Value name;  // StrObj
};

struct ErrorObj : Obj {
    Value kind;     // atom
    Value payload;  // arbitrary value
    /// Where it was first raised, or `UNIT` until it has been: a list of
    /// fixnums the machine fills in as it starts to unwind (`locate_error` in
    /// interp.cpp). It is not part of the value -- equality, hashing and
    /// `to_string` never look at it -- and it means something only to the
    /// image that raised it, so crossing to another image leaves it behind.
    Value where;
};

struct PidObj : Obj {
    uint64_t id;
};

/// An integer outside the fixnum range: a sign and a magnitude in 64-bit
/// limbs, least significant first.
///
/// It is the slow half of `:integer` and nothing else -- `type_of` says
/// `:integer`, and arithmetic reaches it only where a fixnum operation has
/// already overflowed, which is the path that used to answer a float. So a
/// program that never leaves 63 bits never meets one, and every fixnum fast
/// path, in the interpreter and in compiled code, is exactly what it was.
///
/// **Canonical, always.** A `BigIntObj` never holds a value a fixnum could:
/// every operation that makes one trims its top limbs and answers a fixnum when
/// the result fits (`bigint::finish`). That is what lets equality, hashing and
/// `match` treat the two representations as disjoint -- a fixnum is never
/// equal to a bignum -- rather than comparing across them at every test.
///
/// `len` may be less than the allocation holds: an operation allocates for the
/// largest result it could have and trims, and the collector reads the size
/// from the header, not from here.
///
/// **Where the limbs are.** Inline, after these fields, while the whole object
/// fits the heap's largest size class; past that, in memory of their own, the
/// way a GPU tensor's numbers are (`Heap::alloc_bigint`). A heap object that
/// large would be a *big object* -- tenured at birth, a block to itself, and
/// freed only by a major collection -- and a loop of bignums makes one per
/// step, nearly all dead by the next. Held outside, the object is a small
/// young header like any other, and the first minor collection that finds it
/// unreached frees its limbs (`reap_external`), as it frees everything else
/// the loop made. `limbs()` is the one place that knows which.
struct BigIntObj : Obj {
    uint32_t len;      // limbs in use; the top one is nonzero
    uint8_t neg;       // 1 when negative
    uint8_t external;  // 1 when the limbs live outside the heap
    uint16_t pad;
    uint64_t* limbs() {
        return external ? external_limbs() : reinterpret_cast<uint64_t*>(this + 1);
    }
    const uint64_t* limbs() const {
        return external ? *reinterpret_cast<uint64_t* const*>(this + 1)
                        : reinterpret_cast<const uint64_t*>(this + 1);
    }
    // An external bignum's payload: the pointer, then how many limbs it holds.
    uint64_t*& external_limbs() { return *reinterpret_cast<uint64_t**>(this + 1); }
    uint64_t& external_capacity() { return reinterpret_cast<uint64_t*>(this + 1)[1]; }
};

/// A host function: a builtin, a module member, or something registered
/// through the C API. `strict_mask` bit i set means argument i is forced
/// before the call -- `spawn!` clears bit 0, because forcing the thunk it is
/// handed would run the new process's work on the caller.
struct NativeObj : Obj {
    void* fn;
    Value name;
    uint32_t arity;
    uint32_t strict_mask;
    /// Host state the function is bound to -- an FFI binding id, say. A native
    /// receives its own object, so this is how one C function can back many
    /// distinct Dream functions.
    uint64_t user;
};

/// A tensor: numbers packed flat, with a shape. The value the numeric
/// operators (`+ - * / %` elementwise, `@` for products) and `std.tensor` work
/// on, and the reason they can be fast at all.
///
/// An array of floats is an array of *pointers* to float boxes: every element
/// is a separate object, sixteen bytes and a cache miss away from its
/// neighbour, and every arithmetic result is another allocation. Nothing a
/// vector unit does well can be done to that. This holds the doubles
/// themselves, contiguous and row-major, so a kernel reads them the way a C
/// program would and the compiler can vectorize the loop.
///
/// It is a value like any other -- immutable once built, copied between heaps
/// when it is sent -- and holds no references, so the collector treats it as a
/// leaf (a string, to it). One past `kMaxClassSize` is born in old space and
/// never moved, which is what keeps a large one from being copied by every
/// minor collection.
///
/// `dtype` and `device` are the two doors left open. Only `F64` on the host
/// exists today; a `F32` buffer, or one that lives in a GPU's memory with
/// `data` replaced by a handle, would be told apart here. See
/// docs/notes/tensors.md.
///
/// The shape is held inline, up to `TENSOR_MAX_RANK` axes, so that the data
/// starts at a fixed offset. `dims` past `rank` are zero.
constexpr uint32_t TENSOR_MAX_RANK = 6;
enum TensorDtype : uint8_t { TENSOR_F64 = 0, TENSOR_F32 = 1 };
enum TensorDevice : uint8_t { TENSOR_HOST = 0 };

struct TensorObj : Obj {
    uint8_t dtype;
    uint8_t rank;
    uint8_t device;
    uint8_t flags;
    uint32_t pad;
    uint64_t count;  // the product of the dims
    uint32_t dims[TENSOR_MAX_RANK];
    double* data() { return reinterpret_cast<double*>(this + 1); }
    const double* data() const { return reinterpret_cast<const double*>(this + 1); }
};
static_assert(sizeof(TensorObj) % 16 == 0, "a tensor's data starts on a 16-byte boundary");

/// A tensor that has not been computed yet: elementwise work, recorded.
///
/// `a * 2.0 + b` on large tensors used to make two passes over memory and one
/// temporary as large as the answer; `relu (w @ x + b)` three and two. None of
/// that work is visible to the program until something reads the numbers, so
/// the elementwise operators and functions record what they would have done
/// instead -- a small stack program over the tensors they were given -- and
/// whatever finally needs the numbers (a product, an index, printing, a move
/// to or from the GPU, `strict!`) runs the whole chain in one pass. That is
/// fusion; docs/notes/tensors.md, "Fusion", has the measurements.
///
/// A deferred tensor is a `TensorObj` with `TENSOR_DEFERRED` set: the same type
/// and the same shape as the tensor it stands for, so `type_of`, `len` and
/// `tensor.shape` never compute anything. Its payload is a `TensorExpr`
/// instead of numbers, and unlike a computed tensor it holds references --
/// the inputs, and the answer once there is one -- which the collector traces.
/// Computing it stores the answer in `result` and lets the inputs go; every
/// later read finds the answer there.
///
/// The inputs are always computed tensors. Building `x op y` from a deferred
/// `x` copies `x`'s program into the new one rather than pointing at `x`, so
/// a chain is one flat program and never a tree of deferred objects.
constexpr uint8_t TENSOR_DEFERRED = 1;

///
/// A program may also begin from a matrix product (`FUSE_PRODUCT`): `a @ b` on
/// large operands defers too, holding its two operands in `product_a` and
/// `product_b`, so that `relu (w @ x + b)` computes the elementwise part as
/// each band of the product is finished -- in cache, or in the product kernel
/// itself on a GPU -- rather than in a pass over the whole product afterwards.
/// Both are `NIL_SLOT` when the program has no product.
struct TensorExpr {
    Value result;  // NIL_SLOT until computed
    Value product_a;
    Value product_b;
    uint16_t ninputs;
    uint16_t nconsts;
    uint16_t ncode;  // instructions, two bytes each
    uint16_t depth;  // the stack the program needs
    Value* inputs() { return reinterpret_cast<Value*>(this + 1); }
    double* consts() { return reinterpret_cast<double*>(inputs() + ninputs); }
    uint8_t* code() { return reinterpret_cast<uint8_t*>(consts() + nconsts); }
    /// What a payload of this shape occupies, after the tensor's header.
    static size_t bytes_for(size_t ninputs, size_t nconsts, size_t ncode) {
        return sizeof(TensorExpr) + ninputs * sizeof(Value) + nconsts * sizeof(double) +
               ((ncode * 2 + 7) & ~size_t(7));
    }
};

inline bool tensor_deferred(const TensorObj* t) { return (t->flags & TENSOR_DEFERRED) != 0; }
inline TensorExpr* tensor_expr(TensorObj* t) { return reinterpret_cast<TensorExpr*>(t->data()); }

/// The bytes a tensor's numbers occupy wherever they live.
inline size_t tensor_data_bytes(const TensorObj* t) {
    return size_t(t->count) * (t->dtype == TENSOR_F32 ? sizeof(float) : sizeof(double));
}

// ---------------------------------------------------------------------------
// Following indirections
// ---------------------------------------------------------------------------

/// Chase Indirect nodes left behind by thunk updates. Every read of a value
/// that might have been a thunk goes through this.
inline Value resolve(Value v) {
    while (is_ptr(v) && as_obj(v)->type == ObjType::Indirect) {
        v = static_cast<IndirectObj*>(as_obj(v))->target;
    }
    return v;
}

/// True when the value is already in weak head normal form.
inline bool is_whnf(Value v) {
    if (!is_ptr(v)) return true;
    switch (as_obj(v)->type) {
        case ObjType::Thunk:
        case ObjType::Blackhole:
        case ObjType::Indirect:
            return false;
        default:
            return true;
    }
}

// ---------------------------------------------------------------------------
// The few writes that escape a single thread
// ---------------------------------------------------------------------------

/// The mutator only rewrites a live object's header byte or a slot in a couple
/// of places -- a thunk flipped to a blackhole and back, a frame slot bound, a
/// forced cell's fields -- but during a concurrent mark a helper thread can be
/// reading that same byte at the same moment. The reads the helpers make are
/// already atomics (see `read_slot`/`read_gc` in heap.cpp); these are the
/// writes they need to say the same thing, so that the pair is not a data race
/// and a scan never sees half of an update.
///
/// Plain reads stay plain everywhere: a reader that is not the single writer is
/// fine, the racing is between a helper's atomic read and this write. `relaxed`
/// is enough for a header flip or a slot store -- the slot only needs to be one
/// whole value one way or the other. The one exception is the thunk update,
/// which publishes an `Indirect` after storing its target; that uses
/// `obj_type_store_release` and has to put the target in first (see
/// `step_return` in interp.cpp).
inline void obj_type_store(Obj* o, ObjType t) {
    std::atomic_ref<uint8_t>(reinterpret_cast<uint8_t&>(o->type))
        .store(static_cast<uint8_t>(t), std::memory_order_relaxed);
}
inline void obj_type_store_release(Obj* o, ObjType t) {
    std::atomic_ref<uint8_t>(reinterpret_cast<uint8_t&>(o->type))
        .store(static_cast<uint8_t>(t), std::memory_order_release);
}
inline void value_slot_store(Value* slot, Value v) {
    std::atomic_ref<Value>(*slot).store(v, std::memory_order_relaxed);
}
/// A plain read of an object's type. Every read can be plain -- the writes
/// that need atomics are the ones above, and a read racing with nothing (there
/// is only one writer of a given object's type) needs no saying either way.
inline ObjType obj_type(Obj* o) { return o->type; }

// ---------------------------------------------------------------------------
// Strings, in either representation
// ---------------------------------------------------------------------------

/// The bytes behind a string value, and how many of them there are.
///
/// A `Str` carries its bytes in its own allocation; a `BigStr` points at the
/// image's payload. Every operation that only *reads* a string -- compare,
/// hash, index, slice, write -- wants the pointer and the length and has no
/// business knowing which of the two it was handed, so it asks here once
/// instead of switching on the type at each site. The length is 64-bit
/// because one of the two kinds can be.
struct Bytes {
    const char* data = nullptr;
    uint64_t len = 0;
};

/// True when the value is a string of either representation.
inline bool is_stringish(Value v) {
    v = resolve(v);
    if (!is_ptr(v)) return false;
    ObjType t = as_obj(v)->type;
    return t == ObjType::Str || t == ObjType::BigStr;
}

/// The bytes of a string value, or false when it is not one.
inline bool string_bytes(Value v, Bytes* out) {
    v = resolve(v);
    if (!is_ptr(v)) return false;
    Obj* o = as_obj(v);
    if (o->type == ObjType::Str) {
        auto* s = static_cast<StrObj*>(o);
        *out = Bytes{s->data(), s->len};
        return true;
    }
    if (o->type == ObjType::BigStr) {
        auto* b = static_cast<BigStrObj*>(o);
        *out = Bytes{b->data, b->len};
        return true;
    }
    return false;
}

/// Lexicographic byte order, shorter first on a shared prefix. -1, 0 or 1.
inline int bytes_compare(const Bytes& a, const Bytes& b) {
    uint64_t n = a.len < b.len ? a.len : b.len;
    int c = n ? std::memcmp(a.data, b.data, size_t(n)) : 0;
    if (c != 0) return c < 0 ? -1 : 1;
    return a.len < b.len ? -1 : (a.len > b.len ? 1 : 0);
}

inline bool bytes_equal(const Bytes& a, const Bytes& b) {
    return a.len == b.len && (a.len == 0 || std::memcmp(a.data, b.data, size_t(a.len)) == 0);
}

/// FNV-1a over the bytes, never zero -- zero is what a cached hash uses to mean
/// "not worked out yet". Both string representations hash through this, so a
/// `BigStr` slice and the literal it matches land in the same map bucket.
inline uint64_t bytes_hash(const Bytes& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < b.len; ++i) {
        h ^= uint8_t(b.data[i]);
        h *= 1099511628211ull;
    }
    return h ? h : 1;
}

dream_type surface_type(Value v);

}  // namespace dream
