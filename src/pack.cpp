// The packed wire form: every value `std.wire` writes, each one written once.
//
// `std.wire`'s first form spells a value out whole. A value that shares a
// part -- a syntax tree two tables both hold, a list consed onto another --
// is written once for each path to the part, so the bytes grow with the
// paths rather than with the data, and reading them back makes a copy per
// path. A compiler's state is mostly such sharing, and its unit cache wrote
// and read it that way.
//
// The packed form writes each distinct value once. A first pass finds every
// part that occurs more than once *by content* -- by a 128-bit hash of it,
// each match checked by comparing the two exactly, so a collision costs a
// comparison and never a wrong answer -- and the second writes the first
// occurrence behind a `DEF` and every later one as a `REF` to it. A list is
// a run of cells and then its tail, so a shared tail is shared too. What it
// reads back is shared the same way.
//
// It is canonical, which is what lets a cache key be the digest of it: the
// bytes are a function of the value alone. Where the program happened to
// share, and in what order it built a map, change nothing; which parts are
// defined, and in what order, follows the value's content in a walk the
// value fixes. "The same" is the representation, not `==`: `1` and `1.0`
// are equal and are not the same, and are written apart.
//
// And it is compact: counts, lengths and integers are variable-length, and
// an atom's or a record's name is written once a message and then numbered.
//
// A packed message starts with `PACK_MAGIC`, which no message of the first
// form starts with, so `vm.wire_decode` reads both.

#include "builtins.hpp"
#include "bigint.hpp"
#include "interp.hpp"
#include "process.hpp"
#include "runtime.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace dream {

namespace {

enum : uint8_t {
    P_UNIT = 0, P_FALSE = 1, P_TRUE = 2, P_INT = 3, P_FLOAT = 4, P_STRING = 5,
    P_ATOM = 6, P_CHAR = 7, P_LIST = 8, P_ARRAY = 9, P_MAP = 10, P_BIGINT = 11,
    P_TENSOR = 12, P_RECORD = 13, P_TRUNK = 14,
    // What only the packed form has: a value given the next number, a value
    // by its number, the empty list, and an atom's name the first time.
    P_DEF = 15, P_REF = 16, P_NIL = 17, P_ATOM_NEW = 18,
};

constexpr uint8_t PACK_MAGIC = 0xD2;

/// Strings shorter than this are written where they stand: a reference
/// would be as long.
constexpr uint32_t kShareStringFrom = 4;

constexpr int kMaxDepth = 10000;

struct H128 {
    uint64_t a, b;
    bool operator==(const H128& o) const { return a == o.a && b == o.b; }
};

inline uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

inline H128 start(uint8_t tag) { return H128{mix(0x6A09E667F3BCC908ull + tag), mix(0xBB67AE8584CAA73Bull ^ (uint64_t(tag) << 32))}; }

inline H128 add(H128 h, uint64_t v) {
    h.a = mix(h.a ^ v) + 0x9E3779B97F4A7C15ull;
    h.b = mix(h.b + v * 0xC2B2AE3D27D4EB4Full) ^ (h.a >> 7);
    return h;
}

inline H128 add(H128 h, H128 v) { return add(add(h, v.a), v.b); }

H128 add_bytes(H128 h, const char* data, size_t n) {
    h = add(h, uint64_t(n));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, data + i, 8);
        h = add(h, w);
    }
    uint64_t tail = 0;
    for (size_t j = 0; i + j < n; ++j) tail |= uint64_t(uint8_t(data[i + j])) << (8 * j);
    return add(h, tail);
}

void put_varint(std::string* out, uint64_t v) {
    while (v >= 0x80) {
        out->push_back(char(uint8_t(v) | 0x80));
        v >>= 7;
    }
    out->push_back(char(uint8_t(v)));
}

inline uint64_t zigzag(int64_t v) { return (uint64_t(v) << 1) ^ uint64_t(v >> 63); }
inline int64_t unzigzag(uint64_t v) { return int64_t(v >> 1) ^ -int64_t(v & 1); }

/// The kinds a value is written as. Two values are the same only when their
/// kinds are.
enum class Kind : uint8_t { Unit, Bool, Int, Big, Float, Char, Atom, Str, Nil, Cons, Array, Record, Map, Trunk, Tensor, Opaque };

Kind kind_of(Value v) {
    if (is_unit(v)) return Kind::Unit;
    if (is_bool(v)) return Kind::Bool;
    if (is_fixnum(v)) return Kind::Int;
    if (is_char(v)) return Kind::Char;
    if (is_atom(v)) return Kind::Atom;
    if (v == NIL) return Kind::Nil;
    if (is_ptr(v)) {
        switch (as_obj(v)->type) {
            case ObjType::BigInt: {
                int64_t small;
                return bigint::to_int64(v, &small) ? Kind::Int : Kind::Big;
            }
            case ObjType::Float: return Kind::Float;
            case ObjType::Str: case ObjType::BigStr: return Kind::Str;
            case ObjType::Cons: return Kind::Cons;
            case ObjType::Array: return Kind::Array;
            case ObjType::Record: return Kind::Record;
            case ObjType::Map: return Kind::Map;
            case ObjType::Trunk: return Kind::Trunk;
            case ObjType::Tensor: return Kind::Tensor;
            default: break;
        }
    }
    return Kind::Opaque;
}

inline void string_bytes(Value v, const char** data, size_t* n) {
    if (as_obj(v)->type == ObjType::Str) {
        auto* s = static_cast<StrObj*>(as_obj(v));
        *data = s->data();
        *n = s->len;
    } else {
        auto* b = static_cast<BigStrObj*>(as_obj(v));
        *data = b->data;
        *n = size_t(b->len);
    }
}

/// Whether a value of this kind is a part that can be written once and
/// referred to: anything with parts, and anything long.
inline bool shareable(Kind k, Value v) {
    switch (k) {
        case Kind::Cons: case Kind::Array: case Kind::Record: case Kind::Map:
        case Kind::Trunk: case Kind::Tensor: case Kind::Big:
            return true;
        case Kind::Str: {
            const char* d;
            size_t n;
            string_bytes(v, &d, &n);
            return n >= kShareStringFrom;
        }
        default:
            return false;
    }
}

/// An open-addressed table keyed by a nonzero word: an object's address, or
/// a content hash's first half. Every object a value reaches is looked up a
/// few times a pass, and `std::unordered_map` spent more on its nodes than
/// the passes spent on everything else.
template <class V>
class FlatMap {
public:
    FlatMap() : slots_(1024) {}
    V* find(uint64_t key) {
        const size_t mask = slots_.size() - 1;
        for (size_t i = spread(key) & mask;; i = (i + 1) & mask) {
            Slot& s = slots_[i];
            if (s.key == key) return &s.value;
            if (s.key == 0) return nullptr;
        }
    }
    /// The value under `key`, made if there was none. Good until the next
    /// insertion.
    V& at(uint64_t key) {
        if ((used_ + 1) * 2 > slots_.size()) grow();
        const size_t mask = slots_.size() - 1;
        for (size_t i = spread(key) & mask;; i = (i + 1) & mask) {
            Slot& s = slots_[i];
            if (s.key == key) return s.value;
            if (s.key == 0) {
                s.key = key;
                ++used_;
                return s.value;
            }
        }
    }

private:
    struct Slot {
        uint64_t key = 0;
        V value{};
    };
    static size_t spread(uint64_t k) { return size_t((k ^ (k >> 29)) * 0x9E3779B97F4A7C15ull >> 20); }
    void grow() {
        std::vector<Slot> old(slots_.size() * 2);
        old.swap(slots_);
        used_ = 0;
        for (Slot& s : old) {
            if (s.key) at(s.key) = std::move(s.value);
        }
    }
    std::vector<Slot> slots_;
    size_t used_ = 0;
};

inline uint64_t addr_key(Value v) { return uint64_t(reinterpret_cast<uintptr_t>(as_obj(v))); }
inline uint64_t content_key(const H128& h) { return h.a ? h.a : 1; }

/// Both passes, over one value. Nothing here collects: the value was forced
/// all the way down before it was handed over, and the passes allocate only
/// C++ memory, so the objects it keeps pointers to stay where they are.
struct Packer {
    Process& p;
    bool failed = false;

    struct Entry {
        Value rep;
        H128 h;
        uint32_t count = 0;
        int64_t def = -1;
        /// The next entry whose hash has the same first half, or -1.
        int64_t next = -1;
    };
    /// What is known of an object: its content's hash, once worked out, and
    /// the entry of its content, once classified.
    struct Info {
        H128 h{0, 0};
        int64_t entry = -1;
        bool hashed = false;
    };
    std::vector<Entry> entries;
    FlatMap<int64_t> by_hash;  // first entry with that first half, plus one
    FlatMap<Info> info;
    std::unordered_map<const Obj*, std::vector<std::pair<Value, Value>>> sorted;
    std::unordered_map<const Obj*, std::string> opaque_text;
    std::unordered_map<uint32_t, uint32_t> atoms;
    int64_t defs = 0;

    explicit Packer(Process& proc) : p(proc) {}

    const std::vector<std::pair<Value, Value>>& entries_of(Value m) {
        auto it = sorted.find(as_obj(m));
        if (it != sorted.end()) return it->second;
        std::vector<std::pair<Value, Value>> es;
        map_collect(m, es);
        bool ok = true;
        std::stable_sort(es.begin(), es.end(), [&](const auto& x, const auto& y) {
            int cmp = 0;
            if (!compare_for_wire(p, x.first, y.first, &cmp)) ok = false;
            return cmp < 0;
        });
        if (!ok) failed = true;
        return sorted.emplace(as_obj(m), std::move(es)).first->second;
    }

    const std::string& text_of(Value v) {
        auto key = is_ptr(v) ? static_cast<const Obj*>(as_obj(v)) : reinterpret_cast<const Obj*>(v);
        auto it = opaque_text.find(key);
        if (it != opaque_text.end()) return it->second;
        std::string t;
        if (!stringify(p, v, &t)) failed = true;
        return opaque_text.emplace(key, std::move(t)).first->second;
    }

    // --- the hash of a value's content ----------------------------------------

    H128 hash(Value v, int depth) {
        v = resolve(v);
        const Kind k = kind_of(v);
        switch (k) {
            case Kind::Unit: return start(P_UNIT);
            case Kind::Bool: return start(v == TRUE_V ? P_TRUE : P_FALSE);
            case Kind::Int: {
                int64_t n;
                if (is_fixnum(v)) n = fixnum_value(v); else bigint::to_int64(v, &n);
                return add(start(P_INT), uint64_t(n));
            }
            case Kind::Char: return add(start(P_CHAR), uint64_t(imm_payload(v)));
            case Kind::Atom: {
                const std::string& name = p.runtime().atom_name(uint32_t(imm_payload(v)));
                return add_bytes(start(P_ATOM), name.data(), name.size());
            }
            case Kind::Nil: return start(P_NIL);
            case Kind::Float: {
                uint64_t bits;
                const double d = static_cast<FloatObj*>(as_obj(v))->value;
                std::memcpy(&bits, &d, 8);
                return add(start(P_FLOAT), bits);
            }
            case Kind::Opaque: {
                const std::string& t = text_of(v);
                return add_bytes(start(P_STRING), t.data(), t.size());
            }
            default: break;
        }
        if (Info* known = info.find(addr_key(v)); known && known->hashed) return known->h;
        if (depth > kMaxDepth) {
            failed = true;
            return start(0);
        }
        H128 h;
        switch (k) {
            case Kind::Str: {
                const char* d;
                size_t n;
                string_bytes(v, &d, &n);
                h = add_bytes(start(P_STRING), d, n);
                break;
            }
            case Kind::Big: {
                const size_t n = bigint::le_width(v);
                std::string bytes(n, '\0');
                bigint::to_le_bytes(v, n, reinterpret_cast<uint8_t*>(bytes.data()));
                h = add_bytes(start(P_BIGINT), bytes.data(), n);
                break;
            }
            case Kind::Cons: {
                // A spine is hashed from its end, so that a long list costs
                // no C++ stack: each cell's hash is its head's and its tail's.
                std::vector<Value> cells;
                Value cur = v;
                for (;;) {
                    if (!is_obj(cur, ObjType::Cons)) break;
                    Info* known = info.find(addr_key(cur));
                    if (known && known->hashed) break;
                    cells.push_back(cur);
                    cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
                }
                H128 tail = hash(cur, depth + 1);
                for (size_t i = cells.size(); i-- > 0;) {
                    H128 c = add(start(P_LIST), hash(static_cast<ConsObj*>(as_obj(cells[i]))->head, depth + 1));
                    tail = add(c, tail);
                    Info& in = info.at(addr_key(cells[i]));
                    in.h = tail;
                    in.hashed = true;
                }
                return tail;
            }
            case Kind::Array: {
                auto* a = static_cast<ArrayObj*>(as_obj(v));
                h = add(start(P_ARRAY), uint64_t(a->len));
                for (uint32_t i = 0; i < a->len; ++i) h = add(h, hash(a->items()[i], depth + 1));
                break;
            }
            case Kind::Record: {
                auto* r = static_cast<RecordObj*>(as_obj(v));
                const std::string& name = p.runtime().atom_name(uint32_t(imm_payload(r->id())));
                h = add_bytes(start(P_RECORD), name.data(), name.size());
                h = add(add(h, uint64_t(r->variant() ? 1 : 0)), uint64_t(r->len));
                for (uint32_t i = 0; i < r->len; ++i) h = add(h, hash(r->items()[i], depth + 1));
                break;
            }
            case Kind::Map: {
                const auto& es = entries_of(v);
                h = add(start(P_MAP), uint64_t(es.size()));
                for (const auto& [key, val] : es) h = add(add(h, hash(key, depth + 1)), hash(val, depth + 1));
                break;
            }
            case Kind::Trunk: {
                const uint32_t n = trunk_length(v);
                h = add(start(P_TRUNK), uint64_t(n));
                for (uint32_t i = 0; i < n; ++i) h = add(h, hash(trunk_element(v, i), depth + 1));
                break;
            }
            case Kind::Tensor: {
                Value host;
                if (!tensor_host_copy(p, v, &host)) {
                    failed = true;
                    return start(0);
                }
                auto* t = static_cast<TensorObj*>(as_obj(host));
                h = add(start(P_TENSOR), uint64_t(t->rank));
                for (uint32_t axis = 0; axis < t->rank; ++axis) h = add(h, uint64_t(t->dims[axis]));
                h = add_bytes(h, reinterpret_cast<const char*>(t->data()), size_t(t->count) * 8);
                break;
            }
            default:
                h = start(0);
                break;
        }
        Info& in = info.at(addr_key(v));
        in.h = h;
        in.hashed = true;
        return h;
    }

    // --- whether two values are the same ---------------------------------------

    bool same(Value a, Value b, int depth) {
        a = resolve(a);
        b = resolve(b);
        if (a == b) return true;
        const Kind k = kind_of(a);
        if (k != kind_of(b) || depth > kMaxDepth) return false;
        switch (k) {
            case Kind::Unit: case Kind::Nil: return true;
            case Kind::Bool: case Kind::Char: case Kind::Atom: return false;  // immediates: equal only when identical
            case Kind::Int: {
                int64_t x, y;
                if (is_fixnum(a)) x = fixnum_value(a); else bigint::to_int64(a, &x);
                if (is_fixnum(b)) y = fixnum_value(b); else bigint::to_int64(b, &y);
                return x == y;
            }
            case Kind::Float: {
                const double x = static_cast<FloatObj*>(as_obj(a))->value, y = static_cast<FloatObj*>(as_obj(b))->value;
                return std::memcmp(&x, &y, 8) == 0;
            }
            case Kind::Opaque: return text_of(a) == text_of(b);
            default: break;
        }
        // No hashes below the top: `classify` asks only when the hashes of
        // the two already agree, and hashing the parts of a copy to reject
        // it sooner would cost more than the walk that confirms it.
        switch (k) {
            case Kind::Str: {
                const char *x, *y;
                size_t m, n;
                string_bytes(a, &x, &m);
                string_bytes(b, &y, &n);
                return m == n && std::memcmp(x, y, m) == 0;
            }
            case Kind::Big: return bigint::compare(a, b) == 0;
            case Kind::Cons: {
                while (is_obj(a, ObjType::Cons) && is_obj(b, ObjType::Cons)) {
                    if (a == b) return true;
                    if (!same(static_cast<ConsObj*>(as_obj(a))->head, static_cast<ConsObj*>(as_obj(b))->head, depth + 1)) return false;
                    a = resolve(static_cast<ConsObj*>(as_obj(a))->tail);
                    b = resolve(static_cast<ConsObj*>(as_obj(b))->tail);
                }
                return same(a, b, depth + 1);
            }
            case Kind::Array: {
                auto *x = static_cast<ArrayObj*>(as_obj(a)), *y = static_cast<ArrayObj*>(as_obj(b));
                if (x->len != y->len) return false;
                for (uint32_t i = 0; i < x->len; ++i)
                    if (!same(x->items()[i], y->items()[i], depth + 1)) return false;
                return true;
            }
            case Kind::Record: {
                auto *x = static_cast<RecordObj*>(as_obj(a)), *y = static_cast<RecordObj*>(as_obj(b));
                if (x->id() != y->id() || x->variant() != y->variant() || x->len != y->len) return false;
                for (uint32_t i = 0; i < x->len; ++i)
                    if (!same(x->items()[i], y->items()[i], depth + 1)) return false;
                return true;
            }
            case Kind::Map: {
                const auto& xs = entries_of(a);
                const auto& ys = entries_of(b);
                if (xs.size() != ys.size()) return false;
                for (size_t i = 0; i < xs.size(); ++i)
                    if (!same(xs[i].first, ys[i].first, depth + 1) || !same(xs[i].second, ys[i].second, depth + 1)) return false;
                return true;
            }
            case Kind::Trunk: {
                const uint32_t n = trunk_length(a);
                if (n != trunk_length(b)) return false;
                for (uint32_t i = 0; i < n; ++i)
                    if (!same(trunk_element(a, i), trunk_element(b, i), depth + 1)) return false;
                return true;
            }
            case Kind::Tensor:
                // Their hashes cover every double; that is all a tensor is
                // checked by, short of copying both to the host again.
                return hash(a, depth) == hash(b, depth);
            default:
                return false;
        }
    }

    // --- the first pass: what occurs more than once ------------------------------

    /// The entry for a shareable value's content, and whether it was there.
    std::pair<uint32_t, bool> classify(Value v, int depth) {
        Info* known = info.find(addr_key(v));
        if (known && known->entry >= 0) return {uint32_t(known->entry), true};
        const H128 h = known && known->hashed ? known->h : hash(v, depth);
        const int64_t first = by_hash.at(content_key(h)) - 1;
        for (int64_t i = first; i >= 0; i = entries[size_t(i)].next) {
            if (entries[size_t(i)].h == h && same(entries[size_t(i)].rep, v, depth)) {
                info.at(addr_key(v)).entry = i;
                return {uint32_t(i), true};
            }
        }
        const auto i = int64_t(entries.size());
        entries.push_back(Entry{v, h, 0, -1, first});
        by_hash.at(content_key(h)) = i + 1;
        info.at(addr_key(v)).entry = i;
        return {uint32_t(i), false};
    }

    Entry& entry(Value v) { return entries[size_t(info.find(addr_key(v))->entry)]; }

    void count(Value v, int depth) {
        if (failed) return;
        v = resolve(v);
        if (depth > kMaxDepth) {
            failed = true;
            return;
        }
        const Kind k = kind_of(v);
        if (!shareable(k, v)) return;
        if (k == Kind::Cons) {
            // Cell by cell, the spine iteratively: a cell seen before ends
            // the walk, since what follows it has been counted with it.
            Value cur = v;
            while (is_obj(cur, ObjType::Cons)) {
                auto [i, seen] = classify(cur, depth);
                ++entries[i].count;
                if (seen) return;
                count(static_cast<ConsObj*>(as_obj(cur))->head, depth + 1);
                cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
            }
            return;
        }
        auto [i, seen] = classify(v, depth);
        ++entries[i].count;
        if (seen) return;
        switch (k) {
            case Kind::Array: {
                auto* a = static_cast<ArrayObj*>(as_obj(v));
                for (uint32_t j = 0; j < a->len; ++j) count(a->items()[j], depth + 1);
                break;
            }
            case Kind::Record: {
                auto* r = static_cast<RecordObj*>(as_obj(v));
                for (uint32_t j = 0; j < r->len; ++j) count(r->items()[j], depth + 1);
                break;
            }
            case Kind::Map:
                for (const auto& [key, val] : entries_of(v)) {
                    count(key, depth + 1);
                    count(val, depth + 1);
                }
                break;
            case Kind::Trunk: {
                const uint32_t n = trunk_length(v);
                for (uint32_t j = 0; j < n; ++j) count(trunk_element(v, j), depth + 1);
                break;
            }
            default: break;
        }
    }

    // --- the second pass: the bytes ------------------------------------------------

    void write_atom(std::string* out, uint32_t id) {
        auto it = atoms.find(id);
        if (it != atoms.end()) {
            out->push_back(char(P_ATOM));
            put_varint(out, it->second);
            return;
        }
        const auto n = uint32_t(atoms.size());
        atoms.emplace(id, n);
        const std::string& name = p.runtime().atom_name(id);
        out->push_back(char(P_ATOM_NEW));
        put_varint(out, name.size());
        out->append(name);
    }

    void write(std::string* out, Value v, int depth) {
        if (failed) return;
        v = resolve(v);
        const Kind k = kind_of(v);
        if (shareable(k, v)) {
            Entry& e = entry(v);
            if (e.count >= 2) {
                if (e.def >= 0) {
                    out->push_back(char(P_REF));
                    put_varint(out, uint64_t(e.def));
                    return;
                }
                out->push_back(char(P_DEF));
                body(out, v, k, depth);
                // Numbered once read, as the reader numbers it: a part is
                // finished before anything can refer to it.
                entry(v).def = defs++;
                return;
            }
        }
        body(out, v, k, depth);
    }

    void body(std::string* out, Value v, Kind k, int depth) {
        if (depth > kMaxDepth) {
            failed = true;
            return;
        }
        switch (k) {
            case Kind::Unit: out->push_back(char(P_UNIT)); return;
            case Kind::Bool: out->push_back(char(v == TRUE_V ? P_TRUE : P_FALSE)); return;
            case Kind::Nil: out->push_back(char(P_NIL)); return;
            case Kind::Int: {
                int64_t n;
                if (is_fixnum(v)) n = fixnum_value(v); else bigint::to_int64(v, &n);
                out->push_back(char(P_INT));
                put_varint(out, zigzag(n));
                return;
            }
            case Kind::Char:
                out->push_back(char(P_CHAR));
                put_varint(out, uint64_t(imm_payload(v)));
                return;
            case Kind::Atom: write_atom(out, uint32_t(imm_payload(v))); return;
            case Kind::Float: {
                const double d = static_cast<FloatObj*>(as_obj(v))->value;
                char b[8];
                std::memcpy(b, &d, 8);
                out->push_back(char(P_FLOAT));
                out->append(b, 8);
                return;
            }
            case Kind::Str: {
                const char* d;
                size_t n;
                string_bytes(v, &d, &n);
                out->push_back(char(P_STRING));
                put_varint(out, n);
                out->append(d, n);
                return;
            }
            case Kind::Opaque: {
                const std::string& t = text_of(v);
                out->push_back(char(P_STRING));
                put_varint(out, t.size());
                out->append(t);
                return;
            }
            case Kind::Big: {
                const size_t n = bigint::le_width(v);
                out->push_back(char(P_BIGINT));
                put_varint(out, n);
                const size_t at = out->size();
                out->resize(at + n);
                bigint::to_le_bytes(v, n, reinterpret_cast<uint8_t*>(&(*out)[at]));
                return;
            }
            case Kind::Cons: {
                // A run of the cells nothing else shares, then the tail: the
                // empty list, or a shared cell, written by its number when it
                // has been written before.
                std::vector<Value> run{v};
                Value cur = resolve(static_cast<ConsObj*>(as_obj(v))->tail);
                while (is_obj(cur, ObjType::Cons) && entry(cur).count < 2) {
                    run.push_back(cur);
                    cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
                }
                out->push_back(char(P_LIST));
                put_varint(out, run.size());
                for (Value cell : run) write(out, static_cast<ConsObj*>(as_obj(cell))->head, depth + 1);
                write(out, cur, depth + 1);
                return;
            }
            case Kind::Array: {
                auto* a = static_cast<ArrayObj*>(as_obj(v));
                out->push_back(char(P_ARRAY));
                put_varint(out, a->len);
                for (uint32_t i = 0; i < a->len; ++i) write(out, a->items()[i], depth + 1);
                return;
            }
            case Kind::Record: {
                auto* r = static_cast<RecordObj*>(as_obj(v));
                out->push_back(char(P_RECORD));
                write_atom(out, uint32_t(imm_payload(r->id())));
                out->push_back(char(r->variant() ? 1 : 0));
                put_varint(out, r->len);
                for (uint32_t i = 0; i < r->len; ++i) write(out, r->items()[i], depth + 1);
                return;
            }
            case Kind::Map: {
                const auto& es = entries_of(v);
                out->push_back(char(P_MAP));
                put_varint(out, es.size());
                for (const auto& [key, val] : es) {
                    write(out, key, depth + 1);
                    write(out, val, depth + 1);
                }
                return;
            }
            case Kind::Trunk: {
                const uint32_t n = trunk_length(v);
                out->push_back(char(P_TRUNK));
                put_varint(out, n);
                for (uint32_t i = 0; i < n; ++i) write(out, trunk_element(v, i), depth + 1);
                return;
            }
            case Kind::Tensor: {
                Value host;
                if (!tensor_host_copy(p, v, &host)) {
                    failed = true;
                    return;
                }
                auto* t = static_cast<TensorObj*>(as_obj(host));
                out->push_back(char(P_TENSOR));
                out->push_back(char(t->rank));
                for (uint32_t axis = 0; axis < t->rank; ++axis) put_varint(out, t->dims[axis]);
                out->append(reinterpret_cast<const char*>(t->data()), size_t(t->count) * 8);
                return;
            }
        }
    }
};

/// The packed form read back. Never raises: bytes that are not a value are
/// `false`, as the first form's reader says.
struct Unpacker {
    Process& p;
    const unsigned char* data;
    uint64_t len;
    uint64_t at = 0;
    std::vector<Value> defs{};
    std::vector<uint32_t> atom_ids{};

    Unpacker(Process& proc, const unsigned char* d, uint64_t n) : p(proc), data(d), len(n) {}

    bool has(uint64_t n) const { return at <= len && n <= len - at; }

    bool varint(uint64_t* out) {
        uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (!has(1)) return false;
            const uint8_t b = data[at++];
            v |= uint64_t(b & 0x7f) << shift;
            if (!(b & 0x80)) {
                *out = v;
                return true;
            }
        }
        return false;
    }

    bool atom(uint32_t* id) {
        if (!has(1)) return false;
        const uint8_t tag = data[at++];
        uint64_t n;
        if (!varint(&n)) return false;
        if (tag == P_ATOM) {
            if (n >= atom_ids.size()) return false;
            *id = atom_ids[size_t(n)];
            return true;
        }
        if (tag != P_ATOM_NEW || !has(n)) return false;
        // Only ever looked up, never made, as the first form's reader does:
        // a message cannot grow the atom table.
        if (!p.runtime().find_atom(std::string_view(reinterpret_cast<const char*>(data + at), size_t(n)), id)) return false;
        at += n;
        atom_ids.push_back(*id);
        return true;
    }

    bool items(uint64_t n, std::vector<Value>* out, int depth) {
        if (n > len) return false;
        out->reserve(size_t(n));
        for (uint64_t i = 0; i < n; ++i) {
            Value item;
            if (!read(&item, depth + 1)) return false;
            out->push_back(item);
        }
        return true;
    }

    bool read(Value* out, int depth) {
        if (depth > kMaxDepth || !has(1)) return false;
        const uint8_t tag = data[at];
        if (tag == P_ATOM || tag == P_ATOM_NEW) {
            uint32_t id;
            if (!atom(&id)) return false;
            *out = make_atom(id);
            return true;
        }
        ++at;
        uint64_t n;
        switch (tag) {
            case P_UNIT: *out = UNIT; return true;
            case P_FALSE: *out = make_bool(false); return true;
            case P_TRUE: *out = make_bool(true); return true;
            case P_NIL: *out = NIL; return true;
            case P_DEF:
                if (!read(out, depth + 1)) return false;
                defs.push_back(*out);
                return true;
            case P_REF:
                if (!varint(&n) || n >= defs.size()) return false;
                *out = defs[size_t(n)];
                return true;
            case P_INT:
                if (!varint(&n)) return false;
                *out = make_integer(p, unzigzag(n));
                return true;
            case P_CHAR:
                if (!varint(&n) || n > 0x10FFFF || (n >= 0xD800 && n <= 0xDFFF)) return false;
                *out = make_char(uint32_t(n));
                return true;
            case P_FLOAT: {
                if (!has(8)) return false;
                double d;
                std::memcpy(&d, data + at, 8);
                at += 8;
                *out = p.heap().make_float(d);
                return true;
            }
            case P_STRING:
                if (!varint(&n) || !has(n) || n > UINT32_MAX) return false;
                *out = p.heap().make_string(reinterpret_cast<const char*>(data + at), uint32_t(n));
                at += n;
                return true;
            case P_BIGINT:
                if (!varint(&n) || !has(n)) return false;
                *out = bigint::from_le_bytes(p.heap(), data + at, size_t(n));
                at += n;
                return true;
            case P_LIST: {
                std::vector<Value> heads;
                if (!varint(&n) || !items(n, &heads, depth)) return false;
                Value tail;
                if (!read(&tail, depth + 1)) return false;
                tail = resolve(tail);
                if (tail != NIL && !is_obj(tail, ObjType::Cons)) return false;
                for (size_t i = heads.size(); i-- > 0;) tail = p.heap().make_cons(heads[i], tail);
                *out = tail;
                return true;
            }
            case P_ARRAY: {
                std::vector<Value> xs;
                if (!varint(&n) || !items(n, &xs, depth)) return false;
                Value arr = p.heap().make_array(uint32_t(xs.size()));
                for (size_t i = 0; i < xs.size(); ++i) static_cast<ArrayObj*>(as_obj(arr))->items()[i] = xs[i];
                *out = arr;
                return true;
            }
            case P_TRUNK: {
                std::vector<Value> xs;
                if (!varint(&n) || !items(n, &xs, depth)) return false;
                *out = trunk_of_values(p, xs);
                return true;
            }
            case P_RECORD: {
                uint32_t id;
                if (!atom(&id) || !has(1)) return false;
                const bool variant = (data[at++] & 1) != 0;
                std::vector<Value> xs;
                if (!varint(&n) || !items(n, &xs, depth)) return false;
                Value rec = p.heap().make_record(record_tag(make_atom(id), variant), uint32_t(xs.size()));
                for (size_t i = 0; i < xs.size(); ++i) static_cast<RecordObj*>(as_obj(rec))->items()[i] = xs[i];
                *out = rec;
                return true;
            }
            case P_MAP: {
                if (!varint(&n) || n > len) return false;
                Value m = p.heap().make_map(0);
                for (uint64_t i = 0; i < n; ++i) {
                    Value k, v;
                    if (!read(&k, depth + 1) || !read(&v, depth + 1)) return false;
                    m = resolve(map_insert(p, m, k, v));
                }
                *out = m;
                return true;
            }
            case P_TENSOR: {
                if (!has(1)) return false;
                const uint32_t rank = data[at++];
                if (rank == 0 || rank > TENSOR_MAX_RANK) return false;
                uint32_t dims[TENSOR_MAX_RANK];
                uint64_t count = 1;
                const uint64_t limit = (uint64_t(UINT32_MAX) - sizeof(TensorObj) - 64) / 8;
                for (uint32_t axis = 0; axis < rank; ++axis) {
                    if (!varint(&n) || n == 0 || n > UINT32_MAX || count > limit / n) return false;
                    dims[axis] = uint32_t(n);
                    count *= n;
                }
                if (!has(count * 8)) return false;
                Value v = p.heap().make_tensor(rank, dims, count);
                std::memcpy(static_cast<TensorObj*>(as_obj(v))->data(), data + at, size_t(count) * 8);
                at += count * 8;
                *out = v;
                return true;
            }
            default:
                return false;
        }
    }
};

}  // namespace

/// `vm.wire_pack v`: `v` in the packed form. See the top of this file.
NativeResult vm_wire_pack(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    Packer packer(p);
    packer.count(forced, 0);
    std::string out(1, char(PACK_MAGIC));
    if (!packer.failed) packer.write(&out, forced, 0);
    const char* why = packer.failed ? "wire_pack: a map's keys could not be compared, or the value nests too deep"
                    : out.size() > UINT32_MAX ? "wire_pack: the message is larger than a string can be"
                    : nullptr;
    if (why) return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error, why));
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

bool is_packed(const unsigned char* data, uint64_t len) { return len > 0 && data[0] == PACK_MAGIC; }

bool wire_unpack(Process& p, const unsigned char* data, uint64_t len, Value* out) {
    Unpacker u{p, data, len};
    u.at = 1;
    if (!u.read(out, 0)) return false;
    return u.at == len;
}

}  // namespace dream
