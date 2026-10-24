#include "builtins.hpp"
#include "regex.hpp"
#include "digest.hpp"

#include "io.hpp"
#include "os.hpp"

#include <array>
#include <unordered_map>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "bigint.hpp"
#include "interp.hpp"
#include "tensor.hpp"
#include "process.hpp"
#include "scheduler.hpp"

namespace dream {

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

namespace {

uint64_t key_hash(Value v) {
    v = resolve(v);
    // A big string hashes by the same walk over the same bytes, so it lands in
    // the bucket its contents earn and a plain string of those bytes finds it.
    // The answer is kept on the object: everything else here is a few machine
    // words, and this one can be a gigabyte.
    if (is_ptr(v) && as_obj(v)->type == ObjType::BigStr) {
        auto* b = static_cast<BigStrObj*>(as_obj(v));
        if (b->hash == 0) b->hash = bytes_hash(Bytes{b->data, b->len});
        return b->hash;
    }
    Bytes bytes;
    if (is_ptr(v) && as_obj(v)->type == ObjType::Str) {
        string_bytes(v, &bytes);
        return bytes_hash(bytes);
    }
    if (is_ptr(v) && as_obj(v)->type == ObjType::BigInt) {
        uint64_t h = bigint::hash(static_cast<BigIntObj*>(as_obj(v)));
        return h ? h : 1;
    }
    uint64_t h = v;
    if (is_ptr(v)) {
        Obj* o = as_obj(v);
        if (o->type == ObjType::Float) {
            double d = static_cast<FloatObj*>(o)->value;
            std::memcpy(&h, &d, 8);
        } else if (o->type == ObjType::Pid) {
            // Compared by the process it names, so hashed by it too: two
            // boxes of one pid are one key.
            h = static_cast<PidObj*>(o)->id;
        } else if (o->type == ObjType::Record && static_cast<RecordObj*>(o)->len == 0) {
            // A variant with no fields -- `Shape.empty` -- is what an atom
            // was before it was a record, and is a key the way an atom is.
            h = static_cast<RecordObj*>(o)->id();
        } else {
            h = identity_hash(o);
        }
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h ? h : 1;
}

/// Keys are compared by identity or by contents for the flat types. Anything
/// deeper is compared by identity, which is what a hash map can promise
/// without forcing arbitrary structure at lookup time.
bool key_equal(Value a, Value b) {
    a = resolve(a);
    b = resolve(b);
    if (a == b) return true;
    if (!is_ptr(a) || !is_ptr(b)) return false;
    Obj* x = as_obj(a);
    Obj* y = as_obj(b);
    // Before the type test, because the two string representations are one
    // kind of key: they hash alike, so they must compare alike.
    Bytes sa, sb;
    if (string_bytes(a, &sa) && string_bytes(b, &sb)) return bytes_equal(sa, sb);
    if (x->type != y->type) return false;
    if (x->type == ObjType::Float) {
        return static_cast<FloatObj*>(x)->value == static_cast<FloatObj*>(y)->value;
    }
    if (x->type == ObjType::Pid) {
        return static_cast<PidObj*>(x)->id == static_cast<PidObj*>(y)->id;
    }
    if (x->type == ObjType::Record) {
        auto* rx = static_cast<RecordObj*>(x);
        auto* ry = static_cast<RecordObj*>(y);
        return rx->len == 0 && ry->len == 0 && rx->tag == ry->tag;
    }
    if (x->type == ObjType::BigInt) {
        return bigint::equal(static_cast<BigIntObj*>(x), static_cast<BigIntObj*>(y));
    }
    return false;
}

/// The number of entries a node holds, counting a whole subtree.
uint32_t map_node_count(Value v) {
    if (!is_ptr(v)) return 0;
    Obj* o = as_obj(v);
    if (o->type == ObjType::Map) return static_cast<MapObj*>(o)->count;
    if (o->type != ObjType::MapLeaf) return 0;
    uint32_t n = 0;
    for (Value cur = v; is_obj(cur, ObjType::MapLeaf); cur = static_cast<MapLeafObj*>(as_obj(cur))->next) {
        ++n;
    }
    return n;
}

/// A leaf chain with `key` set to `value`. Replacing an existing key rebuilds
/// the chain above it; adding a new one puts it at the front. Either way the
/// chain the caller had is left untouched, which is the whole contract.
Value map_leaf_assoc(Heap& h, Value leaf, uint64_t hash, Value key, Value value, bool* added) {
    // Is the key already here? Walk first so that the common case -- it is not
    // -- costs one pass and no allocation.
    bool present = false;
    for (Value cur = leaf; is_obj(cur, ObjType::MapLeaf); ) {
        auto* l = static_cast<MapLeafObj*>(as_obj(cur));
        if (key_equal(l->key, key)) { present = true; break; }
        cur = l->next;
    }
    if (!present) {
        *added = true;
        return h.make_map_leaf(hash, key, value, leaf);
    }
    // Rebuild only as far as the entry that changed; the rest of the chain is
    // shared.
    Value rebuilt = NIL_SLOT;
    std::vector<MapLeafObj*> before;
    Value cur = leaf;
    while (is_obj(cur, ObjType::MapLeaf)) {
        auto* l = static_cast<MapLeafObj*>(as_obj(cur));
        if (key_equal(l->key, key)) {
            rebuilt = h.make_map_leaf(hash, key, value, l->next);
            break;
        }
        before.push_back(l);
        cur = l->next;
    }
    for (size_t i = before.size(); i > 0; --i) {
        rebuilt = h.make_map_leaf(hash, before[i - 1]->key, before[i - 1]->value, rebuilt);
    }
    return rebuilt;
}

Value map_assoc(Heap& h, Value node, uint32_t shift, uint64_t hash, Value key, Value value,
                bool* added);

/// Two nodes with different hashes, put under a branch deep enough to tell them
/// apart. They may agree for several levels, so this nests until they do not.
Value map_split(Heap& h, uint32_t shift, uint64_t h1, Value n1, uint64_t h2, Value n2) {
    uint32_t i1 = map_index(h1, shift);
    uint32_t i2 = map_index(h2, shift);
    if (i1 == i2) {
        Value deeper = map_split(h, shift + MAP_BITS, h1, n1, h2, n2);
        Value branch = h.make_map_branch(1);
        auto* b = static_cast<MapObj*>(as_obj(branch));
        b->bitmap = 1u << i1;
        b->count = map_node_count(deeper);
        b->slots()[0] = deeper;
        return branch;
    }
    Value branch = h.make_map_branch(2);
    auto* b = static_cast<MapObj*>(as_obj(branch));
    b->bitmap = (1u << i1) | (1u << i2);
    b->count = map_node_count(n1) + map_node_count(n2);
    b->slots()[i1 < i2 ? 0 : 1] = n1;
    b->slots()[i1 < i2 ? 1 : 0] = n2;
    return branch;
}

/// `node` with `key` set to `value`, sharing everything the change does not
/// touch. `added` says whether the map grew, so counts stay right without
/// recounting a subtree.
Value map_assoc(Heap& h, Value node, uint32_t shift, uint64_t hash, Value key, Value value,
                bool* added) {
    if (is_obj(node, ObjType::MapLeaf)) {
        auto* l = static_cast<MapLeafObj*>(as_obj(node));
        if (l->hash == hash) return map_leaf_assoc(h, node, hash, key, value, added);
        // Different hashes: they belong under a branch, not in one chain.
        *added = true;
        Value fresh = h.make_map_leaf(hash, key, value, NIL_SLOT);
        return map_split(h, shift, l->hash, node, hash, fresh);
    }
    auto* m = static_cast<MapObj*>(as_obj(node));
    uint32_t bit = 1u << map_index(hash, shift);
    uint32_t at = map_slot_of(m->bitmap, bit);
    uint32_t slots = map_bit_count(m->bitmap);

    if (m->bitmap & bit) {
        Value child = m->slots()[at];
        Value grown = map_assoc(h, child, shift + MAP_BITS, hash, key, value, added);
        Value branch = h.make_map_branch(slots);
        auto* b = static_cast<MapObj*>(as_obj(branch));
        m = static_cast<MapObj*>(as_obj(node));
        b->bitmap = m->bitmap;
        b->count = m->count + (*added ? 1 : 0);
        for (uint32_t i = 0; i < slots; ++i) b->slots()[i] = m->slots()[i];
        b->slots()[at] = grown;
        return branch;
    }
    // A slot nothing is using yet: widen by one and keep the children packed.
    *added = true;
    Value fresh = h.make_map_leaf(hash, key, value, NIL_SLOT);
    Value branch = h.make_map_branch(slots + 1);
    auto* b = static_cast<MapObj*>(as_obj(branch));
    m = static_cast<MapObj*>(as_obj(node));
    b->bitmap = m->bitmap | bit;
    b->count = m->count + 1;
    for (uint32_t i = 0; i < at; ++i) b->slots()[i] = m->slots()[i];
    b->slots()[at] = fresh;
    for (uint32_t i = at; i < slots; ++i) b->slots()[i + 1] = m->slots()[i];
    return branch;
}

/// `map_assoc` for one of a batch of inserts. A branch in `owned` was made by
/// an earlier insert of the same batch and nothing outside it has seen it yet,
/// so it is changed where it stands instead of being copied again: the batch
/// copies each branch on its keys' paths once, where a run of `map_assoc`s
/// copied the root, and every shared branch, once per key.
///
/// Only branches are reused. A leaf chain is rebuilt as `map_assoc` rebuilds
/// it, and a branch `map_split` makes is not recorded -- both are small, and a
/// batch is a handful of a record's fields, not a bulk load.
///
/// The record of what is owned is a fixed array on the stack, not a vector:
/// a batch is a handful of keys on a short path, and the heap allocation a
/// vector makes cost more than the copies it saved -- a `mapping` pipeline
/// measured 12% slower with one. A branch made after the array is full is
/// simply not recorded, and copied again if it is met again, as before.
struct OwnedBranches {
    Obj* at[32];
    uint32_t n = 0;
    bool has(Obj* o) const {
        for (uint32_t i = 0; i < n; ++i) if (at[i] == o) return true;
        return false;
    }
    void add(Obj* o) { if (n < 32) at[n++] = o; }
};

Value map_assoc_owned(Heap& h, Value node, uint32_t shift, uint64_t hash, Value key, Value value,
                      bool* added, OwnedBranches& owned) {
    if (is_obj(node, ObjType::MapLeaf)) return map_assoc(h, node, shift, hash, key, value, added);
    auto* m = static_cast<MapObj*>(as_obj(node));
    const bool mine = owned.has(as_obj(node));
    uint32_t bit = 1u << map_index(hash, shift);
    if (mine && (m->bitmap & bit)) {
        uint32_t at = map_slot_of(m->bitmap, bit);
        Value grown = map_assoc_owned(h, m->slots()[at], shift + MAP_BITS, hash, key, value, added,
                                      owned);
        m = static_cast<MapObj*>(as_obj(node));
        m->slots()[at] = grown;
        if (*added) m->count += 1;
        h.remember_if_old(m, grown);
        return node;
    }
    if (!mine && (m->bitmap & bit)) {
        // Copied once here, then owned: the child is inserted into below, and
        // the copy points at whatever that answers.
        uint32_t at = map_slot_of(m->bitmap, bit);
        uint32_t slots = map_bit_count(m->bitmap);
        Value branch = h.make_map_branch(slots);
        auto* b = static_cast<MapObj*>(as_obj(branch));
        m = static_cast<MapObj*>(as_obj(node));
        b->bitmap = m->bitmap;
        b->count = m->count;
        for (uint32_t i = 0; i < slots; ++i) b->slots()[i] = m->slots()[i];
        owned.add(as_obj(branch));
        Value grown = map_assoc_owned(h, b->slots()[at], shift + MAP_BITS, hash, key, value, added,
                                      owned);
        b = static_cast<MapObj*>(as_obj(branch));
        b->slots()[at] = grown;
        if (*added) b->count += 1;
        return branch;
    }
    // A new slot widens the branch, which is a new object either way.
    Value branch = map_assoc(h, node, shift, hash, key, value, added);
    owned.add(as_obj(branch));
    return branch;
}

/// `node` without `key`. Answers NIL_SLOT when the node is left empty, so a
/// parent can drop the slot rather than keep an empty branch forever.
Value map_dissoc(Heap& h, Value node, uint32_t shift, uint64_t hash, Value key, bool* removed) {
    if (is_obj(node, ObjType::MapLeaf)) {
        auto* l = static_cast<MapLeafObj*>(as_obj(node));
        if (l->hash != hash) return node;
        std::vector<MapLeafObj*> keep;
        bool found = false;
        for (Value cur = node; is_obj(cur, ObjType::MapLeaf); ) {
            auto* e = static_cast<MapLeafObj*>(as_obj(cur));
            if (!found && key_equal(e->key, key)) { found = true; }
            else { keep.push_back(e); }
            cur = e->next;
        }
        if (!found) return node;
        *removed = true;
        Value rebuilt = NIL_SLOT;
        for (size_t i = keep.size(); i > 0; --i) {
            rebuilt = h.make_map_leaf(hash, keep[i - 1]->key, keep[i - 1]->value, rebuilt);
        }
        return rebuilt;
    }
    auto* m = static_cast<MapObj*>(as_obj(node));
    uint32_t bit = 1u << map_index(hash, shift);
    if (!(m->bitmap & bit)) return node;
    uint32_t at = map_slot_of(m->bitmap, bit);
    uint32_t slots = map_bit_count(m->bitmap);
    Value shrunk = map_dissoc(h, m->slots()[at], shift + MAP_BITS, hash, key, removed);
    if (!*removed) return node;

    if (shrunk == NIL_SLOT) {
        Value branch = h.make_map_branch(slots - 1);
        auto* b = static_cast<MapObj*>(as_obj(branch));
        m = static_cast<MapObj*>(as_obj(node));
        b->bitmap = m->bitmap & ~bit;
        b->count = m->count - 1;
        for (uint32_t i = 0; i < at; ++i) b->slots()[i] = m->slots()[i];
        for (uint32_t i = at + 1; i < slots; ++i) b->slots()[i - 1] = m->slots()[i];
        // An empty branch below the root is not worth keeping.
        return (b->bitmap == 0 && shift > 0) ? NIL_SLOT : branch;
    }
    Value branch = h.make_map_branch(slots);
    auto* b = static_cast<MapObj*>(as_obj(branch));
    m = static_cast<MapObj*>(as_obj(node));
    b->bitmap = m->bitmap;
    b->count = m->count - 1;
    for (uint32_t i = 0; i < slots; ++i) b->slots()[i] = m->slots()[i];
    b->slots()[at] = shrunk;
    return branch;
}

}  // namespace

Value map_insert(Process& p, Value map, Value key, Value value) {
    map = resolve(map);
    if (!is_obj(map, ObjType::Map)) return map;
    bool added = false;
    return map_assoc(p.heap(), map, 0, key_hash(key), resolve(key), value, &added);
}

Value map_insert_many(Process& p, Value map, uint32_t n, const Value* pairs) {
    map = resolve(map);
    if (!is_obj(map, ObjType::Map)) return map;
    OwnedBranches owned;
    for (uint32_t i = 0; i < n; ++i) {
        bool added = false;
        Value key = resolve(pairs[2 * i]);
        map = map_assoc_owned(p.heap(), map, 0, key_hash(key), key, pairs[2 * i + 1], &added, owned);
    }
    return map;
}

Value map_erase(Process& p, Value map, Value key) {
    map = resolve(map);
    if (!is_obj(map, ObjType::Map)) return map;
    bool removed = false;
    Value out = map_dissoc(p.heap(), map, 0, key_hash(key), resolve(key), &removed);
    // The root stays a branch even when the last entry goes.
    return out == NIL_SLOT ? p.heap().make_map(0) : out;
}

/// Every entry, appended to `out`. The order is the trie's, which is to say the
/// hash order -- unspecified, as the language promises.
void map_collect(Value node, std::vector<std::pair<Value, Value>>& out) {
    if (!is_ptr(node)) return;
    Obj* o = as_obj(node);
    if (o->type == ObjType::MapLeaf) {
        for (Value cur = node; is_obj(cur, ObjType::MapLeaf); ) {
            auto* l = static_cast<MapLeafObj*>(as_obj(cur));
            out.emplace_back(l->key, l->value);
            cur = l->next;
        }
        return;
    }
    if (o->type != ObjType::Map) return;
    auto* m = static_cast<MapObj*>(o);
    uint32_t slots = map_bit_count(m->bitmap);
    for (uint32_t i = 0; i < slots; ++i) map_collect(m->slots()[i], out);
}

bool map_lookup(Process&, Value map, Value key, Value* out) {
    uint64_t hash = key_hash(key);
    Value node = resolve(map);
    uint32_t shift = 0;
    // Down five bits of hash at a time until the branch runs out or a leaf
    // answers. Thirteen levels is the most a 64-bit hash can ask for.
    while (is_ptr(node)) {
        Obj* o = as_obj(node);
        if (o->type == ObjType::MapLeaf) {
            for (Value cur = node; is_obj(cur, ObjType::MapLeaf); ) {
                auto* l = static_cast<MapLeafObj*>(as_obj(cur));
                if (l->hash == hash && key_equal(l->key, key)) {
                    if (out) *out = l->value;
                    return true;
                }
                cur = l->next;
            }
            return false;
        }
        if (o->type != ObjType::Map) return false;
        auto* m = static_cast<MapObj*>(o);
        uint32_t bit = 1u << map_index(hash, shift);
        if (!(m->bitmap & bit)) return false;
        node = m->slots()[map_slot_of(m->bitmap, bit)];
        shift += MAP_BITS;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// The shortest decimal that reads back as the same number, written the way
/// Python's `repr` writes one: in positional notation from 1e-4 up to 1e16,
/// and with an exponent only outside that. `%g` alone picks the digits well
/// and the notation badly -- it uses an exponent whenever the digits are
/// fewer than the integer part's, so 10.0, whose shortest form is one digit,
/// came out as `1e+01` in the middle of `[6, 8, 1e+01, 12]`. `f32` asks for the
/// shortest that reads back as the same float32, which is how a float32
/// tensor's 0.1 prints as 0.1 and not as the double it widens to.
void append_shortest(double d, bool f32, std::string* out) {
    char buf[64];
    int prec = 1;
    for (; prec < 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, d);
        double back = std::strtod(buf, nullptr);
        if (f32 ? float(back) == float(d) : back == d) break;
    }
    if (prec == 17) std::snprintf(buf, sizeof buf, "%.17g", d);
    if (std::isfinite(d) && d != 0 && std::strchr(buf, 'e')) {
        int exponent = int(std::floor(std::log10(std::fabs(d))));
        if (exponent >= -4 && exponent < 16) {
            std::snprintf(buf, sizeof buf, "%.*f", std::max(0, prec - 1 - exponent), d);
        }
    }
    out->append(buf);
}

namespace {

bool stringify_into(Process& p, Value v, std::string* out, bool quoted, int depth);

bool stringify_list(Process& p, Value v, std::string* out, int depth) {
    out->push_back('[');
    bool first = true;
    // Rendering an element forces it, which can collect, so the rest of the
    // list is carried on the value stack where the collector will update it.
    const size_t base = p.stack.size();
    p.stack.push_back(v);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return false;
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            // An improper tail: show it rather than pretend the list ended.
            out->append(" | ");
            if (!stringify_into(p, w, out, true, depth + 1)) {
                if (!p.force_blocked) p.stack.resize(base);
                return false;
            }
            break;
        }
        if (!first) out->append(", ");
        first = false;
        p.stack[base] = w;
        if (!stringify_into(p, static_cast<ConsObj*>(as_obj(w))->head, out, true, depth + 1)) {
            if (!p.force_blocked) p.stack.resize(base);
            return false;
        }
        p.stack[base] = static_cast<ConsObj*>(as_obj(p.stack[base]))->tail;
    }
    p.stack.resize(base);
    out->push_back(']');
    return true;
}

bool stringify_into(Process& p, Value v, std::string* out, bool quoted, int depth) {
    if (depth > 128) {
        out->append("...");
        return true;
    }
    Value w;
    if (!force_whnf(p, v, &w)) return false;

    if (is_fixnum(w)) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%" PRId64, fixnum_value(w));
        out->append(buf);
        return true;
    }
    if (is_imm(w)) {
        switch (imm_kind(w)) {
            case IMM_UNIT: out->append("()"); return true;
            case IMM_BOOL: out->append(truthy(w) ? "true" : "false"); return true;
            case IMM_NIL: out->append("[]"); return true;
            case IMM_ATOM:
                out->push_back(':');
                out->append(p.runtime().atom_name(uint32_t(imm_payload(w))));
                return true;
            case IMM_CHAR: {
                uint32_t cp = uint32_t(imm_payload(w));
                if (quoted) out->push_back('\'');
                // Encode as UTF-8.
                if (cp < 0x80) {
                    out->push_back(char(cp));
                } else if (cp < 0x800) {
                    out->push_back(char(0xC0 | (cp >> 6)));
                    out->push_back(char(0x80 | (cp & 0x3F)));
                } else if (cp < 0x10000) {
                    out->push_back(char(0xE0 | (cp >> 12)));
                    out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
                    out->push_back(char(0x80 | (cp & 0x3F)));
                } else {
                    out->push_back(char(0xF0 | (cp >> 18)));
                    out->push_back(char(0x80 | ((cp >> 12) & 0x3F)));
                    out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
                    out->push_back(char(0x80 | (cp & 0x3F)));
                }
                if (quoted) out->push_back('\'');
                return true;
            }
            case IMM_BUILTIN:
                out->append("<builtin ");
                out->append(builtin_def(uint32_t(imm_payload(w))).name);
                out->push_back('>');
                return true;
            default: out->append("<value>"); return true;
        }
    }
    if (!is_ptr(w)) { out->append("<value>"); return true; }

    switch (as_obj(w)->type) {
        case ObjType::BigInt:
            bigint::append_decimal(w, out);
            return true;
        case ObjType::Float: {
            append_shortest(static_cast<FloatObj*>(as_obj(w))->value, false, out);
            return true;
        }
        case ObjType::BigStr:
            // `to_string` refuses outright; here the value is being rendered
            // inside something larger, where raising would lose the rest of the
            // structure. A placeholder says which value it was and how big.
            out->append("<big string, " +
                        std::to_string(static_cast<BigStrObj*>(as_obj(w))->len) + " bytes>");
            return true;
        case ObjType::Str: {
            auto* s = static_cast<StrObj*>(as_obj(w));
            if (!quoted) {
                out->append(s->data(), s->len);
                return true;
            }
            out->push_back('"');
            for (uint32_t i = 0; i < s->len; ++i) {
                char ch = s->data()[i];
                switch (ch) {
                    case '"': out->append("\\\""); break;
                    case '\\': out->append("\\\\"); break;
                    case '\n': out->append("\\n"); break;
                    case '\t': out->append("\\t"); break;
                    default: out->push_back(ch);
                }
            }
            out->push_back('"');
            return true;
        }
        case ObjType::Cons: return stringify_list(p, w, out, depth);
        case ObjType::Array: {
            out->append("#[");
            p.stack.push_back(w);
            uint32_t len = static_cast<ArrayObj*>(as_obj(w))->len;
            for (uint32_t i = 0; i < len; ++i) {
                if (i) out->append(", ");
                Value item = static_cast<ArrayObj*>(as_obj(p.stack.back()))->items()[i];
                if (!stringify_into(p, item, out, true, depth + 1)) {
                    p.stack.pop_back();
                    return false;
                }
            }
            p.stack.pop_back();
            out->push_back(']');
            return true;
        }
        case ObjType::Trunk: {
            // `trunk[1, 2, 3]`: a sequence, said to be a trunk.
            out->append("trunk[");
            p.stack.push_back(w);
            uint32_t len = trunk_length(w);
            for (uint32_t i = 0; i < len; ++i) {
                if (i) out->append(", ");
                if (!stringify_into(p, trunk_element(p.stack.back(), i), out, true, depth + 1)) {
                    p.stack.pop_back();
                    return false;
                }
            }
            p.stack.pop_back();
            out->push_back(']');
            return true;
        }
        case ObjType::Record: {
            // `Shape.circle(1.0)`, `Shape.empty`, `Point(3, 4)`: the
            // declaration's name, then the fields in the order they were
            // declared. A variant with no fields is the name alone.
            out->append(record_shown_name(p, w));
            uint32_t len = static_cast<RecordObj*>(as_obj(w))->len;
            if (len == 0 && (static_cast<RecordObj*>(as_obj(w))->variant())) return true;
            out->push_back('(');
            p.stack.push_back(w);
            for (uint32_t i = 0; i < len; ++i) {
                if (i) out->append(", ");
                Value item = static_cast<RecordObj*>(as_obj(p.stack.back()))->items()[i];
                if (!stringify_into(p, item, out, true, depth + 1)) {
                    p.stack.pop_back();
                    return false;
                }
            }
            p.stack.pop_back();
            out->push_back(')');
            return true;
        }
        case ObjType::Map: {
            out->append("%{");
            // The entries wait on the value stack, a key then its value, for
            // the reason `force_deep`'s map case gives: rendering one forces
            // it, a force under a native that vouches may collect, and a
            // collection rewrites what it can see. A vector of `Value` is not that, so
            // the entries after the first would be husks of moved objects.
            std::vector<std::pair<Value, Value>> entries;
            map_collect(w, entries);
            const size_t base = p.stack.size();
            for (auto& [k, val] : entries) {
                p.stack.push_back(k);
                p.stack.push_back(val);
            }
            const size_t count = entries.size();
            entries.clear();
            for (size_t i = 0; i < count; ++i) {
                if (i) out->append(", ");
                if (!stringify_into(p, p.stack[base + 2 * i], out, true, depth + 1) ||
                    (out->append(" => "),
                     !stringify_into(p, p.stack[base + 2 * i + 1], out, true, depth + 1))) {
                    if (!p.force_blocked) p.stack.resize(base);
                    return false;
                }
            }
            p.stack.resize(base);
            out->push_back('}');
            return true;
        }
        case ObjType::Closure: {
            auto* c = static_cast<ClosureObj*>(as_obj(w));
            const FuncRec& f = p.runtime().image().func(c->func);
            out->append("<fn ");
            out->append(p.runtime().image().str(f.name).str());
            out->push_back('>');
            return true;
        }
        case ObjType::Pap: out->append("<partial application>"); return true;
        case ObjType::Native: {
            auto* nat = static_cast<NativeObj*>(as_obj(w));
            out->append("<fn ");
            std::string tmp;
            stringify_into(p, nat->name, &tmp, false, depth + 1);
            out->append(tmp);
            out->push_back('>');
            return true;
        }
        case ObjType::Module: {
            out->append("<module ");
            std::string tmp;
            stringify_into(p, static_cast<ModuleObj*>(as_obj(w))->name, &tmp, false, depth + 1);
            out->append(tmp);
            out->push_back('>');
            return true;
        }
        case ObjType::ErrorBox: {
            // Kept on the value stack across the first force, and the payload
            // read from there after it, since that force may move the box.
            out->append("<error ");
            p.stack.push_back(w);
            std::string tmp;
            if (!stringify_into(p, static_cast<ErrorObj*>(as_obj(p.stack.back()))->kind, &tmp,
                                true, depth + 1)) {
                if (!p.force_blocked) p.stack.pop_back();
                return false;
            }
            out->append(tmp);
            out->push_back(' ');
            tmp.clear();
            Value payload = static_cast<ErrorObj*>(as_obj(p.stack.back()))->payload;
            p.stack.pop_back();
            stringify_into(p, payload, &tmp, false, depth + 1);
            out->append(tmp);
            out->push_back('>');
            return true;
        }
        case ObjType::Pid: {
            char buf[48];
            std::snprintf(buf, sizeof buf, "<process %" PRIu64 ">",
                          static_cast<PidObj*>(as_obj(w))->id);
            out->append(buf);
            return true;
        }
        case ObjType::Tensor: return tensor_render(p, w, out);
        default: out->append("<value>"); return true;
    }
}

}  // namespace

bool stringify(Process& p, Value v, std::string* out) {
    return stringify_into(p, v, out, false, 0);
}

namespace {

/// `text`, then where the error was raised, a line a place: the lines an
/// uncaught error is reported with.
///
/// Shorter than what `error_trace` answers, in two ways, because this is read
/// by a person looking for their own code. A function that was running and a
/// value it made, side by side, are one place, and are said once. And a run
/// of three or more places in the standard library is one line saying how
/// many: a `console.print!` at the top of a program put six of `std.console`'s
/// and `std.streams`'s functions under every uncaught error, below the line
/// that mattered. The innermost place is always shown -- it is where the error
/// was raised, and `list.head []` raises in `std`. `error.trace!` still
/// answers every place.
std::string with_trace(Process& p, Value err, std::string text) {
    std::vector<TracePlace> places = error_trace(p, err);
    std::vector<const TracePlace*> kept;
    for (const TracePlace& at : places) {
        if (!kept.empty() && kept.back()->name == at.name && kept.back()->path == at.path &&
            kept.back()->line == at.line) {
            continue;
        }
        kept.push_back(&at);
    }
    auto in_std = [](const TracePlace* at) { return at->path.rfind("std/", 0) == 0; };
    for (size_t i = 0; i < kept.size();) {
        size_t run = 0;
        if (i > 0) {
            while (i + run < kept.size() && in_std(kept[i + run])) ++run;
        }
        if (run >= 3) {
            text += "\n    ... " + std::to_string(run) + " more in std";
            i += run;
            continue;
        }
        const TracePlace& at = *kept[i++];
        text += at.made ? "\n    forcing a value made in `" : "\n    in `";
        text += at.name + "`";
        if (at.line != 0) {
            text += " (" + at.path + ":" + std::to_string(at.line) + ":" +
                    std::to_string(at.col) + ")";
        }
    }
    return text;
}

std::string describe_untraced(Process& p, Value err);

}  // namespace

std::string describe_failure(Process& p, Value err) {
    return with_trace(p, err, describe_untraced(p, err));
}

namespace {

std::string describe_untraced(Process& p, Value err) {
    std::string text;
    stringify(p, err, &text);
    Value v = resolve(err);
    if (!is_obj(v, ObjType::ErrorBox)) return text;
    auto* e = static_cast<ErrorObj*>(as_obj(v));
    if (resolve(e->kind) != make_atom(p.runtime().intern_atom("match_error"))) return text;
    // `[value, path, line, col]`, as the lowering of a failed match builds it
    // (see "a failed match" in dreams/src/lower.dr). Anything else -- one raised
    // by hand, or by an older compiler -- is shown as it is.
    Value items[4];
    Value cell = resolve(e->payload);
    for (Value& item : items) {
        if (!is_obj(cell, ObjType::Cons)) return text;
        item = resolve(static_cast<ConsObj*>(as_obj(cell))->head);
        cell = resolve(static_cast<ConsObj*>(as_obj(cell))->tail);
    }
    if (!is_nil(cell) || !is_fixnum(items[2]) || !is_fixnum(items[3])) return text;
    std::string path, shown;
    if (!stringify(p, items[1], &path) || !stringify(p, items[0], &shown)) return text;
    return path + ":" + std::to_string(fixnum_value(items[2])) + ":" +
           std::to_string(fixnum_value(items[3])) + ": no pattern fits " + shown;
}

}  // namespace

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

namespace {

NativeResult bi_spawn(Process& p, Value, Value* args, uint32_t) {
    // args[0] is deliberately unforced: it is the work the new process is
    // meant to do, and forcing it here would do that work on this process.
    uint64_t pid = p.runtime().scheduler()->spawn_from(p, args[0]);
    if (pid == 0) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).error, "cannot spawn: the runtime is shutting down"));
    }
    return NativeResult::ok(p.heap().make_pid(pid));
}

NativeResult bi_join(Process& p, Value, Value* args, uint32_t) {
    if (!is_obj(args[0], ObjType::Pid)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "join! needs a process"));
    }
    uint64_t target = static_cast<PidObj*>(as_obj(args[0]))->id;
    Value out;
    Scheduler::JoinState st = p.runtime().scheduler()->join(p, target, &out);
    switch (st) {
        case Scheduler::JoinState::Ready: return NativeResult::ok(out);
        case Scheduler::JoinState::Failed: return NativeResult::raise(out);
        case Scheduler::JoinState::Blocked: return NativeResult::block();
    }
    return NativeResult::ok(UNIT);
}

NativeResult bi_send(Process& p, Value, Value* args, uint32_t) {
    if (!is_obj(args[0], ObjType::Pid)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "send! needs a process"));
    }
    uint64_t target = static_cast<PidObj*>(as_obj(args[0]))->id;
    // The message must be fully evaluated before it leaves: a thunk carries a
    // frame that points into this process's heap, which the receiver cannot see.
    Value forced;
    if (!force_deep(p, args[1], &forced)) return NativeResult::raise(p.result);
    if (!p.runtime().scheduler()->send(p, target, forced)) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).error,
                                               "no such thread"));
    }
    return NativeResult::ok(UNIT);
}

NativeResult bi_recv(Process& p, Value, Value*, uint32_t) {
    Value out;
    if (p.runtime().scheduler()->receive(p, &out)) return NativeResult::ok(out);
    return NativeResult::block();
}

NativeResult bi_self(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(p.heap().make_pid(p.id()));
}

NativeResult bi_raise(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (is_obj(v, ObjType::ErrorBox)) return NativeResult::raise(v);
    return NativeResult::raise(p.heap().make_error(make_atom(well_known(p.runtime()).error), v));
}

NativeResult bi_defer(Process& p, Value, Value*, uint32_t) {
    return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                         "defer! must be a block statement"));
}

/// `error_new kind payload` -- an error as a value, without raising it.
///
/// An error is a kind and a payload, and until these three existed a program
/// could catch one and learn nothing from it: `catch e` bound a box with no way
/// in, so the only thing to do with a failure was print it. With them, a caught
/// error can be asked what went wrong and a program can raise a *typed* failure
/// of its own -- `raise! (error_new :not_found path)` -- rather than
/// raising a string and hoping the reader parses it.
///
/// Pure, all three: making and reading an error is not an effect. Only raising
/// one is, which is why `raise!` keeps its `!` and these do not.
NativeResult core_error_new(Process& p, Value, Value* args, uint32_t) {
    Value kind = resolve(args[0]);
    if (!is_atom(kind)) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                               "error_new needs an atom for the kind"));
    }
    return NativeResult::ok(p.heap().make_error(kind, args[1]));
}

/// The kind of an error, or `()` for anything else -- so a `match` on the kind
/// needs no type test first.
NativeResult core_error_kind(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::ErrorBox)) return NativeResult::ok(UNIT);
    return NativeResult::ok(static_cast<ErrorObj*>(as_obj(v))->kind);
}

/// The payload of an error, or `()`. Unforced: a payload built lazily by the
/// code that failed stays that way until someone looks.
NativeResult core_error_payload(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::ErrorBox)) return NativeResult::ok(UNIT);
    return NativeResult::ok(static_cast<ErrorObj*>(as_obj(v))->payload);
}

// Only the decision is strict. The predicate decides how much of the value
// must be inspected; the runtime must not force anything else (even on failure).
// This is a partial pure operation, like indexing or arithmetic on wrong types.
// The value is handed back to be entered, not as an answer: it is a lazy
// argument, so a suspension when it was written as anything but a name, and a
// suspension returned as a value reached whatever used it unforced --
// `type_assert ok "why" (k + "!")` printed as `<value>`, and joined into a
// string, raised.
NativeResult bi_type_assert(Process& p, Value, Value* args, uint32_t) {
    if (resolve(args[0]) == TRUE_V) return NativeResult::enter(args[2]);
    return NativeResult::raise(p.heap().make_error(
        make_atom(well_known(p.runtime()).type_error), args[1]));
}

NativeResult bi_type_of(Process& p, Value, Value* args, uint32_t) {
    // The names are interned at startup and looked up by `dream_type`, not
    // built here: this is the cheapest builtin that forces its argument, so
    // it is what strict folds use to force one, and it runs once per element.
    // Interning the name at each call took the atom table's mutex and hashed
    // a string constant twenty million times in a ten-million-element fold.
    const WellKnownAtoms& wk = well_known(p.runtime());
    dream_type t = surface_type(args[0]);
    if (t == DREAM_TYPE_PURE_FN) {
        // Tell the two function types apart via the function record, which is
        // where the compiler recorded the `!` on the name.
        Value v = resolve(args[0]);
        bool impure = true;  // builtins and host natives perform effects
        if (is_obj(v, ObjType::Closure)) {
            auto* c = static_cast<ClosureObj*>(as_obj(v));
            impure = (p.runtime().image().func(c->func).flags & FN_IMPURE) != 0;
        }
        if (impure) t = DREAM_TYPE_IMPURE_FN;
    }
    if (t == DREAM_TYPE_RECORD) {
        return NativeResult::ok(make_atom(p.runtime().intern_atom(record_type_name(p, args[0]))));
    }
    if (unsigned(t) > unsigned(DREAM_TYPE_TRUNK)) t = DREAM_TYPE_UNKNOWN;
    return NativeResult::ok(make_atom(wk.types[t]));
}

NativeResult bi_to_string(Process& p, Value, Value* args, uint32_t) {
    // Its whole point is that it does not fit in a string; rendering it would
    // build the copy the type exists to avoid, and would be capped at 4 GiB
    // while doing so.
    if (is_obj(resolve(args[0]), ObjType::BigStr)) {
        return NativeResult::raise(raise_error(
            p, well_known(p.runtime()).type_error,
            "to_string cannot render a big string: it is a view into the image, larger than a "
            "string may be. Take a `str.slice` of it, or hand the whole of it to `io.write!`."));
    }
    std::string s;
    if (!stringify(p, args[0], &s)) return NativeResult::raise(p.result);
    return NativeResult::ok(p.heap().make_string(s.data(), uint32_t(s.size())));
}

/// `_settled value` -- true, once `value` has been forced all the way down.
///
/// `strict!` without the `!`, for a pure caller that keeps what it makes: a
/// lazy value stored in a persistent table holds the frame that would compute
/// it, and that frame holds the table as it stood when the value was made, so
/// N of them hold N versions of the table (see "A lazy value stored in a map
/// pins the map it was made in"). Forcing is pure wherever the value is, which
/// is why `to_string` could do this job before; it did it by rendering a
/// string nobody read, and without vouching, so a deep lazy chain ran in one
/// nursery no collection could touch -- a gigabyte of it, checking std's
/// tests, before the heap limit stopped it.
///
/// It answers a bool rather than the value because an argument is not
/// evaluated: `record (settled what)` would suspend the call and change
/// nothing, and a condition is the one place the language must evaluate. It
/// vouches as `bi_strict` does, and for the same reason it is safe to.
NativeResult bi_settled(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    Value out;
    if (!force_deep(p, args[0], &out)) return NativeResult::raise(p.result);
    return NativeResult::ok(make_bool(true));
}

/// `_forced value` -- true. Its argument is forced to weak head normal form
/// by the machine before the call (mask 1), which is the whole of what it is
/// for: `lazy.forced`, said where a pure program carries a table and must not
/// let it become a chain of suspended puts. It was `type_of x != :nothing` and
/// `len m >= 0`, which cost the same and read as questions nobody was asking;
/// written as a Dream function with a strict parameter it read well and cost a
/// self-compile 2.3%, a frame and an argument's suspension at every node.
NativeResult bi_forced(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_bool(true));
}

NativeResult bi_len(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    int64_t n = 0;
    if (is_nil(v)) {
        n = 0;
    } else if (is_obj(v, ObjType::Str)) {
        n = static_cast<StrObj*>(as_obj(v))->len;
    } else if (is_obj(v, ObjType::BigStr)) {
        // A fixnum is 63 bits, so a payload would have to be four exabytes
        // before this could not say how long it is.
        n = int64_t(static_cast<BigStrObj*>(as_obj(v))->len);
    } else if (is_obj(v, ObjType::Array)) {
        n = static_cast<ArrayObj*>(as_obj(v))->len;
    } else if (is_obj(v, ObjType::Record)) {
        n = static_cast<RecordObj*>(as_obj(v))->len;
    } else if (is_obj(v, ObjType::Trunk)) {
        n = trunk_length(v);
    } else if (is_obj(v, ObjType::Map)) {
        n = static_cast<MapObj*>(as_obj(v))->count;
    } else if (is_obj(v, ObjType::Tensor)) {
        n = tensor_len(v);
    } else if (is_obj(v, ObjType::Cons)) {
        Value cur = v;
        for (;;) {
            Value w;
            if (!force_whnf(p, cur, &w)) return NativeResult::raise(p.result);
            if (!is_obj(w, ObjType::Cons)) break;
            ++n;
            cur = static_cast<ConsObj*>(as_obj(w))->tail;
        }
    } else {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "len needs a sized value"));
    }
    return NativeResult::ok(make_fixnum(n));
}

// Order defines the builtin id and must match the compiler's table.
/// `strict! e` -- evaluate `e` all the way down, then hand it back.
///
/// Laziness is the default, and usually right, but it has one sharp edge:
/// an effect in a lazy position does not happen until something forces it, and
/// "something" may be much later or never. `list.map (fn x -> spawn! ..) xs`
/// builds a list of *thunks*, and the spawns only run as each element is
/// forced -- which serializes the very thing that was meant to run at once.
///
/// `strict!` is the way to say "now". It forces deeply rather than to weak
/// head normal form, because forcing the list without forcing its elements
/// would leave the effects exactly where they were.
///
/// It is impure by name, which is right: forcing is when effects happen.
///
/// And it vouches (`VouchesForGc`), which of all the natives here it is the
/// one that most had to. `strict!` is how a program says "do the whole of this
/// now", so the work underneath it is not a detail of one native -- it is the
/// program. Without the vouch none of it could reach a safepoint: a benchmark
/// folding ten million elements inside `strict!` grew a 2.3 GB nursery that no
/// collection could touch, because the rule is that a nested force may collect
/// only when every C++ frame above it has said its locals survive one. This
/// frame's do: `args` is read before the force and never again, and everything
/// the force needs is on the value stack or in the process. It is the same
/// claim `str_of_chars` and `str.concat_all` already make, reached
/// through the same audited chain -- `run_process`, `step_eval`,
/// `resume_native` -- and it is what `force_deep`'s map case was rewritten for.
NativeResult bi_strict(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    Value out;
    if (!force_deep(p, args[0], &out)) {
        // Either the value raised, or a blocking operation inside it gave up
        // and asked to be parked. `resume_native` turns the second case into a
        // Block; here it only has to not swallow the first.
        return NativeResult::raise(p.result);
    }
    return NativeResult::ok(out);
}

// --- what `match` compiles to ------------------------------------------------
//
// These exist so pattern matching needs no new opcodes and no import: a
// builtin is resolved by id, so `match` works in a module that imports
// nothing. A pattern forces only as much as deciding takes -- `[x, ..rest]`
// looks at the first cell and neither `x` nor `rest` -- so each hands back the
// piece it found without forcing it, and the machine forces the piece if and
// when something wants it (`NativeResult::enter`).
//
// They used to force it themselves, which is not the same thing. Forcing inside
// a native runs a nested machine loop on the C++ stack, and a piece whose own
// evaluation reads through another pattern nests again: a list rebuilt a few
// hundred thousand times from the rest of the one before was a segmentation
// fault the first time its end was read. Their arguments arrive forced by the
// same machine, through the strictness mask, rather than by a force of their own.

NativeResult bi_match_is_cons(Process&, Value, Value* args, uint32_t) {
    return NativeResult::ok(make_bool(is_obj(resolve(args[0]), ObjType::Cons)));
}

NativeResult bi_match_head(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Cons)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "match_head needs a list cell"));
    }
    return NativeResult::enter(static_cast<ConsObj*>(as_obj(v))->head);
}

NativeResult bi_match_tail(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Cons)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "match_tail needs a list cell"));
    }
    return NativeResult::enter(static_cast<ConsObj*>(as_obj(v))->tail);
}

NativeResult bi_match_at(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    Value i = resolve(args[1]);
    // A constructor pattern's fields are read by position, as an array
    // pattern's elements are, once `_record_is` has said what it is.
    if (is_obj(v, ObjType::Record) && is_fixnum(i)) {
        auto* r = static_cast<RecordObj*>(as_obj(v));
        int64_t k = fixnum_value(i);
        if (k < 0 || k >= r->len) {
            return NativeResult::raise(raise_error(p, p.runtime().intern_atom("out_of_bounds"),
                                                   "match_at is outside the record"));
        }
        return NativeResult::enter(r->items()[k]);
    }
    if (!is_obj(v, ObjType::Array) || !is_fixnum(i)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "match_at needs an array"));
    }
    auto* a = static_cast<ArrayObj*>(as_obj(v));
    int64_t k = fixnum_value(i);
    if (k < 0 || k >= a->len) {
        return NativeResult::raise(raise_error(p, p.runtime().intern_atom("out_of_bounds"),
                                               "match_at is outside the array"));
    }
    return NativeResult::enter(a->items()[k]);
}

/// `[value]` when the key is there, `[]` when it is not -- one call rather
/// than a `has` followed by a `get`, so the map is looked up once and there is
/// no sentinel that a real value could collide with. The value goes into the
/// cell as the map holds it, and `match_head` hands it on unforced.
NativeResult bi_match_key(Process& p, Value, Value* args, uint32_t) {
    Value m = resolve(args[0]);
    if (!is_obj(m, ObjType::Map)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "match_key needs a map"));
    }
    Value found;
    if (!map_lookup(p, m, args[1], &found)) return NativeResult::ok(NIL);
    return NativeResult::ok(p.heap().make_cons(found, NIL));
}

}  // namespace

// External linkage, because `builtin_def` is inline in the header now -- see
// the note there. The entries stay next to the functions they name.
// ---------------------------------------------------------------------------
// std.math
// ---------------------------------------------------------------------------

namespace {

NativeResult math_sqrt(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    double d;
    if (is_obj(v, ObjType::Float)) {
        d = static_cast<FloatObj*>(as_obj(v))->value;
    } else if (bigint::is_integer(v)) {
        d = bigint::to_double(v);
    } else {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "sqrt needs a number"));
    }
    return NativeResult::ok(p.heap().make_float(std::sqrt(d)));
}

/// A number argument as a double, for the functions of a real variable.
bool math_arg(Process& p, Value v, const char* what, double* d, Value* err) {
    v = resolve(v);
    if (is_obj(v, ObjType::Float)) {
        *d = static_cast<FloatObj*>(as_obj(v))->value;
        return true;
    }
    if (bigint::is_integer(v)) {
        *d = bigint::to_double(v);
        return true;
    }
    *err = raise_error(p, well_known(p.runtime()).type_error,
                       std::string(what) + " needs a number, not " + describe(p, v));
    return false;
}

/// `exp`, `log`, `sin`, `cos`, `tan`, `tanh`, `atan`: a float of a number,
/// by the native's `user` field. What a learning-rate schedule or a
/// statistic computes on one number, where a tensor would be ceremony.
NativeResult math_real(Process& p, Value callee, Value* args, uint32_t) {
    static const char* const names[] = {"exp", "log", "sin", "cos", "tan", "tanh", "atan"};
    const uint32_t fn = static_cast<uint32_t>(static_cast<NativeObj*>(as_obj(callee))->user);
    double d;
    Value err;
    if (!math_arg(p, args[0], names[fn], &d, &err)) return NativeResult::raise(err);
    double r;
    switch (fn) {
        case 0: r = std::exp(d); break;
        case 1: r = std::log(d); break;
        case 2: r = std::sin(d); break;
        case 3: r = std::cos(d); break;
        case 4: r = std::tan(d); break;
        case 5: r = std::tanh(d); break;
        default: r = std::atan(d); break;
    }
    return NativeResult::ok(p.heap().make_float(r));
}

/// `pow x y`: `x` to the power `y`, as a float.
NativeResult math_pow(Process& p, Value, Value* args, uint32_t) {
    double x, y;
    Value err;
    if (!math_arg(p, args[0], "pow", &x, &err) || !math_arg(p, args[1], "pow", &y, &err))
        return NativeResult::raise(err);
    return NativeResult::ok(p.heap().make_float(std::pow(x, y)));
}

NativeResult math_abs(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (is_fixnum(v)) {
        int64_t n = fixnum_value(v);
        return NativeResult::ok(make_integer(p, n < 0 ? -n : n));
    }
    if (bigint::is_big(v)) return NativeResult::ok(bigint::abs(p.heap(), v));
    if (is_obj(v, ObjType::Float)) {
        return NativeResult::ok(
            p.heap().make_float(std::fabs(static_cast<FloatObj*>(as_obj(v))->value)));
    }
    return NativeResult::raise(
        raise_error(p, well_known(p.runtime()).type_error, "abs needs a number"));
}

NativeResult math_floor(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (bigint::is_integer(v)) return NativeResult::ok(v);
    if (is_obj(v, ObjType::Float)) {
        Value r = UNIT;
        if (!bigint::from_double(p.heap(), std::floor(static_cast<FloatObj*>(as_obj(v))->value), &r)) {
            return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                                   "floor of an infinity or a NaN is not an integer"));
        }
        return NativeResult::ok(r);
    }
    return NativeResult::raise(
        raise_error(p, well_known(p.runtime()).type_error, "floor needs a number"));
}

// A list's cells store their head and tail as thunks. Both of these *find*
// their half of the cell and hand it back unforced; the machine forces it as
// the continuation of the call, however deep that goes. Forcing it here, as
// the old code did, ran a nested machine loop on the C++ stack, and a tail
// rebuilt a few hundred thousand times from the tail of the one before was a
// segmentation fault the first time the list's length was read.

NativeResult list_tail(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Cons)) {
        return NativeResult::raise(
            raise_error(p, well_known(p.runtime()).type_error, "tail needs a non-empty list"));
    }
    return NativeResult::enter(static_cast<ConsObj*>(as_obj(v))->tail);
}

NativeResult list_cons(Process& p, Value, Value* args, uint32_t) {
    return NativeResult::ok(p.heap().make_cons(args[0], args[1]));
}

NativeResult list_is_empty(Process& p, Value, Value* args, uint32_t) {
    return NativeResult::ok(make_bool(is_nil(resolve(args[0]))));
}

}  // namespace

const NativeDef* ModuleDef::find(const StringRef& member) const {
    for (const auto& m : members) {
        if (member.equals(m.name)) return &m;
    }
    return nullptr;
}

namespace {

NativeResult vm_processes(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(int64_t(p.runtime().live_process_count())));
}

NativeResult vm_reductions(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.total_reductions)));
}

NativeResult vm_collections(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.heap().collections())));
}

/// `vm.share! v` -- `v`, forced all the way down and moved into the runtime's
/// shared area, where every process can read it and none has to copy it.
///
/// The answer is the same value; what changes is what it costs to hand on. A
/// `spawn!` whose work holds a shared table copies a pointer where it copied
/// the table, and a `join!` answering one does the same. It also stops being
/// anyone's to collect, which is why it is for a table built once and read for
/// the rest of the run -- see `SharedArea` for what that means and why it is
/// sound. A value with a suspension left in it after forcing (a closure's
/// captured frame) is refused, loudly, rather than shared by copy.
///
/// It vouches for the collector for the reason `strict!` does: the force is
/// unbounded Dream work. The copy that follows cannot collect, and `out` is
/// not held across anything that can.
NativeResult vm_share(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    Value out;
    std::string why;
    const int64_t copy_start = timeline_on() ? timeline_now() : 0;
    const size_t bytes_before = timeline_on() ? p.runtime().shared().bytes() : 0;
    const bool shared = p.runtime().shared().share(forced, &out, &why);
    if (timeline_on()) {
        timeline_native(p, "share!: copied " + std::to_string(p.runtime().shared().bytes() - bytes_before) +
                               " bytes", copy_start);
    }
    if (!shared) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error, why));
    }
    return NativeResult::ok(out);
}

// --- processes: kill, monitor, and waiting with a deadline ------------------
//
// `spawn!`, `send!`, `recv!` and `join!` are the language's; what follows is
// what `std.proc` builds selective receive, timeouts and supervision out of.
// They are natives of `std.vm` rather than builtins so that adding them needs
// no new compiler: a builtin is a position in the compiler's own table, and
// that table is in the seed.

/// The pid in a process argument, or 0 with `*err` set.
uint64_t pid_arg(Process& p, Value v, const char* who, NativeResult* err) {
    if (is_obj(v, ObjType::Pid)) return static_cast<PidObj*>(as_obj(v))->id;
    *err = NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                           std::string(who) + " needs a process"));
    return 0;
}

/// An integer argument that must fit a fixnum, or false with `*err` set.
bool int_arg(Process& p, Value v, const char* who, int64_t* out, NativeResult* err) {
    if (is_fixnum(v)) {
        *out = fixnum_value(v);
        return true;
    }
    *err = NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                           std::string(who) + " needs an integer"));
    return false;
}

int64_t monotonic_now_ns() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

/// Let go of the timer a timed wait armed. Every way out of a wait but
/// blocking again comes through here, so a timer never outlives the wait.
void drop_timer(Process& p) {
    if (!p.timer_seq) return;
    p.runtime().scheduler()->cancel_timer(p.timer_seq);
    p.timer_seq = 0;
    p.timer_deadline = -1;
}

/// Park until woken, with a timer at `deadline_ns` when it is not negative.
/// Called with `p.sched_mutex` held, which is what makes the condition the
/// caller just tested and the park one step for a waker: a message pushed
/// after the test finds the process still running and leaves a note.
NativeResult park_until(Process& p, int64_t deadline_ns, WaitReason why) {
    if (deadline_ns >= 0 && (p.timer_seq == 0 || p.timer_deadline != deadline_ns)) {
        drop_timer(p);
        p.timer_seq = p.runtime().scheduler()->arm_timer(p.id(), deadline_ns);
        p.timer_deadline = deadline_ns;
    }
    p.park_requested = true;
    p.wait_reason.store(why, std::memory_order_relaxed);
    return NativeResult::block();
}

/// `vm.kill! p reason` -- stop `p`. It ends as a failure of kind `:killed`
/// with `reason` as the payload, at the start of its next slice; it cannot
/// catch that, and nothing after it in its program runs. Answers `:ok`, or
/// raises for something that was never a process. A process can kill itself,
/// in which case this call does not return.
NativeResult vm_kill(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    uint64_t target = pid_arg(p, args[0], "kill!", &err);
    if (!target) return err;
    // Copied into the target's keeping, so it must not hold a suspension
    // pointing back into this heap -- the reason `send!` forces its message.
    Value reason;
    if (!force_deep(p, args[1], &reason)) return NativeResult::raise(p.result);
    Scheduler* sched = p.runtime().scheduler();
    if (!sched->kill(p, target, reason)) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).error, "no such process"));
    }
    if (target == p.id()) {
        // End the slice now. The kill's own wake left a note, so asking to
        // park sends the process straight back to the queue, and the slice
        // that would run it is the one that stops it.
        std::lock_guard<std::mutex> g(p.sched_mutex);
        p.park_requested = true;
        return NativeResult::block();
    }
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// `vm.monitor! p` -- be sent `[:down, p, outcome]` when `p` ends, where
/// `outcome` is `[:ok, value]` or `[:error, e]`. A process already finished is
/// reported at once. Answers `:ok`.
NativeResult vm_monitor(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    uint64_t target = pid_arg(p, args[0], "monitor!", &err);
    if (!target) return err;
    p.runtime().scheduler()->monitor(p, target);
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// `vm.demonitor! p` -- stop monitoring `p`, once. A `[:down, p, ..]` that
/// already arrived is left in the mailbox. Answers `:ok`.
NativeResult vm_demonitor(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    uint64_t target = pid_arg(p, args[0], "demonitor!", &err);
    if (!target) return err;
    p.runtime().scheduler()->demonitor(p, target);
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// `vm.unique! ()` -- an integer no other call of this in this VM has answered.
/// What a request carries so its reply can be told from every other, even one
/// that arrives after the caller stopped waiting for it.
NativeResult vm_unique(Process& p, Value, Value*, uint32_t) {
    static std::atomic<int64_t> next{1};
    return NativeResult::ok(make_integer(p, next.fetch_add(1, std::memory_order_relaxed)));
}

/// `_deadline_in! ms` -- `vm.now_ns!` `ms` milliseconds from now: the deadline
/// `receive! { .., after ms => .. }` waits until.
NativeResult vm_deadline_in(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    int64_t ms;
    if (!int_arg(p, args[0], "after", &ms, &err)) return err;
    return NativeResult::ok(make_integer(p, monotonic_now_ns() + ms * 1000000));
}

/// `vm.mailbox_size! ()` -- messages waiting, received or not looked at.
NativeResult vm_mailbox_size(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(int64_t(p.mailbox.size())));
}

/// `vm.mailbox_peek! i` -- `Result.ok message` for the message at position `i`,
/// left where it is, or `:none` past the end. Positions are stable until this
/// process takes a message: nobody else removes one.
NativeResult vm_mailbox_peek(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    int64_t i;
    if (!int_arg(p, args[0], "mailbox_peek!", &i, &err)) return err;
    Value msg;
    if (i < 0 || !p.mailbox.peek_into(size_t(i), p.heap(), &msg)) {
        return NativeResult::ok(make_atom(p.runtime().intern_atom("none")));
    }
    // `msg` is held across one allocation and no collection: a heap only
    // collects at a safepoint, and an allocation is not one.
    return NativeResult::ok(make_result(p.heap(), p.runtime(), true, msg));
}

/// `vm.mailbox_take! i` -- remove the message at position `i`. Answers `:ok`;
/// raises `:out_of_bounds` when there is none.
NativeResult vm_mailbox_take(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    int64_t i;
    if (!int_arg(p, args[0], "mailbox_take!", &i, &err)) return err;
    if (i < 0 || !p.mailbox.remove_at(size_t(i))) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).out_of_bounds,
                                               "mailbox_take!: no message at that position"));
    }
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// `vm.await_message! n deadline_ns` -- wait until the mailbox holds more than
/// `n` messages, answering `:ok`, or until `vm.now_ns!` reaches `deadline_ns`,
/// answering `:timeout`. A negative deadline waits for ever.
///
/// The deadline is absolute because a blocked native is re-entered from the
/// top on every wake, with the arguments it was first given: "500 ms from now"
/// would start counting again each time.
NativeResult vm_await_message(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    int64_t n, deadline;
    if (!int_arg(p, args[0], "await_message!", &n, &err)) return err;
    if (!int_arg(p, args[1], "await_message!", &deadline, &err)) return err;
    std::lock_guard<std::mutex> g(p.sched_mutex);
    if (int64_t(p.mailbox.size()) > n) {
        drop_timer(p);
        return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
    }
    if (deadline >= 0 && monotonic_now_ns() >= deadline) {
        drop_timer(p);
        return NativeResult::ok(make_atom(well_known(p.runtime()).timeout));
    }
    return park_until(p, deadline, deadline >= 0 ? WaitReason::Timer : WaitReason::Message);
}

/// `vm.sleep_until! deadline_ns` -- park until `vm.now_ns!` reaches it.
/// Answers `:ok`. Messages arriving meanwhile wake the process, which looks at
/// the clock and parks again.
NativeResult vm_sleep_until(Process& p, Value, Value* args, uint32_t) {
    NativeResult err;
    int64_t deadline;
    if (!int_arg(p, args[0], "sleep_until!", &deadline, &err)) return err;
    std::lock_guard<std::mutex> g(p.sched_mutex);
    if (monotonic_now_ns() >= deadline) {
        drop_timer(p);
        return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
    }
    return park_until(p, deadline, WaitReason::Timer);
}

/// `vm.release! v` -- say the program is done with what `vm.share!` gave it.
/// `v` is the answer of a `share!` (or anything inside it). Its region is
/// freed once a census finds nothing anywhere still holding it, so a value
/// still in use simply stays; there is nothing a wrong release can break.
/// Starts that census. Answers `:ok`, or `:not_shared` for a value that is
/// not in the shared area.
NativeResult vm_release(Process& p, Value, Value* args, uint32_t) {
    if (!p.runtime().shared().release(args[0])) {
        return NativeResult::ok(make_atom(p.runtime().intern_atom("not_shared")));
    }
    if (Scheduler* s = p.runtime().scheduler()) s->request_census();
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// `vm.collect_shared! ()` -- run a census again, for regions released while
/// something still held them. Answers `:ok`; `shared_bytes!` shows what it
/// freed once it has.
NativeResult vm_collect_shared(Process& p, Value, Value*, uint32_t) {
    if (Scheduler* s = p.runtime().scheduler(); s && p.runtime().shared().any_condemned()) {
        s->request_census();
    }
    return NativeResult::ok(make_atom(well_known(p.runtime()).ok));
}

/// Bytes in the runtime's shared area.
NativeResult vm_shared_bytes(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.runtime().shared().bytes())));
}

NativeResult vm_heap_bytes(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.heap().bytes_allocated())));
}

NativeResult vm_modules(Process& p, Value, Value*, uint32_t) {
    // Built back to front so the list comes out in image order.
    const Image& img = p.runtime().image();
    Value list = NIL;
    for (uint32_t i = img.module_count(); i-- > 0;) {
        StringRef name = img.str(img.module(i).name);
        list = p.heap().make_cons(p.heap().make_string(name.data, name.len), list);
    }
    return NativeResult::ok(list);
}

// --- introspection ----------------------------------------------------------
//
// What the VM is doing, as ordinary Dream values. This exists because the
// alternative is a debugger and a print statement: when a program stops making
// progress, the question is always "which processes are stuck, and on what",
// and nothing in the language could answer it before.

/// Build a map from key/value pairs, in one place so every report below has
/// the same shape.
Value info_map(Process& p, std::initializer_list<std::pair<const char*, Value>> pairs) {
    Value m = p.heap().make_map(0);
    for (auto& [key, value] : pairs) {
        m = map_insert(p, m, make_atom(p.runtime().intern_atom(key)), value);
    }
    return m;
}

Value atom_of(Process& p, const char* name) {
    return make_atom(p.runtime().intern_atom(name));
}

const char* status_name(ProcStatus s) {
    switch (s) {
        case ProcStatus::Runnable: return "runnable";
        case ProcStatus::Running: return "running";
        case ProcStatus::Waiting: return "waiting";
        case ProcStatus::Finished: return "finished";
        case ProcStatus::Failed: return "failed";
    }
    return "unknown";
}

/// One process, as a map. `waiting_on` is the part worth having: a process
/// parked on a message and one parked on a socket look identical without it.
Value process_info(Process& p, Process& about) {
    ProcStatus st = about.status.load(std::memory_order_relaxed);
    WaitReason reason = about.wait_reason.load(std::memory_order_relaxed);
    int fd = about.wait_fd.load(std::memory_order_relaxed);
    return info_map(
        p, {
               {"id", make_integer(p, int64_t(about.id()))},
               {"status", atom_of(p, status_name(st))},
               {"waiting_on", st == ProcStatus::Waiting ? atom_of(p, wait_reason_name(reason))
                                                        : atom_of(p, "none")},
               {"fd", make_integer(p, int64_t(fd))},
               {"reductions", make_integer(p, int64_t(about.total_reductions))},
               {"heap_bytes", make_integer(p, int64_t(about.heap().bytes_allocated()))},
               {"collections", make_integer(p, int64_t(about.heap().collections()))},
               {"mailbox", make_integer(p, int64_t(about.mailbox.size()))},
               {"failed", make_bool(about.failed)},
           });
}

NativeResult vm_process_list(Process& p, Value, Value*, uint32_t) {
    auto all = p.runtime().all_processes();
    Value list = NIL;
    for (size_t i = all.size(); i-- > 0;) {
        list = p.heap().make_cons(process_info(p, *all[i]), list);
    }
    return NativeResult::ok(list);
}

NativeResult vm_process_info(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    uint64_t id = 0;
    if (is_obj(v, ObjType::Pid)) {
        id = static_cast<PidObj*>(as_obj(v))->id;
    } else if (is_fixnum(v)) {
        id = uint64_t(fixnum_value(v));
    } else {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                               "process_info! needs a process or an id"));
    }
    auto proc = p.runtime().find_process(id);
    if (!proc) return NativeResult::ok(UNIT);
    return NativeResult::ok(process_info(p, *proc));
}

NativeResult vm_scheduler(Process& p, Value, Value*, uint32_t) {
    Scheduler* s = p.runtime().scheduler();
    if (!s) return NativeResult::ok(UNIT);
    return NativeResult::ok(info_map(
        p, {
               {"workers", make_integer(p, int64_t(s->worker_count()))},
               {"idle", make_integer(p, int64_t(s->idle_workers()))},
               {"live", make_integer(p, int64_t(s->live()))},
               {"runnable", make_integer(p, int64_t(s->runnable()))},
               {"queued", make_integer(p, int64_t(s->queued()))},
               // Processes waiting on a descriptor. While this is above zero
               // the system is not deadlocked however idle it looks: the
               // kernel still owes somebody an answer.
               {"io_waiters", make_integer(p, int64_t(s->io_waiters()))},
               {"reductions", make_integer(p, int64_t(s->total_reductions()))},
               {"deadlocked", make_bool(s->deadlocked())},
           }));
}

NativeResult vm_io(Process& p, Value, Value*, uint32_t) {
    auto handles = io_snapshot();
    Value list = NIL;
    for (size_t i = handles.size(); i-- > 0;) {
        const IoHandleInfo& h = handles[i];
        const char* kind = h.kind == HandleKind::File     ? "file"
                         : h.kind == HandleKind::Listener ? "listener"
                                                          : "stream";
        list = p.heap().make_cons(
            info_map(p, {
                            {"handle", make_integer(p, h.handle)},
                            {"fd", make_integer(p, int64_t(h.fd))},
                            {"kind", atom_of(p, kind)},
                            {"busy", make_integer(p, int64_t(h.busy))},
                            {"waiting", h.waiting_pid == 0
                                            ? UNIT
                                            : make_integer(p, int64_t(h.waiting_pid))},
                        }),
            list);
    }
    return NativeResult::ok(list);
}

NativeResult vm_async_io(Process&, Value, Value*, uint32_t) {
    return NativeResult::ok(make_bool(io_async_available()));
}

/// A readable snapshot on stderr, for when a program has stopped making
/// progress and the question is why.
NativeResult vm_dump(Process& p, Value, Value*, uint32_t) {
    std::string out = "--- vm ---\n";
    if (Scheduler* s = p.runtime().scheduler()) {
        out += "scheduler: workers=" + std::to_string(s->worker_count())
             + " idle=" + std::to_string(s->idle_workers())
             + " live=" + std::to_string(s->live())
             + " runnable=" + std::to_string(s->runnable())
             + " queued=" + std::to_string(s->queued())
             + " io_waiters=" + std::to_string(s->io_waiters())
             + (s->deadlocked() ? " DEADLOCKED" : "") + "\n";
    }
    for (auto& proc : p.runtime().all_processes()) {
        ProcStatus st = proc->status.load(std::memory_order_relaxed);
        out += "  process " + std::to_string(proc->id()) + ": " + status_name(st);
        if (st == ProcStatus::Waiting) {
            WaitReason r = proc->wait_reason.load(std::memory_order_relaxed);
            out += " on " + std::string(wait_reason_name(r));
            int fd = proc->wait_fd.load(std::memory_order_relaxed);
            if (r == WaitReason::Io && fd >= 0) out += " fd=" + std::to_string(fd);
        }
        out += " reductions=" + std::to_string(proc->total_reductions)
             + " mailbox=" + std::to_string(proc->mailbox.size()) + "\n";
    }
    for (const IoHandleInfo& h : io_snapshot()) {
        const char* kind = h.kind == HandleKind::File     ? "file"
                         : h.kind == HandleKind::Listener ? "listener"
                                                          : "stream";
        out += "  handle " + std::to_string(h.handle) + ": fd=" + std::to_string(h.fd) + " "
             + kind + " busy=" + std::to_string(h.busy);
        if (h.waiting_pid) out += " waiting=" + std::to_string(h.waiting_pid);
        out += "\n";
    }
    std::fwrite(out.data(), 1, out.size(), stderr);
    std::fflush(stderr);
    return NativeResult::ok(UNIT);
}

}  // namespace

namespace {

// --- measuring -------------------------------------------------------------
//
// A language with no clock can be timed only from outside it, which is no use
// for finding which stage of a long compilation is the slow one. These are the
// smallest set that answers "how long, how much, and how much of it was
// garbage" -- the three questions worth asking before changing anything.

/// Nanoseconds from a monotonic clock. Only differences mean anything: it does
/// not count from any particular moment, and it never goes backwards, which is
/// what makes it the right one to measure with.
NativeResult vm_now_ns(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    // A fixnum is 63 bits, which is 292 years of nanoseconds.
    return NativeResult::ok(make_integer(p, int64_t(ns)));
}

/// Milliseconds since the epoch, for a log line someone has to read.
NativeResult vm_wall_ms(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return NativeResult::ok(make_integer(p, int64_t(ms)));
}

/// Every byte this process has ever allocated, collections included.
NativeResult vm_allocated(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.heap().bytes_total())));
}

/// The largest the live set has been after a collection.
NativeResult vm_peak_bytes(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_integer(p, int64_t(p.heap().bytes_peak())));
}

/// Everything worth knowing about this process at once, so that a measurement
/// takes one call and cannot get its readings from different moments.
NativeResult vm_stats(Process& p, Value, Value*, uint32_t) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return NativeResult::ok(info_map(p, {
        {"now_ns", make_integer(p, int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()))},
        {"reductions", make_integer(p, int64_t(p.total_reductions))},
        {"allocated", make_integer(p, int64_t(p.heap().bytes_total()))},
        {"heap_bytes", make_integer(p, int64_t(p.heap().bytes_allocated()))},
        {"live_bytes", make_integer(p, int64_t(p.heap().bytes_live()))},
        {"peak_bytes", make_integer(p, int64_t(p.heap().bytes_peak()))},
        {"collections", make_integer(p, int64_t(p.heap().collections()))},
        {"minor_collections", make_integer(p, int64_t(p.heap().minor_collections()))},
        {"major_collections", make_integer(p, int64_t(p.heap().major_collections()))},
        {"bytes_promoted", make_integer(p, int64_t(p.heap().bytes_promoted()))},
        {"external_bytes", make_integer(p, int64_t(p.heap().external_bytes()))},
    }));
}

/// Defined further down, next to the error helpers they need.
NativeResult vm_eval_image(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_host_members(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_open_image(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_call_image(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_close_image(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_image_function(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_locate_image(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_image_digest(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_wire_encode(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_share_arenas(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_node_section(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_index_section(Process& p, Value self, Value* args, uint32_t n);
NativeResult vm_wire_decode(Process& p, Value self, Value* args, uint32_t n);
}  // namespace

ModuleDef make_vm_module() {
    ModuleDef def{"std.vm",
                     {
                         {"processes!", 1, 0b1, vm_processes},
                         {"reductions!", 1, 0b1, vm_reductions},
                         {"collections!", 1, 0b1, vm_collections},
                         {"heap_bytes!", 1, 0b1, vm_heap_bytes},
                         {"share!", 1, 0b0, vm_share, 0, true},
                         {"shared_bytes!", 1, 0b1, vm_shared_bytes},
                         {"release!", 1, 0b1, vm_release},
                         {"collect_shared!", 1, 0b1, vm_collect_shared},
                         {"modules!", 1, 0b1, vm_modules},
                         {"host_members", 1, 0b1, vm_host_members},
                         {"async_io", 1, 0b1, vm_async_io},
                         {"processes_info!", 1, 0b1, vm_process_list},
                         {"process_info!", 1, 0b1, vm_process_info},
                         // what `std.proc` builds receive, timeouts and supervision from
                         {"kill!", 2, 0b01, vm_kill},
                         {"monitor!", 1, 0b1, vm_monitor},
                         {"demonitor!", 1, 0b1, vm_demonitor},
                         {"unique!", 1, 0b1, vm_unique},
                         {"mailbox_size!", 1, 0b1, vm_mailbox_size},
                         {"mailbox_peek!", 1, 0b1, vm_mailbox_peek},
                         {"mailbox_take!", 1, 0b1, vm_mailbox_take},
                         {"await_message!", 2, 0b11, vm_await_message},
                         {"sleep_until!", 1, 0b1, vm_sleep_until},
                         {"scheduler!", 1, 0b1, vm_scheduler},
                         {"io!", 1, 0b1, vm_io},
                         {"dump!", 1, 0b1, vm_dump},
                         {"eval_image!", 1, 0b1, vm_eval_image},
                         // An image kept open and called into; see `ImageSession`.
                         {"open_image!", 1, 0b1, vm_open_image},
                         {"call_image!", 4, 0b1111, vm_call_image, 0, true},
                         {"close_image!", 1, 0b1, vm_close_image},
                         // An image as a library; see "Images as libraries".
                         {"image_function", 3, 0b110, vm_image_function, 0},
                         {"pure_image_function", 3, 0b110, vm_image_function, 1},
                         {"locate_image!", 1, 0b0, vm_locate_image},
                         {"image_digest", 1, 0b1, vm_image_digest},
                         // `std.wire`'s format, done here; see "The wire format, natively".
                         {"wire_encode", 1, 0b0, vm_wire_encode, 0, true},
                         {"wire_decode", 1, 0b1, vm_wire_decode},
                         {"wire_pack", 1, 0b0, vm_wire_pack, 0, true},
                         // `opt.optimize_parts`'s walk; see "Sharing a program's parts".
                         {"share_arenas", 1, 0b0, vm_share_arenas, 0, true},
                         // `emit`'s node and kid sections; see "The image's two largest sections".
                         {"node_section", 1, 0b0, vm_node_section, 0, true},
                         {"index_section", 1, 0b0, vm_index_section, 0, true},
                         // `std.regex`'s matcher; see dream/src/regex.cpp.
                         {"regex_run", 4, 0b1111, vm_regex_run},
                         // measuring
                         {"now_ns!", 1, 0b1, vm_now_ns},
                         {"wall_ms!", 1, 0b1, vm_wall_ms},
                         {"allocated!", 1, 0b1, vm_allocated},
                         {"peak_bytes!", 1, 0b1, vm_peak_bytes},
                         {"stats!", 1, 0b1, vm_stats},
                     }};
    // `std.trunk`'s trunks; see trunk.cpp.
    add_trunk_natives(def.members);
    return def;
}

ModuleDef make_math_module() {
    return ModuleDef{"std.math",
                     {
                         {"sqrt", 1, 0b1, math_sqrt},
                         {"abs", 1, 0b1, math_abs},
                         {"floor", 1, 0b1, math_floor},
                         {"exp", 1, 0b1, math_real, 0},
                         {"log", 1, 0b1, math_real, 1},
                         {"sin", 1, 0b1, math_real, 2},
                         {"cos", 1, 0b1, math_real, 3},
                         {"tan", 1, 0b1, math_real, 4},
                         {"tanh", 1, 0b1, math_real, 5},
                         {"atan", 1, 0b1, math_real, 6},
                         {"pow", 2, 0b11, math_pow},
                     }};
}


// ---------------------------------------------------------------------------
// Runtime primitives and language builtins
//
// Everything here is either a primitive the representation hides (a string's
// bytes, a map's buckets) or something that must be a single machine step for
// the rest of the library to be worth writing. Anything that can be written in
// Dream is written in Dream, in mind/std.
// ---------------------------------------------------------------------------

namespace {

StrObj* as_string(Value v) {
    v = resolve(v);
    return is_obj(v, ObjType::Str) ? static_cast<StrObj*>(as_obj(v)) : nullptr;
}

/// The one refusal a big string earns, worded the same everywhere it is made.
///
/// These are the operations that would have to *build* a string out of one --
/// concatenating, searching, decoding to characters. Each would copy the bytes
/// into the heap and each would be capped at the 4 GiB a `StrObj` can count,
/// which between them is the whole reason the payload is not a string. So the
/// error names what a big string can do instead, because a caller who reached
/// for `str.find` wants a way through, not a diagnosis.
NativeResult bigstr_refused(Process& p, const char* what) {
    return NativeResult::raise(raise_error(
        p, well_known(p.runtime()).type_error,
        std::string(what) +
            " cannot take a big string: it is a view into the image, larger than a string may "
            "be. `len`, `str.byte`, `str.slice`, `==` and `io.write!` all work on one without "
            "copying it."));
}

NativeResult type_fail(Process& p, const char* what) {
    return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error, what));
}

/// What a host module provides: `[name, arity]` for each member, in the order
/// the module declares them, or `()` when no module of that name is registered.
/// A variadic member's arity is `-1`, since it has no fixed one.
///
/// The compiler cannot answer this, and that is not an oversight. A host
/// module's member table belongs to the runtime -- `import std.vm` compiles to
/// a lookup that happens while the program runs -- so `dreams` knows such a
/// module by name and by nothing else. `lucid` is the compiler answering an
/// editor, and after a dot on one of these it had nothing to offer at all. It
/// runs *on* the VM that owns the table, so it does not have to guess: it asks.
///
/// Read only by a tool. Nothing in the language needs it, and a program that
/// wanted to reach a member by a computed name still cannot -- this says what
/// the names are, not how to call one.
NativeResult vm_host_members(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Str)) return type_fail(p, "host_members needs a module name");
    auto* s = static_cast<StrObj*>(as_obj(v));
    const ModuleDef* module = p.runtime().find_module(std::string(s->data(), s->len));
    if (module == nullptr) return NativeResult::ok(UNIT);

    // Built back to front so the list comes out in declaration order, which is
    // the order the module was written in and the one a reader expects.
    Value list = NIL;
    for (size_t i = module->members.size(); i-- > 0;) {
        const NativeDef& member = module->members[i];
        Value arity = make_integer(
            p, member.arity == NATIVE_VARIADIC ? int64_t(-1) : int64_t(member.arity));
        Value name = p.heap().make_string(member.name, uint32_t(std::strlen(member.name)));
        list = p.heap().make_cons(p.heap().make_cons(name, p.heap().make_cons(arity, NIL)), list);
    }
    return NativeResult::ok(list);
}

/// Bring a value across from another runtime's heap.
///
/// Only data comes across, which is not a restriction so much as the definition
/// of what a compile-time value is: the answer has to be something a compiler
/// can write into an image, and a closure or a process is not. Atoms are
/// re-interned by name, because an atom is an index into the runtime that made
/// it and the two runtimes have never met.
bool import_across(Process& dest, Runtime& src_rt, Value v, Value* out, int depth) {
    if (depth > 64) return false;
    v = resolve(v);
    if (is_fixnum(v) || v == UNIT || v == NIL || is_bool(v) || is_char(v)) {
        *out = v;
        return true;
    }
    if (is_atom(v)) {
        *out = make_atom(dest.runtime().intern_atom(src_rt.atom_name(uint32_t(imm_payload(v)))));
        return true;
    }
    if (!is_ptr(v)) return false;
    switch (as_obj(v)->type) {
        case ObjType::Float:
            *out = dest.heap().make_float(static_cast<FloatObj*>(as_obj(v))->value);
            return true;
        case ObjType::BigInt: {
            auto* b = static_cast<BigIntObj*>(as_obj(v));
            *out = dest.heap().make_bigint(b->limbs(), b->len, b->neg != 0);
            return true;
        }
        case ObjType::Tensor:
            return tensor_host_copy(dest, v, out);
        case ObjType::Str: {
            auto* s = static_cast<StrObj*>(as_obj(v));
            *out = dest.heap().make_string(s->data(), s->len);
            return true;
        }
        case ObjType::Cons: {
            // Built back to front so the copy is made without recursing down
            // the spine, which for a long list is what would run out of stack.
            std::vector<Value> items;
            for (Value cur = v; is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
                items.push_back(static_cast<ConsObj*>(as_obj(cur))->head);
                if (items.size() > (1u << 24)) return false;
            }
            Value list = NIL;
            for (size_t i = items.size(); i-- > 0;) {
                Value item;
                if (!import_across(dest, src_rt, items[i], &item, depth + 1)) return false;
                list = dest.heap().make_cons(item, list);
            }
            *out = list;
            return true;
        }
        case ObjType::Array: {
            uint32_t n = static_cast<ArrayObj*>(as_obj(v))->len;
            Value arr = dest.heap().make_array(n);
            for (uint32_t i = 0; i < n; ++i) {
                Value item;
                if (!import_across(dest, src_rt, static_cast<ArrayObj*>(as_obj(v))->items()[i],
                                   &item, depth + 1)) {
                    return false;
                }
                static_cast<ArrayObj*>(as_obj(arr))->items()[i] = item;
            }
            *out = arr;
            return true;
        }
        case ObjType::Record: {
            // The id is an atom of the runtime it came from, and is a name
            // again in this one.
            auto* src = static_cast<RecordObj*>(as_obj(v));
            Value id = make_atom(dest.runtime().intern_atom(
                src_rt.atom_name(uint32_t(imm_payload(src->id())))));
            uint32_t n = src->len;
            Value rec = dest.heap().make_record(record_tag(id, src->variant()), n);
            for (uint32_t i = 0; i < n; ++i) {
                Value item;
                if (!import_across(dest, src_rt, static_cast<RecordObj*>(as_obj(v))->items()[i],
                                   &item, depth + 1)) {
                    return false;
                }
                static_cast<RecordObj*>(as_obj(rec))->items()[i] = item;
            }
            *out = rec;
            return true;
        }
        case ObjType::Map: {
            std::vector<std::pair<Value, Value>> entries;
            map_collect(v, entries);
            Value m = dest.heap().make_map(0);
            for (auto& [k, val] : entries) {
                Value ck, cv;
                if (!import_across(dest, src_rt, k, &ck, depth + 1)) return false;
                if (!import_across(dest, src_rt, val, &cv, depth + 1)) return false;
                m = map_insert(dest, m, ck, cv);
            }
            *out = m;
            return true;
        }
        default:
            return false;
    }
}

/// Run an image and answer the value its entry point produced.
///
/// A compiler written in the language it compiles has a problem with
/// compile-time evaluation: working out what an expression comes to means
/// running it, and the only thing that knows how to run Dream is this. So it
/// runs on a fresh VM -- its own runtime, image, heap and scheduler -- which is
/// exactly what the expression would get at run time, and therefore cannot
/// disagree with it. Writing a second evaluator for compile time is how the two
/// come to disagree.
/// Raise with a named kind, the way `std.os` does for its own failures.
NativeResult comp_fail(Process& p, const char* kind, const std::string& message) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom(kind), message));
}

/// Run a process that has already been primed, and bring its answer across.
///
/// Shared by `eval_image!` and `call_image!` because all that separates them
/// is what gets primed -- an image's entry point with no arguments, or a
/// named global with some. Everything after that is the definition of what a
/// compile-time answer *is*: run it to completion, force it all the way down,
/// and copy the data into the heap that asked. There should be one of that,
/// not two that drift.
NativeResult run_compile_time(Process& p, Runtime& rt, Scheduler& sched,
                              const std::shared_ptr<Process>& root) {
    bool clean = sched.run(root);

    // Two different failures, and they used to share one message. Under load
    // `comp_failed` turned up a few runs in twenty-four, and nothing could say
    // whether the expression had raised or its scheduler had given up on it.
    if (root->failed) {
        return comp_fail(p, "comp_failed",
                         "the compile-time expression raised " + describe_failure(*root, root->exit_value));
    }
    if (!clean) {
        return comp_fail(p, "comp_failed",
                         "the compile-time expression stopped making progress: every process in it "
                         "was waiting for something that could not arrive");
    }
    Value deep;
    if (!force_deep(*root, root->exit_value, &deep)) {
        // Forcing the answer raised, and `force_deep` leaves what in `result`:
        // said, as the raise above is, or the author is told only that it did.
        return comp_fail(p, "comp_failed",
                         "the compile-time expression raised " + describe_failure(*root, root->result));
    }
    Value imported;
    if (!import_across(p, rt, deep, &imported, 0)) {
        return comp_fail(p, "comp_failed",
                    "a compile-time expression must produce data, not a function or a process");
    }
    return NativeResult::ok(imported);
}

NativeResult vm_eval_image(Process& p, Value, Value* args, uint32_t) {
    if (p.runtime().confined()) return comp_fail(p, "confined", "a macro cannot run another image");
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Str)) return type_fail(p, "eval_image! needs an image as a string");
    auto* s = static_cast<StrObj*>(as_obj(v));
    std::string bytes(s->data(), s->len);

    Runtime rt(/*owns_host_services=*/false);
    std::string message;
    if (!rt.load_image_bytes(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
                             message)) {
        return comp_fail(p, "bad_image", message);
    }
    uint32_t func = rt.image().entry();
    if (func == NO_NODE) return comp_fail(p, "bad_image", "the image has no entry point");

    // One worker: this runs inside a worker of the outer scheduler, and the
    // work is one expression rather than a program worth parallelising.
    Scheduler sched(rt, 1);
    auto root = sched.create_process();
    Value cl = root->heap().make_closure(func, 0);
    prime_apply(*root, cl, 0);
    return run_compile_time(p, rt, sched, root);
}

/// Load an image and keep it, answering the handle that names it.
///
/// The counterpart of `eval_image!` for the caller that has one image and
/// many questions. See `ImageSession` in dream/src/runtime.hpp for why the
/// handle is an integer rather than an object, and `close_image!` below for
/// what has to happen to it.
NativeResult vm_open_image(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Str)) return type_fail(p, "open_image! needs an image as a string");
    auto* s = static_cast<StrObj*>(as_obj(v));

    if (p.runtime().confined()) return comp_fail(p, "confined", "a macro cannot open another image");
    auto session = std::make_unique<ImageSession>();
    // Copied before it is loaded, and kept for as long as the session is:
    // the image is read where it lies, so these bytes *are* the program.
    session->bytes.assign(s->data(), s->len);
    session->runtime = std::make_unique<Runtime>(/*owns_host_services=*/false);
    // A session is how a compiler runs a program's macros, which it does to
    // check code it has not been asked to run -- see `Runtime::confine`.
    session->runtime->confine();
    std::string message;
    if (!session->runtime->load_image_bytes(
            reinterpret_cast<const uint8_t*>(session->bytes.data()), session->bytes.size(),
            message)) {
        return comp_fail(p, "bad_image", message);
    }
    return NativeResult::ok(
        make_integer(p, int64_t(p.runtime().open_session(std::move(session)))));
}

/// The `FUNC` index of the global `module.name` names, or `NO_NODE`.
///
/// Qualified by module because a global's name is not unique in an image: two
/// modules may each declare `helper`, and a caller that asked by bare name
/// would get whichever was written first. `MODS` is what makes the question
/// answerable at all -- a module record owns a contiguous range of `GLBL`,
/// which is the only statement in the container of which globals are whose.
uint32_t global_in_module(const Image& img, const std::string& module_name,
                          const std::string& name) {
    for (uint32_t mi = 0; mi < img.module_count(); ++mi) {
        const ModuleRec& m = img.module(mi);
        StringRef mn = img.str(m.name);
        if (mn.len != module_name.size() ||
            std::memcmp(mn.data, module_name.data(), mn.len) != 0) {
            continue;
        }
        for (uint32_t i = m.globals_start; i < m.globals_start + m.globals_count; ++i) {
            const GlobalRec& g = img.global(i);
            if (g.kind != GLOBAL_FUNCTION) continue;
            StringRef gn = img.str(g.name);
            if (gn.len == name.size() && std::memcmp(gn.data, name.data(), gn.len) == 0) {
                return g.target;
            }
        }
    }
    return NO_NODE;
}

/// Call `module.name` in an open session, with `args`, and answer what it
/// produced.
///
/// This is what lets an image be a *library* rather than an expression. The
/// arguments cross the runtime boundary as data, by the same rule that governs
/// what comes back -- `import_across` -- which is why a macro can be handed
/// syntax without the syntax being quoted into a program and compiled first.
///
/// Each call gets a scheduler and a process of its own and the session keeps
/// neither. That is not a cost worth avoiding: what was expensive about
/// `eval_image!` was building the image, not starting a process in it. It is
/// also what keeps one call from seeing another's state, which is the only
/// isolation guarantee a compiler can actually use.
NativeResult vm_call_image(Process& p, Value, Value* args, uint32_t) {
    if (!is_fixnum(resolve(args[0]))) return type_fail(p, "call_image! needs a session handle");

    // Forced before anything else is read, and deeply, because `import_across`
    // copies what it is shown and a lazy structure shows it a thunk. This is
    // the one place in this native where the outer heap can collect -- forcing
    // a caller's data is unbounded Dream work, which is what `VouchesForGc`
    // says and why `strict!` says it -- so nothing may be held across it. The
    // handle is a fixnum and the two names are re-read below, from `args`,
    // which the collector rewrites in place.
    Value argv;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[3], &argv)) return NativeResult::raise(p.result);
    }
    if (argv != NIL && !is_obj(argv, ObjType::Cons)) {
        return type_fail(p, "call_image! needs a list of arguments");
    }

    Value hv = resolve(args[0]);
    Value mv = resolve(args[1]);
    Value nv = resolve(args[2]);
    if (!is_obj(mv, ObjType::Str) || !is_obj(nv, ObjType::Str)) {
        return type_fail(p, "call_image! needs a module name and a member name");
    }
    int64_t handle = fixnum_value(hv);
    ImageSession* session = handle < 0 ? nullptr : p.runtime().session(uint64_t(handle));
    if (session == nullptr) return comp_fail(p, "bad_image", "no such open image");
    Runtime& rt = *session->runtime;

    std::string module_name(static_cast<StrObj*>(as_obj(mv))->data(),
                            static_cast<StrObj*>(as_obj(mv))->len);
    std::string name(static_cast<StrObj*>(as_obj(nv))->data(),
                     static_cast<StrObj*>(as_obj(nv))->len);
    uint32_t func = global_in_module(rt.image(), module_name, name);
    if (func == NO_NODE || func >= rt.image().func_count()) {
        return comp_fail(p, "bad_image", "the image has no " + module_name + "." + name);
    }

    Scheduler sched(rt, 1);
    auto root = sched.create_process();

    // Pushed as each arrives, because the inner process's value stack is what
    // its collector walks: an argument held only in a C++ local while the next
    // one is being built is a reference nothing can find. Nothing collects in
    // the *outer* heap from here on -- the force above was the last of it --
    // so walking `argv` a cell at a time is safe.
    uint32_t argc = 0;
    for (Value cur = argv; is_obj(cur, ObjType::Cons);
         cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        Value crossed;
        if (!import_across(*root, p.runtime(), static_cast<ConsObj*>(as_obj(cur))->head, &crossed,
                           0)) {
            return comp_fail(p, "comp_failed",
                             "a compile-time argument must be data, not a function or a process");
        }
        root->stack.push_back(crossed);
        ++argc;
    }

    // A global of arity 0 is a value, and `do_apply` applies the arguments to
    // what it comes to -- so a transformer reached through an alias, which is
    // what a pure nullary global naming another function is, still answers
    // here. A saturated one enters directly.
    Value cl = root->heap().make_closure(func, 0);
    prime_apply(*root, cl, argc);
    return run_compile_time(p, rt, sched, root);
}

/// Close a session, freeing its image, heap and atom table.
///
/// True when the handle named an open session. It is not an error to close one
/// twice -- a compiler that fails partway is entitled to close what it thinks
/// it opened without first working out how far it got -- and a handle is never
/// reused, so a stale one can never name somebody else's session.
NativeResult vm_close_image(Process& p, Value, Value* args, uint32_t) {
    Value hv = resolve(args[0]);
    if (!is_fixnum(hv)) return type_fail(p, "close_image! needs a session handle");
    int64_t handle = fixnum_value(hv);
    bool closed = handle >= 0 && p.runtime().close_session(uint64_t(handle));
    return NativeResult::ok(make_bool(closed));
}

// ---------------------------------------------------------------------------
// Images as libraries
// ---------------------------------------------------------------------------
//
// `std.image` is to a `.dream` image what `std.ffi` is to a C library: the
// image is found and opened the first time something in it is called, kept
// for the life of the runtime, and each function in it is bound once and
// handed to the program as an ordinary function value. Under that it is a
// session (`ImageSession`), the same machinery macro expansion runs on, so the
// rules are a session's: the image runs in a runtime of its own, and what
// crosses in either direction is *data* -- `import_across` -- because a
// closure or a process names things in a heap the other side has never seen.
// Taking functions across is what merging images would buy; see
// docs/dynamic-linking.md.
//
// A call runs on the calling worker, on a one-worker scheduler made for it,
// to completion, and holds the session's lock while it does: one runtime is
// driven by one scheduler at a time.

/// What one descriptor names: a file to read, or bytes already in hand.
struct ImageSource {
    std::string key, label, path;
    const char* bytes = nullptr;
    uint64_t length = 0;
};

const char* const IMAGE_DESCRIPTORS =
    "an image is a path, `[:installed, name]`, `[:payload, name]` or `[:any, images]`";

bool text_of(Value v, std::string* out) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Str)) return false;
    auto* s = static_cast<StrObj*>(as_obj(v));
    out->assign(s->data(), s->len);
    return true;
}

/// The sources a forced descriptor names, in the order to try them; an
/// `[:any, ..]` contributes each of its own. False, with `why`, when the
/// descriptor is not one, or names something that is not there.
bool image_sources(Process& p, Value v, std::vector<ImageSource>* out, std::string* why) {
    v = resolve(v);
    std::string path;
    if (text_of(v, &path)) {
        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(
            std::filesystem::path(std::u8string(path.begin(), path.end())), ec);
        auto u8 = (ec ? std::filesystem::path(path) : canonical).generic_u8string();
        ImageSource s;
        s.path = path;
        s.key = "file:" + std::string(u8.begin(), u8.end());
        s.label = path;
        out->push_back(std::move(s));
        return true;
    }
    std::vector<Value> items;
    for (Value cur = v; is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        items.push_back(resolve(static_cast<ConsObj*>(as_obj(cur))->head));
    }
    std::string tag = items.size() == 2 && is_atom(items[0])
                          ? p.runtime().atom_name(uint32_t(imm_payload(items[0])))
                          : "";
    std::string name;
    if (tag == "installed" && text_of(items[1], &name)) {
        std::vector<std::string> dirs = install_dirs();
        std::string found = installed_image(name, dirs);
        if (found.empty()) {
            std::string where;
            for (const std::string& d : dirs) where += (where.empty() ? "" : ", ") + d;
            *why = "no installed image `" + name + "`" +
                   (dirs.empty() ? std::string(": there is no installation ($MINDV2_PATH, or ~/.mindv2)")
                                 : " in " + where);
            return false;
        }
        return image_sources(p, p.heap().make_string(found.data(), uint32_t(found.size())), out, why);
    }
    if (tag == "payload" && text_of(items[1], &name)) {
        const Image& img = p.runtime().image();
        int64_t index = img.data_index(name);
        if (index < 0) {
            *why = "this image carries no payload called `" + name + "`; build it with `--payload " +
                   name + "=FILE`";
            return false;
        }
        ImageSource s;
        s.bytes = img.data_bytes(uint32_t(index));
        s.length = img.data_length(uint32_t(index));
        s.key = "payload:" + std::to_string(reinterpret_cast<uintptr_t>(s.bytes));
        s.label = "payload `" + name + "`";
        out->push_back(std::move(s));
        return true;
    }
    if (tag == "any") {
        std::string reasons;
        bool any = false;
        for (Value cur = items[1]; is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
            std::string one;
            if (image_sources(p, static_cast<ConsObj*>(as_obj(cur))->head, out, &one)) any = true;
            else reasons += "\n  " + one;
        }
        // Said even when another was found, so that if that one cannot be
        // opened either, the account of what was tried is whole.
        *why = any ? reasons : "none of the images could be found:" + reasons;
        return any;
    }
    *why = IMAGE_DESCRIPTORS;
    return false;
}

/// The session a descriptor opens as -- the one already open, or a new one --
/// or 0 with `why`.
uint64_t open_image_library(Process& p, Value desc, std::string* why) {
    std::vector<ImageSource> sources;
    std::string reasons;
    if (!image_sources(p, desc, &sources, &reasons)) {
        *why = reasons;
        return 0;
    }
    Runtime& rt = p.runtime();
    for (const ImageSource& src : sources) {
        if (uint64_t open = rt.library_session(src.key)) return open;
        auto session = std::make_unique<ImageSession>();
        session->label = src.label;
        if (src.bytes) {
            session->bytes.assign(src.bytes, size_t(src.length));
        } else {
            std::ifstream in(std::filesystem::path(std::u8string(src.path.begin(), src.path.end())),
                             std::ios::binary);
            if (!in) {
                reasons += "\n  " + src.label + (is_file(src.path) ? ": cannot be read" : ": no such file");
                continue;
            }
            session->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        session->runtime = std::make_unique<Runtime>(/*owns_host_services=*/false);
        std::string message;
        if (!session->runtime->load_image_bytes(
                reinterpret_cast<const uint8_t*>(session->bytes.data()), session->bytes.size(), message)) {
            reasons += "\n  " + src.label + ": " + message;
            continue;
        }
        uint64_t handle = rt.open_session(std::move(session));
        uint64_t kept = rt.note_library_session(src.key, handle);
        if (kept != handle) rt.close_session(handle);  // another process opened it first
        return kept;
    }
    *why = sources.size() == 1 && reasons.find('\n', 1) == std::string::npos
               ? "cannot open the image " + reasons.substr(3)
               : "none of the images could be opened:" + reasons;
    return 0;
}

/// The functions an image's root module has, for the message that says the
/// one asked for is not among them.
std::string functions_in(const Image& img, const std::string& module_name) {
    std::string names;
    for (uint32_t mi = 0; mi < img.module_count(); ++mi) {
        const ModuleRec& m = img.module(mi);
        StringRef mn = img.str(m.name);
        if (std::string(mn.data, mn.len) != module_name) continue;
        for (uint32_t i = m.globals_start; i < m.globals_start + m.globals_count; ++i) {
            const GlobalRec& g = img.global(i);
            if (g.kind != GLOBAL_FUNCTION) continue;
            StringRef gn = img.str(g.name);
            names += (names.empty() ? "" : ", ") + std::string(gn.data, gn.len);
        }
    }
    return names;
}

NativeResult image_fail(Process& p, const std::string& message) {
    return comp_fail(p, "image_error", message);
}

/// What a binding does when it is called: the arguments forced and carried
/// across, the function run in the image to completion, and its answer -- or
/// the error it raised -- carried back.
NativeResult image_invoke(Process& p, Value self, Value* args, uint32_t) {
    Runtime::ImageBinding b;
    if (!p.runtime().image_binding(static_cast<NativeObj*>(as_obj(self))->user, &b)) {
        return image_fail(p, "a binding into an image that is not open");
    }
    // The arguments as data, all the way down, before anything is taken
    // across. Written back where the collector can see them.
    for (uint32_t i = 0; i < b.arity; ++i) {
        Value deep;
        if (!force_deep(p, args[i], &deep)) return NativeResult::raise(p.result);
        args[i] = deep;
    }
    ImageSession* session = p.runtime().session(b.session);
    if (session == nullptr) return image_fail(p, "the image `" + b.name + "` was in is closed");
    std::lock_guard<std::mutex> one_at_a_time(session->calls);
    Runtime& rt = *session->runtime;
    Scheduler sched(rt, 1);
    auto root = sched.create_process();
    for (uint32_t i = 0; i < b.arity; ++i) {
        Value crossed;
        if (!import_across(*root, p.runtime(), args[i], &crossed, 0)) {
            return image_fail(p, "argument " + std::to_string(i + 1) + " of `" + b.name +
                                     "` is a function or a process, and only data can be passed "
                                     "to an image");
        }
        root->stack.push_back(crossed);
    }
    prime_apply(*root, root->heap().make_closure(b.func, 0), b.arity);
    bool clean = sched.run(root);

    Value answer = root->exit_value;
    bool raised = root->failed;
    if (!raised && clean) {
        Value deep;
        if (force_deep(*root, answer, &deep)) answer = deep;
        else {
            raised = true;
            answer = root->result;
        }
    }
    if (!clean && !raised) {
        return image_fail(p, "`" + b.name + "` in " + session->label + " did not finish");
    }
    if (raised) {
        // An error is data with a kind, and is raised again here as the
        // caller's own: the image is a library, and a library's errors are
        // its caller's to handle.
        Value err = resolve(answer);
        if (is_obj(err, ObjType::ErrorBox)) {
            auto* e = static_cast<ErrorObj*>(as_obj(err));
            std::string kind = rt.atom_name(uint32_t(imm_payload(resolve(e->kind))));
            Value payload;
            if (!import_across(p, rt, e->payload, &payload, 0)) payload = UNIT;
            Pin kept(p, payload);
            Value k = make_atom(p.runtime().intern_atom(kind));
            return NativeResult::raise(p.heap().make_error(k, kept.get()));
        }
        Value crossed;
        if (import_across(p, rt, err, &crossed, 0)) return NativeResult::raise(crossed);
        return image_fail(p, "`" + b.name + "` in " + session->label + " failed");
    }
    Value out;
    if (!import_across(p, rt, answer, &out, 0)) {
        return image_fail(p, "`" + b.name + "` answered a function or a process, and only data "
                             "comes back from an image");
    }
    return NativeResult::ok(out);
}

/// `image_function lib target arity` and `pure_image_function ..`: open the
/// image if it is not open, find `target` -- `module.member`, or a bare
/// member of the image's root module -- and answer a function of `arity`
/// arguments that calls it. Which of the two is the member's `user`, and
/// decides only the name the function carries, which is the runtime's
/// reading of its purity, as it is for `ffi.function`.
NativeResult vm_image_function(Process& p, Value callee, Value* args, uint32_t) {
    if (p.runtime().confined()) return image_fail(p, "a macro cannot call another image");
    const bool pure = static_cast<NativeObj*>(as_obj(callee))->user == 1;
    Value desc;
    if (!force_deep(p, args[0], &desc)) return NativeResult::raise(p.result);
    std::string target;
    if (!text_of(args[1], &target) || target.empty()) {
        return type_fail(p, "image_function needs the name of a function in the image");
    }
    Value av = resolve(args[2]);
    if (!is_fixnum(av) || fixnum_value(av) < 0) return type_fail(p, "image_function needs an arity");
    uint32_t arity = uint32_t(fixnum_value(av));

    std::string why;
    uint64_t handle = open_image_library(p, desc, &why);
    if (handle == 0) return image_fail(p, why);
    ImageSession* session = p.runtime().session(handle);
    const Image& img = session->runtime->image();

    size_t dot = target.rfind('.');
    StringRef root = img.module_name();
    std::string module_name = dot == std::string::npos ? std::string(root.data, root.len)
                                                       : target.substr(0, dot);
    std::string member = dot == std::string::npos ? target : target.substr(dot + 1);
    uint32_t func = global_in_module(img, module_name, member);
    if (func == NO_NODE || func >= img.func_count()) {
        std::string has = functions_in(img, module_name);
        return image_fail(p, session->label + " has no function `" + module_name + "." + member + "`" +
                                 (has.empty() ? " (and no module `" + module_name + "`)"
                                              : "; `" + module_name + "` has " + has));
    }
    // A function of fewer parameters may answer one that takes the rest, and
    // a value global is applied to whatever it comes to; but one that wants
    // more than it is given would answer a partial application, which cannot
    // come back.
    uint32_t takes = img.func(func).arity;
    if (takes > arity) {
        auto count = [](uint32_t n) {
            return std::to_string(n) + (n == 1 ? " argument" : " arguments");
        };
        return image_fail(p, "`" + module_name + "." + member + "` in " + session->label + " takes " +
                                 count(takes) + ", and the binding gives it " + std::to_string(arity));
    }
    Runtime::ImageBinding b;
    b.session = handle;
    b.func = func;
    b.arity = arity;
    b.name = module_name + "." + member;
    uint64_t id = p.runtime().add_image_binding(b);
    std::string name = pure ? member : (member.size() && member.back() == '!' ? member : member + "!");
    Value name_v = p.heap().make_string(name.data(), uint32_t(name.size()));
    // Dream applies a function of nothing to unit, so one of no arguments
    // takes a unit it ignores -- as a C function of none does.
    return NativeResult::ok(make_native(p, image_invoke, name_v, arity == 0 ? 1 : arity, 0, id));
}

/// `locate_image! lib`: the file an image descriptor names, or why there is
/// none, as `Result.ok label` or `Result.error why`. It opens the image, which is
/// O(1) once read, and is what using it would do anyway.
NativeResult vm_locate_image(Process& p, Value, Value* args, uint32_t) {
    Value desc;
    if (!force_deep(p, args[0], &desc)) return NativeResult::raise(p.result);
    std::string why;
    uint64_t handle = open_image_library(p, desc, &why);
    std::string said = handle ? p.runtime().session(handle)->label : why;
    Value text = p.heap().make_string(said.data(), uint32_t(said.size()));
    return NativeResult::ok(make_result(p.heap(), p.runtime(), handle != 0, text));
}

/// Decode one UTF-8 scalar starting at `i`, advancing it. Invalid bytes are
/// returned as U+FFFD and consume one byte, so decoding always terminates.
///
/// Every char this produces is a Unicode scalar value, which is what a `char`
/// is and what `char_of_code` insists on. The shape of the bytes is not enough
/// for that: `C0 AF` has the shape of a two-byte sequence and spells `/` in
/// more bytes than it needs, `ED A0 80` spells a surrogate, and `F4 90 80 80`
/// spells a codepoint past U+10FFFF. An overlong form is the dangerous one --
/// it lets a `/`, a quote or a NUL through anything that checked the bytes
/// before decoding them -- and all three are ruled out by the same means: the
/// lead byte narrows the range the *second* byte may take (RFC 3629, section 4).
uint32_t utf8_next(const char* data, uint32_t len, uint32_t* i) {
    uint8_t c = uint8_t(data[*i]);
    if (c < 0x80) {
        ++*i;
        return c;
    }
    uint32_t need;
    uint32_t cp;
    uint8_t lo = 0x80;
    uint8_t hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
        need = 1;
        cp = c & 0x1F;
    } else if (c >= 0xE0 && c <= 0xEF) {
        need = 2;
        cp = c & 0x0F;
        if (c == 0xE0) lo = 0xA0;  // below is overlong
        if (c == 0xED) hi = 0x9F;  // above is a surrogate
    } else if (c >= 0xF0 && c <= 0xF4) {
        need = 3;
        cp = c & 0x07;
        if (c == 0xF0) lo = 0x90;  // below is overlong
        if (c == 0xF4) hi = 0x8F;  // above is past U+10FFFF
    } else {
        // A continuation byte with no lead, or a lead (C0, C1, F5..FF) that
        // could only begin an overlong form or something past U+10FFFF.
        ++*i;
        return 0xFFFD;
    }
    // Counted as what is left rather than as `*i + need`, which cannot wrap.
    if (len - *i <= need) {
        ++*i;
        return 0xFFFD;
    }
    for (uint32_t k = 1; k <= need; ++k) {
        uint8_t cc = uint8_t(data[*i + k]);
        if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) {
            ++*i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    *i += need + 1;
    return cp;
}

void utf8_encode(uint32_t cp, std::string* out) {
    if (cp < 0x80) {
        out->push_back(char(cp));
    } else if (cp < 0x800) {
        out->push_back(char(0xC0 | (cp >> 6)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(char(0xE0 | (cp >> 12)));
        out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(char(0xF0 | (cp >> 18)));
        out->push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(char(0x80 | (cp & 0x3F)));
    }
}

// --- strings ---


NativeResult core_str_chars(Process& p, Value, Value* args, uint32_t) {
    if (is_obj(resolve(args[0]), ObjType::BigStr)) return bigstr_refused(p, "str_chars");
    StrObj* s = as_string(args[0]);
    if (!s) return type_fail(p, "str_chars needs a string");
    // Decoded back to front so the list comes out in order without reversing.
    std::vector<uint32_t> cps;
    for (uint32_t i = 0; i < s->len;) cps.push_back(utf8_next(s->data(), s->len, &i));
    Value list = NIL;
    for (size_t k = cps.size(); k-- > 0;) {
        list = p.heap().make_cons(make_char(cps[k]), list);
    }
    return NativeResult::ok(list);
}

NativeResult core_str_of_chars(Process& p, Value, Value* args, uint32_t) {
    std::string out;
    VouchesForGc vouch(p);
    // The same care as `str_of_bytes`: the rest of the list lives on the value
    // stack, because forcing a character can collect and move the cell.
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            p.stack.resize(base);
            return type_fail(p, "str_of_chars needs a list of chars");
        }
        p.stack[base] = w;
        Value head;
        if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &head)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (!is_char(head)) {
            p.stack.resize(base);
            return type_fail(p, "str_of_chars needs a list of chars");
        }
        utf8_encode(uint32_t(imm_payload(head)), &out);
        p.stack[base] = static_cast<ConsObj*>(as_obj(p.stack[base]))->tail;
    }
    p.stack.resize(base);
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// The inverse of `str_byte`: a list of byte values becomes a string holding
/// exactly those bytes.
///
/// `str_of_chars` cannot do this. It takes Unicode scalars and UTF-8-encodes
/// them, so byte 0x80 comes back out as the two bytes 0xC2 0x80 -- which is
/// right for text and wrong for everything else. A string is a length and a
/// byte buffer, `io.write!` puts those bytes out untouched, and `str_byte`
/// reads them back, so this is the one piece missing before a Dream program
/// can produce binary output rather than only consume it.
NativeResult core_str_of_bytes(Process& p, Value, Value* args, uint32_t) {
    std::string out;
    // Safe against a collection, and so allowed to cause one: see
    // `VouchesForGc`. A list of bytes long enough to be worth this native is
    // long enough that forcing it must be able to collect.
    VouchesForGc vouch(p);
    // Forcing a byte runs Dream code, which can collect, and a collection moves
    // every object it keeps. So the walk carries its position on the value
    // stack -- which the collector updates -- rather than in a C++ local, and
    // re-reads the cell after each force. Getting this wrong needs a list long
    // enough to collect part way along, which is why it stayed hidden until a
    // program compiled itself.
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            // A suspended force keeps its working stack above ours; cutting it
            // back would destroy the work that is waiting to be resumed.
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            p.stack.resize(base);
            return type_fail(p, "str_of_bytes needs a list of integers");
        }
        // The cell itself becomes the position, so the tail stays reachable
        // while the head is forced.
        p.stack[base] = w;
        Value head;
        if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &head)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (!is_fixnum(head)) {
            p.stack.resize(base);
            return type_fail(p, "str_of_bytes needs a list of integers");
        }
        int64_t b = fixnum_value(head);
        if (b < 0 || b > 255) {
            p.stack.resize(base);
            return NativeResult::raise(raise_error(
                p, well_known(p.runtime()).type_error,
                "str_of_bytes needs bytes in 0..255, got " + std::to_string(b)));
        }
        out.push_back(char(uint8_t(b)));
        p.stack[base] = static_cast<ConsObj*>(as_obj(p.stack[base]))->tail;
    }
    p.stack.resize(base);
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// `str_concat parts` -- one string from a list of them.
///
/// Concatenating n strings by `+` is n copies of everything written so far,
/// which is quadratic and is why anything that built a large output a piece at
/// a time -- an image, a rendered diagnostic, a dump -- cost more in copying
/// than in the work it was reporting on. This walks the list once and copies
/// each part once.
///
/// The same care with the collector as `str_of_bytes`: forcing a part runs
/// Dream code, which can collect and move every cell, so the position rides on
/// the value stack rather than in a C++ local.
NativeResult core_str_concat(Process& p, Value, Value* args, uint32_t) {
    std::string out;
    // ...which is also what lets the walk collect. This is the native that
    // made the case for `VouchesForGc`: joining the pieces of the compiler's
    // own image forces a lazy list long enough to allocate 354 MB, and with
    // the heap pinned not one byte of it could be reclaimed until the call
    // returned. Nothing below is held in a C++ local across a force.
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            p.stack.resize(base);
            return type_fail(p, "str_concat needs a list of strings");
        }
        p.stack[base] = w;
        Value head;
        if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &head)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_obj(head, ObjType::BigStr)) {
            p.stack.resize(base);
            return bigstr_refused(p, "str_concat");
        }
        if (!is_obj(head, ObjType::Str)) {
            p.stack.resize(base);
            return type_fail(p, "str_concat needs a list of strings");
        }
        auto* part = static_cast<StrObj*>(as_obj(head));
        out.append(part->data(), part->len);
        p.stack[base] = static_cast<ConsObj*>(as_obj(p.stack[base]))->tail;
    }
    p.stack.resize(base);
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// `_str_interp pieces` -- what `$"..{x}.."` comes to: every piece rendered as
/// `to_string` renders it, joined. The parser hands it the literal text and the
/// expressions in order, as one list, and the `str_interp` opcode calls this.
///
/// It walks the list the way `str_concat` does, with the cursor on the value
/// stack, but does not vouch for the collector: a piece is rendered by
/// `stringify`, which keeps C++ locals across its forces, exactly as
/// `to_string` does. A few pieces of a literal are not the walk the vouch was
/// for.
NativeResult core_str_interp(Process& p, Value, Value* args, uint32_t) {
    std::string out;
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            p.stack.resize(base);
            return type_fail(p, "str_interp needs a list");
        }
        p.stack[base] = w;
        Value head;
        if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &head)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_obj(head, ObjType::BigStr)) {
            p.stack.resize(base);
            return bigstr_refused(p, "a string interpolation");
        }
        if (is_obj(head, ObjType::Str)) {
            auto* part = static_cast<StrObj*>(as_obj(head));
            out.append(part->data(), part->len);
        } else if (!stringify(p, head, &out)) {
            p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        p.stack[base] = static_cast<ConsObj*>(as_obj(p.stack[base]))->tail;
    }
    p.stack.resize(base);
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// `_match_fail value path line col` -- what a failed `match` or destructuring
/// comes to: an error of kind `:match_error` whose payload is `[value, path,
/// line, col]`, raised. The value is carried as it is, unforced.
NativeResult core_match_fail(Process& p, Value, Value* args, uint32_t) {
    Heap& h = p.heap();
    // Built from the end; nothing here collects, so the locals stay good.
    Value list = h.make_cons(args[3], NIL);
    list = h.make_cons(args[2], list);
    list = h.make_cons(args[1], list);
    list = h.make_cons(args[0], list);
    return NativeResult::raise(h.make_error(make_atom(p.runtime().intern_atom("match_error")), list));
}

/// `_error_trace! e` -- where `e` was raised, innermost first: `[:in, name,
/// path, line, col]` for a function that was running and `[:made, ..]` for
/// the one that made a value being forced. `[]` for an error never raised.
/// Impure because the answer is filled in by the raise, so asking before and
/// after are different questions about the same value.
// --- records ---------------------------------------------------------------
//
// What a constructor pattern compiles to, and what `std.record` reads and
// builds a record with. A record is made by the `record_make` opcode, whose id
// is a constant; `_record_of` is the same thing for an id in hand, which is
// what a decoder has.

/// `_record_is id x`: is `x` a record whose id is `id`? Never raises, so a
/// pattern can ask it of anything.
NativeResult core_record_is(Process&, Value, Value* args, uint32_t) {
    Value id = resolve(args[0]);
    Value v = resolve(args[1]);
    return NativeResult::ok(make_bool(is_obj(v, ObjType::Record) &&
                                      static_cast<RecordObj*>(as_obj(v))->id() == id));
}

/// `_record_id x`: the atom naming what `x` is, or `()` when it is no record.
NativeResult core_record_id(Process&, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Record)) return NativeResult::ok(UNIT);
    return NativeResult::ok(static_cast<RecordObj*>(as_obj(v))->id());
}

/// `_record_fields x`: the fields, in order, as an array -- unforced, as they
/// are in the record.
NativeResult core_record_fields(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Record)) return type_fail(p, "_record_fields needs a record");
    uint32_t n = static_cast<RecordObj*>(as_obj(v))->len;
    p.stack.push_back(v);
    Value arr = p.heap().make_array(n);
    auto* src = static_cast<RecordObj*>(as_obj(p.stack.back()));
    auto* dst = static_cast<ArrayObj*>(as_obj(arr));
    for (uint32_t i = 0; i < n; ++i) {
        dst->items()[i] = src->items()[i];
        p.heap().remember_if_old(dst, src->items()[i]);
    }
    p.stack.pop_back();
    return NativeResult::ok(arr);
}

/// `_record_of id variant fields`: a record with that id, a variant's when
/// `variant` is true, whose fields are the array's items, unforced.
NativeResult core_record_of(Process& p, Value, Value* args, uint32_t) {
    Value id = resolve(args[0]);
    Value variant = resolve(args[1]);
    Value items = resolve(args[2]);
    if (!is_atom(id)) return type_fail(p, "_record_of needs an atom for the id");
    if (!is_bool(variant)) return type_fail(p, "_record_of needs a bool for whether it is a variant");
    if (!is_obj(items, ObjType::Array)) return type_fail(p, "_record_of needs an array of fields");
    uint32_t n = static_cast<ArrayObj*>(as_obj(items))->len;
    p.stack.push_back(items);
    Value rec = p.heap().make_record(record_tag(id, truthy(variant)), n);
    auto* src = static_cast<ArrayObj*>(as_obj(p.stack.back()));
    auto* dst = static_cast<RecordObj*>(as_obj(rec));
    for (uint32_t i = 0; i < n; ++i) {
        dst->items()[i] = src->items()[i];
        p.heap().remember_if_old(dst, src->items()[i]);
    }
    p.stack.pop_back();
    return NativeResult::ok(rec);
}

NativeResult core_error_trace(Process& p, Value, Value* args, uint32_t) {
    std::vector<TracePlace> places = error_trace(p, args[0]);
    Heap& h = p.heap();
    // Built from the end, each place pinned while the next is made.
    Pin list(p, NIL);
    Value in = make_atom(p.runtime().intern_atom("in"));
    Value made = make_atom(p.runtime().intern_atom("made"));
    for (size_t i = places.size(); i > 0; --i) {
        const TracePlace& at = places[i - 1];
        Pin path(p, h.make_string(at.path.data(), uint32_t(at.path.size())));
        Pin name(p, h.make_string(at.name.data(), uint32_t(at.name.size())));
        Value item = h.make_cons(make_fixnum(at.col), NIL);
        item = h.make_cons(make_fixnum(at.line), item);
        item = h.make_cons(path.get(), item);
        item = h.make_cons(name.get(), item);
        item = h.make_cons(at.made ? made : in, item);
        p.pins[list.at] = h.make_cons(item, list.get());
    }
    return NativeResult::ok(list.get());
}

NativeResult core_str_slice(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    Value from = resolve(args[1]);
    Value count = resolve(args[2]);
    if (!string_bytes(args[0], &b) || !is_fixnum(from) || !is_fixnum(count)) {
        return type_fail(p, "str_slice needs a string, a start and a length");
    }
    // Clamped rather than raising: slicing past the end is how every loop that
    // walks a string finishes, and an error there would be noise.
    int64_t start = fixnum_value(from);
    int64_t n = fixnum_value(count);
    if (start < 0) start = 0;
    if (uint64_t(start) > b.len) start = int64_t(b.len);
    if (n < 0) n = 0;
    if (uint64_t(start) + uint64_t(n) > b.len) n = int64_t(b.len - uint64_t(start));
    // A slice of a big string is another window on the same bytes, not a copy
    // of them -- which is what lets a program walk a payload it could not hold.
    if (is_obj(resolve(args[0]), ObjType::BigStr)) {
        return NativeResult::ok(p.heap().make_bigstr(b.data + start, uint64_t(n)));
    }
    return NativeResult::ok(p.heap().make_string(b.data + start, uint32_t(n)));
}

NativeResult core_str_find(Process& p, Value, Value* args, uint32_t) {
    if (is_obj(resolve(args[0]), ObjType::BigStr) || is_obj(resolve(args[1]), ObjType::BigStr)) {
        return bigstr_refused(p, "str_find");
    }
    StrObj* hay = as_string(args[0]);
    StrObj* needle = as_string(args[1]);
    Value from = resolve(args[2]);
    if (!hay || !needle || !is_fixnum(from)) {
        return type_fail(p, "str_find needs two strings and a start offset");
    }
    int64_t start = fixnum_value(from);
    if (start < 0) start = 0;
    if (needle->len == 0) {
        return NativeResult::ok(make_fixnum(start <= hay->len ? start : -1));
    }
    if (start + needle->len > hay->len) return NativeResult::ok(make_fixnum(-1));
    for (uint32_t i = uint32_t(start); i + needle->len <= hay->len; ++i) {
        if (std::memcmp(hay->data() + i, needle->data(), needle->len) == 0) {
            return NativeResult::ok(make_fixnum(i));
        }
    }
    return NativeResult::ok(make_fixnum(-1));
}

// --- byte scanning ---
//
// A lexer's inner loop is "advance while the byte is one of these", and said in
// Dream that is a recursion: a call, a `str_byte`, and a comparison chain per
// byte. Skipping the indentation of one line cost a dozen reductions and the
// allocations that go with them, and the compiler's own source is mostly
// indentation and comments. These two answer the whole run at once.
//
// The set is a string because a set of bytes is what it is, and because that
// keeps the pair general: `span s i " \t"` and `span s i alnum_bytes` are the
// same operation. It is turned into a 256-bit table first, so a long set costs
// no more per byte than a short one, and the setup is four stores and one per
// member -- less than a single round trip through the machine.

struct ByteSet {
    uint64_t bits[4] = {0, 0, 0, 0};
    explicit ByteSet(const Bytes& b) {
        for (uint64_t i = 0; i < b.len; ++i) {
            uint8_t c = uint8_t(b.data[i]);
            bits[c >> 6] |= uint64_t(1) << (c & 63);
        }
    }
    bool has(uint8_t c) const { return (bits[c >> 6] >> (c & 63)) & 1; }
};

/// Where a scan starts and stops, or false when the arguments are not a
/// string, a set and an offset.
bool scan_args(Process& p, Value* args, Bytes* text, Bytes* set, uint64_t* from) {
    Value i = resolve(args[1]);
    if (!string_bytes(args[0], text) || !is_fixnum(i) || !string_bytes(args[2], set)) return false;
    int64_t k = fixnum_value(i);
    *from = k < 0 ? 0 : uint64_t(k);
    return true;
}

/// The first byte at or after `i` that is *not* in `set`, or the length of the
/// string when there is none -- so a scan that reaches the end needs no more
/// special-casing than `str_byte` does, which answers -1 there.
NativeResult core_str_span(Process& p, Value, Value* args, uint32_t) {
    Bytes text, set;
    uint64_t i;
    if (!scan_args(p, args, &text, &set, &i)) {
        return type_fail(p, "str_span needs a string, an offset and a set of bytes");
    }
    const ByteSet in(set);
    while (i < text.len && in.has(uint8_t(text.data[i]))) ++i;
    return NativeResult::ok(make_fixnum(int64_t(i)));
}

/// The first byte at or after `i` that *is* in `set`, or the length of the
/// string when there is none.
NativeResult core_str_upto(Process& p, Value, Value* args, uint32_t) {
    Bytes text, set;
    uint64_t i;
    if (!scan_args(p, args, &text, &set, &i)) {
        return type_fail(p, "str_upto needs a string, an offset and a set of bytes");
    }
    // One byte is the common case -- the newline that ends a comment -- and
    // `memchr` is worth several times a byte-at-a-time loop for it.
    if (set.len == 1 && i < text.len) {
        const void* hit = std::memchr(text.data + i, uint8_t(set.data[0]), size_t(text.len - i));
        return NativeResult::ok(
            make_fixnum(int64_t(hit ? uint64_t(static_cast<const char*>(hit) - text.data)
                                    : text.len)));
    }
    const ByteSet in(set);
    while (i < text.len && !in.has(uint8_t(text.data[i]))) ++i;
    return NativeResult::ok(make_fixnum(int64_t(i)));
}

/// `span` run backwards: the start of the run of bytes from `set` that ends at
/// offset `until` (clamped to the length), or `until` itself when the byte
/// before it is not one of them. `str.trim_end` is this and a slice, where it
/// used to decode the string into a list of characters, reverse it, drop the
/// spaces, reverse it again and encode what was left.
NativeResult core_str_span_back(Process& p, Value, Value* args, uint32_t) {
    Bytes text, set;
    uint64_t i;
    if (!scan_args(p, args, &text, &set, &i)) {
        return type_fail(p, "str_span_back needs a string, an offset and a set of bytes");
    }
    if (i > text.len) i = text.len;
    const ByteSet in(set);
    while (i > 0 && in.has(uint8_t(text.data[i - 1]))) --i;
    return NativeResult::ok(make_fixnum(int64_t(i)));
}

/// `str_find` from the other end: the greatest offset at or before `from` where
/// `needle` starts, or -1. A file's extension and a path's last component are
/// both "the last one", which in Dream was a loop of forward finds that
/// rescanned everything after the first match.
NativeResult core_str_rfind(Process& p, Value, Value* args, uint32_t) {
    if (is_obj(resolve(args[0]), ObjType::BigStr) || is_obj(resolve(args[1]), ObjType::BigStr)) {
        return bigstr_refused(p, "str_rfind");
    }
    StrObj* hay = as_string(args[0]);
    StrObj* needle = as_string(args[1]);
    Value from = resolve(args[2]);
    if (!hay || !needle || !is_fixnum(from)) {
        return type_fail(p, "str_rfind needs two strings and a start offset");
    }
    int64_t start = fixnum_value(from);
    if (start < 0 || needle->len > hay->len) return NativeResult::ok(make_fixnum(-1));
    const int64_t last = int64_t(hay->len - needle->len);
    for (int64_t i = start < last ? start : last; i >= 0; --i) {
        if (std::memcmp(hay->data() + i, needle->data(), needle->len) == 0) {
            return NativeResult::ok(make_fixnum(i));
        }
    }
    return NativeResult::ok(make_fixnum(-1));
}

/// The number of characters, decoded exactly as `str_chars` decodes them but
/// without building the list -- `str.length` was a cons per character, all of
/// it garbage the moment it was counted.
NativeResult core_str_char_count(Process& p, Value, Value* args, uint32_t) {
    if (is_obj(resolve(args[0]), ObjType::BigStr)) return bigstr_refused(p, "str_char_count");
    StrObj* s = as_string(args[0]);
    if (!s) return type_fail(p, "str_char_count needs a string");
    int64_t n = 0;
    for (uint32_t i = 0; i < s->len; ++n) utf8_next(s->data(), s->len, &i);
    return NativeResult::ok(make_fixnum(n));
}

/// The string with its ASCII letters upper-cased (`upper` true) or lower-cased,
/// every other byte -- the whole of any multi-byte character included -- left
/// as it was. One copy, where `str.map upper_char` decoded, called a Dream
/// function per character and encoded again.
NativeResult core_str_ascii_case(Process& p, Value, Value* args, uint32_t) {
    if (is_obj(resolve(args[0]), ObjType::BigStr)) return bigstr_refused(p, "str_ascii_case");
    StrObj* s = as_string(args[0]);
    Value upper = resolve(args[1]);
    if (!s || !is_bool(upper)) return type_fail(p, "str_ascii_case needs a string and a bool");
    const bool up = upper == TRUE_V;
    std::string out(s->data(), s->len);
    for (char& c : out) {
        if (up ? (c >= 'a' && c <= 'z') : (c >= 'A' && c <= 'Z')) c = char(c ^ 0x20);
    }
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

NativeResult core_str_byte(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    Value i = resolve(args[1]);
    if (!string_bytes(args[0], &b) || !is_fixnum(i)) {
        return type_fail(p, "str_byte needs a string and an index");
    }
    int64_t k = fixnum_value(i);
    if (k < 0 || uint64_t(k) >= b.len) return NativeResult::ok(make_fixnum(-1));
    return NativeResult::ok(make_fixnum(uint8_t(b.data[k])));
}

/// `str_le v n` -- `n` bytes of `v`, little-endian, as a string.
///
/// A binary format is written a field at a time, and a field written in Dream
/// was a list: `str_of_bytes [v % 256, byte_at v 256, ..]`. Four cons
/// cells, four thunks for the lazily-computed elements, a frame apiece, and a
/// nested machine loop inside `str_of_bytes` to force them again -- about 210
/// bytes of heap to produce four bytes of output, which on one self-compile
/// was 128 MB of garbage, the largest single source in the compiler. This is
/// the same four bytes and no list.
///
/// Little-endian because every machine this runs on is, and two's complement
/// for a negative `v`, which is what an `i64` field in an image holds. `n`
/// beyond eight is a mistake rather than a wider number: a Dream integer is
/// sixty-four bits and a ninth byte could only ever be sign.
NativeResult core_str_le(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    Value n = resolve(args[1]);
    if (!bigint::is_integer(v) || !is_fixnum(n)) {
        return type_fail(p, "str_le needs an integer and a width");
    }
    int64_t width = fixnum_value(n);
    if (width < 0 || width > (int64_t(1) << 30)) {
        return NativeResult::raise(raise_error(
            p, well_known(p.runtime()).type_error,
            "str_le cannot write " + std::to_string(width) + " bytes"));
    }
    // Wider than a word, or a bignum: the two's complement at that width,
    // sign extended or cut, which is the same rule the word case follows.
    if (width > 8 || bigint::is_big(v)) {
        Value s = p.heap().make_string(nullptr, uint32_t(width));
        bigint::to_le_bytes(v, size_t(width),
                            reinterpret_cast<uint8_t*>(static_cast<StrObj*>(as_obj(s))->data()));
        return NativeResult::ok(s);
    }
    // Through `uint64_t` so that a negative value is its two's complement and
    // the shift is defined.
    uint64_t bits = static_cast<uint64_t>(fixnum_value(v));
    // Zeroed because a width of zero leaves the loop below writing nothing,
    // and GCC says so -- but only once a profile has shown it that the zero
    // case is reachable, so an ordinary build is quiet and the trained one is
    // not. Nothing reads the bytes either way, since `make_string` is handed
    // the same width; this is one store to keep the trained build clean.
    char buf[8] = {};
    for (int64_t i = 0; i < width; ++i) buf[i] = char(uint8_t(bits >> (8 * i)));
    return NativeResult::ok(p.heap().make_string(buf, uint32_t(width)));
}

// --- chars ---

NativeResult core_char_code(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_char(v)) return type_fail(p, "char_code needs a char");
    return NativeResult::ok(make_fixnum(int64_t(imm_payload(v))));
}

NativeResult core_char_of_code(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return type_fail(p, "char_of_code needs an integer");
    int64_t c = fixnum_value(v);
    // A surrogate is half of a UTF-16 pair and not a character. Encoded as
    // UTF-8 it is bytes that `str_chars` will not decode back, so letting one
    // through here would break the round trip between the two.
    if (c < 0 || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) {
        return type_fail(p, "that is not a Unicode scalar value");
    }
    return NativeResult::ok(make_char(uint32_t(c)));
}

// --- numbers ---

NativeResult core_to_float(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (is_obj(v, ObjType::Float)) return NativeResult::ok(v);
    if (bigint::is_integer(v)) return NativeResult::ok(p.heap().make_float(bigint::to_double(v)));
    return type_fail(p, "to_float needs a number");
}

/// A float's eight bytes, little-endian.
///
/// A bit pattern will not fit in an integer here -- a fixnum is 63 bits and a
/// double is 64 -- and rebuilding one out of halves invites exactly the sign
/// and rounding mistakes that would make a written image differ from the value
/// that went into it. Bytes are what a writer wants anyway.
NativeResult core_float_bytes(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    double d;
    if (is_obj(v, ObjType::Float)) d = static_cast<FloatObj*>(as_obj(v))->value;
    else if (bigint::is_integer(v)) d = bigint::to_double(v);
    else return type_fail(p, "float_bytes needs a number");
    unsigned char b[8];
    std::memcpy(b, &d, 8);
    return NativeResult::ok(p.heap().make_string(reinterpret_cast<const char*>(b), 8));
}

/// The inverse: the float those eight bytes spell.
NativeResult core_float_of_bytes(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Str)) return type_fail(p, "float_of_bytes needs a string");
    auto* s = static_cast<StrObj*>(as_obj(v));
    if (s->len < 8) return type_fail(p, "float_of_bytes needs eight bytes");
    double d;
    std::memcpy(&d, s->data(), 8);
    return NativeResult::ok(p.heap().make_float(d));
}

/// `_pow base exponent` -- `num.pow`. An integer to a power of zero or more is
/// the exact integer, a bignum once it outgrows a fixnum, by repeated squaring:
/// a Dream loop of `*` was a call per factor, and every package that wanted a
/// power wrote that loop again. A float on either side is `std::pow`'s float,
/// which is what `math.pow` answers for everything.
///
/// An integer to a negative power raises rather than answering a float: the
/// answer's kind follows the base's, which is what lets `num.pow` say
/// `n -> :integer -> n` and a caller holding integers keep integers. The
/// fraction is a float base away.
///
/// An exponent past 64 bits is an answer only for a base of 0, 1 or -1; for any
/// other the result would not fit in memory, and that is said rather than
/// attempted. Allocation never collects (see heap.hpp), so the partial products
/// can sit in C++ locals between multiplications.
NativeResult core_pow(Process& p, Value, Value* args, uint32_t) {
    Value base = resolve(args[0]);
    Value exp = resolve(args[1]);
    if (bigint::is_integer(base) && bigint::is_integer(exp)) {
        if (bigint::compare(exp, make_fixnum(0)) < 0) {
            return type_fail(p, "pow of an integer to a negative power is a fraction: use a float base");
        }
        Heap& h = p.heap();
        uint64_t n;
        if (!bigint::to_uint64(exp, &n)) {
            if (base == make_fixnum(0) || base == make_fixnum(1)) return NativeResult::ok(base);
            if (base == make_fixnum(-1)) {
                bool odd = bigint::rem(h, exp, make_fixnum(2)) != make_fixnum(0);
                return NativeResult::ok(make_fixnum(odd ? -1 : 1));
            }
            return type_fail(p, "pow of an exponent this large would not fit in memory");
        }
        Value acc = make_fixnum(1);
        while (n != 0) {
            if (n & 1) acc = bigint::mul(h, acc, base);
            n >>= 1;
            if (n != 0) base = bigint::mul(h, base, base);
        }
        return NativeResult::ok(acc);
    }
    double x, y;
    Value err;
    if (!math_arg(p, base, "pow", &x, &err) || !math_arg(p, exp, "pow", &y, &err))
        return NativeResult::raise(err);
    return NativeResult::ok(p.heap().make_float(std::pow(x, y)));
}

NativeResult core_to_int(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (bigint::is_integer(v)) return NativeResult::ok(v);
    if (is_obj(v, ObjType::Float)) {
        double d = static_cast<FloatObj*>(as_obj(v))->value;
        // Truncates toward zero, like C. `math.floor` is there for the other
        // rounding, so this one does not have to guess. Exactly, however large:
        // a double past 2^62 is an integer already, and its bignum is it.
        Value r = UNIT;
        if (!bigint::from_double(p.heap(), d, &r)) {
            return type_fail(p, "to_int of an infinity or a NaN is not an integer");
        }
        return NativeResult::ok(r);
    }
    return type_fail(p, "to_int needs a number");
}

NativeResult core_parse_int(Process& p, Value, Value* args, uint32_t) {
    StrObj* s = as_string(args[0]);
    if (!s) return type_fail(p, "parse_int needs a string");
    // Space either side, a sign, and digits -- the whole of the rest, or
    // "12abc" would parse as 12. However many digits: past eighteen the answer
    // is a bignum, which is also how a literal too big for a fixnum is built
    // (`const_int` in dreams/src/lower.dr).
    const char* d = s->data();
    size_t lo = 0, hi = s->len;
    while (lo < hi && std::isspace(uint8_t(d[lo]))) ++lo;
    while (hi > lo && std::isspace(uint8_t(d[hi - 1]))) --hi;
    Value v = UNIT;
    if (!bigint::parse(p.heap(), d + lo, hi - lo, &v)) return NativeResult::ok(UNIT);
    return NativeResult::ok(v);
}

/// `to_existing_atom` -- the atom of this name, if there already is one.
///
/// There is no `to_atom` beside this, and that is the design rather than an
/// omission. An atom is an index into a table that only grows, so a program
/// that interned whatever text arrived on a socket would leak until it died.
/// This asks instead of asserting, and answers `()` where the name has never
/// been written as an atom anywhere in the program -- which, since the image's
/// atoms are interned at load, means the program does not mention it.
///
/// The answer is monotone and that is what makes it usable: the table never
/// forgets, so a name that is an atom stays one, and a `()` can only become an
/// atom if something else creates it -- loading another image, or a value
/// copied in from another VM. Nothing here can turn an atom back into a `()`.
///
/// A big string is accepted for the reason `len` and `str.byte` accept one:
/// this reads the bytes and builds nothing. The cost is hashing the bytes it
/// was given, so asking it about a megabyte is a megabyte of hashing to be
/// told `()`.
NativeResult core_to_existing_atom(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return type_fail(p, "to_existing_atom needs a string");
    uint32_t id = 0;
    if (!p.runtime().find_atom(std::string_view(b.data, b.len), &id)) {
        return NativeResult::ok(UNIT);
    }
    return NativeResult::ok(make_atom(id));
}

NativeResult core_parse_float(Process& p, Value, Value* args, uint32_t) {
    StrObj* s = as_string(args[0]);
    if (!s) return type_fail(p, "parse_float needs a string");
    std::string text(s->data(), s->len);
    size_t used = 0;
    double v = 0;
    try {
        v = std::stod(text, &used);
    } catch (...) {
        return NativeResult::ok(UNIT);
    }
    while (used < text.size() && std::isspace(uint8_t(text[used]))) ++used;
    if (used != text.size()) return NativeResult::ok(UNIT);
    return NativeResult::ok(p.heap().make_float(v));
}

// --- arrays ---

NativeResult core_array_new(Process& p, Value, Value* args, uint32_t) {
    Value n = resolve(args[0]);
    if (!is_fixnum(n) || fixnum_value(n) < 0) return type_fail(p, "array_new needs a length");
    Value arr = p.heap().make_array(uint32_t(fixnum_value(n)));
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    for (uint32_t i = 0; i < a->len; ++i) a->items()[i] = args[1];
    return NativeResult::ok(arr);
}



NativeResult core_array_of_list(Process& p, Value, Value* args, uint32_t) {
    // Two passes: count, then fill. The elements stay lazy.
    //
    // The spine is forced here, and whatever made it lazily runs underneath:
    // `array.of_list (list.reverse acc)` at the end of a pass is the rest of
    // the pass. So it vouches (`VouchesForGc`), as `strict!` does. Unvouched,
    // that work ran with nothing able to collect -- 808 MB of a self-compile,
    // in one nursery per call, and how much depended on what an unrelated
    // `if` elsewhere had happened to force first (see "Strict builtins and
    // pinned forces" in docs/notes/vm-performance.md). Every head waits on the
    // value stack, and `cur` is only ever handed straight to the force, so a
    // collection underneath loses nothing.
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    Value cur = args[0];
    for (;;) {
        Value w;
        if (!force_whnf(p, cur, &w)) {
            // A suspended force keeps its working stack above ours; cutting it
            // back would destroy the work that is waiting to be resumed.
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) {
            p.stack.resize(base);
            return type_fail(p, "array_of_list needs a list");
        }
        auto* c = static_cast<ConsObj*>(as_obj(w));
        p.stack.push_back(c->head);
        cur = c->tail;
    }
    uint32_t n = uint32_t(p.stack.size() - base);
    Value arr = p.heap().make_array(n);
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    for (uint32_t i = 0; i < n; ++i) a->items()[i] = p.stack[base + i];
    p.stack.resize(base);
    return NativeResult::ok(arr);
}

/// `_indexed`: a list read into an array, and anything else as it is. It is
/// never written in a program: the supercompiler puts it where a loop reads a
/// list it is handed only by position and has measured it already, so that
/// each read is constant time rather than a walk (dreams/src/hoist.dr). What
/// it is handed may not be a list -- the loop's `.[i]` and `len` mean the same
/// of an array or a string -- and then nothing changes. It forces the spine,
/// which is why the compiler only puts it after the loop's own `len`.
NativeResult core_indexed(Process& p, Value self, Value* args, uint32_t argc) {
    Value w;
    {
        VouchesForGc vouch(p);
        if (!force_whnf(p, args[0], &w)) return NativeResult::raise(p.result);
    }
    if (!is_nil(w) && !is_obj(w, ObjType::Cons)) return NativeResult::ok(w);
    Value list = w;
    return core_array_of_list(p, self, &list, argc);
}

// --- what the supercompiler writes instead ---------------------------------------
//
// Builtins no program names: the supercompiler puts them where a loop was
// written the slow way (dreams/src/accum.dr, dreams/src/hoist.dr). Each means
// what the code it replaces means -- the same values, forced no further --
// and where it cannot be sure of that it says so, and the code as written
// runs instead.

/// `_list_drop n xs`: `xs` past its first `n` cells, as `list.drop` is,
/// forcing those cells and no element. The supercompiler writes it where an
/// aggregate of a growing prefix (`maximum (take (i + 1) xs)`) is carried from
/// one step to the next: the rest past the prefix is where the next element
/// comes from, and `take n` forced the same cells to make the prefix.
NativeResult core_list_drop(Process& p, Value, Value* args, uint32_t) {
    if (!is_fixnum(args[0])) return type_fail(p, "_list_drop needs a count");
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[1]);
    for (int64_t n = fixnum_value(args[0]); n > 0; --n) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (is_nil(w)) { p.stack[base] = w; break; }
        if (!is_obj(w, ObjType::Cons)) { p.stack.resize(base); return type_fail(p, "_list_drop needs a list"); }
        p.stack[base] = static_cast<ConsObj*>(as_obj(w))->tail;
    }
    Value rest = p.stack[base];
    p.stack.resize(base);
    return NativeResult::ok(rest);
}

// A sorting heap is what `list.sort q` would answer, kept for a loop that
// sorts its queue each step only to take the smallest and hand the rest on
// with something added: `drain (list.tail (list.sort q) + [x])`. Each step
// was a sort; with this it is a pop and a push.
//
// `#[mode, base, pending, root, size, seq]`. `pending` is the pieces handed
// on since the heap was last settled, newest first, each a list not yet
// read. In mode 0 the heap is still the list it was made from, `base`,
// untouched, so asking whether it is empty forces only its first cell, as
// `list.is_empty q` does. In mode 1 it is a leftist heap, `root`, holding
// `size` elements, each numbered in the order the sort would have met it
// (`seq`), and ordered by `compare` and then by that number: a stable
// sort's order, so the smallest is the head `list.sort` would answer. In
// mode 2 it is a list again, `base`, sorted the way `list.sort` sorts it:
// what a heap becomes on meeting an element `compare` does not order
// totally -- a NaN, a map -- for which only the sort itself answers right.
//
// Settling -- reading what is pending, and building the heap from the list
// the first time -- forces every element, which is what the sort it stands
// for does before it compares anything, and happens where that sort would
// have been forced: at a minimum or a pop. It is done in place, with the
// write barrier, since what the heap stands for does not change. A heap
// that is pushed or popped is a new one: the old one is still the queue it
// was.

bool compare_values(Process& p, Value a, Value b, int depth, int* out);

enum { HEAP_MODE, HEAP_BASE, HEAP_PENDING, HEAP_ROOT, HEAP_SIZE, HEAP_SEQ, HEAP_FIELDS };
// A node: `#[element, seq, rank, left, right]`; the empty heap is `()`.
enum { NODE_ELEM, NODE_SEQ, NODE_RANK, NODE_LEFT, NODE_RIGHT, NODE_FIELDS };

static Value heap_field(Value h, int f) { return static_cast<ArrayObj*>(as_obj(h))->items()[f]; }
static void heap_store(Process& p, Value h, int f, Value v) {
    auto* a = static_cast<ArrayObj*>(as_obj(h));
    a->items()[f] = v;
    p.heap().remember_if_old(a, v);
}

static Value make_heap(Process& p, int mode, Value base, Value pending, Value root, int64_t size, int64_t seq) {
    Value h = p.heap().make_array(HEAP_FIELDS);
    auto* a = static_cast<ArrayObj*>(as_obj(h));
    a->items()[HEAP_MODE] = make_fixnum(mode);
    a->items()[HEAP_BASE] = base;
    a->items()[HEAP_PENDING] = pending;
    a->items()[HEAP_ROOT] = root;
    a->items()[HEAP_SIZE] = make_fixnum(size);
    a->items()[HEAP_SEQ] = make_fixnum(seq);
    return h;
}

/// Whether `compare` orders `v` against anything of its kind totally: no
/// NaN, no map, nothing nested too deep to say.
static bool heap_orderly(Value v, int depth) {
    v = resolve(v);
    if (depth > 64) return false;
    if (is_obj(v, ObjType::Float)) return !std::isnan(static_cast<FloatObj*>(as_obj(v))->value);
    if (is_obj(v, ObjType::Cons)) {
        for (Value c = v; ; ) {
            c = resolve(c);
            if (is_nil(c)) return true;
            if (!is_obj(c, ObjType::Cons)) return heap_orderly(c, depth + 1);
            if (!heap_orderly(static_cast<ConsObj*>(as_obj(c))->head, depth + 1)) return false;
            c = static_cast<ConsObj*>(as_obj(c))->tail;
        }
    }
    if (is_obj(v, ObjType::Array)) {
        auto* a = static_cast<ArrayObj*>(as_obj(v));
        for (uint32_t i = 0; i < a->len; ++i) if (!heap_orderly(a->items()[i], depth + 1)) return false;
        return true;
    }
    if (is_obj(v, ObjType::Record)) {
        auto* r = static_cast<RecordObj*>(as_obj(v));
        for (uint32_t i = 0; i < r->len; ++i) if (!heap_orderly(r->items()[i], depth + 1)) return false;
        return true;
    }
    return is_fixnum(v) || bigint::is_big(v) || is_atom(v) || is_char(v) || is_bool(v) || is_unit(v)
        || is_obj(v, ObjType::Str) || is_nil(v);
}

static int64_t node_rank(Value n) { return is_unit(n) ? 0 : fixnum_value(heap_field(n, NODE_RANK)); }

/// Whether node `a` comes before node `b`: smaller, or as small and met first.
static bool node_before(Process& p, Value a, Value b, bool* ok) {
    int cmp = 0;
    if (!compare_values(p, heap_field(a, NODE_ELEM), heap_field(b, NODE_ELEM), 0, &cmp)) { *ok = false; return false; }
    if (cmp != 0) return cmp < 0;
    return fixnum_value(heap_field(a, NODE_SEQ)) < fixnum_value(heap_field(b, NODE_SEQ));
}

static Value make_node(Process& p, Value elem, Value seq, Value left, Value right) {
    if (node_rank(left) < node_rank(right)) std::swap(left, right);
    Value n = p.heap().make_array(NODE_FIELDS);
    auto* a = static_cast<ArrayObj*>(as_obj(n));
    a->items()[NODE_ELEM] = elem;
    a->items()[NODE_SEQ] = seq;
    a->items()[NODE_RANK] = make_fixnum(node_rank(right) + 1);
    a->items()[NODE_LEFT] = left;
    a->items()[NODE_RIGHT] = right;
    return n;
}

/// Two leftist heaps as one, sharing what it does not change. Its depth is
/// the right spines', a logarithm of the size. Nothing here forces.
static Value heap_merge(Process& p, Value a, Value b, bool* ok) {
    if (is_unit(a)) return b;
    if (is_unit(b)) return a;
    if (!node_before(p, a, b, ok)) {
        if (!*ok) return UNIT;
        std::swap(a, b);
    }
    Value right = heap_merge(p, heap_field(a, NODE_RIGHT), b, ok);
    if (!*ok) return UNIT;
    return make_node(p, heap_field(a, NODE_ELEM), heap_field(a, NODE_SEQ), heap_field(a, NODE_LEFT), right);
}

/// The elements of a heap, smallest first, by popping a copy: what the list
/// it stands for, sorted, is.
static bool heap_drain(Process& p, Value root, std::vector<Value>* out) {
    bool ok = true;
    while (!is_unit(root)) {
        out->push_back(heap_field(root, NODE_ELEM));
        root = heap_merge(p, heap_field(root, NODE_LEFT), heap_field(root, NODE_RIGHT), &ok);
        if (!ok) return false;
    }
    return true;
}

/// Sort `items` as `list.sort` does -- stable, by `compare` -- into a list.
static bool sort_as_written(Process& p, std::vector<Value>& items, Value* out) {
    std::vector<uint32_t> order(items.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    bool ok = true;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (!ok) return false;
        int cmp = 0;
        if (!compare_values(p, items[a], items[b], 0, &cmp)) { ok = false; return false; }
        return cmp < 0;
    });
    if (!ok) return false;
    Value list = NIL;
    for (size_t i = order.size(); i-- > 0;) list = p.heap().make_cons(items[order[i]], list);
    *out = list;
    return true;
}

/// Read what is pending into the heap, and the list it was made from the
/// first time, forcing each element in the order the sort would. False,
/// with the reason in `p.result`, when a force raised or the list is not one.
static bool heap_settle(Process& p, size_t at) {
    int mode = int(fixnum_value(heap_field(p.stack[at], HEAP_MODE)));
    if (mode != 0 && is_nil(resolve(heap_field(p.stack[at], HEAP_PENDING)))) return true;
    // Everything to be read, oldest first: the list itself, then each piece.
    std::vector<Value> chunks;
    if (mode == 0) chunks.push_back(heap_field(p.stack[at], HEAP_BASE));
    {
        std::vector<Value> newest_first;
        for (Value c = resolve(heap_field(p.stack[at], HEAP_PENDING)); is_obj(c, ObjType::Cons);
             c = resolve(static_cast<ConsObj*>(as_obj(c))->tail))
            newest_first.push_back(static_cast<ConsObj*>(as_obj(c))->head);
        chunks.insert(chunks.end(), newest_first.rbegin(), newest_first.rend());
    }
    // Each chunk forced whole, as the sort forces its list, and kept on the
    // value stack while the next is.
    const size_t first = p.stack.size();
    for (size_t i = 0; i < chunks.size(); ++i) p.stack.push_back(chunks[i]);
    for (size_t i = 0; i < chunks.size(); ++i) {
        Value w;
        if (!force_deep(p, p.stack[first + i], &w)) return false;
        p.stack[first + i] = w;
    }
    std::vector<Value> fresh;
    for (size_t i = 0; i < chunks.size(); ++i) {
        Value c = resolve(p.stack[first + i]);
        while (is_obj(c, ObjType::Cons)) {
            fresh.push_back(resolve(static_cast<ConsObj*>(as_obj(c))->head));
            c = resolve(static_cast<ConsObj*>(as_obj(c))->tail);
        }
        if (!is_nil(c)) {
            p.stack.resize(first);
            p.result = raise_error(p, well_known(p.runtime()).type_error, "array_of_list needs a list");
            return false;
        }
    }
    p.stack.resize(first);
    Value h = p.stack[at];
    bool orderly = mode != 2;
    for (Value v : fresh) if (orderly && !heap_orderly(v, 0)) orderly = false;
    bool ok = true;
    if (!orderly) {
        // A list from here on, sorted as the program sorts it: what was in
        // the heap, in its order, then what came after it.
        std::vector<Value> items;
        if (mode == 1 && !heap_drain(p, heap_field(h, HEAP_ROOT), &items)) return false;
        if (mode == 2) {
            for (Value c = resolve(heap_field(h, HEAP_BASE)); is_obj(c, ObjType::Cons); c = resolve(static_cast<ConsObj*>(as_obj(c))->tail))
                items.push_back(static_cast<ConsObj*>(as_obj(c))->head);
        }
        items.insert(items.end(), fresh.begin(), fresh.end());
        Value sorted;
        if (!sort_as_written(p, items, &sorted)) return false;
        heap_store(p, h, HEAP_BASE, sorted);
        heap_store(p, h, HEAP_ROOT, UNIT);
        heap_store(p, h, HEAP_PENDING, NIL);
        heap_store(p, h, HEAP_MODE, make_fixnum(2));
        return true;
    }
    Value root = mode == 1 ? heap_field(h, HEAP_ROOT) : UNIT;
    int64_t seq = fixnum_value(heap_field(h, HEAP_SEQ));
    int64_t size = mode == 1 ? fixnum_value(heap_field(h, HEAP_SIZE)) : 0;
    for (Value v : fresh) {
        root = heap_merge(p, root, make_node(p, v, make_fixnum(seq++), UNIT, UNIT), &ok);
        if (!ok) return false;
        ++size;
    }
    heap_store(p, h, HEAP_ROOT, root);
    heap_store(p, h, HEAP_SIZE, make_fixnum(size));
    heap_store(p, h, HEAP_SEQ, make_fixnum(seq));
    heap_store(p, h, HEAP_PENDING, NIL);
    heap_store(p, h, HEAP_BASE, NIL);
    heap_store(p, h, HEAP_MODE, make_fixnum(1));
    return true;
}

/// `_heap_of xs`: the heap `xs` sorts to, nothing read yet.
NativeResult core_heap_of(Process& p, Value, Value* args, uint32_t) {
    return NativeResult::ok(make_heap(p, 0, args[0], NIL, UNIT, 0, 0));
}

/// `_heap_push h piece`: the heap of the list `h` stands for with `piece`, a
/// list, after it. Nothing is read.
NativeResult core_heap_push(Process& p, Value, Value* args, uint32_t) {
    Value h = resolve(args[0]);
    Value pending = p.heap().make_cons(args[1], heap_field(h, HEAP_PENDING));
    h = resolve(args[0]);
    return NativeResult::ok(make_heap(p, int(fixnum_value(heap_field(h, HEAP_MODE))), heap_field(h, HEAP_BASE), pending,
                                      heap_field(h, HEAP_ROOT), fixnum_value(heap_field(h, HEAP_SIZE)),
                                      fixnum_value(heap_field(h, HEAP_SEQ))));
}

/// `_heap_empty h`: whether the list `h` stands for is empty, forcing what
/// `list.is_empty` of it would: its first cell, and nothing in a piece after
/// anything already there.
NativeResult core_heap_empty(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[0]));
    auto fail = [&]() {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    };
    int mode = int(fixnum_value(heap_field(p.stack[base], HEAP_MODE)));
    bool something = false;
    if (mode == 1) something = fixnum_value(heap_field(p.stack[base], HEAP_SIZE)) > 0;
    else {
        Value w;
        if (!force_whnf(p, heap_field(p.stack[base], HEAP_BASE), &w)) return fail();
        // Not a list at all is not empty, as `list.is_empty` says.
        something = !is_nil(w);
    }
    if (!something) {
        // The pieces after it, oldest first, until one has a cell.
        std::vector<Value> newest_first;
        for (Value c = resolve(heap_field(p.stack[base], HEAP_PENDING)); is_obj(c, ObjType::Cons);
             c = resolve(static_cast<ConsObj*>(as_obj(c))->tail))
            newest_first.push_back(static_cast<ConsObj*>(as_obj(c))->head);
        const size_t first = p.stack.size();
        for (auto it = newest_first.rbegin(); it != newest_first.rend(); ++it) p.stack.push_back(*it);
        for (size_t i = first; i < p.stack.size() && !something; ++i) {
            Value w;
            if (!force_whnf(p, p.stack[i], &w)) return fail();
            something = !is_nil(w);
        }
    }
    p.stack.resize(base);
    return NativeResult::ok(make_bool(!something));
}

/// `_heap_min h`: the head of the list `h` stands for, sorted. Asked only of
/// a heap that is not empty.
NativeResult core_heap_min(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[0]));
    if (!heap_settle(p, base)) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    Value h = p.stack[base];
    p.stack.resize(base);
    if (fixnum_value(heap_field(h, HEAP_MODE)) == 2)
        return NativeResult::ok(static_cast<ConsObj*>(as_obj(resolve(heap_field(h, HEAP_BASE))))->head);
    return NativeResult::ok(heap_field(heap_field(h, HEAP_ROOT), NODE_ELEM));
}

/// `_heap_pop h`: the heap of the tail of the list `h` stands for, sorted.
/// Asked only of a heap that is not empty.
NativeResult core_heap_pop(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[0]));
    if (!heap_settle(p, base)) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    Value h = p.stack[base];
    p.stack.resize(base);
    if (fixnum_value(heap_field(h, HEAP_MODE)) == 2)
        return NativeResult::ok(make_heap(p, 2, static_cast<ConsObj*>(as_obj(resolve(heap_field(h, HEAP_BASE))))->tail,
                                          NIL, UNIT, 0, 0));
    bool ok = true;
    Value root = heap_field(h, HEAP_ROOT);
    Value rest = heap_merge(p, heap_field(root, NODE_LEFT), heap_field(root, NODE_RIGHT), &ok);
    if (!ok) return NativeResult::raise(p.result);
    return NativeResult::ok(make_heap(p, 1, NIL, NIL, rest, fixnum_value(heap_field(h, HEAP_SIZE)) - 1,
                                      fixnum_value(heap_field(h, HEAP_SEQ))));
}

/// `_memo_cells n cell`: an array of `n` cells, cell `i` the call `cell i`
/// suspended. The supercompiler writes it for a function whose calls of
/// itself it keeps in a table (`memo_table` in dreams/src/accum.dr): `cell`
/// is the function's body over a flat index, capturing the table it reads.
/// Each cell is the frame that call would have, its argument in slot 0, and
/// a thunk of the body in it -- what a top-level value is (`global_value`) --
/// so a cell costs two objects, and one no call reaches is never run. Nothing
/// is forced here and nothing can collect, so what is made waits on the value
/// stack until the array holding it is made, as `_array_of_list` does.
NativeResult core_memo_cells(Process& p, Value, Value* args, uint32_t) {
    Value cl = resolve(args[1]);
    if (!is_fixnum(args[0]) || fixnum_value(args[0]) < 0 || !is_obj(cl, ObjType::Closure))
        return type_fail(p, "_memo_cells needs a count and a function");
    const FuncRec& f = p.code->func(static_cast<ClosureObj*>(as_obj(cl))->func);
    if (f.arity != 1) return type_fail(p, "_memo_cells needs a function of one argument");
    const int64_t n = fixnum_value(args[0]);
    const size_t base = p.stack.size();
    for (int64_t i = 0; i < n; ++i) {
        Value fr = p.heap().make_frame_filling(cl, f.slots, 1);
        static_cast<FrameObj*>(as_obj(fr))->slots()[0] = make_fixnum(i);
        p.stack.push_back(p.heap().make_thunk(f.body, fr));
    }
    Value arr = p.heap().make_array(uint32_t(n));
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    for (int64_t i = 0; i < n; ++i) a->items()[i] = p.stack[base + size_t(i)];
    p.stack.resize(base);
    return NativeResult::ok(arr);
}

/// Force the spine of the list in `p.stack[at]`, leaving each cell's head on
/// the stack above `base` in order. False, with the reason in `p.result`, if
/// a force raised or the list is not a proper one (`what` names the operator
/// whose error it is).
static bool spill_spine(Process& p, size_t at, const char* what) {
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[at], &w)) return false;
        if (is_nil(w)) return true;
        if (!is_obj(w, ObjType::Cons)) {
            p.result = raise_error(p, well_known(p.runtime()).type_error, what);
            return false;
        }
        auto* c = static_cast<ConsObj*>(as_obj(w));
        p.stack[at] = c->tail;
        p.stack.push_back(c->head);
    }
}

/// `_append_rev a r`: `a + reverse r`. A loop that grew `acc` by `acc + [x]`
/// -- a copy of `acc` per step -- grows `r` by a cons instead, newest first,
/// and this is what `acc` was at the end: `a` copied once, then what was
/// appended in the order it was.
NativeResult core_append_rev(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);   // a
    p.stack.push_back(args[1]);   // r, walked
    if (!spill_spine(p, base + 1, "the left side of a list `+` is not a proper list")) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    if (p.stack.size() == base + 2) {
        Value a = p.stack[base];
        p.stack.resize(base);
        return NativeResult::ok(a);
    }
    // The heads above base + 2 are r's, newest first: consed in that order
    // onto nothing they come out oldest first. The tail is kept on the stack,
    // since forcing `a`'s spine below may collect.
    Value tail = NIL;
    for (size_t i = base + 2; i < p.stack.size(); ++i) tail = p.heap().make_cons(p.stack[i], tail);
    p.stack.resize(base + 1);
    p.stack.push_back(tail);           // base + 1: the tail
    p.stack.push_back(p.stack[base]);  // base + 2: `a`, walked
    if (!spill_spine(p, base + 2, "the left side of a list `+` is not a proper list")) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    // `a + tail`, as `+` makes it: `a`'s spine copied, its end the tail.
    Value out = p.stack[base + 1];
    for (size_t i = p.stack.size(); i-- > base + 3;) out = p.heap().make_cons(p.stack[i], out);
    p.stack.resize(base);
    return NativeResult::ok(out);
}

/// `_rev_onto xs r`: `reverse xs` in front of `r` -- what `acc + xs` puts on
/// the end of `acc`, kept newest first.
NativeResult core_rev_onto(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    if (!spill_spine(p, base, "the right side of a list `+` is not a proper list")) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    Value out = args[1];
    for (size_t i = base + 1; i < p.stack.size(); ++i) out = p.heap().make_cons(p.stack[i], out);
    p.stack.resize(base);
    return NativeResult::ok(out);
}

/// `_str_append_rev a r`: `a` followed by the strings of `r`, oldest first --
/// what `acc` was after a loop of `acc + piece`, each of which copied `acc`.
NativeResult core_str_append_rev(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    p.stack.push_back(args[1]);
    if (!spill_spine(p, base + 1, "a string `+` was handed something that is not a list of pieces")) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    // Each piece forced where the `+` it replaces would have forced it.
    size_t total = 0;
    for (size_t i = base; i < p.stack.size(); ++i) {
        if (i == base + 1) continue;
        Value w;
        if (!force_whnf(p, p.stack[i], &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        if (!is_obj(w, ObjType::Str)) {
            p.stack.resize(base);
            return type_fail(p, "cannot add a string to something that is not one");
        }
        p.stack[i] = w;
        total += static_cast<StrObj*>(as_obj(w))->len;
    }
    Value s = p.heap().make_string(nullptr, total);
    auto* z = static_cast<StrObj*>(as_obj(s));
    size_t at = 0;
    auto put = [&](Value v) {
        auto* x = static_cast<StrObj*>(as_obj(v));
        std::memcpy(z->data() + at, x->data(), x->len);
        at += x->len;
    };
    put(p.stack[base]);
    for (size_t i = p.stack.size(); i-- > base + 2;) put(p.stack[i]);
    p.stack.resize(base);
    return NativeResult::ok(s);
}

/// A value a membership set can hold by hashing: one that is evaluated, and
/// is equal to another exactly when a map would find it as a key.
static bool set_flat(Value v) {
    return is_fixnum(v) || is_atom(v) || is_char(v) || is_bool(v) || is_unit(v) || is_obj(v, ObjType::Str);
}

/// `==` between two flat values, as the language decides it.
static bool flat_equal(Value a, Value b) {
    if (is_obj(a, ObjType::Str) && is_obj(b, ObjType::Str)) {
        auto* x = static_cast<StrObj*>(as_obj(a));
        auto* y = static_cast<StrObj*>(as_obj(b));
        return x->len == y->len && std::memcmp(x->data(), y->data(), x->len) == 0;
    }
    return a == b;
}

// A membership set is what `list.contains x xs` would find, kept so that a
// lookup answers exactly what the scan would -- the same answer, with the
// same elements forced -- or says how much of the scan is left to do.
//
// `#[map, hi, lo, pending, broken, cut]`. Every element has a position in
// the scan, larger for one scanned earlier: one consed onto the front takes
// `hi` and counts it up, one found further along the list takes `lo` and
// counts it down. `map` hashes each flat element to the largest position it
// has; `pending` holds the elements not yet evaluated when they came, as
// `#[position, value]`, newest first; `cut` is the first cell of the list not
// yet looked at, `NIL` once there is none; `broken` says an element is not
// flat, and nothing can be said at all.

enum { SET_MAP, SET_HI, SET_LO, SET_PENDING, SET_BROKEN, SET_CUT, SET_SIZE };

static Value set_field(Value s, int i) { return static_cast<ArrayObj*>(as_obj(s))->items()[i]; }

static Value make_set(Process& p, Value map, int64_t hi, int64_t lo, Value pending, bool broken, Value cut) {
    Value arr = p.heap().make_array(SET_SIZE);
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    a->items()[SET_MAP] = map;
    a->items()[SET_HI] = make_fixnum(hi);
    a->items()[SET_LO] = make_fixnum(lo);
    a->items()[SET_PENDING] = pending;
    a->items()[SET_BROKEN] = make_bool(broken);
    a->items()[SET_CUT] = cut;
    return arr;
}

static Value broken_set(Process& p) { return make_set(p, p.heap().make_map(0), 1, 0, NIL, true, NIL); }

static Value pending_pair(Process& p, int64_t at, Value v) {
    Value pair = p.heap().make_array(2);
    static_cast<ArrayObj*>(as_obj(pair))->items()[0] = make_fixnum(at);
    static_cast<ArrayObj*>(as_obj(pair))->items()[1] = v;
    return pair;
}

/// Hash `h` at position `at`, unless it is there at a later one already.
static Value hash_at(Process& p, Value map, Value h, int64_t at) {
    Value had;
    if (map_lookup(p, map, h, &had) && fixnum_value(had) >= at) return map;
    return map_insert(p, map, h, make_fixnum(at));
}

/// The set with everything that has been evaluated since taken in: the cells
/// past `cut` that are evaluated now, and the pending elements that are. No
/// value is forced. Nothing here is a safepoint: the walk allocates, and does
/// not evaluate.
static Value settle_set(Process& p, Value s) {
    if (set_field(s, SET_BROKEN) == TRUE_V) return s;
    Value map = set_field(s, SET_MAP);
    int64_t lo = fixnum_value(set_field(s, SET_LO));
    // Pending elements now evaluated are hashed; the rest stay, newest first.
    Value kept_rev = NIL;
    bool changed = false;
    for (Value q = set_field(s, SET_PENDING); !is_nil(q); q = static_cast<ConsObj*>(as_obj(q))->tail) {
        Value pair = static_cast<ConsObj*>(as_obj(q))->head;
        auto* pr = static_cast<ArrayObj*>(as_obj(pair));
        Value v = resolve(pr->items()[1]);
        if (!is_whnf(v)) { kept_rev = p.heap().make_cons(pair, kept_rev); continue; }
        if (!set_flat(v)) return broken_set(p);
        map = hash_at(p, map, v, fixnum_value(pr->items()[0]));
        changed = true;
    }
    // The cells past the cut that have been evaluated, each further along
    // than everything before it.
    Value cut = resolve(set_field(s, SET_CUT));
    while (!is_nil(cut) && is_whnf(cut)) {
        if (!is_obj(cut, ObjType::Cons)) return broken_set(p);
        auto* c = static_cast<ConsObj*>(as_obj(cut));
        Value h = resolve(c->head);
        Value next = c->tail;
        if (!is_whnf(h)) {
            // Oldest of all, so last: the pending list keeps newest first.
            kept_rev = p.heap().make_cons(pending_pair(p, lo, c->head), kept_rev);
        } else if (!set_flat(h)) {
            return broken_set(p);
        } else {
            map = hash_at(p, map, h, lo);
        }
        --lo;
        changed = true;
        cut = resolve(next);
    }
    if (!changed) return s;
    Value pending = NIL;
    for (Value q = kept_rev; !is_nil(q); q = static_cast<ConsObj*>(as_obj(q))->tail) {
        pending = p.heap().make_cons(static_cast<ConsObj*>(as_obj(q))->head, pending);
    }
    return make_set(p, map, fixnum_value(set_field(s, SET_HI)), lo, pending, false, cut);
}

/// `_set_of xs`: the set of a list, as much of it as is evaluated.
NativeResult core_set_of(Process& p, Value, Value* args, uint32_t) {
    return NativeResult::ok(settle_set(p, make_set(p, p.heap().make_map(0), 1, 0, NIL, false, args[0])));
}

/// `_set_add s e`: the set of `e :: xs`. Nothing is forced: an `e` not yet
/// evaluated is pending until a scan would reach it.
NativeResult core_set_add(Process& p, Value, Value* args, uint32_t) {
    Value s = resolve(args[0]);
    if (set_field(s, SET_BROKEN) == TRUE_V) return NativeResult::ok(s);
    int64_t hi = fixnum_value(set_field(s, SET_HI));
    Value e = resolve(args[1]);
    Value map = set_field(s, SET_MAP);
    Value pending = set_field(s, SET_PENDING);
    if (!is_whnf(e)) pending = p.heap().make_cons(pending_pair(p, hi, args[1]), pending);
    else if (!set_flat(e)) return NativeResult::ok(broken_set(p));
    else map = map_insert(p, map, e, make_fixnum(hi));
    return NativeResult::ok(make_set(p, map, hi + 1, fixnum_value(set_field(s, SET_LO)), pending, false, set_field(s, SET_CUT)));
}

/// `_set_refresh s xs`: `s` with what has been evaluated since taken in. The
/// scans a set stands in for evaluate the list as they run, so a set that
/// could only say part at first says more each time.
NativeResult core_set_refresh(Process& p, Value, Value* args, uint32_t) {
    return NativeResult::ok(settle_set(p, resolve(args[0])));
}

/// `_set_lookup s x`: what `list.contains x xs` answers, forcing what its scan
/// would force -- `x`, when there is anything to compare it with, and each
/// pending element the scan reaches before it finds `x`. When the scan would
/// go on past what the set has seen, the rest of the list, for the caller to
/// scan; `()` when the set cannot say at all.
NativeResult core_set_lookup(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    Value s0 = resolve(args[0]);
    if (set_field(s0, SET_BROKEN) == TRUE_V) return NativeResult::ok(UNIT);
    // Nothing added and nothing found along the list: no element seen.
    bool empty = fixnum_value(set_field(s0, SET_HI)) == 1 && fixnum_value(set_field(s0, SET_LO)) == 0;
    // The rest of the list, for the scan to go on with, evaluated to its
    // first cell: the scan's own next step forces exactly that, and what
    // reads this answer tells a list from a boolean with `type_is`, which a
    // suspension is neither of.
    auto rest_of = [&p](Value cut, Value* out) -> bool {
        Value w;
        if (!force_whnf(p, cut, &w)) return false;
        *out = is_nil(w) ? FALSE_V : w;
        return true;
    };
    if (empty) {
        // Nothing seen to compare with: the scan goes on, or there is none.
        Value out;
        if (!rest_of(set_field(s0, SET_CUT), &out)) return NativeResult::raise(p.result);
        return NativeResult::ok(out);
    }
    const size_t base = p.stack.size();
    p.stack.push_back(s0);
    p.stack.push_back(args[1]);
    Value x;
    if (!force_whnf(p, p.stack[base + 1], &x)) {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    }
    if (!set_flat(x)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
    p.stack[base + 1] = x;
    Value found;
    bool hit = map_lookup(p, set_field(p.stack[base], SET_MAP), x, &found);
    int64_t at = hit ? fixnum_value(found) : INT64_MIN;
    // Each pending element the scan reaches before the hit, forced and
    // compared, newest first.
    p.stack.push_back(set_field(p.stack[base], SET_PENDING));
    for (;;) {
        Value q = p.stack[base + 2];
        if (is_nil(q)) break;
        auto* pair = static_cast<ArrayObj*>(as_obj(static_cast<ConsObj*>(as_obj(q))->head));
        if (fixnum_value(pair->items()[0]) < at) break;
        p.stack.push_back(pair->items()[1]);
        Value w;
        if (!force_whnf(p, p.stack.back(), &w)) {
            if (!p.force_blocked) p.stack.resize(base);
            return NativeResult::raise(p.result);
        }
        p.stack.pop_back();
        if (!set_flat(w)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        if (flat_equal(w, p.stack[base + 1])) { p.stack.resize(base); return NativeResult::ok(TRUE_V); }
        p.stack[base + 2] = static_cast<ConsObj*>(as_obj(p.stack[base + 2]))->tail;
    }
    Value cut = set_field(p.stack[base], SET_CUT);
    p.stack.resize(base);
    if (hit) return NativeResult::ok(TRUE_V);
    Value out;
    if (!rest_of(cut, &out)) return NativeResult::raise(p.result);
    return NativeResult::ok(out);
}

// A tally is what `count x xs` -- `list.length (list.filter (fn y -> y ==
// x) xs)`, as `std`'s fusion writes it, a scan that compares each element
// with `x` -- would find for every `x` at once: each flat element of `xs`
// mapped to how often it occurs, or `()` when an element is not flat and
// nothing can be said. The count scans the whole list, forcing every
// element, so making the tally forces nothing a count would not; a lookup
// forces the list's first cell, its head and `x` first, in the scan's
// order, and only then the rest, through the tally.

/// `_tally xs`: each flat element of `xs` to its count, or `()`.
NativeResult core_tally(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    p.stack.push_back(p.heap().make_map(0));
    auto fail = [&]() {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    };
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[base], &w)) return fail();
        if (is_nil(w)) break;
        if (!is_obj(w, ObjType::Cons)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        p.stack[base] = static_cast<ConsObj*>(as_obj(w))->tail;
        Value h;
        if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &h)) return fail();
        if (!set_flat(h)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        Value had;
        int64_t n = map_lookup(p, p.stack[base + 1], h, &had) ? fixnum_value(had) : 0;
        p.stack[base + 1] = map_insert(p, p.stack[base + 1], h, make_fixnum(n + 1));
    }
    Value map = p.stack[base + 1];
    p.stack.resize(base);
    return NativeResult::ok(map);
}

/// `_tally_lookup t xs x`: how often `x` occurs in `xs`, `t` being `xs`'s
/// tally; `()` when the tally cannot say, and the count runs as written.
NativeResult core_tally_lookup(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    p.stack.push_back(args[1]);
    p.stack.push_back(args[2]);
    auto fail = [&]() {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    };
    Value w;
    if (!force_whnf(p, p.stack[base + 1], &w)) return fail();
    // An empty list: the count is nothing, and `x` is never compared.
    if (is_nil(w)) { p.stack.resize(base); return NativeResult::ok(make_fixnum(0)); }
    if (!is_obj(w, ObjType::Cons)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
    Value h, x, t;
    if (!force_whnf(p, static_cast<ConsObj*>(as_obj(w))->head, &h)) return fail();
    if (!force_whnf(p, p.stack[base + 2], &x)) return fail();
    p.stack[base + 2] = x;
    if (!set_flat(x)) { p.stack.resize(base); return NativeResult::ok(UNIT); }
    if (!force_whnf(p, p.stack[base], &t)) return fail();
    x = p.stack[base + 2];
    p.stack.resize(base);
    if (is_unit(t)) return NativeResult::ok(UNIT);
    Value n;
    return NativeResult::ok(map_lookup(p, t, x, &n) ? n : make_fixnum(0));
}

// A key index is what `list.find (fn e -> e.[0] == x) xs` -- a scan comparing
// a key read out of each element with `x`, as `std`'s fusion writes one --
// would find for every `x` at once, filled only as far as the scans it stands
// for would have read. `#[map, cut, broken, path, x_first, read]`: `map` takes each
// flat key to the first element that has it, `cut` is the list past what has
// been read, `path` the keys the scan reads an element's key by (`e.[0]` is
// `[0]`), `x_first` whether the scan forces `x` before the key, and `read`
// how many elements have been read.
//
// A lookup of a key already read answers from the map: the scan would have
// forced everything before it, which the index has. Otherwise the index reads
// on from the cut, an element and its key at a time, as the scan would, until
// it finds the key or the list ends. It is filled in place -- each answer is
// what the scan answers, so no one can tell -- with the write barrier on each
// store, as a thunk's update has. A key that is not flat, or a path the index
// cannot follow, breaks it: it answers `()` from then on, and the scan runs as
// written.

enum { KEY_MAP, KEY_CUT, KEY_BROKEN, KEY_PATH, KEY_X_FIRST, KEY_READ, KEY_SIZE };

static void key_store(Process& p, Value idx, int field, Value v) {
    auto* a = static_cast<ArrayObj*>(as_obj(idx));
    a->items()[field] = v;
    p.heap().remember_if_old(a, v);
}

/// `_index_new xs path x_first`: an index of `xs` with nothing read yet.
NativeResult core_index_new(Process& p, Value, Value* args, uint32_t) {
    Value idx = p.heap().make_array(KEY_SIZE);
    auto* a = static_cast<ArrayObj*>(as_obj(idx));
    a->items()[KEY_MAP] = p.heap().make_map(0);
    a->items()[KEY_CUT] = args[0];
    a->items()[KEY_BROKEN] = FALSE_V;
    a->items()[KEY_PATH] = args[1];
    a->items()[KEY_X_FIRST] = args[2];
    a->items()[KEY_READ] = make_fixnum(0);
    return NativeResult::ok(idx);
}

/// Read the key of the element at `p.stack[at]` along `path` into the same
/// slot, forcing what each `.[k]` forces. 1 when it is a flat key, 0 when
/// the index cannot follow, -1 when a force raised.
static int key_read(Process& p, size_t at, Value path) {
    p.stack.push_back(path);
    const size_t path_at = p.stack.size() - 1;
    for (;;) {
        Value w;
        if (!force_whnf(p, p.stack[at], &w)) return -1;
        p.stack[at] = w;
        Value rest = resolve(p.stack[path_at]);
        if (is_nil(rest)) {
            p.stack.pop_back();
            return set_flat(w) ? 1 : 0;
        }
        if (!is_obj(rest, ObjType::Cons)) { p.stack.pop_back(); return 0; }
        Value k = resolve(static_cast<ConsObj*>(as_obj(rest))->head);
        p.stack[path_at] = static_cast<ConsObj*>(as_obj(rest))->tail;
        if (is_obj(w, ObjType::Map)) {
            Value found;
            if (!map_lookup(p, w, k, &found)) { p.stack.pop_back(); return 0; }
            p.stack[at] = found;
        } else if (is_obj(w, ObjType::Record) && is_fixnum(k)) {
            auto* r = static_cast<RecordObj*>(as_obj(w));
            int64_t i = fixnum_value(k);
            if (i < 0 || i >= int64_t(r->len)) { p.stack.pop_back(); return 0; }
            p.stack[at] = r->items()[i];
        } else if (is_obj(w, ObjType::Array) && is_fixnum(k)) {
            auto* r = static_cast<ArrayObj*>(as_obj(w));
            int64_t i = fixnum_value(k);
            if (i < 0 || i >= int64_t(r->len)) { p.stack.pop_back(); return 0; }
            p.stack[at] = r->items()[i];
        } else if ((is_nil(w) || is_obj(w, ObjType::Cons)) && is_fixnum(k) && fixnum_value(k) >= 0) {
            // A list read by position: its spine forced up to it.
            for (int64_t i = fixnum_value(k); ; --i) {
                Value c;
                if (!force_whnf(p, p.stack[at], &c)) return -1;
                if (!is_obj(c, ObjType::Cons)) { p.stack.pop_back(); return 0; }
                if (i == 0) { p.stack[at] = static_cast<ConsObj*>(as_obj(c))->head; break; }
                p.stack[at] = static_cast<ConsObj*>(as_obj(c))->tail;
            }
        } else {
            p.stack.pop_back();
            return 0;
        }
    }
}

/// `_index_find idx x`: what the scan finds for `x` -- `[e]`, the first
/// element whose key is `x`, or `[]` -- forcing what it would; `()` when the
/// index cannot say, and the scan runs as written.
NativeResult core_index_find(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[0]));   // base: the index
    p.stack.push_back(args[1]);            // base + 1: x
    p.stack.push_back(UNIT);               // base + 2: scratch
    auto fail = [&]() {
        if (!p.force_blocked) p.stack.resize(base);
        return NativeResult::raise(p.result);
    };
    auto field = [&](int f) { return static_cast<ArrayObj*>(as_obj(p.stack[base]))->items()[f]; };
    auto broken = [&]() {
        key_store(p, p.stack[base], KEY_BROKEN, TRUE_V);
        p.stack.resize(base);
        return NativeResult::ok(UNIT);
    };
    if (field(KEY_BROKEN) == TRUE_V) { p.stack.resize(base); return NativeResult::ok(UNIT); }
    bool x_ready = false;
    auto force_x = [&]() -> int {
        if (x_ready) return 1;
        Value x;
        if (!force_whnf(p, p.stack[base + 1], &x)) return -1;
        p.stack[base + 1] = x;
        x_ready = true;
        return set_flat(x) ? 1 : 0;
    };
    // Nothing read yet: the scan's first step, in its own order.
    if (fixnum_value(field(KEY_READ)) > 0) {
        int fx = force_x();
        if (fx < 0) return fail();
        if (fx == 0) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        Value found;
        if (map_lookup(p, field(KEY_MAP), p.stack[base + 1], &found)) {
            Value hit = p.heap().make_cons(found, NIL);
            p.stack.resize(base);
            return NativeResult::ok(hit);
        }
    }
    for (;;) {
        Value cell;
        if (!force_whnf(p, field(KEY_CUT), &cell)) return fail();
        key_store(p, p.stack[base], KEY_CUT, cell);
        if (is_nil(cell)) {
            // The end: forced `x` only if there was something to compare.
            p.stack.resize(base);
            return NativeResult::ok(NIL);
        }
        if (!is_obj(cell, ObjType::Cons)) return broken();
        if (field(KEY_X_FIRST) == TRUE_V) {
            int fx = force_x();
            if (fx < 0) return fail();
            if (fx == 0) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        }
        auto* c = static_cast<ConsObj*>(as_obj(cell));
        p.stack[base + 2] = c->head;
        int got = key_read(p, base + 2, field(KEY_PATH));
        if (got < 0) return fail();
        if (got == 0) return broken();
        int fx = force_x();
        if (fx < 0) return fail();
        if (fx == 0) { p.stack.resize(base); return NativeResult::ok(UNIT); }
        Value key = p.stack[base + 2];
        Value elem = static_cast<ConsObj*>(as_obj(resolve(field(KEY_CUT))))->head;
        Value had;
        if (!map_lookup(p, field(KEY_MAP), key, &had)) {
            key_store(p, p.stack[base], KEY_MAP, map_insert(p, field(KEY_MAP), key, elem));
        }
        key_store(p, p.stack[base], KEY_CUT, static_cast<ConsObj*>(as_obj(resolve(field(KEY_CUT))))->tail);
        key_store(p, p.stack[base], KEY_READ, make_fixnum(fixnum_value(field(KEY_READ)) + 1));
        if (flat_equal(key, p.stack[base + 1])) {
            Value hit = p.heap().make_cons(elem, NIL);
            p.stack.resize(base);
            return NativeResult::ok(hit);
        }
    }
}

NativeResult core_array_to_list(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_obj(v, ObjType::Array)) return type_fail(p, "array_to_list needs an array");
    uint32_t n = static_cast<ArrayObj*>(as_obj(v))->len;
    Value list = NIL;
    for (uint32_t i = n; i-- > 0;) {
        Value item = static_cast<ArrayObj*>(as_obj(v))->items()[i];
        list = p.heap().make_cons(item, list);
    }
    return NativeResult::ok(list);
}

// --- maps ---



NativeResult core_map_has(Process& p, Value, Value* args, uint32_t) {
    Value m = resolve(args[0]);
    if (!is_obj(m, ObjType::Map)) return type_fail(p, "map_has needs a map");
    Value found;
    return NativeResult::ok(make_bool(map_lookup(p, m, args[1], &found)));
}


NativeResult core_map_remove(Process& p, Value, Value* args, uint32_t) {
    Value m = resolve(args[0]);
    if (!is_obj(m, ObjType::Map)) return type_fail(p, "map_remove needs a map");
    return NativeResult::ok(map_erase(p, m, args[1]));
}

NativeResult core_map_pairs(Process& p, Value, Value* args, uint32_t) {
    Value m = resolve(args[0]);
    if (!is_obj(m, ObjType::Map)) return type_fail(p, "map_pairs needs a map");
    std::vector<std::pair<Value, Value>> entries;
    map_collect(m, entries);
    Value list = NIL;
    for (size_t i = entries.size(); i-- > 0;) {
        Value pair = p.heap().make_cons(entries[i].first,
                                        p.heap().make_cons(entries[i].second, NIL));
        list = p.heap().make_cons(pair, list);
    }
    return NativeResult::ok(list);
}

// --- ordering ---

/// A total order, so sorting can be written in Dream. Values of different
/// types order by type, which keeps it total without pretending an integer and
/// a string are comparable.
///
/// Lists and arrays order element by element, the shorter first when one is a
/// prefix of the other, forcing as they go the way `==` does. They used to
/// rank with functions and maps and all compare equal, and a stable sort on a
/// list key then left everything exactly where it was: no error, just an order
/// that was never applied. Maps, functions and pids still compare equal within
/// their kind; there is no order on them worth promising.
///
/// The force is not vouched (see "VouchesForGc"), so a collection it wants
/// waits for the machine, and holding `Obj*` across it is safe.
int compare_rank(Value v) {
    if (is_fixnum(v) || is_obj(v, ObjType::Float) || is_obj(v, ObjType::BigInt)) return 0;
    if (is_char(v)) return 1;
    if (is_bool(v)) return 2;
    if (is_atom(v)) return 3;
    if (is_stringish(v)) return 4;
    if (is_unit(v)) return 5;
    if (v == NIL || is_obj(v, ObjType::Cons)) return 6;
    if (is_obj(v, ObjType::Array)) return 7;
    if (is_obj(v, ObjType::Tensor)) return 8;
    if (is_obj(v, ObjType::Record)) return 9;
    return 10;
}

bool compare_values(Process& p, Value a, Value b, int depth, int* out) {
    if (depth > 512) {
        p.result = raise_error(p, well_known(p.runtime()).type_error,
                               "structure is too deeply nested to compare");
        return false;
    }
    a = resolve(a);
    b = resolve(b);
    if ((!is_whnf(a) && !force_whnf(p, a, &a)) || (!is_whnf(b) && !force_whnf(p, b, &b)))
        return false;

    int ra = compare_rank(a), rb = compare_rank(b);
    if (ra != rb) {
        *out = ra < rb ? -1 : 1;
        return true;
    }
    auto order = [](auto x, auto y) { return x < y ? -1 : (x > y ? 1 : 0); };
    switch (ra) {
        case 0: {
            if (bigint::is_integer(a) && bigint::is_integer(b)) {
                *out = bigint::compare(a, b);
            } else {
                auto as_double = [](Value v) {
                    return is_obj(v, ObjType::Float) ? static_cast<FloatObj*>(as_obj(v))->value
                                                     : bigint::to_double(v);
                };
                *out = order(as_double(a), as_double(b));
            }
            return true;
        }
        case 1:
        case 2: *out = order(imm_payload(a), imm_payload(b)); return true;
        case 3: {
            const std::string& x = p.runtime().atom_name(uint32_t(imm_payload(a)));
            const std::string& y = p.runtime().atom_name(uint32_t(imm_payload(b)));
            *out = order(x, y);
            return true;
        }
        case 4: {
            Bytes x, y;
            string_bytes(a, &x);
            string_bytes(b, &y);
            *out = bytes_compare(x, y);
            return true;
        }
        case 6: {
            // Iterate down the spine rather than recursing on the tail, so a
            // long list is not a deep one; only the elements recurse.
            for (;;) {
                // An improper tail is compared as whatever it is.
                if (compare_rank(a) != 6 || compare_rank(b) != 6)
                    return compare_values(p, a, b, depth + 1, out);
                if (a == NIL || b == NIL) {
                    *out = a == b ? 0 : (a == NIL ? -1 : 1);
                    return true;
                }
                auto* x = static_cast<ConsObj*>(as_obj(a));
                auto* y = static_cast<ConsObj*>(as_obj(b));
                if (!compare_values(p, x->head, y->head, depth + 1, out)) return false;
                if (*out != 0) return true;
                Value ta = resolve(x->tail), tb = resolve(y->tail);
                if ((!is_whnf(ta) && !force_whnf(p, ta, &ta)) ||
                    (!is_whnf(tb) && !force_whnf(p, tb, &tb)))
                    return false;
                a = ta;
                b = tb;
            }
        }
        case 7: {
            auto* x = static_cast<ArrayObj*>(as_obj(a));
            auto* y = static_cast<ArrayObj*>(as_obj(b));
            uint32_t n = std::min(x->len, y->len);
            for (uint32_t i = 0; i < n; ++i) {
                if (!compare_values(p, x->items()[i], y->items()[i], depth + 1, out)) return false;
                if (*out != 0) return true;
            }
            *out = order(x->len, y->len);
            return true;
        }
        case 8: return tensor_compare(p, a, b, out);
        case 9: {
            // By what they are, then field by field: records of one
            // declaration sort as their fields do, which is what sorting a
            // list of them by `compare` wants.
            auto* x = static_cast<RecordObj*>(as_obj(a));
            auto* y = static_cast<RecordObj*>(as_obj(b));
            if (x->id() != y->id()) {
                *out = order(p.runtime().atom_name(uint32_t(imm_payload(x->id()))),
                             p.runtime().atom_name(uint32_t(imm_payload(y->id()))));
                return true;
            }
            uint32_t n = std::min(x->len, y->len);
            for (uint32_t i = 0; i < n; ++i) {
                if (!compare_values(p, static_cast<RecordObj*>(as_obj(a))->items()[i],
                                    static_cast<RecordObj*>(as_obj(b))->items()[i], depth + 1, out))
                    return false;
                if (*out != 0) return true;
            }
            *out = order(static_cast<RecordObj*>(as_obj(a))->len,
                         static_cast<RecordObj*>(as_obj(b))->len);
            return true;
        }
        default: *out = 0; return true;
    }
}

NativeResult core_compare(Process& p, Value, Value* args, uint32_t) {
    int cmp = 0;
    if (!compare_values(p, args[0], args[1], 0, &cmp)) return NativeResult::raise(p.result);
    return NativeResult::ok(make_fixnum(cmp));
}

/// `sort_keyed keys xs` -- the elements of the array `xs`, as a list, in the
/// order `compare` puts `keys` in, equal keys keeping their order. What
/// `std.list.sort` and `sort_on` are: a merge sort written in Dream took apart
/// and rebuilt its list at every level and asked for the key twice a
/// comparison, and a compile sorts tens of thousands of things.
///
/// The keys are forced whole first, under a vouch, and the sort then compares
/// values that are already forced, so it forces nothing and nothing can be
/// collected under it. The elements are carried, never looked at: an element
/// the Dream sort never forced is not forced here either.
NativeResult core_sort_keyed(Process& p, Value, Value* args, uint32_t) {
    Value keys;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &keys)) return NativeResult::raise(p.result);
    }
    Value xs = resolve(args[1]);
    if (!is_obj(xs, ObjType::Array)) return type_fail(p, "sort_keyed needs a list of keys and an array");
    auto* arr = static_cast<ArrayObj*>(as_obj(xs));
    std::vector<Value> ks;
    for (Value cur = resolve(keys); is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail))
        ks.push_back(static_cast<ConsObj*>(as_obj(cur))->head);
    if (ks.size() != arr->len) return type_fail(p, "sort_keyed needs a key for every element");
    std::vector<uint32_t> order(ks.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    bool ok = true;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (!ok) return false;
        int cmp = 0;
        if (!compare_values(p, ks[a], ks[b], 0, &cmp)) { ok = false; return false; }
        return cmp < 0;
    });
    if (!ok) return NativeResult::raise(p.result);
    Value list = NIL;
    for (size_t i = order.size(); i-- > 0;) list = p.heap().make_cons(arr->items()[order[i]], list);
    return NativeResult::ok(list);
}

// --- the wire format, natively ---
//
// `std.wire` is written in Dream, a byte at a time, and it is exact and slow:
// 150,000 small lists took 7.7 s to write and 70 s to read, which is a
// language describing its own bytes through `str_le` and `str_byte`. Anything
// that keeps a compiled program's state between runs -- a build's compile
// units -- is that size, so the format is done here as well, byte for byte
// the same: `std.wire.encode` and `decode` call these, and every message
// either side writes the other reads. The Dream writer stays the definition
// of the format, and `std.wire`'s tests hold the two to each other.
//
// The same rules, in the same places. A map's entries are written in the order
// `compare` puts their keys, so equal maps are equal bytes. Anything that is
// not data -- a function, a process -- is written as the string `to_string`
// gives it, as `std.codec`'s `write_opaque` does. Reading never raises: a
// value that is not one is `[:error, why]`. And an atom is only ever looked
// up, never made, so a message cannot grow the atom table.

enum : uint8_t {
    WIRE_UNIT = 0, WIRE_FALSE = 1, WIRE_TRUE = 2, WIRE_INT = 3, WIRE_FLOAT = 4,
    WIRE_STRING = 5, WIRE_ATOM = 6, WIRE_CHAR = 7, WIRE_LIST = 8, WIRE_ARRAY = 9,
    WIRE_MAP = 10, WIRE_BIGINT = 11, WIRE_TENSOR = 12,
    // The id as an atom is written, a byte of flags, a count, then the
    // fields. Read back only where the atom already exists, as an atom is:
    // a record crosses to a program that declares it.
    WIRE_RECORD = 13,
    // A trunk: a count, then the elements, as an array is.
    WIRE_TRUNK = 14,
};

void wire_le(std::string* out, uint64_t v, int width) {
    for (int i = 0; i < width; ++i) out->push_back(char((v >> (8 * i)) & 0xff));
}

void wire_bytes(std::string* out, uint8_t tag, const char* data, uint64_t len) {
    out->push_back(char(tag));
    wire_le(out, len, 4);
    out->append(data, size_t(len));
}

/// Write a value that has already been forced all the way down. False when
/// a map's keys could not be compared, which is the only way this can fail.
bool wire_write(Process& p, Value v, std::string* out, int depth) {
    v = resolve(v);
    if (depth > 10000) return false;
    if (is_unit(v)) { out->push_back(char(WIRE_UNIT)); return true; }
    if (is_bool(v)) { out->push_back(char(v == TRUE_V ? WIRE_TRUE : WIRE_FALSE)); return true; }
    if (is_fixnum(v)) {
        out->push_back(char(WIRE_INT));
        wire_le(out, uint64_t(fixnum_value(v)), 8);
        return true;
    }
    if (bigint::is_big(v)) {
        // Eight bytes when it fits in them, as every integer did before there
        // were bignums; past 64 bits, as many as its two's complement needs.
        int64_t small;
        if (bigint::to_int64(v, &small)) {
            out->push_back(char(WIRE_INT));
            wire_le(out, uint64_t(small), 8);
            return true;
        }
        const size_t n = bigint::le_width(v);
        out->push_back(char(WIRE_BIGINT));
        wire_le(out, n, 4);
        const size_t at = out->size();
        out->resize(at + n);
        bigint::to_le_bytes(v, n, reinterpret_cast<uint8_t*>(&(*out)[at]));
        return true;
    }
    if (is_char(v)) {
        out->push_back(char(WIRE_CHAR));
        wire_le(out, uint64_t(imm_payload(v)), 4);
        return true;
    }
    if (is_atom(v)) {
        const std::string& name = p.runtime().atom_name(uint32_t(imm_payload(v)));
        wire_bytes(out, WIRE_ATOM, name.data(), name.size());
        return true;
    }
    if (v == NIL) { out->push_back(char(WIRE_LIST)); wire_le(out, 0, 4); return true; }
    if (is_ptr(v)) {
        switch (as_obj(v)->type) {
            case ObjType::Float: {
                double d = static_cast<FloatObj*>(as_obj(v))->value;
                char b[8];
                std::memcpy(b, &d, 8);
                out->push_back(char(WIRE_FLOAT));
                out->append(b, 8);
                return true;
            }
            case ObjType::Tensor: {
                Value host;
                if (!tensor_host_copy(p, v, &host)) return false;
                auto* t = static_cast<TensorObj*>(as_obj(host));
                out->push_back(char(WIRE_TENSOR));
                out->push_back(char(t->rank));
                for (uint32_t axis = 0; axis < t->rank; ++axis) wire_le(out, t->dims[axis], 4);
                for (uint64_t i = 0; i < t->count; ++i) {
                    uint64_t bits;
                    std::memcpy(&bits, t->data() + i, sizeof bits);
                    wire_le(out, bits, 8);
                }
                return true;
            }
            case ObjType::Str: {
                auto* s = static_cast<StrObj*>(as_obj(v));
                wire_bytes(out, WIRE_STRING, s->data(), s->len);
                return true;
            }
            case ObjType::Cons: {
                // The count comes first and is not known until the spine has
                // been walked, so a place is kept for it and filled after.
                out->push_back(char(WIRE_LIST));
                size_t at = out->size();
                wire_le(out, 0, 4);
                uint64_t n = 0;
                for (Value cur = v; is_obj(cur, ObjType::Cons);
                     cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
                    if (!wire_write(p, static_cast<ConsObj*>(as_obj(cur))->head, out, depth + 1)) return false;
                    ++n;
                }
                for (int i = 0; i < 4; ++i) (*out)[at + i] = char((n >> (8 * i)) & 0xff);
                return true;
            }
            case ObjType::Record: {
                auto* r = static_cast<RecordObj*>(as_obj(v));
                const std::string& name = p.runtime().atom_name(uint32_t(imm_payload(r->id())));
                wire_bytes(out, WIRE_RECORD, name.data(), name.size());
                out->push_back(char(r->variant() ? 1 : 0));
                wire_le(out, r->len, 4);
                for (uint32_t i = 0; i < r->len; ++i) {
                    if (!wire_write(p, static_cast<RecordObj*>(as_obj(v))->items()[i], out, depth + 1)) return false;
                }
                return true;
            }
            case ObjType::Trunk: {
                const uint32_t n = trunk_length(v);
                out->push_back(char(WIRE_TRUNK));
                wire_le(out, n, 4);
                for (uint32_t i = 0; i < n; ++i) {
                    if (!wire_write(p, trunk_element(v, i), out, depth + 1)) return false;
                }
                return true;
            }
            case ObjType::Array: {
                auto* a = static_cast<ArrayObj*>(as_obj(v));
                out->push_back(char(WIRE_ARRAY));
                wire_le(out, a->len, 4);
                for (uint32_t i = 0; i < a->len; ++i) {
                    if (!wire_write(p, static_cast<ArrayObj*>(as_obj(v))->items()[i], out, depth + 1)) return false;
                }
                return true;
            }
            case ObjType::Map: {
                std::vector<std::pair<Value, Value>> entries;
                map_collect(v, entries);
                bool ok = true;
                std::stable_sort(entries.begin(), entries.end(), [&](const auto& x, const auto& y) {
                    int cmp = 0;
                    if (!compare_values(p, x.first, y.first, 0, &cmp)) ok = false;
                    return cmp < 0;
                });
                if (!ok) return false;
                out->push_back(char(WIRE_MAP));
                wire_le(out, entries.size(), 4);
                for (auto& [k, val] : entries) {
                    if (!wire_write(p, k, out, depth + 1) || !wire_write(p, val, out, depth + 1)) return false;
                }
                return true;
            }
            case ObjType::BigStr: {
                auto* b = static_cast<BigStrObj*>(as_obj(v));
                wire_bytes(out, WIRE_STRING, b->data, b->len);
                return true;
            }
            default: break;
        }
    }
    // Not data: its text, as `std.codec`'s `write_opaque` writes it.
    std::string text;
    if (!stringify(p, v, &text)) return false;
    wire_bytes(out, WIRE_STRING, text.data(), text.size());
    return true;
}

NativeResult vm_wire_encode(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    std::string out;
    if (!wire_write(p, forced, &out, 0)) {
        return type_fail(p, "wire_encode: a map's keys could not be compared, or the value nests too deep");
    }
    if (out.size() > UINT32_MAX) return type_fail(p, "wire_encode: the message is larger than a string can be");
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

struct WireReader {
    Process& p;
    const unsigned char* data;
    uint64_t len;
    uint64_t at = 0;

    bool has(uint64_t n) const { return at + n <= len; }
    uint64_t le(int width) {
        uint64_t v = 0;
        for (int i = width - 1; i >= 0; --i) v = (v << 8) | data[at + i];
        at += width;
        return v;
    }

    /// One value, or false for bytes that are not one.
    bool read(Value* out, int depth) {
        if (depth > 10000 || !has(1)) return false;
        uint8_t tag = data[at++];
        switch (tag) {
            case WIRE_UNIT: *out = UNIT; return true;
            case WIRE_FALSE: *out = make_bool(false); return true;
            case WIRE_TRUE: *out = make_bool(true); return true;
            case WIRE_INT: {
                if (!has(8)) return false;
                *out = make_integer(p, int64_t(le(8)));
                return true;
            }
            case WIRE_BIGINT: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                if (!has(n)) return false;
                *out = bigint::from_le_bytes(p.heap(), data + at, size_t(n));
                at += n;
                return true;
            }
            case WIRE_FLOAT: {
                if (!has(8)) return false;
                double d;
                std::memcpy(&d, data + at, 8);
                at += 8;
                *out = p.heap().make_float(d);
                return true;
            }
            case WIRE_TENSOR: {
                if (!has(1)) return false;
                const uint32_t rank = data[at++];
                if (rank == 0 || rank > TENSOR_MAX_RANK || !has(rank * 4)) return false;
                uint32_t dims[TENSOR_MAX_RANK];
                uint64_t count = 1;
                const uint64_t limit = (uint64_t(UINT32_MAX) - sizeof(TensorObj) - 64) / 8;
                for (uint32_t axis = 0; axis < rank; ++axis) {
                    dims[axis] = uint32_t(le(4));
                    if (!dims[axis] || count > limit / dims[axis]) return false;
                    count *= dims[axis];
                }
                if (!has(count * 8)) return false;
                Value v = p.heap().make_tensor(rank, dims, count);
                auto* t = static_cast<TensorObj*>(as_obj(v));
                for (uint64_t i = 0; i < count; ++i) {
                    const uint64_t bits = le(8);
                    std::memcpy(t->data() + i, &bits, sizeof bits);
                }
                *out = v;
                return true;
            }
            case WIRE_STRING: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                if (!has(n)) return false;
                *out = p.heap().make_string(reinterpret_cast<const char*>(data + at), uint32_t(n));
                at += n;
                return true;
            }
            case WIRE_ATOM: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                if (!has(n)) return false;
                uint32_t id;
                if (!p.runtime().find_atom(std::string_view(reinterpret_cast<const char*>(data + at), n), &id)) {
                    return false;
                }
                at += n;
                *out = make_atom(id);
                return true;
            }
            case WIRE_CHAR: {
                if (!has(4)) return false;
                uint64_t code = le(4);
                if (code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) return false;
                *out = make_char(uint32_t(code));
                return true;
            }
            case WIRE_LIST: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                std::vector<Value> items;
                items.reserve(size_t(std::min<uint64_t>(n, len)));
                for (uint64_t i = 0; i < n; ++i) {
                    Value item;
                    if (!read(&item, depth + 1)) return false;
                    items.push_back(item);
                }
                Value list = NIL;
                for (size_t i = items.size(); i-- > 0;) list = p.heap().make_cons(items[i], list);
                *out = list;
                return true;
            }
            case WIRE_ARRAY: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                if (n > len) return false;
                std::vector<Value> items;
                items.reserve(size_t(n));
                for (uint64_t i = 0; i < n; ++i) {
                    Value item;
                    if (!read(&item, depth + 1)) return false;
                    items.push_back(item);
                }
                Value arr = p.heap().make_array(uint32_t(n));
                for (uint64_t i = 0; i < n; ++i) static_cast<ArrayObj*>(as_obj(arr))->items()[i] = items[i];
                *out = arr;
                return true;
            }
            case WIRE_TRUNK: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                if (n > len) return false;
                std::vector<Value> items;
                items.reserve(size_t(n));
                for (uint64_t i = 0; i < n; ++i) {
                    Value item;
                    if (!read(&item, depth + 1)) return false;
                    items.push_back(item);
                }
                *out = trunk_of_values(p, items);
                return true;
            }
            case WIRE_RECORD: {
                if (!has(4)) return false;
                uint64_t name_len = le(4);
                if (!has(name_len)) return false;
                uint32_t id;
                if (!p.runtime().find_atom(std::string_view(reinterpret_cast<const char*>(data + at), name_len), &id)) {
                    return false;
                }
                at += name_len;
                if (!has(5)) return false;
                const bool variant = (data[at] & 1) != 0;
                at += 1;
                uint64_t n = le(4);
                if (n > len) return false;
                std::vector<Value> items;
                items.reserve(size_t(n));
                for (uint64_t i = 0; i < n; ++i) {
                    Value item;
                    if (!read(&item, depth + 1)) return false;
                    items.push_back(item);
                }
                Value rec = p.heap().make_record(record_tag(make_atom(id), variant), uint32_t(n));
                for (uint64_t i = 0; i < n; ++i) static_cast<RecordObj*>(as_obj(rec))->items()[i] = items[i];
                *out = rec;
                return true;
            }
            case WIRE_MAP: {
                if (!has(4)) return false;
                uint64_t n = le(4);
                Value m = p.heap().make_map(0);
                for (uint64_t i = 0; i < n; ++i) {
                    Value k, v;
                    if (!read(&k, depth + 1) || !read(&v, depth + 1)) return false;
                    m = resolve(map_insert(p, m, k, v));
                }
                *out = m;
                return true;
            }
            default: return false;
        }
    }
};

/// The SHA-256 of the running image, as `io.digest` spells one: which program
/// this is, exactly. A compiler keys what it caches between runs by it, since
/// a unit one compiler made means nothing to another. Pure, because an image
/// cannot change while it runs; worked out once per runtime.
NativeResult vm_image_digest(Process& p, Value, Value*, uint32_t) {
    static std::mutex mutex;
    static std::unordered_map<const void*, std::string> known;
    const Image& img = p.runtime().image();
    std::string hex;
    {
        std::lock_guard<std::mutex> g(mutex);
        auto it = known.find(&img);
        if (it != known.end()) hex = it->second;
    }
    if (hex.empty()) {
        Sha256 sha;
        sha.update(img.bytes(), img.byte_count());
        hex = sha.hex();
        std::lock_guard<std::mutex> g(mutex);
        known[&img] = hex;
    }
    return NativeResult::ok(p.heap().make_string(hex.data(), uint32_t(hex.size())));
}

// ---------------------------------------------------------------------------
// Sharing a program's parts into one arena
// ---------------------------------------------------------------------------
//
// `opt.optimize_parts` in the compiler, done here: the last whole-program walk
// of a build, which hash-conses every part's arena into one. The parts arrive
// already shared within themselves, so all this does is rename -- each part's
// constants into the program's pools, its functions and invented globals past
// the parts before it -- and keep each node once by its whole record. Written
// in Dream it was a map insert and a `to_string` per node, threaded through a
// state record, and the largest stage left in a warm build of a big program.
//
// It is the Dream walk, step for step, and must stay so: the image it writes
// is compared byte for byte with the one the Dream walk writes (`opt`'s tests
// hold the two to each other on parts with every kind of node in them). The
// opcode tables are the compiler's, handed in, so there is no second copy of
// which operand is an edge.
//
// Everything is forced before the walk starts, and nothing is forced during
// it, so the values read here stay where they are: allocation never collects.

struct ShareArenas {
    Process& p;
    int64_t part_base = 0;
    int64_t invented_from = 0;
    Value filled = NIL;
    Value none = 0, nop = 0, field = 0, unit = 0, global = 0, make_closure = 0, make_thunk = 0;
    Value const_int = 0, const_float = 0, const_str = 0, const_atom = 0, record_make = 0;
    std::unordered_map<Value, std::vector<int>> edges;
    std::unordered_map<Value, std::array<int64_t, 3>> runs;

    struct Part {
        Value nodes = NIL, kids = NIL, remap = UNIT;
        int64_t fo = 0, go = 0;
        Value ints = NIL, floats = NIL, strs = NIL, atoms = NIL;
    };
    std::vector<Part> parts;
    size_t cur = 0;

    using Node = std::array<Value, 5>;
    std::vector<Node> out;
    std::vector<Value> kids;
    std::unordered_map<std::string, int64_t> seen;
    std::unordered_map<std::string, int64_t> runs_seen;
    std::unordered_map<int64_t, Value> memo;
    const char* failed = nullptr;

    explicit ShareArenas(Process& proc) : p(proc) {}

    Value atom(const char* name) { return make_atom(p.runtime().intern_atom(name)); }

    /// `c.[i else d]` for a list, an array or a map.
    Value at(Value c, int64_t i, Value d) {
        c = resolve(c);
        if (i < 0) return d;
        if (is_obj(c, ObjType::Array)) {
            auto* a = static_cast<ArrayObj*>(as_obj(c));
            return uint64_t(i) < a->len ? resolve(a->items()[i]) : d;
        }
        // A node is a record (`ir.Node`), read by position as its list was.
        if (is_obj(c, ObjType::Record)) {
            auto* r = static_cast<RecordObj*>(as_obj(c));
            return uint64_t(i) < r->len ? resolve(r->items()[i]) : d;
        }
        if (is_obj(c, ObjType::Map)) {
            Value found;
            return map_lookup(p, c, make_fixnum(i), &found) ? resolve(found) : d;
        }
        for (Value cell = c; is_obj(cell, ObjType::Cons); cell = resolve(static_cast<ConsObj*>(as_obj(cell))->tail)) {
            if (i-- == 0) return resolve(static_cast<ConsObj*>(as_obj(cell))->head);
        }
        return d;
    }
    Value field_of(Value m, Value key, Value d) {
        Value found;
        return is_obj(resolve(m), ObjType::Map) && map_lookup(p, resolve(m), key, &found) ? resolve(found) : d;
    }
    static int64_t num(Value v) { return fixnum_value(resolve(v)); }

    bool is_node(Value v) const { return resolve(v) != none; }

    Node node_at(int64_t i) {
        Value n = at(parts[cur].nodes, i, UNIT);
        if (n == UNIT) return Node{nop, make_fixnum(0), none, none, none};
        Node node{};
        for (int k = 0; k < 5; ++k) node[k] = at(n, k, none);
        return node;
    }
    Value kid_at(int64_t i) { return at(parts[cur].kids, i, none); }

    void key_of(const Value* vs, size_t n, std::string* key) {
        key->clear();
        for (size_t i = 0; i < n; ++i) {
            Value v = resolve(vs[i]);
            if (is_ptr(v)) {
                std::string text;
                if (!stringify(p, v, &text)) failed = "a node holds something that cannot be written";
                key->push_back('\xff');
                key->append(text);
                key->push_back('\0');
            } else {
                key->append(reinterpret_cast<const char*>(&v), sizeof v);
            }
        }
    }

    Value keep(const Node& n) {
        if (n[0] != field) {
            std::string key;
            key_of(n.data(), 5, &key);
            auto it = seen.find(key);
            if (it != seen.end()) return make_fixnum(it->second);
            seen.emplace(std::move(key), int64_t(out.size()));
        }
        out.push_back(n);
        return make_fixnum(int64_t(out.size()) - 1);
    }

    Value keep_run(const std::vector<Value>& run) {
        std::string key;
        key_of(run.data(), run.size(), &key);
        auto it = runs_seen.find(key);
        if (it != runs_seen.end()) return make_fixnum(it->second);
        int64_t offset = int64_t(kids.size());
        runs_seen.emplace(std::move(key), offset);
        kids.insert(kids.end(), run.begin(), run.end());
        return make_fixnum(offset);
    }

    Value remapped(Value table, Value i) {
        Value v = at(table, num(i), 0);
        if (v == 0) failed = "a constant is not in its part's pool";
        return v;
    }

    Node remap_leaf(Node n) {
        const Part& part = parts[cur];
        if (resolve(part.remap) == UNIT) return n;
        Value op = n[0];
        if (op == make_closure || op == make_thunk) n[2] = make_fixnum(num(n[2]) + part.fo);
        else if (op == unit) { if (is_node(n[2])) n[2] = make_fixnum(num(n[2]) + part.fo); }
        else if (op == global) { if (num(n[2]) >= invented_from) n[2] = make_fixnum(num(n[2]) + part.go); }
        else if (op == const_int) n[2] = remapped(part.ints, n[2]);
        else if (op == const_float) n[2] = remapped(part.floats, n[2]);
        else if (op == const_str) n[2] = remapped(part.strs, n[2]);
        else if (op == const_atom) n[2] = remapped(part.atoms, n[2]);
        return n;
    }

    Value rebuild(Value i) {
        i = resolve(i);
        if (!is_node(i) || failed) return i;
        Node n = node_at(num(i));
        int64_t key = num(i) + int64_t(cur) * part_base;
        auto hit = memo.find(key);
        if (hit != memo.end()) return hit->second;
        bool leaf = !edges.count(n[0]) && !runs.count(n[0]);
        Value got = leaf ? rebuild_leaf(remap_leaf(n)) : rebuild_node(n);
        memo[key] = got;
        return got;
    }

    Value rebuild_leaf(const Node& n) {
        if (n[0] == unit && is_node(n[2])) {
            Value value;
            if (is_obj(resolve(filled), ObjType::Map) && map_lookup(p, resolve(filled), resolve(n[2]), &value)) {
                size_t back = cur;
                cur = 0;
                Value got = rebuild(value);
                cur = back;
                return got;
            }
        }
        return keep(n);
    }

    Value rebuild_node(Node n) {
        const Part& part = parts[cur];
        if (n[0] == field && resolve(part.remap) != UNIT) n[3] = remapped(part.strs, n[3]);
        // `record_make`'s atom, as `const_atom`'s, with its variant bit kept.
        if (n[0] == record_make && resolve(part.remap) != UNIT) {
            const int64_t c = num(n[4]);
            const int64_t bit = c & int64_t(RECORD_MAKE_VARIANT);
            n[4] = make_fixnum(num(remapped(part.atoms, make_fixnum(c & int64_t(RECORD_MAKE_ATOM)))) | bit);
        }
        auto e = edges.find(n[0]);
        if (e != edges.end()) {
            for (int k : e->second) n[2 + k] = rebuild(n[2 + k]);
        }
        auto r = runs.find(n[0]);
        if (r != runs.end()) {
            auto [off_k, count_k, width] = r->second;
            int64_t off = num(n[2 + off_k]);
            int64_t many = num(n[2 + count_k]) * width;
            std::vector<Value> run;
            run.reserve(size_t(std::max<int64_t>(many, 0)));
            for (int64_t j = 0; j < many; ++j) run.push_back(rebuild(kid_at(off + j)));
            n[2 + off_k] = keep_run(run);
        }
        return keep(n);
    }

    Value place_captures(Value f) {
        cur = 0;
        int64_t off = num(field_of(f, atom("captures_off"), make_fixnum(0)));
        int64_t many = num(field_of(f, atom("n_captures"), make_fixnum(0)));
        if (many == 0) return make_fixnum(0);
        std::vector<Value> run;
        for (int64_t j = 0; j < many; ++j) run.push_back(kid_at(off + j));
        return keep_run(run);
    }
};

/// `[funcs, parts, invented_from, filled, edges, runs, part_base]` to
/// `[nodes, kids, funcs]`: see "Sharing a program's parts into one arena".
NativeResult vm_share_arenas(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    ShareArenas s(p);
    Value in[7];
    for (int i = 0; i < 7; ++i) in[i] = s.at(forced, i, UNIT);
    s.invented_from = ShareArenas::num(in[2]);
    s.filled = in[3];
    s.part_base = ShareArenas::num(in[6]);
    s.none = s.atom("none"); s.nop = s.atom("nop"); s.field = s.atom("field"); s.unit = s.atom("unit");
    s.global = s.atom("global"); s.make_closure = s.atom("make_closure"); s.make_thunk = s.atom("make_thunk");
    s.const_int = s.atom("const_int"); s.const_float = s.atom("const_float");
    s.const_str = s.atom("const_str"); s.const_atom = s.atom("const_atom");
    s.record_make = s.atom("record_make");
    std::vector<std::pair<Value, Value>> table;
    map_collect(resolve(in[4]), table);
    for (auto& [op, ks] : table) {
        std::vector<int>& into = s.edges[resolve(op)];
        for (int k = 0; s.at(ks, k, UNIT) != UNIT; ++k) into.push_back(int(ShareArenas::num(s.at(ks, k, UNIT))));
    }
    table.clear();
    map_collect(resolve(in[5]), table);
    for (auto& [op, spec] : table) {
        s.runs[resolve(op)] = {ShareArenas::num(s.at(spec, 0, UNIT)), ShareArenas::num(s.at(spec, 1, UNIT)),
                               ShareArenas::num(s.at(spec, 2, UNIT))};
    }
    Value nodes_k = s.atom("nodes"), kids_k = s.atom("kids"), remap_k = s.atom("remap"), fo_k = s.atom("fo"), go_k = s.atom("go");
    for (int64_t k = 0;; ++k) {
        Value part = s.at(in[1], k, UNIT);
        if (part == UNIT) break;
        ShareArenas::Part pv;
        pv.nodes = s.field_of(part, nodes_k, NIL);
        pv.kids = s.field_of(part, kids_k, NIL);
        pv.remap = s.field_of(part, remap_k, UNIT);
        pv.fo = ShareArenas::num(s.field_of(part, fo_k, make_fixnum(0)));
        pv.go = ShareArenas::num(s.field_of(part, go_k, make_fixnum(0)));
        if (pv.remap != UNIT) {
            pv.ints = s.field_of(pv.remap, s.atom("ints"), NIL);
            pv.floats = s.field_of(pv.remap, s.atom("floats"), NIL);
            pv.strs = s.field_of(pv.remap, s.atom("strs"), NIL);
            pv.atoms = s.field_of(pv.remap, s.atom("atoms"), NIL);
        }
        s.parts.push_back(pv);
    }
    if (s.parts.empty()) s.parts.push_back(ShareArenas::Part{});
    Value body_k = s.atom("body"), caps_k = s.atom("captures_off");
    std::vector<Value> funcs;
    for (Value cur = resolve(in[0]); is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        Value f = resolve(static_cast<ConsObj*>(as_obj(cur))->head);
        Value body = s.field_of(f, body_k, s.none);
        Value got = body;
        if (s.is_node(body)) {
            s.cur = size_t(ShareArenas::num(body) / s.part_base);
            if (s.cur >= s.parts.size()) return type_fail(p, "share_arenas: a body names a part that is not there");
            got = s.rebuild(make_fixnum(ShareArenas::num(body) % s.part_base));
        }
        Value caps = s.place_captures(f);
        if (s.failed) return type_fail(p, s.failed);
        funcs.push_back(map_insert(p, map_insert(p, f, body_k, got), caps_k, caps));
    }
    if (s.failed) return type_fail(p, s.failed);
    Value nodes = NIL;
    for (size_t i = s.out.size(); i-- > 0;) {
        Value rec = NIL;
        for (int k = 5; k-- > 0;) rec = p.heap().make_cons(s.out[i][k], rec);
        nodes = p.heap().make_cons(rec, nodes);
    }
    Value kids = NIL;
    for (size_t i = s.kids.size(); i-- > 0;) kids = p.heap().make_cons(s.kids[i], kids);
    Value fl = NIL;
    for (size_t i = funcs.size(); i-- > 0;) fl = p.heap().make_cons(funcs[i], fl);
    Value result = p.heap().make_cons(fl, NIL);
    result = p.heap().make_cons(kids, result);
    result = p.heap().make_cons(nodes, result);
    return NativeResult::ok(result);
}

// ---------------------------------------------------------------------------
// The image's two largest sections, written here
// ---------------------------------------------------------------------------
//
// `emit.node_bytes` and `emit.index` in the compiler write a node as twelve
// bytes and a kid as four, through a function call and a `str_le` a field, and
// an image is tens of thousands of each. These write the same bytes in one
// call a section -- `emit`'s tests hold them to the Dream writers -- with the
// opcode numbers the compiler's own table gives (`ir.op_codes`), handed in.

namespace {

/// A node index as the image stores one: the word, or all ones for `:none`.
bool section_index(Process& p, Value v, Value none, std::string* out) {
    v = resolve(v);
    uint32_t w;
    if (v == none) w = 0xFFFFFFFFu;
    else if (is_fixnum(v)) w = uint32_t(uint64_t(fixnum_value(v)));
    else return false;
    (void)p;
    for (int i = 0; i < 4; ++i) out->push_back(char((w >> (8 * i)) & 0xff));
    return true;
}

}  // namespace

/// `node_section [nodes, op_codes]` -- every node as `u8 opcode, u8 flags,
/// u16 0, index a, index b, index c`.
NativeResult vm_node_section(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    forced = resolve(forced);
    if (!is_obj(forced, ObjType::Cons)) return type_fail(p, "node_section needs [nodes, op_codes]");
    Value nodes = resolve(static_cast<ConsObj*>(as_obj(forced))->head);
    Value rest = resolve(static_cast<ConsObj*>(as_obj(forced))->tail);
    if (!is_obj(rest, ObjType::Cons)) return type_fail(p, "node_section needs [nodes, op_codes]");
    Value codes = resolve(static_cast<ConsObj*>(as_obj(rest))->head);
    Value none = make_atom(p.runtime().intern_atom("none"));
    std::string out;
    for (Value cur = nodes; is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        Value n = resolve(static_cast<ConsObj*>(as_obj(cur))->head);
        Value f[5];
        if (is_obj(n, ObjType::Record) && static_cast<RecordObj*>(as_obj(n))->len == 5) {
            // `ir.Node`, the compiler's; the list is what `share_arenas` hands back.
            for (int k = 0; k < 5; ++k) f[k] = resolve(static_cast<RecordObj*>(as_obj(n))->items()[k]);
        } else {
            Value walk = n;
            for (int k = 0; k < 5; ++k) {
                if (!is_obj(walk, ObjType::Cons)) return type_fail(p, "node_section: a node is [op, flags, a, b, c]");
                f[k] = resolve(static_cast<ConsObj*>(as_obj(walk))->head);
                walk = resolve(static_cast<ConsObj*>(as_obj(walk))->tail);
            }
        }
        Value code;
        int64_t op = map_lookup(p, codes, f[0], &code) && is_fixnum(resolve(code)) ? fixnum_value(resolve(code)) : 0;
        if (!is_fixnum(f[1])) return type_fail(p, "node_section: a node's flags are an integer");
        out.push_back(char(op & 0xff));
        out.push_back(char(fixnum_value(f[1]) & 0xff));
        out.push_back('\0');
        out.push_back('\0');
        for (int k = 2; k < 5; ++k) {
            if (!section_index(p, f[k], none, &out)) return type_fail(p, "node_section: an operand is an integer or :none");
        }
    }
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

/// `index_section xs` -- every element as an index: the kids section.
NativeResult vm_index_section(Process& p, Value, Value* args, uint32_t) {
    Value forced;
    {
        VouchesForGc vouch(p);
        if (!force_deep(p, args[0], &forced)) return NativeResult::raise(p.result);
    }
    Value none = make_atom(p.runtime().intern_atom("none"));
    std::string out;
    for (Value cur = resolve(forced); is_obj(cur, ObjType::Cons); cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        if (!section_index(p, static_cast<ConsObj*>(as_obj(cur))->head, none, &out))
            return type_fail(p, "index_section: an index is an integer or :none");
    }
    return NativeResult::ok(p.heap().make_string(out.data(), uint32_t(out.size())));
}

NativeResult vm_wire_decode(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return type_fail(p, "wire_decode needs a string");
    const auto* bytes = reinterpret_cast<const unsigned char*>(b.data);
    WireReader r{p, bytes, b.len};
    // Read only when `read` succeeded, which -O1 cannot see: a TSan build
    // warned that it may be used uninitialized.
    Value v = UNIT;
    const char* why = nullptr;
    // The packed form (pack.cpp) says so in its first byte.
    if (is_packed(bytes, b.len)) {
        if (!wire_unpack(p, bytes, b.len, &v)) why = "not a packed wire value";
    } else if (!r.read(&v, 0)) {
        why = "not a wire value";
    } else if (r.at != r.len) {
        why = "trailing bytes after a wire value";
    }
    Value second = why ? p.heap().make_string(why, uint32_t(std::strlen(why))) : v;
    return NativeResult::ok(make_result(p.heap(), p.runtime(), why == nullptr, second));
}

// --- large data ---
//
// The payload region is the one part of an image the container addresses in 64
// bits (docs/bytecode-format.md, `LDAT`/`PAYL`). It holds whatever the program
// is *about* rather than what the program is -- a corpus, a model, an asset --
// and it is reached from here and nowhere else: no opcode names it, so an image
// that never asks for its payload is an image any reader can run.
//
// Both of these are pure. The payload is fixed when the image is written and
// nothing can alter it, so asking for datum `i` is a function of `i` in exactly
// the way `str_byte` is a function of its index.

NativeResult core_data_count(Process& p, Value, Value*, uint32_t) {
    return NativeResult::ok(make_fixnum(int64_t(p.runtime().image().data_count())));
}

NativeResult core_data_at(Process& p, Value, Value* args, uint32_t) {
    Value v = resolve(args[0]);
    if (!is_fixnum(v)) return type_fail(p, "data_at needs an index");
    int64_t i = fixnum_value(v);
    const Image& img = p.runtime().image();
    if (i < 0 || uint64_t(i) >= img.data_count()) {
        return NativeResult::raise(raise_error(
            p, well_known(p.runtime()).type_error,
            "data_at " + std::to_string(i) + ": this image carries " +
                std::to_string(img.data_count()) +
                " large data. They are declared at compile time with `dreams --payload`."));
    }
    // O(1) and no copy: the value is a length and a pointer into the mapping.
    uint32_t k = uint32_t(i);
    return NativeResult::ok(p.heap().make_bigstr(img.data_bytes(k), img.data_length(k)));
}

}  // namespace

const BuiltinDef BUILTINS[] = {
    {"spawn!", 1, 0b0, bi_spawn},
    {"join!", 1, 0b1, bi_join},
    {"send!", 2, 0b11, bi_send},
    {"recv!", 1, 0b1, bi_recv},
    {"self!", 1, 0b1, bi_self},
    {"raise!", 1, 0b1, bi_raise},
    {"type_of", 1, 0b1, bi_type_of},
    {"to_string", 1, 0b1, bi_to_string},
    {"len", 1, 0b1, bi_len},
    // Mask 0: the argument must arrive unforced, or the deep force below
    // would be handed something already reduced to weak head normal form by
    // the caller and could not report a raise from inside it. The last field
    // is the vouch: `strict!` is the native that most needed one.
    {"strict!", 1, 0b0, bi_strict, true},
    // The container and the key are forced to WHNF for them, by the machine's
    // own continuation (never by a force of their own, which would run a
    // nested machine loop on the C++ stack); the piece they hand back stays
    // unforced, because that is the point.
    {"_match_is_cons", 1, 0b1, bi_match_is_cons},
    {"_match_head", 1, 0b1, bi_match_head},
    {"_match_tail", 1, 0b1, bi_match_tail},
    {"_match_at", 2, 0b11, bi_match_at},
    {"_match_key", 2, 0b11, bi_match_key},
    {"type_assert", 3, 0b001, bi_type_assert},
    {"_list_tail", 1, 0b1, list_tail},
    {"_list_cons", 2, 0b0, list_cons},
    {"_list_is_empty", 1, 0b1, list_is_empty},
    {"_str_chars", 1, 0b1, core_str_chars},
    {"_str_of_chars", 1, 0b1, core_str_of_chars, true},
    {"_str_of_bytes", 1, 0b1, core_str_of_bytes, true},
    {"_str_concat", 1, 0b1, core_str_concat, true},
    {"_error_new", 2, 0b01, core_error_new},
    {"_error_kind", 1, 0b1, core_error_kind},
    {"_error_payload", 1, 0b1, core_error_payload},
    {"_str_slice", 3, 0b111, core_str_slice},
    {"_str_find", 3, 0b111, core_str_find},
    {"_str_byte", 2, 0b11, core_str_byte},
    {"_str_le", 2, 0b11, core_str_le},
    {"_str_span", 3, 0b111, core_str_span},
    {"_str_upto", 3, 0b111, core_str_upto},
    {"_char_code", 1, 0b1, core_char_code},
    {"_char_of_code", 1, 0b1, core_char_of_code},
    {"_to_float", 1, 0b1, core_to_float},
    {"_float_bytes", 1, 0b1, core_float_bytes},
    {"_float_of_bytes", 1, 0b1, core_float_of_bytes},
    {"_to_int", 1, 0b1, core_to_int},
    {"_parse_int", 1, 0b1, core_parse_int},
    {"_parse_float", 1, 0b1, core_parse_float},
    {"_to_existing_atom", 1, 0b1, core_to_existing_atom},
    {"_array_new", 2, 0b01, core_array_new},
    {"_array_of_list", 1, 0b0, core_array_of_list},
    {"_array_to_list", 1, 0b1, core_array_to_list},
    {"_map_has", 2, 0b11, core_map_has},
    {"_map_remove", 2, 0b11, core_map_remove},
    {"_map_pairs", 1, 0b1, core_map_pairs},
    {"_data_count", 1, 0b1, core_data_count},
    {"_data_at", 1, 0b1, core_data_at},
    {"compare", 2, 0b11, core_compare},
    {"_sort_keyed", 2, 0b10, core_sort_keyed, true},
    {"_tensor_matmul", 2, 0b11, tensor_matmul_builtin},
    {"_str_interp", 1, 0b1, core_str_interp},
    {"_mailbox_peek!", 1, 0b1, vm_mailbox_peek},
    {"_mailbox_take!", 1, 0b1, vm_mailbox_take},
    {"_await_message!", 2, 0b11, vm_await_message},
    {"_deadline_in!", 1, 0b1, vm_deadline_in},
    {"_match_fail", 4, 0b1110, core_match_fail},
    {"_error_trace!", 1, 0b1, core_error_trace},
    {"_tensor_of_bytes", 2, 0b11, tensor_of_bytes_builtin},
    {"_record_is", 2, 0b11, core_record_is},
    {"_record_id", 1, 0b1, core_record_id},
    {"_record_fields", 1, 0b1, core_record_fields},
    {"_record_of", 3, 0b111, core_record_of},
    {"_pow", 2, 0b11, core_pow},
    {"_str_char_count", 1, 0b1, core_str_char_count},
    {"_str_ascii_case", 2, 0b11, core_str_ascii_case},
    {"_str_span_back", 3, 0b111, core_str_span_back},
    {"_str_rfind", 3, 0b111, core_str_rfind},
    // Mask 0, as `strict!`, which this is without the effect. It vouches, but
    // is not marked as vouching: the mark keeps the JIT off every function
    // that calls it, and the compiler calls it once per node it emits and per
    // name it resolves -- marked, a self-compile was 1.3% slower, unmarked as
    // fast as `to_string` was, and std's test check fits in its heap either
    // way, since the deep chains it forces are met from interpreted code.
    {"_settled", 1, 0b0, bi_settled},
    {"_forced", 1, 0b1, bi_forced},
    {"defer!", 1, 0b0, bi_defer},
    {"_indexed", 1, 0b0, core_indexed},
    {"_append_rev", 2, 0b01, core_append_rev},
    {"_rev_onto", 2, 0b0, core_rev_onto},
    {"_str_append_rev", 2, 0b01, core_str_append_rev},
    {"_set_of", 1, 0b1, core_set_of},
    {"_set_add", 2, 0b01, core_set_add},
    {"_set_lookup", 2, 0b01, core_set_lookup},
    {"_set_refresh", 2, 0b01, core_set_refresh},
    {"_memo_cells", 2, 0b11, core_memo_cells},
    {"_tally", 1, 0b1, core_tally},
    {"_tally_lookup", 3, 0b000, core_tally_lookup},
    {"_list_drop", 2, 0b01, core_list_drop},
    {"_index_new", 3, 0b110, core_index_new},
    {"_index_find", 2, 0b01, core_index_find},
    {"_heap_of", 1, 0b0, core_heap_of},
    {"_heap_push", 2, 0b01, core_heap_push},
    {"_heap_empty", 1, 0b1, core_heap_empty},
    {"_heap_min", 1, 0b1, core_heap_min},
    {"_heap_pop", 1, 0b1, core_heap_pop},
    {"_trunk_of", 1, 0b0, core_trunk_of, true},
    {"_trunk_to_list", 1, 0b1, core_trunk_to_list},
    {"_trunk_lookup", 2, 0b10, core_trunk_lookup, true},
    {"_trunk_at", 2, 0b11, core_trunk_at},
    {"_trunk_push", 2, 0b10, core_trunk_push},
    {"_trunk_set", 3, 0b101, core_trunk_set},
    {"_trunk_recall", 2, 0b10, core_trunk_recall},
};

uint32_t builtin_count() { return uint32_t(sizeof(BUILTINS) / sizeof(BUILTINS[0])); }


Value make_result(Heap& h, Runtime& rt, bool ok, Value value) {
    const WellKnownAtoms& wk = well_known(rt);
    Value r = h.make_record((ok ? wk.result_ok : wk.result_error) | RECORD_VARIANT, 1);
    static_cast<RecordObj*>(as_obj(r))->items()[0] = value;
    return r;
}

bool compare_for_wire(Process& p, Value a, Value b, int* out) { return compare_values(p, a, b, 0, out); }

}  // namespace dream
