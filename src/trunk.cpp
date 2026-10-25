// Trunks: `std.trunk`'s one container. See TrunkObj in value.hpp for the
// representation and docs/notes/trunks.md for why it is this one.
//
// A program says what it does with a sequence -- push, prepend, read a
// position, search, look a key up, put one -- and this file decides how,
// from what it has seen done. Every answer is the one a list would give, and
// forces the elements a scan of that list would force, in the same order: a
// representation is a cache of the sequence, never a different meaning.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "builtins.hpp"
#include "heap.hpp"
#include "interp.hpp"
#include "process.hpp"
#include "runtime.hpp"
#include "value.hpp"

namespace dream {

namespace {

// --- what was done, for `--stats` -----------------------------------------------

struct TrunkCounters {
    std::atomic<uint64_t> made{0};
    std::atomic<uint64_t> in_place{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<uint64_t> grown{0};
    std::atomic<uint64_t> from_index{0};
    std::atomic<uint64_t> scanned{0};
    std::atomic<uint64_t> indexed{0};
    std::atomic<uint64_t> rerooted{0};
};

TrunkCounters& counters() {
    static TrunkCounters c;
    return c;
}

void count(std::atomic<uint64_t>& c, uint64_t n = 1) { c.fetch_add(n, std::memory_order_relaxed); }

// --- versions -----------------------------------------------------------------

TrunkObj* trunk_of(Value v) { return static_cast<TrunkObj*>(as_obj(resolve(v))); }
bool is_root(const TrunkObj* c) { return is_obj(c->link, ObjType::TrunkStore); }
TrunkStoreObj* store_of(const TrunkObj* c) { return static_cast<TrunkStoreObj*>(as_obj(c->link)); }
uint32_t length(const TrunkObj* c) { return c->end - c->start; }

Value read(Value v, uint32_t i) { return trunk_read(as_obj(resolve(v)), i); }

/// How many versions lie between `c` and its root, counting up to `limit`.
uint32_t depth(const TrunkObj* c, uint32_t limit) {
    uint32_t n = 0;
    while (!is_root(c) && n < limit) {
        c = static_cast<const TrunkObj*>(as_obj(c->link));
        ++n;
    }
    return n;
}

/// Store `v` in a field of `o`, telling the collector: `o` may be old and `v`
/// young, and a mark may be running.
void put_field(Process& p, Obj* o, Value* field, Value v) {
    *field = v;
    if (o->type == ObjType::TrunkStore) {
        auto* st = static_cast<TrunkStoreObj*>(o);
        p.heap().remember_once(o, v, &st->remembered);
    } else {
        p.heap().remember_if_old(o, v);
    }
}

/// May versions of this store be written in place? Not one in the shared area,
/// which every process reads at once.
bool writable(const TrunkObj* c) { return is_root(c) && !is_shared_obj(c) && !is_shared_obj(as_obj(c->link)); }

/// A fresh store holding `[from, to)` as `p.stack[at]` sees it, with `room`
/// free slots on each side, and the root over `[from, to)` it answers.
/// Nothing of the version is changed.
Value copy_out(Process& p, size_t at, uint32_t from, uint32_t to, uint32_t room) {
    const uint32_t n = to - from;
    const uint32_t cap = n + 2 * room;
    Value store = p.heap().make_trunk_store(TRUNK_ORIGIN - n / 2 - room, cap);
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    st->lo = TRUNK_ORIGIN - n / 2;
    st->hi = st->lo + n;
    st->ilo = st->ihi = st->klo = st->khi = st->lo;
    // Allocation is not a safepoint, so neither the version nor the store
    // moves while the slots are filled.
    for (uint32_t i = 0; i < n; ++i) st->slots()[st->lo - st->base + i] = read(p.stack[at], from + i);
    count(counters().copied);
    return p.heap().make_trunk(store, st->lo, st->hi);
}

/// Make the version at `p.stack[at]` read its slots directly when its chain
/// has grown long: a store of its own, holding every slot as it sees them, so
/// that the older versions reading through it still read what they did. The
/// version stays the same object; only how it reads changes.
void reroot_if_deep(Process& p, size_t at) {
    TrunkObj* c = trunk_of(p.stack[at]);
    if (is_root(c) || is_shared_obj(c) || depth(c, 9) < 9) return;
    // The slots every version of the chain can reach are those of the root's
    // store; the root is where they are counted.
    const TrunkObj* r = c;
    while (!is_root(r)) r = static_cast<const TrunkObj*>(as_obj(r->link));
    const uint32_t lo = store_of(r)->lo, hi = store_of(r)->hi;
    Value fresh = copy_out(p, at, lo, hi, 0);
    c = trunk_of(p.stack[at]);
    Value store = trunk_of(fresh)->link;
    // The copy put position `lo` at its own origin; renumber it to `lo`.
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    const int64_t shift = int64_t(lo) - int64_t(st->lo);
    st->base = uint32_t(int64_t(st->base) + shift);
    st->lo = lo;
    st->hi = hi;
    st->ilo = st->ihi = st->klo = st->khi = lo;
    c->pos = TRUNK_NO_POS;
    put_field(p, c, &c->value, UNIT);
    put_field(p, c, &c->link, store);
    count(counters().rerooted);
}

// --- keys ------------------------------------------------------------------------

/// What an evaluated value is to an index.
enum class Kind {
    Key,        // flat: equal to another value exactly when their keys are
    Unequal,    // structured: equal to no flat value
    Never,      // NaN: equal to nothing
    Irregular,  // a big integer, or a float an integer could equal: not keyed
};

/// The key an evaluated value is filed under. Numbers are equal across
/// representations (`1 == 1.0`), so an integral float that fits a fixnum is
/// filed as that fixnum and NaN, equal to nothing, under nothing; a big
/// integer, or a float an integer too large for a fixnum could equal, is not
/// filed at all, and a search for one scans.
Kind key_of(Value v, Value* key) {
    v = resolve(v);
    if (!is_ptr(v)) {
        *key = v;
        return Kind::Key;
    }
    Obj* o = as_obj(v);
    switch (o->type) {
        case ObjType::Str:
        case ObjType::BigStr:
        case ObjType::Pid:
            *key = v;
            return Kind::Key;
        case ObjType::Record:
            if (static_cast<RecordObj*>(o)->len == 0) {
                *key = v;
                return Kind::Key;
            }
            return Kind::Unequal;
        case ObjType::Float: {
            double d = static_cast<FloatObj*>(o)->value;
            if (d != d) return Kind::Never;
            if (d == double(int64_t(d)) && d > -4.0e18 && d < 4.0e18) {
                *key = make_fixnum(int64_t(d));
                return Kind::Key;
            }
            if (d == std::floor(d)) return Kind::Irregular;
            *key = v;
            return Kind::Key;
        }
        case ObjType::BigInt:
            return Kind::Irregular;
        default:
            return Kind::Unequal;
    }
}

/// The key of an element that is a list, by its first item: what an
/// association list is searched by. A list whose first item is not yet
/// evaluated is `Irregular` to the caller, who forces it first.
Kind pair_key(Value elem, Value* key) {
    elem = resolve(elem);
    if (!is_obj(elem, ObjType::Cons)) return Kind::Unequal;
    Value head = resolve(static_cast<ConsObj*>(as_obj(elem))->head);
    if (!is_whnf(head)) return Kind::Irregular;
    return key_of(head, key);
}

/// `positions` with `pos` in front.
Value with_position(Process& p, Value map, Value key, uint32_t pos) {
    Value had;
    Value list = map_lookup(p, map, key, &had) ? had : NIL;
    return map_insert(p, map, key, p.heap().make_cons(make_fixnum(int64_t(pos)), list));
}

/// `positions` without `pos`.
Value without_position(Process& p, Value map, Value key, uint32_t pos) {
    Value had;
    if (!map_lookup(p, map, key, &had)) return map;
    std::vector<Value> keep;
    for (Value c = resolve(had); is_obj(c, ObjType::Cons); c = resolve(static_cast<ConsObj*>(as_obj(c))->tail)) {
        Value at = resolve(static_cast<ConsObj*>(as_obj(c))->head);
        if (fixnum_value(at) != int64_t(pos)) keep.push_back(at);
    }
    if (keep.empty()) return map_erase(p, map, key);
    Value list = NIL;
    for (size_t i = keep.size(); i-- > 0;) list = p.heap().make_cons(keep[i], list);
    return map_insert(p, map, key, list);
}

// --- the two indexes -------------------------------------------------------------
//
// `index` is by element, `keyed` by an element's first item. They are kept
// the same way, so each is a `Which` and the code below is written once.

enum class Which { Elements, Keys };

struct Cover {
    Value* map;
    uint32_t* lo;
    uint32_t* hi;
};

Cover cover(TrunkStoreObj* st, Which w) {
    if (w == Which::Elements) return {&st->index, &st->ilo, &st->ihi};
    return {&st->keyed, &st->klo, &st->khi};
}

Kind kind_for(Which w, Value elem, Value* key) {
    return w == Which::Elements ? key_of(elem, key) : pair_key(elem, key);
}

/// Forget the index, keeping nothing: what a write the index cannot follow
/// leaves.
void drop_index(Process& p, TrunkStoreObj* st, Which w, uint32_t at) {
    Cover c = cover(st, w);
    put_field(p, st, c.map, UNIT);
    *c.lo = *c.hi = at;
}

/// Slot `q` of the root's store is about to hold `now` rather than `was`:
/// keep the index true. `was` is evaluated wherever the index covers it.
void note_write(Process& p, Value store, Which w, uint32_t q, Value was, Value now) {
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    Cover c = cover(st, w);
    if (*c.map == UNIT) return;
    const bool inside = q >= *c.lo && q < *c.hi;
    const bool next = q == *c.hi;
    if (!inside && !next) return;
    Value key;
    if (inside && kind_for(w, was, &key) == Kind::Key) {
        Value m = without_position(p, *c.map, key, q);
        st = static_cast<TrunkStoreObj*>(as_obj(store));
        put_field(p, st, cover(st, w).map, m);
    }
    st = static_cast<TrunkStoreObj*>(as_obj(store));
    c = cover(st, w);
    // What is written in is only filed if it is already evaluated: filing it
    // would otherwise force it, which the program as written never asked for.
    Value v = resolve(now);
    Kind k = is_whnf(v) ? kind_for(w, v, &key) : Kind::Irregular;
    if (k == Kind::Irregular) {
        if (next) return;
        if (q + 1 == *c.hi) *c.hi = q;
        else if (q == *c.lo) *c.lo = q + 1;
        else drop_index(p, st, w, *c.lo);
        return;
    }
    if (k == Kind::Key) {
        Value m = with_position(p, *c.map, key, q);
        st = static_cast<TrunkStoreObj*>(as_obj(store));
        put_field(p, st, cover(st, w).map, m);
    }
    if (next) ++*cover(static_cast<TrunkStoreObj*>(as_obj(store)), w).hi;
}

// --- writing, by the root --------------------------------------------------------

/// The root at `p.stack[at]`, its store grown to hold position `q`. Positions
/// never change, so the indexes carry over as they are, and the old store,
/// which only this root read, is garbage. Room is left on the side that grew,
/// as much again as the trunk holds, so a run of pushes or of prepends
/// grows the store a logarithmic number of times.
void grow_to(Process& p, size_t at, uint32_t q) {
    TrunkObj* c = trunk_of(p.stack[at]);
    TrunkStoreObj* st = store_of(c);
    if (q >= st->base && q - st->base < st->cap) return;
    const uint32_t lo = std::min(st->lo, q), hi = std::max(st->hi, q + 1);
    const uint32_t n = hi - lo;
    const uint32_t room = std::max<uint32_t>(n, 8);
    const uint32_t left = q < st->lo ? room : room / 8;
    const uint32_t right = q >= st->hi ? room : room / 8;
    Value store = p.heap().make_trunk_store(lo - left, n + left + right);
    c = trunk_of(p.stack[at]);
    st = store_of(c);
    auto* fresh = static_cast<TrunkStoreObj*>(as_obj(store));
    fresh->lo = st->lo;
    fresh->hi = st->hi;
    fresh->ilo = st->ilo;
    fresh->ihi = st->ihi;
    fresh->klo = st->klo;
    fresh->khi = st->khi;
    fresh->work = st->work;
    fresh->index = st->index;
    fresh->keyed = st->keyed;
    for (uint32_t i = st->lo; i < st->hi; ++i) fresh->slots()[i - fresh->base] = st->slots()[i - st->base];
    put_field(p, c, &c->link, store);
    count(counters().grown);
}

/// The next version after the root at `p.stack[at]`: bounds `[start, end)`,
/// and, when `write`, position `q` holding `p.stack[val]`. The root becomes
/// a version like the new one but for the bounds and, where it held one,
/// slot `q`'s old value. Answers the new root.
Value advance(Process& p, size_t at, uint32_t start, uint32_t end, bool write, uint32_t q, size_t val) {
    if (write) grow_to(p, at, q);
    Value store = trunk_of(p.stack[at])->link;
    Value next = p.heap().make_trunk(store, start, end);
    TrunkObj* old = trunk_of(p.stack[at]);
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    if (write) {
        Value now = p.stack[val];
        const bool used = q >= st->lo && q < st->hi;
        Value was = used ? st->slots()[q - st->base] : UNIT;
        if (used) {
            old->pos = q;
            put_field(p, old, &old->value, was);
        }
        note_write(p, store, Which::Elements, q, was, now);
        note_write(p, store, Which::Keys, q, was, now);
        st = static_cast<TrunkStoreObj*>(as_obj(store));
        put_field(p, st, &st->slots()[q - st->base], p.stack[val]);
        if (q < st->lo) st->lo = q;
        if (q >= st->hi) st->hi = q + 1;
        old = trunk_of(p.stack[at]);
    }
    put_field(p, old, &old->link, next);
    count(counters().in_place);
    return next;
}

/// A version of `p.stack[at]` that may be written, at `p.stack[at]`: the root
/// itself, or a copy when it is not one -- an old version written again is a
/// branch, and a branch is a store of its own.
void writable_at(Process& p, size_t at) {
    TrunkObj* c = trunk_of(p.stack[at]);
    if (writable(c)) return;
    const uint32_t n = length(c);
    p.stack[at] = copy_out(p, at, c->start, c->end, std::max<uint32_t>(n / 2, 4));
}

// --- scans -------------------------------------------------------------------------

/// What a search is asked.
enum class Ask { Contains, IndexOf, Count, Lookup };

/// The answer to a search of `p.stack[base]` for `p.stack[base + 1]`, the
/// rest of the stack from `base` its own: a position relative to the start
/// (`Contains`, `IndexOf`, `Lookup`) or -1, or how many (`Count`). False
/// when a force raised or parked.
///
/// In three steps, which together force exactly what a scan of the sequence
/// from its start would, in that order: the positions before what the index
/// covers, scanned; those it covers, answered by it, all already evaluated;
/// and those after it, scanned, the index growing behind the scan. A version
/// that is not the root, or a search for something no key can stand for,
/// is all scan.
bool search(Process& p, size_t base, Ask ask, int64_t* out) {
    const Which w = ask == Ask::Lookup ? Which::Keys : Which::Elements;
    const bool all = ask == Ask::Count;
    int64_t found = all ? 0 : -1;
    reroot_if_deep(p, base);
    TrunkObj* c = trunk_of(p.stack[base]);
    const uint32_t s = c->start, e = c->end;
    if (s == e) {
        *out = found;
        return true;
    }
    // The needle is evaluated first, as `==` evaluates its left side first --
    // and only when there is something to compare it with.
    Value x;
    if (!force_whnf(p, p.stack[base + 1], &x)) return false;
    p.stack[base + 1] = x;
    Value qkey = UNIT;
    const bool keyed_query = key_of(x, &qkey) == Kind::Key;

    // One step of the scan: evaluate position `i` (and, for a lookup, its
    // first item) and say whether it matches. -1 when a force failed.
    auto step = [&](uint32_t i) -> int {
        p.stack[base + 2] = read(p.stack[base], i);
        Value elem;
        if (!force_whnf(p, p.stack[base + 2], &elem)) return -1;
        p.stack[base + 2] = elem;
        count(counters().scanned);
        if (ask == Ask::Lookup) {
            if (!is_obj(elem, ObjType::Cons)) return 0;
            Value head;
            if (!force_whnf(p, static_cast<ConsObj*>(as_obj(elem))->head, &head)) return -1;
            elem = head;
        }
        bool raised = false;
        bool same = equal_values(p, p.stack[base + 1], elem, &raised, 0);
        if (raised) return -1;
        return same ? 1 : 0;
    };
    auto scan = [&](uint32_t from, uint32_t to) -> int {
        for (uint32_t i = from; i < to; ++i) {
            int r = step(i);
            if (r < 0) return -1;
            if (r == 1) {
                if (!all) {
                    found = int64_t(i - s);
                    return 1;
                }
                ++found;
            }
        }
        return 0;
    };

    c = trunk_of(p.stack[base]);
    if (!keyed_query || !writable(c)) {
        int r = scan(s, e);
        if (r < 0) return false;
        *out = found;
        return true;
    }
    {
        // Index once scanning has cost as much as the trunk is long:
        // one search is cheaper as a scan, and the second pays for it.
        TrunkStoreObj* st = store_of(c);
        Cover cv = cover(st, w);
        if (*cv.map == UNIT) {
            st->work += 1;
            if (st->work < uint64_t(st->hi - st->lo) + 16) {
                st->work += e - s;
                int r = scan(s, e);
                if (r < 0) return false;
                *out = found;
                return true;
            }
            Value m = p.heap().make_map(0);
            st = store_of(trunk_of(p.stack[base]));
            put_field(p, st, cover(st, w).map, m);
            *cover(st, w).lo = *cover(st, w).hi = s;
        } else if (s > *cv.hi || e < *cv.lo) {
            // The index covers nothing next to this version: start again here.
            drop_index(p, st, w, s);
            Value m = p.heap().make_map(0);
            st = store_of(trunk_of(p.stack[base]));
            put_field(p, st, cover(st, w).map, m);
        }
    }

    // Before what the index covers. Filed only when scanned to the end, so
    // that what it covers stays one run.
    {
        TrunkStoreObj* st = store_of(trunk_of(p.stack[base]));
        const uint32_t ilo = *cover(st, w).lo;
        if (s < ilo) {
            const uint32_t to = std::min(e, ilo);
            int r = scan(s, to);
            if (r < 0) return false;
            if (r == 1) {
                *out = found;
                return true;
            }
            if (to == ilo) {
                bool filed = true;
                for (uint32_t i = ilo; i-- > s;) {
                    st = store_of(trunk_of(p.stack[base]));
                    Value key;
                    Kind k = kind_for(w, st->slots()[i - st->base], &key);
                    if (k == Kind::Irregular) {
                        filed = false;
                        break;
                    }
                    if (k == Kind::Key) {
                        Value m = with_position(p, *cover(st, w).map, key, i);
                        st = store_of(trunk_of(p.stack[base]));
                        put_field(p, st, cover(st, w).map, m);
                    }
                    *cover(st, w).lo = i;
                    count(counters().indexed);
                }
                (void)filed;
            }
        }
    }

    // What the index covers.
    {
        TrunkStoreObj* st = store_of(trunk_of(p.stack[base]));
        Cover cv = cover(st, w);
        const uint32_t from = std::max(s, *cv.lo), to = std::min(e, *cv.hi);
        if (from < to) {
            Value positions;
            int64_t best = -1;
            if (map_lookup(p, *cv.map, qkey, &positions)) {
                for (Value l = resolve(positions); is_obj(l, ObjType::Cons);
                     l = resolve(static_cast<ConsObj*>(as_obj(l))->tail)) {
                    int64_t at = fixnum_value(resolve(static_cast<ConsObj*>(as_obj(l))->head));
                    if (at < int64_t(from) || at >= int64_t(to)) continue;
                    if (all) ++found;
                    else if (best < 0 || at < best) best = at;
                }
            }
            count(counters().from_index);
            if (!all && best >= 0) {
                *out = best - int64_t(s);
                return true;
            }
        }
    }

    // After it: scanned, and filed as the scan goes while it is next to it.
    {
        TrunkStoreObj* st = store_of(trunk_of(p.stack[base]));
        uint32_t from = std::max(s, *cover(st, w).hi);
        for (uint32_t i = from; i < e; ++i) {
            int r = step(i);
            if (r < 0) return false;
            st = store_of(trunk_of(p.stack[base]));
            Cover cv = cover(st, w);
            if (*cv.map != UNIT && *cv.hi == i) {
                Value key;
                Kind k = kind_for(w, p.stack[base + 2], &key);
                if (k == Kind::Key) {
                    Value m = with_position(p, *cv.map, key, i);
                    st = store_of(trunk_of(p.stack[base]));
                    put_field(p, st, cover(st, w).map, m);
                }
                if (k != Kind::Irregular) {
                    *cover(st, w).hi = i + 1;
                    count(counters().indexed);
                }
            }
            if (r == 1) {
                if (!all) {
                    *out = int64_t(i - s);
                    return true;
                }
                ++found;
            }
        }
    }
    *out = found;
    return true;
}

NativeResult failed(Process& p, size_t base) {
    if (!p.force_blocked) p.stack.resize(base);
    return NativeResult::raise(p.result);
}

NativeResult answer(Process& p, size_t base, Value v) {
    p.stack.resize(base);
    return NativeResult::ok(v);
}

NativeResult not_a_trunk(Process& p, const char* who) {
    return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                           std::string(who) + " needs a trunk"));
}

bool is_trunk(Value v) { return is_obj(resolve(v), ObjType::Trunk); }

// --- the natives -------------------------------------------------------------------

/// `vm.trunk_of xs`: the trunk of a list's elements, which stay as lazy as
/// they were. The spine is walked, so a trunk is finite.
NativeResult trunk_from_list(Process& p, Value, Value* args, uint32_t) {
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(args[0]);
    uint32_t n = 0;
    for (Value cur = args[0];;) {
        Value cell;
        if (!force_whnf(p, cur, &cell)) return failed(p, base);
        if (is_nil(cell)) break;
        if (!is_obj(cell, ObjType::Cons)) {
            p.stack.resize(base);
            return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                                   "trunk.of needs a list"));
        }
        ++n;
        cur = static_cast<ConsObj*>(as_obj(cell))->tail;
    }
    const uint32_t room = std::max<uint32_t>(n / 4, 4);
    Value store = p.heap().make_trunk_store(TRUNK_ORIGIN - n / 2 - room, n + 2 * room);
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    st->lo = TRUNK_ORIGIN - n / 2;
    st->hi = st->lo + n;
    st->ilo = st->ihi = st->klo = st->khi = st->lo;
    uint32_t i = 0;
    for (Value cur = resolve(p.stack[base]); is_obj(cur, ObjType::Cons);
         cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail)) {
        st->slots()[st->lo - st->base + i++] = static_cast<ConsObj*>(as_obj(cur))->head;
    }
    count(counters().made);
    return answer(p, base, p.heap().make_trunk(store, st->lo, st->hi));
}

/// `vm.trunk_to_list c`: its elements as a list, in order and as lazy as they are.
NativeResult trunk_to_list(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[0])) return not_a_trunk(p, "trunk.to_list");
    Value c = resolve(args[0]);
    TrunkObj* o = trunk_of(c);
    Value list = NIL;
    for (uint32_t i = o->end; i-- > o->start;) list = p.heap().make_cons(read(c, i), list);
    return NativeResult::ok(list);
}

/// `vm.trunk_len c`.
NativeResult trunk_len(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[0])) return not_a_trunk(p, "trunk.len");
    return NativeResult::ok(make_fixnum(int64_t(length(trunk_of(args[0])))));
}

/// `vm.trunk_at i c`: the element at `i`, counted from 0, unevaluated.
NativeResult trunk_at(Process& p, Value, Value* args, uint32_t) {
    Value i = resolve(args[0]);
    if (!is_trunk(args[1])) return not_a_trunk(p, "trunk.at");
    TrunkObj* c = trunk_of(args[1]);
    if (!is_fixnum(i) || fixnum_value(i) < 0 || fixnum_value(i) >= int64_t(length(c))) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).out_of_bounds,
                                               "trunk.at: no element at that position"));
    }
    return NativeResult::enter(read(args[1], c->start + uint32_t(fixnum_value(i))));
}

/// `vm.trunk_push x c`: `c` with `x` after its last element.
NativeResult trunk_push(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[1])) return not_a_trunk(p, "trunk.push");
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[1]));
    p.stack.push_back(args[0]);
    writable_at(p, base);
    TrunkObj* c = trunk_of(p.stack[base]);
    return answer(p, base, advance(p, base, c->start, c->end + 1, true, c->end, base + 1));
}

/// `vm.trunk_cons x c`: `c` with `x` before its first element.
NativeResult trunk_cons(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[1])) return not_a_trunk(p, "trunk.cons");
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[1]));
    p.stack.push_back(args[0]);
    writable_at(p, base);
    TrunkObj* c = trunk_of(p.stack[base]);
    return answer(p, base, advance(p, base, c->start - 1, c->end, true, c->start - 1, base + 1));
}

/// `vm.trunk_set i x c`: `c` with the element at `i` replaced by `x`.
NativeResult trunk_set(Process& p, Value, Value* args, uint32_t) {
    Value i = resolve(args[0]);
    if (!is_trunk(args[2])) return not_a_trunk(p, "trunk.set");
    if (!is_fixnum(i) || fixnum_value(i) < 0 || fixnum_value(i) >= int64_t(length(trunk_of(args[2])))) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).out_of_bounds,
                                               "trunk.set: no element at that position"));
    }
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[2]));
    p.stack.push_back(args[1]);
    writable_at(p, base);
    TrunkObj* c = trunk_of(p.stack[base]);
    return answer(p, base, advance(p, base, c->start, c->end, true, c->start + uint32_t(fixnum_value(i)), base + 1));
}

/// `vm.trunk_slice from to c`: the elements at `[from, to)`, clamped to `c`.
/// What `rest`, `drop` and `take` are. The root moves to it, which is what
/// lets a queue -- push at one end, take from the other -- be written in place.
NativeResult trunk_slice(Process& p, Value, Value* args, uint32_t) {
    Value a = resolve(args[0]), b = resolve(args[1]);
    if (!is_trunk(args[2])) return not_a_trunk(p, "trunk.slice");
    if (!is_fixnum(a) || !is_fixnum(b)) {
        return NativeResult::raise(raise_error(p, well_known(p.runtime()).type_error,
                                               "trunk.slice needs integer bounds"));
    }
    TrunkObj* c = trunk_of(args[2]);
    const int64_t n = length(c);
    const int64_t from = std::clamp<int64_t>(fixnum_value(a), 0, n);
    const int64_t to = std::clamp<int64_t>(fixnum_value(b), from, n);
    if (from == 0 && to == n) return NativeResult::ok(resolve(args[2]));
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[2]));
    if (!writable(c)) {
        p.stack[base] = copy_out(p, base, c->start + uint32_t(from), c->start + uint32_t(to), 0);
        return answer(p, base, p.stack[base]);
    }
    // Moving the root only narrows what it reads: the store still holds every
    // slot it ever had, the slices of earlier versions among them, and
    // `grow_to` carries all of them on. So a queue -- pushed at the back,
    // taken from the front -- would keep everything it was ever given. Once
    // the slots outside the slice outnumber those in it, the slice gets a
    // store of its own instead, and the old one goes with the old versions.
    // That copies what is kept only after as many slots have been let go,
    // which keeps a queue's push and `rest` constant time, amortized.
    const TrunkStoreObj* st = store_of(c);
    const uint32_t kept = uint32_t(to - from);
    const uint32_t outside = (st->hi - st->lo) - kept;
    if (outside > std::max<uint32_t>(kept, 32)) {
        p.stack[base] = copy_out(p, base, c->start + uint32_t(from), c->start + uint32_t(to),
                                 std::max<uint32_t>(kept / 2, 4));
        return answer(p, base, p.stack[base]);
    }
    return answer(p, base, advance(p, base, c->start + uint32_t(from), c->start + uint32_t(to), false, 0, 0));
}

/// `vm.trunk_concat a b`: `a`'s elements, then `b`'s. `b`'s are written after
/// `a`'s, in place when `a` is a root.
NativeResult trunk_concat(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[0]) || !is_trunk(args[1])) return not_a_trunk(p, "trunk.concat");
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[0]));
    p.stack.push_back(resolve(args[1]));
    p.stack.push_back(UNIT);
    const uint32_t n = length(trunk_of(p.stack[base + 1]));
    if (n == 0) return answer(p, base, p.stack[base]);
    writable_at(p, base);
    const uint32_t from = trunk_of(p.stack[base + 1])->start;
    // Past the root's end, where no version has ever written, there is no old
    // value for the root to keep: the store grows once and every element goes
    // in behind one new version. A version per element is what the general
    // way below makes, and a line of terminal output written that way was
    // a hundred of them.
    if (TrunkObj* c = trunk_of(p.stack[base]); c->end >= store_of(c)->hi) {
        const uint32_t start = c->start, end = c->end;
        grow_to(p, base, end + n - 1);
        Value store = trunk_of(p.stack[base])->link;
        for (uint32_t i = 0; i < n; ++i) {
            // Allocation is not a safepoint, so the store stays where it is.
            p.stack[base + 2] = read(p.stack[base + 1], from + i);
            note_write(p, store, Which::Elements, end + i, UNIT, p.stack[base + 2]);
            note_write(p, store, Which::Keys, end + i, UNIT, p.stack[base + 2]);
            auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
            put_field(p, st, &st->slots()[end + i - st->base], p.stack[base + 2]);
            st->hi = end + i + 1;
        }
        return answer(p, base, advance(p, base, start, end + n, false, 0, 0));
    }
    for (uint32_t i = 0; i < n; ++i) {
        p.stack[base + 2] = read(p.stack[base + 1], from + i);
        TrunkObj* c = trunk_of(p.stack[base]);
        p.stack[base] = advance(p, base, c->start, c->end + 1, true, c->end, base + 2);
    }
    return answer(p, base, p.stack[base]);
}

/// Whether two keys are the same key without forcing either: the same object,
/// or equal strings. What a lookup was asked then holds exactly.
bool same_key(Value k, Value m) {
    k = resolve(k);
    m = resolve(m);
    if (k == m) return true;
    if (!is_obj(k, ObjType::Str) || !is_obj(m, ObjType::Str)) return false;
    auto *a = static_cast<StrObj*>(as_obj(k)), *b = static_cast<StrObj*>(as_obj(m));
    return a->len == b->len && std::memcmp(a->data(), b->data(), a->len) == 0;
}

/// What version `c` remembers of a lookup of `k` (`TrunkObj::memo_key`), or
/// `TRUNK_NO_MEMO`.
int64_t remembered(Value c, Value k) {
    const TrunkObj* t = trunk_of(c);
    if (t->memo_at == TRUNK_NO_MEMO || !same_key(k, t->memo_key)) return TRUNK_NO_MEMO;
    return t->memo_at;
}

NativeResult searched(Process& p, Value* args, Ask ask, const char* who) {
    if (!is_trunk(args[1])) return not_a_trunk(p, who);
    if (ask == Ask::Lookup) {
        const int64_t at = remembered(resolve(args[1]), args[0]);
        if (at != TRUNK_NO_MEMO) return NativeResult::ok(make_fixnum(at));
    }
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[1]));
    p.stack.push_back(args[0]);
    p.stack.push_back(UNIT);
    int64_t got = 0;
    if (!search(p, base, ask, &got)) return failed(p, base);
    // Remembered by the version, unless it is in a shared area, which
    // nothing writes once it is made.
    if (ask == Ask::Lookup && !is_shared_obj(trunk_of(p.stack[base]))) {
        TrunkObj* c = trunk_of(p.stack[base]);
        put_field(p, c, &c->memo_key, p.stack[base + 1]);
        c->memo_at = got;
    }
    if (ask == Ask::Contains) return answer(p, base, make_bool(got >= 0));
    return answer(p, base, make_fixnum(got));
}

/// `vm.trunk_contains x c`: whether some element is `== x`.
NativeResult trunk_contains(Process& p, Value, Value* args, uint32_t) {
    return searched(p, args, Ask::Contains, "trunk.contains");
}
/// `vm.trunk_index_of x c`: the position of the first element `== x`, or -1.
NativeResult trunk_index_of(Process& p, Value, Value* args, uint32_t) {
    return searched(p, args, Ask::IndexOf, "trunk.index_of");
}
/// `vm.trunk_count x c`: how many elements are `== x`.
NativeResult trunk_count(Process& p, Value, Value* args, uint32_t) {
    return searched(p, args, Ask::Count, "trunk.count");
}
/// `vm.trunk_lookup k c`: the position of the first element that is a list
/// whose first item is `== k`, or -1.
NativeResult trunk_lookup(Process& p, Value, Value* args, uint32_t) {
    return searched(p, args, Ask::Lookup, "trunk.lookup");
}

/// `vm.trunk_put k v c`: `c` with the first element whose first item is `== k`
/// replaced by `[k, v]`, or with `[k, v]` pushed when there is none. The
/// search is `lookup`'s, forcing what it forces.
NativeResult trunk_put(Process& p, Value, Value* args, uint32_t) {
    if (!is_trunk(args[2])) return not_a_trunk(p, "trunk.put");
    VouchesForGc vouch(p);
    const size_t base = p.stack.size();
    p.stack.push_back(resolve(args[2]));
    p.stack.push_back(args[0]);
    p.stack.push_back(UNIT);
    p.stack.push_back(args[1]);
    int64_t at = 0;
    if (!search(p, base, Ask::Lookup, &at)) return failed(p, base);
    // The key as evaluated: the search evaluated it, unless there was nothing
    // to search, and then it is filed unevaluated, as a list literal would be.
    Value pair = p.heap().make_cons(p.stack[base + 1], p.heap().make_cons(p.stack[base + 3], NIL));
    p.stack[base + 2] = pair;
    writable_at(p, base);
    TrunkObj* c = trunk_of(p.stack[base]);
    if (at >= 0) {
        return answer(p, base, advance(p, base, c->start, c->end, true, c->start + uint32_t(at), base + 2));
    }
    return answer(p, base, advance(p, base, c->start, c->end + 1, true, c->end, base + 2));
}

}  // namespace

// The same operations as builtins, for what the supercompiler writes: a loop
// it carries in a trunk where the program kept a list (`upserted` in
// dreams/src/accum.dr) may belong to a program that never imports `std.trunk`,
// and a builtin is reached by its index rather than through a module.
NativeResult core_trunk_of(Process& p, Value self, Value* args, uint32_t n) { return trunk_from_list(p, self, args, n); }
NativeResult core_trunk_to_list(Process& p, Value self, Value* args, uint32_t n) { return trunk_to_list(p, self, args, n); }
NativeResult core_trunk_lookup(Process& p, Value self, Value* args, uint32_t n) { return trunk_lookup(p, self, args, n); }
NativeResult core_trunk_at(Process& p, Value self, Value* args, uint32_t n) { return trunk_at(p, self, args, n); }
NativeResult core_trunk_push(Process& p, Value self, Value* args, uint32_t n) { return trunk_push(p, self, args, n); }
NativeResult core_trunk_set(Process& p, Value self, Value* args, uint32_t n) { return trunk_set(p, self, args, n); }

/// `_trunk_recall k c`: the position the last lookup of `k` in this very
/// version `c` found (`TrunkObj::memo_key`), -1 for an empty trunk, or `()`
/// when its last lookup was not of `k`. Forces nothing: `k` arrives as it is,
/// and is the same key when it is the same object or an equal string.
NativeResult core_trunk_recall(Process& p, Value, Value* args, uint32_t) {
    const Value c = resolve(args[1]);
    if (!is_trunk(c)) return not_a_trunk(p, "_trunk_recall");
    if (length(trunk_of(c)) == 0) return NativeResult::ok(make_fixnum(-1));
    const int64_t at = remembered(c, args[0]);
    return NativeResult::ok(at != TRUNK_NO_MEMO ? make_fixnum(at) : UNIT);
}

void add_trunk_natives(std::vector<NativeDef>& members) {
    members.push_back({"trunk_of", 1, 0b0, trunk_from_list, 0, true});
    members.push_back({"trunk_to_list", 1, 0b1, trunk_to_list});
    members.push_back({"trunk_len", 1, 0b1, trunk_len});
    members.push_back({"trunk_at", 2, 0b11, trunk_at});
    members.push_back({"trunk_push", 2, 0b10, trunk_push});
    members.push_back({"trunk_cons", 2, 0b10, trunk_cons});
    members.push_back({"trunk_set", 3, 0b101, trunk_set});
    members.push_back({"trunk_slice", 3, 0b111, trunk_slice});
    members.push_back({"trunk_concat", 2, 0b11, trunk_concat});
    members.push_back({"trunk_contains", 2, 0b10, trunk_contains, 0, true});
    members.push_back({"trunk_index_of", 2, 0b10, trunk_index_of, 0, true});
    members.push_back({"trunk_count", 2, 0b10, trunk_count, 0, true});
    members.push_back({"trunk_lookup", 2, 0b10, trunk_lookup, 0, true});
    members.push_back({"trunk_put", 3, 0b100, trunk_put, 0, true});
}

bool trunk_equal(Process& p, Value a, Value b, bool* raised, int depth) {
    TrunkObj* x = trunk_of(a);
    TrunkObj* y = trunk_of(b);
    if (length(x) != length(y)) return false;
    const uint32_t n = length(x), xs = x->start, ys = y->start;
    for (uint32_t i = 0; i < n; ++i) {
        if (!equal_values(p, read(a, xs + i), read(b, ys + i), raised, depth + 1)) return false;
        if (*raised) return false;
    }
    return true;
}

uint32_t trunk_length(Value c) { return length(trunk_of(c)); }

Value trunk_of_values(Process& p, const std::vector<Value>& items) {
    const uint32_t n = uint32_t(items.size());
    const uint32_t room = std::max<uint32_t>(n / 4, 4);
    Value store = p.heap().make_trunk_store(TRUNK_ORIGIN - n / 2 - room, n + 2 * room);
    auto* st = static_cast<TrunkStoreObj*>(as_obj(store));
    st->lo = TRUNK_ORIGIN - n / 2;
    st->hi = st->lo + n;
    st->ilo = st->ihi = st->klo = st->khi = st->lo;
    for (uint32_t i = 0; i < n; ++i) st->slots()[st->lo - st->base + i] = items[i];
    count(counters().made);
    return p.heap().make_trunk(store, st->lo, st->hi);
}
Value trunk_element(Value c, uint32_t i) { return read(c, trunk_of(c)->start + i); }

void print_trunk_stats() {
    TrunkCounters& c = counters();
    const uint64_t made = c.made.load();
    if (made == 0 && c.in_place.load() == 0) return;
    std::fprintf(stderr,
                 "; trunk: %llu made, %llu written in place, %llu copied, %llu stores grown, "
                 "%llu elements scanned, %llu filed in an index, %llu searches answered by one, "
                 "%llu old versions rerooted\n",
                 (unsigned long long)made, (unsigned long long)c.in_place.load(),
                 (unsigned long long)c.copied.load(), (unsigned long long)c.grown.load(),
                 (unsigned long long)c.scanned.load(), (unsigned long long)c.indexed.load(),
                 (unsigned long long)c.from_index.load(), (unsigned long long)c.rerooted.load());
}

}  // namespace dream
