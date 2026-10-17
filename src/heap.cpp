#include <unordered_map>
#include "heap.hpp"

#include <shared_mutex>
#include <unordered_set>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <unordered_set>

#include "gc_pool.hpp"
#include "interp.hpp"

namespace dream {

namespace {
constexpr size_t ALIGN = 8;
inline size_t align_up(size_t n) { return (n + ALIGN - 1) & ~(ALIGN - 1); }

/// A free chunk is an object-shaped slot whose header still answers the chunk
/// size (so the sweep can walk a block as a run of chunks) and whose first
/// payload word links to the next free chunk of the same size class.
inline Obj* free_next(Obj* o) {
    return reinterpret_cast<Obj*>(reinterpret_cast<Value*>(o + 1)[0]);
}
inline void set_free_next(Obj* o, Obj* next) {
    reinterpret_cast<Value*>(o + 1)[0] = reinterpret_cast<Value>(next);
}

/// True for the object kinds whose payload holds no references -- Float, Str,
/// Pid, BigInt and BigStr carry only raw bytes, a host id, or a pointer into
/// the image, which is not heap memory and is never collected. Marking one is a
/// no-op, so the collector need not put it on the worklist; the mark bit alone
/// keeps it.
///
/// A tensor is not on the list although a computed one is only numbers: a
/// deferred one holds its inputs (see `TensorExpr`), and the flag that says
/// which is in the payload, not the type. Scanning a computed tensor finds
/// nothing and costs a worklist slot.
inline bool is_atom_object(ObjType t) {
    return t == ObjType::Float || t == ObjType::Str || t == ObjType::Pid ||
           t == ObjType::BigStr || t == ObjType::BigInt;
}

/// Allocation rounds every object up to one of these sizes, and every chunk in
/// the heap is exactly one of them -- which is what makes the free lists
/// uniform and O(1) to fill and drain. Sizes below 128 are every multiple of 8
/// (objects all the way up to a ~8-slot frame are some $24+8k$); above that the
/// classes spread, first by 16, then geometrically, so a large object wastes at
/// most ~7% of its chunk.
constexpr uint32_t kClassSizes[] = {
    16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,  104, 112, 120, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 416, 448, 480, 512,
    576, 640, 704, 768, 832, 896, 960, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
    2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
};
static_assert(std::size(kClassSizes) == kHeapClassCount,
              "the free list array must match the size-class table");
constexpr uint32_t kMaxClassSize = kClassSizes[kHeapClassCount - 1];

/// Every class size is a multiple of 8 and the largest is 4096, so the whole
/// table inverts into 512 bytes: `kClassOf[bytes / 8]` is the class index of
/// anything that size, rounded up. Built once, at load.
///
/// This replaces a binary search. Allocation is the single most frequent thing
/// the VM does -- three hundred million times in one fold -- and six dependent
/// branches to answer a question with 512 possible inputs is the wrong shape.
struct ClassTable {
    uint8_t of[(kMaxClassSize / 8) + 1];
    constexpr ClassTable() : of{} {
        size_t cls = 0;
        for (size_t words = 0; words <= kMaxClassSize / 8; ++words) {
            while (kClassSizes[cls] < words * 8) ++cls;
            of[words] = uint8_t(cls);
        }
    }
};
constexpr ClassTable kClassOf{};
static_assert(kHeapClassCount <= 256, "a class index has to fit in the table's byte");

/// Set from DREAM_GC_TRACE: report every collection on stderr as it happens.
/// Read once, because the collector asks twice per collection and the answer
/// cannot change. `--stats` gives the totals; this gives the shape of them,
/// which is what says whether a pause is one long collection or forty short
/// ones, and whether the threads that joined in found anything to do.
static bool gc_trace() {
    static const bool on = [] {
        const char* v = std::getenv("DREAM_GC_TRACE");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

/// A monotonic clock read, in nanoseconds. Two of these per collection is
/// nothing against the collection itself, and a pause nobody measured is a
/// pause nobody has diagnosed.
inline uint64_t now_nanos() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}

/// The index of the smallest class that holds `bytes`. `bytes` must be a
/// multiple of 8 no larger than the top class -- every caller has already
/// rounded up, and sweep only ever asks about a chunk it found in the heap.
inline size_t class_index(uint32_t bytes) {
    return kClassOf.of[bytes >> 3];
}

/// The class size that holds `bytes`: the class, rounding up.
inline uint32_t class_size(uint32_t bytes) {
    return kClassSizes[class_index(bytes)];
}
}  // namespace

/// How large a nursery may grow, and where the number comes from.
///
/// A nursery is a bet that most objects die young, and its size is how long
/// they are given to do it. Too small and a collection promotes objects that
/// were about to die anyway -- which is the expensive mistake, because
/// promotion is a copy and everything it copies has to be scanned and then
/// swept later. Too large and the collection's working set falls out of cache
/// and every process pays for memory it is not using.
///
/// So the size is not chosen: it is *earned*. Every process starts at 64 KB,
/// and only one that keeps promoting a large share of what it allocates grows
/// -- doubling each time, up to this. A language that expects hundreds of
/// thousands of processes cannot afford a large nursery by default, and a
/// compiler churning through a syntax tree cannot afford a small one; letting
/// the survival rate decide gives each of them what it needs.
///
/// The default cap is 32 MiB, which is where the self-compile stops improving.
/// A program can ask for another in its root package's manifest (`[vm]
/// nursery_max`, carried in the image's `VMOP` section), and the person
/// running it can overrule that: `dream --nursery-max SIZE` first, then
/// `DREAM_NURSERY_MAX`, then the image, then the default. `set_nursery_max` is
/// how the command line and the image get here; it is called before the
/// first process, so every heap is made with the same cap.
static std::atomic<size_t> g_nursery_max{0};

void set_nursery_max(size_t bytes) {
    // Below the nursery a process starts with, a cap would only ever shrink it.
    g_nursery_max.store(std::max(bytes, size_t(64) << 10));
}

static size_t nursery_max_bytes() {
    if (const size_t set = g_nursery_max.load()) return set;
    if (const char* env = std::getenv("DREAM_NURSERY_MAX")) {
        size_t n = 0;
        if (parse_size(env, &n)) return std::max(n, size_t(64) << 10);
    }
    return size_t(32) << 20;
}

Heap::Heap(size_t initial_bytes)
    : nursery_hi_(initial_bytes),
      nursery_max_(nursery_max_bytes()),
      gc_threshold_(initial_bytes),
      initial_bytes_(initial_bytes) {
    // The nursery starts with one block; old space grows its own as objects
    // survive into it. The constructor's block is the nursery's, so a process
    // that dies young -- most programs -- never grows an old space at all.
    nursery_.push_back(new_block(initial_bytes));
}

Heap::Block* Heap::new_block(size_t bytes, bool exact) {
    size_t size = bytes < initial_bytes_ && !exact ? initial_bytes_ : bytes;
    auto* b = static_cast<Block*>(std::malloc(sizeof(Block)));
    if (!b) throw std::bad_alloc();
    b->data = static_cast<uint8_t*>(std::malloc(size));
    if (!b->data) {
        std::free(b);
        throw std::bad_alloc();
    }
    b->next = nullptr;
    b->size = size;
    b->used = 0;
    b->big = false;
    b->dead = false;
    b->evacuate = false;
    b->pinned = false;
    b->held_end = b->held_from = b->held_count = 0;
    block_bytes_ += size;
    if (block_bytes_ > peak_block_bytes_) {
        peak_block_bytes_ = block_bytes_;
        peak_allocated_ = allocated_;
    }
    return b;
}

void Heap::free_block(Block* b) {
    block_bytes_ -= b->size;
    std::free(b->data);
    std::free(b);
}

void Heap::retire_big(Block* b) {
    big_pool_[b->size].push_back(PooledBlock{b->data, b->size, sweeps_});
    big_pool_bytes_ += b->size;
    std::free(b);
}

void Heap::trim_big_pool() {
    const size_t cap = std::max(kBigPoolFloor, live_after_gc_);
    for (auto it = big_pool_.begin(); it != big_pool_.end();) {
        std::vector<PooledBlock>& same = it->second;
        size_t keep = 0;
        for (size_t i = 0; i < same.size(); ++i) {
            PooledBlock& pb = same[i];
            // Oldest first within a size, so what goes over the cap is what
            // has waited longest there.
            const bool stale = pb.sweep < sweeps_;
            if (stale || big_pool_bytes_ > cap) {
                big_pool_bytes_ -= pb.size;
                block_bytes_ -= pb.size;
                std::free(pb.data);
            } else {
                same[keep++] = pb;
            }
        }
        same.resize(keep);
        it = same.empty() ? big_pool_.erase(it) : std::next(it);
    }
}

void Heap::free_blocks(Block* b) {
    while (b) {
        Block* next = b->next;
        free_block(b);
        b = next;
    }
}

void Heap::free_nursery() {
    // But a block holding a pinned object, which stays: see `hold_pinned_blocks`.
    size_t kept = 0;
    for (Block* b : nursery_) {
        if (b->pinned) nursery_[kept++] = b; else free_block(b);
    }
    nursery_.resize(kept);
    nursery_at_ = 0;
    nursery_bytes_ = 0;
    nursery_held_ = 0;
}

Obj* Heap::carve(size_t sz) {
    // A same-sized chunk is already free: pop it. The free list holds only
    // chunks this size class knows, so the header is a valid run member; the
    // `GC_FREE` bit tells sweep that a dead chunk is already on a list and must
    // not be pushed a second time. Only the collector's promotes carve here,
    // so a popped chunk is always about to become a tenured object.
    size_t i = class_index(static_cast<uint32_t>(sz));
    if (Obj* o = free_lists_[i]) {
        free_lists_[i] = free_next(o);
        allocated_ += sz;
        return o;
    }

    // None free. Carve a fresh chunk off the tail of a block that has room --
    // the current carve block first, then any other shared old block (so the
    // strands of space left when a run ends and a new one begins are still
    // used), then a brand-new block. Dedicated big-object blocks are never a
    // target: their headroom is reserved, and a chunk cut into one would be
    // orphaned the day the big object dies and its whole block is handed back.
    // The carve is why collection can never run here: the block's tail belongs
    // to no object yet, but the caller may hold raw pointers that a collection
    // would have to treat as roots or lose. A block waiting to be evacuated is
    // not a target either: whatever went into it would only be copied out
    // again, and its tail is about to be handed back with the rest of it.
    Block* b = carve_block_;
    if (b && (b->big || b->evacuate || b->used + sz > b->size)) b = nullptr;
    for (Block* c = blocks_; b == nullptr && c; c = c->next) {
        if (!c->big && !c->evacuate && c->used + sz <= c->size) b = c;
    }
    if (!b) {
        b = new_block(sz);
        b->next = blocks_;
        blocks_ = b;
    }
    carve_block_ = b;
    auto* o = reinterpret_cast<Obj*>(b->data + b->used);
    b->used += sz;
    allocated_ += sz;
    return o;
}

namespace {
/// The size a big block is made in: `sz` rounded up to one of eight steps
/// per doubling, at most 12.5% over what was asked for. Classes rather than
/// exact sizes so that a pooled block fits the next object of nearly the same
/// size, and only eight per doubling so that it is never much too big.
size_t big_size_class(size_t sz) {
    size_t top = size_t(1) << (63 - std::countl_zero(uint64_t(sz)));
    size_t step = top >= 8 ? top / 8 : 1;
    return (sz + step - 1) / step * step;
}
}  // namespace

Obj* Heap::carve_big(size_t sz) {
    // Anything past the top class gets a block to itself: a run of even a few
    // percent slack is one large object's worth of waste, and big objects are
    // rare enough that dedicating a block never fragments anything smaller.
    // A pooled block of the same class is taken first; see `big_pool_`.
    const size_t cls = big_size_class(sz);
    Block* b = nullptr;
    // Best fit among the pooled blocks no more than half as large again as
    // the class: a loop's sizes vary, and an exact match missed whenever they
    // did -- the pool then held memory the program could not use while it
    // asked the system for more. Of that size, the block retired last, whose
    // pages are the warmest.
    auto fit = big_pool_.lower_bound(cls);
    if (fit != big_pool_.end() && fit->first <= cls + cls / 2) {
        b = static_cast<Block*>(std::malloc(sizeof(Block)));
        if (!b) throw std::bad_alloc();
        const PooledBlock pb = fit->second.back();
        fit->second.pop_back();
        if (fit->second.empty()) big_pool_.erase(fit);
        b->data = pb.data;
        b->size = pb.size;
        b->dead = false;
        b->evacuate = false;
        b->pinned = false;
        b->held_end = b->held_from = b->held_count = 0;
        big_pool_bytes_ -= b->size;
    }
    // Exactly the class, not the heap's minimum block. That minimum is for
    // shared blocks, which many objects carve; a dedicated block holds one
    // object and nothing is ever cut from its tail, so rounding a 5 KiB
    // object up to a 64 KiB block was 59 KiB of nothing -- held until the
    // object died. A loop of bignums past the top class made one per step,
    // and held a gigabyte from the OS for 85 MB of numbers.
    if (!b) b = new_block(cls, true);
    b->big = true;
    b->used = sz;
    b->next = blocks_;
    blocks_ = b;
    allocated_ += sz;
    return reinterpret_cast<Obj*>(b->data);
}

/// Bump-allocate from the nursery. Collection can never run mid-reduction, so
/// no allocation-time safepoint is needed: the high-water mark only decides
/// what the *next* safepoint check does. When the current block is full the
/// next one is taken, and when there is no next one a fresh block is added --
/// never a collection, and never a failure.
Obj* Heap::alloc_nursery(uint32_t sz) {
    if (nursery_at_ < nursery_.size()) {
        Block* b = nursery_[nursery_at_];
        if (b->used + sz <= b->size) {
            auto* o = reinterpret_cast<Obj*>(b->data + b->used);
            b->used += sz;
            nursery_bytes_ += sz;
            allocated_ += sz;
            return o;
        }
    }
    return grow_nursery(sz);
}

Obj* Heap::grow_nursery(uint32_t sz) {
    // Step past the blocks with no room. A minor collection empties every
    // block and rewinds to the first, so the ones behind us have already been
    // refilled; a major frees them all, and then there are none at all.
    while (nursery_at_ < nursery_.size()) {
        Block* b = nursery_[nursery_at_];
        if (b->used + sz <= b->size) break;
        ++nursery_at_;
    }
    if (nursery_at_ == nursery_.size()) {
        Block* b = new_block(sz);
        b->next = nullptr;
        nursery_.push_back(b);
    }
    Block* b = nursery_[nursery_at_];
    auto* o = reinterpret_cast<Obj*>(b->data + b->used);
    b->used += sz;
    nursery_bytes_ += sz;
    allocated_ += sz;
    return o;
}

Obj* Heap::alloc_bare(ObjType type, size_t extra) {
    size_t bytes = align_up(sizeof(Obj) + extra);
    uint32_t sz;
    Obj* o;
    uint8_t gen;
    if (bytes > kMaxClassSize) {
        // Large objects tenure immediately: they are rare, often long-lived,
        // and copying one once to save the next copy is a bad swap.
        sz = static_cast<uint32_t>(bytes);
        o = carve_big(sz);
        gen = GC_OLD;
    } else {
        sz = class_size(static_cast<uint32_t>(bytes));
        o = alloc_nursery(sz);
        gen = GC_YOUNG;
    }
    // The header is written whole rather than cleared and then filled: these
    // four fields are the whole of it. `gc` is published last, because it is
    // what a concurrent mark's helper looks at first -- the other bytes must
    // all be there by the time the byte that says "reachable" is.
    //
    // A big object allocated during a concurrent mark is born *marked*: born
    // grey, it could be scanned by a helper before its caller has filled the
    // payload in. So it carries GC_MARK, and is logged for the finalize to
    // scan when the fills are done. The young objects it is filled with can
    // wait for that scan; a helper would have skipped them anyway, because
    // young objects are precisely what a concurrent mark leaves to the final.
    //
    // With no mark running the same object has the same problem for a simpler
    // reason: it is born *old*, and what its caller fills it with is young.
    // That is an old-to-young edge made by no store the barrier ever sees --
    // `make_array` hands back an array and the caller writes the elements
    // straight into it -- so the next minor collection would not scan the
    // array, and the elements it cannot see are elements it frees. A 3000-cell
    // array of unforced thunks loses every one of them, and what the program
    // gets back is whatever the nursery has since put there.
    //
    // So an object born old is remembered at birth. One entry covers the whole
    // object however many slots it has, it is paid only by allocations over
    // `kMaxClassSize` -- which tenure precisely because they are rare -- and it
    // is what makes filling one safe without a barrier at every fill site.
    // Nothing can collect between the allocation and the fill: a collection
    // runs only at a safepoint, and there is none inside a reduction.
    by_type_[size_t(type) & 31] += sz;
    o->type = type;
    o->aux = 0;
    o->bytes = sz;
    if (marking_ && bytes > kMaxClassSize) {
        std::atomic_ref<uint8_t>(o->gc).store(GC_OLD | GC_MARK, std::memory_order_release);
        mark_born_.push_back(o);
    } else {
        std::atomic_ref<uint8_t>(o->gc).store(gen, std::memory_order_relaxed);
        if (gen == GC_OLD) remembered_.push_back(o);
    }
    total_allocated_ += sz;
    return o;
}

Obj* Heap::alloc(ObjType type, size_t extra) {
    Obj* o = alloc_bare(type, extra);
    std::memset(o + 1, 0, size_t(o->bytes) - sizeof(Obj));
    return o;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Value Heap::make_float(double v) {
    auto* o = static_cast<FloatObj*>(alloc_bare(ObjType::Float, sizeof(double)));
    o->value = v;
    return from_obj(o);
}

BigIntObj* Heap::alloc_bigint(uint32_t limbs) {
    // Inline while the object fits a size class: a young chunk like any other.
    if (align_up(sizeof(BigIntObj) + size_t(limbs) * 8) <= kMaxClassSize) {
        auto* o = static_cast<BigIntObj*>(alloc_bare(ObjType::BigInt, 8 + size_t(limbs) * 8));
        o->external = 0;
        o->pad = 0;
        return o;
    }
    // Past it, a young header and limbs of their own; see `BigIntObj`. The
    // header goes on `external_`, so the collector says when they die.
    auto* data = static_cast<uint64_t*>(std::malloc(size_t(limbs) * 8));
    if (!data) throw std::bad_alloc();
    auto* o = static_cast<BigIntObj*>(alloc_bare(ObjType::BigInt, 8 + 16));
    o->external = 1;
    o->pad = 0;
    o->external_limbs() = data;
    o->external_capacity() = limbs;
    external_.push_back(o);
    external_bytes_ += size_t(limbs) * 8;
    return o;
}

Value Heap::make_bigint(const uint64_t* limbs, uint32_t len, bool neg) {
    BigIntObj* o = alloc_bigint(len);
    o->len = len;
    o->neg = neg ? 1 : 0;
    std::memcpy(o->limbs(), limbs, size_t(len) * 8);
    return from_obj(o);
}

Value Heap::make_string(const char* data, uint32_t len) {
    auto* o = static_cast<StrObj*>(alloc_bare(ObjType::Str, 8 + size_t(len) + 1));
    o->len = len;
    o->hash = 0;
    // A null `data` reserves room to be filled by the caller, which is how
    // concatenation builds its result in one allocation.
    if (len && data) std::memcpy(o->data(), data, len);
    o->data()[len] = '\0';
    return from_obj(o);
}

Value Heap::make_bigstr(const char* data, uint64_t len) {
    // Sized from the struct rather than counted by hand: unlike a StrObj there
    // is no trailing payload here, so the whole object is its fields.
    auto* o = static_cast<BigStrObj*>(alloc_bare(ObjType::BigStr, sizeof(BigStrObj) - sizeof(Obj)));
    o->len = len;
    o->hash = 0;
    o->data = data;
    return from_obj(o);
}

Value Heap::make_cons(Value head, Value tail) {
    auto* o = static_cast<ConsObj*>(alloc_bare(ObjType::Cons, 2 * sizeof(Value)));
    o->head = head;
    o->tail = tail;
    return from_obj(o);
}

Value Heap::make_array(uint32_t len) {
    auto* o = static_cast<ArrayObj*>(alloc(ObjType::Array, 8 + size_t(len) * sizeof(Value)));
    o->len = len;
    return from_obj(o);
}

Value Heap::make_record(uint32_t tag, uint32_t len) {
    auto* o = static_cast<RecordObj*>(
        alloc(ObjType::Record, 8 + size_t(len) * sizeof(Value)));
    o->len = len;
    o->tag = tag;
    return from_obj(o);
}

Value Heap::make_map(uint32_t) {
    return make_map_branch(0);
}

Value Heap::make_map_branch(uint32_t nslots) {
    auto* o = static_cast<MapObj*>(alloc(ObjType::Map, 8 + size_t(nslots) * sizeof(Value)));
    o->count = 0;
    o->bitmap = 0;
    return from_obj(o);
}

Value Heap::make_map_leaf(uint64_t hash, Value key, Value value, Value next) {
    auto* o = static_cast<MapLeafObj*>(alloc_bare(ObjType::MapLeaf, 8 + 3 * sizeof(Value)));
    o->hash = hash;
    o->key = key;
    o->value = value;
    o->next = next;
    return from_obj(o);
}

Value Heap::make_closure(uint32_t func, uint32_t ncaps) {
    auto* o = static_cast<ClosureObj*>(alloc(ObjType::Closure, 8 + size_t(ncaps) * sizeof(Value)));
    o->func = func;
    o->ncaps = ncaps;
    return from_obj(o);
}

// `DREAM_PROBE_FRAMES`: how many frames a run makes, and how many of them a
// suspension is ever made against. A frame no suspension holds is done with
// when its call returns, so it could be freed then rather than collected: the
// count is the most that freeing frames on return could save, measured before
// anyone builds the analysis that would prove which frames those are. A
// frame's unused `pad` marks it held, so each is counted once.
extern const bool g_probe_frames = std::getenv("DREAM_PROBE_FRAMES") != nullptr;
std::atomic<uint64_t> g_frames_made{0}, g_frame_bytes{0}, g_frames_held{0}, g_held_bytes{0};

Value Heap::make_thunk(uint32_t node, Value frame) {
    auto* o = static_cast<ThunkObj*>(alloc_bare(ObjType::Thunk, 8 + sizeof(Value)));
    o->node = node;
    o->pad = 0;
    o->frame = frame;
    if (g_probe_frames && is_obj(frame, ObjType::Frame)) {
        auto* fo = static_cast<FrameObj*>(as_obj(frame));
        if (fo->pad == 0) {
            fo->pad = 1;
            g_frames_held.fetch_add(1, std::memory_order_relaxed);
            g_held_bytes.fetch_add(fo->bytes, std::memory_order_relaxed);
        }
    }
    return from_obj(o);
}

Value Heap::make_frame(Value closure, uint32_t nslots) {
    return make_frame_filling(closure, nslots, 0);
}

Value Heap::make_frame_filling(Value closure, uint32_t nslots, uint32_t filled) {
    auto* o = static_cast<FrameObj*>(
        alloc_bare(ObjType::Frame, sizeof(Value) + 8 + size_t(nslots) * sizeof(Value)));
    o->closure = closure;
    o->nslots = nslots;
    o->pad = 0;
    if (g_probe_frames) {
        g_frames_made.fetch_add(1, std::memory_order_relaxed);
        g_frame_bytes.fetch_add(o->bytes, std::memory_order_relaxed);
    }
    // The slots the caller is about to write need no clearing; the rest must
    // read as `NIL_SLOT`, and so must the padding to the size class, which the
    // collector walks as part of the object.
    Value* slots = o->slots();
    auto* from = reinterpret_cast<uint8_t*>(slots + filled);
    auto* end = reinterpret_cast<uint8_t*>(o) + o->bytes;
    if (end > from) std::memset(from, 0, size_t(end - from));
    return from_obj(o);
}

Value Heap::make_pap(Value fn, uint32_t nargs) {
    auto* o = static_cast<PapObj*>(
        alloc(ObjType::Pap, sizeof(Value) + 8 + size_t(nargs) * sizeof(Value)));
    o->fn = fn;
    o->nargs = nargs;
    return from_obj(o);
}

Value Heap::make_error(Value kind, Value payload, Value where) {
    auto* o = static_cast<ErrorObj*>(alloc_bare(ObjType::ErrorBox, 3 * sizeof(Value)));
    o->kind = kind;
    o->payload = payload;
    o->where = where;
    return from_obj(o);
}

namespace {
/// The header of a tensor, written the same way whichever area holds it.
void init_tensor(TensorObj* t, uint32_t rank, const uint32_t* dims, uint64_t count) {
    t->dtype = TENSOR_F64;
    t->rank = uint8_t(rank);
    t->device = TENSOR_HOST;
    t->flags = 0;
    t->pad = 0;
    t->count = count;
    for (uint32_t i = 0; i < TENSOR_MAX_RANK; ++i) t->dims[i] = i < rank ? dims[i] : 0;
}
}  // namespace

Value Heap::make_tensor(uint32_t rank, const uint32_t* dims, uint64_t count) {
    // Bare: the data is about to be written whole, and clearing megabytes the
    // caller overwrites is the waste `alloc_bare` exists to avoid. Nothing here
    // is a reference, so an unwritten element is never mistaken for one.
    auto* t = static_cast<TensorObj*>(alloc_bare(
        ObjType::Tensor, sizeof(TensorObj) - sizeof(Obj) + size_t(count) * sizeof(double)));
    init_tensor(t, rank, dims, count);
    return from_obj(t);
}

Value Heap::make_external_tensor(uint32_t rank, const uint32_t* dims, uint64_t count,
                                 uint8_t dtype, uint8_t device, void* handle) {
    auto* t = static_cast<TensorObj*>(
        alloc_bare(ObjType::Tensor, sizeof(TensorObj) - sizeof(Obj) + sizeof(void*)));
    init_tensor(t, rank, dims, count);
    t->dtype = dtype;
    t->device = device;
    std::memcpy(t->data(), &handle, sizeof handle);
    external_.push_back(t);
    external_bytes_ += tensor_data_bytes(t);
    return from_obj(t);
}

Value Heap::make_deferred_tensor(uint32_t rank, const uint32_t* dims, uint64_t count,
                                 uint8_t dtype, uint8_t device, uint32_t ninputs,
                                 uint32_t nconsts, uint32_t ncode) {
    // Cleared, not bare: the inputs are references, and until the caller has
    // written them the collector must read them as empty slots.
    auto* t = static_cast<TensorObj*>(
        alloc(ObjType::Tensor, sizeof(TensorObj) - sizeof(Obj) +
                                   TensorExpr::bytes_for(ninputs, nconsts, ncode)));
    init_tensor(t, rank, dims, count);
    t->dtype = dtype;
    t->device = device;
    t->flags = TENSOR_DEFERRED;
    TensorExpr* e = tensor_expr(t);
    e->result = NIL_SLOT;
    e->ninputs = uint16_t(ninputs);
    e->nconsts = uint16_t(nconsts);
    e->ncode = uint16_t(ncode);
    return from_obj(t);
}

size_t Heap::external_size(Obj* o) {
    if (o->type == ObjType::BigInt) return size_t(static_cast<BigIntObj*>(o)->external_capacity()) * 8;
    return tensor_data_bytes(static_cast<TensorObj*>(o));
}

void Heap::drop_external(Obj* o) {
    if (o->type == ObjType::BigInt) {
        std::free(static_cast<BigIntObj*>(o)->external_limbs());
        return;
    }
    release_external(o);
}

void Heap::reap_external(bool full) {
    size_t keep = 0;
    const size_t before = external_bytes_;
    auto drop = [&](Obj* o) {
        external_bytes_ -= external_size(o);
        drop_external(o);
    };
    for (Obj* o : external_) {
        const uint8_t gc = o->gc;
        if (gc == GC_FORWARDED) {
            // Promoted: the copy is the object now, and the stub is about to
            // be nursery space again.
            external_[keep++] = as_obj(forward_target(o));
        } else if (gc & GC_PINNED) {
            // Young and not copied out, because a compiled frame holds it.
            external_[keep++] = o;
        } else if (gc & GC_YOUNG) {
            // Young and not copied out: nothing reached it.
            drop(o);
        } else if (!full || (gc & GC_MARK)) {
            external_[keep++] = o;
        } else {
            drop(o);
        }
    }
    external_.resize(keep);
    // A minor that was called for external memory and freed less than half of
    // it found the memory held by old objects, which only a major can judge.
    // A major settles the question either way.
    external_major_ = !full && before >= external_trigger_ && external_bytes_ * 2 > before;
    //
    // The doubling is capped by the device. GPU memory is not the heap's to
    // grow: a training loop makes a gigabyte of temporaries a second on the
    // device, and with the trigger at twice what survived, the peaks between
    // collections grew until an 8 GB card ran out (`external_budget`).
    const size_t doubled = std::min(external_bytes_ * 2, external_budget());
    external_trigger_ = std::max(doubled, external_bytes_ + kExternalSlack);
}

Value Heap::make_pid(uint64_t id) {
    auto* o = static_cast<PidObj*>(alloc_bare(ObjType::Pid, sizeof(uint64_t)));
    o->id = id;
    return from_obj(o);
}

Value Heap::make_module(uint32_t import_index, Value name) {
    auto* o = static_cast<ModuleObj*>(alloc(ObjType::Module, 8 + sizeof(Value)));
    o->import_index = import_index;
    o->name = name;
    return from_obj(o);
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// One collector thread's private state, for the length of one collection.
///
/// Everything a tracing thread touches often lives here rather than in the
/// heap: the grey set it is working through, the old-space chunks it promotes
/// into, and the counters it will add to the heap's when it is finished. That
/// is what lets a parallel round take a lock only when a thread runs out of
/// something -- a batch of free chunks, a block to carve, work to do -- rather
/// than once per object.
struct Heap::GcCtx {
    /// The grey set: marked or promoted, not yet scanned. A serial collection
    /// points this at the heap's own queue, which keeps its capacity from one
    /// collection to the next; a parallel one gives every thread its own and
    /// balances them through the round's shared stack.
    std::vector<Obj*>* q = nullptr;
    std::vector<Obj*> own;
    /// Whether the object being scanned has a field naming a pinned young
    /// object, and the old ones found to. See `GC_PINNED`.
    bool met_pinned = false;
    std::vector<Obj*> pin_refs;
    /// Objects still to scan before this thread next asks whether anyone else
    /// has run out of work. Checking on every object would put a shared
    /// counter on the hottest loop in the collector to answer a question whose
    /// answer changes a few thousand times a second.
    int check_in = 0;

    /// Old space to promote into, private to this thread: chunks split off the
    /// heap's free lists a batch at a time, and a block of its own to carve
    /// from when they run out. Both are handed back when the round ends.
    Block* carve_block = nullptr;
    Obj* free_head[kHeapClassCount] = {};
    Obj* free_tail[kHeapClassCount] = {};
    /// Size classes whose shared list was found empty. Nothing puts a chunk
    /// back on a list until the round is over, so "empty" stays true once it
    /// is true -- and without remembering it, every promotion of a shape the
    /// heap has none of spare would take the allocation lock to be told so
    /// again, which is a lock per object and the whole cost of the round.
    bool free_dry[kHeapClassCount] = {};

    /// Added to the heap's counters when the round ends. Kept apart until then
    /// so that no collector thread ever writes a heap-wide number.
    size_t allocated = 0;
    uint64_t promoted = 0;
    uint64_t total = 0;
    size_t live = 0;
    /// What this thread's share of the sweep found alive, by object kind.
    /// Summed into the heap's own when the round ends, like every other
    /// counter here, so that no collector thread writes a heap-wide number.
    uint64_t by_type[64] = {};
    /// Objects this thread took off a queue and scanned. Counted only in a
    /// parallel round, where the question it answers -- did the work divide?
    /// -- is the only one worth asking about a collector thread.
    uint64_t scanned = 0;
};

/// What a parallel round shares. It lives on the collecting thread's stack for
/// the length of the round and nowhere else.
struct Heap::GcRound {
    /// Guards the shared stack below, and the counters that decide when the
    /// trace is over.
    std::mutex mutex;
    std::condition_variable cv;
    /// Guards the heap's free lists and its block list. A separate lock from
    /// the one above, and not for tidiness: a thread refills its chunk lists
    /// while promoting, which is exactly when the other threads are handing
    /// work around, and one lock for both would have promotion waiting on the
    /// load balancer thousands of times a collection.
    std::mutex alloc_mutex;

    /// Work no thread has claimed. A thread with a long queue gives half of it
    /// away and a thread with none takes a batch back, which is the whole of
    /// the load balancing: the shape of a heap is not known before it is
    /// traced, so the only honest way to divide the trace is to divide it as
    /// it is found.
    std::vector<Obj*> stack;
    /// `stack.size()`, readable without the lock.
    std::atomic<size_t> shared{0};
    /// Threads that have run out of work and not yet found any. A tracing
    /// thread reads this far more often than it changes, which is why it is an
    /// atomic beside the count under the lock rather than the count itself.
    ///
    /// This, and not the size of the local queue, is the question worth
    /// asking before giving work away: handing over a hundred objects costs a
    /// lock, a copy and possibly a wake-up, and is worth it exactly when
    /// somebody would otherwise be doing nothing.
    std::atomic<unsigned> idle{0};

    /// Threads that have reached the body, and threads that still have work.
    /// Both are guarded by `mutex`, and it matters that they are: a thread
    /// claiming work from the stack and a thread declaring itself out of work
    /// have to be ordered against each other, or the second could conclude
    /// that everyone is idle in the moment before the first says otherwise.
    ///
    /// The trace is over when every thread has arrived and every one of them
    /// has run out at once -- a grey object could only be on a thread's own
    /// queue or on the shared stack, and then neither holds one. Counting
    /// arrivals is what makes a thread that was slow to wake safe: until it
    /// has said it is here, the others may not conclude anything.
    unsigned size = 0;
    unsigned arrived = 0;
    unsigned active = 0;
    /// Threads asleep waiting for work. A thread giving work away wakes the
    /// others only when there is someone to wake: a `notify_all` nobody is
    /// listening for is a system call for nothing, and sharing happens often
    /// enough that the nothings add up to more than the collection.
    unsigned waiting = 0;
    /// Read while spinning, so it cannot be guarded by the lock alone.
    std::atomic<bool> done{false};

    std::vector<GcCtx> ctxs;

    /// The old blocks, snapshotted so that a parallel sweep can hand them out
    /// by index, and the list can be rebuilt afterwards by the one thread that
    /// owns it.
    std::vector<Block*> blocks;
    std::atomic<size_t> next_block{0};
};

namespace {

/// Read an object's collector byte. In a parallel round another thread may be
/// claiming the same object at this moment, so the read has to be spelled as
/// an atomic one -- on the machines this runs on it is the same instruction,
/// and saying so is what makes the claim protocol below mean anything.
template <bool Par>
inline uint8_t read_gc(Obj* o) {
    if constexpr (Par)
        return std::atomic_ref<uint8_t>(o->gc).load(std::memory_order_acquire);
    else
        return o->gc;
}

/// Read an object's type byte. In a parallel round another thread may be
/// claiming the object's `gc` at this moment -- but that is not the byte
/// being read here, and the mutator, which is stopped in a parallel round,
/// writes no headers then. What needs saying is the *concurrent* mark, where
/// the mutator is very much running and can flip a live thunk's type (to
/// Blackhole, to Indirect, back) while a helper scans: a plain read against
/// that atomic store is a data race, and reading a type mid-store is reading
/// a byte that is whole one way or the other anyway.
template <bool Par>
inline ObjType read_type(Obj* o) {
    if constexpr (Par)
        return static_cast<ObjType>(
            std::atomic_ref<uint8_t>(reinterpret_cast<uint8_t&>(o->type))
                .load(std::memory_order_acquire));
    else
        return o->type;
}

/// The same for a reference the collector is about to rewrite.
///
/// Two threads can reach one slot: an object logged twice in the remembered
/// set is scanned twice, and both scans write the same answer into it. That
/// much would be benign on every machine here. The ordering is not: the value
/// written is usually a copy the writing thread has *just* made, and a slot is
/// how the rest of the collection finds it. Without the release the reader can
/// see the pointer before it sees the bytes -- and then read a header that is
/// still whatever the chunk held before it was reused. The pair below is the
/// second way a promoted object is published; the forwarding stub in `promote`
/// is the first, and both need the same edge for the same reason.
///
/// On x86-64 these are the same two instructions a plain load and store
/// compile to. Saying which is not a cost; it is the difference between a
/// collector that works and one that works on this machine.
template <bool Par>
inline Value read_slot(Value* slot) {
    if constexpr (Par)
        return std::atomic_ref<Value>(*slot).load(std::memory_order_acquire);
    else
        return *slot;
}

template <bool Par>
inline void write_slot(Value* slot, Value v) {
    if constexpr (Par)
        std::atomic_ref<Value>(*slot).store(v, std::memory_order_release);
    else
        *slot = v;
}

/// `resolve`, for a collector thread.
///
/// The chain it walks is made of ordinary object fields, and in a parallel
/// round another thread may have written the last link a moment ago -- so
/// every hop is an acquiring read, which is what makes the object behind it
/// safe to look at at all. The mutator's `resolve` in value.hpp is the same
/// walk without that, which is right for the mutator: it is stopped.
template <bool Par>
inline Value gc_resolve(Value v) {
    if constexpr (!Par) {
        return resolve(v);
    } else {
        while (is_ptr(v) && as_obj(v)->type == ObjType::Indirect) {
            v = std::atomic_ref<Value>(static_cast<IndirectObj*>(as_obj(v))->target)
                    .load(std::memory_order_acquire);
        }
        return v;
    }
}

/// One spin of a wait that is expected to be over almost immediately.
inline void spin_once() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

/// How long a local queue may get before half of it is offered to the others
/// whatever they are doing, how short it may be and still be worth splitting
/// for a thread that has nothing, and how much an idle thread takes back at a
/// time. All three are in objects.
///
/// The floor is low because the cost it guards against is not the copy but the
/// wake-up, and `idle` above already answers whether anyone needs waking. What
/// no floor can help with is a trace with no parallelism in it: a list is the
/// honest example -- scanning one cell yields exactly one more, so the queue
/// stays at length one however many threads are watching it, and the right
/// thing for the others to do is go back to sleep.
constexpr size_t kShareAbove = 4096;
constexpr size_t kShareMin = 32;
constexpr size_t kTakeBatch = 512;
/// Objects scanned between two looks at whether anyone else is idle.
constexpr int kCheckEvery = 64;
/// How long a thread with nothing to do waits before it sleeps, in spins.
///
/// The thread that is about to hand work over is running *now*, so the wait is
/// usually a fraction of a microsecond -- and a futex round trip is tens of
/// them. Sleeping immediately is what made the smaller collections run on one
/// thread while seven others were still being woken: by the time they were up
/// the trace was over. Sleeping eventually is what keeps a long serial tail
/// -- a list, whose trace has no parallelism in it at all -- from burning
/// seven cores to watch one.
constexpr unsigned kIdleSpins = 2000;
/// Chunks split off a shared free list in one go.
constexpr unsigned kFreeBatch = 256;

}  // namespace

/// Keep the actual destructor here: it joins a concurrent mark in flight --
/// the helpers may be picking their way through an object in this heap at
/// this very moment, so they must stand down before a single block below is
/// freed -- and `mark_round_`'s `unique_ptr` needs the round's shape complete,
/// which it only is once `GcRound` has been defined above.
Heap::~Heap() {
    if (marking_) {
        marking_ = false;
        GcPool::instance().join();
    }
    // The process is gone, so everything it held outside the heap goes too.
    for (Obj* o : external_) drop_external(o);
    free_blocks(blocks_);
    for (Block* b : nursery_) free_block(b);
    for (auto& [size, same] : big_pool_) {
        for (PooledBlock& pb : same) std::free(pb.data);
    }
}

/// How much work makes the handshake worth it. Waking a pool of threads and
/// joining them again costs tens of microseconds, so a collection that would
/// have finished inside that is better done alone -- and most collections in
/// most processes are exactly that. A minor is measured by the nursery it is
/// about to empty, a major by the whole heap it is about to walk, and the
/// major's floor is four times the minor's because a major also sweeps.
///
/// `DREAM_GC_PAR_MIN` moves the floor. Zero puts every collection through the
/// parallel path however small it is, which is how the race detector gets to
/// see the parallel collector on a program small enough to run under it.
static size_t parallel_floor(bool minor) {
    static const size_t base = [] {
        if (const char* env = std::getenv("DREAM_GC_PAR_MIN")) {
            long long n = std::atoll(env);
            if (n >= 0) return size_t(n);
        }
        return size_t(1) << 20;
    }();
    return minor ? base : base * 4;
}

Value Heap::await_forward(Obj* o) {
    std::atomic_ref<uint8_t> gc(o->gc);
    unsigned spins = 0;
    while ((gc.load(std::memory_order_acquire) & GC_FORWARDED) == 0) {
        if (++spins < 64) {
            spin_once();
        } else {
            // The claim covers one memcpy of at most four kilobytes, so
            // getting this far means the claiming thread lost its core rather
            // than that the copy is slow. Stand aside for it.
            std::this_thread::yield();
            spins = 0;
        }
    }
    return forward_target(o);
}

template <bool Par>
void Heap::mark_object(GcCtx& c, Value v) {
    if (!is_ptr(v)) return;
    Obj* o = as_obj(v);
    if (is_shared_obj(o)) {
        if (observe_round_) shared_observe(o, observe_round_);
        return;
    }
    if constexpr (Par) {
        // The mark is the claim: whoever sets the bit owns the scan, so an
        // object two threads reach at once is still scanned exactly once.
        uint8_t prev = std::atomic_ref<uint8_t>(o->gc).fetch_or(
            GC_MARK, std::memory_order_acq_rel);
        if (prev & GC_MARK) return;
    } else {
        if (o->gc & GC_MARK) return;
        o->gc |= GC_MARK;
    }

    // Collapse indirection chains while marking. A long-running loop that
    // repeatedly updates thunks would otherwise accumulate hops that cost
    // time on every read; a collection is the natural place to shorten them.
    // Only the thread that claimed the mark writes here, and the mutator is
    // stopped, so this is still the single writer it was.
    if (o->type == ObjType::Indirect) {
        auto* ind = static_cast<IndirectObj*>(o);
        write_slot<Par>(&ind->target, gc_resolve<Par>(read_slot<Par>(&ind->target)));
    }

    // An atom object (Float, Str, Pid) has no references to forward, so it is
    // grey only in name; it never needs the worklist.
    if (!is_atom_object(o->type)) c.q->push_back(o);
}

bool Heap::claim_mark(Obj* o) {
    // The concurrent mark's claim, and its rule in one line: old objects only.
    // Young ones move at the finalize, so a mark on one would name a corpse;
    // they are reached again there, from the live roots. The mark is the claim
    // exactly as in `mark_object`, except there is no `Par` shape to speak for:
    // the seed and the helpers may as well all use the atomic, since it is the
    // same instruction and only turns up during a mark.
    auto gc = std::atomic_ref<uint8_t>(o->gc);
    uint8_t g = gc.load(std::memory_order_acquire);
    if (g & (GC_YOUNG | GC_SHARED)) return false;
    if (g & GC_MARK) return false;
    uint8_t prev = gc.fetch_or(GC_MARK, std::memory_order_acq_rel);
    return (prev & GC_MARK) == 0 && !(prev & GC_YOUNG);
}

bool Heap::mark_ref(GcCtx& c, Value v) {
    if (!is_ptr(v)) return false;
    Obj* o = as_obj(v);
    if (!claim_mark(o)) return false;
    // The type is read atomically because the mutator is running: it may be
    // flipping a live thunk's type (to Blackhole, to Indirect, back) at this
    // very moment, and a helper reading the byte plainly against that store is
    // a data race. The byte is whole one way or the other either way, but the
    // says-so is what ThreadSanitizer needs.
    //
    // Atom objects have nothing to scan; the mark bit alone keeps them.
    if (!is_atom_object(read_type<true>(o))) c.q->push_back(o);
    return true;
}

/// Hand one store's object to the concurrent mark's helpers and note it for
/// the finalize. The caller has already established that a mark is running and
/// the object carries the mark, so the work is purely the bookkeeping that
/// turns "the helpers may have finished with this neighbourhood" into "it will
/// be walked again": `mark_log_` is the finalize's memory of it, and the
/// round's shared stack gets it at once, so a helper that is still awake takes
/// it now instead of the finalize taking it later.
void Heap::log_mutation(Obj* dst) {
    mark_log_.push_back(dst);
    GcRound* r = round_;
    if (!r) return;
    bool wake;
    {
        std::lock_guard<std::mutex> g(r->mutex);
        // The trace may have finished a moment ago, with every helper asleep;
        // work pushed now would sit there until the finalize's join, useless.
        // The finalize re-walks `mark_log_` either way, so it is safe to let
        // the object wait for it.
        if (r->done.load(std::memory_order_relaxed)) return;
        r->stack.push_back(dst);
        r->shared.store(r->stack.size(), std::memory_order_relaxed);
        wake = r->waiting > 0;
    }
    if (wake) r->cv.notify_all();
}

template <bool Par, bool Con>
void Heap::forward_in(GcCtx& c, Value* slot) {
    // A concurrent mark does not promote -- young objects are the finalize's
    // job -- so its "forwarding" is marking, and nothing more.
    if constexpr (Con) {
        mark_ref(c, read_slot<Par>(slot));
        return;
    }
    Value v = read_slot<Par>(slot);
    if (!is_ptr(v)) return;
    Obj* o = as_obj(v);
    uint8_t gc = read_gc<Par>(o);
    // Not this heap's, and nothing under it is: see `SharedArea`. A census
    // major notes it and goes no further either -- what a region reaches is
    // the region's dependencies, not this heap's business.
    if (gc & GC_SHARED) {
        if (observe_round_) shared_observe(o, observe_round_);
        return;
    }
    if (gc & (GC_YOUNG | GC_FORWARDED | GC_BUSY)) {
        forward_slow<Par>(c, slot);
        return;
    }
    // An Indirect is folded away by the slow path, which rewrites the slot
    // with what it resolves to.
    if (o->type == ObjType::Indirect) {
        forward_slow<Par>(c, slot);
        return;
    }
    if (full_trace_ && !(gc & GC_MARK)) mark_object<Par>(c, v);
}

template <bool Par>
void Heap::forward_slow(GcCtx& c, Value* slot) {
    // A slot may hold an Indirect whose work is done. The copying collector
    // folded these by copying through them into the slot, so every reachable
    // slot came out of a collection holding the real value; readers rely on
    // that, not on resolving at the point of use. Nothing moves here, so the
    // way to keep the promise is to write the resolved value into the slot.
    Value v = read_slot<Par>(slot);
    if (is_ptr(v) && as_obj(v)->type == ObjType::Indirect) {
        v = gc_resolve<Par>(v);
        write_slot<Par>(slot, v);
    }
    if (!is_ptr(v)) return;
    Obj* o = as_obj(v);
    uint8_t gc = read_gc<Par>(o);
    if (gc & GC_SHARED) {
        if (observe_round_) shared_observe(o, observe_round_);
        return;
    }
    // Pinned: young, and staying where a compiled frame can find it. The slot
    // keeps it, and if the slot is in an old object that object is owed to
    // the next minor's remembered set -- it now points at a young one.
    if (gc & GC_PINNED) {
        c.met_pinned = true;
        return;
    }
    // A young object is copied to old space and the slot rewritten to the
    // copy. A forwarded one is a young object already copied this cycle; its
    // first payload word holds the copy, so sharing survives promotion.
    if (gc & GC_YOUNG) {
        if constexpr (Par) {
            // Claim it before copying it. Two copies of one object would be
            // two objects, and everything the language says about sharing --
            // that forcing a thunk is visible to everyone holding it -- would
            // stop being true across a collection.
            uint8_t expect = GC_YOUNG;
            if (!std::atomic_ref<uint8_t>(o->gc).compare_exchange_strong(
                    expect, uint8_t(GC_YOUNG | GC_BUSY),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                write_slot<Par>(slot, await_forward(o));
                return;
            }
        }
        write_slot<Par>(slot, from_obj(promote<Par>(c, o)));
        return;
    }
    if (gc & GC_FORWARDED) {
        write_slot<Par>(slot, forward_target(o));
        return;
    }
    if constexpr (Par) {
        if (gc & GC_BUSY) {
            write_slot<Par>(slot, await_forward(o));
            return;
        }
    }
    // Old object: only a major collection wants it marked; a minor scans the
    // remembered set and promotes, and has no need to touch the rest.
    if (full_trace_) mark_object<Par>(c, v);
}

/// Copy a nursery object to old space. The from-space chunk becomes a
/// forwarding stub -- its generation bit flips to GC_FORWARDED and its first
/// payload word points at the copy -- so the second reference to it is
/// rewritten to the same copy and sharing survives. Promoted objects are
/// marked too during a major collection, so the sweep keeps them.
///
/// In a parallel round the caller has already claimed `o`, and the flip to
/// GC_FORWARDED is what publishes the copy: every write below it is ordered
/// before the release, so a thread that sees the bit sees a finished object.
template <bool Par>
Obj* Heap::promote(GcCtx& c, Obj* o) {
    uint32_t sz = o->bytes;
    Obj* copy = gc_carve<Par>(c, sz);
    // The header travels field by field rather than inside the payload copy,
    // because `o->gc` is the one byte of the object that is not this thread's
    // to read plainly: another thread may be attempting the claim on it at
    // this very moment, and a plain read against an atomic write is a race
    // whatever the bytes turn out to be. Everything else -- `aux`, the cached
    // hash in the first payload word, the bytes themselves -- travels exactly
    // as it was, which is what `AUX_DEEP_FORCED` and the string hash rely on.
    copy->type = o->type;
    copy->aux = o->aux;
    copy->bytes = sz;
    copy->gc = full_trace_ ? uint8_t(GC_MARK | GC_OLD) : uint8_t(GC_OLD);
    std::memcpy(copy + 1, o + 1, size_t(sz) - sizeof(Obj));
    // Every class size is at least 16 bytes, so every chunk has a first
    // payload word to park the forwarding pointer in. The from-space chunk is
    // dead once the nursery is emptied, so the overwrite costs nothing.
    *reinterpret_cast<Value*>(o + 1) = from_obj(copy);
    if constexpr (Par) {
        std::atomic_ref<uint8_t>(o->gc).store(GC_FORWARDED, std::memory_order_release);
    } else {
        o->gc = GC_FORWARDED;
    }
    c.promoted += sz;
    c.total += sz;
    if (!is_atom_object(o->type)) c.q->push_back(copy);
    return copy;
}

namespace {

/// Every reference an object holds, handed to `visit` as a slot to read and
/// perhaps rewrite. Written once for everything that walks an object's fields:
/// the trace, which promotes and marks through them, and the sweep after an
/// evacuation, which points them at where their targets went. Two copies of
/// this list would be two places to forget a field, and a forgotten field is a
/// reference the collector frees or leaves pointing at a block it handed back.
///
/// `type` is passed in rather than read here because the concurrent mark has to
/// read it atomically and nobody else does.
template <class F>
inline void for_each_slot(Obj* o, ObjType type, F&& visit) {
    switch (type) {
        case ObjType::Cons: {
            auto* x = static_cast<ConsObj*>(o);
            visit(&x->head);
            visit(&x->tail);
            break;
        }
        case ObjType::Array: {
            auto* a = static_cast<ArrayObj*>(o);
            for (uint32_t i = 0; i < a->len; ++i) visit(&a->items()[i]);
            break;
        }
        case ObjType::Record: {
            // The id is an atom, an immediate, so the fields are all of it.
            auto* r = static_cast<RecordObj*>(o);
            for (uint32_t i = 0; i < r->len; ++i) visit(&r->items()[i]);
            break;
        }
        case ObjType::Map: {
            auto* m = static_cast<MapObj*>(o);
            uint32_t n = map_bit_count(m->bitmap);
            for (uint32_t i = 0; i < n; ++i) visit(&m->slots()[i]);
            break;
        }
        case ObjType::MapLeaf: {
            auto* l = static_cast<MapLeafObj*>(o);
            visit(&l->key);
            visit(&l->value);
            visit(&l->next);
            break;
        }
        case ObjType::Closure: {
            auto* x = static_cast<ClosureObj*>(o);
            for (uint32_t i = 0; i < x->ncaps; ++i) visit(&x->caps()[i]);
            break;
        }
        case ObjType::Thunk:
        case ObjType::Blackhole: {
            auto* t = static_cast<ThunkObj*>(o);
            visit(&t->frame);
            break;
        }
        case ObjType::Indirect: {
            auto* ind = static_cast<IndirectObj*>(o);
            visit(&ind->target);
            break;
        }
        case ObjType::Tensor: {
            auto* t = static_cast<TensorObj*>(o);
            if (!tensor_deferred(t)) break;
            TensorExpr* e = tensor_expr(t);
            visit(&e->result);
            visit(&e->product_a);
            visit(&e->product_b);
            for (uint32_t i = 0; i < e->ninputs; ++i) visit(&e->inputs()[i]);
            break;
        }
        case ObjType::Pap: {
            auto* p = static_cast<PapObj*>(o);
            visit(&p->fn);
            for (uint32_t i = 0; i < p->nargs; ++i) visit(&p->args()[i]);
            break;
        }
        case ObjType::Frame: {
            auto* f = static_cast<FrameObj*>(o);
            visit(&f->closure);
            for (uint32_t i = 0; i < f->nslots; ++i) visit(&f->slots()[i]);
            break;
        }
        case ObjType::ErrorBox: {
            auto* e = static_cast<ErrorObj*>(o);
            visit(&e->kind);
            visit(&e->payload);
            visit(&e->where);
            break;
        }
        case ObjType::Module: {
            visit(&static_cast<ModuleObj*>(o)->name);
            break;
        }
        case ObjType::Native: {
            visit(&static_cast<NativeObj*>(o)->name);
            break;
        }
        // Float, Str, Pid, BigStr and BigInt hold no references.
        default:
            break;
    }
}

}  // namespace

template <bool Par, bool Con>
void Heap::scan_object(GcCtx& c, Obj* o) {
    // Each reference is either forwarded (the copying collector: promote a
    // young one, mark an old one, fold an indirection) or, when `Con`, merely
    // marked -- the concurrent mark's whole scan.
    // A root that named a pinned object set it too, and roots are nobody's.
    if constexpr (!Con) c.met_pinned = false;
    for_each_slot(o, read_type<Par>(o), [&](Value* slot) {
        if constexpr (Con)
            mark_ref(c, read_slot<Par>(slot));
        else
            forward_in<Par, Con>(c, slot);
    });
    // Once per object, however many of its fields named one: a record per
    // field went back on the remembered set as that many entries, each of
    // which was scanned and recorded again, and the set doubled at every
    // minor -- 33 million entries in a self-compile, and a 759 ms minor.
    if constexpr (!Con) {
        if (c.met_pinned) {
            c.met_pinned = false;
            if (!(read_gc<Par>(o) & GC_YOUNG)) c.pin_refs.push_back(o);
        }
    }
}

void Heap::forward(Value* slot) {
    // While verifying, `forward` records roots instead of marking them.
    if (recording_) {
        recording_->push_back(*slot);
        return;
    }
    // After an evacuation, the roots are pointed at the copies and nothing
    // else is done: the trace is over, and the sweep that follows does the
    // same for every reference inside the heap.
    if (fixing_roots_) {
        Value v = *slot;
        if (is_ptr(v) && as_obj(v)->gc == GC_FORWARDED) *slot = forward_target(as_obj(v));
        return;
    }
    if (parallel_)
        forward_in<true, false>(*root_ctx_, slot);
    else
        forward_in<false, false>(*root_ctx_, slot);
}

// ---------------------------------------------------------------------------
// Promoting into old space from several threads at once
// ---------------------------------------------------------------------------

bool Heap::refill_free(GcCtx& c, size_t cls) {
    if (c.free_dry[cls]) return false;
    std::lock_guard<std::mutex> g(round_->alloc_mutex);
    Obj* head = free_lists_[cls];
    if (!head) {
        c.free_dry[cls] = true;
        return false;
    }
    // Take a bounded batch rather than the whole chain: the whole chain would
    // leave every other thread carving fresh blocks while the space it was
    // holding went unused.
    Obj* tail = head;
    for (unsigned n = 1; n < kFreeBatch && free_next(tail); ++n) tail = free_next(tail);
    free_lists_[cls] = free_next(tail);
    set_free_next(tail, nullptr);
    c.free_head[cls] = head;
    c.free_tail[cls] = tail;
    return true;
}

Heap::Block* Heap::thread_block(uint32_t sz) {
    // A block per thread, not a cursor shared between them. The waste is the
    // tail of each thread's last block, which the next serial carve finds and
    // fills; the alternative is a lock on every object promoted.
    std::lock_guard<std::mutex> g(round_->alloc_mutex);
    Block* b = new_block(sz);
    b->next = blocks_;
    blocks_ = b;
    return b;
}

void Heap::merge_free_lists(Obj** head, Obj** tail) {
    for (size_t i = 0; i < kHeapClassCount; ++i) {
        if (!head[i]) continue;
        set_free_next(tail[i], free_lists_[i]);
        free_lists_[i] = head[i];
        head[i] = nullptr;
        tail[i] = nullptr;
    }
}

template <bool Par>
Obj* Heap::gc_carve(GcCtx& c, uint32_t sz) {
    if constexpr (!Par) {
        return carve(sz);
    } else {
        size_t i = class_index(sz);
        Obj* o = c.free_head[i];
        if (!o && refill_free(c, i)) o = c.free_head[i];
        if (o) {
            c.free_head[i] = free_next(o);
            if (!c.free_head[i]) c.free_tail[i] = nullptr;
            c.allocated += sz;
            return o;
        }
        Block* b = c.carve_block;
        if (!b || b->used + sz > b->size) {
            b = thread_block(sz);
            c.carve_block = b;
        }
        o = reinterpret_cast<Obj*>(b->data + b->used);
        b->used += sz;
        c.allocated += sz;
        return o;
    }
}

// ---------------------------------------------------------------------------
// Dividing the trace
// ---------------------------------------------------------------------------

void Heap::share_work(GcCtx& c) {
    std::vector<Obj*>& q = *c.q;
    size_t give = q.size() / 2;
    if (give == 0) return;
    bool wake;
    {
        std::lock_guard<std::mutex> g(round_->mutex);
        // The *bottom* half. Work found early is nearer the roots, so it
        // stands for a larger subgraph than the leaves on top, and a thread
        // that takes it has something to do for a while.
        round_->stack.insert(round_->stack.end(), q.begin(), q.begin() + ptrdiff_t(give));
        round_->shared.store(round_->stack.size(), std::memory_order_relaxed);
        wake = round_->waiting > 0;
    }
    q.erase(q.begin(), q.begin() + ptrdiff_t(give));
    if (wake) round_->cv.notify_all();
}

/// Move a batch off the shared stack into `c`. The caller holds the lock.
static void claim_batch(std::vector<Obj*>& stack, std::atomic<size_t>& shared,
                        std::vector<Obj*>& into, size_t batch) {
    size_t take = std::min(stack.size(), batch);
    into.insert(into.end(), stack.end() - ptrdiff_t(take), stack.end());
    stack.resize(stack.size() - take);
    shared.store(stack.size(), std::memory_order_relaxed);
}

bool Heap::take_work(GcCtx& c) {
    GcRound& r = *round_;
    // `idle` is this thread's half of `r.active`: it has said it has nothing
    // to do and has not taken it back. Every change of it happens under the
    // lock, because it is what the end of the trace is decided from.
    bool idle = false;
    unsigned spins = 0;
    for (;;) {
        if (r.shared.load(std::memory_order_acquire) != 0) {
            std::lock_guard<std::mutex> g(r.mutex);
            if (!r.stack.empty()) {
                if (idle) {
                    ++r.active;
                    r.idle.fetch_sub(1, std::memory_order_relaxed);
                    idle = false;
                }
                claim_batch(r.stack, r.shared, *c.q, kTakeBatch);
                return true;
            }
        }
        if (!idle) {
            // Out of work, and so is the shared stack. Say so; when the last
            // thread that has arrived says so too, there is nowhere a grey
            // object could still be and the trace is over.
            std::unique_lock<std::mutex> lk(r.mutex);
            idle = true;
            r.idle.fetch_add(1, std::memory_order_relaxed);
            if (--r.active == 0 && r.arrived == r.size) {
                r.done.store(true, std::memory_order_release);
                lk.unlock();
                r.cv.notify_all();
                return false;
            }
            continue;
        }
        if (r.done.load(std::memory_order_acquire)) return false;
        if (++spins < kIdleSpins) {
            spin_once();
            continue;
        }
        // Long enough. Sleep, and let whoever finds work wake us.
        std::unique_lock<std::mutex> lk(r.mutex);
        ++r.waiting;
        r.cv.wait(lk, [&] {
            return r.done.load(std::memory_order_relaxed) || !r.stack.empty();
        });
        --r.waiting;
        if (r.done.load(std::memory_order_relaxed)) return false;
        ++r.active;
        r.idle.fetch_sub(1, std::memory_order_relaxed);
        claim_batch(r.stack, r.shared, *c.q, kTakeBatch);
        return true;
    }
}

template <bool Par, bool Con>
void Heap::drain(GcCtx& c) {
    std::vector<Obj*>& q = *c.q;
    for (;;) {
        while (!q.empty()) {
            Obj* o = q.back();
            q.pop_back();
            scan_object<Par, Con>(c, o);
            if constexpr (Par) {
                ++c.scanned;
                if (--c.check_in <= 0) {
                    c.check_in = kCheckEvery;
                    // Give work away when somebody is waiting for it, or when
                    // the local queue has grown past what one thread can be
                    // expected to get through on its own.
                    if (q.size() >= kShareAbove ||
                        (q.size() >= kShareMin &&
                         round_->idle.load(std::memory_order_relaxed) != 0)) {
                        share_work(c);
                    }
                }
            }
        }
        if constexpr (!Par) {
            return;
        } else {
            if (!take_work(c)) return;
        }
    }
}

bool Heap::trace_in_parallel(RootSource& roots, bool minor) {
    size_t work = minor ? nursery_bytes_ : allocated_;
    if (work < parallel_floor(minor)) return false;
    GcPool& pool = GcPool::instance();
    if (pool.capacity() < 2) return false;

    GcRound round;
    round.ctxs.resize(pool.capacity());
    round_ = &round;
    parallel_ = true;

    bool ran = pool.run(pool.capacity(), [&](unsigned index, unsigned count) {
        GcCtx& c = round.ctxs[index];
        c.q = &c.own;
        c.check_in = kCheckEvery;
        {
            std::lock_guard<std::mutex> g(round.mutex);
            round.size = count;
            ++round.arrived;
            ++round.active;
        }
        // The roots are the collecting thread's own business: they are few,
        // and they are the only part of the heap that lives outside it.
        if (index == 0) {
            root_ctx_ = &c;
            roots.visit_roots(*this);
            root_ctx_ = nullptr;
            trace_pinned<true>(c);
        }
        if (minor) {
            // The remembered set divides by index. An object logged twice is
            // scanned twice, possibly by two threads at once -- which is why
            // a slot the collector rewrites is read and written atomically.
            size_t n = remembered_.size();
            size_t lo = n * index / count;
            size_t hi = n * (index + 1) / count;
            for (size_t k = lo; k < hi; ++k) scan_object<true, false>(c, remembered_[k]);
        }
        drain<true, false>(c);
    });

    parallel_ = false;
    round_ = nullptr;
    if (ran && gc_trace()) {
        std::fprintf(stderr, ";   traced by");
        for (GcCtx& c : round.ctxs) {
            if (c.scanned) std::fprintf(stderr, " %llu", (unsigned long long)c.scanned);
        }
        std::fprintf(stderr, " objects per thread\n");
    }
    if (ran) {
        for (GcCtx& c : round.ctxs) {
            merge_free_lists(c.free_head, c.free_tail);
            pin_refs_.insert(pin_refs_.end(), c.pin_refs.begin(), c.pin_refs.end());
            allocated_ += c.allocated;
            promoted_bytes_ += c.promoted;
            total_allocated_ += c.total;
        }
        ++parallel_collections_;
    }
    return ran;
}

void Heap::trace_alone(RootSource& roots, bool minor) {
    // Every object that survives lands on the worklist at some point, so it
    // wants room for a big fraction of everything that was alive. The queue
    // retains its capacity across collects, but the live set grows, so a
    // fresh peak collection reallocates it once -- then again at the next
    // peak. Giving it a generous head start floors the size of the span it
    // holds, so each collection amortizes the growth instead of reallocating
    // a handful of times mid-collect. Nothing here is exact; it only has to
    // be in the right decade.
    if (scan_queue_.capacity() < allocated_ / size_t(16) + 1)
        scan_queue_.reserve(allocated_ / size_t(16) + 1);
    scan_queue_.clear();

    GcCtx c;
    c.q = &scan_queue_;
    root_ctx_ = &c;
    roots.visit_roots(*this);
    root_ctx_ = nullptr;
    trace_pinned<false>(c);
    if (minor) {
        for (Obj* o : remembered_) scan_object<false, false>(c, o);
    }
    drain<false, false>(c);
    promoted_bytes_ += c.promoted;
    total_allocated_ += c.total;
    pin_refs_.insert(pin_refs_.end(), c.pin_refs.begin(), c.pin_refs.end());
}

// ---------------------------------------------------------------------------
// Collecting under compiled code
// ---------------------------------------------------------------------------
//
// See `set_conservative_roots` for what this is and docs/gc.md for why. What
// follows is the mechanics, in the order a collection meets them: find the
// objects, keep them where they are, trace from them, and give their nursery
// blocks to old space.

void Heap::pin_conservative(bool full) {
    pinned_.clear();
    pinned_blocks_.clear();
    pin_refs_.clear();
    young_pins_.clear();
    if (!cons_lo_) return;
    const uint64_t started = now_nanos();

    // Every block's used extent, sorted, so that a word is placed by a binary
    // search. Built afresh: this runs once per collection under compiled code,
    // and a table kept up to date would cost every block that is ever made.
    struct Range {
        uintptr_t lo, hi;
        Block* b;
    };
    std::vector<Range> ranges;
    ranges.reserve(nursery_.size() + 64);
    auto add = [&](Block* b) {
        if (!b->used) return;
        const auto lo = reinterpret_cast<uintptr_t>(b->data);
        ranges.push_back({lo, lo + b->used, b});
    };
    for (Block* b : nursery_) add(b);
    // Old space only for a full collection. A minor moves no old object and
    // marks none, and what an old object points at in the nursery it reaches
    // through the remembered set, as it would interpreted -- so a word naming
    // one asks nothing of a minor, and finding its object would be a walk of
    // an old block for every such word, at every minor. That walk was the
    // first version's, and it made a self-compile two and a half times slower.
    if (full) {
        for (Block* b = blocks_; b; b = b->next) add(b);
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const Range& x, const Range& y) { return x.lo < y.lo; });

    // The words that land in a block, as (block, address), sorted so that
    // each block is walked once however many words point into it. A nursery
    // block is one run of objects with no index, so finding the object an
    // address is inside is a walk from the block's start.
    //
    // Most words are not candidates at all, and saying so costs two compares
    // rather than a binary search: a fixnum or any other immediate has a tag
    // bit set, an object is eight-byte aligned, and a return address or a
    // pointer into the stack is usually outside every block. On a precompiled
    // self-compile, where almost every collection runs under compiled code,
    // the search was most of this function and this function was 8% of the
    // run.
    if (ranges.empty()) return;
    const uintptr_t lowest = ranges.front().lo;
    uintptr_t highest = 0;
    for (const Range& r : ranges) highest = std::max(highest, r.hi);
    //
    // A stack names the same object many times over -- the thunk being forced
    // is in every frame between the force and the loop -- and every word that
    // lands is a hit to sort. A small table of the addresses already taken
    // drops most repeats before they cost a place in the sort: on a
    // precompiled self-compile the sort had become as large as the scan.
    std::vector<std::pair<size_t, uintptr_t>> hits;
    uintptr_t seen[256] = {};
    for (const uintptr_t* w = cons_lo_; w < cons_hi_; ++w) {
        const uintptr_t a = *w;
        if ((a & 7) != 0 || a < lowest || a >= highest) continue;
        uintptr_t& slot = seen[(a >> 3) & 255];
        if (slot == a) continue;
        slot = a;
        auto it = std::upper_bound(ranges.begin(), ranges.end(), a,
                                   [](uintptr_t key, const Range& r) { return key < r.lo; });
        if (it == ranges.begin()) continue;
        --it;
        if (a >= it->hi) continue;
        hits.push_back({size_t(it - ranges.begin()), a});
    }
    std::sort(hits.begin(), hits.end());

    size_t i = 0;
    while (i < hits.size()) {
        const Range& r = ranges[hits[i].first];
        Block* b = r.b;
        size_t pos = 0;
        Obj* last = nullptr;
        for (; i < hits.size() && hits[i].first == size_t(&r - ranges.data()); ++i) {
            const uintptr_t at = hits[i].second - r.lo;
            // Step to the object that holds `at`. The hits are in address
            // order, so the walk only ever goes forward.
            Obj* o = nullptr;
            while (pos < b->used) {
                auto* here = reinterpret_cast<Obj*>(b->data + pos);
                if (at < pos + here->bytes) {
                    o = here;
                    break;
                }
                pos += here->bytes;
            }
            if (!o || o == last) continue;
            last = o;
            // Below the line of a block the last collection held, only what
            // it pinned is alive: everything else there was dead or moved
            // then, and is left as it was rather than marked so.
            if (pos < b->held_end &&
                !std::binary_search(held_pins_.begin() + b->held_from,
                                    held_pins_.begin() + b->held_from + b->held_count,
                                    uint32_t(pos))) {
                continue;
            }
            const uint8_t gc = o->gc;
            // A hole on a free list, or what a word left over from a call long
            // returned happens to point at: nothing to keep.
            if (gc & (GC_FREE | GC_FORWARDED | GC_PINNED)) continue;
            if (gc & GC_YOUNG) {
                // Left where it stands and left young. Promoting it in place
                // was the first version, and it was wrong in a way worth
                // knowing: the object a compiled loop holds is usually the
                // thunk it is forcing, which is updated a moment later to
                // point at what it produced -- an old object pointing at the
                // young, remembered, and every minor after that promoted
                // whatever had been produced since. A young pinned object
                // makes no such edge. The next collection judges it like any
                // other: copied out if it is still reached, gone if not.
                o->gc = uint8_t(GC_YOUNG | GC_PINNED);
                young_pins_.push_back({b, uint32_t(pos)});
                if (!b->pinned) {
                    b->pinned = true;
                    pinned_blocks_.push_back(b);
                }
            } else if (b->evacuate) {
                // An old object never moves except by evacuation, so keeping
                // its block out of this one is all keeping it takes. The block
                // is left off the free lists until the next sweep, which is
                // what the choice had already arranged.
                b->evacuate = false;
            }
            pinned_.push_back(o);
        }
    }
    if (gc_trace())
        std::fprintf(stderr,
                     "; under compiled code: %zu words, %zu pinned, %zu blocks held, %.3f ms\n",
                     size_t(cons_hi_ - cons_lo_), pinned_.size(), pinned_blocks_.size(),
                     double(now_nanos() - started) / 1e6);
}

template <bool Par>
void Heap::trace_pinned(GcCtx& c) {
    for (Obj* o : pinned_) {
        if (!(o->gc & GC_YOUNG) && full_trace_) {
            // Marks the object itself, an `Indirect` included -- a compiled
            // frame may hold the indirection rather than what it leads to.
            mark_object<Par>(c, from_obj(o));
        } else if (!is_atom_object(o->type)) {
            // Its fields, as a remembered object's are: a young pinned object
            // is never marked, because the sweep never sees the nursery, and
            // a minor marks nothing.
            scan_object<Par, false>(c, o);
        }
    }
}

void Heap::hold_pinned_blocks() {
    // Everything in a held block but the pinned is dead or the stub of
    // something promoted, and stays there until a collection empties the
    // block. A stale word on a later scan must not bring one back: its fields
    // name objects that have moved or been overwritten. That used to be said by
    // marking each one free, a walk of every held block -- 3.2 GB of them on a
    // precompiled self-compile, 2% of the run. Now the block says it: how far
    // it was filled (`held_end`) and which offsets below that were pinned,
    // which is all `pin_conservative` asks, and nothing else walks a nursery
    // block.
    held_pins_.clear();
    std::sort(young_pins_.begin(), young_pins_.end());
    std::sort(pinned_blocks_.begin(), pinned_blocks_.end());
    size_t at = 0;
    for (Block* b : pinned_blocks_) {
        b->held_end = uint32_t(b->used);
        b->held_from = uint32_t(held_pins_.size());
        // `young_pins_` is sorted by block, then offset; this block's run.
        while (at < young_pins_.size() && young_pins_[at].first < b) ++at;
        while (at < young_pins_.size() && young_pins_[at].first == b) {
            held_pins_.push_back(young_pins_[at++].second);
        }
        b->held_count = uint32_t(held_pins_.size()) - b->held_from;
        // Held, so still allocated -- but not new allocation, which is what
        // `nursery_bytes_` measures and a minor is due on. Counted there, a
        // held block made the next minor due the moment this one ended.
        nursery_held_ += b->used;
        allocated_ += b->used;
        b->pinned = false;
    }
    pinned_blocks_.clear();
    // A survivor that points at a pinned object is an old object pointing at
    // a young one, which is exactly what the remembered set is for. Once
    // each: an object scanned twice -- remembered twice, say -- is recorded
    // twice, and the next minor would scan it twice and record it twice more.
    std::sort(pin_refs_.begin(), pin_refs_.end());
    pin_refs_.erase(std::unique(pin_refs_.begin(), pin_refs_.end()), pin_refs_.end());
    remembered_.insert(remembered_.end(), pin_refs_.begin(), pin_refs_.end());
    pin_refs_.clear();
}

void Heap::forward_pin_refs() {
    if (!fixing_refs_) return;
    // An evacuation may have moved one of them; the stub names the copy until
    // the sweep frees its block.
    for (Obj*& o : pin_refs_) {
        if (o->gc == GC_FORWARDED) o = as_obj(forward_target(o));
    }
    // And a pinned young object may point at something it moved. The sweep
    // rewrites those references in every live object it walks, and it never
    // walks the nursery -- so these are rewritten here, or keep naming a stub
    // in a block about to be handed back.
    for (Obj* o : pinned_) {
        if (o->gc & GC_YOUNG) fix_refs(o);
    }
}

void Heap::collect(RootSource& roots) {
    major_collect(roots);
}

/// A minor collection's cost is what it promoted, not what it looked at: the
/// nursery is emptied for nothing whatever is in it. So a collection that
/// copied out a large share of the nursery was simply too early -- most of
/// what it copied would have died had it waited -- and the answer is a bigger
/// nursery, not a cleverer collection. A quarter is the line: above it the
/// nursery doubles, below it nothing changes, and a process whose garbage
/// really is long-lived stops at the cap rather than growing for ever.
void Heap::grow_nursery_if_crowded(size_t promoted, size_t looked_at) {
    if (nursery_hi_ >= nursery_max_) return;
    if (promoted * 4 <= looked_at) return;
    nursery_hi_ *= 2;
    if (nursery_hi_ > nursery_max_) nursery_hi_ = nursery_max_;
}

void Heap::major_collect(RootSource& roots) {
    const uint64_t started = now_nanos();
    pin_conservative(true);

    // Mark. `forward` promotes every reachable young object -- copying it to
    // old space -- and marks the old ones. The roots are grey before anything
    // else; scanning turns them black and enqueues everything they point at,
    // until the worklist drains and the grey set is empty.
    full_trace_ = true;
    if (!trace_in_parallel(roots, /*minor=*/false)) trace_alone(roots, /*minor=*/false);
    full_trace_ = false;
    reap_external(true);
    // Everything alive is marked now, and that is all evacuation needs: the
    // blocks the last sweep chose are emptied into the holes of the rest,
    // and the sweep below rewrites the references as it passes them.
    evacuate(roots);
    forward_pin_refs();
    const uint64_t marked = now_nanos();

    // Sweep. Every old block's chunk headers tile the block, so one walk over
    // each block sees every object: the marked ones stay put (and lose their
    // mark, for the next cycle), the rest go on their size class's free list.
    // Everything reachable in the nursery was promoted, so the whole nursery
    // is dead and is handed back.
    if (!sweep_in_parallel()) sweep();
    free_nursery();
    remembered_.clear();
    hold_pinned_blocks();

    ++collections_;
    ++major_collections_;

    // Collect again once the heap has grown well past what survived, so that
    // programs with a large live set do not collect continuously. The `live * 2`
    // floor means a steady state with a modest live set (a program chewing
    // through transient lists) never pays to collect more than a couple of
    // times before the picture settles -- an allocate-heavy program that makes
    // little that survives wastes a copy of everything it ever builds if it
    // collects on every doubling.
    gc_threshold_ = live_after_gc_ * 3 + initial_bytes_;
    // The nursery scales with the live set too: a process that holds a lot
    // wants a bigger nursery before it pays for a minor collection. The cap
    // is the same one the survival rate grows towards, so a process cannot
    // reach a nursery through this that it could not have earned.
    nursery_hi_ = live_after_gc_ / 4 + initial_bytes_;
    if (nursery_hi_ > nursery_max_) nursery_hi_ = nursery_max_;
    // A major has just emptied the nursery and refitted both thresholds, so
    // whatever a minor last measured is about a heap that no longer exists.
    // Give the generational bet its odds back and let the next minor judge.
    minor_pays_ = true;

    major_nanos_ += now_nanos() - started;
    if (gc_trace())
        std::fprintf(stderr,
                     "; major: %zu live, %zu held, %.2f ms mark, %.2f ms sweep,"
                     " %zu evacuated from %zu blocks, %zu chosen%s\n",
                     live_after_gc_, block_bytes_, double(marked - started) / 1e6,
                     double(now_nanos() - marked) / 1e6, last_evacuated_,
                     last_evacuated_blocks_, evacuate_pending_, last_kinds_.c_str());
    verify_collect(roots, false);
}

void Heap::minor_collect(RootSource& roots) {
    const uint64_t started = now_nanos();
    const uint64_t promoted_before = promoted_bytes_;
    const size_t looked_at = nursery_bytes_;
    const size_t remembered_n = remembered_.size();
    pin_conservative(false);

    // Trace. Roots have their young values promoted directly; each remembered
    // old object is scanned so a young value it was made to point at is
    // promoted too. Promotion copies reach every other young object through
    // the promoted one, so when the worklist drains, everything reachable is
    // in old space and the nursery is empty by construction.
    if (!trace_in_parallel(roots, /*minor=*/true)) trace_alone(roots, /*minor=*/true);
    reap_external(false);

    // Every nursery block is now dead space; reset them all and rewind to the
    // first, so the next bump allocation refills the blocks we already have.
    // (They are kept, not freed -- the process will fill them again at once.)
    // A block holding a pinned object is the exception, and is counted again
    // by `hold_pinned_blocks`.
    for (Block* b : nursery_) {
        if (!b->pinned) b->used = b->held_end = 0;
    }
    nursery_at_ = 0;
    allocated_ -= nursery_bytes_ + nursery_held_;
    nursery_bytes_ = 0;
    nursery_held_ = 0;
    remembered_.clear();
    hold_pinned_blocks();

    ++collections_;
    ++minor_collections_;
    if (allocated_ > peak_live_) peak_live_ = allocated_;
    // What this minor cost against what it recovered. A nursery that was
    // nearly all survivors is one a minor cannot help with -- it copies the
    // lot into old space and the lot dies there -- and `major_due` reads this
    // to stop choosing minors in that case. The same ratio, at a different
    // threshold, is what decides the nursery should grow.
    const size_t promoted_now = size_t(promoted_bytes_ - promoted_before);
    minor_pays_ = promoted_now * 4 <= looked_at * 3;
    grow_nursery_if_crowded(promoted_now, looked_at);
    minor_nanos_ += now_nanos() - started;
    if (gc_trace())
        std::fprintf(stderr,
                     "; minor: %zu nursery, %llu promoted, %zu remembered, %zu held, %.2f ms\n",
                     looked_at, (unsigned long long)(promoted_bytes_ - promoted_before),
                     remembered_n, block_bytes_, double(now_nanos() - started) / 1e6);

    // A minor never touches old space, so `live_after_gc_` still says what the
    // last full collection measured; `bytes_allocated` reports the whole live
    // picture. The check a minor adds is that nothing reachable is young.
    verify_collect(roots, true);
}

// ---------------------------------------------------------------------------
// A major whose marking overlaps the mutator
// ---------------------------------------------------------------------------
//
// Phase 2's step 3: the pool that a parallel collection borrows to shorten its
// pause also works *during* the pause on a major's marking. A heap that is due
// for a full collection snapshots its roots, hands the snapshot to the helper
// threads, and keeps running -- the helpers walk old objects; the mutator
// keeps allocating and storing; `should_collect` insists the only collection
// the mutator may do meanwhile is the finalize of the one in flight. What the
// helpers grey is a picture of the old graph as it was at the safepoint; the
// finalize re-walks exactly what changed after it (see below), so the two
// together divide the marking, with the younger, hotter half on the helpers
// and the delta -- usually small -- at the finalize.

bool Heap::start_concurrent_mark(RootSource& roots) {
    // Overlapping the mark costs a pool round and, for every store into an old
    // object while the helpers run, a couple of atomic loads and a log. A heap
    // that could have finished its marking inside the handshake pays all of
    // that for nothing, so size is the gate -- the same per-arg idea as
    // `parallel_floor`, with a heavier thumb on the scale. `DREAM_GC_CONCURRENT=1`
    // forces every major through this path (the race detector's door in), and
    // `=0` switches it off regardless of size.
    static const int mode = [] {
        if (const char* v = std::getenv("DREAM_GC_CONCURRENT")) {
            if (std::strcmp(v, "1") == 0) return 1;
            if (std::strcmp(v, "0") == 0) return 0;
        }
        return -1;
    }();
    if (mode == 0) return false;
    if (mode < 0 && allocated_ < (size_t(1) << 24)) return false;
    GcPool& pool = GcPool::instance();
    if (pool.capacity() < 2) return false;
    const unsigned want = pool.capacity();

    // The round must outlive this call: the helpers work on it after the
    // function has returned, and the barrier feeds it until the finalize. It
    // lives on the heap, not the stack, for exactly that reason.
    if (!mark_round_) mark_round_ = std::make_unique<GcRound>();
    GcRound& round = *mark_round_;
    // `GcRound` is a stack-shaped object (see `trace_in_parallel`), so a roll
    // like this has to reset every field one by one rather than start fresh.
    round.stack.clear();
    round.shared.store(0, std::memory_order_relaxed);
    round.idle.store(0, std::memory_order_relaxed);
    round.arrived = 0;
    round.active = 0;
    round.waiting = 0;
    round.done.store(false, std::memory_order_relaxed);
    round.ctxs.assign(want, GcCtx());
    // The caller does not run the body, so the round belongs to the helpers
    // alone; each announces the count it was given when it arrives.
    round.size = 0;
    round_ = &round;
    marking_ = true;
    mark_done_.store(false, std::memory_order_relaxed);
    concurrent_started_ = now_nanos();

    // Snapshot the roots at this safepoint, while they are still the
    // mutator's. `forward` doubles as a recorder when `recording_` is set -- the
    // same trick `verify_internal` uses, this time collecting entry points
    // rather than walking anything.
    mark_roots_.clear();
    recording_ = &mark_roots_;
    roots.visit_roots(*this);
    recording_ = nullptr;

    // Seed the grey set: every old root is claimed and handed to the shared
    // stack, so the first helper to wake finds something to draw from. Nothing
    // races here -- the helpers have not been told the round exists yet.
    std::vector<Obj*> claimed_roots;
    for (Value v : mark_roots_) {
        if (!is_ptr(v)) continue;
        Obj* o = as_obj(v);
        if (claim_mark(o)) {
            claimed_roots.push_back(o);
            if (!is_atom_object(o->type)) round.stack.push_back(o);
        }
    }
    round.shared.store(round.stack.size(), std::memory_order_relaxed);

    // The helpers' job for this round: arrive, drain the graph, say done. A
    // member -- not a stack object -- for the same reason the round itself is:
    // the pool invokes it from another thread after this function has gone.
    mark_body_ = [this](unsigned index, unsigned count) {
        GcCtx& c = mark_round_->ctxs[index];
        c.q = &c.own;
        c.check_in = kCheckEvery;
        {
            std::lock_guard<std::mutex> g(mark_round_->mutex);
            mark_round_->size = count - 1;  // the caller is not among us
            ++mark_round_->arrived;
            ++mark_round_->active;
        }
        drain<true, true>(c);
        mark_done_.store(true, std::memory_order_relaxed);
    };
    if (!pool.start(want, mark_body_)) {
        // The pool declined between our check and the launch -- another heap
        // took it, or the scheduler's idle hint moved. Back the sealed round
        // out so the caller can do the usual thing, and leave no trace.
        // No helper ran, but seeding already marked the roots. The fallback
        // tracer must claim and scan them again or it will reclaim their
        // children. Include atom roots and undo only marks we acquired here.
        for (Obj* o : claimed_roots) o->gc &= ~GC_MARK;
        marking_ = false;
        round_ = nullptr;
        mark_round_.reset();
        mark_roots_.clear();
        mark_log_.clear();
        mark_born_.clear();
        return false;
    }
    return true;
}

void Heap::finalize_concurrent_mark(RootSource& roots) {
    const uint64_t started = now_nanos();
    const uint64_t overlap_ns = now_nanos() - concurrent_started_;

    // The mutator may not collect while the helpers have old space open, so the
    // finalize's first act is to take the helpers away. They work on this
    // heap's memory, and the sweep below is about to reconsider all of it.
    marking_ = false;
    GcPool::instance().join();
    // Only now: a helper may still have been claiming marks until the join,
    // and pinning writes the same byte.
    pin_conservative(true);

    // The snapshot was read while the helpers were still running, so it may
    // name young objects this finalize is about to move. Let it go before the
    // nursery is emptied.
    mark_roots_.clear();

    // Re-walk what the overlap could not have seen: the old objects the
    // mutator stored into while the helpers were working (a child can arrive
    // after its parent was scanned), and the old objects born during the mark,
    // whose fresh payloads were filled in after their headers were published.
    // Both go on the remembered set, which the minor-shaped trace below scans
    // first -- and which is sitting ready for exactly that walk.
    remembered_.insert(remembered_.end(), mark_log_.begin(), mark_log_.end());
    mark_log_.clear();
    remembered_.insert(remembered_.end(), mark_born_.begin(), mark_born_.end());
    mark_born_.clear();
    mark_round_.reset();
    round_ = nullptr;

    // Everything the helpers greyed lay in the seeded graph; the mutator has
    // moved on since, so what is left for the trace is the delta, and the
    // trace skips whatever already carries the mark. This is the rest of a
    // major -- promote the young through the remembered set, sweep, empty the
    // nursery, call it done -- with the minor's tail shape on top of the
    // major's mark.
    full_trace_ = true;
    if (!trace_in_parallel(roots, /*minor=*/true)) trace_alone(roots, /*minor=*/true);
    full_trace_ = false;
    reap_external(true);
    // The helpers are joined and the last of the graph has been traced, so
    // this is the same moment as a stopped major's: see `major_collect`.
    evacuate(roots);
    forward_pin_refs();
    if (!sweep_in_parallel()) sweep();
    free_nursery();
    remembered_.clear();
    hold_pinned_blocks();
    ++collections_;
    ++major_collections_;
    ++concurrent_marks_;

    gc_threshold_ = live_after_gc_ * 3 + initial_bytes_;
    nursery_hi_ = live_after_gc_ / 4 + initial_bytes_;
    if (nursery_hi_ > nursery_max_) nursery_hi_ = nursery_max_;
    // As in `major_collect`: the nursery is empty and both thresholds are
    // newly fitted, so the last minor's verdict is about a heap that is gone.
    minor_pays_ = true;

    const uint64_t final_ns = now_nanos() - started;
    major_nanos_ += final_ns;
    concurrent_nanos_ += overlap_ns;
    if (gc_trace())
        std::fprintf(stderr,
                     "; concurrent major: %zu live, %zu held, "
                     "%.2f ms overlap + %.2f ms stop,"
                     " %zu evacuated from %zu blocks, %zu chosen%s\n",
                     live_after_gc_, block_bytes_, double(overlap_ns) / 1e6,
                     double(final_ns) / 1e6, last_evacuated_, last_evacuated_blocks_,
                     evacuate_pending_, last_kinds_.c_str());
    verify_collect(roots, false);
}

// ---------------------------------------------------------------------------
// Sweeping
// ---------------------------------------------------------------------------

/// Whether a major empties the blocks its sweep finds mostly dead, and how dead
/// "mostly" is: a block is chosen when what survives in it is under this share
/// of its size. `DREAM_GC_EVACUATE=0` turns evacuation off -- and with it the
/// rebuilt free lists and the freeing of empty blocks, so that the switch is a
/// clean A/B against the collector that never moved an old object -- and a
/// number from 1 to 99 sets the share in percent. `all` evacuates every block
/// that is not empty, at every major, whatever it holds -- which is useless for
/// memory and is how the tests get a small program to move nearly every old
/// object it has, every time (`just test-heap`).
///
/// A quarter is not a tuned number. Nothing measured so far tells 25 from 50
/// apart -- sparse blocks turned out to be rare, and nearly all of what the
/// switch buys is the freeing of empty ones -- so the default is the one that
/// copies least. "Evacuating sparse blocks" in docs/gc.md has the numbers.
static unsigned evacuate_below_percent() {
    static const unsigned pct = [] {
        if (const char* v = std::getenv("DREAM_GC_EVACUATE")) {
            if (std::strcmp(v, "all") == 0) return 100u;
            long n = std::atol(v);
            if (n <= 0) return 0u;
            if (n < 100) return unsigned(n);
        }
        return 25u;
    }();
    return pct;
}

void Heap::fix_refs(Obj* o) {
    for_each_slot(o, o->type, [](Value* slot) {
        Value v = *slot;
        if (!is_ptr(v)) return;
        // Another sweep thread may be clearing this target's mark at the same
        // moment, so the byte is read as an atomic; the one bit asked about
        // here is never set or cleared by anybody during a sweep. A stub's
        // payload is read plainly because nothing writes it: an evacuated
        // block is off the list, and no thread sweeps it.
        Obj* t = as_obj(v);
        if (std::atomic_ref<uint8_t>(t->gc).load(std::memory_order_relaxed) == GC_FORWARDED)
            *slot = forward_target(t);
    });
}

bool Heap::choose_to_evacuate(const Block* b, size_t live_here) {
    const unsigned pct = evacuate_below_percent();
    if (b->big) return false;
    if (pct >= 100) return true;
    // A block whose tail is still mostly unused is not fragmented, it is new:
    // the carve will fill it.
    if (b->used * 2 < b->size) return false;
    if (live_here * 100 >= size_t(pct) * b->size) return false;
    const size_t before = evacuate_chosen_.fetch_add(live_here, std::memory_order_relaxed);
    if (before + live_here <= evacuate_budget_) return true;
    evacuate_chosen_.fetch_sub(live_here, std::memory_order_relaxed);
    return false;
}

void Heap::sweep_block(Block* b, Obj** head, Obj** tail, size_t& live, uint64_t* by_type) {
    // Every store this sweep makes to a header is atomic, relaxed, because
    // after an evacuation another sweep thread reads headers it does not own:
    // `fix_refs` asks each reference's target whether it is a stub. On the
    // machines this runs on that is the same instruction as a plain store.
    auto set_gc = [](Obj* o, uint8_t g) {
        std::atomic_ref<uint8_t>(o->gc).store(g, std::memory_order_relaxed);
    };
    const bool fixing = fixing_refs_;
    if (b->big) {
        // A dedicated block carries exactly one big object. It never joins
        // a size class (its size is past the table), so the choice is
        // keep it or hand the whole block back. Nothing else was ever
        // carved into it, so freeing it cannot strand a live chunk.
        Obj* o = reinterpret_cast<Obj*>(b->data);
        if (o->gc & GC_MARK) {
            if (fixing) fix_refs(o);
            set_gc(o, o->gc & ~GC_MARK);
            live += o->bytes;
            by_type[size_t(o->type) & 31] += o->bytes;
            ++by_type[32 + (size_t(o->type) & 31)];
        } else {
            b->dead = true;
        }
        return;
    }

    // What the chains looked like before this block, so that a block chosen
    // to be evacuated -- or found empty -- can take back what it put on them.
    // Its chunks are then on no list at all, which is the point: nothing may
    // be allocated into a block that is about to be handed back.
    const bool moving = evacuate_below_percent() != 0;
    Obj* head_before[kHeapClassCount];
    Obj* tail_before[kHeapClassCount];
    if (moving) {
        std::memcpy(head_before, head, sizeof head_before);
        std::memcpy(tail_before, tail, sizeof tail_before);
    }

    // A shared block: its chunks tile it exactly, so one walk sees every
    // object and always ends precisely at `used`.
    size_t here = 0;
    size_t pos = 0;
    while (pos < b->used) {
        auto* o = reinterpret_cast<Obj*>(b->data + pos);
        uint32_t sz = o->bytes;
        pos += sz;
        if (o->gc & GC_MARK) {
            // Tenured objects keep their generation bit; only the cycle's
            // mark is cleared.
            if (fixing) fix_refs(o);
            set_gc(o, o->gc & ~GC_MARK);
            here += sz;
            by_type[size_t(o->type) & 31] += sz;
            ++by_type[32 + (size_t(o->type) & 31)];
        } else if ((o->gc & GC_FREE) && !moving) {
            // Alive on a list from an earlier cycle, untouched since. The
            // free bit is what keeps us from pushing it a second time; it
            // still points off its list head. Leave the header alone.
        } else {
            // Dead, or free from an earlier cycle. When blocks can be chosen
            // for evacuation the lists are rebuilt from nothing at every sweep
            // -- the only way to leave a chosen block's holes off them -- so a
            // chunk that was already free is threaded again with the rest.
            set_gc(o, GC_FREE);
            size_t i = class_index(sz);
            set_free_next(o, head[i]);
            if (!head[i]) tail[i] = o;
            head[i] = o;
        }
    }
    live += here;
    if (!moving) return;
    // An empty block goes back now; a sparse one waits to be emptied by the
    // next major. Either way its chunks come off the lists again.
    if (here == 0 || choose_to_evacuate(b, here)) {
        std::memcpy(head, head_before, sizeof head_before);
        std::memcpy(tail, tail_before, sizeof tail_before);
        if (here == 0) b->dead = true;
        else b->evacuate = true;
    }
}

void Heap::evacuate(RootSource& roots) {
    last_evacuated_ = 0;
    last_evacuated_blocks_ = 0;
    if (!evacuate_pending_) return;
    // Every marked object in a chosen block is copied, exactly as promotion
    // copies a young one: header and payload as they were, `aux` and the
    // cached hash with them, and a stub left behind whose first payload word
    // names the copy. The copy is carved from the free lists, which the last
    // sweep rebuilt without the chosen blocks' holes, so it lands in a hole
    // somewhere else -- which is the whole of what this is for. It is born
    // marked, so the sweep keeps it.
    //
    // A new block the carve adds goes on the front of the list, behind this
    // walk, so the walk never meets it.
    size_t moved = 0;
    size_t blocks = 0;
    for (Block* b = blocks_; b; b = b->next) {
        if (!b->evacuate) continue;
        ++blocks;
        for (size_t pos = 0; pos < b->used;) {
            auto* o = reinterpret_cast<Obj*>(b->data + pos);
            const uint32_t sz = o->bytes;
            pos += sz;
            if (!(o->gc & GC_MARK)) continue;
            Obj* copy = carve(sz);
            copy->type = o->type;
            copy->aux = o->aux;
            copy->bytes = sz;
            copy->gc = GC_OLD | GC_MARK;
            std::memcpy(copy + 1, o + 1, size_t(sz) - sizeof(Obj));
            *reinterpret_cast<Value*>(o + 1) = from_obj(copy);
            o->gc = GC_FORWARDED;
            moved += sz;
        }
    }
    // What lives outside the heap is named by its object, and the object may
    // be one that moved. `reap_external` has already run, so every entry here
    // is alive.
    for (Obj*& o : external_) {
        if (o->gc == GC_FORWARDED) o = as_obj(forward_target(o));
    }
    // The roots are the only references outside the heap, and the sweep
    // rewrites every one inside it.
    fixing_roots_ = true;
    roots.visit_roots(*this);
    fixing_roots_ = false;
    fixing_refs_ = true;
    evacuated_bytes_ += moved;
    evacuated_blocks_ += blocks;
    last_evacuated_ = moved;
    last_evacuated_blocks_ = blocks;
}

std::vector<Heap::Block*> Heap::detach_evacuated() {
    std::vector<Block*> gone;
    if (!evacuate_pending_) return gone;
    Block** link = &blocks_;
    while (Block* b = *link) {
        if (b->evacuate) {
            *link = b->next;
            gone.push_back(b);
        } else {
            link = &b->next;
        }
    }
    if (carve_block_ && carve_block_->evacuate) carve_block_ = nullptr;
    evacuate_pending_ = 0;
    return gone;
}

void Heap::free_evacuated(std::vector<Block*>& gone) {
    for (Block* b : gone) free_block(b);
    gone.clear();
    fixing_refs_ = false;
}

/// What a sweep may choose to evacuate before the next major, in live bytes.
/// An eighth of what old space holds now bounds the copying the next major
/// does to a fraction of what its own trace already touched.
static size_t evacuate_budget_for(size_t old_bytes) { return old_bytes / 8; }

/// The kinds that make up a live set, largest first, as ` map 55% list 16%`.
///
/// A trace line that says only how many bytes survived says nothing about
/// what they are, and "what are they?" is the only question a live set of
/// hundreds of megabytes actually poses. Three kinds is enough to tell the
/// shapes apart -- a compile holding its syntax from one holding its tables --
/// and short enough to leave the timings on the same line readable.
std::string Heap::live_kinds_line(const uint64_t* by_type, size_t live) {
    if (!live) return "";
    std::vector<std::pair<uint64_t, size_t>> rows;
    for (size_t i = 0; i < 32; ++i)
        if (by_type[i]) rows.push_back({by_type[i], i});
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::string out;
    for (size_t i = 0; i < rows.size() && i < 3; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof buf, " %s %.0f%%", obj_type_name(ObjType(rows[i].second)),
                      100.0 * double(rows[i].first) / double(live));
        out += buf;
    }
    return out;
}

/// Keep this sweep's tally if it found more live than any before it.
///
/// A live set is only worth reporting with the moment it belongs to, and the
/// largest one is the moment that decides whether a program fits in a heap.
/// Keeping the largest rather than the last is what makes the number answer
/// "what is this compile held back by?" instead of "what was left at the end?"
void Heap::note_live_by_type(size_t live, const uint64_t* by_type) {
    if (live <= peak_major_live_ && peak_major_live_) return;
    peak_major_live_ = live;
    for (size_t i = 0; i < 64; ++i) peak_by_type_[i] = by_type[i];
}

void Heap::sweep() {
    ++sweeps_;
    std::vector<Block*> gone = detach_evacuated();
    evacuate_budget_ = evacuate_budget_for(allocated_);
    evacuate_chosen_.store(0, std::memory_order_relaxed);
    Obj* head[kHeapClassCount] = {};
    Obj* tail[kHeapClassCount] = {};
    size_t live = 0;
    uint64_t by_type[64] = {};
    Block* prev = nullptr;
    for (Block* b = blocks_; b;) {
        Block* next = b->next;
        sweep_block(b, head, tail, live, by_type);
        if (b->dead) {
            if (prev) prev->next = next;
            else blocks_ = next;
            if (carve_block_ == b) carve_block_ = nullptr;
            if (b->big) retire_big(b);
            else free_block(b);
        } else {
            if (b->evacuate) ++evacuate_pending_;
            prev = b;
        }
        b = next;
    }
    // The chunks the sweep found are every free chunk there is when the lists
    // are rebuilt, so the old lists go first.
    if (evacuate_below_percent() != 0) free_lists_.fill(nullptr);
    merge_free_lists(head, tail);
    free_evacuated(gone);
    Block* cb = blocks_;
    while (cb && (cb->big || cb->evacuate)) cb = cb->next;
    carve_block_ = cb;
    live_after_gc_ = live;
    if (live > peak_live_) peak_live_ = live;
    note_live_by_type(live, by_type);
    last_kinds_ = gc_trace() ? live_kinds_line(by_type, live) : std::string();
    allocated_ = live;
    trim_big_pool();
}

bool Heap::sweep_in_parallel() {
    if (allocated_ < parallel_floor(/*minor=*/false)) return false;
    GcPool& pool = GcPool::instance();
    if (pool.capacity() < 2) return false;

    // A sweep divides itself: the blocks are independent, a thread that
    // finishes one takes the next, and nothing needs a lock until the lists
    // are put back together. The one thing a thread may not do is touch the
    // block list, which is why a dead big block is flagged and unlinked
    // afterwards rather than where it is found.
    GcRound round;
    round.ctxs.resize(pool.capacity());
    std::vector<Block*> gone = detach_evacuated();
    evacuate_budget_ = evacuate_budget_for(allocated_);
    evacuate_chosen_.store(0, std::memory_order_relaxed);
    for (Block* b = blocks_; b; b = b->next) round.blocks.push_back(b);
    round_ = &round;

    ++sweeps_;
    bool ran = pool.run(pool.capacity(), [&](unsigned index, unsigned) {
        GcCtx& c = round.ctxs[index];
        for (;;) {
            size_t i = round.next_block.fetch_add(1, std::memory_order_relaxed);
            if (i >= round.blocks.size()) break;
            sweep_block(round.blocks[i], c.free_head, c.free_tail, c.live, c.by_type);
        }
    });
    round_ = nullptr;
    if (!ran) {
        // Nothing was swept, so nothing was chosen and the stubs are still
        // owed their rewrite: put the detached blocks back for the serial
        // sweep the caller falls back on.
        for (Block* b : gone) {
            b->next = blocks_;
            blocks_ = b;
            ++evacuate_pending_;
        }
        --sweeps_;
        return false;
    }

    size_t live = 0;
    uint64_t by_type[64] = {};
    if (evacuate_below_percent() != 0) free_lists_.fill(nullptr);
    for (GcCtx& c : round.ctxs) {
        merge_free_lists(c.free_head, c.free_tail);
        live += c.live;
        for (size_t i = 0; i < 64; ++i) by_type[i] += c.by_type[i];
    }

    Block* head = nullptr;
    Block* tail = nullptr;
    for (Block* b : round.blocks) {
        if (b->dead) {
            if (b->big) retire_big(b);
            else free_block(b);
            continue;
        }
        if (b->evacuate) ++evacuate_pending_;
        b->next = nullptr;
        if (tail) tail->next = b;
        else head = b;
        tail = b;
    }
    blocks_ = head;
    free_evacuated(gone);
    Block* cb = blocks_;
    while (cb && (cb->big || cb->evacuate)) cb = cb->next;
    carve_block_ = cb;
    live_after_gc_ = live;
    if (live > peak_live_) peak_live_ = live;
    note_live_by_type(live, by_type);
    last_kinds_ = gc_trace() ? live_kinds_line(by_type, live) : std::string();
    allocated_ = live;
    trim_big_pool();
    ++parallel_collections_;
    return true;
}

void Heap::verify_collect(RootSource& roots, bool expect_no_young) {
    if (!verify_after_gc()) return;
    std::string problem = verify_internal(roots, expect_no_young);
    if (!problem.empty()) {
        std::fprintf(stderr, "dream: heap corrupted after collection %llu: %s\n",
                     static_cast<unsigned long long>(collections_), problem.c_str());
        std::abort();
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

bool Heap::owns(const void* p, size_t bytes) const {
    auto addr = reinterpret_cast<uintptr_t>(p);
    for (Block* b = blocks_; b; b = b->next) {
        auto start = reinterpret_cast<uintptr_t>(b->data);
        // Only the used part counts: an object cannot legitimately live in the
        // unallocated tail of a block.
        if (addr >= start && addr + bytes <= start + b->used) return true;
    }
    for (Block* b : nursery_) {
        auto start = reinterpret_cast<uintptr_t>(b->data);
        if (addr >= start && addr + bytes <= start + b->used) return true;
    }
    return false;
}

bool Heap::verify_after_gc() {
    // Read once and answered from a bool thereafter, so that the collector
    // pays one predictable branch for it. It used to be compiled out unless
    // the build defined `DREAM_DEBUG` or `DREAM_VERIFY_HEAP` -- which no build
    // here ever did, so the environment variable did nothing and `just
    // test-heap` verified nothing. A branch per collection is not a cost worth
    // a switch that is off by accident.
    static const bool on = [] {
        const char* v = std::getenv("DREAM_VERIFY_HEAP");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

namespace {

struct VerifyWalk {
    const Heap& heap;
    bool no_young;
    std::string error;
    std::vector<Value> work;
    std::vector<Value> parents;
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;

    bool owns(const void* p, size_t bytes) const {
        const auto addr = reinterpret_cast<uintptr_t>(p);
        auto it = std::upper_bound(ranges.begin(), ranges.end(), addr,
            [](uintptr_t key, const auto& range) { return key < range.first; });
        if (it == ranges.begin()) return false;
        --it;
        return addr >= it->first && addr <= it->second && bytes <= it->second - addr;
    }


    bool problem(const std::string& what) {
        if (error.empty()) error = what;
        return false;
    }

    static std::string addr(const void* p) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%p", p);
        return buf;
    }

    void push(Value v, Value parent) {
        if (is_ptr(v)) {
            work.push_back(v);
            parents.push_back(parent);
        }
    }

    bool check_object(Value v, Value parent) {
        Obj* o = as_obj(v);
        if (reinterpret_cast<uintptr_t>(o) % 8 != 0) {
            return problem("object at " + addr(o) + " is not 8-byte aligned");
        }
        if (!owns(o, sizeof(Obj))) {
            std::string via = "a root";
            if (is_ptr(parent)) {
                Obj* pp = as_obj(parent);
                via = addr(pp) + " type " + std::to_string(static_cast<int>(pp->type)) +
                      " gc " + std::to_string(static_cast<int>(pp->gc));
            }
            return problem("pointer " + addr(o) + " does not land in this heap (reached from " +
                           via + ")");
        }
        if (no_young && (o->gc & GC_YOUNG) && !(o->gc & GC_PINNED)) {
            std::string via = "root";
            if (is_ptr(parent)) {
                Obj* pp = as_obj(parent);
                via = addr(pp) + " type " + std::to_string(static_cast<int>(pp->type)) +
                      " gc " + std::to_string(static_cast<int>(pp->gc));
            }
            return problem("a minor collection left a young object reachable at " +
                           addr(o) + " type " + std::to_string(static_cast<int>(o->type)) +
                           " (reached from " + via + ")");
        }
        if (o->bytes < sizeof(Obj) || o->bytes > (1u << 30)) {
            return problem("object at " + addr(o) + " claims an implausible size " +
                           std::to_string(o->bytes));
        }
        if (!owns(o, o->bytes)) {
            return problem("object at " + addr(o) + " of " + std::to_string(o->bytes) +
                           " bytes runs past the end of its block");
        }
        if (static_cast<uint8_t>(o->type) >= static_cast<uint8_t>(ObjType::Count)) {
            return problem("object at " + addr(o) + " has unknown type " +
                           std::to_string(static_cast<int>(o->type)));
        }
        return true;
    }

    void run() {
        // `seen` needs O(1) membership: the walk meets every live object once,
        // and a vector + linear find would make verification itself
        // quadratic -- survivable when a collection happens a handful of times,
        // not when a generational collector verifies after every minor.
        std::unordered_set<const Obj*> seen;
        while (!work.empty() && error.empty()) {
            Value v = work.back();
            work.pop_back();
            Value parent = parents.empty() ? 0 : parents.back();
            parents.pop_back();
            if (!is_ptr(v)) continue;
            Obj* o = as_obj(v);
            if (!seen.insert(o).second) continue;
            // A shared object is no heap's, so it is not this one's to vouch
            // for; the area it lives in is checked by nothing but its own
            // construction, which is one copy under one lock.
            if (!owns(o, sizeof(Obj)) && SharedArea::any_contains(o)) continue;
            if (!check_object(v, parent)) return;

            switch (o->type) {
                case ObjType::Cons: {
                    auto* c = static_cast<ConsObj*>(o);
                    push(c->head, v);
                    push(c->tail, v);
                    break;
                }
                case ObjType::Array: {
                    auto* a = static_cast<ArrayObj*>(o);
                    size_t need = sizeof(Obj) + 8 + size_t(a->len) * sizeof(Value);
                    if (a->bytes < need) {
                        problem("array at " + addr(o) + " says it holds " +
                                std::to_string(a->len) + " items but is too small");
                        return;
                    }
                    for (uint32_t i = 0; i < a->len; ++i) push(a->items()[i], v);
                    break;
                }
                case ObjType::Record: {
                    auto* r = static_cast<RecordObj*>(o);
                    size_t need = sizeof(Obj) + 8 + size_t(r->len) * sizeof(Value);
                    if (r->bytes < need) {
                        problem("record at " + addr(o) + " says it holds " +
                                std::to_string(r->len) + " fields but is too small");
                        return;
                    }
                    if (!is_atom(r->id())) {
                        problem("record at " + addr(o) + " has an id that is not an atom");
                        return;
                    }
                    for (uint32_t i = 0; i < r->len; ++i) push(r->items()[i], v);
                    break;
                }
                case ObjType::MapLeaf: {
                    auto* l = static_cast<MapLeafObj*>(o);
                    if (l->bytes < sizeof(Obj) + 8 + 3 * sizeof(Value)) {
                        problem("map entry at " + addr(o) + " is too small");
                        return;
                    }
                    push(l->key, v);
                    push(l->value, v);
                    push(l->next, v);
                    break;
                }
                case ObjType::Map: {
                    auto* m = static_cast<MapObj*>(o);
                    uint32_t slots = map_bit_count(m->bitmap);
                    size_t need = sizeof(Obj) + 8 + size_t(slots) * sizeof(Value);
                    if (m->bytes < need) {
                        problem("map at " + addr(o) + " holds " +
                                std::to_string(slots) + " children but is too small");
                        return;
                    }
                    // A branch with children must hold at least as many
                    // entries as it has children, since every child holds one.
                    if (m->count < slots) {
                        problem("map at " + addr(o) + " counts fewer entries than it has children");
                        return;
                    }
                    for (uint32_t i = 0; i < slots; ++i) push(m->slots()[i], v);
                    break;
                }
                case ObjType::Closure: {
                    auto* c = static_cast<ClosureObj*>(o);
                    if (c->bytes < sizeof(Obj) + 8 + size_t(c->ncaps) * sizeof(Value)) {
                        problem("closure at " + addr(o) + " is too small for its captures");
                        return;
                    }
                    for (uint32_t i = 0; i < c->ncaps; ++i) push(c->caps()[i], v);
                    break;
                }
                case ObjType::Frame: {
                    auto* f = static_cast<FrameObj*>(o);
                    if (f->bytes < sizeof(Obj) + sizeof(Value) + 8 +
                                       size_t(f->nslots) * sizeof(Value)) {
                        problem("frame at " + addr(o) + " is too small for its slots");
                        return;
                    }
                    push(f->closure, v);
                    for (uint32_t i = 0; i < f->nslots; ++i) {
                        // An unbound slot is legitimate: a `let` later in the
                        // block has not run yet.
                        if (f->slots()[i] != NIL_SLOT) push(f->slots()[i], v);
                    }
                    break;
                }
                case ObjType::Pap: {
                    auto* p = static_cast<PapObj*>(o);
                    push(p->fn, v);
                    for (uint32_t i = 0; i < p->nargs; ++i) push(p->args()[i], v);
                    break;
                }
                case ObjType::Thunk:
                case ObjType::Blackhole:
                    push(static_cast<ThunkObj*>(o)->frame, v);
                    break;
                case ObjType::Indirect:
                    push(static_cast<IndirectObj*>(o)->target, v);
                    break;
                case ObjType::ErrorBox: {
                    auto* e = static_cast<ErrorObj*>(o);
                    push(e->kind, v);
                    push(e->payload, v);
                    push(e->where, v);
                    break;
                }
                case ObjType::Module:
                    push(static_cast<ModuleObj*>(o)->name, v);
                    break;
                case ObjType::Native:
                    push(static_cast<NativeObj*>(o)->name, v);
                    break;
                case ObjType::Str: {
                    auto* s = static_cast<StrObj*>(o);
                    if (s->bytes < sizeof(Obj) + 8 + size_t(s->len) + 1) {
                        problem("string at " + addr(o) + " is shorter than its length claims");
                        return;
                    }
                    break;
                }
                case ObjType::BigInt: {
                    // Canonical as well as in bounds: a top limb of zero is a
                    // bignum some operation forgot to trim, and equality would
                    // then call it different from the same number trimmed.
                    auto* b = static_cast<BigIntObj*>(o);
                    const size_t room = b->external ? size_t(b->external_capacity()) * 8
                                                    : b->bytes - sizeof(BigIntObj);
                    if (b->len == 0 || room < size_t(b->len) * 8) {
                        problem("bignum at " + addr(o) + " is shorter than its length claims");
                        return;
                    }
                    if (b->limbs()[b->len - 1] == 0) {
                        problem("bignum at " + addr(o) + " is not trimmed");
                        return;
                    }
                    break;
                }
                case ObjType::Tensor: {
                    auto* t = static_cast<TensorObj*>(o);
                    if (tensor_deferred(t)) {
                        TensorExpr* e = tensor_expr(t);
                        if (t->bytes < sizeof(TensorObj) +
                                           TensorExpr::bytes_for(e->ninputs, e->nconsts, e->ncode)) {
                            problem("deferred tensor at " + addr(o) +
                                    " is smaller than its program");
                            return;
                        }
                        push(e->result, v);
                        push(e->product_a, v);
                        push(e->product_b, v);
                        for (uint32_t i = 0; i < e->ninputs; ++i) push(e->inputs()[i], v);
                        break;
                    }
                    uint64_t count = 1;
                    for (uint32_t i = 0; i < t->rank; ++i) count *= t->dims[i];
                    const size_t payload = t->device == TENSOR_HOST
                                               ? t->count * sizeof(double) : sizeof(void*);
                    if (t->rank == 0 || t->rank > TENSOR_MAX_RANK || count != t->count ||
                        t->bytes < sizeof(TensorObj) + payload) {
                        problem("tensor at " + addr(o) + " has a shape its size does not hold");
                        return;
                    }
                    break;
                }
                case ObjType::BigStr: {
                    // Its bytes are not in the heap, so there is nothing here
                    // to measure against its length -- only that it points
                    // somewhere at all. A null view would be read as an empty
                    // string forever rather than failing where it was made.
                    if (static_cast<BigStrObj*>(o)->data == nullptr) {
                        problem("big string at " + addr(o) + " points nowhere");
                        return;
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
};

}  // namespace

std::string Heap::verify(RootSource& roots) {
    return verify_internal(roots, false);
}

std::string Heap::verify_internal(RootSource& roots, bool no_young) {
    // Gather the roots without moving anything. `forward` is the only way a
    // root source hands over its references, so it doubles as a recorder while
    // `recording_` is set.
    std::vector<Value> collected;
    recording_ = &collected;
    roots.visit_roots(*this);
    recording_ = nullptr;

    VerifyWalk walk{*this, no_young, {}, {}, {}, {}};
    // Ownership checks used to walk every block for every object, making a
    // verified self-compile quadratic in heap size. Index the used ranges
    // once per verification, without changing the allocator's hot path.
    auto add_range = [&](Block* b) {
        const auto start = reinterpret_cast<uintptr_t>(b->data);
        walk.ranges.emplace_back(start, start + b->used);
    };
    for (Block* b = blocks_; b; b = b->next) add_range(b);
    for (Block* b : nursery_) add_range(b);
    std::sort(walk.ranges.begin(), walk.ranges.end());
    for (Value v : collected) walk.push(v, 0);
    // What a collection under compiled code pinned is as much a root as the
    // process's own, and has to have come through it as sound.
    for (Obj* o : pinned_) walk.push(from_obj(o), 0);
    walk.run();
    return walk.error;
}

// ---------------------------------------------------------------------------
// Cross-heap copying
// ---------------------------------------------------------------------------

namespace {

/// What has been copied already, so that sharing survives the copy and a cycle
/// ends. A hash map: it was a vector searched from the front for every object,
/// which made a copy quadratic in the size of what crossed -- nothing for a
/// message, and minutes for a syntax tree.
///
/// Open addressing rather than `std::unordered_map`, whose every insert is a
/// heap allocation: a copy inserts one entry per object that crossed, and for
/// a compiler handing an arena to another process that is a million of them.
class CopySeen {
public:
    CopySeen() : slots_(64) {}
    struct Found {
        Value second;
    };
    const Found* find(Value key) const {
        size_t mask = slots_.size() - 1;
        for (size_t i = hash(key) & mask;; i = (i + 1) & mask) {
            const Slot& s = slots_[i];
            if (s.key == 0) return nullptr;
            if (s.key == key) return reinterpret_cast<const Found*>(&s.value);
        }
    }
    const Found* end() const { return nullptr; }
    void emplace(Value key, Value value) {
        if ((used_ + 1) * 2 > slots_.size()) grow();
        put(key, value);
    }

private:
    struct Slot {
        Value key = 0;  // a pointer value is never 0, so 0 is "empty"
        Value value = 0;
    };
    static size_t hash(Value v) { return size_t((v >> 3) * 0x9E3779B97F4A7C15ull >> 16); }
    void put(Value key, Value value) {
        size_t mask = slots_.size() - 1;
        for (size_t i = hash(key) & mask;; i = (i + 1) & mask) {
            Slot& s = slots_[i];
            if (s.key == 0) { s.key = key; s.value = value; ++used_; return; }
            if (s.key == key) { s.value = value; return; }
        }
    }
    void grow() {
        std::vector<Slot> old(slots_.size() * 2);
        old.swap(slots_);
        used_ = 0;
        for (const Slot& s : old) if (s.key) put(s.key, s.value);
    }
    std::vector<Slot> slots_;
    size_t used_ = 0;
};

/// Deep-copy with cycle handling. Values crossing a heap boundary are copied
/// eagerly, which is why forcing must happen before a send: a thunk carries a
/// frame that points back into the sender's world.
///
/// Written once for both destinations: another process's heap, and the
/// runtime's shared area. The only differences are what a shared object is to
/// each -- to either, something already where it needs to be, so it crosses by
/// pointer -- and that the shared area refuses a suspension where a heap takes
/// one with its frame.
Value copy_suspension(Heap& dest, Value v, CopySeen& seen);

template <class Dest>
Value copy_object(Dest& dest, Value v, CopySeen& seen);

/// A copy keeps the identity hash of what it copies (see `AUX_IDENTITY`): a
/// map crosses with the hashes its leaves were filed under, so a key that
/// crossed with it has to answer the same one. `copy_object` makes the copy;
/// this is the entry every recursion goes through, so every object it makes is
/// stamped -- except the cells of a list's spine, which the loop there makes
/// itself and stamps as it goes.
inline void carry_identity(Value from, Value to) {
    if (!is_ptr(to)) return;
    if (uint16_t id = as_obj(from)->aux & AUX_IDENTITY) as_obj(to)->aux |= id;
}

template <class Dest>
Value copy_value(Dest& dest, Value v, CopySeen& seen) {
    if (!is_ptr(v)) return v;
    v = resolve(v);
    if (!is_ptr(v)) return v;
    if (is_shared_obj(as_obj(v))) {
        // Crossing by pointer, which is the point of sharing -- and the one
        // moment a census has to hear about. See `note_shared_crossing`.
        dest.met_shared(as_obj(v));
        return v;
    }

    if (auto* it = seen.find(v)) return it->second;

    Value out = copy_object(dest, v, seen);
    carry_identity(v, out);
    return out;
}

template <class Dest>
Value copy_object(Dest& dest, Value v, CopySeen& seen) {
    constexpr bool to_shared = std::is_same_v<Dest, SharedArea>;
    Obj* o = as_obj(v);
    switch (o->type) {
        case ObjType::Float:
            return dest.make_float(static_cast<FloatObj*>(o)->value);
        case ObjType::Str: {
            auto* s = static_cast<StrObj*>(o);
            return dest.make_string(s->data(), s->len);
        }
        case ObjType::BigStr: {
            // The view crosses, not the bytes. They live in the image, which
            // every heap of this runtime shares and none of them owns.
            auto* b = static_cast<BigStrObj*>(o);
            return dest.make_bigstr(b->data, b->len);
        }
        case ObjType::Pid:
            return dest.make_pid(static_cast<PidObj*>(o)->id);
        case ObjType::BigInt: {
            auto* b = static_cast<BigIntObj*>(o);
            return dest.make_bigint(b->limbs(), b->len, b->neg != 0);
        }
        case ObjType::Tensor: {
            auto* src = static_cast<TensorObj*>(o);
            if (tensor_deferred(src)) {
                // Computed already: what crosses is the answer.
                TensorExpr* e = tensor_expr(src);
                if (e->result != NIL_SLOT) return copy_value(dest, e->result, seen);
                // Not yet: the program crosses with copies of its inputs, and
                // the receiver computes it if it ever reads it. The shared
                // area has nobody to do that, so it is asked for the answer.
                if constexpr (to_shared) {
                    dest.refuse("unevaluated tensor (`strict!` computes it)");
                    return UNIT;
                } else {
                    Value t = dest.make_deferred_tensor(src->rank, src->dims, src->count,
                                                        src->dtype, src->device, e->ninputs,
                                                        e->nconsts, e->ncode);
                    seen.emplace(v, t);
                    auto* to = static_cast<TensorObj*>(as_obj(t));
                    TensorExpr* d = tensor_expr(to);
                    d->depth = e->depth;
                    std::memcpy(d->consts(), e->consts(),
                                e->nconsts * sizeof(double) + e->ncode * 2);
                    for (uint32_t i = 0; i < e->ninputs; ++i) {
                        Value in = copy_value(dest, e->inputs()[i], seen);
                        tensor_expr(static_cast<TensorObj*>(as_obj(t)))->inputs()[i] = in;
                    }
                    if (e->product_a != NIL_SLOT) {
                        Value pa = copy_value(dest, e->product_a, seen);
                        Value pb = copy_value(dest, e->product_b, seen);
                        tensor_expr(static_cast<TensorObj*>(as_obj(t)))->product_a = pa;
                        tensor_expr(static_cast<TensorObj*>(as_obj(t)))->product_b = pb;
                    }
                    return t;
                }
            }
            if (src->device != TENSOR_HOST) {
                // A GPU tensor crosses as its handle: both processes then hold
                // a reference to the one device buffer, which is immutable like
                // every other value. The shared area has no collector to
                // release a reference, so it cannot hold one.
                if constexpr (to_shared) {
                    dest.refuse("GPU tensor");
                    return UNIT;
                } else {
                    retain_external(src);
                    return dest.make_external_tensor(src->rank, src->dims, src->count,
                                                     src->dtype, src->device,
                                                     tensor_handle(src));
                }
            }
            // The numbers cross by copy, as a string's bytes do. A tensor
            // sent to another process is one memcpy, not one box per element.
            Value t = dest.make_tensor(src->rank, src->dims, src->count);
            std::memcpy(static_cast<TensorObj*>(as_obj(t))->data(), src->data(),
                        size_t(src->count) * sizeof(double));
            return t;
        }
        case ObjType::Cons: {
            // The spine is walked in a loop rather than by recursing on the
            // tail, so a long list costs no C++ stack: a hundred thousand
            // cells was a hundred thousand frames on a worker's 8 MB.
            Value first = dest.make_cons(UNIT, NIL);
            seen.emplace(v, first);
            Value cell = first;
            Value cur = v;
            for (;;) {
                auto* src = static_cast<ConsObj*>(as_obj(cur));
                Value head = copy_value(dest, src->head, seen);
                static_cast<ConsObj*>(as_obj(cell))->head = head;
                Value next = resolve(src->tail);
                if (is_ptr(next) && as_obj(next)->type == ObjType::Cons && !seen.find(next)) {
                    Value fresh = dest.make_cons(UNIT, NIL);
                    carry_identity(next, fresh);
                    seen.emplace(next, fresh);
                    static_cast<ConsObj*>(as_obj(cell))->tail = fresh;
                    cell = fresh;
                    cur = next;
                    continue;
                }
                Value tail = copy_value(dest, next, seen);
                static_cast<ConsObj*>(as_obj(cell))->tail = tail;
                return first;
            }
        }
        case ObjType::Array: {
            auto* src = static_cast<ArrayObj*>(o);
            Value arr = dest.make_array(src->len);
            seen.emplace(v, arr);
            for (uint32_t i = 0; i < src->len; ++i) {
                Value item = copy_value(dest, src->items()[i], seen);
                static_cast<ArrayObj*>(as_obj(arr))->items()[i] = item;
            }
            return arr;
        }
        case ObjType::Record: {
            // The id is an atom of this runtime, as the copy is.
            auto* src = static_cast<RecordObj*>(o);
            Value rec = dest.make_record(src->tag, src->len);
            seen.emplace(v, rec);
            for (uint32_t i = 0; i < src->len; ++i) {
                Value item = copy_value(dest, static_cast<RecordObj*>(o)->items()[i], seen);
                static_cast<RecordObj*>(as_obj(rec))->items()[i] = item;
            }
            return rec;
        }
        case ObjType::Map: {
            // Copied branch for branch, so the shape -- and with it the
            // lookups -- comes out identical on the other side.
            auto* src = static_cast<MapObj*>(o);
            uint32_t slots = map_bit_count(src->bitmap);
            Value map = dest.make_map_branch(slots);
            seen.emplace(v, map);
            auto* out = static_cast<MapObj*>(as_obj(map));
            out->count = src->count;
            out->bitmap = src->bitmap;
            for (uint32_t i = 0; i < slots; ++i) {
                Value item = copy_value(dest, src->slots()[i], seen);
                static_cast<MapObj*>(as_obj(map))->slots()[i] = item;
            }
            return map;
        }
        case ObjType::MapLeaf: {
            auto* src = static_cast<MapLeafObj*>(o);
            Value leaf = dest.make_map_leaf(src->hash, NIL_SLOT, NIL_SLOT, NIL_SLOT);
            seen.emplace(v, leaf);
            Value key = copy_value(dest, src->key, seen);
            Value val = copy_value(dest, src->value, seen);
            Value next = copy_value(dest, src->next, seen);
            auto* out = static_cast<MapLeafObj*>(as_obj(leaf));
            out->key = key;
            out->value = val;
            out->next = next;
            return leaf;
        }
        case ObjType::ErrorBox: {
            auto* src = static_cast<ErrorObj*>(o);
            Value kind = copy_value(dest, src->kind, seen);
            Value payload = copy_value(dest, src->payload, seen);
            Value where = copy_value(dest, src->where, seen);
            return dest.make_error(kind, payload, where);
        }
        case ObjType::Module: {
            auto* src = static_cast<ModuleObj*>(o);
            Value name = copy_value(dest, src->name, seen);
            return dest.make_module(src->import_index, name);
        }
        case ObjType::Closure: {
            auto* src = static_cast<ClosureObj*>(o);
            Value cl = dest.make_closure(src->func, src->ncaps);
            seen.emplace(v, cl);
            for (uint32_t i = 0; i < src->ncaps; ++i) {
                Value cap = copy_value(dest, src->caps()[i], seen);
                static_cast<ClosureObj*>(as_obj(cl))->caps()[i] = cap;
            }
            return cl;
        }
        case ObjType::Native: {
            // A host function is a pointer into the VM and whatever id it was
            // bound with -- both good for every process of the runtime -- so it
            // travels as itself. A foreign function sent to a library's server
            // process is the case that needs it; it used to arrive as `()`.
            if constexpr (to_shared) {
                dest.refuse(obj_type_name(o->type));
                return UNIT;
            } else {
                auto* src = static_cast<NativeObj*>(o);
                Value name = copy_value(dest, src->name, seen);
                auto* nat = static_cast<NativeObj*>(dest.alloc(
                    ObjType::Native, sizeof(void*) + sizeof(Value) + 8 + sizeof(uint64_t)));
                nat->fn = src->fn;
                nat->name = name;
                nat->arity = src->arity;
                nat->strict_mask = src->strict_mask;
                nat->user = src->user;
                Value out = from_obj(nat);
                seen.emplace(v, out);
                return out;
            }
        }
        case ObjType::Pap: {
            auto* src = static_cast<PapObj*>(o);
            Value fn = copy_value(dest, src->fn, seen);
            Value pap = dest.make_pap(fn, src->nargs);
            seen.emplace(v, pap);
            for (uint32_t i = 0; i < src->nargs; ++i) {
                Value arg = copy_value(dest, src->args()[i], seen);
                static_cast<PapObj*>(as_obj(pap))->args()[i] = arg;
            }
            return pap;
        }
        case ObjType::Frame:
        case ObjType::Thunk:
        case ObjType::Blackhole:
            if constexpr (to_shared) {
                dest.refuse(obj_type_name(o->type));
                return UNIT;
            } else {
                return copy_suspension(dest, v, seen);
            }
        default:
            return UNIT;
    }
}

/// The half of `copy_value` only a heap can take: a frame or a thunk, which
/// travel between processes with everything they would need to run.
Value copy_suspension(Heap& dest, Value v, CopySeen& seen) {
    Obj* o = as_obj(v);
    switch (o->type) {
        case ObjType::Frame: {
            auto* src = static_cast<FrameObj*>(o);
            Value fr = dest.make_frame(UNIT, src->nslots);
            seen.emplace(v, fr);
            Value closure = copy_value(dest, src->closure, seen);
            static_cast<FrameObj*>(as_obj(fr))->closure = closure;
            for (uint32_t i = 0; i < src->nslots; ++i) {
                Value slot = src->slots()[i];
                Value copied = slot == NIL_SLOT ? NIL_SLOT : copy_value(dest, slot, seen);
                static_cast<FrameObj*>(as_obj(fr))->slots()[i] = copied;
            }
            return fr;
        }
        case ObjType::Thunk: {
            // Thunks travel with their frame, so a spawned computation carries
            // everything it needs and evaluates independently in its new home.
            auto* src = static_cast<ThunkObj*>(o);
            Value th = dest.make_thunk(src->node, UNIT);
            seen.emplace(v, th);
            Value frame = copy_value(dest, src->frame, seen);
            static_cast<ThunkObj*>(as_obj(th))->frame = frame;
            return th;
        }
        case ObjType::Blackhole:
            // A thunk under evaluation belongs to the sender's stack; the
            // receiver cannot finish it, so hand over an error instead.
            return dest.make_error(make_atom(0), UNIT);
        default:
            return UNIT;
    }
}

}  // namespace

Value Heap::copy_between(Heap& dest, Value v) {
    CopySeen seen;
    return copy_value(dest, v, seen);
}

// ---------------------------------------------------------------------------
// The shared area
// ---------------------------------------------------------------------------

std::atomic<int> g_condemned_regions{0};

namespace {
/// A region's first block. Each next one doubles, up to `kSharedBlockMax`, so
/// a small share wastes little and a large one takes few blocks.
constexpr size_t kSharedBlockMin = size_t(64) << 10;
constexpr size_t kSharedBlockMax = size_t(1) << 20;

/// Every block of every live shared area, sorted by address, with the region
/// it belongs to: `[start, end, region]`.
///
/// Two readers. The heap verifier asks `any_contains` of pointers it must not
/// dereference until it knows whose they are. A census asks `region_of` of
/// every shared object a major meets and every one a copy carries while a
/// region is condemned -- often, so a reader takes the lock shared and keeps
/// the last block it found in a thread-local cache, which is valid while the
/// table's generation has not moved. Nothing reads it outside a census or a
/// verification.
struct SharedRange {
    uintptr_t lo;
    uintptr_t hi;
    SharedRegion* region;
};
std::shared_mutex g_shared_ranges_mutex;
std::vector<SharedRange> g_shared_ranges;
std::atomic<uint64_t> g_shared_generation{1};

void note_shared_block(const uint8_t* at, size_t bytes, SharedRegion* r) {
    std::unique_lock<std::shared_mutex> g(g_shared_ranges_mutex);
    auto start = reinterpret_cast<uintptr_t>(at);
    SharedRange range{start, start + bytes, r};
    auto it = std::lower_bound(g_shared_ranges.begin(), g_shared_ranges.end(), start,
                               [](const SharedRange& x, uintptr_t v) { return x.lo < v; });
    g_shared_ranges.insert(it, range);
    g_shared_generation.fetch_add(1, std::memory_order_release);
}

void forget_shared_block(const uint8_t* at) {
    std::unique_lock<std::shared_mutex> g(g_shared_ranges_mutex);
    auto start = reinterpret_cast<uintptr_t>(at);
    auto it = std::lower_bound(g_shared_ranges.begin(), g_shared_ranges.end(), start,
                               [](const SharedRange& x, uintptr_t v) { return x.lo < v; });
    if (it != g_shared_ranges.end() && it->lo == start) g_shared_ranges.erase(it);
    g_shared_generation.fetch_add(1, std::memory_order_release);
}

/// The range holding `at`, or one with a null region. The caller holds the
/// lock shared.
SharedRange find_range(uintptr_t at) {
    auto it = std::upper_bound(g_shared_ranges.begin(), g_shared_ranges.end(), at,
                               [](uintptr_t v, const SharedRange& x) { return v < x.lo; });
    if (it == g_shared_ranges.begin()) return {0, 0, nullptr};
    --it;
    if (at >= it->lo && at < it->hi) return *it;
    return {0, 0, nullptr};
}
}  // namespace

SharedRegion* SharedArea::region_of(const Obj* o) {
    thread_local SharedRange last{0, 0, nullptr};
    thread_local uint64_t last_generation = 0;
    auto at = reinterpret_cast<uintptr_t>(o);
    uint64_t gen = g_shared_generation.load(std::memory_order_acquire);
    if (gen == last_generation && at >= last.lo && at < last.hi) return last.region;
    std::shared_lock<std::shared_mutex> g(g_shared_ranges_mutex);
    SharedRange r = find_range(at);
    last = r;
    last_generation = g_shared_generation.load(std::memory_order_relaxed);
    return r.region;
}

bool SharedArea::any_contains(const void* p) {
    auto at = reinterpret_cast<uintptr_t>(p);
    std::shared_lock<std::shared_mutex> g(g_shared_ranges_mutex);
    SharedRange r = find_range(at);
    return r.region && at + sizeof(Obj) <= r.hi;
}

void shared_crossing_slow(const Obj* o) {
    SharedRegion* r = SharedArea::region_of(o);
    if (r && r->condemned.load(std::memory_order_relaxed)) r->area->spoil();
}

void shared_observe(const Obj* o, uint64_t round) {
    SharedRegion* r = SharedArea::region_of(o);
    if (r && r->condemned.load(std::memory_order_relaxed)) r->seen_round.store(round, std::memory_order_relaxed);
}

void shared_census_walk(Value v, uint64_t round) {
    // Read-only: no mark bit is set, so this may run over a heap another
    // thread is reading -- a finished process's, which a `join!` copies from
    // -- and over a message's, which nobody collects.
    std::unordered_set<Obj*> visited;
    std::vector<Obj*> todo;
    auto push = [&](Value w) {
        if (is_ptr(w)) todo.push_back(as_obj(w));
    };
    push(v);
    while (!todo.empty()) {
        Obj* o = todo.back();
        todo.pop_back();
        if (is_shared_obj(o)) {
            shared_observe(o, round);
            continue;
        }
        if (!visited.insert(o).second) continue;
        for_each_slot(o, o->type, [&](Value* slot) { push(*slot); });
    }
}

void Heap::census_collect(RootSource& roots, uint64_t round) {
    if (marking_) finalize_concurrent_mark(roots);
    observe_round_ = round;
    major_collect(roots);
    observe_round_ = 0;
}

SharedArea::~SharedArea() {
    for (auto& r : regions_) {
        for (uint8_t* b : r->blocks) {
            forget_shared_block(b);
            std::free(b);
        }
        if (r->condemned.load(std::memory_order_relaxed)) g_condemned_regions.fetch_sub(1);
    }
}

Obj* SharedArea::alloc(ObjType type, size_t extra) {
    SharedRegion* r = building_;
    size_t bytes = align_up(sizeof(Obj) + extra);
    uint8_t* at;
    if (bytes > kSharedBlockMax / 4) {
        // A large object gets a block of its own, so the bump block it would
        // have ended is not thrown away half used.
        at = static_cast<uint8_t*>(std::malloc(bytes));
        if (!at) throw std::bad_alloc();
        r->blocks.push_back(at);
        note_shared_block(at, bytes, r);
    } else {
        if (bytes > r->left) {
            size_t size = r->next_block ? r->next_block : kSharedBlockMin;
            while (size < bytes) size *= 2;
            r->next_block = std::min(size * 2, kSharedBlockMax);
            r->cursor = static_cast<uint8_t*>(std::malloc(size));
            if (!r->cursor) throw std::bad_alloc();
            r->blocks.push_back(r->cursor);
            note_shared_block(r->cursor, size, r);
            r->left = size;
        }
        at = r->cursor;
        r->cursor += bytes;
        r->left -= bytes;
    }
    r->bytes += bytes;
    bytes_.fetch_add(bytes, std::memory_order_relaxed);
    std::memset(at, 0, bytes);
    auto* o = reinterpret_cast<Obj*>(at);
    o->type = type;
    o->gc = GC_SHARED;
    o->aux = AUX_DEEP_FORCED;
    o->bytes = uint32_t(bytes);
    return o;
}

void SharedArea::met_shared(const Obj* o) {
    note_shared_crossing(o);
    // An object already shared is not copied again, so the region being built
    // points into the one it is in, and must keep it.
    SharedRegion* on = region_of(o);
    if (!on || on == building_) return;
    for (SharedRegion* d : building_->deps)
        if (d == on) return;
    building_->deps.push_back(on);
}

void SharedArea::free_region(SharedRegion* r) {
    for (SharedRegion* d : r->deps) --d->dependents;
    for (uint8_t* b : r->blocks) {
        forget_shared_block(b);
        std::free(b);
    }
    bytes_.fetch_sub(r->bytes, std::memory_order_relaxed);
    if (r->condemned.load(std::memory_order_relaxed)) g_condemned_regions.fetch_sub(1);
}

bool SharedArea::release(Value v) {
    v = resolve(v);
    if (!is_ptr(v) || !is_shared_obj(as_obj(v))) return false;
    std::lock_guard<std::mutex> g(mutex_);
    SharedRegion* r = region_of(as_obj(v));
    if (!r || r->area != this) return false;
    if (!r->condemned.exchange(true)) {
        r->condemned_at = rounds_;
        g_condemned_regions.fetch_add(1);
    }
    return true;
}

bool SharedArea::any_condemned() {
    std::lock_guard<std::mutex> g(mutex_);
    for (auto& r : regions_)
        if (r->condemned.load(std::memory_order_relaxed)) return true;
    return false;
}

uint64_t SharedArea::begin_round() {
    std::lock_guard<std::mutex> g(mutex_);
    spoiled_.store(false, std::memory_order_relaxed);
    return ++rounds_;
}

size_t SharedArea::finish_round(uint64_t round) {
    std::lock_guard<std::mutex> g(mutex_);
    if (spoiled_.load(std::memory_order_relaxed)) return 0;
    size_t freed = 0;
    // A fixpoint: freeing a region can leave one it depended on with no
    // dependents, which may then go in the same round.
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 0; i < regions_.size(); ++i) {
            SharedRegion* r = regions_[i].get();
            if (!r->condemned.load(std::memory_order_relaxed)) continue;
            if (r->condemned_at >= round) continue;  // condemned after the round began
            if (r->seen_round.load(std::memory_order_relaxed) == round) continue;
            if (r->dependents > 0) continue;
            freed += r->bytes;
            free_region(r);
            regions_[i] = std::move(regions_.back());
            regions_.pop_back();
            --i;
            changed = true;
        }
    }
    return freed;
}

Value SharedArea::make_float(double v) {
    auto* o = static_cast<FloatObj*>(alloc(ObjType::Float, sizeof(double)));
    o->value = v;
    return from_obj(o);
}

Value SharedArea::make_bigint(const uint64_t* limbs, uint32_t len, bool neg) {
    auto* o = static_cast<BigIntObj*>(alloc(ObjType::BigInt, 8 + size_t(len) * 8));
    o->len = len;
    o->neg = neg ? 1 : 0;
    std::memcpy(o->limbs(), limbs, size_t(len) * 8);
    return from_obj(o);
}

Value SharedArea::make_string(const char* data, uint32_t len) {
    auto* o = static_cast<StrObj*>(alloc(ObjType::Str, 8 + size_t(len) + 1));
    o->len = len;
    o->hash = 0;
    if (len && data) std::memcpy(o->data(), data, len);
    o->data()[len] = '\0';
    return from_obj(o);
}

Value SharedArea::make_bigstr(const char* data, uint64_t len) {
    auto* o = static_cast<BigStrObj*>(alloc(ObjType::BigStr, sizeof(BigStrObj) - sizeof(Obj)));
    o->len = len;
    // A big string caches its hash the first time a map asks for it. Here
    // that would be a write every process could race to make, so the answer
    // is worked out now, while the object is still this thread's alone.
    o->hash = bytes_hash(Bytes{data, len});
    o->data = data;
    return from_obj(o);
}

Value SharedArea::make_tensor(uint32_t rank, const uint32_t* dims, uint64_t count) {
    auto* t = static_cast<TensorObj*>(alloc(
        ObjType::Tensor, sizeof(TensorObj) - sizeof(Obj) + size_t(count) * sizeof(double)));
    init_tensor(t, rank, dims, count);
    return from_obj(t);
}

Value SharedArea::make_pid(uint64_t id) {
    auto* o = static_cast<PidObj*>(alloc(ObjType::Pid, sizeof(uint64_t)));
    o->id = id;
    return from_obj(o);
}

Value SharedArea::make_cons(Value head, Value tail) {
    auto* o = static_cast<ConsObj*>(alloc(ObjType::Cons, 2 * sizeof(Value)));
    o->head = head;
    o->tail = tail;
    return from_obj(o);
}

Value SharedArea::make_array(uint32_t len) {
    auto* o = static_cast<ArrayObj*>(alloc(ObjType::Array, 8 + size_t(len) * sizeof(Value)));
    o->len = len;
    return from_obj(o);
}

Value SharedArea::make_record(uint32_t tag, uint32_t len) {
    auto* o = static_cast<RecordObj*>(
        alloc(ObjType::Record, 8 + size_t(len) * sizeof(Value)));
    o->len = len;
    o->tag = tag;
    return from_obj(o);
}

Value SharedArea::make_map_branch(uint32_t nslots) {
    auto* o = static_cast<MapObj*>(alloc(ObjType::Map, 8 + size_t(nslots) * sizeof(Value)));
    o->count = 0;
    o->bitmap = 0;
    return from_obj(o);
}

Value SharedArea::make_map_leaf(uint64_t hash, Value key, Value value, Value next) {
    auto* o = static_cast<MapLeafObj*>(alloc(ObjType::MapLeaf, 8 + 3 * sizeof(Value)));
    o->hash = hash;
    o->key = key;
    o->value = value;
    o->next = next;
    return from_obj(o);
}

Value SharedArea::make_error(Value kind, Value payload, Value where) {
    auto* o = static_cast<ErrorObj*>(alloc(ObjType::ErrorBox, 3 * sizeof(Value)));
    o->kind = kind;
    o->payload = payload;
    o->where = where;
    return from_obj(o);
}

Value SharedArea::make_module(uint32_t import_index, Value name) {
    auto* o = static_cast<ModuleObj*>(alloc(ObjType::Module, 8 + sizeof(Value)));
    o->import_index = import_index;
    o->name = name;
    return from_obj(o);
}

Value SharedArea::make_closure(uint32_t func, uint32_t ncaps) {
    auto* o = static_cast<ClosureObj*>(alloc(ObjType::Closure, 8 + size_t(ncaps) * sizeof(Value)));
    o->func = func;
    o->ncaps = ncaps;
    return from_obj(o);
}

Value SharedArea::make_pap(Value fn, uint32_t nargs) {
    auto* o = static_cast<PapObj*>(
        alloc(ObjType::Pap, sizeof(Value) + 8 + size_t(nargs) * sizeof(Value)));
    o->fn = fn;
    o->nargs = nargs;
    return from_obj(o);
}

bool SharedArea::share(Value v, Value* out, std::string* why) {
    // One copy at a time. A share is rare and large, and the region being
    // built is the area's only state, so a lock around the whole copy is the
    // simple answer and costs nothing anyone would measure.
    std::lock_guard<std::mutex> g(mutex_);
    refused_.clear();
    auto region = std::make_unique<SharedRegion>();
    region->area = this;
    building_ = region.get();
    CopySeen seen;
    Value copied = copy_value(*this, v, seen);
    building_ = nullptr;
    if (!refused_.empty()) {
        // Nothing can reach what was copied before the refusal, so the region
        // goes now, as it is.
        free_region(region.get());
        if (why) *why = "cannot share a " + refused_ + ": force it first";
        return false;
    }
    // A share that copied nothing -- an immediate, or a value already shared
    // -- leaves no region behind.
    if (!region->blocks.empty()) {
        for (SharedRegion* d : region->deps) ++d->dependents;
        regions_.push_back(std::move(region));
    }
    *out = copied;
    return true;
}

}  // namespace dream
