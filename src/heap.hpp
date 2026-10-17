// Per-process heaps.
//
// Every process owns its heap outright: nothing in it is reachable from
// another process, because messages are copied on send. That is what BEAM buys
// with copying semantics, and it buys the same things here -- collection never
// stops the world, never takes a lock, and never has to consider another
// thread's view of an object. It is also what makes lazy thunk update safe
// without atomics: only one process can ever force a given thunk.
//
// The heap is generational: fresh objects bump-allocate in the nursery, and a
// minor collection promotes everything still reachable into old space by
// copying. Old space is a mark-sweep that moves only what it chooses to: a
// major collection evacuates the few blocks its last sweep found mostly empty,
// copying their survivors into the holes of the rest and handing the blocks
// back whole (see "Evacuating sparse blocks" in docs/gc.md). Collection is
// still safe at an interpreter safepoint with no native-stack scanning, for the
// reason promotion always was: `alloc` never collects, and what can collect
// underneath a native is exactly what has vouched that its locals survive one.
//
// The design this implements is written down in docs/gc.md ("Phase 1").

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <vector>

#include "value.hpp"

namespace dream {

/// One free list per size class; the class table lives in heap.cpp.
inline constexpr size_t kHeapClassCount = 55;

/// The collector bits, all in the one-byte `gc` header field. The low two are
/// the mark-sweep cycle's; the generation bits are permanent once set and must
/// survive a collection untouched.
constexpr uint8_t GC_MARK = 1;       ///< in the live set of a full collection
constexpr uint8_t GC_FREE = 2;       ///< already threaded onto a free list
constexpr uint8_t GC_YOUNG = 4;      ///< a nursery object
constexpr uint8_t GC_OLD = 8;        ///< a tenured object
/// A nursery object whose copy already lives in old space: its first payload
/// word holds the forwarding pointer. Only valid while a collection is running.
constexpr uint8_t GC_FORWARDED = 16;
/// A nursery object one collector thread has claimed and is copying right now.
/// Only a parallel collection ever sets it, and only for the length of one
/// `memcpy`: another thread that finds it waits for `GC_FORWARDED` rather than
/// making a second copy, because two copies of one object would be two objects
/// and sharing would not survive the collection.
constexpr uint8_t GC_BUSY = 32;
/// An object in the runtime's shared area (`SharedArea`), not in any heap.
/// Permanent and exclusive: an object carrying it carries nothing else, no
/// collector ever sets or clears a bit of it, and every trace stops at it --
/// it is not this heap's to mark, move or free, and nothing under it can be.
constexpr uint8_t GC_SHARED = 64;
/// A nursery object a conservative root pinned (`Heap::set_conservative_roots`):
/// young, and not to be copied. Only set while a collection is running; the
/// object is an ordinary young one again once it ends.
constexpr uint8_t GC_PINNED = 128;

class Heap;
class SharedArea;
struct SharedRegion;

/// Regions waiting to be freed, in every runtime. While it is zero -- every
/// program that never calls `vm.release!` -- a copy that meets a shared object
/// pays one relaxed load for it and nothing else. See "Freeing what was
/// shared" in docs/gc.md.
extern std::atomic<int> g_condemned_regions;

/// A shared object met while copying from one heap to another: if its region
/// is condemned, a census in progress can no longer trust what it has heard.
/// The slow half of `note_shared_crossing`.
void shared_crossing_slow(const Obj* o);
inline void note_shared_crossing(const Obj* o) {
    if (g_condemned_regions.load(std::memory_order_relaxed) > 0) shared_crossing_slow(o);
}

/// A census major met a shared object: if its region is condemned, it is seen
/// in `round`.
void shared_observe(const Obj* o, uint64_t round);

/// Walk everything `v` reaches without touching a mark bit, noting every
/// condemned region it meets as seen in `round`. For values no collector of
/// the round will trace: messages in flight, kill reasons, and what finished
/// processes left for `join!`.
void shared_census_walk(Value v, uint64_t round);

/// What a tensor that lives outside the heap holds, and how its reference is
/// taken and given back. Implemented in tensor.cpp, which knows what a GPU
/// buffer is; the heap only knows that some objects hold one.
void* tensor_handle(const TensorObj* t);
void retain_external(Obj* o);
void release_external(Obj* o);
/// The most external memory a heap should let accumulate before it collects,
/// whatever the doubling rule says: a quarter of the GPU's memory once a GPU
/// is in use, and no bound before then. Defined beside the tensors.
size_t external_budget();

/// Supplies the roots for a collection. Implemented by Process.
struct RootSource {
    virtual ~RootSource() = default;
    /// Call `heap.forward(&value)` for every live reference.
    virtual void visit_roots(Heap& heap) = 0;
};

/// The most any process's nursery may grow to, from here on; see
/// `nursery_max_bytes` in heap.cpp for who sets it and in what order.
void set_nursery_max(size_t bytes);

class Heap {
public:
    explicit Heap(size_t initial_bytes = 64 * 1024);
    ~Heap();
    Heap(const Heap&) = delete;
    Heap& operator=(const Heap&) = delete;

    /// Allocate a zeroed object with `extra` payload bytes after the header.
    /// Routes to the nursery, except for objects past the top size class,
    /// which tenure immediately in old space. Never collects: growing here
    /// would invalidate references held in C++ locals by the caller.
    /// Collection happens at safepoints instead.
    Obj* alloc(ObjType type, size_t extra);

    /// The same, without clearing the payload. For the constructors below that
    /// write every field they own.
    ///
    /// The clear is pure waste for those: a thunk is twenty-four bytes and all
    /// twenty-four are written immediately afterwards. Anything whose payload a
    /// *caller* fills in later -- a frame's unbound slots, an array handed to a
    /// host -- must still arrive zeroed, because until it is filled the
    /// collector may walk it, and an unwritten slot has to read as `NIL_SLOT`.
    Obj* alloc_bare(ObjType type, size_t extra);

    /// A frame whose first `filled` slots the caller is about to write. Only
    /// the rest is cleared -- an unwritten slot must read as `NIL_SLOT`,
    /// because that is what "not bound yet" is -- and the caller's own stores
    /// do the work the clear would have done twice.
    Value make_frame_filling(Value closure, uint32_t nslots, uint32_t filled);

    // Convenience constructors for the common shapes.
    Value make_float(double v);
    Value make_string(const char* data, uint32_t len);
    /// A view onto bytes the heap does not own -- the image's payload region.
    /// Nothing is copied, so `data` must outlive every heap that can reach it;
    /// today the only such bytes are the `Runtime`'s mapped image.
    Value make_bigstr(const char* data, uint64_t len);
    Value make_cons(Value head, Value tail);
    Value make_array(uint32_t len);
    /// A record of `len` fields, left for the caller to fill; see RecordObj.
    Value make_record(uint32_t tag, uint32_t len);
    /// An empty map: a branch with no children. The capacity argument is a
    /// leftover of the open-addressed table and is ignored -- a trie sizes
    /// itself -- but it is kept so that callers with a sensible hint need not
    /// all change.
    Value make_map(uint32_t capacity);
    /// A branch with room for `nslots` packed children. The caller fills in the
    /// bitmap and the slots.
    Value make_map_branch(uint32_t nslots);
    Value make_map_leaf(uint64_t hash, Value key, Value value, Value next);
    Value make_closure(uint32_t func, uint32_t ncaps);
    Value make_thunk(uint32_t node, Value frame);
    Value make_frame(Value closure, uint32_t nslots);
    Value make_pap(Value fn, uint32_t nargs);
    Value make_error(Value kind, Value payload, Value where = UNIT);
    Value make_pid(uint64_t id);
    /// A bignum with room for `limbs` limbs, uninitialised: the caller writes
    /// them and sets `len` and `neg`. See `bigint::finish`.
    BigIntObj* alloc_bigint(uint32_t limbs);
    /// A copy of a bignum's limbs, sign and all. Not canonicalised: the source
    /// already is.
    Value make_bigint(const uint64_t* limbs, uint32_t len, bool neg);
    /// A tensor of the given shape, its data uninitialised: the caller fills
    /// every element. `dims` holds `rank` axes; `count` is their product, which
    /// the caller has already checked fits (see `tensor_bytes_ok`).
    Value make_tensor(uint32_t rank, const uint32_t* dims, uint64_t count);
    /// A tensor whose numbers live somewhere else -- a GPU's memory -- named by
    /// `handle`, of which the new object takes ownership of one reference. The
    /// heap keeps a list of such objects and releases the handle when the
    /// collector finds the object dead; see `reap_external`.
    Value make_external_tensor(uint32_t rank, const uint32_t* dims, uint64_t count,
                               uint8_t dtype, uint8_t device, void* handle);
    /// A deferred tensor (`TensorExpr`) with room for a program of this size,
    /// its inputs empty and its result unset, for the caller to fill.
    Value make_deferred_tensor(uint32_t rank, const uint32_t* dims, uint64_t count,
                               uint8_t dtype, uint8_t device, uint32_t ninputs,
                               uint32_t nconsts, uint32_t ncode);
    Value make_module(uint32_t import_index, Value name);

    /// True when the process should collect at its next safepoint: old space
    /// past its threshold, or the nursery past its high-water mark.
    ///
    /// With a concurrent mark in flight neither triggers a *new* collection --
    /// the helpers may be walking an old object's fields at this very moment,
    /// so the only collection this process may do is the finalize of the one
    /// it already owes. The finalize becomes due when the helpers have drained,
    /// or when the nursery has outgrown twice its high-water mark (the normal
    /// mark, which would have triggered a minor, is not worth the handshake;
    /// twice the normal mark is). The doubling is capped at the nursery's
    /// maximum so it cannot handshake for a grow that was not allowed.
    bool should_collect() const {
        if (marking_)
            return mark_done_.load(std::memory_order_relaxed) ||
                   nursery_bytes_ >= std::min(nursery_max_, nursery_hi_ * 2);
        return major_due() || minor_due();
    }
    /// Bytes held in old space: everything allocated that the nursery is not
    /// still holding. `allocated_` counts both generations, because it is what
    /// says how full the heap is; this is the half a major is about.
    size_t old_bytes() const {
        // A held nursery block (`nursery_held_`) is young memory too.
        const size_t young = nursery_bytes_ + nursery_held_;
        return allocated_ >= young ? allocated_ - young : 0;
    }

    /// Old space past the threshold the last full collection fitted to it.
    ///
    /// The nursery is deliberately not in that comparison -- `gc_threshold_`
    /// is fitted to what a major found live in *old* space, so counting the
    /// nursery against it measures a quantity a minor is about to empty
    /// against one only a major can move. On the self-compile, separating them
    /// took the majors from 176 ms to 16 ms: a major that fires while the
    /// nursery is full promotes the whole of it first, so the conflation was
    /// not just miscounting, it was picking the expensive collection.
    ///
    /// The second clause is what the first one costs. A nursery can be
    /// hundreds of megabytes past its high-water mark by the time a safepoint
    /// comes round, because a deep force (`strict!`) runs the interpreter in a
    /// nested loop that may not collect -- a native above it holds raw
    /// pointers, which is the "alloc never collects" rule -- and the safepoint
    /// after it lands at the one instant when everything just allocated is
    /// still live. A minor there is a copy with no collection in it: it
    /// promotes the lot, and the lot dies immediately afterwards. So when the
    /// last minor promoted more than three quarters of what it looked at, the
    /// nursery is treated as old space's problem again and the whole heap is
    /// the trigger. A `strict!` in a loop collected 2.4x faster for it, and
    /// the compile does not notice, because its minors promote about a
    /// quarter.
    bool major_due() const {
        if (old_bytes() >= gc_threshold_ || external_major_) return true;
        return !minor_pays_ && allocated_ >= gc_threshold_;
    }
    /// The nursery is full -- or the memory held *outside* the heap has grown
    /// past its trigger. A GPU tensor is fifty-six bytes of heap and megabytes
    /// of device memory, so a loop making them fills the device long before it
    /// fills the nursery; without the second clause nothing would ever collect
    /// and nothing would ever be released. See `reap_external`.
    bool minor_due() const {
        return nursery_bytes_ >= nursery_hi_ || external_bytes_ >= external_trigger_;
    }

    /// Full collection: promote every reachable young object and mark-sweep
    /// old space. Safe only at an interpreter safepoint.
    void collect(RootSource& roots);
    /// The same, under its proper name.
    void major_collect(RootSource& roots);
    /// Minor collection: promote every reachable young object out of the
    /// nursery into old space, leaving it empty. The trace starts from the
    /// roots and the remembered set; old space is neither marked nor swept.
    void minor_collect(RootSource& roots);

    /// How many objects this heap holds that own something outside it.
    size_t external_count() const { return external_.size(); }

    /// Phase 2's step 3: a major whose *marking* overlaps the mutator.
    ///
    /// `start_concurrent_mark` snapshots the roots at this safepoint, seeds
    /// the helper threads with them, and returns at once -- the helpers mark
    /// the old objects they can reach while the mutator keeps running, and
    /// skip young ones, which move and so cannot be marked ahead of time.
    /// `finalize_concurrent_mark` then does what a major does: finish the
    /// graph with one more pass over what the overlap could have missed, sweep,
    /// and empty the nursery.
    ///
    /// The mutator runs no collection of any kind between the two calls --
    /// `should_collect` says "finalize" when `marking()` is true, and nothing
    /// else. `finalize_concurrent_mark` is the one legal way out.
    ///
    /// `start_concurrent_mark` returns false when the helpers cannot help (the
    /// heap is too small, the pool has no threads, another heap owns it); the
    /// caller then collects in the old whole-world way and the pair is not
    /// entered. `marking()` is the third state, told from the two methods by
    /// who is asking: `should_collect`/`maybe_collect` drive it, and the rest
    /// of the runtime only ever reads `marking()` to pick the legal next move.
    bool start_concurrent_mark(RootSource& roots);
    void finalize_concurrent_mark(RootSource& roots);

    /// A census major: a whole-world `major_collect` that notes every
    /// condemned shared region it meets as seen in `round`. A concurrent mark
    /// in flight is finalized first, because its helpers marked without
    /// looking. Safe only where a major is.
    void census_collect(RootSource& roots, uint64_t round);
    /// Where a copy into this heap meets a shared object. See `copy_value`.
    void met_shared(const Obj* o) { note_shared_crossing(o); }
    /// True between `start_concurrent_mark` and `finalize_concurrent_mark`.
    bool marking() const { return marking_; }
    /// True once the helpers have drained their last batch: the mark is done
    /// and only the finalize's housekeeping is owed. Read by `should_collect`
    /// to make the finalize prompt when nothing more can be gained from the
    /// overlap, so a heap does not drift on with a finished mark behind it.
    bool mark_done() const { return mark_done_.load(std::memory_order_relaxed); }

    /// The machine stack of a process collecting underneath JIT-compiled
    /// code, as the words from `lo` up to `hi`. The next collection pins in
    /// place every object one of those words points into, and `clear` ends it.
    ///
    /// A compiled frame keeps its values in machine registers and stack
    /// slots, which nothing can find precisely and nothing may rewrite. So the
    /// collection that runs under one does not move what such a frame could
    /// be holding: any word that lands inside an object of this heap -- its
    /// start or anywhere in it, because a compiled loop may keep only a field
    /// address -- is treated as naming that object. A young one stays young
    /// and stays put (`GC_PINNED`), in a nursery block the collection does not
    /// empty; the next collection judges it like any other young object. An
    /// old one is kept out of the evacuation. Everything they point at is
    /// traced and moved as usual, because the collector rewrites the fields of
    /// a pinned object and never the word that pinned it. A word that is not
    /// a pointer at all, but looks like one, keeps something alive for one
    /// collection more; that is the whole cost of guessing. "Collecting under
    /// compiled code" in docs/gc.md.
    void set_conservative_roots(const void* lo, const void* hi) {
        cons_lo_ = static_cast<const uintptr_t*>(lo);
        cons_hi_ = static_cast<const uintptr_t*>(hi);
    }
    void clear_conservative_roots() {
        cons_lo_ = cons_hi_ = nullptr;
        for (Obj* o : pinned_) o->gc &= uint8_t(~GC_PINNED);
        pinned_.clear();
    }

    /// Hand one root to the collector: mark what `*slot` holds, and rewrite
    /// the slot when promotion moves it. This is the entry point a
    /// `RootSource` uses, and the only one; the scanner reaches the same code
    /// through the templates in heap.cpp, where the collector knows whether it
    /// is running on one thread or several and can afford to say so at compile
    /// time.
    ///
    /// A collection asks the templated form ninety million times in one
    /// self-compile, so it is worth an inlined fast path and two compiled
    /// shapes. The roots are a few thousand, so this one can be a call.
    void forward(Value* slot);

    /// The write barrier. The mutator calls this on an in-place store into an
    /// existing object, so that a minor collection can still find every
    /// old-to-young reference. Cheap, and never needs a lock: only the owning
    /// process writes its own heap.
    ///
    /// The reads are atomics because the one byte being read is the same one a
    /// concurrent mark's helper claims with `fetch_or` at the same moment --
    /// same instruction as a plain load, but a plain load against an atomic
    /// write is a data race whether or not it matters.
    ///
    /// During a concurrent mark the same store can also uncover a subgraph the
    /// helpers have already drawn: an object is logged the first time a store
    /// lands in one that is marked, so the finalize re-walks it -- and the
    /// work is handed to the helpers now, so the re-walk happens concurrently
    /// rather than at the finalize. A store into an object the helpers have
    /// not marked yet needs none of this: their scan, when it comes, sees it.
    void remember_if_old(Obj* dst, Value value) {
        uint8_t dgc = std::atomic_ref<uint8_t>(dst->gc).load(std::memory_order_acquire);
        if ((dgc & GC_OLD) && is_ptr(value) &&
            (std::atomic_ref<uint8_t>(as_obj(value)->gc).load(std::memory_order_acquire) &
             GC_YOUNG))
            remembered_.push_back(dst);
        if (marking_ && (dgc & GC_MARK)) log_mutation(dst);
    }

    size_t bytes_allocated() const { return allocated_; }
    size_t bytes_live() const { return live_after_gc_; }
    uint64_t collections() const { return collections_; }
    /// Memory held outside the heap by objects in it: GPU buffers, a
    /// bignum's limbs.
    size_t external_bytes() const { return external_bytes_; }
    uint64_t major_collections() const { return major_collections_; }
    uint64_t minor_collections() const { return minor_collections_; }
    /// Rounds that were divided across the helper threads rather than run on
    /// the collecting thread alone. A mark and a sweep count separately: they
    /// are separate rounds, and either may fall back on its own.
    uint64_t parallel_rounds() const { return parallel_collections_; }
    /// Bytes of nursery space copied into old space by promotion.
    uint64_t bytes_promoted() const { return promoted_bytes_; }
    /// Bytes of old space copied out of sparse blocks by evacuation, and the
    /// blocks that were handed back for it.
    uint64_t bytes_evacuated() const { return evacuated_bytes_; }
    uint64_t blocks_evacuated() const { return evacuated_blocks_; }
    /// Nanoseconds this process spent stopped in a collection, split by kind.
    /// Wall time, not CPU time: what a collection costs is what the process
    /// could not do while it ran, and a parallel collector's whole claim is
    /// that those two numbers come apart.
    uint64_t nanos_minor() const { return minor_nanos_; }
    uint64_t nanos_major() const { return major_nanos_; }
    /// Nanoseconds the mutator spent running *over* a concurrent mark -- the
    /// window in which the helpers were working in the background. Not stopped
    /// time; it is reported beside the stopped time precisely so the two can
    /// be told apart.
    uint64_t nanos_concurrent() const { return concurrent_nanos_; }
    /// Full collections whose marking ran on the helper threads rather than on
    /// the collecting thread alone.
    uint64_t concurrent_marks() const { return concurrent_marks_; }
    /// Every byte this heap has ever handed out, collections included. What
    /// `bytes_allocated` reports is reset by a collection, so it says how full
    /// the heap is rather than how much work has gone through it -- and the
    /// second question is the one that finds a program allocating quadratically.
    uint64_t bytes_total() const { return total_allocated_; }
    /// The largest the live set has ever been after a collection: the memory
    /// the program actually needs, as opposed to the garbage it made getting
    /// there.
    size_t bytes_peak() const { return peak_live_; }
    /// The most this heap has ever held *from the OS*: every block it had
    /// malloc'd at once, nursery and old space together, whether or not the
    /// objects in them were live.
    ///
    /// This is the number a person watching `top` sees, and it is not
    /// `bytes_peak`. The difference between them is everything the collector
    /// is carrying and not using -- headroom above the live set, chunks on the
    /// free lists, and the blocks a sweep emptied but cannot hand back because
    /// something else in them is still alive.
    size_t bytes_peak_held() const { return peak_block_bytes_; }
    /// Bytes of dead big blocks kept for reuse (`big_pool_`).
    size_t bytes_pooled() const { return big_pool_bytes_; }
    /// What was *allocated* at the moment the heap held the most -- objects
    /// that existed, live or merely not yet proven dead. Reported beside the
    /// held figure because the gap between the two is the collector's own
    /// waste (headroom and holes), and the gap between *this* and the live set
    /// is what a different trigger policy could in principle recover. On the
    /// self-compile it is 88% of the peak, which is what says the peak is a
    /// heap full of objects rather than a heap full of holes -- and so why
    /// four different threshold policies all failed to move it. See docs/gc.md.
    size_t bytes_alloc_at_peak() const { return peak_allocated_; }
    const std::array<uint64_t, 32>& bytes_by_type() const { return by_type_; }
    /// The live set of the largest major collection, and what it was made of.
    ///
    /// `bytes_by_type` says where a program's *allocation* went, which is the
    /// question "what is this program making?" -- and for a lazy language the
    /// answer is always thunks and frames, because making a call is most of
    /// what any program does. This is the other question: of what survived,
    /// what is it? A compile that holds 400 MB at its peak is held back by
    /// whatever those bytes are, and nothing else in this runtime could say.
    ///
    /// A *major* because only a major proves anything: a minor leaves old
    /// space untouched, so its idea of "live" is every object old space has,
    /// dead ones included. So this is not necessarily the moment `bytes_peak`
    /// reports -- it is the largest live set anything here can vouch for, and
    /// the two are printed together for that reason.
    size_t bytes_live_at_major_peak() const { return peak_major_live_; }
    const std::array<uint64_t, 64>& live_by_type_at_peak() const { return peak_by_type_; }

    /// Deep-copy `v` out of this heap into `dest`. Used for message sends and
    /// for spawning, which are the only two places a value crosses heaps.
    static Value copy_between(Heap& dest, Value v);

    /// Walk everything reachable from `roots` and check the object graph holds
    /// together: every pointer lands on an object this heap owns, every header
    /// is plausible, and no object straddles the end of its block. Returns an
    /// empty string when all is well.
    ///
    /// This is the check that catches a collector or allocator bug at the
    /// moment it happens rather than as a crash somewhere unrelated later.
    std::string verify(RootSource& roots);

    /// True when this heap owns `bytes` starting at `p`.
    bool owns(const void* p, size_t bytes) const;

    /// Set from the DREAM_VERIFY_HEAP environment variable: verify after every
    /// collection. Slow, and meant for chasing exactly this class of bug.
    static bool verify_after_gc();

private:
    /// Objects that hold a reference to something the heap does not own -- a
    /// GPU buffer, a large bignum's limbs -- and so must say when they die.
    /// Nothing else in this language needs a finalizer, which is why this is a
    /// list beside the
    /// heap and not a bit on every object: the collector looks at these and at
    /// nothing else when it asks what to release.
    std::vector<Obj*> external_;
    /// What `external_` holds outside the heap, in bytes, and the level that
    /// asks for a collection: twice what survived the last one, and never less
    /// than `kExternalSlack` above it.
    size_t external_bytes_ = 0;
    size_t external_trigger_ = kExternalSlack;
    /// A minor collection freed too little of it: what is held is old, and only
    /// a major can tell whether it is still reached.
    bool external_major_ = false;
    static constexpr size_t kExternalSlack = size_t(64) << 20;
    /// Release what the dead among `external_` hold, and follow the living to
    /// where a minor moved them. Called once the trace is complete and before
    /// the nursery is emptied or old space swept, which is the one moment both
    /// "was it reached" and "where is it now" can be read off the headers.
    /// `full` says old objects were marked, so an unmarked one is dead.
    void reap_external(bool full);
    /// What one of `external_` holds outside the heap, and letting it go:
    /// a tensor's GPU buffer, or a bignum's limbs.
    static size_t external_size(Obj* o);
    static void drop_external(Obj* o);

    struct Block {
        Block* next;
        size_t size;
        size_t used;
        uint8_t* data;
        /// True when the whole block is one big object's. Nothing else may be
        /// carved into it, and sweep may then free the whole block without
        /// stranding any live chunk.
        bool big;
        /// Set by the sweep of a big block whose object died. The block list
        /// is rebuilt afterwards rather than unlinked from as it is walked,
        /// because in a parallel sweep the thread that finds a dead block is
        /// not the one that owns the list.
        bool dead;
        /// Chosen by the last sweep to be emptied by the next major: its live
        /// share was small enough that copying it out is cheaper than holding
        /// the rest. None of its holes are on a free list and nothing is carved
        /// from its tail, so nothing new lands in it while it waits.
        bool evacuate;
        /// A nursery block holding an object a conservative root pinned. The
        /// collection running now leaves it as it is rather than emptying it.
        bool pinned;
        /// A nursery block the last collection held: how far it was filled
        /// then, and which of `held_pins_` were its pinned objects. Below
        /// `held_end` everything else is dead; see `hold_pinned_blocks`.
        /// Zero `held_end` for any other block.
        uint32_t held_end;
        uint32_t held_from;
        uint32_t held_count;
    };

    /// One collector thread's private state for the length of one collection.
    /// Defined in heap.cpp: nothing outside the collector has any business
    /// with it, but a parallel round needs an array of them and the array has
    /// to live somewhere.
    struct GcCtx;
    /// The state a parallel round shares: the overflow work stack that
    /// balances the trace, the block snapshot the sweep divides, and the lock
    /// that guards the free lists while several threads carve from them.
    struct GcRound;

    /// A block of at least `bytes`, and at least the heap's minimum block
    /// unless `exact` -- which a dedicated big-object block is.
    Block* new_block(size_t bytes, bool exact = false);
    void free_block(Block* b);
    void free_blocks(Block* b);
    /// Free every nursery block: after a major collection every reachable
    /// young object has been promoted, so the whole nursery is dead.
    void free_nursery();

    /// Take `sz` bytes of new space in old space: a fresh chunk carved off the
    /// tail of the carve block (or a new block, or a split from the class
    /// lists). Used by promotion; normal allocation goes to the nursery.
    Obj* carve(size_t sz);
    /// An entire block dedicated to one large object.
    Obj* carve_big(size_t sz);
    /// A dead big object's block, kept for the next big object of its size
    /// class rather than handed back to the system. See `big_pool_`.
    void retire_big(Block* b);
    /// After a sweep: hand back what the last sweep retired and nothing has
    /// reused since, and anything past the pool's cap.
    void trim_big_pool();
    /// Bump-allocate `sz` bytes from the nursery blocks.
    Obj* alloc_nursery(uint32_t sz);
    /// Add a block to the nursery and allocate from it. The cold half of
    /// `alloc_nursery`, reached only when no existing block has room.
    Obj* grow_nursery(uint32_t sz);

    /// Widen the nursery when a collection promoted a large share of what it
    /// looked at. Called at the end of a minor collection.
    void grow_nursery_if_crowded(size_t promoted, size_t looked_at);

    // The collector proper, in two compiled shapes. `Par` is what the code
    // would otherwise have had to ask at runtime, ninety million times a
    // collection: whether another thread can be looking at the same object.
    // With it false these are exactly the single-threaded collector, plain
    // loads and stores throughout; with it true every read of a `gc` byte is
    // an atomic, the mark is a claim, and promotion is a claim followed by a
    // copy that other threads wait for. Both are instantiated in heap.cpp.

    /// Mark what `*slot` holds, rewriting the slot if promotion moves it.
    /// `Con` is the concurrent-mark shape: instead of promoting and absorbing,
    /// it hands each reference to `mark_ref`, which marks what the helpers may
    /// reach and skips the young.
    template <bool Par, bool Con> void forward_in(GcCtx& c, Value* slot);
    /// The half of `forward_in` that has something to do: a young object to
    /// promote, one already promoted this cycle, or an indirection to fold
    /// away.
    template <bool Par> void forward_slow(GcCtx& c, Value* slot);
    /// Place one reference onto the live set.
    template <bool Par> void mark_object(GcCtx& c, Value v);
    /// The concurrent mark's own claim on one reference. Unlike `mark_object`
    /// it will not touch a young object -- those move at the finalize, and a
    /// mark on a soon-to-be-corpse is a mark on nothing -- and it collapses no
    /// indirection chains, because a chain it "fixed" would be a write the
    /// mutator is reading. It is called by the helpers on every reference they
    /// scan and by nothing else.
    bool mark_ref(GcCtx& c, Value v);
    /// Claim an old object for the concurrent mark: set GC_MARK, say whether
    /// this caller won. Shared by `mark_ref` and the seed in
    /// `start_concurrent_mark`.
    static bool claim_mark(Obj* o);
    /// Copy a nursery object to old space and leave a forwarding pointer
    /// behind, so that the second reference to it finds the same copy.
    template <bool Par> Obj* promote(GcCtx& c, Obj* o);
    /// Update the references inside one object, pushing each further out.
    template <bool Par, bool Con> void scan_object(GcCtx& c, Obj* o);
    /// Take `sz` bytes of old space to promote into. Serially that is the
    /// heap's own `carve`; in parallel it is the thread's private chunk lists
    /// and its private carve block, so that promotion needs no lock per
    /// object.
    template <bool Par> Obj* gc_carve(GcCtx& c, uint32_t sz);
    /// Find what the conservative roots point into and pin it; see
    /// `set_conservative_roots`. The first thing every collection does, so
    /// that nothing it then does can move a pinned object. `full` for one that
    /// marks old space; a minor looks only at the nursery.
    void pin_conservative(bool full);
    /// The pinned objects as roots of the trace now running in `c`: their
    /// fields traced, and an old one marked in a full trace.
    template <bool Par> void trace_pinned(GcCtx& c);
    /// The end of a collection that pinned anything young: what is left in
    /// each pinned block besides the pinned objects is made inert, and the old
    /// objects that point at them go on the remembered set for the next minor.
    /// `allocated_` is left counting the held blocks, and `nursery_held_`.
    void hold_pinned_blocks();
    /// After an evacuation, point `pin_refs_` at where its objects went, and
    /// the pinned young objects' fields at where theirs did.
    void forward_pin_refs();
    /// Drain the thread's own queue, taking from and giving to the round's
    /// shared stack, until every thread has run out of work at once.
    template <bool Par, bool Con> void drain(GcCtx& c);

    /// The old-space copy a forwarded nursery object points at, read back out
    /// of its first payload word.
    static Value forward_target(Obj* o) {
        return *reinterpret_cast<Value*>(o + 1);
    }
    /// Wait for the thread that claimed `o` to finish copying it, and answer
    /// with the copy. A claim covers one `memcpy` of at most four kilobytes,
    /// so this spins rather than sleeps.
    static Value await_forward(Obj* o);

    /// Trace the live set, in parallel when the pool can spare the threads and
    /// the collection is large enough to be worth the handshake. Returns false
    /// when it did not run, and the caller must trace on its own thread.
    bool trace_in_parallel(RootSource& roots, bool minor);
    /// The same trace on the collecting thread alone: what every collection
    /// did before there was a pool, and what most of them still do.
    void trace_alone(RootSource& roots, bool minor);
    /// Sweep old space across several threads, the blocks divided between
    /// them. Returns false when it did not run.
    bool sweep_in_parallel();
    /// Split up to a batch of chunks off the shared free list for `cls` into
    /// `c`. False when the list is empty and the caller must carve.
    bool refill_free(GcCtx& c, size_t cls);
    /// A block for one collector thread to carve promoted objects out of.
    Block* thread_block(uint32_t sz);
    /// Give half of this thread's queue to the round's shared stack.
    void share_work(GcCtx& c);
    /// Take a batch back when the local queue has run dry, waiting for one to
    /// appear. False when the trace is over and no batch is coming.
    bool take_work(GcCtx& c);

    /// Sweep one block into `head`/`tail`, a chain per size class, counting
    /// what survives. Shared by the serial sweep and the parallel one, which
    /// differ only in whose chains they fill and who rebuilds the block list.
    void sweep_block(Block* b, Obj** head, Obj** tail, size_t& live, uint64_t* by_type);

    /// Move every marked object out of the blocks the last sweep chose, leave a
    /// forwarding stub behind, and point the roots and `external_` at the
    /// copies. Called once a major's trace is complete and before its sweep,
    /// which rewrites the references inside the heap as it passes them.
    void evacuate(RootSource& roots);
    /// Take the blocks `evacuate` emptied off the block list, so that no sweep
    /// walks them; the sweep frees them once it has finished reading their
    /// stubs.
    std::vector<Block*> detach_evacuated();
    void free_evacuated(std::vector<Block*>& gone);
    /// Whether a block the sweep just walked should be evacuated by the next
    /// major, given the bytes it found alive there. Claims the bytes from the
    /// cycle's budget when it says yes.
    bool choose_to_evacuate(const Block* b, size_t live_here);
    /// Rewrite every reference in `o` that names an evacuated object.
    static void fix_refs(Obj* o);
    void note_live_by_type(size_t live, const uint64_t* by_type);
    static std::string live_kinds_line(const uint64_t* by_type, size_t live);
    /// What the last major's sweep found, for its trace line. Built only
    /// when the trace is on, because formatting it is not free.
    std::string last_kinds_;
    /// Prepend one thread's swept chains to the heap's free lists.
    void merge_free_lists(Obj** head, Obj** tail);

    /// Thread every dead object in every old block onto the size-class free
    /// lists. Also clears the mark bits of what survives and recounts the
    /// live set. Called only after a major collection.
    void sweep();

    /// Run the graph walk after a collection when `DREAM_VERIFY_HEAP` asks
    /// for it. `expect_no_young` is the promise a minor collection makes:
    /// everything reachable was promoted, so nothing may still be young.
    void verify_collect(RootSource& roots, bool expect_no_young);
    /// The walk itself, shared by `verify` and `verify_collect`.
    std::string verify_internal(RootSource& roots, bool no_young);

    std::array<Obj*, kHeapClassCount> free_lists_{};
    /// Old-space blocks. Everything that has survived a collection lives here.
    Block* blocks_ = nullptr;
    Block* carve_block_ = nullptr;
    /// The nursery: fresh allocations bump through these and every reachable
    /// one is copied to old space at the next collection, after which the
    /// blocks are reset empty (or freed whole by a major).
    std::vector<Block*> nursery_;
    /// Which block is being bumped in. An *index*, not a pointer, because a
    /// minor collection empties every block and allocation has to start over
    /// at the first one: a cursor left at the end would add a block per
    /// collection and never touch the ones it had just emptied, so the nursery
    /// grew by its whole size at every minor until the next major freed it.
    size_t nursery_at_ = 0;
    size_t nursery_bytes_ = 0;
    /// The nursery's high-water mark: past it, a minor collection is due.
    size_t nursery_hi_;
    /// How large the nursery may grow. See `grow_nursery_if_crowded`.
    size_t nursery_max_;
    /// Old objects the mutator has been made to point at young ones. Drained
    /// (not deduplicated) by scanning each entry at the next minor collection.
    std::vector<Obj*> remembered_;
    /// See `set_conservative_roots`. Null outside a collection under compiled
    /// code, which is every collection but those.
    const uintptr_t* cons_lo_ = nullptr;
    const uintptr_t* cons_hi_ = nullptr;
    /// What they pinned, for the length of one collection; the nursery blocks
    /// it holds; and the old objects found pointing at a pinned young one,
    /// which are owed to the next minor's remembered set.
    std::vector<Obj*> pinned_;
    std::vector<Block*> pinned_blocks_;
    std::vector<Obj*> pin_refs_;
    /// The young objects this collection pinned, as (block, offset), and the
    /// offsets the last one pinned in each block it held -- `Block::held_from`
    /// indexes it. See `hold_pinned_blocks`.
    std::vector<std::pair<Block*, uint32_t>> young_pins_;
    std::vector<uint32_t> held_pins_;
    /// What the held blocks had in them when they were held: allocated, and
    /// left out of `nursery_bytes_`. See `hold_pinned_blocks`.
    size_t nursery_held_ = 0;
    size_t allocated_ = 0;
    uint64_t total_allocated_ = 0;
    size_t peak_live_ = 0;
    /// Bytes currently malloc'd for blocks, and the most there have ever been.
    size_t block_bytes_ = 0;
    /// Big blocks whose objects died, kept to be reused.
    ///
    /// A large object -- a tensor of a few million numbers -- has a block to
    /// itself, and handing that block back to `free` when the object dies
    /// returns it to the system: glibc serves anything this large with `mmap`
    /// and gives it back with `munmap`. The next one is then fresh pages, and
    /// the first write to each page faults. A loop that makes a 32 MB tensor
    /// per iteration measured 40 ms an iteration in faults against 8 ms for the
    /// work, whenever it was not lucky enough to land on reused memory.
    ///
    /// So a dead big block keeps its memory, filed by size class
    /// (`big_size_class`, eight classes per doubling, so a loop's tensors of
    /// one shape always match), and the next big object of that class takes
    /// the most recently retired one, whose pages are warmest. What one sweep
    /// retires and the program does not reuse before the next is handed back
    /// then, and the pool never holds more than the larger of
    /// `kBigPoolFloor` and the live heap. Pooled bytes still count in
    /// `block_bytes_`: they are held from the system, which is what that says.
    ///
    /// Filed by block size, each size's blocks in the order they were retired,
    /// so taking one is a `lower_bound` and a `pop_back`. It was one flat list
    /// searched end to end, which was right for tensors -- a handful at a time
    /// -- and wrong for a loop of bignums past the top class, which retires a
    /// thousand per sweep and then searched all of them, and erased from the
    /// middle, at every allocation.
    struct PooledBlock {
        uint8_t* data;
        size_t size;
        uint64_t sweep;
    };
    std::map<size_t, std::vector<PooledBlock>> big_pool_;
    size_t big_pool_bytes_ = 0;
    uint64_t sweeps_ = 0;
    static constexpr size_t kBigPoolFloor = size_t(64) << 20;
    /// Bytes handed out, by object kind. Where a program's garbage actually
    /// comes from -- a lazy language's answer is usually "thunks and frames",
    /// and knowing the share is what says whether a strictness analysis would
    /// be worth writing. One add per allocation, indexed by the type byte.
    std::array<uint64_t, 32> by_type_{};
    /// The live set of the largest major collection so far, by object kind.
    /// Written only by the thread that finishes a sweep, which is the same
    /// thread that writes every other heap-wide number.
    /// Bytes in the low half, object counts in the high half: a live set is
    /// only half described by its bytes, since "map 55%" reads very
    /// differently as a thousand fat branches and as a million thin ones.
    std::array<uint64_t, 64> peak_by_type_{};
    size_t peak_major_live_ = 0;
    size_t peak_block_bytes_ = 0;
    size_t peak_allocated_ = 0;
    size_t gc_threshold_;
    size_t initial_bytes_;
    size_t live_after_gc_ = 0;
    uint64_t collections_ = 0;
    uint64_t major_collections_ = 0;
    uint64_t minor_collections_ = 0;
    uint64_t promoted_bytes_ = 0;
    uint64_t evacuated_bytes_ = 0;
    uint64_t evacuated_blocks_ = 0;
    /// Blocks chosen for the next major to empty, and whether the sweep running
    /// now has stubs to rewrite references through.
    size_t evacuate_pending_ = 0;
    bool fixing_refs_ = false;
    /// What the sweep running now may still choose to evacuate, in live bytes,
    /// and how much it has chosen. Atomic because the parallel sweep's threads
    /// choose as they go.
    size_t evacuate_budget_ = 0;
    std::atomic<size_t> evacuate_chosen_{0};
    /// What the last evacuation moved, for the trace line.
    size_t last_evacuated_ = 0;
    size_t last_evacuated_blocks_ = 0;
    uint64_t parallel_collections_ = 0;
    uint64_t minor_nanos_ = 0;
    uint64_t major_nanos_ = 0;
    /// True while a major collection is running: `forward` marks the old
    /// objects it reaches as well as promoting the young ones.
    bool full_trace_ = false;
    /// True while a parallel round is running, which is what tells `forward`
    /// -- the root entry point, and the only part of the collector that is not
    /// compiled twice -- which shape of the collector to call into.
    bool parallel_ = false;
    /// The context the roots are handed to: the collecting thread's own.
    GcCtx* root_ctx_ = nullptr;
    /// Non-null only while a parallel round is running.
    GcRound* round_ = nullptr;
    /// The grey set while collecting: objects marked but not yet scanned.
    std::vector<Obj*> scan_queue_;
    /// Whether the last minor collection was worth doing: it promoted no more
    /// than three quarters of what it looked at, so the nursery was mostly
    /// garbage and copying the survivors bought something. False after a minor
    /// that promoted nearly everything -- see `major_due`, which then stops
    /// preferring a minor. Starts true: a heap that has never collected has no
    /// evidence against the generational bet.
    bool minor_pays_ = true;

    /// Non-null while verifying: `forward` records roots rather than marking.
    std::vector<Value>* recording_ = nullptr;
    /// True while `evacuate` hands the roots their objects' new addresses:
    /// `forward` then rewrites a slot naming a stub and does nothing else.
    bool fixing_roots_ = false;

    /// Note a store into a marked object during a concurrent mark: the object
    /// goes on the re-walk list (`mark_log_`, drained by the finalize) and on
    /// the helpers' shared stack, so its newly uncovered subgraph is drawn in
    /// the background rather than at the finalize. The caller -- `remember_if_old`
    /// -- has already checked that a mark is running and the object is marked.
    void log_mutation(Obj* dst);

    /// True while a concurrent mark's helpers are out. Nothing else in the
    /// heap may collect then, which is what `should_collect` enforces.
    bool marking_ = false;
    /// Set by a helper the moment its drain returns, meaning the mark's last
    /// batch was the last anywhere: the finalize has nothing further to gain
    /// by waiting. Written by the helpers, read by the process's own thread.
    std::atomic<bool> mark_done_{false};
    /// The roots, snapshotted at the safepoint where `start_concurrent_mark`
    /// was called. The helpers see the heap as it was then, not as it becomes
    /// while they work; young values in the snapshot are skipped and are
    /// re-found by the finalize's walk of the live roots. Kept until the
    /// finalize, because the values answer for objects the helpers still hold.
    std::vector<Value> mark_roots_;
    /// Old objects a store reached while a mark was running and the helpers
    /// had moved past. The finalize scans these once more.
    std::vector<Obj*> mark_log_;
    /// Old objects allocated during a mark. They are born marked -- grey would
    /// let a helper scan a half-built body -- and each carries a fresh young
    /// payload written after its header was published, so marking them while
    /// still building would miss the point. The finalize scans these once, at
    /// the end, when the fills are all done.
    std::vector<Obj*> mark_born_;
    /// The concurrent mark's round, alive from `start_concurrent_mark` until
    /// the finalize's join. It must outlive `start_concurrent_mark`'s stack,
    /// and the finalize may be many reductions later.
    std::unique_ptr<GcRound> mark_round_;
    /// The mark's body, bound to the heap and the round. Same lifetime as
    /// `mark_round_`: the pool calls it on another thread, so it must be a
    /// member rather than a stack object the start function would leave.
    std::function<void(unsigned, unsigned)> mark_body_;
    uint64_t concurrent_marks_ = 0;
    uint64_t concurrent_nanos_ = 0;
    /// The clock reading taken when the helpers were launched; the overlap
    /// window is the difference to the start of the finalize.
    uint64_t concurrent_started_ = 0;
    /// Nonzero while `census_collect` runs: the round a shared object met by
    /// the trace is seen in. Read by every marking thread, written only
    /// between collections.
    uint64_t observe_round_ = 0;
};

/// The size in bytes of an object, from its header.
inline size_t object_size(const Obj* o) { return o->bytes; }

/// True for an object in the shared area. The byte is read atomically because
/// it may be an object of another process's heap whose concurrent mark is
/// setting a bit of the same byte right now; the answer for this bit cannot
/// change either way.
inline bool is_shared_obj(const Obj* o) {
    return std::atomic_ref<uint8_t>(const_cast<Obj*>(o)->gc).load(std::memory_order_relaxed) &
           GC_SHARED;
}

/// The hash a map gives a key it compares by identity. See `AUX_IDENTITY`.
///
/// Assigned the first time anything asks, by the process that owns the object
/// -- only the owner ever writes a header, and `aux` is not a byte a collector
/// thread reads -- and kept from then on. An object in the shared area cannot
/// be written by anybody, but it cannot move either, so one with no identity of
/// its own is hashed by its address, which is as stable there as the bits are
/// everywhere else. The numbers come from a counter per thread mixed into
/// fifteen bits; nothing needs them to differ between threads, only to spread.
inline uint64_t identity_hash(Obj* o) {
    if (uint16_t id = o->aux & AUX_IDENTITY) return id;
    if (is_shared_obj(o)) return reinterpret_cast<uint64_t>(o);
    thread_local uint32_t next = 0;
    uint32_t mixed = ++next * 0x9e3779b9u;
    uint16_t id = uint16_t(((mixed >> 17) % 0x7fffu + 1) << 1);
    o->aux |= id;
    return id;
}

/// Values every process of a runtime can read and none of them owns.
///
/// Processes share nothing -- a value crossing between two is copied, which is
/// what lets each heap collect on its own -- and for a message that is the
/// right bargain. It is the wrong one for a large table many processes only
/// read: a compiler handing its resolution to four workers paid for four
/// copies of it, 0.43 s of a self-compile, before any of them did anything.
///
/// So a value can be copied here once instead (`vm.share!`), and from then on
/// it crosses by pointer. What makes that sound is that nothing here can
/// change and nothing here points out:
///
///   - A value is forced all the way down before it is copied, and the copy
///     refuses a suspension (a thunk, a frame) rather than bringing one: a
///     thunk is overwritten when it is forced, and a write by one process into
///     memory another is reading is exactly what per-process heaps exist to
///     rule out. Every object here is born with `AUX_DEEP_FORCED`, which is
///     what keeps `force_deep` from so much as touching its header.
///   - Every object carries `GC_SHARED` and nothing else in its `gc` byte, and
///     every collector stops at it (`forward_in`, `mark_object`,
///     `claim_mark`). A heap's objects may point in; nothing here points out,
///     because everything reachable from a shared object was copied in with
///     it. So no heap's collection ever has a reason to look inside.
///   - Nothing is freed until nothing can reach it, and nothing is even looked
///     at until a program says it is done with something (`release`). Each
///     `share` copies into a **region** of its own; a released region is
///     *condemned*, and a census -- every heap's major, every message, every
///     finished process -- frees the condemned regions nobody turned out to
///     hold. See "Freeing what was shared" in docs/gc.md for why each step is
///     there.
struct SharedRegion {
    SharedArea* area = nullptr;
    std::vector<uint8_t*> blocks;
    uint8_t* cursor = nullptr;
    size_t left = 0;
    size_t next_block = 0;
    size_t bytes = 0;
    std::atomic<bool> condemned{false};
    /// The last round started when this was condemned: only a later round
    /// asked every process about it.
    uint64_t condemned_at = 0;
    /// The last round a census met it in.
    std::atomic<uint64_t> seen_round{0};
    /// Regions this one points into, which it keeps alive; and how many live
    /// regions point into this one.
    std::vector<SharedRegion*> deps;
    uint32_t dependents = 0;
};

class SharedArea {
public:
    SharedArea() = default;
    ~SharedArea();
    SharedArea(const SharedArea&) = delete;
    SharedArea& operator=(const SharedArea&) = delete;

    /// Copy an already deeply forced value in, answering the shared copy. A
    /// value holding a suspension answers false with `*why` saying so, and
    /// nothing it allocated is used. Objects already shared are not copied
    /// again, so sharing a table that contains a shared table costs only the
    /// new part.
    bool share(Value v, Value* out, std::string* why);
    /// Bytes held now: handed out and not yet freed.
    size_t bytes() const { return bytes_.load(std::memory_order_relaxed); }

    /// Condemn the region `v` was shared into. Answers false for a value that
    /// is not a shared object. Frees nothing by itself: a census does that.
    bool release(Value v);
    /// Any condemned region still here?
    bool any_condemned();
    /// Start a census round, answering its number.
    uint64_t begin_round();
    /// A copy carried a condemned pointer while a round was running.
    void spoil() { spoiled_.store(true, std::memory_order_relaxed); }
    /// End round `round`: free every condemned region it neither met nor
    /// found depended upon, unless it was spoiled. Answers the bytes freed.
    size_t finish_round(uint64_t round);
    /// Where the copy that builds a region meets an object already shared.
    void met_shared(const Obj* o);
    /// The region an object of this area lives in, or null.
    static SharedRegion* region_of(const Obj* o);
    /// True when `p` lies in a block of some live shared area. For the heap
    /// verifier, which meets pointers it must not dereference until it knows
    /// whose they are -- including, on purpose, ones that are nobody's.
    static bool any_contains(const void* p);

    // The constructors the cross-heap copier asks a destination for. See
    // `copy_value` in heap.cpp, which is written once for both kinds.
    Value make_float(double v);
    Value make_string(const char* data, uint32_t len);
    Value make_bigstr(const char* data, uint64_t len);
    Value make_pid(uint64_t id);
    Value make_bigint(const uint64_t* limbs, uint32_t len, bool neg);
    Value make_tensor(uint32_t rank, const uint32_t* dims, uint64_t count);
    Value make_cons(Value head, Value tail);
    Value make_array(uint32_t len);
    /// A record of `len` fields, left for the caller to fill; see RecordObj.
    Value make_record(uint32_t tag, uint32_t len);
    Value make_map_branch(uint32_t nslots);
    Value make_map_leaf(uint64_t hash, Value key, Value value, Value next);
    Value make_error(Value kind, Value payload, Value where = UNIT);
    Value make_module(uint32_t import_index, Value name);
    Value make_closure(uint32_t func, uint32_t ncaps);
    Value make_pap(Value fn, uint32_t nargs);
    /// What the copier calls on meeting something that cannot be shared.
    void refuse(const char* what) { if (refused_.empty()) refused_ = what; }

private:
    Obj* alloc(ObjType type, size_t extra);
    void free_region(SharedRegion* r);
    std::mutex mutex_;
    std::vector<std::unique_ptr<SharedRegion>> regions_;
    /// The region the share in progress copies into.
    SharedRegion* building_ = nullptr;
    std::atomic<size_t> bytes_{0};
    std::atomic<bool> spoiled_{false};
    uint64_t rounds_ = 0;
    std::string refused_;
};

}  // namespace dream