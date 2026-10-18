// Unit tests for the VM's building blocks. End-to-end behaviour is covered by
// tests/e2e.sh, which compiles real Dream programs and checks their output.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "arithmetic.hpp"
#include "builtins.hpp"
#include "gc_pool.hpp"
#include "heap.hpp"
#include "image.hpp"
#include "process.hpp"
#include "runtime.hpp"
#include "tensor_kernels.hpp"
#include "tls.hpp"
#include "tls/tls_fixtures.hpp"
#include "value.hpp"

using namespace dream;

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++checks;                                                              \
        if (!(cond)) {                                                         \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        ++checks;                                                              \
        auto va = (a);                                                         \
        auto vb = (b);                                                         \
        if (!(va == vb)) {                                                     \
            std::printf("  FAIL %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static void test_value_tagging() {
    std::printf("value tagging\n");
    for (int64_t n : {int64_t(0), int64_t(1), int64_t(-1), int64_t(42), int64_t(-99999),
                      (int64_t(1) << 61), -(int64_t(1) << 61)}) {
        Value v = make_fixnum(n);
        CHECK(is_fixnum(v));
        CHECK(!is_ptr(v));
        CHECK(!is_imm(v));
        CHECK_EQ(fixnum_value(v), n);
    }

    CHECK(is_unit(UNIT));
    CHECK(is_bool(TRUE_V));
    CHECK(is_bool(FALSE_V));
    CHECK(truthy(TRUE_V));
    CHECK(!truthy(FALSE_V));
    CHECK(is_nil(NIL));
    CHECK(!is_fixnum(UNIT));
    CHECK(!is_ptr(UNIT));

    CHECK(is_atom(make_atom(7)));
    CHECK_EQ(imm_payload(make_atom(7)), 7u);
    CHECK(is_char(make_char(0x1F600)));
    CHECK_EQ(imm_payload(make_char(0x1F600)), 0x1F600u);
    CHECK(is_builtin(make_builtin(3)));

    // Distinct immediates must never collide.
    CHECK(UNIT != NIL);
    CHECK(TRUE_V != FALSE_V);
    CHECK(make_atom(0) != make_char(0));
    CHECK(make_atom(0) != UNIT);
}

static void test_heap_alloc() {
    std::printf("heap allocation\n");
    Heap h(4096);
    Value f = h.make_float(2.5);
    CHECK(is_ptr(f));
    CHECK(is_obj(f, ObjType::Float));
    CHECK_EQ(static_cast<FloatObj*>(as_obj(f))->value, 2.5);

    Value s = h.make_string("hello", 5);
    CHECK(is_obj(s, ObjType::Str));
    CHECK_EQ(static_cast<StrObj*>(as_obj(s))->len, 5u);
    CHECK(std::memcmp(static_cast<StrObj*>(as_obj(s))->data(), "hello", 5) == 0);

    // A null pointer reserves space for the caller to fill; concatenation
    // depends on it, and getting it wrong reads from address zero.
    Value blank = h.make_string(nullptr, 4);
    CHECK_EQ(static_cast<StrObj*>(as_obj(blank))->len, 4u);
    std::memcpy(static_cast<StrObj*>(as_obj(blank))->data(), "abcd", 4);
    CHECK(std::memcmp(static_cast<StrObj*>(as_obj(blank))->data(), "abcd", 4) == 0);

    // Allocation past the initial block must chain, not corrupt.
    std::vector<Value> many;
    for (int i = 0; i < 2000; ++i) many.push_back(h.make_float(double(i)));
    for (int i = 0; i < 2000; ++i) {
        CHECK_EQ(static_cast<FloatObj*>(as_obj(many[size_t(i)]))->value, double(i));
    }
}

/// A root source that holds a fixed list of values, so the collector can be
/// tested without a whole process.
struct VectorRoots : RootSource {
    std::vector<Value> values;
    void visit_roots(Heap& h) override {
        for (Value& v : values) h.forward(&v);
    }
};

static void test_gc_keeps_live_and_drops_dead() {
    std::printf("garbage collection\n");
    Heap h(4096);
    VectorRoots roots;

    // A list of 100 floats, held live.
    Value list = NIL;
    for (int i = 0; i < 100; ++i) list = h.make_cons(h.make_float(double(i)), list);
    roots.values.push_back(list);

    // Ten thousand unreachable floats.
    for (int i = 0; i < 10000; ++i) (void)h.make_float(double(i));

    size_t before = h.bytes_allocated();
    h.collect(roots);
    CHECK(h.bytes_live() < before / 2);
    CHECK_EQ(h.collections(), uint64_t(1));

    // The live list must have survived intact, with its contents in order.
    Value cur = roots.values[0];
    for (int i = 99; i >= 0; --i) {
        CHECK(is_obj(cur, ObjType::Cons));
        auto* c = static_cast<ConsObj*>(as_obj(cur));
        CHECK_EQ(static_cast<FloatObj*>(as_obj(c->head))->value, double(i));
        cur = c->tail;
    }
    CHECK(is_nil(cur));
}

static void test_gc_preserves_sharing() {
    std::printf("garbage collection preserves sharing\n");
    Heap h(4096);
    VectorRoots roots;
    Value shared = h.make_float(1.25);
    // The same object referenced twice must still be one object afterwards,
    // or thunk update would stop being observable to every sharer.
    roots.values.push_back(h.make_cons(shared, shared));
    h.collect(roots);

    auto* c = static_cast<ConsObj*>(as_obj(roots.values[0]));
    CHECK_EQ(c->head, c->tail);
    CHECK_EQ(static_cast<FloatObj*>(as_obj(c->head))->value, 1.25);
}

static void test_gc_collapses_indirections() {
    std::printf("garbage collection collapses indirections\n");
    Heap h(4096);
    VectorRoots roots;
    Value target = h.make_float(7.0);
    Value ind = h.make_thunk(0, UNIT);
    as_obj(ind)->type = ObjType::Indirect;
    static_cast<IndirectObj*>(as_obj(ind))->target = target;
    roots.values.push_back(h.make_cons(ind, NIL));

    h.collect(roots);
    auto* c = static_cast<ConsObj*>(as_obj(roots.values[0]));
    CHECK(is_obj(c->head, ObjType::Float));  // the hop is gone, not just followed
}

static void test_gc_handles_cycles() {
    std::printf("garbage collection handles cycles\n");
    Heap h(4096);
    VectorRoots roots;
    Value a = h.make_cons(UNIT, NIL);
    Value b = h.make_cons(UNIT, a);
    static_cast<ConsObj*>(as_obj(a))->tail = b;  // a -> b -> a
    roots.values.push_back(a);
    h.collect(roots);

    Value ra = roots.values[0];
    Value rb = static_cast<ConsObj*>(as_obj(ra))->tail;
    CHECK_EQ(static_cast<ConsObj*>(as_obj(rb))->tail, ra);
}

/// A block that the sweep finds mostly dead is emptied by the next major: its
/// survivors are copied into the holes of other blocks and the block is handed
/// back. What has to come through is everything a promotion has to keep --
/// values, sharing, the references between the survivors -- and the roots have
/// to name the copies.
static void test_major_evacuates_sparse_blocks() {
    std::printf("a major evacuates sparse blocks\n");
    Heap h(4096);
    VectorRoots roots;

    // A chain of kept cells with nineteen others between each pair of them,
    // every one rooted, so a first collection tenures them all densely.
    const int n = 4000;
    Value prev_kept = NIL;
    for (int i = 0; i < n; ++i) {
        if (i % 20 == 0) {
            prev_kept = h.make_cons(h.make_float(double(i)), prev_kept);
            roots.values.push_back(prev_kept);
        } else {
            roots.values.push_back(h.make_cons(h.make_float(double(i)), NIL));
        }
    }
    h.collect(roots);
    CHECK_EQ(h.blocks_evacuated(), uint64_t(0));

    // Drop all but the kept cells, and root the newest of them twice: one
    // object reached two ways must still be one object after it moves.
    std::vector<Value> kept;
    for (int i = 0; i < n; i += 20) kept.push_back(roots.values[size_t(i)]);
    roots.values = kept;
    roots.values.push_back(kept.back());
    std::vector<Value> addresses = roots.values;

    // The first of these sweeps finds the blocks sparse and chooses them; the
    // second empties them.
    h.collect(roots);
    h.collect(roots);
    CHECK(h.blocks_evacuated() > 0);
    CHECK(h.bytes_evacuated() > 0);

    size_t moved = 0;
    for (size_t k = 0; k < kept.size(); ++k) {
        if (roots.values[k] != addresses[k]) ++moved;
        // The chain still runs from each kept cell to every one before it.
        Value cur = roots.values[k];
        for (size_t j = k + 1; j-- > 0;) {
            CHECK(is_obj(cur, ObjType::Cons));
            auto* c = static_cast<ConsObj*>(as_obj(cur));
            CHECK_EQ(static_cast<FloatObj*>(as_obj(c->head))->value, double(j * 20));
            cur = c->tail;
            // A cell's tail is the cell kept before it, which is rooted too.
            if (j > 0) CHECK_EQ(cur, roots.values[j - 1]);
        }
        CHECK(is_nil(cur));
    }
    CHECK(moved > 0);
    CHECK_EQ(roots.values.back(), roots.values[kept.size() - 1]);
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_minor_collection_promotes_reachable() {
    std::printf("minor collection promotes the reachable young\n");
    Heap h(4096);
    VectorRoots roots;

    // All young. A chain of conses is held live; thousands of floats are not.
    Value list = NIL;
    for (int i = 0; i < 50; ++i) list = h.make_cons(h.make_float(double(i)), list);
    for (int i = 0; i < 5000; ++i) (void)h.make_float(double(i));
    roots.values.push_back(list);

    CHECK(as_obj(list)->gc & GC_YOUNG);
    h.minor_collect(roots);
    CHECK_EQ(h.minor_collections(), uint64_t(1));
    CHECK_EQ(h.major_collections(), uint64_t(0));

    // The reachable chain survived, now tenured, in order.
    Value cur = roots.values[0];
    for (int i = 49; i >= 0; --i) {
        CHECK(is_ptr(cur) && (as_obj(cur)->gc & GC_OLD));
        CHECK(is_obj(static_cast<ConsObj*>(as_obj(cur))->head, ObjType::Float));
        CHECK_EQ(static_cast<FloatObj*>(as_obj(
            static_cast<ConsObj*>(as_obj(cur))->head))->value, double(i));
        cur = static_cast<ConsObj*>(as_obj(cur))->tail;
    }
    CHECK(is_nil(cur));
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_minor_collection_leaves_old_space_alone() {
    std::printf("minor collection leaves old space alone\n");
    Heap h(4096);
    VectorRoots roots;

    // Tenure a chain, then drop every reference to it.
    Value keep = NIL;
    for (int i = 0; i < 10; ++i) keep = h.make_cons(h.make_float(double(i)), keep);
    roots.values.push_back(keep);
    h.minor_collect(roots);
    Value kept = roots.values[0];
    roots.values.clear();

    // Reclaiming old objects is a major's job. A minor sweeps nothing, so the
    // dead chain survives untouched.
    uint64_t majors_before = h.major_collections();
    h.minor_collect(roots);
    CHECK_EQ(h.major_collections(), majors_before);
    CHECK(h.owns(as_obj(kept), object_size(as_obj(kept))));

    // A major actually lets it go: the live set falls to nothing, and the
    // following trace shows a heap with no holes in it.
    h.collect(roots);
    CHECK_EQ(h.major_collections(), majors_before + 1);
    CHECK_EQ(h.bytes_live(), size_t(0));
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_write_barrier_keeps_old_to_young() {
    std::printf("write barrier keeps an old-to-young edge live\n");
    Heap h(4096);
    VectorRoots roots;

    // An old array, tenured by a minor while still empty. The root is
    // rewritten to the promoted copy; the local taken before stayed put.
    Value arr = h.make_array(1);
    roots.values.push_back(arr);
    h.minor_collect(roots);
    arr = roots.values[0];
    CHECK(as_obj(arr)->gc & GC_OLD);

    // A fresh young float that only the old array points at -- exactly the
    // edge only the remembered set knows about.
    Value fresh = h.make_float(9.5);
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    a->items()[0] = fresh;                  // the mutator's in-place store
    h.remember_if_old(as_obj(arr), fresh);  // and the barrier it must call
    CHECK_EQ(h.verify(roots), std::string());

    h.minor_collect(roots);

    // The float survived, promoted, and the store now points at the copy.
    // Without the barrier the array would have been scanned from the roots
    // alone, never seen the float, and lost the reference.
    Value slot = static_cast<ArrayObj*>(as_obj(arr))->items()[0];
    CHECK(is_obj(slot, ObjType::Float));
    CHECK(as_obj(slot)->gc & GC_OLD);
    CHECK_EQ(static_cast<FloatObj*>(as_obj(slot))->value, 9.5);
    CHECK_EQ(h.verify(roots), std::string());
}

// ---------------------------------------------------------------------------
// The parallel collector
//
// A collection is divided across the helper threads only when there is enough
// of it to be worth the handshake, so these heaps are deliberately large: a
// smaller one would quietly test the single-threaded collector a second time.
// When the pool has no threads to give -- one core, or DREAM_GC_THREADS=1 --
// there is nothing here to test and the case says so rather than passing.
// ---------------------------------------------------------------------------

/// True when the pool can divide a collection at all.
static bool parallel_collection_available(const char* what) {
    if (GcPool::instance().capacity() >= 2) return true;
    std::printf("  skipped (%s): the pool has no threads to give\n", what);
    return false;
}

static void test_parallel_collection_preserves_sharing() {
    std::printf("a parallel collection preserves sharing\n");
    if (!parallel_collection_available("parallel sharing")) return;

    Heap h(64 * 1024);
    VectorRoots roots;

    // The shape here is the whole point, so it is worth saying what it is for.
    //
    // Several threads must reach the *same young object at the same moment*,
    // or the claim in `promote` is never tested: whoever gets there first
    // promotes it and everyone after finds a forwarding pointer, which is the
    // easy case. A list will not do it -- scanning one cell yields exactly one
    // more, so a chain is traced by one thread however many are watching.
    //
    // So: a few hundred floats, and many arrays that each name all of them in
    // the same order. Every array is a separate piece of work, the trace hands
    // them out as it finds them, and the threads then walk in step through the
    // same few hundred objects. If two of them may copy one object, they will.
    const uint32_t kFloats = 400;
    const int kArrays = 600;

    std::vector<Value> floats;
    for (uint32_t i = 0; i < kFloats; ++i) floats.push_back(h.make_float(double(i)));
    for (int a = 0; a < kArrays; ++a) {
        Value arr = h.make_array(kFloats);
        for (uint32_t i = 0; i < kFloats; ++i)
            static_cast<ArrayObj*>(as_obj(arr))->items()[i] = floats[i];
        roots.values.push_back(arr);
    }

    h.minor_collect(roots);
    CHECK(h.parallel_rounds() > 0);

    // One object copied twice would be two objects: the arrays that named it
    // would come back pointing at different floats, and a thunk forced through
    // one of them would still look unforced through the other. Every array
    // must agree with the first, slot for slot.
    //
    // This checks the *outcome*, which is what a test can check. It cannot be
    // relied on to catch the claim in `promote` going missing: the window
    // between reading an object's generation bits and publishing its copy is
    // tens of nanoseconds, and losing a race that narrow on purpose is not
    // something a test can arrange. What catches that is `just test-races` --
    // ThreadSanitizer reports the missing claim as the data race it is, on the
    // first collection it sees.
    auto* first = static_cast<ArrayObj*>(as_obj(roots.values[0]));
    for (int a = 1; a < kArrays; ++a) {
        auto* arr = static_cast<ArrayObj*>(as_obj(roots.values[size_t(a)]));
        for (uint32_t i = 0; i < kFloats; ++i) {
            if (arr->items()[i] != first->items()[i]) {
                CHECK_EQ(arr->items()[i], first->items()[i]);
                a = kArrays;  // one report is enough
                break;
            }
        }
    }
    // And each of them is still the float it was, tenured.
    for (uint32_t i = 0; i < kFloats; ++i) {
        Value v = first->items()[i];
        CHECK(is_obj(v, ObjType::Float));
        CHECK(as_obj(v)->gc & GC_OLD);
        CHECK_EQ(static_cast<FloatObj*>(as_obj(v))->value, double(i));
    }
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_parallel_major_collects_across_threads() {
    std::printf("a parallel major collection marks and sweeps across threads\n");
    if (!parallel_collection_available("parallel major")) return;

    Heap h(64 * 1024);
    VectorRoots roots;

    // Enough live data to be worth dividing, with a cycle in it -- the mark
    // has to claim each object exactly once however many threads reach it --
    // and enough garbage beside it that the sweep has something to reclaim.
    Value keep = NIL;
    for (int i = 0; i < 60000; ++i) keep = h.make_cons(h.make_float(double(i)), keep);
    Value cycle = h.make_cons(UNIT, NIL);
    static_cast<ConsObj*>(as_obj(cycle))->tail = cycle;
    roots.values.push_back(keep);
    roots.values.push_back(cycle);
    for (int i = 0; i < 60000; ++i) (void)h.make_cons(h.make_float(double(i)), NIL);

    h.major_collect(roots);
    CHECK(h.parallel_rounds() > 0);
    CHECK_EQ(h.major_collections(), uint64_t(1));
    CHECK_EQ(h.verify(roots), std::string());

    // Everything kept is there, in order, and nothing else is: what a major
    // reports live is the chain and the cycle and no part of the garbage.
    Value cur = roots.values[0];
    for (int i = 59999; i >= 0; --i) {
        auto* cell = static_cast<ConsObj*>(as_obj(cur));
        CHECK_EQ(static_cast<FloatObj*>(as_obj(cell->head))->value, double(i));
        cur = cell->tail;
    }
    CHECK(is_nil(cur));
    Value c = roots.values[1];
    CHECK_EQ(static_cast<ConsObj*>(as_obj(c))->tail, c);

    // 60000 cells and 60000 floats, tenured; the garbage was the same again
    // and is gone. The bound is loose on purpose -- what matters is that the
    // sweep freed a whole generation of it rather than a little.
    size_t live = h.bytes_live();
    CHECK(live > 60000 * 2 * 16);
    CHECK(live < 60000 * 2 * 48);

    // A second major with nothing held finds the whole heap dead.
    roots.values.clear();
    h.major_collect(roots);
    CHECK_EQ(h.bytes_live(), size_t(0));
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_declined_concurrent_mark() {
    std::printf("declined concurrent marking rolls back root marks\n");
    if (GcPool::instance().capacity() < 2) return;
#ifdef _WIN32
    _putenv_s("DREAM_GC_CONCURRENT", "1");
#else
    setenv("DREAM_GC_CONCURRENT", "1", 1);
#endif
    Heap h(64 * 1024);
    VectorRoots roots;
    roots.values.push_back(h.make_cons(h.make_float(42.0), NIL));
    roots.values.push_back(h.make_float(7.0));
    h.minor_collect(roots);
    // Duplicate roots must not change rollback behavior.
    roots.values.push_back(roots.values[0]);
    std::atomic<unsigned> idle{0};
    GcPool::instance().set_idle_hint(&idle);
    CHECK(!h.start_concurrent_mark(roots));
    for (Value v : roots.values) CHECK(!(as_obj(v)->gc & GC_MARK));
    h.major_collect(roots);
    GcPool::instance().set_idle_hint(nullptr);
    CHECK_EQ(h.verify(roots), std::string());
    auto* cell = static_cast<ConsObj*>(as_obj(roots.values[0]));
    CHECK(is_obj(cell->head, ObjType::Float));
    CHECK_EQ(static_cast<FloatObj*>(as_obj(cell->head))->value, 42.0);
    CHECK_EQ(roots.values[0], roots.values[2]);
}

static void test_concurrent_mark_period() {
    std::printf("a concurrent mark overlaps the mutator\n");
    if (GcPool::instance().capacity() < 2) {
        std::printf("  skipped (concurrent mark): the pool has no threads to give\n");
        return;
    }
    // The mode is read once, by the first `start_concurrent_mark` call, so the
    // knob must be set before this test's own call -- which is also the
    // binary's first. Forcing it is what makes the size gate below not matter.
#ifdef _WIN32
    _putenv_s("DREAM_GC_CONCURRENT", "1");
#else
    setenv("DREAM_GC_CONCURRENT", "1", 1);
#endif

    Heap h(64 * 1024);
    VectorRoots roots;

    // Tenure a wide graph first: a concurrent mark is a *major*'s marking, so
    // there has to be an old graph for the helpers to draw. The shape is the
    // same as the parallel-sharing test's -- many arrays naming the same
    // floats -- because the mark divides across threads at exactly these
    // widths.
    const uint32_t kFloats = 200;
    const int kArrays = 300;
    std::vector<Value> floats;
    for (uint32_t i = 0; i < kFloats; ++i) floats.push_back(h.make_float(double(i)));
    for (int a = 0; a < kArrays; ++a) {
        Value arr = h.make_array(kFloats);
        for (uint32_t i = 0; i < kFloats; ++i)
            static_cast<ArrayObj*>(as_obj(arr))->items()[i] = floats[i];
        roots.values.push_back(arr);
    }
    h.minor_collect(roots);

    // Everything that survived is old; grow a little more on top of it so the
    // graph is not its own seed's shadow.
    std::vector<Value> extra;
    for (int a = 0; a < 8; ++a) {
        Value arr = h.make_array(2);
        static_cast<ArrayObj*>(as_obj(arr))->items()[0] = roots.values[size_t(a)];
        static_cast<ArrayObj*>(as_obj(arr))->items()[1] = h.make_float(-1.0);
        extra.push_back(arr);
        roots.values.push_back(arr);
    }

    CHECK_EQ(h.marking(), false);
    CHECK(h.start_concurrent_mark(roots));
    CHECK_EQ(h.marking(), true);

    // The helpers are walking the graph while the mutator keeps going: more
    // old-to-young stores -- the barrier logs each one -- and young garbage
    // for the finalize to be wrong about if it guesses. Indices are stepped
    // from 5 so that array 0 is never written: the check below walks it whole.
    // Both the store and the barrier are the runtime's own spellings: a plain
    // store here would race the helpers' atomic read of the same slot, which
    // is precisely the race `value_slot_store` exists to make unused.
    for (int a = 5; a < kArrays; a += 7) {
        auto* arr = static_cast<ArrayObj*>(as_obj(roots.values[size_t(a)]));
        Value fresh = h.make_cons(make_fixnum(a), h.make_float(0.5));
        value_slot_store(&arr->items()[0], fresh);
        h.remember_if_old(arr, fresh);
        (void)h.make_cons(make_fixnum(a), NIL);  // young garbage
    }

    // Finalize; everything the delta reached must come back old.
    h.finalize_concurrent_mark(roots);
    CHECK_EQ(h.marking(), false);
    CHECK_EQ(h.concurrent_marks(), uint64_t(1));
    CHECK_EQ(h.major_collections(), uint64_t(1));

    // The original floats survived the overlap, tenured, in the same order.
    auto* first = static_cast<ArrayObj*>(as_obj(roots.values[0]));
    for (uint32_t i = 0; i < kFloats; ++i) {
        Value v = first->items()[i];
        CHECK(is_obj(v, ObjType::Float));
        CHECK(as_obj(v)->gc & GC_OLD);
        CHECK_EQ(static_cast<FloatObj*>(as_obj(v))->value, double(i));
    }

    // The stores made during the mark landed, and what they pointed at was
    // found by the barrier log and promoted rather than freed.
    for (int a = 5; a < kArrays; a += 7) {
        auto* arr = static_cast<ArrayObj*>(as_obj(roots.values[size_t(a)]));
        Value popped = arr->items()[0];
        if (!is_ptr(popped) || as_obj(popped)->gc & GC_YOUNG) {
            CHECK(is_ptr(popped));
            CHECK(!(as_obj(popped)->gc & GC_YOUNG));
            continue;
        }
        CHECK_EQ(fixnum_value(static_cast<ConsObj*>(as_obj(popped))->head),
                 int64_t(a));
    }
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_heap_verifier_accepts_a_healthy_heap() {
    std::printf("heap verifier accepts a healthy heap\n");
    Heap h(4096);
    VectorRoots roots;
    Value list = NIL;
    for (int i = 0; i < 20; ++i) {
        list = h.make_cons(h.make_string("item", 4), list);
    }
    Value arr = h.make_array(3);
    for (uint32_t i = 0; i < 3; ++i) static_cast<ArrayObj*>(as_obj(arr))->items()[i] = list;
    Value map = h.make_map(8);
    roots.values = {list, arr, map, h.make_float(1.0)};
    CHECK_EQ(h.verify(roots), std::string());

    // And it must still be happy after a collection.
    h.collect(roots);
    CHECK_EQ(h.verify(roots), std::string());
}

static void test_shared_area() {
    std::printf("shared area\n");

    SharedArea area;
    Heap h(4096);
    // A list of a string and a map, forced -- everything here is a value.
    Value m = h.make_map_branch(0);
    Value str = h.make_string("shared", 6);
    Value list = h.make_cons(str, h.make_cons(m, NIL));
    Value shared = UNIT;
    std::string why;
    CHECK(area.share(list, &shared, &why));
    CHECK(is_ptr(shared) && is_shared_obj(as_obj(shared)));
    CHECK(as_obj(shared)->aux & AUX_DEEP_FORCED);
    CHECK(!h.owns(as_obj(shared), sizeof(Obj)));
    CHECK(SharedArea::any_contains(as_obj(shared)));
    CHECK(area.bytes() > 0);

    // Crossing heaps, a shared value is the same pointer, and so is anything
    // that reaches it -- only the part that is not shared is copied.
    Heap other(4096);
    CHECK(Heap::copy_between(other, shared) == shared);
    Value wrapped = h.make_cons(shared, NIL);
    Value moved = Heap::copy_between(other, wrapped);
    CHECK(moved != wrapped);
    CHECK(static_cast<ConsObj*>(as_obj(moved))->head == shared);

    // Sharing what is already shared copies nothing.
    size_t before = area.bytes();
    Value again = UNIT;
    CHECK(area.share(shared, &again, &why));
    CHECK(again == shared && area.bytes() == before);

    // A heap that points at it collects without touching it: the verifier
    // accepts the edge, and a full collection neither marks nor frees it.
    VectorRoots roots;
    roots.values = {wrapped};
    CHECK(h.verify(roots).empty());
    h.collect(roots);
    CHECK(as_obj(shared)->gc == GC_SHARED);
    CHECK(static_cast<ConsObj*>(as_obj(resolve(roots.values[0])))->head == shared);
    CHECK(h.verify(roots).empty());

    // A suspension is refused, not shared.
    Value th = h.make_thunk(0, UNIT);
    Value out = UNIT;
    CHECK(!area.share(h.make_cons(th, NIL), &out, &why));
    CHECK(why.find("thunk") != std::string::npos);
}

static void test_heap_verifier_catches_corruption() {
    std::printf("heap verifier catches corruption\n");

    {   // A pointer that does not land in this heap.
        Heap h(4096);
        VectorRoots roots;
        Value cell = h.make_cons(UNIT, NIL);
        roots.values = {cell};
        static_cast<ConsObj*>(as_obj(cell))->head = 0x1000;  // aligned but foreign
        CHECK(h.verify(roots).find("does not land in this heap") != std::string::npos);
    }
    {   // A header claiming a size that runs past its block.
        Heap h(4096);
        VectorRoots roots;
        Value f = h.make_float(1.0);
        roots.values = {f};
        as_obj(f)->bytes = 1u << 20;
        CHECK(h.verify(roots).find("runs past the end") != std::string::npos);
    }
    {   // An unknown type tag.
        Heap h(4096);
        VectorRoots roots;
        Value f = h.make_float(1.0);
        roots.values = {f};
        as_obj(f)->type = static_cast<ObjType>(99);
        CHECK(h.verify(roots).find("unknown type") != std::string::npos);
    }
    {   // An array whose length does not fit its allocation. This is the shape
        // the map-growth bug took: a header describing more than it owns.
        Heap h(4096);
        VectorRoots roots;
        Value a = h.make_array(2);
        roots.values = {a};
        static_cast<ArrayObj*>(as_obj(a))->len = 1000;
        CHECK(h.verify(roots).find("too small") != std::string::npos);
    }
    {   // Tagging already rules out a misaligned pointer: such a word is not
        // classified as a pointer at all, so it is never dereferenced. The
        // verifier's alignment check is a backstop for a hand-built object.
        Heap h(4096);
        Value cell = h.make_cons(UNIT, NIL);
        CHECK(is_ptr(cell));
        CHECK(!is_ptr(cell + 4));
        VectorRoots roots;
        roots.values = {cell};
        static_cast<ConsObj*>(as_obj(cell))->head = cell + 4;
        CHECK_EQ(h.verify(roots), std::string());
    }
}

static void test_heap_verifier_follows_every_object_kind() {
    std::printf("heap verifier follows every object kind\n");
    // Each kind has its own traversal; a kind the walker forgets would hide
    // corruption underneath it, so plant a bad pointer under each in turn.
    Heap h(8192);

    {
        VectorRoots roots;
        Value cl = h.make_closure(0, 2);
        roots.values = {cl};
        static_cast<ClosureObj*>(as_obj(cl))->caps()[1] = 0x2000;
        CHECK(!h.verify(roots).empty());
    }
    {
        VectorRoots roots;
        Value fr = h.make_frame(h.make_closure(0, 0), 3);
        roots.values = {fr};
        static_cast<FrameObj*>(as_obj(fr))->slots()[2] = 0x2000;
        CHECK(!h.verify(roots).empty());
    }
    {
        VectorRoots roots;
        Value th = h.make_thunk(0, UNIT);
        roots.values = {th};
        static_cast<ThunkObj*>(as_obj(th))->frame = 0x2000;
        CHECK(!h.verify(roots).empty());
    }
    {
        VectorRoots roots;
        Value e = h.make_error(make_atom(1), UNIT);
        roots.values = {e};
        static_cast<ErrorObj*>(as_obj(e))->payload = 0x2000;
        CHECK(!h.verify(roots).empty());
    }
    {
        // A branch whose child is not a pointer to anything.
        VectorRoots roots;
        Value m = h.make_map_branch(1);
        auto* branch = static_cast<MapObj*>(as_obj(m));
        branch->bitmap = 1;
        branch->count = 1;
        branch->slots()[0] = 0x2000;
        roots.values = {m};
        CHECK(!h.verify(roots).empty());
    }
    {
        // And a leaf with a bad key, which is where a map's own values live.
        VectorRoots roots;
        Value leaf = h.make_map_leaf(1, UNIT, UNIT, NIL_SLOT);
        Value m = h.make_map_branch(1);
        auto* branch = static_cast<MapObj*>(as_obj(m));
        branch->bitmap = 1;
        branch->count = 1;
        branch->slots()[0] = leaf;
        static_cast<MapLeafObj*>(as_obj(leaf))->key = 0x2000;
        roots.values = {m};
        CHECK(!h.verify(roots).empty());
    }
    {
        VectorRoots roots;
        Value pap = h.make_pap(h.make_closure(0, 0), 1);
        roots.values = {pap};
        static_cast<PapObj*>(as_obj(pap))->args()[0] = 0x2000;
        CHECK(!h.verify(roots).empty());
    }
}

static void test_cross_heap_copy() {
    std::printf("cross-heap copying\n");
    Heap a(4096), b(4096);
    Value list = a.make_cons(a.make_float(1.0),
                             a.make_cons(a.make_string("two", 3), NIL));
    Value copied = Heap::copy_between(b, list);

    CHECK(copied != list);
    auto* c = static_cast<ConsObj*>(as_obj(copied));
    CHECK_EQ(static_cast<FloatObj*>(as_obj(c->head))->value, 1.0);
    auto* c2 = static_cast<ConsObj*>(as_obj(c->tail));
    CHECK_EQ(static_cast<StrObj*>(as_obj(c2->head))->len, 3u);

    // A cycle must not send the copier into a loop.
    Value x = a.make_cons(UNIT, NIL);
    static_cast<ConsObj*>(as_obj(x))->tail = x;
    Value cx = Heap::copy_between(b, x);
    CHECK_EQ(static_cast<ConsObj*>(as_obj(cx))->tail, cx);

    // A tensor crosses as its numbers, in one piece, with its shape.
    const uint32_t dims[2] = {2, 3};
    Value t = a.make_tensor(2, dims, 6);
    for (int i = 0; i < 6; ++i) static_cast<TensorObj*>(as_obj(t))->data()[i] = i * 1.5;
    Value ct = Heap::copy_between(b, t);
    auto* tc = static_cast<TensorObj*>(as_obj(ct));
    CHECK(ct != t);
    CHECK_EQ(tc->rank, 2u);
    CHECK_EQ(tc->dims[1], 3u);
    CHECK_EQ(tc->count, 6u);
    CHECK_EQ(tc->data()[5], 7.5);
}

/// A dead big object's block is kept and handed to the next big object that
/// fits, and what nothing reuses goes back to the system a collection later.
static void test_big_block_pool() {
    std::printf("big blocks are pooled\n");
    Heap h(8192);
    VectorRoots roots;
    const uint32_t dims[1] = {1u << 20};  // 8 MB, far past the largest class
    Obj* first = as_obj(h.make_tensor(1, dims, dims[0]));
    h.collect(roots);
    CHECK(h.bytes_pooled() >= size_t(dims[0]) * sizeof(double));
    // Unreached, so swept and pooled: the next one of the same size is it.
    Obj* second = as_obj(h.make_tensor(1, dims, dims[0]));
    CHECK_EQ(second, first);
    CHECK_EQ(h.bytes_pooled(), size_t(0));
    // A little larger still fits the same block.
    roots.values = {};
    h.collect(roots);
    const uint32_t more[1] = {dims[0] + 1000};
    CHECK_EQ(as_obj(h.make_tensor(1, more, more[0])), first);
    // Nothing reuses what this collection retires, so the next one gives it
    // back.
    h.collect(roots);
    CHECK(h.bytes_pooled() > 0);
    h.collect(roots);
    CHECK_EQ(h.bytes_pooled(), size_t(0));
}

/// The matrix product against the textbook triple loop, at sizes chosen to
/// land on every edge of the blocking: tiles of 6 x 8 cut short in each
/// direction, a K past one 256-wide panel, the one-row and one-column shapes
/// that take their own paths. Both instruction-set tables, where the machine
/// can run the second, because a CPU without AVX2 runs only the first.
static void test_tensor_kernels() {
    std::printf("tensor kernels\n");
    std::vector<const TensorKernels*> tables = tensor_kernel_tables();
    const size_t shapes[][3] = {{1, 1, 1}, {1, 7, 9}, {5, 3, 1}, {6, 8, 8},  {7, 9, 17},
                                {13, 300, 11}, {97, 31, 100}, {2, 513, 3}};
    for (const TensorKernels* k : tables) {
        for (auto& s : shapes) {
            const size_t M = s[0], K = s[1], N = s[2];
            std::vector<double> A(M * K), B(K * N), C(M * N, -1.0);
            for (size_t i = 0; i < A.size(); ++i) A[i] = double(i % 7) - 3.0;
            for (size_t i = 0; i < B.size(); ++i) B[i] = double(i % 5) * 0.5 - 1.0;
            // The same product with its operands read three ways: as stored;
            // B as the transpose of its own transpose, by strides; and A
            // computed a segment at a time by a callback. Every way must give
            // the textbook answer.
            std::vector<double> Bt(N * K);
            for (size_t p = 0; p < K; ++p)
                for (size_t j = 0; j < N; ++j) Bt[j * K + p] = B[p * N + j];
            struct Rows {
                const std::vector<double>* a;
                size_t K;
            } rows{&A, K};
            auto segment = [](void* ctx, size_t row, size_t col, size_t len, double* out) {
                auto* r = static_cast<Rows*>(ctx);
                for (size_t l = 0; l < len; ++l) out[l] = (*r->a)[row * r->K + col + l];
            };
            const GemmOperand plainA{A.data(), K, 1, nullptr, nullptr, 0};
            const GemmOperand plainB{B.data(), N, 1, nullptr, nullptr, 0};
            const GemmOperand turnedB{Bt.data(), 1, K, nullptr, nullptr, 0};
            const GemmOperand computedA{nullptr, 0, 0, segment, &rows, 0};
            const GemmOperand* as[] = {&plainA, &plainA, &computedA};
            const GemmOperand* bs[] = {&plainB, &turnedB, &plainB};
            for (int way = 0; way < 3; ++way) {
                std::fill(C.begin(), C.end(), -1.0);
                k->gemm(*as[way], *bs[way], C.data(), M, K, N, nullptr);
                double worst = 0;
                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < N; ++j) {
                        double want = 0;
                        for (size_t p = 0; p < K; ++p) want += A[i * K + p] * B[p * N + j];
                        worst = std::max(worst, std::abs(C[i * N + j] - want));
                    }
                // Small integers and halves: every product and partial sum is
                // exact, so any reordering the kernel does still gives the
                // same number.
                CHECK_EQ(worst, 0.0);
            }
        }
        double x[19], y[19], out[19];
        for (int i = 0; i < 19; ++i) {
            x[i] = i;
            y[i] = 19 - i;
        }
        CHECK_EQ(k->dot(x, y, 19), 1140.0);
        CHECK_EQ(k->reduce(KRED_SUM, x, 19), 171.0);
        CHECK_EQ(k->reduce(KRED_MAX, y, 19), 19.0);
        CHECK_EQ(k->reduce(KRED_MIN, y, 19), 1.0);
        k->binary(KOP_SUB, x, y, out, 19);
        CHECK_EQ(out[18], -1.0 * (19 - 18) + 18.0);
        k->scalar(KOP_DIV, x, 2.0, true, out, 19);
        CHECK_EQ(out[4], 0.5);
        double t[6] = {1, 2, 3, 4, 5, 6}, tt[6];
        k->transpose(t, tt, 2, 3);
        CHECK_EQ(tt[1], 4.0);
        CHECK_EQ(tt[4], 3.0);
    }
}

static void test_image_rejects_bad_input() {
    std::printf("image validation\n");
    std::string err;

    Image img;
    CHECK(!img.load_bytes(reinterpret_cast<const uint8_t*>("short"), 5, err));

    uint8_t header[32] = {};
    std::memcpy(header, "NOTMAGIC", 8);
    Image img2;
    CHECK(!img2.load_bytes(header, sizeof header, err));
    CHECK(err.find("magic") != std::string::npos);

    // Right magic, impossible version.
    uint8_t good[32] = {};
    std::memcpy(good, "DAGNCAAF", 8);
    good[8] = 99;
    Image img3;
    CHECK(!img3.load_bytes(good, sizeof good, err));
}

// ---------------------------------------------------------------------------
// Large data: the LDAT/PAYL pair
//
// Building an image by hand, because that is the only way to write the ones a
// loader must *reject* -- the compiler cannot be asked for a malformed file.
// The shape is what `dreams/src/emit.dr` writes: a 32-byte header, a table of
// 16-byte entries, then each section on an 8-byte boundary.
// ---------------------------------------------------------------------------

namespace {

struct ImageBuilder {
    struct Sec {
        const char* kind;
        std::vector<uint8_t> body;
        uint32_t count;
        /// What the table claims, when a test wants it to differ from the body.
        uint32_t declared_length;
    };
    std::vector<Sec> secs;

    void add(const char* kind, std::vector<uint8_t> body, uint32_t count) {
        uint32_t len = uint32_t(body.size());
        secs.push_back(Sec{kind, std::move(body), count, len});
    }
    /// The same, but the table lies about how long the section is.
    void add_claiming(const char* kind, std::vector<uint8_t> body, uint32_t count,
                      uint32_t declared) {
        secs.push_back(Sec{kind, std::move(body), count, declared});
    }

    static void put32(std::vector<uint8_t>& out, uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
    }
    static void put64(std::vector<uint8_t>& out, uint64_t v) {
        for (int i = 0; i < 8; ++i) out.push_back(uint8_t(v >> (8 * i)));
    }

    std::vector<uint8_t> build() const {
        const size_t table_end = 32 + secs.size() * 16;
        // Where each section lands, once the table that names them is sized.
        std::vector<uint32_t> at;
        size_t cursor = (table_end + 7) & ~size_t(7);
        for (const Sec& s : secs) {
            at.push_back(uint32_t(cursor));
            cursor = (cursor + s.body.size() + 7) & ~size_t(7);
        }

        std::vector<uint8_t> out;
        out.insert(out.end(), {'D', 'A', 'G', 'N', 'C', 'A', 'A', 'F'});
        put32(out, 0 | (1u << 16));           // version 0.1
        put32(out, 0);                        // flags
        put32(out, 0);                        // module_name
        put32(out, 0);                        // source_name
        put32(out, 0xFFFFFFFFu);              // entry: none
        put32(out, uint32_t(secs.size()));
        for (size_t i = 0; i < secs.size(); ++i) {
            const Sec& s = secs[i];
            put32(out, uint32_t(uint8_t(s.kind[0])) | (uint32_t(uint8_t(s.kind[1])) << 8) |
                           (uint32_t(uint8_t(s.kind[2])) << 16) |
                           (uint32_t(uint8_t(s.kind[3])) << 24));
            put32(out, at[i]);
            put32(out, s.declared_length);
            put32(out, s.count);
        }
        for (size_t i = 0; i < secs.size(); ++i) {
            out.resize(at[i], 0);
            out.insert(out.end(), secs[i].body.begin(), secs[i].body.end());
        }
        return out;
    }
};

/// A one-string table, which the header's module and source names point at.
/// Every image needs it; nothing here is testing strings.
void add_minimum(ImageBuilder& b) {
    std::vector<uint8_t> kstr;
    ImageBuilder::put32(kstr, 0);
    ImageBuilder::put32(kstr, 0);
    b.add("KSTR", kstr, 1);
    b.add("SBLB", {}, 0);
}

static void test_type_test_nodes() {
    std::printf("type test nodes\n");
    for (int mode = 0; mode < 6; ++mode) {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> nodes(16, 0);
        nodes[0] = uint8_t(Op::Unit);
        nodes.insert(nodes.end(), {uint8_t(Op::TypeIs), 0, 0, 0});
        ImageBuilder::put32(nodes, mode == 2 ? 2 : mode == 5 ? 1 : 0);
        ImageBuilder::put32(nodes, mode == 3 ? DREAM_TYPE_PURE_FN : DREAM_TYPE_MAP);
        ImageBuilder::put32(nodes, mode == 4 ? 2 : mode == 1 ? 1 : 0);
        b.add("NODE", nodes, 2);
        auto bytes = b.build();
        Image img;
        std::string error;
        // Valid equality/inequality; reject bad subject, kind, inversion and cycle.
        CHECK_EQ(img.load_bytes(bytes.data(), bytes.size(), error), mode < 2);
    }
}

static void test_image_frame_validation() {
    std::printf("image frame validation\n");
    // Every Get/Set operand, including a Get fallback, must be traversed.
    for (Op parent : {Op::Get, Op::Set}) {
        for (Op access : {Op::Local, Op::Capture, Op::Bind, Op::Try}) {
            for (unsigned operand = 0; operand < 3; ++operand) {
                for (bool invalid : {false, true}) {
                    ImageBuilder b;
                    add_minimum(b);
                    std::vector<uint8_t> nodes;
                    auto node = [&](Op op, uint32_t a, uint32_t c1, uint32_t c2) {
                        nodes.insert(nodes.end(), {uint8_t(op), 0, 0, 0});
                        ImageBuilder::put32(nodes, a);
                        ImageBuilder::put32(nodes, c1);
                        ImageBuilder::put32(nodes, c2);
                    };
                    node(Op::Unit, 0, 0, 0);
                    node(access, access == Op::Try ? 0 : unsigned(invalid), 0,
                         access == Op::Try ? unsigned(invalid) : 0);
                    node(parent, operand == 0 ? 1 : 0, operand == 1 ? 1 : 0,
                         operand == 2 ? 1 : 0);
                    b.add("NODE", nodes, 3);
                    std::vector<uint8_t> func(32, 0);
                    func[4] = 2; // body
                    func[12] = 1; // slots
                    func[14] = 1; // captures
                    b.add("FUNC", func, 1);
                    b.add("KIDS", std::vector<uint8_t>(4, 0), 1);
                    auto bytes = b.build();
                    Image img;
                    std::string error;
                    CHECK_EQ(img.load_bytes(bytes.data(), bytes.size(), error), !invalid);
                }
            }
        }
    }
    for (bool invalid : {false, true}) {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> nodes(16, 0);
        nodes[0] = uint8_t(Op::Unit);
        b.add("NODE", nodes, 1);
        std::vector<uint8_t> func(32, 0);
        func[8] = invalid ? 2 : 1;
        func[12] = 1;
        b.add("FUNC", func, 1);
        auto bytes = b.build();
        Image img;
        std::string error;
        CHECK_EQ(img.load_bytes(bytes.data(), bytes.size(), error), !invalid);
    }
    for (uint32_t scalar : {0xD7FFu, 0xD800u, 0xDFFFu, 0xE000u, 0x10FFFFu, 0x110000u}) {
        ImageBuilder chars;
        add_minimum(chars);
        std::vector<uint8_t> node{uint8_t(Op::ConstChar), 0, 0, 0};
        ImageBuilder::put32(node, scalar);
        ImageBuilder::put32(node, 0);
        ImageBuilder::put32(node, 0);
        chars.add("NODE", node, 1);
        auto bytes = chars.build();
        Image img;
        std::string error;
        CHECK_EQ(img.load_bytes(bytes.data(), bytes.size(), error),
                 scalar <= 0x10FFFF && (scalar < 0xD800 || scalar > 0xDFFF));
    }
    // A wide DAG with repeated children must load in linear time.
    ImageBuilder b;
    add_minimum(b);
    std::vector<uint8_t> nodes(16, 0);
    nodes[0] = uint8_t(Op::Unit);
    nodes.insert(nodes.end(), {uint8_t(Op::MakeList), 0, 0, 0});
    ImageBuilder::put32(nodes, 0);
    ImageBuilder::put32(nodes, 100000);
    ImageBuilder::put32(nodes, 0);
    b.add("NODE", nodes, 2);
    b.add("KIDS", std::vector<uint8_t>(400000, 0), 100000);
    auto bytes = b.build();
    Image img;
    std::string error;
    CHECK(img.load_bytes(bytes.data(), bytes.size(), error));
}

static void test_integer_hints() {
    std::printf("integer signature hints\n");
    for (int mode = 0; mode < 4; ++mode) {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> node(16, 0);
        node[0] = uint8_t(Op::Unit);
        b.add("NODE", node, 1);
        std::vector<uint8_t> func(32, 0);
        func[8] = 1;  // arity
        func[12] = 1; // slots
        b.add("FUNC", func, 1);
        if (mode != 0) {
            std::vector<uint8_t> hint;
            ImageBuilder::put32(hint, mode == 3 ? 1 : 0);
            ImageBuilder::put32(hint, 0);
            ImageBuilder::put64(hint, 1);
            b.add_claiming("ITYP", hint, 1, mode == 2 ? 8 : 16);
        }
        auto bytes = b.build();
        Image img;
        std::string error;
        bool ok = img.load_bytes(bytes.data(), bytes.size(), error);
        CHECK_EQ(ok, mode < 2);
        if (ok) {
            CHECK_EQ(img.integer_params(0), mode == 1 ? 1u : 0u);
            CHECK_EQ(img.integer_params(1), 0u);
            CHECK_EQ(img.float_params(0), 0u);
        }
    }
}

/// `LDAT` from a list of (offset, length), and a `PAYL` holding `bytes`.
void add_payload(ImageBuilder& b, const std::vector<std::pair<uint64_t, uint64_t>>& table,
                 const std::string& bytes) {
    std::vector<uint8_t> ldat;
    for (auto& [off, len] : table) {
        ImageBuilder::put64(ldat, off);
        ImageBuilder::put64(ldat, len);
    }
    b.add("LDAT", ldat, uint32_t(table.size()));

    std::vector<uint8_t> payl;
    ImageBuilder::put64(payl, bytes.size());
    payl.insert(payl.end(), bytes.begin(), bytes.end());
    // The section is its 8-byte header; the bytes after it are the payload and
    // are not part of what the table measures.
    b.add_claiming("PAYL", payl, 0, 8);
}

}  // namespace

static void test_payload_sections() {
    std::printf("large data\n");
    std::string err;

    // A payload that is there, correct, and reachable.
    {
        ImageBuilder b;
        add_minimum(b);
        add_payload(b, {{0, 5}, {5, 6}}, "helloworld!");
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK_EQ(img.data_count(), 2u);
        CHECK_EQ(img.data_length(0), uint64_t(5));
        CHECK_EQ(std::string(img.data_bytes(0), 5), std::string("hello"));
        CHECK_EQ(std::string(img.data_bytes(1), 6), std::string("world!"));
    }

    // An image with no payload at all is the ordinary case and must stay one.
    {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK_EQ(img.data_count(), 0u);
    }

    // Each section is meaningless without the other: a table describing
    // nothing, or bytes nothing can name.
    {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> ldat;
        ImageBuilder::put64(ldat, 0);
        ImageBuilder::put64(ldat, 4);
        b.add("LDAT", ldat, 1);
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(!img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK(err.find("without a PAYL") != std::string::npos);
    }
    {
        ImageBuilder b;
        add_minimum(b);
        std::vector<uint8_t> payl;
        ImageBuilder::put64(payl, 4);
        payl.insert(payl.end(), {'a', 'b', 'c', 'd'});
        b.add_claiming("PAYL", payl, 0, 8);
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(!img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK(err.find("without an LDAT") != std::string::npos);
    }

    // A datum reaching past the end of the payload.
    {
        ImageBuilder b;
        add_minimum(b);
        add_payload(b, {{3, 10}}, "hello");
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(!img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK(err.find("past the payload") != std::string::npos);
    }

    // The offset alone is past the end, and `offset + length` would wrap were
    // it added rather than subtracted -- which is the sum this check exists for.
    {
        ImageBuilder b;
        add_minimum(b);
        add_payload(b, {{~uint64_t(0) - 2, 8}}, "hello");
        std::vector<uint8_t> bytes = b.build();
        Image img;
        CHECK(!img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK(err.find("past the payload") != std::string::npos);
    }

    // A payload the file is too short to hold: the header says more bytes
    // follow than there are.
    {
        ImageBuilder b;
        add_minimum(b);
        add_payload(b, {{0, 5}}, "hello");
        std::vector<uint8_t> bytes = b.build();
        // Rewrite the payload's length header to claim a gigabyte.
        size_t payl_at = bytes.size() - 8 - 5;
        for (int i = 0; i < 8; ++i) {
            bytes[payl_at + i] = uint8_t((uint64_t(1) << 30) >> (8 * i));
        }
        Image img;
        CHECK(!img.load_bytes(bytes.data(), bytes.size(), err));
        CHECK(err.find("past end of file") != std::string::npos);
    }
}

static void test_bigstr_values() {
    std::printf("big strings\n");
    // The bytes are not heap memory, which is the whole point: a view can be
    // made, promoted and copied between heaps without any of them moving.
    static const char corpus[] = "the quick brown fox";
    Heap a, b;
    Value v = a.make_bigstr(corpus, 19);
    CHECK_EQ(surface_type(v), DREAM_TYPE_BIGSTR);

    Bytes bytes;
    CHECK(string_bytes(v, &bytes));
    CHECK_EQ(bytes.len, uint64_t(19));
    CHECK_EQ(bytes.data, corpus);

    // A plain string of the same bytes is equal to it and hashes with it,
    // which is what lets `str.slice payload == "quick"` mean anything.
    Value s = a.make_string(corpus, 19);
    Bytes sb;
    CHECK(string_bytes(s, &sb));
    CHECK(bytes_equal(bytes, sb));
    CHECK_EQ(bytes_hash(bytes), bytes_hash(sb));
    CHECK_EQ(bytes_compare(bytes, sb), 0);

    // Crossing to another heap copies the view, not the bytes.
    Value cv = Heap::copy_between(b, v);
    CHECK(is_obj(cv, ObjType::BigStr));
    CHECK_EQ(static_cast<BigStrObj*>(as_obj(cv))->data, corpus);
    CHECK_EQ(static_cast<BigStrObj*>(as_obj(cv))->len, uint64_t(19));
    CHECK(as_obj(cv) != as_obj(v));
}

static void test_atom_interning() {
    std::printf("atom interning\n");
    Runtime rt;
    uint32_t a = rt.intern_atom("hello");
    uint32_t b = rt.intern_atom("hello");
    uint32_t c = rt.intern_atom("world");
    CHECK_EQ(a, b);
    CHECK(a != c);
    CHECK_EQ(rt.atom_name(a), std::string("hello"));

    // The well-known atoms must exist and be distinct.
    const WellKnownAtoms& wk = well_known(rt);
    CHECK(wk.divide_by_zero != wk.type_error);
    CHECK_EQ(rt.atom_name(wk.divide_by_zero), std::string("divide_by_zero"));

    // `find_atom` answers the same id as `intern_atom` for a name that is
    // already an atom -- and, the point of it, does not create one for a name
    // that is not. The count is what says so: asking about a name the table
    // has never seen must leave the table exactly the size it was.
    uint32_t found = 0;
    CHECK(rt.find_atom("hello", &found));
    CHECK_EQ(found, a);
    const uint32_t before = rt.atom_count();
    CHECK(!rt.find_atom("nothing has ever interned this", &found));
    CHECK_EQ(rt.atom_count(), before);
    // And the id it does not find is not written over the caller's.
    CHECK_EQ(found, a);
}

static void test_builtin_table_matches_compiler() {
    std::printf("_builtin table\n");
    // The ids are baked into every image, so this order is a wire format.
    const char* expected[] = {"spawn!",  "join!",   "send!",     "recv!", "self!",
                              "raise!",  "type_of", "to_string", "len",   "strict!",
                              "_match_is_cons", "_match_head", "_match_tail",
                              "_match_at", "_match_key", "type_assert",
                              "_list_tail",
                              "_list_cons",
                              "_list_is_empty",
                              "_str_chars",
                              "_str_of_chars",
                              "_str_of_bytes",
                              "_str_concat",
                              "_error_new",
                              "_error_kind",
                              "_error_payload",
                              "_str_slice",
                              "_str_find",
                              "_str_byte",
                              "_str_le",
                              "_str_span",
                              "_str_upto",
                              "_char_code",
                              "_char_of_code",
                              "_to_float",
                              "_float_bytes",
                              "_float_of_bytes",
                              "_to_int",
                              "_parse_int",
                              "_parse_float",
                              "_to_existing_atom",
                              "_array_new",
                              "_array_of_list",
                              "_array_to_list",
                              "_map_has",
                              "_map_remove",
                              "_map_pairs",
                              "_data_count",
                              "_data_at",
                              "compare",
                              "_sort_keyed",
                              "_tensor_matmul",
                              "_str_interp",
                              "_mailbox_peek!",
                              "_mailbox_take!",
                              "_await_message!",
                              "_deadline_in!",
                              "_match_fail",
                              "_error_trace!",
                              "_tensor_of_bytes",
                              "_record_is",
                              "_record_id",
                              "_record_fields",
                              "_record_of",
                              "_pow",
                              "_str_char_count",
                              "_str_ascii_case",
                              "_str_span_back",
                              "_str_rfind",
                              "_settled",
                              "_forced",
                              "defer!",
                              "_indexed",
                              "_append_rev",
                              "_rev_onto",
                              "_str_append_rev",
                              "_set_of",
                              "_set_add",
                              "_set_lookup",
    };
    const uint32_t n = uint32_t(sizeof(expected) / sizeof(expected[0]));
    CHECK_EQ(builtin_count(), n);
    // Bounded by both, so a table that has grown past this list reports the
    // mismatch above rather than reading off the end of it.
    for (uint32_t i = 0; i < builtin_count() && i < n; ++i) {
        CHECK_EQ(std::string(builtin_def(i).name), std::string(expected[i]));
    }
    // `spawn!` must not force its argument, or spawning would run the work here.
    CHECK_EQ(builtin_def(0).strict_mask, 0u);
    CHECK_EQ(builtin_def(15).strict_mask, 1u);
#define CHECK_PRIMITIVE(name, code, builtin, nargs) \
    CHECK_EQ(primitive_builtin(Op::name), uint32_t(builtin)); \
    CHECK_EQ(builtin_def(builtin).arity, uint32_t(nargs));
    DREAM_PRIMITIVES(CHECK_PRIMITIVE)
#undef CHECK_PRIMITIVE

}

static void test_maps() {
    std::printf("maps\n");
    Runtime rt;
    auto proc = rt.spawn_process();
    Process& p = *proc;

    // A put answers a new map and leaves the one it was given alone. The
    // caller keeps what comes back -- that is the whole contract, and the
    // reason a map can be shared without being copied.
    Value m = p.heap().make_map(0);
    for (int i = 0; i < 200; ++i) {
        m = map_insert(p, m, make_fixnum(i), make_fixnum(i * 2));
    }
    CHECK_EQ(static_cast<MapObj*>(as_obj(m))->count, 200u);
    for (int i = 0; i < 200; ++i) {
        Value out;
        CHECK(map_lookup(p, m, make_fixnum(i), &out));
        CHECK_EQ(fixnum_value(out), int64_t(i * 2));
    }
    Value missing;
    CHECK(!map_lookup(p, m, make_fixnum(9999), &missing));

    // The map put into is untouched: the older version still answers as it did,
    // and does not see the newer one's entry.
    Value grown = map_insert(p, m, make_fixnum(1000), make_fixnum(7));
    CHECK_EQ(static_cast<MapObj*>(as_obj(m))->count, 200u);
    CHECK_EQ(static_cast<MapObj*>(as_obj(grown))->count, 201u);
    Value peek;
    CHECK(!map_lookup(p, m, make_fixnum(1000), &peek));
    CHECK(map_lookup(p, grown, make_fixnum(1000), &peek));

    // Replacing a key changes the value without changing the size.
    Value replaced = map_insert(p, m, make_fixnum(5), make_fixnum(99));
    CHECK_EQ(static_cast<MapObj*>(as_obj(replaced))->count, 200u);
    CHECK(map_lookup(p, replaced, make_fixnum(5), &peek));
    CHECK_EQ(fixnum_value(peek), int64_t(99));
    CHECK(map_lookup(p, m, make_fixnum(5), &peek));
    CHECK_EQ(fixnum_value(peek), int64_t(10));

    // Removing likewise leaves the original standing.
    Value without = map_erase(p, m, make_fixnum(5));
    CHECK_EQ(static_cast<MapObj*>(as_obj(without))->count, 199u);
    CHECK(!map_lookup(p, without, make_fixnum(5), &peek));
    CHECK(map_lookup(p, m, make_fixnum(5), &peek));
    // Removing something that was never there changes nothing.
    Value same = map_erase(p, m, make_fixnum(9999));
    CHECK_EQ(static_cast<MapObj*>(as_obj(same))->count, 200u);

    // Emptying a map leaves a map, not nothing.
    Value one = map_insert(p, p.heap().make_map(0), make_fixnum(1), UNIT);
    Value none = map_erase(p, one, make_fixnum(1));
    CHECK(is_obj(none, ObjType::Map));
    CHECK_EQ(static_cast<MapObj*>(as_obj(none))->count, 0u);

    // Every entry comes back exactly once, however the trie is shaped.
    std::vector<std::pair<Value, Value>> entries;
    map_collect(m, entries);
    CHECK_EQ(entries.size(), size_t(200));

    // String keys compare by contents, not identity.
    Value m2 = map_insert(p, p.heap().make_map(0), p.heap().make_string("key", 3),
                          make_fixnum(1));
    Value found;
    CHECK(map_lookup(p, m2, p.heap().make_string("key", 3), &found));
    CHECK_EQ(fixnum_value(found), int64_t(1));
}

static void test_arithmetic_boundaries() {
    int64_t result = 0;
    CHECK(add_overflow(INT64_MAX, 1, &result));
    CHECK(add_overflow(INT64_MIN, -1, &result));
    CHECK(!add_overflow(INT64_MIN, INT64_MAX, &result));
    CHECK_EQ(result, int64_t(-1));
    CHECK(sub_overflow(INT64_MIN, 1, &result));
    CHECK(sub_overflow(INT64_MAX, -1, &result));
    CHECK(!sub_overflow(INT64_MIN, INT64_MIN, &result));
    CHECK_EQ(result, int64_t(0));
    CHECK(mul_overflow(INT64_MIN, -1, &result));
    CHECK(mul_overflow(-1, INT64_MIN, &result));
    CHECK(mul_overflow(INT64_MAX, 2, &result));
    CHECK(mul_overflow(INT64_MIN, 2, &result));
    CHECK(!mul_overflow(INT64_MIN, 1, &result));
    CHECK_EQ(result, INT64_MIN);
    CHECK(!mul_overflow(0, INT64_MIN, &result));
    CHECK_EQ(result, int64_t(0));
    CHECK(!mul_overflow(-3, -7, &result));
    CHECK_EQ(result, int64_t(21));
}


// --- TLS --------------------------------------------------------------------
//
// The engine is driven from memory, so two of them can talk to each other with
// nothing between but this loop: what one writes the other is fed. That tests
// the backend this VM was built with -- OpenSSL here, SChannel on Windows -- on
// its own, with no socket and no scheduler to blame when it fails.

struct TlsPair {
    std::unique_ptr<tls::Engine> client;
    std::unique_ptr<tls::Engine> server;
    tls::Status client_status = tls::Status::WantRead;
    tls::Status server_status = tls::Status::WantRead;
};

static tls::Config tls_server_config() {
    tls::Config c;
    c.server = true;
    c.verify = false;
    c.identity_p12 = tls_fixtures::server_p12();
    c.identity_password = tls_fixtures::password;
    return c;
}

static tls::Config tls_client_config() {
    tls::Config c;
    c.host = "localhost";
    c.ca_pem = tls_fixtures::ca_pem();
    return c;
}

/// Run both handshakes until each has finished or failed, passing records
/// across. Answers false when they stopped making progress.
static bool tls_shake(TlsPair& t) {
    for (int round = 0; round < 50; ++round) {
        if (t.client_status == tls::Status::WantRead) t.client_status = t.client->handshake();
        std::string to_server = t.client->take_output();
        t.server->feed(to_server.data(), to_server.size());
        if (t.server_status == tls::Status::WantRead) t.server_status = t.server->handshake();
        std::string to_client = t.server->take_output();
        t.client->feed(to_client.data(), to_client.size());
        bool client_done = t.client_status != tls::Status::WantRead;
        bool server_done = t.server_status != tls::Status::WantRead;
        if (client_done && server_done && to_server.empty() && to_client.empty()) return true;
        // One side failed and said so; the other is waiting for what will not come.
        if ((t.client_status == tls::Status::Error || t.server_status == tls::Status::Error)
            && to_server.empty() && to_client.empty()) {
            return true;
        }
    }
    return false;
}

static TlsPair tls_pair(const tls::Config& client, const tls::Config& server) {
    TlsPair t;
    tls::Failure why;
    t.client = tls::make_engine(client, &why);
    if (!t.client) std::printf("  client engine: %s %s\n", why.kind.c_str(), why.message.c_str());
    t.server = tls::make_engine(server, &why);
    if (!t.server) std::printf("  server engine: %s %s\n", why.kind.c_str(), why.message.c_str());
    return t;
}

/// Plaintext sent one way: written by `from`, fed across, read by `to`.
static std::string tls_send(tls::Engine& from, tls::Engine& to, const std::string& text) {
    if (from.write(text.data(), text.size()) != tls::Status::Ok) return "<write failed>";
    std::string records = from.take_output();
    to.feed(records.data(), records.size());
    std::string got;
    char buf[4096];
    for (;;) {
        size_t n = 0;
        tls::Status st = to.read(buf, sizeof buf, &n);
        // A read may answer with records of its own (a TLS 1.3 ticket's
        // acknowledgement, say); they go back the other way.
        std::string back = to.take_output();
        from.feed(back.data(), back.size());
        if (st != tls::Status::Ok) break;
        got.append(buf, n);
    }
    return got;
}

static void test_tls_engines() {
    std::printf("tls engines (%s)\n", tls::backend().c_str());

    // A whole conversation, both ways, and the server's name checked.
    {
        tls::Config server = tls_server_config();
        server.alpn = {"dream/1", "http/1.1"};
        tls::Config client = tls_client_config();
        client.alpn = {"dream/1"};
        TlsPair t = tls_pair(client, server);
        CHECK(t.client && t.server);
        if (t.client && t.server) {
            CHECK(tls_shake(t));
            CHECK(t.client_status == tls::Status::Ok);
            CHECK(t.server_status == tls::Status::Ok);
            CHECK_EQ(t.client->info().alpn, std::string("dream/1"));
            CHECK_EQ(t.server->info().alpn, std::string("dream/1"));
            CHECK_EQ(t.client->info().peer, std::string("CN=localhost"));
            CHECK(!t.client->info().version.empty());
            CHECK_EQ(tls_send(*t.client, *t.server, "hello, server"), std::string("hello, server"));
            CHECK_EQ(tls_send(*t.server, *t.client, "hello, client"), std::string("hello, client"));
            std::string big(200000, 'x');
            for (size_t i = 0; i < big.size(); ++i) big[i] = char('a' + i % 26);
            CHECK(tls_send(*t.client, *t.server, big) == big);

            // A clean close reads as the end of the stream on the other side.
            t.client->close();
            std::string bye = t.client->take_output();
            t.server->feed(bye.data(), bye.size());
            char buf[16];
            size_t n = 0;
            tls::Status at_close = t.server->read(buf, sizeof buf, &n);
            if (at_close != tls::Status::Closed) {
                std::printf("  close_notify: %zu bytes sent, %s, server read answered %d (%s)\n", bye.size(),
                            t.client->info().version.c_str(), int(at_close),
                            t.server->failure().message.c_str());
            }
            CHECK(at_close == tls::Status::Closed);
        }
    }

    // Each way verification fails is reported as what it is.
    struct Refusal { const char* what; std::string ca; std::string host; std::string identity; const char* kind; };
    std::vector<Refusal> refusals = {
        {"untrusted", tls_fixtures::other_ca_pem(), "localhost", tls_fixtures::server_p12(), "certificate_untrusted"},
        {"misnamed", tls_fixtures::ca_pem(), "example.com", tls_fixtures::server_p12(), "hostname_mismatch"},
        {"expired", tls_fixtures::ca_pem(), "localhost", tls_fixtures::expired_p12(), "certificate_expired"},
    };
    for (const Refusal& r : refusals) {
        tls::Config client = tls_client_config();
        client.ca_pem = r.ca;
        client.host = r.host;
        tls::Config server = tls_server_config();
        server.identity_p12 = r.identity;
        TlsPair t = tls_pair(client, server);
        CHECK(t.client && t.server);
        if (!t.client || !t.server) continue;
        tls_shake(t);
        CHECK(t.client_status == tls::Status::Error);
        if (t.client->failure().kind != r.kind) {
            std::printf("  %s: got %s (%s)\n", r.what, t.client->failure().kind.c_str(),
                        t.client->failure().message.c_str());
        }
        CHECK_EQ(t.client->failure().kind, std::string(r.kind));
    }

    // Verification off accepts what it would have refused.
    {
        tls::Config client = tls_client_config();
        client.ca_pem = tls_fixtures::other_ca_pem();
        client.verify = false;
        TlsPair t = tls_pair(client, tls_server_config());
        CHECK(t.client && t.server && tls_shake(t) && t.client_status == tls::Status::Ok);
    }

    // The chain alone: a trusted certificate for another name is accepted,
    // an untrusted one is not.
    {
        tls::Config client = tls_client_config();
        client.host = "example.com";
        client.check_name = false;
        TlsPair t = tls_pair(client, tls_server_config());
        CHECK(t.client && t.server && tls_shake(t) && t.client_status == tls::Status::Ok);
        client.ca_pem = tls_fixtures::other_ca_pem();
        TlsPair u = tls_pair(client, tls_server_config());
        CHECK(u.client && u.server);
        if (u.client && u.server) {
            tls_shake(u);
            CHECK_EQ(u.client->failure().kind, std::string("certificate_untrusted"));
        }
    }

    // Revocation from a list: what it names is refused, what it does not is
    // accepted, and without the list the revoked certificate's status is
    // unknown and it goes through, as tls.hpp says it must on every backend.
    {
        tls::Config client = tls_client_config();
        client.crl_pem = tls_fixtures::ca_crl();
        tls::Config revoked = tls_server_config();
        revoked.identity_p12 = tls_fixtures::revoked_p12();
        TlsPair t = tls_pair(client, revoked);
        CHECK(t.client && t.server);
        if (t.client && t.server) {
            tls_shake(t);
            if (t.client->failure().kind != "certificate_revoked") {
                std::printf("  revoked: got %s (%s)\n", t.client->failure().kind.c_str(),
                            t.client->failure().message.c_str());
            }
            CHECK_EQ(t.client->failure().kind, std::string("certificate_revoked"));
        }
        TlsPair good = tls_pair(client, tls_server_config());
        CHECK(good.client && good.server && tls_shake(good) && good.client_status == tls::Status::Ok);
        TlsPair unlisted = tls_pair(tls_client_config(), revoked);
        CHECK(unlisted.client && unlisted.server && tls_shake(unlisted) && unlisted.client_status == tls::Status::Ok);
        tls::Failure why;
        tls::Config bad = tls_client_config();
        bad.crl_pem = "no lists here";
        CHECK(!tls::make_engine(bad, &why));
        CHECK_EQ(why.kind, std::string("tls_config"));
    }

    // Mutual TLS: the server sees who the client is.
    {
        tls::Config server = tls_server_config();
        server.ca_pem = tls_fixtures::ca_pem();
        server.verify = true;
        tls::Config client = tls_client_config();
        client.identity_p12 = tls_fixtures::client_p12();
        client.identity_password = tls_fixtures::password;
        TlsPair t = tls_pair(client, server);
        CHECK(t.client && t.server);
        if (t.client && t.server) {
            CHECK(tls_shake(t));
            CHECK(t.client_status == tls::Status::Ok);
            CHECK(t.server_status == tls::Status::Ok);
            CHECK_EQ(t.server->info().peer, std::string("CN=Dream Test Client"));
            CHECK_EQ(tls_send(*t.client, *t.server, "it is me"), std::string("it is me"));
        }
    }

    // What cannot be used is refused before any handshake.
    {
        tls::Failure why;
        tls::Config c = tls_server_config();
        c.identity_password = "wrong";
        CHECK(!tls::make_engine(c, &why));
        CHECK_EQ(why.kind, std::string("tls_config"));
        tls::Config d = tls_client_config();
        d.ca_pem = "no certificates here";
        why = {};
        CHECK(!tls::make_engine(d, &why));
        CHECK_EQ(why.kind, std::string("tls_config"));
    }

    // The PEM reader the backends without one share.
    {
        std::vector<std::string> certs = tls::pem_certificates(tls_fixtures::ca_pem() + tls_fixtures::other_ca_pem());
        CHECK_EQ(certs.size(), size_t(2));
        CHECK(!certs.empty() && certs[0].size() > 100 && uint8_t(certs[0][0]) == 0x30);  // a DER SEQUENCE
    }
}

int main() {
    // Unbuffered, so that a test that crashes the process still leaves the
    // name of the section it was in: under CI stdout is a pipe, and a pipe is
    // block-buffered, so a crash used to take every line with it.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_arithmetic_boundaries();
    test_value_tagging();
    test_heap_alloc();
    test_gc_keeps_live_and_drops_dead();
    test_gc_preserves_sharing();
    test_gc_collapses_indirections();
    test_gc_handles_cycles();
    test_major_evacuates_sparse_blocks();
    test_minor_collection_promotes_reachable();
    test_minor_collection_leaves_old_space_alone();
    test_write_barrier_keeps_old_to_young();
    test_parallel_collection_preserves_sharing();
    test_parallel_major_collects_across_threads();
    test_declined_concurrent_mark();
    test_concurrent_mark_period();
    test_heap_verifier_accepts_a_healthy_heap();
    test_heap_verifier_catches_corruption();
    test_shared_area();
    test_heap_verifier_follows_every_object_kind();
    test_cross_heap_copy();
    test_big_block_pool();
    test_tensor_kernels();
    test_image_rejects_bad_input();
    test_image_frame_validation();
    test_integer_hints();
    test_type_test_nodes();
    test_payload_sections();
    test_bigstr_values();
    test_atom_interning();
    test_builtin_table_matches_compiler();
    test_maps();
    test_tls_engines();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
