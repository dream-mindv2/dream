// A C library written to be wrapped by std.ffi's tests.
//
// Every destructor writes to a log, so a test can say not only that a resource
// was let go but when, and in what order relative to the others: a cursor must
// be closed before the counter it reads from, whoever asked for either.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char destroy_log[4096];

static void note(const char* what, long id) {
    size_t at = strlen(destroy_log);
    snprintf(destroy_log + at, sizeof destroy_log - at, "%s%s%ld", at ? "," : "", what, id);
}

const char* sample_log(void) { return destroy_log; }
void sample_clear_log(void) { destroy_log[0] = 0; }

// --- an opaque handle, and one made from it ---------------------------------

typedef struct counter {
    long id;
    long value;
    int alive;
} counter;

static long next_id = 1;

counter* counter_new(long start) {
    counter* c = malloc(sizeof *c);
    c->id = next_id++;
    c->value = start;
    c->alive = 1;
    return c;
}

void counter_free(counter* c) {
    note("counter", c->id);
    c->alive = 0;
    free(c);
}

long counter_add(counter* c, long n) { return c->value += n; }
long counter_id(counter* c) { return c->id; }

/// Fails with -1 for a negative start, as C APIs do, and hands the counter
/// back through a pointer rather than as its return.
int counter_make(long start, counter** out) {
    if (start < 0) {
        *out = NULL;
        return -1;
    }
    *out = counter_new(start);
    return 0;
}

typedef struct cursor {
    counter* of;
} cursor;

cursor* cursor_open(counter* c) {
    cursor* k = malloc(sizeof *k);
    k->of = c;
    return k;
}

void cursor_close(cursor* k) {
    // A cursor that outlived its counter would read freed memory here. Logging
    // what it read keeps the order visible: `alive` is 1 exactly when the
    // counter was still there.
    note(k->of->alive ? "cursor" : "cursor-after-free", k->of->id);
    free(k);
}

long cursor_read(cursor* k) { return k->of->value; }

counter* counter_null(void) { return NULL; }

// --- numbers, out-parameters and strings -------------------------------------

int divmod(int a, int b, int* q, int* r) {
    if (b == 0) return -1;
    *q = a / b;
    *r = a % b;
    return 0;
}

double scale(float x, double by) { return x * by; }
uint8_t low_byte(uint32_t x) { return (uint8_t)x; }
int8_t negate8(int8_t x) { return (int8_t)-x; }

char* greet(const char* name) {
    char* s = malloc(strlen(name) + 8);
    strcpy(s, "hello, ");
    strcat(s, name);
    return s;
}

const char* static_name(void) { return "sample"; }

size_t count_byte(const char* data, size_t n, int c) {
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) k += (unsigned char)data[i] == c;
    return k;
}

// --- callbacks --------------------------------------------------------------

long fold_range(long n, long (*f)(long, long), long init) {
    long acc = init;
    for (long i = 1; i <= n; ++i) acc = f(acc, i);
    return acc;
}

double apply_twice(double (*f)(double), double x) { return f(f(x)); }

static int (*compare_by)(int, int);
static int trampoline(const void* a, const void* b) {
    return compare_by(*(const int*)a, *(const int*)b);
}

void sort_ints(int* xs, size_t n, int (*cmp)(int, int)) {
    compare_by = cmp;
    qsort(xs, n, sizeof *xs, trampoline);
}

// --- a struct, by pointer -----------------------------------------------------

typedef struct point {
    int32_t x;
    double y;
    uint8_t tag;
} point;

double point_sum(const point* p) { return p->x + p->y + p->tag; }

void point_fill(point* p, int32_t x, double y) {
    p->x = x;
    p->y = y;
    p->tag = 7;
}

size_t point_size(void) { return sizeof(point); }
