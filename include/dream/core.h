/* Dream VM — core C ABI types.
 *
 * This header is C-compatible so the VM can be embedded from C and C++.
 */
#ifndef DREAM_CORE_H
#define DREAM_CORE_H 1

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DREAM_BAD 0
#define DREAM_OK  1
#define DREAM_IS_OK(x) ((x) != DREAM_BAD)

typedef int      dream_result;
typedef int64_t  dream_integer;
typedef double   dream_float;

/* A Dream value: an immediate, or a tagged pointer into a process heap.
 * Values are only meaningful relative to the process that owns them. */
typedef uint64_t dream_value;

/* An arity meaning "however many arguments the call site passed".
 *
 * Dream functions are curried, so a variadic function cannot be recognised by
 * saturation -- `f a b` and `f a b c` differ only in the application node. A
 * variadic host function therefore receives exactly the arguments of the
 * application that reached it, and is never partially applied. Every argument
 * is forced, whatever the registered strict mask says, because a 32-bit mask
 * cannot describe an unbounded argument list. `std.console.print!` is one. */
#define DREAM_VARIADIC 0xFFFFFFFFu

/* The language's surface types, as reported by `type_of`.
 * The list in dreams must stay in step. */
typedef enum dream_type {
    DREAM_TYPE_INTEGER = 0,
    DREAM_TYPE_FLOAT,
    DREAM_TYPE_CHAR,
    DREAM_TYPE_BOOL,
    DREAM_TYPE_UNIT,
    DREAM_TYPE_STRING,
    DREAM_TYPE_ATOM,
    DREAM_TYPE_LIST,
    DREAM_TYPE_ARRAY,
    DREAM_TYPE_MAP,
    DREAM_TYPE_PURE_FN,
    DREAM_TYPE_IMPURE_FN,
    DREAM_TYPE_MODULE,
    DREAM_TYPE_ERROR,
    /* A green process: isolated heap, mailbox, scheduler slot. */
    DREAM_TYPE_PROCESS,
    DREAM_TYPE_UNKNOWN,
    /* A read-only view into the image's payload region, which may be larger
     * than the 4 GiB the rest of the container can address. Appended after
     * DREAM_TYPE_UNKNOWN rather than filed beside DREAM_TYPE_STRING, because
     * every value here is already compiled into somebody's embedder. */
    DREAM_TYPE_BIGSTR,
    /* Packed numbers with a shape: what the numeric operators and std.tensor
     * work on. Appended for the same reason DREAM_TYPE_BIGSTR was. */
    DREAM_TYPE_TENSOR,
    /* A value of a declared record or union: a fixed run of fields and the
     * name of what it is. Appended for the same reason as the two above. */
    DREAM_TYPE_RECORD
} dream_type;

/* `thread` was the original spelling in the language spec and is kept as an
 * alias so existing embedders keep compiling. */
#define DREAM_TYPE_THREAD DREAM_TYPE_PROCESS

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DREAM_CORE_H */
