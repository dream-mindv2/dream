/* Dream VM — public embedding API.
 *
 * The VM is usable from C and C++: create one, load a bytecode image, run an
 * entry point. Host functions can be registered as modules that Dream code
 * reaches through `import`.
 */
#ifndef DREAM_H
#define DREAM_H 1

#include "dream/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dream_vm dream_vm;
typedef struct dream_process dream_process;

/* A host function. Arguments arrive already forced unless the registration
 * cleared their bit in `strict_mask`. Return DREAM_OK and set *out, or return
 * DREAM_BAD and set *out to an error value. */
typedef dream_result (*dream_native_fn)(dream_process* p, const dream_value* args,
                                      uint32_t argc, dream_value* out);

dream_vm* dream_vm_new(void);
void dream_vm_free(dream_vm* vm);

/* Load a `.dream` image. On failure writes a message into `err`. */
dream_result dream_vm_load_file(dream_vm* vm, const char* path, char* err, size_t err_size);
dream_result dream_vm_load_bytes(dream_vm* vm, const uint8_t* data, size_t size,
                               char* err, size_t err_size);

/* Register a module before running. `member_names`, `member_arities`,
 * `member_strict_masks` and `member_fns` are parallel arrays of `count`
 * entries. An arity of DREAM_VARIADIC makes a member take however many
 * arguments its call site passed, with every one forced. */
dream_result dream_vm_register_module(dream_vm* vm, const char* module_name,
                                    const char* const* member_names,
                                    const uint32_t* member_arities,
                                    const uint32_t* member_strict_masks,
                                    const dream_native_fn* member_fns, size_t count);

/* Number of scheduler threads. 0 means "one per hardware thread". */
void dream_vm_set_workers(dream_vm* vm, unsigned workers);
/* Turn the JIT tier on or off, and set how many calls make a function hot. */
void dream_vm_set_jit(dream_vm* vm, int enabled, uint32_t threshold);

/* Run `entry` (defaults to "main!" when NULL) to completion. Returns DREAM_OK
 * when the root process finished without raising. */
dream_result dream_vm_run(dream_vm* vm, const char* entry);

/* The root process's result, rendered as text. Valid until the next run. */
const char* dream_vm_result_text(dream_vm* vm);
/* Non-zero when the last run raised. */
int dream_vm_failed(const dream_vm* vm);

/* Statistics from the last run. */
uint64_t dream_vm_reductions(const dream_vm* vm);
uint64_t dream_vm_collections(const dream_vm* vm);

/* ------------------------------------------------------------------ */
/* Running and reading a result                                        */
/*                                                                     */
/* `dream_vm_run_value` is the entry point a tool uses when it wants the */
/* answer rather than a transcript -- the compiler's `comp!` runs an    */
/* expression this way and reads the value back. The result is forced   */
/* all the way down before it is handed over, so every nested value is  */
/* safe to inspect, and it stays valid until the next run or until the  */
/* VM is freed.                                                         */
/* ------------------------------------------------------------------ */

dream_result dream_vm_run_value(dream_vm* vm, const char* entry, dream_value* out);

/* Force a value and everything reachable through it. */
dream_result dream_vm_force_deep(dream_vm* vm, dream_value v, dream_value* out);

/* The name behind an atom, or NULL. Valid for the life of the VM. */
const char* dream_vm_atom_name(dream_vm* vm, dream_value v);

/* --- structural inspection --- */

uint32_t dream_value_char(dream_value v);

/* A list is a chain of cells. Returns DREAM_OK and fills head/tail while `v`
 * is a cell; returns DREAM_BAD at the end of the list. */
dream_result dream_value_list_next(dream_value v, dream_value* head, dream_value* tail);
int dream_value_list_is_empty(dream_value v);

uint32_t dream_value_array_len(dream_value v);
dream_value dream_value_array_at(dream_value v, uint32_t index);

uint32_t dream_value_map_count(dream_value v);
/* Walk a map. Start with *cursor = 0; returns DREAM_BAD when done. */
dream_result dream_value_map_next(dream_value v, uint32_t* cursor, dream_value* key,
                                dream_value* value);

dream_value dream_value_error_kind(dream_value v);
dream_value dream_value_error_payload(dream_value v);

/* The process id behind a process value. */
uint64_t dream_value_process_id(dream_value v);

/* --- building structures, for host functions --- */

dream_value dream_make_list_empty(void);
dream_value dream_make_cons(dream_process* p, dream_value head, dream_value tail);
dream_value dream_make_array(dream_process* p, uint32_t len);
/* `p` owns `array`, which the C API needs so the written value's reachability
   is tracked between collections. */
dream_result dream_array_set(dream_process* p, dream_value array, uint32_t index, dream_value v);
dream_value dream_make_map(dream_process* p, uint32_t capacity_hint);
/* Returns the map to use afterwards: inserting may replace the object. */
dream_value dream_map_insert(dream_process* p, dream_value map, dream_value key, dream_value v);
dream_result dream_map_get(dream_process* p, dream_value map, dream_value key, dream_value* out);

/* --- runtime facts, for tooling --- */

/* The modules this VM provides natively. The compiler keeps its own list of
 * these, so exposing them is what lets a test prove the two agree. */
uint32_t dream_vm_native_module_count(const dream_vm* vm);
const char* dream_vm_native_module_name(const dream_vm* vm, uint32_t index);

uint64_t dream_vm_process_count(const dream_vm* vm);
uint32_t dream_vm_module_count(const dream_vm* vm);
const char* dream_vm_module_name(const dream_vm* vm, uint32_t index);

/* --- values, for host functions --- */
dream_type dream_value_type(dream_value v);
dream_integer dream_value_integer(dream_value v);
dream_float dream_value_float(dream_value v);
int dream_value_bool(dream_value v);
/* Borrowed pointer into the process heap; valid until the next allocation. */
const char* dream_value_string(dream_value v, uint32_t* len);

dream_value dream_make_integer(dream_process* p, dream_integer n);
dream_value dream_make_float(dream_process* p, dream_float d);
dream_value dream_make_bool(int b);
dream_value dream_make_unit(void);
dream_value dream_make_string(dream_process* p, const char* s, uint32_t len);
dream_value dream_make_atom(dream_process* p, const char* name);
dream_value dream_make_error(dream_process* p, const char* kind, const char* message);

/* Force a value to weak head normal form. Returns DREAM_BAD if it raised. */
dream_result dream_force(dream_process* p, dream_value v, dream_value* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DREAM_H */
