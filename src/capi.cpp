// The C embedding API, implemented over the C++ runtime.

#include "dream/dream.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bigint.hpp"
#include "builtins.hpp"
#include "interp.hpp"
#include "jit.hpp"
#include "process.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

using namespace dream;

// The C header cannot include the C++ one, so the sentinel is spelled twice.
// This is what keeps the two spellings the same value.
static_assert(DREAM_VARIADIC == NATIVE_VARIADIC,
              "DREAM_VARIADIC and NATIVE_VARIADIC must agree");

namespace {

/// One registered host function, kept alive for the lifetime of the VM so the
/// ModuleDef can point at stable storage.
struct HostFn {
    std::string name;
    dream_native_fn fn;
};

}  // namespace

struct dream_vm {
    Runtime rt;
    std::unique_ptr<Scheduler> sched;
    /// Kept alive after a run so the result value, which lives in this
    /// process's heap, stays readable.
    std::shared_ptr<Process> root;
    unsigned workers = 0;
    bool jit_enabled = true;
    uint32_t jit_threshold = 0;

    std::string result_text;
    bool failed = false;
    uint64_t reductions = 0;
    uint64_t collections = 0;

    // Registered host functions, by module then member.
    std::vector<std::unique_ptr<std::vector<HostFn>>> host_fns;
};

namespace {

/// Trampoline from the VM's native calling convention to the C one.
///
/// One trampoline serves every registered host function: the C function
/// pointer travels on the function value itself, in `user`. That is what makes
/// the number of host functions unlimited rather than capped by a table of
/// generated thunks.
NativeResult host_thunk(Process& p, Value callee, Value* args, uint32_t argc) {
    auto* nat = static_cast<NativeObj*>(as_obj(callee));
    auto fn = reinterpret_cast<dream_native_fn>(static_cast<uintptr_t>(nat->user));
    dream_value out = UNIT;
    dream_result r = fn(reinterpret_cast<dream_process*>(&p),
                       reinterpret_cast<const dream_value*>(args), argc, &out);
    return r == DREAM_OK ? NativeResult::ok(out) : NativeResult::raise(out);
}

void copy_error(char* err, size_t err_size, const std::string& msg) {
    if (!err || err_size == 0) return;
    std::snprintf(err, err_size, "%s", msg.c_str());
}

}  // namespace

extern "C" {

dream_vm* dream_vm_new(void) { return new dream_vm(); }

void dream_vm_free(dream_vm* vm) {
    if (!vm) return;
    if (vm->sched) vm->sched->stop();
    delete vm;
}

dream_result dream_vm_load_file(dream_vm* vm, const char* path, char* err, size_t err_size) {
    std::string message;
    if (!vm->rt.load_image_file(path, message)) {
        copy_error(err, err_size, message);
        return DREAM_BAD;
    }
    return DREAM_OK;
}

dream_result dream_vm_load_bytes(dream_vm* vm, const uint8_t* data, size_t size, char* err,
                               size_t err_size) {
    std::string message;
    if (!vm->rt.load_image_bytes(data, size, message)) {
        copy_error(err, err_size, message);
        return DREAM_BAD;
    }
    return DREAM_OK;
}

dream_result dream_vm_register_module(dream_vm* vm, const char* module_name,
                                    const char* const* member_names,
                                    const uint32_t* member_arities,
                                    const uint32_t* member_strict_masks,
                                    const dream_native_fn* member_fns, size_t count) {
    ModuleDef def;
    def.name = module_name;
    // The names have to outlive this call, so keep a copy alongside the VM.
    auto storage = std::make_unique<std::vector<HostFn>>();
    storage->reserve(count);
    for (size_t i = 0; i < count; ++i) {
        storage->push_back(HostFn{member_names[i], member_fns[i]});
    }
    for (size_t i = 0; i < count; ++i) {
        def.members.push_back(NativeDef{
            (*storage)[i].name.c_str(), member_arities[i], member_strict_masks[i], host_thunk,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(member_fns[i]))});
    }
    vm->host_fns.push_back(std::move(storage));
    vm->rt.register_module(std::move(def));
    return DREAM_OK;
}

void dream_vm_set_workers(dream_vm* vm, unsigned workers) { vm->workers = workers; }

void dream_vm_set_jit(dream_vm* vm, int enabled, uint32_t threshold) {
    vm->jit_enabled = enabled != 0;
    vm->jit_threshold = threshold;
}

dream_result dream_vm_run(dream_vm* vm, const char* entry) {
    if (!vm->rt.has_image()) return DREAM_BAD;
    const Image& img = vm->rt.image();

    uint32_t func = NO_NODE;
    if (entry && *entry) {
        int g = img.find_global(entry);
        if (g < 0) return DREAM_BAD;
        const GlobalRec& gr = img.global(uint32_t(g));
        if (gr.kind != GLOBAL_FUNCTION) return DREAM_BAD;
        func = gr.target;
    } else {
        func = img.entry();
    }
    if (func == NO_NODE) return DREAM_BAD;

    unsigned workers = vm->workers;
    if (workers == 0) workers = Scheduler::default_workers();
    vm->sched = std::make_unique<Scheduler>(vm->rt, workers);

    auto root = vm->sched->create_process();
    vm->root = root;
    Value cl = root->heap().make_closure(func, 0);
    prime_apply(*root, cl, 0);

    bool clean = vm->sched->run(root);

    vm->failed = root->failed || !clean;
    vm->reductions = vm->sched->total_reductions();
    vm->collections = root->heap().collections();

    vm->result_text.clear();
    if (!clean && !root->failed) {
        vm->result_text = "deadlock: every thread is waiting for a message that cannot arrive";
    } else {
        stringify(*root, root->exit_value, &vm->result_text);
    }
    return vm->failed ? DREAM_BAD : DREAM_OK;
}

const char* dream_vm_result_text(dream_vm* vm) { return vm->result_text.c_str(); }
int dream_vm_failed(const dream_vm* vm) { return vm->failed ? 1 : 0; }
uint64_t dream_vm_reductions(const dream_vm* vm) { return vm->reductions; }
uint64_t dream_vm_collections(const dream_vm* vm) { return vm->collections; }

dream_type dream_value_type(dream_value v) { return surface_type(v); }

dream_integer dream_value_integer(dream_value v) {
    v = resolve(v);
    if (is_fixnum(v)) return fixnum_value(v);
    if (is_obj(v, ObjType::Float)) return dream_integer(static_cast<FloatObj*>(as_obj(v))->value);
    if (bigint::is_big(v)) {
        // Saturated: a host asking for an int64 of something wider gets the
        // nearest one it can hold, never a wrapped one of the wrong sign.
        int64_t n;
        if (bigint::to_int64(v, &n)) return n;
        return bigint::compare(v, make_fixnum(0)) < 0 ? INT64_MIN : INT64_MAX;
    }
    return 0;
}

dream_float dream_value_float(dream_value v) {
    v = resolve(v);
    if (is_obj(v, ObjType::Float)) return static_cast<FloatObj*>(as_obj(v))->value;
    if (bigint::is_integer(v)) return dream_float(bigint::to_double(v));
    return 0.0;
}

int dream_value_bool(dream_value v) { return truthy(resolve(v)) ? 1 : 0; }

const char* dream_value_string(dream_value v, uint32_t* len) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Str)) {
        if (len) *len = 0;
        return nullptr;
    }
    auto* s = static_cast<StrObj*>(as_obj(v));
    if (len) *len = s->len;
    return s->data();
}

dream_value dream_make_integer(dream_process* p, dream_integer n) {
    return make_integer(*reinterpret_cast<Process*>(p), n);
}
dream_value dream_make_float(dream_process* p, dream_float d) {
    return reinterpret_cast<Process*>(p)->heap().make_float(d);
}
dream_value dream_make_bool(int b) { return make_bool(b != 0); }
dream_value dream_make_unit(void) { return UNIT; }

dream_value dream_make_string(dream_process* p, const char* s, uint32_t len) {
    return reinterpret_cast<Process*>(p)->heap().make_string(s, len);
}

dream_value dream_make_atom(dream_process* p, const char* name) {
    auto* proc = reinterpret_cast<Process*>(p);
    return make_atom(proc->runtime().intern_atom(name));
}

dream_value dream_make_error(dream_process* p, const char* kind, const char* message) {
    auto* proc = reinterpret_cast<Process*>(p);
    return raise_error(*proc, proc->runtime().intern_atom(kind), message);
}

dream_result dream_force(dream_process* p, dream_value v, dream_value* out) {
    auto* proc = reinterpret_cast<Process*>(p);
    Value result;
    if (!force_whnf(*proc, v, &result)) {
        *out = proc->result;
        return DREAM_BAD;
    }
    *out = result;
    return DREAM_OK;
}

/* ------------------------------------------------------------------ */
/* Running and reading a result                                        */
/* ------------------------------------------------------------------ */

dream_result dream_vm_run_value(dream_vm* vm, const char* entry, dream_value* out) {
    dream_result r = dream_vm_run(vm, entry);
    if (!vm->root) return DREAM_BAD;
    Value v = vm->root->exit_value;
    if (r == DREAM_OK) {
        // Force everything before handing it over: the caller has no way to
        // drive evaluation, and a nested thunk would be unreadable.
        Value deep;
        if (force_deep(*vm->root, v, &deep)) {
            v = deep;
        } else {
            v = vm->root->result;
            r = DREAM_BAD;
        }
    }
    if (out) *out = v;
    return r;
}

dream_result dream_vm_force_deep(dream_vm* vm, dream_value v, dream_value* out) {
    if (!vm->root) return DREAM_BAD;
    Value deep;
    if (!force_deep(*vm->root, v, &deep)) {
        if (out) *out = vm->root->result;
        return DREAM_BAD;
    }
    if (out) *out = deep;
    return DREAM_OK;
}

const char* dream_vm_atom_name(dream_vm* vm, dream_value v) {
    v = resolve(v);
    if (!is_atom(v)) return nullptr;
    return vm->rt.atom_name(uint32_t(imm_payload(v))).c_str();
}

/* --- structural inspection --- */

uint32_t dream_value_char(dream_value v) {
    v = resolve(v);
    return is_char(v) ? uint32_t(imm_payload(v)) : 0;
}

dream_result dream_value_list_next(dream_value v, dream_value* head, dream_value* tail) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Cons)) return DREAM_BAD;
    auto* c = static_cast<ConsObj*>(as_obj(v));
    if (head) *head = c->head;
    if (tail) *tail = c->tail;
    return DREAM_OK;
}

int dream_value_list_is_empty(dream_value v) { return is_nil(resolve(v)) ? 1 : 0; }

uint32_t dream_value_array_len(dream_value v) {
    v = resolve(v);
    return is_obj(v, ObjType::Array) ? static_cast<ArrayObj*>(as_obj(v))->len : 0;
}

dream_value dream_value_array_at(dream_value v, uint32_t index) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Array)) return UNIT;
    auto* a = static_cast<ArrayObj*>(as_obj(v));
    return index < a->len ? a->items()[index] : UNIT;
}

uint32_t dream_value_map_count(dream_value v) {
    v = resolve(v);
    return is_obj(v, ObjType::Map) ? static_cast<MapObj*>(as_obj(v))->count : 0;
}

dream_result dream_value_map_next(dream_value v, uint32_t* cursor, dream_value* key,
                                dream_value* value) {
    v = resolve(v);
    if (!is_obj(v, ObjType::Map) || !cursor) return DREAM_BAD;
    // The cursor counts entries. A trie has no flat slot array to index, and a
    // path does not fit in the 32 bits this interface hands out, so each step
    // walks to the nth entry. Maps that cross the C API are host-facing --
    // configuration and results -- and small enough that this costs nothing
    // worth a wider interface.
    std::vector<std::pair<Value, Value>> entries;
    map_collect(v, entries);
    if (*cursor >= entries.size()) {
        *cursor = uint32_t(entries.size());
        return DREAM_BAD;
    }
    if (key) *key = entries[*cursor].first;
    if (value) *value = entries[*cursor].second;
    *cursor += 1;
    return DREAM_OK;
}

dream_value dream_value_error_kind(dream_value v) {
    v = resolve(v);
    return is_obj(v, ObjType::ErrorBox) ? static_cast<ErrorObj*>(as_obj(v))->kind : UNIT;
}

dream_value dream_value_error_payload(dream_value v) {
    v = resolve(v);
    return is_obj(v, ObjType::ErrorBox) ? static_cast<ErrorObj*>(as_obj(v))->payload : UNIT;
}

uint64_t dream_value_process_id(dream_value v) {
    v = resolve(v);
    return is_obj(v, ObjType::Pid) ? static_cast<PidObj*>(as_obj(v))->id : 0;
}

/* --- building structures --- */

dream_value dream_make_list_empty(void) { return NIL; }

dream_value dream_make_cons(dream_process* p, dream_value head, dream_value tail) {
    return reinterpret_cast<Process*>(p)->heap().make_cons(head, tail);
}

dream_value dream_make_array(dream_process* p, uint32_t len) {
    return reinterpret_cast<Process*>(p)->heap().make_array(len);
}

dream_result dream_array_set(dream_process* p, dream_value array, uint32_t index, dream_value v) {
    array = resolve(array);
    if (!is_obj(array, ObjType::Array)) return DREAM_BAD;
    auto* a = static_cast<ArrayObj*>(as_obj(array));
    if (index >= a->len) return DREAM_BAD;
    // A host writing into an array it did not just build may be putting an
    // unforced value into one a deep force has already walked, so the promise
    // that flag makes no longer holds -- and the target may be old while the
    // value is young, which the next minor collection needs to hear about.
    a->aux &= uint16_t(~AUX_DEEP_FORCED);
    reinterpret_cast<Process*>(p)->heap().remember_if_old(a, v);
    value_slot_store(&a->items()[index], v);
    return DREAM_OK;
}

dream_value dream_make_map(dream_process* p, uint32_t) {
    return reinterpret_cast<Process*>(p)->heap().make_map(0);
}

dream_value dream_map_insert(dream_process* p, dream_value map, dream_value key, dream_value v) {
    auto* proc = reinterpret_cast<Process*>(p);
    // A put shares rather than mutates, so the answer is a different map and
    // the caller has to keep it -- which is what this has always returned.
    return map_insert(*proc, resolve(map), key, v);
}

dream_result dream_map_get(dream_process* p, dream_value map, dream_value key, dream_value* out) {
    auto* proc = reinterpret_cast<Process*>(p);
    Value found;
    if (!map_lookup(*proc, resolve(map), key, &found)) return DREAM_BAD;
    if (out) *out = found;
    return DREAM_OK;
}

/* --- runtime facts --- */

uint32_t dream_vm_native_module_count(const dream_vm* vm) {
    return uint32_t(vm->rt.modules().size());
}

const char* dream_vm_native_module_name(const dream_vm* vm, uint32_t index) {
    if (index >= vm->rt.modules().size()) return nullptr;
    return vm->rt.modules()[index].name.c_str();
}

uint64_t dream_vm_process_count(const dream_vm* vm) {
    return const_cast<dream_vm*>(vm)->rt.live_process_count();
}

uint32_t dream_vm_module_count(const dream_vm* vm) {
    return vm->rt.has_image() ? vm->rt.image().module_count() : 0;
}

const char* dream_vm_module_name(const dream_vm* vm, uint32_t index) {
    if (!vm->rt.has_image() || index >= vm->rt.image().module_count()) return nullptr;
    // The image is mapped for the life of the VM and the string blob is
    // NUL-terminated by the emitter's padding, but do not rely on that: hand
    // back a stable copy instead.
    static thread_local std::string scratch;
    scratch = vm->rt.image().str(vm->rt.image().module(index).name).str();
    return scratch.c_str();
}

}  // extern "C"
