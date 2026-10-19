// Builtins, host modules, and the map primitives shared with the interpreter.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <string>

#include "runtime.hpp"
#include "value.hpp"

namespace dream {

class Process;
class Heap;

struct BuiltinDef {
    const char* name;
    uint32_t arity;
    /// Bit i set means argument i is forced before the call. `spawn!` clears
    /// bit 0 deliberately: forcing the thunk would run the new process's work
    /// on the caller's stack, which is exactly what spawning must not do.
    uint32_t strict_mask;
    NativeFn fn;
    /// This builtin declares `VouchesForGc`. Same meaning and same consequence
    /// as `NativeDef::vouches`, which is where it is written down.
    bool vouches = false;
};

/// The table itself, declared here rather than hidden in `builtins.cpp` so
/// that looking a builtin up is the array index it is.
///
/// It was a call across a translation unit for a body of `return BUILTINS[id]`,
/// and the interpreter asks the question on the way into every native -- 1.15%
/// of a self-compile to fetch a pointer the caller could have computed itself.
/// Nothing writes to it; `builtin_count` still reports its extent, because the
/// size belongs to the definition.
///
/// Being data rather than a function is what makes it need `DREAM_DATA_API`.
/// On Windows the VM is a DLL built with `WINDOWS_EXPORT_ALL_SYMBOLS`, which
/// exports every function and no variable, so `dream.exe` and `dream_tests`
/// inlining `builtin_def` reached for a symbol the DLL never offered. CMake
/// defines `dream_EXPORTS` while it builds the library itself.
#if defined(_WIN32) && defined(dream_EXPORTS)
#define DREAM_DATA_API __declspec(dllexport)
#elif defined(_WIN32)
#define DREAM_DATA_API __declspec(dllimport)
#else
#define DREAM_DATA_API
#endif
extern DREAM_DATA_API const BuiltinDef BUILTINS[];
inline const BuiltinDef& builtin_def(uint32_t id) { return BUILTINS[id]; }
uint32_t builtin_count();

/// A float as the shortest decimal that reads back as it (as a float32 when
/// `f32`), positional from 1e-4 to 1e16 and with an exponent outside.
void append_shortest(double d, bool f32, std::string* out);

/// Insert into an open-addressed map, growing it when it gets crowded.
/// `map` with `key` set to `value`. The map handed in is unchanged and stays
/// valid: what comes back shares all of it but the path to the new entry.
Value map_insert(Process& p, Value map, Value key, Value value);
/// `std.trunk`'s natives, added to `std.vm`'s members (trunk.cpp).
void add_trunk_natives(std::vector<NativeDef>& members);
NativeResult core_trunk_of(Process& p, Value self, Value* args, uint32_t n);
NativeResult core_trunk_to_list(Process& p, Value self, Value* args, uint32_t n);
NativeResult core_trunk_lookup(Process& p, Value self, Value* args, uint32_t n);
NativeResult core_trunk_at(Process& p, Value self, Value* args, uint32_t n);
NativeResult core_trunk_push(Process& p, Value self, Value* args, uint32_t n);
NativeResult core_trunk_set(Process& p, Value self, Value* args, uint32_t n);
/// What `--stats` says trunks did, if any were made.
void print_trunk_stats();
/// `map` with `n` keys set, in order, a later one winning: what a run of
/// `map_insert`s answers, with each branch on the way copied once rather than
/// once per key. `pairs` is `k0, v0, k1, v1, ..`. `SetMany` on a map is this.
Value map_insert_many(Process& p, Value map, uint32_t n, const Value* pairs);
/// `map` without `key`, on the same terms.
Value map_erase(Process& p, Value map, Value key);
/// Every entry of a map, in the trie's order.
void map_collect(Value node, std::vector<std::pair<Value, Value>>& out);
/// `compare`'s order, which the wire formats write a map's entries in.
bool compare_for_wire(Process& p, Value a, Value b, int* out);
/// The packed wire form (pack.cpp): `vm.wire_pack`, and reading it back.
NativeResult vm_wire_pack(Process& p, Value self, Value* args, uint32_t n);
bool is_packed(const unsigned char* data, uint64_t len);
bool wire_unpack(Process& p, const unsigned char* data, uint64_t len, Value* out);
/// Look up a key. Returns false when it is absent.
bool map_lookup(Process& p, Value map, Value key, Value* out);

/// `Result.ok value`, or `Result.error value` when `ok` is false: the
/// prelude's union (`std.prelude`), which is how a native answers something
/// that can fail without raising. A variant record of one field, stamped as
/// the compiler stamps the union's own constructors, so a program's
/// `Result.ok(v)` pattern takes it apart. `value` is held across the one
/// allocation, which is no safepoint.
Value make_result(Heap& h, Runtime& rt, bool ok, Value value);

/// Modules this runtime provides to `import`.
ModuleDef make_math_module();
ModuleDef make_vm_module();
ModuleDef make_ffi_module();

}  // namespace dream
