// Tensors: what `+ - * / %` do to them, what `@` is, and `std.tensor`.
//
// A tensor is a `TensorObj` (value.hpp): packed doubles with a shape, on the
// host, or a handle to a buffer on the GPU (gpu.hpp). Everything a program can
// do to one is here, and every operation answers a new tensor -- the value
// semantics every other Dream value has. docs/notes/tensors.md is the design.

#pragma once

#include <cstdint>
#include <string>

#include "image.hpp"
#include "runtime.hpp"
#include "value.hpp"

namespace dream {

class Process;

inline bool is_tensor(Value v) { return is_obj(v, ObjType::Tensor); }

/// `a op b` for `+ - * / %` where at least one side is a tensor: elementwise
/// between two tensors of one shape (or where one's shape is the trailing part
/// of the other's, which repeats it), and against a number on either side.
/// False with the error in `*out` when it raises.
bool tensor_arith(Process& p, Op op, Value a, Value b, Value* out);
/// The same for any `KernelOp` (tensor_kernels.hpp): the operators, and the
/// `std.tensor` functions `max`, `min`, `pow` and the comparisons.
bool tensor_binary(Process& p, int kop, Value a, Value b, Value* out);

/// Compute a deferred tensor (see `TensorExpr`) and keep the answer in it;
/// what `strict!` does to a tensor. False with the error when the device
/// refused.
bool tensor_force(Process& p, Value t, Value* err);

/// `==` between two tensors: the same shape and the same numbers.
bool tensor_equal(Process& p, Value a, Value b, bool* raised);
/// `compare` between two tensors: by shape, then element by element.
bool tensor_compare(Process& p, Value a, Value b, int* out);
/// How `to_string` and `console.print!` render one.
bool tensor_render(Process& p, Value t, std::string* out);
/// `t.[i]`: the element of a vector, or a copy of row `i` of anything with
/// more axes. 1 with the answer, 2 when `i` is out of range (the caller has an
/// `else` or raises), 0 with the error when it raised.
int tensor_index(Process& p, Value t, int64_t i, Value* out);
/// `len t`: the length of the first axis.
inline int64_t tensor_len(Value t) {
    return int64_t(static_cast<TensorObj*>(as_obj(resolve(t)))->dims[0]);
}

/// `a @ b`, which the compiler writes as this builtin: the matrix product, or
/// the dot product of two vectors.
NativeResult tensor_matmul_builtin(Process& p, Value callee, Value* args, uint32_t argc);

/// Materialize and copy numbers to a host tensor in the caller's heap.
/// Also used when data crosses an image or wire boundary.
bool tensor_host_copy(Process& p, Value t, Value* out);
NativeResult tensor_of_bytes_builtin(Process& p, Value callee, Value* args, uint32_t argc);

ModuleDef make_tensor_module();

}  // namespace dream
