#include "value.hpp"

namespace dream {

const char* obj_type_name(ObjType t) {
    switch (t) {
        case ObjType::Float: return "float";
        case ObjType::BigInt: return "integer";
        case ObjType::Str: return "string";
        case ObjType::BigStr: return "big string";
        case ObjType::Tensor: return "tensor";
        case ObjType::Cons: return "list";
        case ObjType::Array: return "array";
        case ObjType::Record: return "record";
        case ObjType::Map: return "map";
        // Never user-visible: a leaf only ever hangs off a branch.
        case ObjType::MapLeaf: return "map entry";
        case ObjType::Closure: return "function";
        case ObjType::Thunk: return "thunk";
        case ObjType::Blackhole: return "blackhole";
        case ObjType::Indirect: return "indirect";
        case ObjType::Pap: return "function";
        case ObjType::Frame: return "frame";
        case ObjType::Module: return "module";
        case ObjType::ErrorBox: return "error";
        case ObjType::Pid: return "process";
        case ObjType::Native: return "function";
        default: return "unknown";
    }
}

dream_type surface_type(Value v) {
    v = resolve(v);
    if (is_fixnum(v)) return DREAM_TYPE_INTEGER;
    if (is_imm(v)) {
        switch (imm_kind(v)) {
            case IMM_UNIT: return DREAM_TYPE_UNIT;
            case IMM_BOOL: return DREAM_TYPE_BOOL;
            case IMM_CHAR: return DREAM_TYPE_CHAR;
            case IMM_ATOM: return DREAM_TYPE_ATOM;
            case IMM_NIL: return DREAM_TYPE_LIST;
            default: return DREAM_TYPE_IMPURE_FN;  // builtins are impure by nature
        }
    }
    if (!is_ptr(v)) return DREAM_TYPE_UNKNOWN;
    switch (as_obj(v)->type) {
        case ObjType::Float: return DREAM_TYPE_FLOAT;
        case ObjType::BigInt: return DREAM_TYPE_INTEGER;
        case ObjType::Str: return DREAM_TYPE_STRING;
        case ObjType::BigStr: return DREAM_TYPE_BIGSTR;
        case ObjType::Tensor: return DREAM_TYPE_TENSOR;
        case ObjType::Cons: return DREAM_TYPE_LIST;
        case ObjType::Array: return DREAM_TYPE_ARRAY;
        case ObjType::Record: return DREAM_TYPE_RECORD;
        case ObjType::Map: return DREAM_TYPE_MAP;
        case ObjType::MapLeaf: return DREAM_TYPE_MAP;
        case ObjType::Module: return DREAM_TYPE_MODULE;
        case ObjType::ErrorBox: return DREAM_TYPE_ERROR;
        case ObjType::Pid: return DREAM_TYPE_PROCESS;
        case ObjType::Closure:
        case ObjType::Pap:
        case ObjType::Native:
            // Purity is a property of the function record; the caller
            // consults the image when it needs to tell the two apart.
            return DREAM_TYPE_PURE_FN;
        default: return DREAM_TYPE_UNKNOWN;
    }
}

}  // namespace dream
