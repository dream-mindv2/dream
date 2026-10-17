// Fixed-arity runtime primitives. X(enum name, opcode, builtin ID, arity).
// Keep IDs in sync with BUILTINS and dreams/src/builtins.dr.
#pragma once

#define DREAM_PRIMITIVES(X) \
    X(StrChars, 47, 19, 1) \
    X(ErrorNew, 48, 23, 2) \
    X(ErrorKind, 49, 24, 1) \
    X(ErrorPayload, 50, 25, 1) \
    X(StrSlice, 51, 26, 3) \
    X(StrFind, 52, 27, 3) \
    X(StrByte, 53, 28, 2) \
    X(StrLe, 54, 29, 2) \
    X(StrSpan, 55, 30, 3) \
    X(StrUpto, 56, 31, 3) \
    X(CharCode, 57, 32, 1) \
    X(CharOfCode, 58, 33, 1) \
    X(ToFloat, 59, 34, 1) \
    X(FloatBytes, 60, 35, 1) \
    X(FloatOfBytes, 61, 36, 1) \
    X(ToInt, 62, 37, 1) \
    X(ParseInt, 63, 38, 1) \
    X(ParseFloat, 64, 39, 1) \
    X(ToExistingAtom, 65, 40, 1) \
    X(ArrayNew, 66, 41, 2) \
    X(ArrayToList, 67, 43, 1) \
    X(MapHas, 68, 44, 2) \
    X(MapRemove, 69, 45, 2) \
    X(MapPairs, 70, 46, 1) \
    X(DataCount, 71, 47, 1) \
    X(DataAt, 72, 48, 1) \
    X(Compare, 73, 49, 2) \
    X(StrInterp, 74, 52, 1)

#define DREAM_PRIMITIVE_CASE(name, code, builtin, arity) case Op::name:
#define DREAM_PRIMITIVE_CASES DREAM_PRIMITIVES(DREAM_PRIMITIVE_CASE)
