#pragma once
#include <cstdint>
#include <limits>

namespace dream {
inline bool add_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(a, b, result);
#else
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return true;
    *result = a + b;
    return false;
#endif
}
inline bool sub_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_sub_overflow(a, b, result);
#else
    if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b)) return true;
    *result = a - b;
    return false;
#endif
}
inline bool mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, result);
#else
    if (a > 0 ? (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
              : (a < 0 && (b > 0 ? a < INT64_MIN / b : b < 0 && a < INT64_MAX / b))) return true;
    *result = a * b;
    return false;
#endif
}
}
