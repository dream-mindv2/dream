// Integers past 63 bits.
//
// `:integer` has two representations: a fixnum, which is every integer a
// program is likely to meet and which costs a shift to read, and a
// `BigIntObj`, which is the rest. Nothing chooses between them but the value:
// an operation whose answer leaves the fixnum range answers a bignum, and an
// operation on bignums whose answer comes back into it answers a fixnum. The
// boundary is invisible to a program -- `type_of` says `:integer` for both,
// `==` and `compare` and map keys agree across it -- and it is the *only*
// place the bignum code is reached from: the fixnum fast paths of the
// interpreter and of compiled code are untouched, and each of them already
// handed overflow to `arith`, which used to answer a float there. That is the
// whole of the cost to a program that does not use bignums: nothing.
//
// **Representation.** A sign and a magnitude in 64-bit limbs, least
// significant first. Sign-magnitude rather than two's complement because
// the arithmetic algorithms are the textbook ones over unsigned words.
// Bitwise operations convert to two's complement temporarily.
//
// **The algorithms.** Addition and subtraction are linear. Multiplication is
// schoolbook below `kKaratsubaThreshold` limbs and Karatsuba above, with an
// unbalanced product split into balanced ones. Division is Knuth's algorithm
// D, with a single-limb divisor taken by a loop of its own, which is also what
// printing in base 10^19 is made of. Conversion to and from decimal is
// quadratic, as CPython's is; a program printing a hundred-thousand-digit
// number pays a tenth of a second for it.
//
// **Where the words come from.** Each operation that makes a bignum allocates
// the largest result it could have straight on the process heap, computes into
// it, and trims (`finish`). `Heap::alloc` never collects, so the limbs of the
// operands -- also on the heap -- stay where they are throughout, and nothing
// is staged through malloc except Karatsuba's and division's scratch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "value.hpp"

namespace dream {

class Heap;

namespace bigint {

/// Whether `v` is an integer in either representation. `v` must be resolved.
inline bool is_integer(Value v) { return is_fixnum(v) || is_obj(v, ObjType::BigInt); }
inline bool is_big(Value v) { return is_obj(v, ObjType::BigInt); }

// The arithmetic. Both operands are integers (fixnums or bignums) and the
// answer is canonical: a fixnum whenever it fits. Division truncates toward
// zero and the remainder takes the dividend's sign, as fixnum `/` and `%` do;
// a zero divisor is the caller's to refuse before asking.
Value add(Heap& h, Value a, Value b);
Value sub(Heap& h, Value a, Value b);
Value mul(Heap& h, Value a, Value b);
Value quot(Heap& h, Value a, Value b);
Value rem(Heap& h, Value a, Value b);
Value negate(Heap& h, Value a);
Value abs(Heap& h, Value a);

// The bitwise operations, on the infinite two's complement every integer has
// in principle: a negative number has ones above its top bit forever, so
// `-1 & x` is `x` and `~x` is `-x - 1`, as in Python. Both operands are
// integers; the answer is canonical.
Value bit_and(Heap& h, Value a, Value b);
Value bit_or(Heap& h, Value a, Value b);
Value bit_xor(Heap& h, Value a, Value b);
Value bit_not(Heap& h, Value a);
/// `a * 2^k`.
Value shift_left(Heap& h, Value a, uint64_t k);
/// `a / 2^k` rounded toward negative infinity, which is an arithmetic shift:
/// a negative number stays negative and never reaches zero, only -1.
Value shift_right(Heap& h, Value a, uint64_t k);

/// -1, 0 or 1, exactly.
int compare(Value a, Value b);
/// Content equality of two bignums. A fixnum is never equal to a bignum,
/// because a bignum is never a value a fixnum could hold.
bool equal(const BigIntObj* a, const BigIntObj* b);
uint64_t hash(const BigIntObj* a);

/// The nearest double, ties to even; past the range of a double, infinity.
double to_double(Value v);
/// A finite double truncated toward zero, exactly. False for NaN or infinity.
bool from_double(Heap& h, double d, Value* out);

Value from_int64(Heap& h, int64_t v);
Value from_uint64(Heap& h, uint64_t v);
/// The value if it fits; false otherwise.
bool to_int64(Value v, int64_t* out);
bool to_uint64(Value v, uint64_t* out);

/// The low `width` bytes of `v`'s two's complement, little-endian, sign
/// extended past its top: what `_str_le` writes and the wire format carries.
void to_le_bytes(Value v, size_t width, uint8_t* out);
/// The fewest bytes whose two's complement holds `v`.
size_t le_width(Value v);
/// The integer whose two's complement is these `n` bytes, little-endian.
Value from_le_bytes(Heap& h, const uint8_t* bytes, size_t n);

/// Decimal, with a leading `-` when negative.
std::string to_string(Value v);
void append_decimal(Value v, std::string* out);
/// An optional sign and one or more decimal digits, the whole of the text.
/// False for anything else, including an empty string.
bool parse(Heap& h, const char* s, size_t n, Value* out);

}  // namespace bigint
}  // namespace dream
