# Bignums, and the cryptography beside them

## Integers past 63 bits (2026-10-03)

`:integer` is now unbounded. A value that fits in 63 bits is a fixnum, as
before; one that does not is a `BigIntObj` -- a sign and 64-bit limbs, least
significant first -- and `type_of`, `==`, `compare`, map keys, `match` and the
wire format all treat the two as one type. The design is in
[dream/src/bigint.hpp](../../src/bigint.hpp); this is what was measured.

### What it costs a program that never makes one: nothing measurable

Every fixnum fast path -- `finish_binary`, `operand_value`'s eager arithmetic,
the JIT's inline add/sub/mul and its tagged compares -- was already handing
overflow to `arith`, which answered a float there. The bignum code hangs off
that one exit and nowhere else, so the paths a fixnum takes are the same
instructions they were. The places that did change are all off the fixnum
path: `arith`'s float case asks one more type question, and the JIT's inline
`:integer` type test and its int-to-double conversion grew a second branch
that a fixnum never takes.

A/B against the commit before, same compiler, runs interleaved:

| | HEAD VM | bignum VM |
|---|---|---|
| self-compile, best of 6 | 1.717 s | 1.714 s |
| self-compile, median of 6 | 1.753 s | 1.735 s |
| `benchmark/benchmark` (7 workloads) | | equal to the millisecond |

The two VMs write byte-identical images.

### What it costs a program that does: CPython's ballpark, and ahead on products

`fact` is `acc * n` 20,000 times (a bignum times a fixnum), `fib` is 100,000
additions of numbers up to 1,100 limbs, `pow` is 3^1,000,000 by squaring, and
`divmod` divides a 506,000-digit number by a 253,000-digit one. Milliseconds,
Python 3.13.

| | Dream | CPython |
|---|---|---|
| fact 20000 | 72-80 | 55 |
| fib 100000 | 47 | 54 |
| 3 ^ 1000000 | 16 | 42 |
| divmod | 167 | 121 |

The products are ahead because the limbs are 64 bits (CPython's are 30) and
multiplication switches to Karatsuba at 32 limbs. Division is Knuth's
algorithm D with `divq` doing each step on x86-64 (`__int128` division calls
`__udivti3`, a general 128-by-128 routine several times slower).

### The first version was 6x slower on `fib`, and the heap was why

`fib 100000` took **336 ms** at first, against 21 ms for the first 45,000
steps -- every number under 4 KiB. Past 4 KiB a heap object is a *big object*:
tenured at birth into a block of its own, and freed only by a major
collection. A loop of bignums makes one per step and drops it the next, which
is the opposite of what that path was built for (tensors: few, large,
long-lived). `--stats` said it plainly: 1.1 GB held from the OS for 85 MB
allocated, 85 majors.

Three things were wrong with that path for this load, and fixing the first
two helped the third most:

- **Every big block was at least 64 KiB** (`new_block` rounds up to the heap's
  minimum block, which is right for a shared block and nothing but waste for
  a dedicated one). A 5 KiB bignum held 64. Sized exactly: 1.1 GB -> 190 MB
  held, 336 -> 230 ms.
- **The pool of dead big blocks was a flat list** searched end to end, and
  erased from the middle, at every allocation -- a thousand entries after a
  sweep of bignums. Filed by size instead (`std::map` of vectors, take the
  last retired): 230 -> 184-216 ms. Both changes help any program that makes
  big strings or arrays in a loop as well.
- **The limbs should not be on the heap at all.** A large bignum is now a
  24-byte young header with its limbs in `malloc`'d memory, listed on
  `external_` exactly as a GPU tensor's buffer is, so the first *minor*
  collection that finds the header unreached frees them (`reap_external`).
  Nothing tenures, nothing waits for a major: **47 ms**, 17 MB held, 2 ms of
  collection pauses. A bignum small enough for a size class stays inline,
  where the nursery already does the right thing.

The idea for the third was Blake's: "handle bignums kind of like we handle
tensors".

### Literals

A literal past the fixnum range lowers to `_parse_int "<digits>"`
(`const_int` in dreams/src/lower.dr), so the image format's `KINT` stays an
`i64`. The seed was regenerated with that lowering in it, so a big literal is
fine anywhere, the compiler's own source included; a seed older than it dies
on one with `str_le needs an integer`.

### Semantics that changed

- Overflow is exact where it was a float. Three recorded outputs moved
  (`jit_integer`, `jit_lists`, `type_tests`), each to the exact answer.
- `num.parse_int` reads any number of digits; `num.to_int` and `math.floor` of
  a float past 2^62 answer the exact integer (and of an infinity or NaN raise
  `:type_error`, where they were undefined behaviour in C++).
- `std.wire` carries any integer: tag 3 is now the whole 64-bit range, and tag
  11 is wider, as long as its two's complement needs.
- `std.sql.pg` decodes every `int8` and every whole `numeric` as an integer;
  they used to come back as text past 2^62.
- `_str_le` writes a bignum, and any width.

## `std.crypto` (2026-10-03)

The hashes that were Dream code -- MD5 over nibble tables in
`std.sql.pg.crypto`, PBKDF2 as four thousand rounds of Dream around native
SHA-256 -- are natives now: MD5, SHA-1, SHA-256/384/512, HMAC, PBKDF2, HKDF,
constant-time comparison, hex and base64, and the OS's random bytes.
[docs/builtins.md](../builtins.md) has the table. Written out in
`dream/src/digest.cpp` rather than taken from OpenSSL, because on Windows the
TLS library is SChannel and a hash from libcrypto would be one Windows lacked.
PBKDF2 hashes the key's two padded blocks once and copies the state for every
round, so a round is two compressions. `dream/tests/programs/crypto.dr` checks
58 vectors against Python's `hashlib` and `hmac`.
