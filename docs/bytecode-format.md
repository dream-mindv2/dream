# Dream bytecode format (`.dream`), version 0.1

A `.dream` file holds one compiled module. It is designed around a single
constraint: **the VM evaluates lazily, so the program *is* a tree of nodes, and
loading must not rebuild that tree.**

Everything is therefore stored as flat arrays of fixed-size, little-endian
records, and every edge is an *index* rather than a pointer or an offset. A
loader can `mmap` the file, slice the `NODE` and `KIDS` sections, and begin
forcing nodes immediately — no parsing, no relocation, no allocation
proportional to program size.

A compiled image is shared unchanged across Linux, Windows, and macOS VMs.
Native addresses and JIT machine code are never serialized into it. See
[platform portability](platforms.md) for the host requirements and exceptions.

All multi-byte integers are little-endian. All section offsets are 8-byte
aligned. The sentinel `0xFFFFFFFF` (`NO_NODE`) marks an absent edge.

## Layout

```
+--------------------------+  0
| Header                   |  32 bytes
+--------------------------+  32
| Section table            |  16 bytes x section_count
+--------------------------+
| Sections, in table order |  each padded to an 8-byte boundary
+--------------------------+
```

### Header (32 bytes)

| Offset | Size | Field           | Notes                                        |
|-------:|-----:|-----------------|----------------------------------------------|
| 0      | 8    | `magic`         | ASCII `DAGNCAAF`                             |
| 8      | 2    | `version_major` | `0`; a mismatch must be rejected             |
| 10     | 2    | `version_minor` | `1`; a higher minor is forward-compatible    |
| 12     | 4    | `flags`         | bit 0: a `SPAN` debug section is present; bits 8-23: the target (below) |
| 16     | 4    | `module_name`   | string-table index                           |
| 20     | 4    | `source_name`   | string-table index; the original source path, or as `--remap-path-prefix` rewrote it |
| 24     | 4    | `entry`         | function index of `main!`, or `NO_NODE`      |
| 28     | 4    | `section_count` |                                              |

### The image's target

Bits 8-15 of `flags` are the operating systems the image may run on and bits
16-23 the architectures, each a set with one bit per member:

| Bits | Field | `0x01` | `0x02` | `0x04` | `0x08` |
|---|---|---|---|---|---|
| 8-15 | systems | linux | macos | windows | android |
| 16-23 | architectures | x86_64 | aarch64 | | |

An empty set means **any**, so every image written before these bits existed,
and every image whose program does not care, means what it always did. A
non-empty set must contain the running machine's bit. The two sets are
independent: an image carrying a Linux x86-64 library and a Windows ARM64 one
records both systems and both architectures, and so also admits Linux on
ARM64. The header cannot say more, and a library for the wrong machine fails
where it always did.

The compiler decides the sets (`dreams/src/target.dr`). `--target` states them,
and `--target any` leaves them empty. Otherwise they come from what the
program is made of: a payload that is an ELF, Mach-O or PE binary contributes
its system and machine, and a `when` condition that read `os`, `family` or
`arch` pins the image to the value it was compiled with. The `dream` command
refuses a mismatched image before running it, and `--any-target` overrides
that. The check belongs to the command and not to the loader, because the
compiler runs images of a cross-built program for `comp` on its own machine.

Constraints that are not a set of platforms, such as a minimum libc or a
minimum VM, would go in a section of their own, which does not exist yet.

### Section table entry (16 bytes)

| Offset | Size | Field    | Notes                                          |
|-------:|-----:|----------|------------------------------------------------|
| 0      | 4    | `kind`   | four ASCII bytes, read as a little-endian `u32` |
| 4      | 4    | `offset` | absolute byte offset into the file              |
| 8      | 4    | `length` | byte length, excluding alignment padding        |
| 12     | 4    | `count`  | number of records (bytes, for `SBLB`)           |

Unknown section kinds must be skipped, not treated as an error.

## Sections

| Kind   | Record                                              | Purpose |
|--------|-----------------------------------------------------|---------|
| `KINT` | `i64`                                               | integer constants |
| `KFLT` | `f64` (raw bit pattern)                             | float constants |
| `KSTR` | `u32 offset, u32 length`                            | string table, into `SBLB` |
| `SBLB` | raw UTF-8 bytes                                     | string character data |
| `KATM` | `u32`                                               | atom table; each entry is a `KSTR` index |
| `NODE` | 16 bytes (below)                                    | the execution trees |
| `KIDS` | `u32`                                               | variadic child lists and capture descriptors |
| `FUNC` | 32 bytes (below)                                    | function records |
| `GLBL` | 16 bytes (below)                                    | module-level bindings |
| `IMPT` | `u32 path, u32 alias`                               | imports; both are `KSTR` indices |
| `MODS` | 24 bytes (below)                                    | module records: which globals belong to which module |
| `SPAN` | `u32 start, u32 end`                                 | per-node source spans, parallel to `NODE`; reserved, never written |
| `ITYP` | `u32 func, u32 reserved, u64 integer_params`        | integer signature hints for the JIT |
| `TYPE` | `u32 func, u32 reserved, u64 float_params`          | what signatures declared, for the JIT (below) |
| `VMOP` | `u32 key, u32 reserved, u64 value`                  | VM settings from the root package's `[vm]` table; optional (below) |
| `LDAT` | `u64 offset, u64 length`                            | large-data table, into `PAYL` |
| `LNAM` | `u32 length`, then the bytes, per datum, unpadded   | payload names; optional |
| `PAYL` | `u64 byte_length` header, then the bytes            | the payload; last in the file |

Atoms are interned separately from strings so the VM can compare them by
identity: two occurrences of `:ok` anywhere in a module share one atom index.

### Node (16 bytes)

| Offset | Size | Field   |
|-------:|-----:|---------|
| 0      | 1    | `op`    |
| 1      | 1    | `flags` |
| 2      | 2    | `aux`   | reserved |
| 4      | 4    | `a`     |
| 8      | 4    | `b`     |
| 12     | 4    | `c`     |

**Flags.** These are the compiler's answer to "when does this run?", which a
lazy VM cannot work out for itself:

- `0x01 STRICT` — force this node where it appears. Set on block statements
  whose effects must be ordered, on `if` conditions, and on `try!` bodies
  (whose result must materialize while the handler is still installed).
- `0x02 IMPURE` — this node performs, or transitively contains, an effect.
- `0x04 TAIL` — a call in tail position; the VM may reuse the frame.

**Opcodes.**

| # | Op | `a` | `b` | `c` |
|--:|----|-----|-----|-----|
| 0 | `nop` | | | |
| 1 | `int` | `KINT` index | | |
| 2 | `float` | `KFLT` index | | |
| 3 | `str` | `KSTR` index | | |
| 4 | `char` | Unicode scalar value | | |
| 5 | `bool` | 0 or 1 | | |
| 6 | `atom` | `KATM` index | | |
| 7 | `unit` | | | |
| 8 | `local` | frame slot | | |
| 9 | `capture` | capture index | | |
| 10 | `global` | `GLBL` index | | |
| 11 | `builtin` | builtin id | | |
| 12 | `field` | object node | `KSTR` index of the name | |
| 13 | `apply` | callee node | `KIDS` offset | argument count |
| 14 | `if` | condition | then | else |
| 15 | `block` | `KIDS` offset | statement count | |
| 16 | `bind` | frame slot | value node | |
| 17 | `closure` | `FUNC` index | | |
| 18 | `thunk` | `FUNC` index | | |
| 19 | `try` | body | handler | slot bound to the error |
| 20 | `force` | node | | |
| 21–25 | `add` `sub` `mul` `div` `mod` | lhs | rhs | |
| 26–31 | `eq` `ne` `lt` `le` `gt` `ge` | lhs | rhs | |
| 32–33 | `and` `or` | lhs | rhs (forced only if needed) | |
| 34–35 | `neg` `not` | operand | | |
| 36–37 | `list` `array` | `KIDS` offset | element count | |
| 38 | `map` | `KIDS` offset | pair count (`2n` kids: `k0 v0 k1 v1 …`) | |
| 39 | `get` | container | key | fallback, or `NO_NODE` |
| 40 | `set` | container | key | value |
| 41 | `switch_head` | subject | `KIDS` offset | kid count (`2n + 1`: `key0 target0 … default`) |
| 42 | `switch_atom` | subject | `KIDS` offset | kid count, as above |
| 43 | `type_is` | subject | surface type (0–9) | invert (0 or 1) |
| 44 | `cons` | head node (lazy) | tail node (lazy) | — |
| 45 | `list_tail` | list node | — | — |
| 46 | `list_is_empty` | value node | — | — |
| 47 | `str_chars` | builtin ID 19 | kids offset | argument count 1 |
| 48 | `error_new` | builtin ID 23 | kids offset | argument count 2 |
| 49 | `error_kind` | builtin ID 24 | kids offset | argument count 1 |
| 50 | `error_payload` | builtin ID 25 | kids offset | argument count 1 |
| 51 | `str_slice` | builtin ID 26 | kids offset | argument count 3 |
| 52 | `str_find` | builtin ID 27 | kids offset | argument count 3 |
| 53 | `str_byte` | builtin ID 28 | kids offset | argument count 2 |
| 54 | `str_le` | builtin ID 29 | kids offset | argument count 2 |
| 55 | `str_span` | builtin ID 30 | kids offset | argument count 3 |
| 56 | `str_upto` | builtin ID 31 | kids offset | argument count 3 |
| 57 | `char_code` | builtin ID 32 | kids offset | argument count 1 |
| 58 | `char_of_code` | builtin ID 33 | kids offset | argument count 1 |
| 59 | `to_float` | builtin ID 34 | kids offset | argument count 1 |
| 60 | `float_bytes` | builtin ID 35 | kids offset | argument count 1 |
| 61 | `float_of_bytes` | builtin ID 36 | kids offset | argument count 1 |
| 62 | `to_int` | builtin ID 37 | kids offset | argument count 1 |
| 63 | `parse_int` | builtin ID 38 | kids offset | argument count 1 |
| 64 | `parse_float` | builtin ID 39 | kids offset | argument count 1 |
| 65 | `to_existing_atom` | builtin ID 40 | kids offset | argument count 1 |
| 66 | `array_new` | builtin ID 41 | kids offset | argument count 2 |
| 67 | `array_to_list` | builtin ID 43 | kids offset | argument count 1 |
| 68 | `map_has` | builtin ID 44 | kids offset | argument count 2 |
| 69 | `map_remove` | builtin ID 45 | kids offset | argument count 2 |
| 70 | `map_pairs` | builtin ID 46 | kids offset | argument count 1 |
| 71 | `data_count` | builtin ID 47 | kids offset | argument count 1 |
| 72 | `data_at` | builtin ID 48 | kids offset | argument count 1 |
| 73 | `compare` | builtin ID 49 | kids offset | argument count 2 |
| 74 | `str_interp` | builtin ID 52 | kids offset | argument count 1 |
| 75 | `band` | left node | right node | — |
| 76 | `bor` | left node | right node | — |
| 77 | `bxor` | left node | right node | — |
| 78 | `bnot` | operand node | — | — |
| 79 | `shl` | left node | count node | — |
| 80 | `shr` | left node | count node | — |
| 81 | `record_make` | kids offset | field count | atom index, top bit set for a variant |
| 82 | `set_many` | container node | kids offset | pair count |
| 83 | `ensure` | body node | cleanup node | |

`record_make` builds a value of a declared `group`, `struct` or union variant
(`RecordObj`): the fields in a lazy position, as `array`'s items are, and the
atom naming the declaration -- `app.Point`, or `app.Shape.circle` with
`c`'s top bit set. Validation checks the kids and that the atom index is in
the atom table. A part of a parallel build numbers atoms for itself, so the
merge renumbers this operand as it does `atom`'s.

`set_many` is a chain of `set`s at constant keys made one: its kids are
`k0, v0, k1, v1, ..`, every key an `int` or `atom` node (validation refuses
anything else, since the machine reads a key without evaluating it), every
value stored unforced. It answers what the chain would -- a later key wins, and
a bad key raises what that `set` would have -- but copies a record or an array
once. A map takes the pairs as inserts and a list as `set`s, one at a time.
The compiler emits it for a pipeline of setters, `p |> P.set_a 1 |> P.set_b 2`.


`type_is` forces its subject to WHNF and compares its runtime type, answering a
bool. The kinds follow `dream_type`: integer, float, char, bool, unit, string,
atom, list, array, map (0 through 9). With `c = 1` the answer is inverted.
The optimizer fuses `type_of x == :kind` and `!=` (also with the literal on
the left) into this instruction. This includes the checks emitted for container
patterns. Function kinds and other atoms retain the ordinary comparison.
These are runtime checks, never assumptions based on gradual signatures.
The JIT tests tags and guarded object headers directly; a value already carried
as an unboxed double has a known type, but its computation still runs.

A `block` evaluates to its last statement. An empty block evaluates to unit.

`ensure` evaluates its body, saves the answer or error, then evaluates cleanup
in the same frame. It returns the saved answer, or propagates the saved error.
A cleanup error replaces a normal answer but preserves an existing error.
Nested ensures unwind from the inside out, including when a process is killed.

`get` and `set` are generic over containers: a map by key, an array or a list by
position. Both force the container and then the key. A `get` enters the element
it finds -- the machine forces it, not the operation -- and, when there is none,
evaluates its fallback, or raises `:no_such_key` for a map and `:out_of_bounds`
for an array or a list when it has none. A `set` stores its value unforced and
answers a new container: a copy of an array, the changed path of a map's trie,
or the cells of a list in front of the position, with everything behind it
shared. The container it was given is unchanged. A list is walked a cell at a
time on the continuation stack, never recursively.

`switch_head` and `switch_atom` are how a `match` skips the arms that cannot
match. Each key is an `atom` node, followed by the node to evaluate when the
key is that atom; the last kid is the node for every other key. `switch_atom`
keys on its subject, when the subject is an atom. `switch_head` keys on the
head of a list cell, or the first element of a non-empty array (a `union
struct` variant), which it forces first; any other subject, or one whose head
is not an atom, takes the default. The subject is forced to weak
head normal form. The compiler emits these only where the tests they skip could
not have matched and where the chain of arm tests would itself have forced the
same head first, so a VM that ran every target's arm tests from the top would
give the same answers. It is only doing more work. Validation: `c` is odd, the
run is in range, and every key is an `atom` node.

### Function record (32 bytes)

| Offset | Size | Field          | Notes |
|-------:|-----:|----------------|-------|
| 0      | 4    | `name`         | `KSTR` index |
| 4      | 4    | `body`         | `NODE` index |
| 8      | 2    | `arity`        | declared parameters; they occupy slots `0..arity` |
| 10     | 2    | `flags`        | see below |
| 12     | 2    | `slots`        | frame size: parameters plus block-local bindings |
| 14     | 2    | `n_captures`   | |
| 16     | 4    | `captures_off` | `KIDS` offset of `n_captures` capture descriptors |
| 20     | 4    | `line`         | 1-based line the function was written on; 0 when not known |
| 24     | 4    | `col`          | 1-based column (bytes) |
| 28     | 4    | `source`       | `KSTR` index of the file, **plus one**; 0 when not known |

The place is what an error reports where it was raised from: the VM walks
the continuation stack when an error first unwinds and keeps the functions it
finds there, running and having made a value being forced, and an uncaught
error is printed with a line for each (`locate_error` in
dream/src/interp.cpp, `error.trace!` in std/error.dr). It is a function's place
and not a node's because the optimizer shares identical nodes across the whole
program. These bytes were a byte span and a reserved zero once, which nothing
read; an image written then reads as "not known" everywhere, since `source` is
zero.

Function flags: `0x01 IMPURE`, `0x02 REC`, `0x04 THUNK` (a generated 0-arity
body), `0x08 GLOBAL_VALUE` (a pure top-level value: force once, then memoize).

A 0-arity function is a lazy value when pure and an action when impure. That
distinction is the whole difference between `let x = expensive ()` — computed at
most once — and `let go! = { print! "hi" }`, which must print on every call.

**Capture descriptors.** Each is a `u32` read at closure-creation time against
the *defining* frame:

- high bit clear: take slot `value` of the parent frame
- high bit set: take capture `value & 0x7FFFFFFF` of the parent's capture list

Chaining through the second form is what lets a deeply nested closure reach a
variable several frames up without the intermediate frames referring to it.
Captures bind the parent's *cell*, not its current value, so a `let rec` closure
can capture its own binding before it is filled in.

### Global record (16 bytes)

| Offset | Size | Field    | Notes |
|-------:|-----:|----------|-------|
| 0      | 4    | `name`   | `KSTR` index |
| 4      | 4    | `kind`   | 0 = function, 1 = module |
| 8      | 4    | `target` | `FUNC` index, or `IMPT` index for a module |
| 12     | 4    | `flags`  | `0x01` exported, `0x02` impure |

### Module record (24 bytes)

| Offset | Size | Field           | Notes |
|-------:|-----:|-----------------|-------|
| 0      | 4    | `name`          | `KSTR` index |
| 4      | 4    | `source`        | `KSTR` index; the original source path, or as `--remap-path-prefix` rewrote it |
| 8      | 4    | `globals_start` | first global index belonging to this module |
| 12     | 4    | `globals_count` | |
| 16     | 4    | `flags`         | `0x01` host-provided, `0x02` declares virtuals |
| 20     | 4    | `derives`       | `MODS` index of the derived module, or `NO_NODE` |

Compilation is whole-program, so an image carries every module it needs, and a
module's globals form a contiguous range `globals_start .. globals_start +
globals_count`. The range is what lets a reader say which global belongs to
which module without every `GLBL` record carrying a module id of its own — and
it is the identity that survives a merge of two images, which is why the record
is here rather than being reconstructible from the rest.

## Declared types (`TYPE`)

One record per function whose signature declares a `:float` parameter:
`func` is a `FUNC` index, `reserved` is zero, and bit `i` of `float_params` is
set when parameter `i` is declared a float. A function with no such parameter
has no record, and a program with none has no section, so an image built from
a program without a float in a signature is byte-identical to one built before
the section existed. Masks cover the first 62 parameters.

It is a **hint**. Dream's types are gradual -- code with no signature may call
a `:float` function with an integer, and is entitled to the answer the
interpreter gives -- so nothing may be *trusted* on the strength of this
section. The JIT uses it to choose a representation (a raw double rather than
a tagged value) and guards every value it describes, handing the call back to
the interpreter when the guard fails. A reader that ignores the section runs
every program correctly; one that ignores it loses only speed.

**Validation.** `count * 16 <= length`, and every `func` is below the number of
function records.

## Integer hints (`ITYP`)

The optional `ITYP` section uses the same 16-byte record layout and validation
as `TYPE`, with a mask of declared `:integer` parameters instead. It covers the
first 62 parameters and is omitted when no signature supplies a hint. Aliases,
integer literal types, and unions consisting entirely of integer types qualify.
Older readers can skip the section without changing program behavior.

The JIT uses these hints to favor the checked fixnum arithmetic path in code
layout. Values remain tagged; neither the signature nor propagated arithmetic
hints prove that a value fits a fixnum. Runtime type checks, overflow handling,
and slow paths remain active, including for calls from untyped code.

## VM settings (`VMOP`)

The optional `VMOP` section is what the program's root package asked of the VM
in its manifest's `[vm]` table, one 16-byte record per setting it made: a key,
a reserved word (zero), and a `u64` value. A program that made none has no
section, and its image is the bytes it would have been without one. The image
carries these because it is the one thing every way of running the program has
in hand; a build tool may not be there at all.

| Key | `[vm]` setting | Value |
|---|---|---|
| 1 | `nursery_max` | the most a process's nursery may grow to, in bytes |

The VM reads the section when it loads the image, before the first process, and
the person running the program overrules it: `dream --nursery-max` first, then
`DREAM_NURSERY_MAX`, then this. A key the VM does not know is ignored, so a
setting added later does not stop an older VM from running the image.

## Large data (`LDAT` and `PAYL`)

Everything above is addressed with `u32`: a section table entry is `u32 offset,
u32 length`, a `KSTR` record is `u32 offset, u32 length`, and the VM's `StrObj`
counts its bytes in a `uint32_t`. Between them they cap a `.dream` file, and any
string in it, at 4 GiB. That is plenty for a program and nothing for the datum a
program is *about* — a corpus, a model, an asset. The payload is the one region
allowed past that line, and two properties keep it from disturbing anything
else:

- **Nothing already in the file moves.** Everything up to and including `LDAT`
  stays inside its `u32` addresses. Only the payload crosses, because it is the
  last thing in the file and nothing else has to name a position inside it.
- **Loading does not copy it.** A large datum materializes as a *view* into the
  mapped image, not as a heap string. That is the whole point: the region may be
  larger than the address space the rest of the format can describe, and larger
  than memory.

Both sections are additive. A reader that knows neither skips them as unknown
kinds, exactly as it does any section it does not recognise, so an image that
never uses its payload runs anywhere.

**`PAYL`** is the payload: an 8-byte little-endian `byte_length`, then that many
raw bytes. Its section-table entry is ordinary — `offset` names the header (a
start below 2^32, since everything precedes it), `length` is `8`, the honest
size of the section as laid out, and `count` is `0`. The real size is the `u64`
at the section start, and it may exceed 4 GiB. The payload ends the file and its
bytes are not padded.

**`LDAT`** is the table of payload descriptors, `u64 offset, u64 length`, with
offsets counted from the start of the payload bytes (`PAYL.offset + 8`). It
exists so that a `u32` index can name a datum without a 64-bit offset appearing
at every use. Descriptors are consecutive and unpadded.

Neither may appear without the other: a table describing nothing, or bytes
nothing can name, is an image that was built wrong.

**`LNAM`** names the data, in `LDAT`'s order: for each, a `u32 length` and
that many bytes, consecutive and unpadded, with `count` the number of names.
`dreams --payload NAME=FILE` writes it (a bare `FILE` is named for its last
path component), and it is what lets a program ask for a datum by what it is
-- `std.foreign`'s `[:payload, "sqlite"]` -- rather than by where it fell on
the command line. It is optional and additive like the other two: an image
without it is reached by index, and a reader that does not know it skips it.
It must not appear without a payload, and may not name more data than `LDAT`
describes.

**Validation.** `count * 16 <= length` for `LDAT`; each descriptor must satisfy
`offset + length <= byte_length` in 64-bit arithmetic, written as a subtraction
because that addition is the one sum in the container that can wrap; and the
payload must satisfy `PAYL.offset + 8 + byte_length <= file size`.

**No opcode names it.** The payload is a host-level feature: nothing in `NODE`,
`KIDS`, `FUNC` or `KSTR` points at it. A program reaches it through the
primitives `_data_count` and `_data_at` (see [builtins.md](builtins.md)),
which is what keeps the addition additive — the format needed no new opcode and
no wider index.

**What a reader hands back.** `_data_at i` (`payload.at i` in `std`) is O(1) and copies nothing: it is a
length and a pointer into the mapping. The reference VM materializes it as a
`BigStrObj`, which the collector treats as an atom object — no internal
references — so promotion is a byte copy and a `spawn!` heap copy shares the
view. That is sound because the bytes live in the program image, which is
immutable and outlives every process. A value of this kind cannot cross
*runtimes* (`std.vm`), where the pointer would name someone else's mapping.

## Builtins

Resolved by index, so the order is append-only:

| id | name | signature |
|---:|------|-----------|
| 0 | `spawn!` | thunk → process |
| 1 | `join!` | process → value |
| 2 | `send!` | process → value → unit |
| 3 | `recv!` | unit → value |
| 4 | `self!` | unit → process |
| 5 | `raise!` | value → never |
| 6 | `type_of` | value → atom |
| 7 | `to_string` | value → string |
| 8 | `len` | list \| array \| map \| string → integer |

A binding of the same name shadows the builtin.

## Notes for the VM

- **Forcing.** Every node is a suspension. Force a node to weak head normal
  form; memoize the result in the node's activation, not in the node itself —
  the `NODE` section is shared across every activation and every thread.
- **Ordering.** Ignore evaluation order except where `STRICT` says otherwise.
  Statements in an impure block carry it; bindings deliberately do not, unless
  their value is effectful.
- **Processes.** `spawn!` takes a `thunk` node's closure and runs it as a new
  process. Whether captured state is shared or copied is the VM's choice: the
  reference implementation copies, giving each process an isolated heap, which
  is what lets it collect and force thunks without atomics. A VM that shares
  instead must blackhole or lock every thunk, because the compiler guarantees
  only that captures are indices — not that forcing is single-threaded.
  The `process` object type is one of the VM's builtin object types; its
  fields are documented in [builtins.md](builtins.md), and `type_of` names it
  from `bi_type_of` in `dream/src/builtins.cpp`.
- **Validation.** A loader should bounds-check every index against its
  section's `count` before use, since a `.dream` file may be untrusted.

Optional type descriptions use ordinary value nodes and need no new opcode.
The builtin table appends `type_assert` at ID 15 (arity 3, strict mask `0b001`);
existing builtin IDs are unchanged. Images that use it require an updated VM.

### Primitive calls

Opcodes 47–74 invoke a fixed primitive without evaluating a callee. Their
argument run uses the corresponding builtin's strictness mask, in argument
order; unforced arguments remain lazy. The loader validates the builtin ID,
arity and every child index. Returning a stored value resumes evaluation in
the machine rather than forcing on the C++ stack. The JIT shares these handlers
and strictness rules. Opcodes 44–46 implement lazy cons construction, tail
selection, and the empty-list predicate directly.

The former core functions are builtin IDs 16–49, appended in this order.
All but `compare` are now spelled with a leading `_`, which marks a primitive
`std` wraps (see [builtins.md](builtins.md#runtime-primitives)). An image names
a builtin by ID and never by spelling, so the rename changed no ID and an image
written before it runs unchanged:

| ID | Name |
|---:|------|
| 16 | `_list_tail` |
| 17 | `_list_cons` |
| 18 | `_list_is_empty` |
| 19 | `_str_chars` |
| 20 | `_str_of_chars` |
| 21 | `_str_of_bytes` |
| 22 | `_str_concat` |
| 23 | `_error_new` |
| 24 | `_error_kind` |
| 25 | `_error_payload` |
| 26 | `_str_slice` |
| 27 | `_str_find` |
| 28 | `_str_byte` |
| 29 | `_str_le` |
| 30 | `_str_span` |
| 31 | `_str_upto` |
| 32 | `_char_code` |
| 33 | `_char_of_code` |
| 34 | `_to_float` |
| 35 | `_float_bytes` |
| 36 | `_float_of_bytes` |
| 37 | `_to_int` |
| 38 | `_parse_int` |
| 39 | `_parse_float` |
| 40 | `_to_existing_atom` |
| 41 | `_array_new` |
| 42 | `_array_of_list` |
| 43 | `_array_to_list` |
| 44 | `_map_has` |
| 45 | `_map_remove` |
| 46 | `_map_pairs` |
| 47 | `_data_count` |
| 48 | `_data_at` |
| 49 | `compare` |

Later appends: `_sort_keyed` (50), `_tensor_matmul` (51) and `_str_interp`
(52), the last the builtin behind opcode 74, which an interpolated string
`$"..{x}.."` lowers to directly: one kid, the list of its pieces. Then `_mailbox_peek!`,
`_mailbox_take!`, `_await_message!`, `_deadline_in!` (53-56), `_match_fail`
(57) and `_error_trace!` (58). Then `_tensor_of_bytes` (59), and the record
primitives `_record_is` (60), `_record_id` (61), `_record_fields` (62) and
`_record_of` (63). Then `_pow` (64), `num.pow`, and the string walks `_str_char_count` (65),
`_str_ascii_case` (66), `_str_span_back` (67) and `_str_rfind` (68). Then
`_settled` (69), `lazy.settled`, and `_forced` (70), `lazy.forced`.

Images importing the removed `std.core` or `std.native` modules must be recompiled.
