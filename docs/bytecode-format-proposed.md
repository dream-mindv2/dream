# Proposed bytecode-format changes

> **Landed.** Everything below is implemented and has been folded into
> [bytecode-format.md](bytecode-format.md), which is the spec to read and the
> one to keep current. `MODS` is documented there in the section table and
> beside the global record; `LDAT`/`PAYL` have a section of their own, and the
> primitives that reach the payload are in [builtins.md](builtins.md).
> This file is kept only as the record of what was argued for and why; it can
> go whenever that is no longer worth having.
>
> Two things the proposal did not say, settled during the work: a bigstr and a
> string of the same bytes compare equal and hash alike (otherwise
> `str.slice 0 4 data == "%PDF"` is quietly false), and `data_count` was
> added beside `data_at`, because without it nothing can ask how many data an
> image carries. `str.byte`, which the proposal lists as supported, did not
> exist in `std.str` and was added.

This file is the *proposal* for additions to `bytecode-format.md`, kept
separate until the format work lands.

## Why

`MODS` exists in the format already — it is emitted (`dreams/src/emit.dr:213`),
validated (`dream/src/image.cpp:284`), and read by the VM
(`dream/src/image.cpp:296-304`) — but it is missing from the section table in
`docs/bytecode-format.md`. Merging images (see `docs/dynamic-linking.md`) makes
a module's global range the identity that survives the merge, so the record
deserves documenting. Two additions:

1. A `MODS` row in the section table.
2. A module-record description beside "Global record".

## Published spec, current

From `docs/bytecode-format.md`:

| Kind   | Record                                              | Purpose |
|--------|-----------------------------------------------------|---------|
| `IMPT` | `u32 path, u32 alias`                               | imports; both are `KSTR` indices |
| `SPAN` | `u32 start, u32 end`                                 | per-node source spans, parallel to `NODE` |

## Proposed

| Kind   | Record                                              | Purpose |
|--------|-----------------------------------------------------|---------|
| `IMPT` | `u32 path, u32 alias`                               | imports; both are `KSTR` indices |
| `MODS` | 24 bytes (below)                                    | module records: which globals belong to which module |
| `SPAN` | `u32 start, u32 end`                                 | per-node source spans, parallel to `NODE` |

### Module record (24 bytes)

| Offset | Size | Field           | Notes |
|-------:|-----:|-----------------|-------|
| 0      | 4    | `name`          | `KSTR` index |
| 4      | 4    | `source`        | `KSTR` index; the original source path |
| 8      | 4    | `globals_start` | first global index belonging to this module |
| 12     | 4    | `globals_count` | |
| 16     | 4    | `flags`         | `0x01` host-provided, `0x02` declares virtuals |
| 20     | 4    | `derives`       | `MODS` index of the derived module, or `NO_NODE` |

A module's globals form a contiguous range `globals_start .. globals_start +
globals_count`; the range is what lets a reader say which module a global
belongs to without the `GLBL` records carrying a module id of their own.

The record matches `ModuleRec` in `dream/src/image.hpp` and `module_bytes` in
`dreams/src/emit.dr`. The `MODS` section is validated like any other record array at
load; unknown sections are skipped, so a reader predating this documentation is
unaffected.

## Large data: a payload region beyond the 4 GiB image

### Why

The container addresses everything with `u32`: a section table entry is
`u32 offset, u32 length`, a `StrRec` is `u32 offset, u32 length`, and the VM's
`StrObj` carries a `uint32_t len`. Together they cap a `.dream` file, and any
string in it, at ~4 GiB. That is plenty for a program but nothing for the datum
a program is *about* -- a corpus, a model, an asset. Two constraints shape the
addition:

- The payload must not move anything the existing sections need. Everything up
  to and including `LDAT` stays inside its `u32` addresses; only the payload
  itself is allowed to cross the line, because it is the last thing in the file
  and is addressed by 64-bit numbers nowhere else has to know.
- Loading must not copy it. The whole point of growth is a big datum, and the
  VM already copies a string once on materialization and again at every
  collection and `spawn!`. A big string is a *view* into the mapped image, not
  a `StrObj`.

### Published spec, current

The section table is a `16`-byte entry of `u32`s, strings are `KSTR`
offset-and-length records into the `SBLB` blob, and the VM materializes
`ConstStr` by copying the bytes into its heap.

### Proposed

Two new section kinds, both additive: a reader that knows neither skips them as
unknown kinds, exactly as it does any section today.

| Kind   | Record                                        | Purpose |
|--------|-----------------------------------------------|---------|
| `LDAT` | `u64 offset, u64 length`                      | large-data table, into `PAYL` |
| `PAYL` | `u64 byte_length` header, then the bytes      | the payload, last in the file |

- `PAYL` is the payload: an 8-byte little-endian `byte_length`, then that many
  raw bytes. Its section-table entry is ordinary: `offset` names the header
  (a start below 2^32, since everything precedes it), `length` is `8` -- the
  honest size of the section as laid out -- and `count` is `0`. The real size
  is the `u64` at the section start, and it may exceed 4 GiB. The payload ends
  the file; its bytes are not padded.
- `LDAT` is the table of payload descriptors, `u64 offset, u64 length`, offsets
  counted from the start of the payload bytes (`PAYL.offset + 8`). It exists
  so a `u32` index can name a payload without a 64-bit offset in every use.
  `LDAT` must not appear without `PAYL`, nor `PAYL` without `LDAT`.
- The payload is a *host-level* feature, not an instruction: nothing in `NODE`,
  `KIDS`, `FUNC`, or `KSTR` points at it. The primitive `data_at`
  (see `builtins.md`) materializes payload `i` as a zero-copy view. This is
  what keeps the change additive -- an image that never uses its payload runs
  on any reader, and the format does not need a new opcode.

Validation: `fits(16)` for `LDAT`, and each record must satisfy
`offset + length` (u64 arithmetic) against the payload's `byte_length`; the
payload must satisfy `PAYL.offset + 8 + byte_length` against the file size.

### The VM side

The value is a new heap object:

```
struct BigStrObj : Obj {
    uint64_t len;          // bytes
    uint64_t hash;         // filled like a StrObj's, so a big string costs one hash walk
    const char* data;      // into the image mapping; not heap memory
};
```

- Creating one is O(1) and copies nothing: `data_at` reads `LDAT[i]` and points
  at `payload + offset`.
- The collector treats it as an "atom" object -- no internal references -- so
  promotion is a byte copy and a `spawn!` heap copy shares the view. That is
  safe because the bytes live in the program image and the image outlives every
  process. A big string cannot cross *runtimes* (`std.vm`), since the pointer
  would point into someone else's image.
- `type_of` names it `:bigstr`, so no `:string` code path is handed a
  multi-gigabyte value it will choke on.
- Supported -- none of these copy:
  - `len`, `str.byte_length` -- the 64-bit length, as an integer.
  - `str.byte` -- one byte, pointer math.
  - `str.slice` -- another `BigStr` view, clamped like the string case.
  - `==` and the ordering operators, and use as a map key.
  - `io.write!` -- written straight from the image, no intermediate buffer.
- Unsupported, and a clean error: `+`, `str.concat_all`, `to_string`, and
  every `str` op that would have to produce a `StrObj` (concat, find, join_on).
  The message points at slicing and `io.write!`.

### The compiler side

The payload is declared on the command line: `dreams --payload PATH`, one `LDAT`
entry per file in order. A Dream string is itself capped at `uint32_t` bytes,
so the compiler never holds a payload: it asks each file's `io.size!`, lays out
`LDAT`/`PAYL` from the sizes, writes the ordinary prefix as today, and then
streams each file into the output in bounded `io.read!`/`io.write!` chunks. A
build with no `--payload` emits the exact bytes it emits today.