# Dream Built-in Functions Reference

This document covers every built-in function available in Dream, organized by module. There are three layers:

1. **Language builtins** — available everywhere, no import needed
2. **Native modules** — C++ modules imported with `import std.<name>`
3. **Dream stdlib** — modules written in Dream, in `mind/std/`

---

## Language Builtins

These are resolved by the compiler without any import. They can be shadowed by a local binding.

| Name | Signature | Description |
|------|-----------|-------------|
| `spawn!` | `thunk → process` | Starts a new process to evaluate `thunk`. The argument is **not** forced before being handed off — the new process does the work. |
| `join!` | `process → value` | Waits for a process to finish and returns its result. If the process raised, the error propagates here. Parks the calling process, not the worker thread. |
| `send!` | `process → value → unit` | Deep-copies `value` into the mailbox of `process`. The message is fully evaluated before leaving — thunks carrying this process's heap cannot be shared. |
| `recv!` | `unit → value` | Takes the next message from this process's mailbox. Parks until one arrives. |
| `self!` | `unit → process` | Returns the current process's handle. |
| `raise!` | `value → never` | Raises `value` as an error. If `value` is already an error box it is re-raised as-is; otherwise it is wrapped in one. Never returns. |
| `type_assert` | `bool → description → value → value` | Contract primitive used by `std.types.check`. Forces only the decision; returns the original value for `true`, otherwise raises `:type_error` with the description as payload. |
| `type_of` | `value → atom` | Returns an atom naming the type of its argument: `:integer`, `:float`, `:char`, `:bool`, `:unit`, `:string`, `:atom`, `:list`, `:array`, `:map`, `:module`, `:error`, `:process`, `:pure_fn`, `:impure_fn`, `:tensor`, or `:bigstr` (a view into the image's payload — see [Large data](#large-data) below; deliberately *not* `:string`, so that no code path written for a string is handed one). |
| `to_string` | `value → string` | Renders any value as a human-readable string. Lists print as `[a, b, c]`, arrays as `#[a, b, c]`, maps as `%{:k => v}`, strings are quoted, chars as `'c'`. Raises `:type_error` on a bigstr, which by definition may not fit in a string; one nested inside a larger value renders as `<big string, N bytes>` rather than losing the rest of the structure. |
| `len` | `list\|array\|map\|string\|bigstr\|tensor → integer` | Returns the number of elements (list), slots (array), entries (map), bytes (string or bigstr), or the length of a tensor's first axis. Raises `:type_error` for anything else. For lists, walks the entire spine. |
| `strict!` | `value → value` | Forces `value` all the way to normal form (deeply, not just to weak head normal form), then returns it unchanged. Use this when laziness would defer an effect — `list.map (fn x -> spawn! ..) xs` builds thunks; `strict! (list.map ...)` runs the spawns immediately. |
| `defer! expression` | `expression → unit` | A direct block statement: evaluate the expression when that block exits, including on errors and kills. Defers run in reverse order. See [Blocks](https://github.com/dream-mindv2/dreams/blob/main/docs/language-spec.md#blocks). |

### Pattern-match internals

These are emitted by the compiler for `match` expressions. They are primitives (see below) with no `std` wrapper: nothing but the lowering of a pattern has any business calling them.

| Name | Description |
|------|-------------|
| `_match_is_cons v` | Returns `true` if `v` is a cons cell (a non-empty list). |
| `_match_head v` | Returns the head of a cons cell, unforced. |
| `_match_tail v` | Returns the tail of a cons cell, unforced. |
| `_match_at v i` | Returns element `i` of array `v`, unforced. |
| `_match_key map key` | Returns `[value]` if `key` is in `map`, `[]` if absent; the value unforced. |

Each takes its arguments forced by the machine, through its strictness mask, and hands back what it found without forcing it: the machine forces the piece as the continuation of the call. That is what lets a chain of reads through patterns — a list rebuilt many times from the rest of the one before — go as deep as the heap allows rather than as deep as the C++ stack. The list-tail opcode and container reads use the same machine continuation.

---

## Runtime primitives

Operations the machine has to provide because Dream cannot write them. Each is
spelled with a leading `_`, and each is **`std`'s to call**: `std` wraps it in a
function under an ordinary name -- the second column below -- and a program is
meant to call that. A primitive named anywhere outside `std` still compiles,
with a warning saying which call to write instead; the underscore is there so
that one is never reached for by accident, and so `std` is free to change them.

Calling the wrapper costs nothing. A function whose body only passes its
parameters on is a *wrapper*, and the compiler emits the call it stands for in
its place, so `str.byte i s` compiles to exactly what `_str_byte s i` does --
the same opcode, the same reductions, the same code from the JIT. Passing one as
a value (`list.map char.of_code cs`) passes the primitive itself when the
wrapper keeps its parameters in order. Six of the wrappers take theirs in a
different order from the primitive, subject last, as the rest of `std` does;
the column says which.

Saturated calls compile directly to opcodes; partial applications and function
values retain ordinary builtin semantics. `_str_of_chars`, `_str_of_bytes`,
`_str_concat`, and `_array_of_list` remain builtin calls because they traverse
lazy lists. None of the language builtins above is a primitive, and
neither is anything a native module (`std.vm`, `std.math`, `std.io`, ..)
provides: those are called by their own names.

Container access and updates use `xs.[0]`, `m.[key else default]`, and
`m.[key => value]`. An empty map is `%{}`; byte length is `len s`.
The former `std.core` and `std.native` modules have been removed.
Scalar type descriptions such as `Integer` and `Number` live in `std.types`.

### Lists

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_list_tail` | `list.tail xs` | `list → list` | Everything after the first element. Raises `:type_error` on an empty list. |
| `_list_cons` | `list.cons x xs` | `value → list → list` | A new cons cell with the given head and tail. Both sides stay lazy. |
| `_list_is_empty` | `list.is_empty xs` | `list → bool` | `true` if the list is `[]`. |

### Strings

Strings are byte-indexed internally (UTF-8 storage). Offsets in the functions below are **byte** offsets, not character offsets.

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_str_chars` | `str.chars s` | `string → list of char` | Decodes the string to a list of Unicode codepoints (characters). Bytes that do not spell a Unicode scalar value — a stray or truncated sequence, an overlong form, a surrogate, anything past U+10FFFF — each become U+FFFD, one per byte, so every char it yields is one `_char_of_code` would accept. Raises `:type_error` on a bigstr. |
| `_str_of_chars` | `str.of_chars cs` | `list of char → string` | Encodes a list of characters into a UTF-8 string. |
| `_str_of_bytes` | `str.of_bytes bs` | `list of integer → string` | Builds a string from raw byte values, each `0`–`255`. The inverse of `_str_byte`, and the way to produce **binary** output: `_str_of_chars` UTF-8-encodes its input, so byte `0x80` would become two bytes. Raises `:type_error` for a non-integer or a value outside `0`–`255`. A `0` byte is an ordinary byte and does not end the string. |
| `_str_slice` | `str.slice from count s` | `string\|bigstr → start:integer → len:integer → string\|bigstr` | Returns `len` bytes starting at byte offset `start`. Clamped silently — running past the end is how string-walking loops finish. A slice of a bigstr is another bigstr view, however small: no copy, at any size. |
| `_str_find` | `str.find_from from needle s` | `haystack:string → needle:string → from:integer → integer` | Returns the byte offset of the first occurrence of `needle` at or after `from`, or `-1` if not found. Raises `:type_error` on a bigstr. |
| `_str_byte` | `str.byte i s` | `string\|bigstr → index:integer → integer` | The raw byte value (0–255) at byte `index`, or `-1` if out of range. |
| `_str_span` | `str.span set from s` | `string\|bigstr → from:integer → set:string → integer` | The byte offset of the first byte at or after `from` that is **not** in `set`, or the string's byte length when there is none. `set` is read as the set of its bytes, so `_str_span s i " \t"` skips indentation and `_str_span s i digits` skips a number. A whole run of a byte class in one operation — the same walk written in Dream is a call and a comparison chain per byte, which is what a lexer spends its time on. |
| `_str_upto` | `str.upto set from s` | `string\|bigstr → from:integer → set:string → integer` | The other way round: the offset of the first byte at or after `from` that **is** in `set`, or the byte length when there is none — so `_str_upto s i "\n"` is the end of the line whether or not the text ends with one. |
| `_str_span_back` | `str.span_back set until s` | `string\|bigstr → until:integer → set:string → integer` | `_str_span` run backwards: the start of the run of bytes from `set` that ends at `until` (clamped to the length). `str.trim_end` is this and a slice. |
| `_str_rfind` | `str.find_last_from from needle s` | `haystack:string → needle:string → from:integer → integer` | The greatest byte offset at or before `from` where `needle` starts, or `-1`. Raises `:type_error` on a bigstr. |
| `_str_char_count` | `str.length s` | `string → integer` | The number of characters, decoded as `_str_chars` decodes them, without building the list. |
| `_str_ascii_case` | `str.upper s` · `str.lower s` | `string → bool → string` | The string with ASCII letters upper-cased (`true`) or lower-cased (`false`); every other byte, all of a multi-byte character included, unchanged. |
| `_str_concat` | `str.concat_all parts` | `list of string → string` | Joins the parts, copying each exactly once. Raises `:type_error` on a bigstr. |

### Chars

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_char_code` | `char.code c` | `char → integer` | The Unicode scalar value (codepoint) of a character. |
| `_char_of_code` | `char.of_code n` | `integer → char` | The character for a Unicode scalar value. Raises `:type_error` if the integer is not a valid Unicode scalar (must be 0–0x10FFFF, excluding surrogates). |

### Numbers

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_to_float` | `num.to_float n` | `integer\|float → float` | Converts a number to `float`. Returns a `float` unchanged. |
| `_to_int` | `num.to_int x` | `integer\|float → integer` | Converts a number to `integer`. Truncates toward zero (like C cast). For floor-division behavior use `math.floor` first. |
| `_parse_int` | `num.parse_int s` | `string → integer\|unit` | Parses a base-10 integer from a string. Returns `unit` on failure. The entire string must be a valid integer (trailing non-numeric characters cause failure). |
| `_parse_float` | `num.parse_float s` | `string → float\|unit` | Parses a floating-point number from a string. Returns `unit` on failure. |
| `_pow` | `num.pow base exponent` | `number → number → number` | Two integers, the exponent zero or more, answer the exact integer, a bignum when it grows past a fixnum (by repeated squaring). A float on either side answers the float `math.pow` does. Raises `:type_error` for an integer base and a negative integer exponent -- the answer's kind follows the base's -- and for an exponent past 64 bits unless the base is 0, 1 or -1. |

### Atoms

There is no `to_atom`, and that is the design rather than a gap. An atom is an
index into a table the runtime never shrinks — every atom value in every
process is one of those indexes, so none can be reused — and a program that
interned text it did not write would grow that table until it died, with no
collector able to reach it. So the only direction offered is the one that
cannot leak.

The other direction needs no primitive: `atom.name :ok` is `"ok"`, the atom as
`to_string` writes it less its leading `:`. Use it rather than slicing a
rendering by hand.

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_to_existing_atom` | `atom.existing name` | `string\|bigstr → atom\|unit` | The atom of this name if there already is one, `unit` if there is not. Never creates one. A name is an atom when the program writes it as one anywhere — an image's atoms are interned when it loads — so this is how text from outside is matched against names the program does know. Accepts a bigstr, since it reads the bytes and builds nothing. Raises `:type_error` on anything that is not a string. The answer only moves one way: the table never forgets, so an atom stays one. |

### Arrays

Arrays are fixed-length, eagerly allocated sequences. Indexing is O(1). All update operations return a new array (values are immutable).

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_array_new` | `array.new n fill` | `length:integer → fill:value → array` | Creates a new array of `length` slots, each initialized to `fill`. |
| `_array_of_list` | `array.of_list xs` | `list → array` | Converts a list to an array. Elements remain lazy. |
| `_array_to_list` | `array.to_list a` | `array → list` | Converts an array to a list. Elements remain lazy. |

### Maps

Maps are persistent hash maps — a hash array mapped trie, branching 32 ways on five bits of the key's hash at a time. Keys are compared by value for flat types (integers, floats, strings, atoms, chars, bools, pids) and by identity for everything else. An identity key keeps its hash in its header, so it is found again after the collector has moved it and on the far side of a `send!` that carried it with the map. All updates return a new map.

Persistent means *shared*, not copied: `m.[key => value]` rebuilds only the path from the root to the entry it changes — about `log32(n)` nodes — and the map it was given keeps every other node and stays valid. So the ordinary functional way to build a map, folding `m.[key => value]` over a sequence, costs `O(n log n)` in total rather than the `O(n²)` a copy-on-write table would. `len` is constant time: every node knows how many entries hang below it.

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_map_has` | `map.has k m` | `map → key → bool` | Returns `true` if `key` is in the map. |
| `_map_remove` | `map.remove k m` | `map → key → map` | Returns a new map with `key` removed. |
| `_map_pairs` | `map.pairs m` | `map → list` | Returns a list of `[key, value]` pairs in unspecified order. |

### Ordering

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `compare` | `compare a b` | `a → b → integer` | Total order comparison. Returns `-1`, `0`, or `1`. Ranks, in order: integers and floats (numerically), chars, bools, atoms, strings, unit, lists, arrays, tensors, records. A bigstr ranks with the strings and compares by its bytes. Lists and arrays compare element by element, forcing as they go, and a prefix sorts before what it prefixes. Tensors compare by shape, then element by element. Records compare by the name of their declaration, then field by field. Values of different types order by their rank. Maps, functions and pids compare equal to anything of their own kind. |
| `_tensor_matmul` | `a @ b` | `tensor\|list\|array → tensor\|list\|array → tensor\|float` | What `a @ b` is written as. See [`std.tensor`](#stdtensor). |
| `_mailbox_peek!` · `_mailbox_take!` · `_await_message!` · `_deadline_in!` | `receive! { .. }` | | What `receive!` is written with: the `std.vm` natives of the same names (`_deadline_in! ms` is `vm.now_ns!` that many milliseconds on), as builtins so the syntax needs no import. |
| `_match_fail` | a `match` no arm fits | `value → path → line → col → never` | Raises `:match_error` with `[value, path, line, col]`: one call per failure site, so a site costs one node and its kids. |
| `_error_trace!` | `error.trace! e` | `error → list` | Where the error was first raised, innermost first: `[:in, name, path, line, col]` for a function running then, `[:made, ..]` for the function that made a value being forced. `[]` for one never raised. |
| `_str_interp` | `$"..{x}.."` | `list → string` | Every element rendered as `to_string` renders it, joined: what an interpolated string is written as, lowered straight to the `str_interp` opcode. |
| `_record_is` | a constructor pattern | `atom → value → bool` | Whether the value is a record whose id is the atom: `Shape.circle(r)` is this, then `_match_at` per field. Never raises. |
| `_record_id` | `types.record_id v` | `value → atom\|unit` | The atom naming what made a `group`, `struct` or union value -- `app.Point`, `app.Shape.circle` -- or `()` for anything no constructor made. |
| `_record_fields` | `types.record_fields v` | `record → array` | Its fields in declared order, unforced. |
| `_record_of` | a constructor's body | `atom → bool → array → record` | A record with that id, a variant's when the bool is true, its fields the array's items, unforced. What `syntax.record` writes as a constructor, which lowering turns into `record_make`; `types.make_record` for a decoder. |
| `_settled` | `lazy.settled v` | `value → bool` | `true`, once the value has been forced all the way down, as `strict!` forces it but pure. For a pure program that keeps what it makes, where a lazy value held in a persistent table would hold every version of the table it was made from. Vouches for the collector, so a deep force may collect, except under a compiled caller, which pins the heap as it does for every native. Raises what forcing raises. |
| `_forced` | `lazy.forced v` | `value → bool` | `true`, once the value has been forced to its outermost constructor and no further. For a pure loop that carries a table, so each step's put is made rather than suspended. Costs what `type_of` costs. |
| `_sort_keyed` | `list.sort_on key xs` | `keys:list -> array -> list` | The array's elements, as a list, in the order `compare` puts `keys` in, equal keys keeping their order. The keys are forced whole first; the elements are carried and never forced. `std.list.sort` and `sort_on` are this. |

### Large data

The payload region of the image: bytes put there at compile time with
`dreams --payload FILE`, one entry per file in the order given. See
[bytecode-format.md](bytecode-format.md) for the `LDAT` and `PAYL` sections
these read.

| Primitive | Write | Signature | Description |
|-----------|-------|-----------|-------------|
| `_data_count` | `payload.count ()` | `unit → integer` | How many large data this image carries. `0` for an image built without `--payload`, which is every ordinary image. |
| `_data_at` | `payload.at i` | `integer → bigstr` | Datum `i`, as a **bigstr**: a length and a pointer into the mapped image. Constant time and copies nothing, whatever its size. Raises `:type_error` for an index outside `0 .. payload.count () - 1`. |

Both are pure. The payload is fixed when the image is written and nothing can
alter it, so asking for datum `i` is a function of `i` in the way `str.byte` is
a function of its index.

**What a bigstr is for.** A `.dream` file addresses itself with 32-bit offsets,
so the file and every string in it stop at 4 GiB. A bigstr is the exception: it
is not a string but a *view*, and it is never copied — not when it is made, not
when the collector promotes it, and not when `spawn!` hands it to another
process, all of which share the pointer because the bytes belong to the
runtime's image and outlive every process in it.

Operations that need only to read it all work, and none of them copies:

| Works | |
|-------|--|
| `len`, `str.byte_length` | the 64-bit length, as an integer |
| `str.byte` | one byte; pointer arithmetic |
| `str.slice` | another view, clamped like the string case |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | by bytes |
| a map key | hashed once, then remembered on the value |
| `io.write!` | written straight from the mapping, with no intermediate buffer |

Operations that would have to *build* a string out of it raise `:type_error`
with a message naming what to reach for instead: `+`, `to_string`,
`str.concat_all`, `str.find` and `str.chars`. Each would copy the bytes into the
heap and each would be capped at the 4 GiB a string can count, which between
them is the reason the payload is not a string in the first place.

**A bigstr and a string of the same bytes are equal, and hash alike.** That is
what makes the first thing anyone writes — `str.slice 0 4 data == "%PDF"` — mean
what it looks like. `type_of` still tells them apart, so a branch written for
`:string` is never handed one.

```dream
import std.io;
import std.payload;

let main! = {
    let data = payload.at 0;
    if str.slice 0 4 data == "%PDF" {
        io.write! (io.stdout! ()) data       // the whole of it, never in memory
    } else { console.error! "not a PDF" }
};
```

---

## `std.console`

Implemented in Dream over `std.io`. Existing `print!`, `write!`, `line!` and
`error!` calls take one value, render it with `to_string`, and return unit.
Writes complete before returning; I/O failures propagate.

```dream
import std.console;
console.printf! "{} scored {}" ["Ada", 42]
let name = console.prompt! "Name? ";
console.warn! "Using the default configuration"
```

| Function | Behavior |
|---|---|
| `print! value`, `line! value` | stdout, with a newline |
| `write! value` | stdout, without a newline |
| `error! value`, `error_write! value` | stderr, with/without a newline |
| `print_to! handle value`, `write_to! handle value` | caller-supplied output |
| `text values` | concatenate rendered values |
| `format template values` | `Result.ok text` or `Result.error message`; `{}`, `{{`, `}}` |
| `format! template values` | formatted text, or `:bad_argument` |
| `printf!`, `writef!`, `errorf!` | formatted output |
| `read_line! ()`, `prompt! question` | a line, or `()` at EOF |
| `confirm! question default` | yes/no with retry; `()` at EOF |
| `prompt_int! question`, `ask! question parser` | validated input with retry |
| `styled! styles value` | terminal-aware styled stdout line |
| `log! level value`, `info! value`, `warn! value` | labeled stderr logging |
| `logf! level template values` | formatted logging |

See [Console](https://github.com/dream-mindv2/std/blob/main/docs/console.md) for stream injection, line limits, colors and logger
configuration. Prompts use stderr by default so stdout can carry program results.

---

## `std.math`

```dream
import std.math;
```

| Name | Signature | Description |
|------|-----------|-------------|
| `sqrt` | `number → float` | Square root. |
| `abs` | `integer\|float → integer\|float` | Absolute value. Returns the same type as the input. |
| `floor` | `integer\|float → integer` | Rounds down to the nearest integer. |
| `exp` · `log` · `sin` · `cos` · `tan` · `tanh` · `atan` | `number → float` | The functions of a real variable. |
| `pow x y` | `number → number → float` | `x` to the power `y`. |

---

## `std.tensor`

```dream
import std.tensor;

let a = tensor.of_list [[1, 2], [3, 4]];
let b = tensor.random 1 [2, 2];
a @ b                       // matrix product
a * b + 1.0                 // elementwise, and with a number
tensor.relu (a @ b - 0.5)   // elementwise functions
let g = tensor.gpu a;       // the same operations, now on the GPU
tensor.host (g @ g)         // and back
```

A tensor is packed numbers with a shape: up to six axes, row-major. On the
host the numbers are doubles; on the GPU, float32 (`tensor.gpu`) or float64
(`tensor.gpu64`). It is a value: nothing changes one, and every operation
answers a new tensor. `type_of` answers `:tensor`, and `:tensor` is a type in
signatures. [docs/notes/tensors.md](notes/tensors.md) is the design.

**Operators.** `+ - * / %` are elementwise between two tensors, or between a
tensor and a number on either side. The shapes must be equal, or one must be
the trailing part of the other, which repeats it (a vector added to every row
of a matrix). Unary `-` negates. `a @ b` is the matrix product: matrix by
matrix, matrix by vector, vector by matrix, or vector by vector, which is the
dot product and answers a float -- or a batch of matrices `[b, m, k]` by a batch
`[b, k, n]` (each pair multiplied; one launch on the GPU) or by one matrix `[k,
n]`. `@` binds like `*`. `t.[i]` is an element of
a vector, or a copy of row `i` of anything larger; `len t` is the length of the
first axis. `==` compares shapes and numbers.

**Lists in, tensors out.** Every function here and `@` take a tensor *or*
nested lists and arrays of numbers, converted on the way in, so
`[1, 2, 3] @ [4, 5, 6]` is `32`. The operators `+ - * / %` do not, because `+`
on lists already joins them.

**Fusion.** On a large tensor (a few thousand numbers or more, and always on
the GPU), the elementwise operators and functions do not compute at once. They
record the work, and whatever reads the numbers runs the whole chain in one
pass: `tensor.relu (a * 2.0 + b) * 0.5` is one pass over memory, or one GPU
kernel, not four. A product defers too, so the chain after it runs inside it:
`tensor.relu (w @ x + b)` applies `+ b` and `relu` to each block of the
product as it is finished, or in the product kernel itself on the GPU. A
transpose defers too, and is read where it lies: by a chain, or by a
product's strides (`a @ tensor.transpose b` copies nothing). A chain feeding
a product on the host is computed as the product packs it. A reduction of
such a chain never stores it, and neither do `sum_axis` and `dot`. A reshape
copies nothing. Nothing about
this is visible except the speed: a deferred tensor has its type and shape,
and shape errors are raised where they always were. `strict!` computes one,
which is what timing or `vm.share!` wants.

**Where it runs.** An operation runs where its operands are. The CPU kernels
are blocked and vectorized (AVX2 and FMA where the CPU has them, chosen at
start-up), and a large product is split across threads (capped by
`DREAM_TENSOR_THREADS`). The GPU is reached through OpenCL, loaded at run time
(`DREAM_OPENCL_LIB` names the library), and its operations are queued and
return at once; only reading a result back waits. A GPU tensor that dies has
its device memory freed by the collector. Combining a host tensor with a GPU
tensor raises `:device_error`.

| Name | Signature | Description |
|------|-----------|-------------|
| `of_list xs` | `nested lists/arrays → tensor` | A host tensor. The shape is read off the nesting, and every row must match it. |
| `to_list t` | `tensor → nested lists` | The numbers as nested lists of floats. |
| `zeros shape` · `ones shape` | `[integer] → tensor` | Filled with 0 or 1. |
| `fill shape x` | `[integer] → number → tensor` | Filled with `x`. |
| `identity n` | `integer → tensor` | The `n x n` identity matrix. |
| `range n` | `integer → tensor` | `[0, 1, .., n - 1]`. |
| `random seed shape` | `integer → [integer] → tensor` | Uniform in `[0, 1)`, the same for the same seed on every machine. |
| `shape t` · `rank t` · `size t` | `tensor → ..` | The axes' lengths, how many, and how many numbers in all. |
| `device t` · `dtype t` | `tensor → atom` | `:host` or `:gpu`; `:f64` or `:f32`. |
| `reshape shape t` | `[integer] → tensor → tensor` | The same numbers in a new shape of the same size. Free on the GPU. |
| `transpose t` | `tensor → tensor` | A matrix transposed; a vector unchanged. |
| `at t indices` | `tensor → [integer] → float` | One element, one index per axis. |
| `matmul a b` · `dot a b` · `outer a b` | `tensor → tensor → ..` | `a @ b`; `@` of two vectors only; every `a[i] * b[j]`. |
| `sum` · `mean` · `minimum` · `maximum` · `norm` | `tensor → float` | Over every number. `norm` is the Euclidean length. |
| `sum_axis axis t` | `integer → tensor → tensor` | Sums along one axis, which drops out of the shape. A vector's is a number. |
| `max_axis` · `min_axis` | `integer → tensor → tensor` | The largest or smallest along one axis, which drops out. |
| `argmax_axis` · `argmin_axis` | `integer → tensor → tensor` | Where along the axis the largest or smallest first is, as numbers (so they stay on the GPU); a vector's is an integer. |
| `sqrt` · `exp` · `log` · `abs` · `tanh` · `sin` · `cos` · `relu` · `sigmoid` · `floor` · `ceil` · `round` · `sign` · `erf` | `tensor → tensor` | Elementwise. `round` takes a half away from zero. |
| `max a b` · `min a b` · `pow a b` | `tensor\|number → tensor\|number → tensor` | Elementwise, broadcast and fused as the operators are. (One argument's `maximum` and `minimum` are the reductions.) |
| `lt` · `le` · `gt` · `ge` · `eq` · `ne` | `tensor\|number → tensor\|number → tensor` | Elementwise comparisons answering 1 where they hold and 0 where not: a mask, which multiplies into a chain. |
| `fill_like t x` | `tensor → number → tensor` | `t`'s shape, where `t` is, every number `x`. Free on the GPU: nothing is uploaded. |
| `repeat n t` | `integer → tensor → tensor` | `t` with a new last axis of length `n`, each number repeated along it: a value per row stretched across the row. Read in place by a chain, so `x - repeat k (max_axis 1 x)` is one pass. |
| `slice start count t` | `integer → integer → tensor → tensor` | Rows `start` to `start + count` of the first axis. |
| `take rows t` | `[integer]\|tensor → tensor → tensor` | The rows named, in that order, any row any number of times. One gather on the GPU. |
| `concat ts` | `[tensor] → tensor` | Joined along the first axis; the rest of the shapes must agree. |
| `one_hot n classes` | `integer → [integer]\|tensor → tensor` | A row per class, `n` wide, 1 at the class. |
| `permute axes t` | `[integer] → tensor → tensor` | The axes reordered: axis `k` of the answer is axis `axes[k]` of `t`. |
| `im2col kh kw stride pad x` | `integer → integer → integer → integer → tensor → tensor` | Every `kh x kw` window of images `[n, h, w, c]` (channels last), moved `stride` at a time over them padded with `pad` zeros, as a row: `[n * oh * ow, kh * kw * c]`. A convolution is one `@` of these rows by the kernel. |
| `col2im shape kh kw stride pad cols` | `[integer] → .. → tensor → tensor` | The reverse: images of `shape` whose each number is the sum of the places `im2col` copied it to. |
| `normal seed shape` | `integer → [integer] → tensor` | Standard normal numbers, the same for the same seed everywhere. |
| `to_bytes t` · `of_bytes shape bytes` | `tensor → string` · `[integer] → string → tensor` | The numbers as little-endian doubles, and back, exactly. |
| `gpu t` · `gpu64 t` | `tensor → tensor` | Onto the GPU as float32 or float64. Raises `:no_gpu` naming what is missing. |
| `host t` | `tensor → tensor` | Back onto the host. |
| `gpu_available ()` · `gpu_name ()` | `unit → ..` | Whether a GPU can be used, and its name (or `()`). |
| `cpu_kernels ()` | `unit → string` | `"avx2"` or `"baseline"`: which CPU kernels this machine runs. |

---

## `std.io`

File and stream I/O. Handles are opaque integers with generation tracking. A stale or closed handle raises `:io_closed` rather than operating on an unrelated file.

On Linux, stream reads and writes that would block park the **calling process**, not the worker thread, so other Dream processes continue running.

```dream
import std.io;
```

### Opening and closing

| Name | Signature | Description |
|------|-----------|-------------|
| `open!` | `path:string → mode:atom → handle` | Opens a file. `mode` must be one of `:read`, `:write`, `:append`, or `:update`. `:write` truncates; `:update` is read-write without truncating. |
| `close!` | `handle → unit` | Closes a handle. Safe to call from a different process than the one that opened it. |
| `is_open!` | `handle → bool` | Returns whether a handle is still open. |

### Reading and writing

| Name | Signature | Description |
|------|-----------|-------------|
| `read!` | `handle → bytes:integer → string` | Reads at most `bytes` bytes. Returns an empty string `""` at end of file. A caller loops until it sees `""`. |
| `write!` | `handle → data:string → integer` | Writes as much of `data` as the descriptor will accept and returns how many bytes were written. A short write is not an error — use a loop (or `write_all!` from the Dream stdlib) to ensure all bytes go out. |
| `flush!` | `handle → unit` | Flushes kernel buffers for regular files (fsync). A no-op for sockets and pipes. |

### Positioning

| Name | Signature | Description |
|------|-----------|-------------|
| `seek!` | `handle → offset:integer → integer` | Seeks to `offset` bytes from the start of the file. Returns the new position. Raises on streams that are not seekable. |
| `size!` | `handle → integer` | Returns the file size in bytes via `fstat`. |

### Querying

| Name | Signature | Description |
|------|-----------|-------------|
| `kind!` | `handle → atom` | Returns `:file`, `:stream`, or `:listener`. |
| `stdin!` | `unit → handle` | A handle for fd 0. Not owned — do not close it. |
| `stdout!` | `unit → handle` | A handle for fd 1. Not owned — do not close it. |
| `stderr!` | `unit → handle` | A handle for fd 2. Not owned — do not close it. |
| `async` | `unit → bool` | Whether epoll-based async I/O is available on this platform. |

### Filesystem operations

| Name | Signature | Description |
|------|-----------|-------------|
| `exists!` | `path:string → bool` | Returns whether a path exists. |
| `is_dir!` | `path:string → bool` | Returns whether a path is a directory. |
| `remove!` | `path:string → unit` | Deletes a file. Raises on failure. |
| `rename!` | `from:string → to:string → unit` | Renames or moves a file. Raises on failure. |
| `mkdir!` | `path:string → unit` | Creates a directory. Silently succeeds if it already exists. |
| `stat!` | `path:string → [size:integer, modified:integer, kind:atom] \| unit` | A path's stamp: its size, when it was last written in nanoseconds since the epoch (seconds' resolution on Windows), and `:file`, `:dir` or `:other`. `()` when nothing is there, which is an answer rather than a failure: a deleted input has moved as surely as a rewritten one. |
| `digest!` | `path:string → string` | The SHA-256 of a file's contents, as 64 lowercase hex digits, read in the VM rather than a chunk at a time through the heap. Raises as `open!` does. |
| `digest` | `data:string → string` | The SHA-256 of a string, the digest `digest!` gives the file holding the same bytes. Pure. |
| `mkdir_all!` | `path:string → unit` | The directory and every parent it needs. Already there is not an error. |
| `remove_all!` | `path:string → integer` | The path and everything under it, answering how many entries went. Nothing there answers 0 rather than failing. |
| `copy!` | `from:string → to:string → unit` | `to` becomes a copy of `from`, with its permissions, replacing it if it is there: a file of its own, so that what edits one later cannot reach the other. |
| `link!` | `from:string → to:string → atom` | `to` becomes another name for `from`: a hard link (`:linked`), or a copy where one cannot be made (`:copied`). `to` must not exist yet. For a file nothing will write to again, such as another build step's output. |
| `chmod!` | `path:string → mode:integer → unit` | Set the permission bits, `0o755` being `493`. On Windows only the owner's write bit means anything, and the rest are accepted and ignored. |
| `mode!` | `path:string → integer \| unit` | The permission bits of what the path leads to, as `chmod!` takes them (`0o755` is `493`); `()` when nothing is there. Windows reports only its read-only attribute: `0o666` or `0o444`, plus execute bits for a directory or a program. |
| `lstat!` | `path:string → [size:integer, modified:integer, kind:atom] \| unit` | `stat!` of the name itself rather than where it leads: a symbolic link answers `:link`, its size the length of its target, and a dangling one is still there. Anything else answers what `stat!` does. |
| `is_link!` | `path:string → bool` | Whether the name is a symbolic link, dangling or not. |
| `symlink!` | `target:string → to:string → unit` | `to` becomes a symbolic link holding `target` as written, in `ln -s`'s order. A relative target is read from the link's directory and need not exist; `to` must not. On Windows it needs developer mode or the privilege, and otherwise raises `:permission_denied`. |
| `read_link!` | `path:string → string` | The target a symbolic link holds, as written rather than resolved. `:wrong_kind` when the path is not a link. |
| `walk!` | `dir:string → list of string` | Every file under `dir`, relative to it with `/` between the parts, sorted. Directories are walked and not listed, and a link to a directory is not followed, so a tree that links into itself is still finite. |

### Error atoms

`:io_closed` · `:not_found` · `:permission_denied` · `:already_exists` · `:wrong_kind` · `:io_error` · `:type_error`

---

## `std.net`

TCP networking. Sockets are regular I/O handles — `io.read!` and `io.write!` work on them directly. This module adds only the TCP-specific operations.

All blocking operations park the calling process, not the worker thread.

```dream
import std.net;
```

| Name | Signature | Description |
|------|-----------|-------------|
| `listen!` | `port:integer → handle` | Binds a TCP listener on `port`, on every interface. Use port `0` to let the OS pick a port, then read it back with `port!`. |
| `listen_on!` | `host:string → port:integer → handle` | `listen!` at one address: `"127.0.0.1"` or `"localhost"` for this machine alone, `"0.0.0.0"` for every interface. IPv4, as `connect!` is. A server meant only for this machine should bind this way. |
| `accept!` | `listener → handle` | Waits for an incoming connection and returns a stream handle. Parks until a connection arrives. |
| `connect!` | `host:string → port:integer → handle` | Connects to a TCP server. Parks during the connect. Raises `:connection_refused` if the server is not listening. |
| `peer!` | `socket → string` | Returns the remote address as `"host:port"`. |
| `port!` | `socket\|listener → integer` | Returns the local port number. Useful after binding to port `0`. |
| `shutdown!` | `socket → unit` | Half-closes the connection (stops sending). The peer sees end-of-file on their read, but this end can still receive. |

### Error atoms

`:connection_refused` · `:connection_lost` · `:address_in_use` · `:timed_out` · `:io_closed` · `:io_error`

---

## `std.tls`

TLS on a socket. A connected stream handle is upgraded **in place**: `connect!` and `accept!` answer the same handle, and from then on `io.read!`, `io.write!`, `io.close!` and `net.shutdown!` on it carry plaintext through the session — so everything written over sockets (`std.streams`, `std.remote`, a program's own reader) works over TLS unchanged. Upgrading a connection that has already spoken plain text is how PostgreSQL's SSLRequest and SMTP's STARTTLS work, and is why there is no separate `dial`.

The library is the platform's and invisible to Dream: OpenSSL 3 on Linux and macOS, SChannel on Windows. The same program behaves the same on each: the options are what every backend can honour, and a failure is one of the kinds below whatever the library called it. Every blocking step — the handshake, a read waiting for a record, a write waiting for the socket — parks the process, not the worker. See `dream/src/tls.hpp` for the design.

```dream
import std.net;
import std.tls;

let sock = net.connect! "example.com" 443;
tls.connect! sock %{ :host => "example.com", :alpn => ["http/1.1"] }
io.write! sock "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
```

| Name | Signature | Description |
|------|-----------|-------------|
| `connect!` | `socket → options:map → socket` | The client's handshake. Raises the kind of failure below when the server cannot be trusted. |
| `accept!` | `socket → options:map → socket` | The server's handshake, on a socket from `net.accept!`. |
| `info!` | `socket → map\|unit` | `%{ :version, :cipher, :alpn, :peer }` for a TLS socket — `:peer` the subject of the other side's certificate, `:alpn` the protocol chosen, either `()` when there is none — and `()` for a plain one. Version and cipher are spelled by the library. |
| `backend` | `unit → string` | The TLS library this VM was built with, and its version. |

Options, all optional except as noted:

| Key | Meaning |
|-----|---------|
| `:host` | The name (or address) the server's certificate must carry, also sent as SNI. **Required** for a client that verifies. |
| `:verify` | Client: check the server's certificate (default `true`). Server: require a client certificate and check it against `:ca` (default `false`). |
| `:check_name` | Client: whether checking includes `:host` (default `true`). `false` checks the chain alone — libpq's `verify-ca`. |
| `:ca` | PEM certificates to trust **instead of** the system's store. Exclusive rather than additional, so a program that names its roots trusts the same thing on every machine. |
| `:crl` | PEM certificate revocation lists. A certificate one of them lists is refused as `:certificate_revoked`, on either side. |
| `:identity` | This side's certificate, chain and key, as the bytes of a **PKCS#12** bundle — the one format every backend imports. Required for a server. |
| `:password` | The PKCS#12 bundle's password. |
| `:alpn` | Protocol names to offer (client) or accept (server), in preference order. |

TLS 1.2 is the oldest version either side will speak.

**Revocation** is checked from what the handshake already has, never fetched: the lists given as `:crl`, and an OCSP response the server staples (OpenSSL's client asks for one; SChannel consults whatever Windows has cached, a staple included). A certificate one of them says is revoked is refused; a certificate nothing speaks for is accepted, because its status is unknown rather than bad — and because a download in the middle of a handshake would hold a worker for as long as a remote server took.

On Windows SChannel does its cryptography outside the process, so a PKCS#12 identity's key is imported into the user's key store for the life of the VM and deleted when it exits. Each such key is also written down in `%LOCALAPPDATA%\dream\tls-keys`, and a VM that was killed before it could delete its keys has them deleted by the next VM to import an identity.

### Error atoms

`:certificate_untrusted` · `:certificate_expired` · `:certificate_revoked` · `:hostname_mismatch` · `:handshake_failed` · `:tls_config` (an unreadable `:ca`, `:crl` or `:identity`, a wrong password) · `:tls_error` · `:bad_argument` · `:wrong_kind` · `:io_closed`

---

## `std.crypto`

Hashes, MACs, key derivation and randomness, done by the machine. Writing these in Dream costs thousands of reductions per block; here each is one call. Everything is **bytes in, bytes out**: a digest, a MAC or a derived key is a string of raw bytes, ready to be the next step's key or salt, and `hex`/`base64` spell one for printing. The hashes are written out in the VM (`dream/src/digest.cpp`), not taken from the TLS library, so they are the same on every platform.

Where the algorithm is a parameter it is an atom: `:md5`, `:sha1`, `:sha256`, `:sha384` or `:sha512`. Anything else raises `:type_error`.

```dream
import std.crypto;

crypto.hex (crypto.sha256 "abc")               // "ba7816bf..."
crypto.hmac :sha256 key message                // 32 raw bytes
crypto.pbkdf2 :sha256 password salt 600000 32  // a 32-byte key
crypto.equal expected_mac given_mac            // constant time
```

| Name | Signature | Description |
|------|-----------|-------------|
| `md5`, `sha1`, `sha256`, `sha384`, `sha512` | `string → string` | The digest, as raw bytes (16, 20, 32, 48, 64). |
| `digest` | `alg → string → string` | The same, with the hash chosen by the caller. |
| `digest_file!` | `alg → path → string` | The digest of a file's contents, read by the VM a block at a time, so a file of any size costs the heap nothing. Raises `:not_found`, `:wrong_kind` (a directory) or `:permission_denied`. |
| `hmac` | `alg → key → message → string` | HMAC (RFC 2104). |
| `pbkdf2` | `alg → password → salt → iterations → length → string` | PBKDF2 with HMAC (RFC 8018). The key's padding is hashed once, so each round costs two compressions. |
| `hkdf` | `alg → key → salt → info → length → string` | HKDF extract-and-expand (RFC 5869). An empty salt is the RFC's default. |
| `equal` | `string → string → bool` | Whether two strings hold the same bytes, in time that depends only on their lengths — for checking a MAC. |
| `xor` | `string → string → string` | Two strings of one length, exclusive-or'd byte by byte. |
| `hex`, `unhex` | `string → string`, `string → string\|unit` | Lowercase hex, and back (either case); `()` for text that is not hex. |
| `base64`, `unbase64` | `string → string`, `string → string\|unit` | Padded RFC 4648 base64, and back; `()` for text that is not. |
| `random_bytes!` | `integer → string` | That many bytes from the operating system's generator (`getrandom`, `arc4random_buf`, `BCryptGenRandom`), up to 2^24. |

A native runs to completion on its worker, so an expensive `pbkdf2` holds that worker for as long as it takes; the iteration count is the caller's choice.

---

## `std.os`

Operating system interface: arguments, environment, filesystem traversal, subprocesses.

```dream
import std.os;
```

### Environment

| Name | Signature | Description |
|------|-----------|-------------|
| `args!` | `unit → list of string` | The command-line arguments that follow the image name. |
| `env!` | `name:string → string\|unit` | The value of an environment variable, or `unit` if unset. |
| `set_env!` | `name:string → value:string → unit` | Sets an environment variable for the current process. |
| `install_dirs!` | `unit → list of string` | Every directory of the installation, as the VM searches them for `dream NAME`: `$MINDV2_PATH` split and `~`-expanded, or `~/.mindv2` when that is unset and exists, or `[]`. `std.install` is the module to use. |

### Directories

| Name | Signature | Description |
|------|-----------|-------------|
| `cwd!` | `unit → string` | The current working directory. |
| `chdir!` | `path:string → unit` | Changes the working directory. Raises `:not_found` or `:os_error` on failure. |
| `list_dir!` | `path:string → list of string` | The entries in a directory, sorted alphabetically, without `.` or `..`. Raises `:not_found` if the path does not exist. |

### Subprocesses

| Name | Signature | Description |
|------|-----------|-------------|
| `exec!` | `program:string → args:list of string → map` | Runs `program` to completion and returns a map `%{ :code, :out, :err, :timed_out }`. `program` is resolved via `PATH`. Parks the calling process — not the worker thread — while the child runs. |
| `exec_for!` | `program:string → args:list of string → timeout_ms:integer → map` | Same as `exec!` but kills the child after `timeout_ms` milliseconds. Sets `:timed_out true` in the result map when the deadline is hit, so the caller can distinguish that from an ordinary non-zero exit code. |
| `exec_in!` | `dir:string → program:string → args:list of string → timeout_ms:integer → map` | `exec_for!` with the child started in `dir`, and `0` for no deadline. The directory is the child's own: `chdir!` moves the whole VM's, which two concurrent build steps cannot share. A relative `program` is looked up from `dir`. |
| `exec_with!` | `dir:string → env:list of string → program:string → args:list of string → timeout_ms:integer → map` | `exec_in!` with variables of the child's own: each `NAME=value` in `env` is laid over this VM's environment for that child alone, since `set_env!` changes the whole VM's. |
| `exec_joined!` | `dir:string → env:list of string → program:string → args:list of string → timeout_ms:integer → map` | `exec_with!` with the child's errors written into its output, in the order they happened: `:out` is everything it printed and `:err` is empty. What a check comparing a program's output with a recorded file runs it with, since two streams read apart cannot be put back in order. |
| `replace!` | `program:string → args:list of string → never` | **Becomes** `program`: `execvp`, so this VM — image, heap and every thread — is gone and the named program takes over the process, inheriting the terminal and every open descriptor. Stdio is flushed first. It returns only by failing, raising `:not_found` when the program cannot be run. Use it to hand over to something interactive; `exec!` gives its child pipes, which is right for a compiler and useless for anything that prompts. |

The result map fields:

| Field | Type | Description |
|-------|------|-------------|
| `:code` | `integer` | Exit code. 128 + signal number if killed by a signal. |
| `:out` | `string` | Everything the child wrote to stdout. |
| `:err` | `string` | Everything the child wrote to stderr. |
| `:timed_out` | `bool` | `true` only when killed by `exec_for!`'s timeout. |

### Process and platform

| Name | Signature | Description |
|------|-----------|-------------|
| `monotonic!` | `unit → integer` | Milliseconds from a fixed point in this process's life, from a steady clock that never jumps. Only *differences* between two readings mean anything — that difference is a duration. Timing anything lazy means forcing it first: an unforced value has not run, so a reading around one times the building of a thunk. |
| `now!` | `unit → integer` | Milliseconds since the Unix epoch, from the wall clock. It can jump, forwards or back, so it is what to stamp a log line with and never what to measure a duration with. |
| `pid!` | `unit → integer` | The OS process ID of the running VM. |
| `platform` | `unit → atom` | The current platform: `:linux`, `:macos`, `:windows`, or `:unknown`. |
| `arch` | `unit → atom` | `:x86_64`, `:aarch64` or `:unknown`: the architecture the VM was built for, which is the one a program's native code has to match. |
| `exit!` | `code:integer → never` | Terminates the entire VM immediately with the given exit code. Flushes stdio first. Never returns. |

---

## `std.vm`

Runtime introspection. Useful for monitoring, debugging, and detecting deadlocks.

```dream
import std.vm;
```

### Per-process stats

These report figures for the **calling** process.

| Name | Signature | Description |
|------|-----------|-------------|
| `processes!` | `unit → integer` | The total number of live processes in the runtime right now. |
| `reductions!` | `unit → integer` | The number of reduction steps this process has taken. |
| `collections!` | `unit → integer` | The number of GC collections on this process's heap. |
| `heap_bytes!` | `unit → integer` | Bytes currently allocated in this process's heap. |

### Runtime-wide

| Name | Signature | Description |
|------|-----------|-------------|
| `modules!` | `unit → list of string` | The names of all loaded modules. |
| `async_io` | `unit → bool` | Whether epoll-based async I/O is available. |

### Sharing between processes

| Name | Signature | Description |
|------|-----------|-------------|
| `share!` | `value → value` | The same value, forced all the way down and moved into the runtime's shared area. From then on a `spawn!`, `send!` or `join!` carrying it copies a pointer rather than the value. |
| `wire_encode` | `value → string` | The value, forced all the way down, in `std.wire`'s format, byte for byte what `std.wire`'s Dream writer produces, and thousands of times faster: `std.wire.encode` is this. A map is written in the order `compare` puts its keys; what is not data is written as its `to_string`. |
| `wire_decode` | `string → Result.ok value \| Result.error string` | One `std.wire` message, read whole. Never raises. Trailing bytes, a truncated field, an integer past 63 bits and an atom this program does not have are all `Result.error why` -- an atom is looked up, never made. `std.wire.decode` is this. |
| `regex_run` | `code:string → subject:string → from:integer → mode:integer → value` | Run a pattern `std.regex` compiled (`Regex.code`) over a string: mode 0 answers whether it matches, 1 the first match at or after `from` as an array of capture offsets (unit for none), 2 every match as a list of those, 3 how many there are, 4 the match starting exactly at `from`. A lazy DFA and a Pike VM, linear in the text; engines are cached per worker. `std.regex` is the interface; dream/src/regex.cpp is the design. |
| `share_arenas` | `[funcs, parts, invented_from, filled, edges, runs, part_base] → [nodes, kids, funcs]` | The compiler's `opt.optimize_parts`: a program's parts, each an arena already shared within itself, hash-consed into one -- constants renamed through each part's pools, functions and invented globals moved past the parts before, a settled `comp` placeholder read from part 0. Step for step the walk `opt.optimize_parts_by_hand` writes in Dream, which `opt`'s tests hold it to; the opcode tables are the compiler's, handed in. |
| `node_section` | `[nodes, op_codes] → string` | An image's NODE section: each node `[op, flags, a, b, c]` as `u8 opcode, u8 flags, u16 0` and three indices, `:none` as all ones, the opcode from the compiler's table. The bytes `emit.node_bytes` writes, which `emit`'s tests hold it to. |
| `index_section` | `list → string` | An image's KIDS section: each index as a little-endian `u32`, `:none` as all ones. The bytes `emit.index` writes. |
| `shared_bytes!` | `unit → integer` | Bytes in the runtime's shared area. |
| `release!` | `value → :ok \| :not_shared` | Say the program is done with what a `share!` answered. Its region is freed once a census finds nothing anywhere still holding it -- a value still in use simply stays, so a wrong release breaks nothing. Starts that census. |
| `collect_shared!` | `unit → :ok` | Run a census again, for regions released while something still held them. |

Processes share nothing: a value that crosses between two is copied, which is
what lets each heap collect on its own. For a large table that many processes
only *read*, the copy is the cost — a compiler handing its resolution to four
workers spent more copying it than the split saved. `share!` copies it once,
into memory every process of the runtime can read and none owns.

Three things follow, and they are the contract:

- The answer is equal to the argument in every way a program can observe; only
  where it lives changes. A map put, a list cons or anything else built *on*
  a shared value is an ordinary value of the process that built it.
- A value must be data after forcing. A closure whose captured state is still
  a suspension is refused with `:type_error` rather than shared, because
  forcing a suspension writes to it and nothing shared may ever be written.
- **Nothing shared is freed until it is released and nobody holds it.** Each
  `share!` copies into a region of its own; `release!` asks for it back, and a
  census -- a major in every process, a walk of every message waiting and of
  what finished processes left -- frees it once nothing reaches it. A census
  costs a major per process, so it runs only when asked. `dreams` shares only
  when it is compiling from the command line and never releases: it exits.

### Watching, stopping and waiting

What `std.proc` builds selective receive, timeouts and supervision from; a
program reaches for `std.proc`, which says the same things in fewer words.
They are natives of this module rather than builtins so that adding them took
no new compiler.

| Name | Signature | Description |
|------|-----------|-------------|
| `monitor!` | `process → :ok` | Be sent `[:down, p, outcome]` when `p` ends -- at once if it has -- where `outcome` is `Result.ok value` or `Result.error e`. A failure delivered this way counts as handled, as one delivered to a joiner does. |
| `demonitor!` | `process → :ok` | Undo one `monitor!`. A report already sent stays in the mailbox. |
| `kill!` | `process → value → :ok` | End the process as a failure of kind `:killed` whose payload is the value. It is not unwound and cannot catch it: it is not run again after its current slice. A parked process is woken to be stopped. A process can kill itself, and then the call does not return. |
| `unique!` | `unit → integer` | An integer no other call in this VM has answered: the reference a request carries so its reply cannot be mistaken for another's. |
| `mailbox_size!` | `unit → integer` | Messages waiting. |
| `mailbox_peek!` | `integer → Result.ok value \| :none` | The message at a position, left where it is. Only the owner ever removes a message, so a position keeps naming the same one until this process takes it. |
| `mailbox_take!` | `integer → :ok` | Remove the message at a position; `:out_of_bounds` when there is none. |
| `await_message!` | `integer → integer → :ok \| :timeout` | Wait until the mailbox holds more than `n` messages, or until `now_ns!` reaches the deadline; a negative deadline waits for ever. The deadline is absolute because a blocked native is re-entered from the top on every wake. |
| `sleep_until!` | `integer → :ok` | Park until `now_ns!` reaches the deadline. |

A timed wait counts, for the deadlock check, as a wait on something outside the
scheduler -- as a descriptor does -- so a program whose every process is asleep
is not reported as deadlocked.

### Inspection

| Name | Signature | Description |
|------|-----------|-------------|
| `processes_info!` | `unit → list of map` | A snapshot of every live process. |
| `process_info!` | `process\|integer → map\|unit` | Info for one process by handle or numeric ID. Returns `unit` if the process is not found. |
| `scheduler!` | `unit → map` | Scheduler state: `:workers`, `:idle`, `:live`, `:runnable`, `:queued`, `:io_waiters`, `:reductions`, `:deadlocked`. |
| `io!` | `unit → list of map` | Every open I/O handle: `:handle`, `:fd`, `:kind`, `:busy`, `:waiting` (pid or `unit`). |
| `dump!` | `unit → unit` | Prints the full VM state (scheduler + all processes + all handles) to stderr. Same output as `DREAM_STUCK_SECONDS`. |

Process info map fields: `:id` · `:status` (`:runnable`, `:running`, `:waiting`, `:finished`, `:failed`) · `:waiting_on` (`:message`, `:join`, `:io`, `:timer`, `:none`) · `:fd` · `:reductions` · `:heap_bytes` · `:collections` · `:mailbox` (count) · `:failed`

### Compile-time evaluation

A compiler written in the language it compiles has to *run* Dream to work out
what a `comp` expression comes to or what a macro expands to, and the only
thing that knows how to run Dream is the VM. These are how it asks. Each
answer is produced in a runtime of its own — its own image, heap, atom table
and scheduler — and crosses back as **data**: a closure, a process or an I/O
handle cannot be a compile-time answer, and neither can be a compile-time
argument.

| Name | Signature | Description |
|------|-----------|-------------|
| `eval_image!` | `string → value` | Load an image, run its entry point, and answer what it produced, forced all the way down. The runtime is discarded afterwards. |
| `open_image!` | `string → integer` | Load an image and **keep** it, answering a handle that names it. |
| `call_image!` | `integer → string → string → list → value` | Call `module.member` in an open image with the arguments in the list, and answer what it produced. |
| `close_image!` | `integer → bool` | Free an open image. `false` when the handle named none; closing twice is not an error. |
| `image_function` | `image → string → integer → fn` | A function in another image as a Dream function of that many arguments, named with a `!`; see [images.md](images.md). The image is opened once per runtime, on first use. |
| `pure_image_function` | same | The same, named without one: for a function with no effects. |
| `locate_image!` | `image → Result.ok file \| Result.error why` | Where an image descriptor is found, or why it is not. |

`eval_image!` is for one image and one question: the image *is* the expression.
The other three are for one image and many questions — a package's macros,
compiled once and then called — which is what keeps the arguments out of the
image. Since the arguments are values rather than quoted code, nothing has to
be recompiled per call.

Handles are integers rather than objects on purpose: what one names is a whole
runtime, and tying its lifetime to a collector is how it ends up freed while
something inside it is running. Handles are never reused, so a stale one names
nothing rather than someone else's image. Whatever a program leaves open is
freed when its own runtime goes away.

A member is named by its module *and* its own name, because a global's name is
not unique in an image — two modules may each declare `helper`. A pure nullary
global is a value, so naming one whose value is a function applies the
arguments to what it comes to; that is what makes an alias callable.

| Failure | When |
|---|---|
| `:bad_image` | the bytes are not a loadable image, the handle names no open image, or the image has no such member |
| `:comp_failed` | the call raised or did not finish, or a value crossing either way is not data |

---

## `std.ffi`

Calling C. A required module: every VM provides it, and the VM's build fails
without libffi rather than produce one that cannot call C.
[`std.foreign`](ffi.md) is the Dream-level half, and the one to write a wrapper against: it gives the vocabulary below
types, so a signature written as a literal is checked while compiling.

```dream
import std.ffi;

let sqrt  = ffi.pure_function "libm.so.6" "sqrt" [:f64] :f64;
let puts! = ffi.function () "puts" [:cstr] :int;
```

### Foreign functions

| Name | Signature | Description |
|------|-----------|-------------|
| `function` | `library → symbol → [arg type] → result type → fn` | A C function as a Dream function of its arguments -- every one that is not `[:out, t]`. Named `symbol!`, so the runtime treats calling it as an effect; bind it to a `!` name. |
| `pure_function` | same | The same, named `symbol`: for a C function with no effects, which a pure function may call. |
| `call!` | `fn → [args] → value` | A foreign function applied to a list -- what lets a function and its arguments travel in one message. |
| `locate!` | `library → Result.ok file \| Result.error why` | Open a library and say which file it came from, or what was tried. |

Both constructors are pure and lazy: nothing is loaded until the function is
made, and a library is opened once per VM. A library is a path, `()` or `""`
for the running program (the C library on any normal system),
`[:payload, name]` / `[:payload, index]` for one carried in the image by
`dreams --payload NAME=FILE`, `[:search, name, files, dirs]` for one installed
on the system (`foreign.system` makes it; [ffi.md](ffi.md#finding-a-library-on-the-system)
is the search), or `[:any, libraries]` for the first of several that opens.

**Types.** Scalars: `:void` (result only), `:bool`, `:i8`…`:i64`, `:u8`…`:u64`,
`:f32`, `:f64`, and C's own names, whose widths are the platform's: `:char`,
`:uchar`, `:short`, `:ushort`, `:int`, `:uint`, `:long`, `:ulong`, `:longlong`,
`:ulonglong`, `:size`, `:ssize`, `:intptr`, `:uintptr`, `:double`. There is no
`:float`, since in Dream that is a double; C's `float` is `:f32`.

| Type | Where | Meaning |
|------|-------|---------|
| `:ptr` | anywhere | a raw address, as an integer, or `()` for null. Accepts any handle as an argument. Checks nothing |
| `:cstr` | anywhere | a string as a NUL-terminated pointer; a result is copied, `NULL` is `()` |
| `:bytes` | argument | a string's bytes, or a payload view's, with no terminator promised |
| `[:handle, tag]` | argument | a live handle of that tag, owned by this process; passes its pointer |
| `:buffer` | argument | `[:handle, "buffer"]` |
| `[:own, tag, destructor]` | result, `out` | the pointer becomes a handle this process owns; `destructor` is a symbol in the library, run on release |
| `[:borrow, tag]` | result, `out` | a handle with no destructor, for a pointer something else frees |
| `[:cstr, destructor]` | result, `out` | copy the string, then free the pointer with `destructor` |
| `[:out, t]` | argument | not passed by the caller: C is given somewhere to write a `t`, and the call answers `[result, out1, ..]` |
| `[:callback, [args], result]` | argument | a pure Dream function C may call during this call; numbers, `:ptr` and `:cstr` only |

### Owning C resources

A handle is `[:foreign, tag, id]`. The pointer stays in the VM, in a table
belonging to the process that made the call; the handle is only a name for an
entry. So a released handle, one belonging to another process, and one of the
wrong tag are all `:ffi_error`s rather than undefined behaviour.

A resource made by a call that took handles records them as its *parents*.
Releasing a parent releases its children first, newest first. Whatever a
process has not released is released when it ends, whether it returned, raised
or called `os.exit!`, newest first, which is always children before parents.

| Name | Signature | Description |
|------|-----------|-------------|
| `release!` | `handle → unit` | Run the destructor now, after releasing everything made from it. Raises for a handle that is not live here. |
| `alive!` | `handle → bool` | Whether this process still owns it. |
| `owned!` | `unit → [handle]` | Every handle this process owns, oldest first. |
| `address!` | `handle → integer` | The raw address. The one way to take a pointer out of the table. |

### Buffers

Memory this process allocated, of a known size, so that every access is
checked against it. A pointer C hands back has no size and cannot be read
directly; that is the point.

| Name | Signature | Description |
|------|-----------|-------------|
| `buffer!` | `size → handle` | `size` zeroed bytes, owned, tagged `"buffer"`. |
| `size!` | `handle → integer\|unit` | A buffer's size; `()` for a handle whose size is unknown. |
| `peek!` | `buffer → offset → type → value` | Read one scalar or `:ptr`. |
| `poke!` | `buffer → offset → type → value → unit` | Write one. |
| `read!` | `buffer → offset → length → string` | Bytes, as a string. |
| `read_string!` | `buffer → offset → string` | Up to a NUL or the end of the buffer. |
| `write!` | `buffer → offset → string → unit` | A string's (or payload view's) bytes. |
| `sizeof` | `type → integer` | Size in bytes of a scalar type. |
| `alignof` | `type → integer` | Its alignment. |
| `payload_index` | `name → integer\|unit` | Which payload `--payload NAME=FILE` called `name`. |

### The first interface

Kept, and now read by the same signature reader, so `bind!` takes every type
above: `open! name → handle`, `close! handle`, `bind! handle symbol args result`,
`load! name symbol args result`, and raw addresses as integers: `alloc! size`,
`free! address`, `read_cstr! address`, `read_u8! base offset`,
`write_u8! base offset byte`. None of the raw-address members is checked;
new code wants buffers.

---

## Dream-level stdlib (`mind/std/`)

These modules are written in Dream and ship with the compiler. They are built on the native primitives above.

### `std.list`

```dream
import std.list;
```

A lazy singly-linked list. Most operations work on infinite lists. Functions that must see the entire list (`length`, `reverse`, `sort`) say so in their descriptions.

| Name | Description |
|------|-------------|
| `empty` | The empty list `[]`. |
| `head xs` | First element. Raises on empty; see `first_or`. |
| `tail xs` | Everything after the first element. |
| `cons x xs` | Prepends `x`. Lazy in both arguments. |
| `is_empty xs` | `true` if empty. |
| `first_or default xs` | First element, or `default` if empty. |
| `length xs` | Forces the entire spine. Does not terminate on infinite lists. |
| `nth n xs` | Element at index `n` (0-based), or `unit` if out of range — including a negative `n`. This is `xs.[n else ()]`, so a saturated call is the `get` opcode. |
| `last xs` | Last element, or `unit` if empty. Forces the entire spine. |
| `append xs ys` | Concatenates two lists; this is what `xs + ys` means. `ys` and every element are left untouched, but the spine of `xs` is walked when the result is forced, so appending to an *infinite* left-hand list does not terminate. |
| `reverse xs` | Forces the entire spine. |
| `range from until` | `[from, from+1, ..., until-1]`. The end is exclusive. |
| `from n` | Infinite list `n, n+1, n+2, ...` |
| `repeat x` | Infinite list of `x`. |
| `replicate n x` | `x` repeated `n` times. |
| `map f xs` | Apply `f` to every element. Lazy. |
| `filter keep xs` | Elements for which `keep` returns `true`. |
| `fold f acc xs` | Left fold. Forces the entire spine and each accumulator step to its outer value; elements and accumulator contents stay lazy. The seed reaches the first callback unforced. |
| `fold_lazy f acc xs` | Left fold with suspended accumulator steps, so later steps can discard earlier ones. Also available on arrays and maps. |
| `fold_strict f acc xs` | Compatible spelling for `fold`. |
| `fold_right f acc xs` | Right fold. |
| `take n xs` | The first `n` elements. Terminates on infinite lists. |
| `drop n xs` | Everything after the first `n` elements. |
| `take_while keep xs` | Elements while `keep` holds, then stops. |
| `drop_while skip xs` | Drops elements while `skip` holds, then returns the rest. |
| `zip xs ys` | Pairs of `[x, y]`. Stops at the shorter list. |
| `concat xss` | Flattens a list of lists. |
| `flat_map f xs` | `map` then `concat`. |
| `enumerate xs` | Pairs each element with its index: `[[0, x], [1, y], ...]`. |
| `any pred xs` | `true` if any element satisfies `pred`. Short-circuits. |
| `all pred xs` | `true` if every element satisfies `pred`. Short-circuits. |
| `contains x xs` | `true` if `x` appears in the list. |
| `count pred xs` | Number of elements satisfying `pred`. |
| `index_of x xs` | Index of the first occurrence of `x`, or `-1`. |
| `sum xs` | Sum of all elements. |
| `product xs` | Product of all elements. |
| `minimum xs` | Smallest element by `compare`, or `unit` if empty. |
| `maximum xs` | Largest element by `compare`, or `unit` if empty. |
| `partition pred xs` | `[passing, failing]` — two lists. |
| `unique xs` | Removes duplicates, keeping the first occurrence. |
| `sort xs` | Stable sort in ascending order by `compare`. Forces the entire spine. |
| `sort_by before xs` | Stable sort with a custom comparator `before a b → bool`. |
| `sort_on key xs` | Stable sort ascending by `key` applied to each element. |
| `to_array xs` | Converts to an array (`array.of_list`). |
| `of_array a` | Converts an array to a list. |
| `force xs` | Forces every element. Useful before `send!`. |

**Accumulator-passing forms.** Several functions above are thin wrappers over a
recursive worker that carries its accumulator as the first argument. The worker
is reachable too, and is the form to use when you already have a partial result
to continue from.

| Name | Description |
|------|-------------|
| `length_from acc xs` | `acc` plus the length of `xs`. |
| `reverse_from acc xs` | `xs` reversed, with `acc` left on the end: `reverse_from [9] [1,2,3]` is `[3, 2, 1, 9]`. This is `reverse` and `append` in one pass. |
| `index_of_from i x xs` | Index of the first `x`, counting as if `xs` started at index `i`; `-1` if absent. `i` is an offset added to the answer, **not** a position to start searching from. |
| `min_by_from best less xs` | The smallest of `best` and the elements of `xs`, by the comparator `less a b → bool`. Returns `best` unchanged on an empty list, which is how `minimum`/`maximum` get a seed without a special case for one-element lists. |
| `unique_from seen xs` | The elements of `xs` not already in the list `seen`, with duplicates removed. Elements in `seen` are dropped from the output. |
| `merge_by before xs ys` | Merges two lists that are **already sorted** by `before`, preserving order. The merge step of `sort_by`; on unsorted input it interleaves rather than sorts. |

---

### `std.str`

```dream
import std.str;
```

UTF-8 text. Derives `std.seq`, so it also exposes `sum`, `any`, `all`, `contains`, `minimum`, `maximum`, `join`, `describe`, and `count` over characters.

**Characters and bytes are distinct.** `length` counts characters; `byte_length`, `slice`, and `find` operate on bytes.

| Name | Description |
|------|-------------|
| `length s` | Number of Unicode characters (decodes the string). |
| `byte_length s` | Number of bytes. Constant time. |
| `is_empty s` | `true` if `s` has zero bytes. |
| `chars s` | List of characters (decoded codepoints). |
| `of_chars cs` | String from a list of characters. |
| `first_or default s` | First character, or `default`. |
| `slice from count s` | `count` bytes from byte offset `from`. Clamped silently. |
| `byte i s` | The byte at offset `i` as an integer, or `-1`. Works on a bigstr, where it is the way in. |
| `find needle s` | Byte offset of `needle` in `s`, or `-1`. |
| `find_from from needle s` | Byte offset of `needle` at or after `from`, or `-1`. |
| `span set from s` | Past the run of bytes from `set` starting at byte offset `from`: the offset of the first byte that is not one of them, or `byte_length s`. `set` is a string read as the set of its bytes. |
| `upto set from s` | The offset of the first byte from `set` at or after `from`, or `byte_length s` when there is none. |
| `contains_str needle s` | `true` if `needle` appears anywhere in `s`. |
| `starts_with prefix s` | `true` if `s` begins with `prefix`. |
| `ends_with suffix s` | `true` if `s` ends with `suffix`. |
| `concat a b` | Concatenates two strings (same as `a + b`). |
| `join_str sep parts` | Joins a list of strings with `sep` between them. Unlike `seq.join`, does not call `to_string` on each part. |
| `repeat n s` | `s` repeated `n` times. |
| `split sep s` | Splits `s` on every occurrence of `sep`. An empty separator returns `[s]`. |
| `map f s` | Applies `f` to every character and rebuilds the string. |
| `filter keep s` | Keeps characters that satisfy `keep`. |
| `reverse s` | Reverses the string by characters (not bytes). |
| `upper s` | ASCII uppercase. Non-ASCII characters are unchanged. |
| `lower s` | ASCII lowercase. Non-ASCII characters are unchanged. |
| `trim s` | Removes leading and trailing whitespace (space, tab, newline, carriage return). |
| `trim_start s` | Removes leading whitespace. |
| `trim_end s` | Removes trailing whitespace. |

**Character-level helpers.** These take and return a `char`, not a string, and
are what `upper`, `lower` and the `trim` family are written in terms of.

| Name | Description |
|------|-------------|
| `is_space c` | `true` for space, tab, newline or carriage return. |
| `upper_char c` | ASCII uppercase of one character. Anything outside `a`–`z` is returned unchanged. |
| `lower_char c` | ASCII lowercase of one character. Anything outside `A`–`Z` is returned unchanged. |

---

### `std.array`

```dream
import std.array;
```

Fixed-length sequences with O(1) indexing, written `#[a, b, c]`. Derives `std.seq`, so it also exposes `sum`, `any`, `all`, `contains`, `minimum`, `maximum`, `join`, `count`, `describe`, and related functions.

| Name | Description |
|------|-------------|
| `length xs` | Number of elements. O(1). |
| `is_empty xs` | `true` if empty. |
| `get xs i` | Element at index `i`. Raises `:out_of_bounds` if out of range. |
| `get_or default xs i` | Element at index `i`, or `default` if out of range. |
| `first_or default xs` | First element, or `default`. |
| `last_or default xs` | Last element, or `default`. |
| `new n fill` | Array of `n` slots all set to `fill`. |
| `of_list xs` | Array from a list. |
| `to_list xs` | List from an array. |
| `build n f` | `#[f 0, f 1, ..., f (n-1)]`. |
| `set xs i v` | A copy of `xs` with index `i` replaced by `v`. |
| `map f xs` | Apply `f` to every element, return a new array. |
| `filter keep xs` | Elements satisfying `keep`, as a new (shorter) array. |
| `reverse xs` | Elements in reverse order. |
| `sort xs` | Sorted ascending by `compare`. |
| `sort_by before xs` | Sorted with a custom comparator. |
| `append xs ys` | Concatenation of two arrays. |
| `slice from until xs` | Elements from index `from` up to (not including) `until`. |

---

### `std.seq`

```dream
derive std.seq;
```

Generic sequence interface. Not imported directly — a module derives it and provides `fold`. `std.array` and `std.str` both do this. Operations over characters or array elements are all written in terms of `fold`.

| Name | Description |
|------|-------------|
| `fold f init xs` | *(virtual — must be implemented by the deriving module)* |
| `name xs` | *(virtual, optional — returns a string for use in `describe`)* |
| `length xs` | Count by folding (override in the deriving module if O(1) is available). |
| `is_empty xs` | `true` if `length` is 0. |
| `sum xs` | Sum of all elements. |
| `product xs` | Product of all elements. |
| `count keep xs` | Elements satisfying `keep`. |
| `any keep xs` | `true` if any element satisfies `keep`. Folds the whole sequence. |
| `all keep xs` | `true` if every element satisfies `keep`. |
| `contains wanted xs` | `true` if any element equals `wanted`. |
| `minimum xs` | Smallest element, or `unit` if empty. |
| `maximum xs` | Largest element, or `unit` if empty. |
| `minimum_by rank xs` | Element for which `rank` is smallest. |
| `maximum_by rank xs` | Element for which `rank` is largest. |
| `extreme_by better xs` | The element that `better a b → bool` prefers over every other, or `unit` if empty. `minimum` is `extreme_by (fn a b -> a < b)` and `maximum` is `extreme_by (fn a b -> a > b)`. |
| `to_list xs` | Convert to a list by folding. |
| `join sep xs` | Render each element with `to_string` and join with `sep`. |
| `describe xs` | `"<name> of <length>"`, e.g. `"array of 3"`. |

---

### `std.json`

```dream
import std.json;
```

JSON, parsed and rendered. **Nothing here raises.** A parser that raised could
not be called from a pure function, and reading a configuration file is not an
effect — so `parse` answers with a value describing the outcome instead.

| JSON | Dream |
|------|-------|
| object | `map`, with string keys |
| array | `list` |
| string | `string` |
| number | `integer` when it has no `.`, `e` or `E`; otherwise `float` |
| `true` / `false` | `bool` |
| `null` | `()` |

`null` and "absent" are therefore the same value, which is how the rest of the
library already answers a missing thing — and it means a round trip cannot tell
a member that was absent from one that was explicitly `null`. Neither can JSON.

#### Reading

| Name | Description |
|------|-------------|
| `parse text` | `Result.ok value`, or `Result.error message`. Trailing characters after the value are an error. |
| `parse_or default text` | The parsed value, or `default` if `text` is not valid JSON. The form to use when a fallback is more useful than a diagnosis. |

```dream
let r = json.parse "{\"a\": 1, \"b\": [true, null, 2.5]}";
is_ok r                         // true
unwrap_or %{} r                 // %{"a" => 1, "b" => [true, (), 2.5]}

json.parse "{oops}"             // Result.error("not valid JSON")
json.parse_or %{} "not json"    // %{}
```

#### Writing

| Name | Description |
|------|-------------|
| `write v` | `v` as compact JSON on one line, with no spaces. |
| `write_indented depth v` | `v` as JSON indented two spaces per level, starting at `depth`. |
| `pretty v` | `write_indented 0 v` — the usual entry point for readable output. |
| `quote s` | One string as a quoted, escaped JSON string, including the surrounding `"`. |
| `escape_char c` | One character as its JSON representation. Escapes `"`, `\`, and the characters with their own shorthand (`\n`, `\t`, `\r`, `\b`, `\f`); anything else below `0x20` becomes `\u00XX`. **Non-ASCII is left as itself**, so output stays UTF-8 rather than becoming escapes. Used by `quote`. |

```dream
json.write %{ "a" => 1, "b" => [1, true, ()] }   // {"a":1,"b":[1,true,null]}
json.pretty %{ "a" => [1, 2] }                   // multi-line, two-space indent
json.quote "he said \"hi\""                      // 16 characters, including the outer quotes
```

> **Map order is not insertion order.** A map is a hash table, so `write` emits
> members in whatever order the map holds them, and that order is not the one
> they were written in. JSON objects are unordered, so this is valid output —
> but it does mean two maps that compare equal can render as different text,
> and that output is not stable enough to compare byte-for-byte in a test. Sort
> `map.pairs` yourself if you need a canonical rendering.

#### Parser internals

The parser is recursive descent over a list of characters. Every step takes the
characters still to read and returns `[value, rest]` — or `()` when it does not
match — which is what lets the steps compose with no parser state threaded
through. These are not part of the interface, but they are reachable, and they
are the shape to copy when writing a parser of your own.

| Name | Description |
|------|-------------|
| `step value rest` | Builds the `[value, rest]` pair every step returns. |
| `step_value s` / `step_rest s` | The two halves back out of one. |
| `is_ws c` / `skip_ws cs` | Whitespace, and dropping a run of it. |
| `is_digit c` | `'0'`–`'9'`. |
| `starts cs word` | Does the character list `cs` begin with the characters of `word`? |
| `drop_n n cs` | Drops `n` characters. |
| `hex_value c` / `hex4 cs n acc` | One hex digit as a number (`-1` if it is not one), and the four digits of a `\uXXXX` escape accumulated into `acc` over `n` digits. |
| `string_body cs acc` | A string's characters up to the closing quote, handling escapes. |
| `number_chars cs acc floaty` | The characters of a number, and whether it had a `.` or an exponent. |
| `parse_number cs` · `parse_value cs` | The steps for a number and for any value. |
| `parse_object cs acc first` · `parse_array cs acc first` | The steps for `{…}` and `[…]`, accumulating into `acc`; `first` tracks whether a separating comma is required yet. |

---

### `std.toml`

```dream
import std.toml;
```

TOML — **deliberately a subset**, and the subset is the one a `mind.toml`
manifest uses: comments, `[section]` and `[dotted.section]` headers,
`key = value`, and values that are strings, integers, floats, booleans, arrays
or inline tables.

What is missing is what a manifest has no use for: array-of-table headers
`[[x]]`, multi-line strings, dates, and dotted keys inside a section. Using one
is an **error rather than a quietly wrong parse** — though the message says
only where the parse gave up, not which unsupported construct caused it:
`[[x]]` reports `"expected a key"`. Like `std.json`, nothing here raises.

The result is a map of section name to a map of that section's keys. Keys
written before any header are collected under the empty string `""`:

```toml
name = "top"           %{ ""             => %{ "name" => "top" },
[package]                 "package"      => %{ "name" => "demo", "ver" => 2 },
name = "demo"             "dependencies" => %{ "a" => %{ "path" => "../a" } } }
ver = 2
[dependencies]
a = { path = "../a" }
```

An inline table becomes a nested map, so `a = { path = "../a" }` reads back as a
map — which is what lets a dependency carry `path`, `git`, `tag` and the rest.

#### Reading

| Name | Description |
|------|-------------|
| `parse text` | `Result.ok table`, or `Result.error message`. |
| `ok result` / `value result` | As `std.json` — did it parse, and the table out of it. |
| `parse_or default text` | The parsed table, or `default`. |

#### Reading what was parsed

| Name | Description |
|------|-------------|
| `section table name` | The map for `[name]`, or an **empty map** if there is no such section. A missing section reads exactly like an empty one, so a manifest with no `[dependencies]` needs no special case. |
| `get default table sec key` | `table[sec][key]`, or `default` if either the section or the key is absent. |
| `sections table` | The name of every section. Includes `""` when the file had keys before its first header. |
| `entries table name` | One section's `key = value` pairs as `[key, value]` lists — `map.pairs` of that section. |

```dream
let t = toml.parse_or %{} text;
toml.get "0.0.0" t "package" "version"
toml.sections t                        // ["", "package", "dependencies"]
toml.entries t "package"               // [["name", "demo"], ["ver", 2]]
```

> **Order is the map's, not the file's.** `sections` and `entries` return
> whatever order the underlying map holds, which is not the order the file
> wrote them in. Read a manifest by name; do not depend on the sequence.

#### Parser internals

The same `[value, rest]` shape as `std.json`, over a list of characters.

| Name | Description |
|------|-------------|
| `is_ws c` · `is_digit c` · `is_bare c` | Character classes; a bare key is `[A-Za-z0-9_-]`. |
| `skip_ws cs` | Spaces and tabs — **not** newlines, which are significant here. |
| `skip_blank cs` | Whitespace, comments and line breaks: everything between one item and the next. |
| `drop_line cs` / `end_of_line cs` | Discarding a comment, and checking nothing but a comment follows a value. |
| `step` · `step_value` · `step_rest` · `starts` · `drop_n` | As in `std.json`. |
| `quoted_body cs acc` / `literal_body cs acc` | The body of a `"…"` and of a `'…'` string. |
| `number_chars` / `parse_number` | Numbers, with `_` accepted as a digit separator (`1_000` is `1000`). |
| `parse_key cs` | A key: bare, or quoted when it holds characters a bare key may not. |
| `parse_header cs` | A `[section]` or `[a.b]` header, as the name between the brackets. |
| `parse_value cs` · `parse_array cs acc` · `parse_inline cs acc first` | A value, an `[…]` array, and a `{…}` inline table. |
| `parse_items cs table section current` | The top-level loop: `table` is the sections finished so far, `section` the name of the one being read, and `current` its keys. |

---

### `std.test`

```dream
import std.test;
```

The test framework. **Each case runs in its own process.** That is not
ceremony: a case that raises, or that loops forever, is isolated from the rest
of the suite, and its failure arrives at the runner as an ordinary value
through `join!` rather than as something that has already unwound the runner's
own stack.

`std.test` depends only on native I/O and OS operations, avoiding cycles with the modules it tests.
A module's own tests import this framework, so anything the framework depended
on could not have tests of its own; the import would be a cycle. Walking lists
with the primitives directly is the price of letting every module test itself.

#### Cases

A case is a name and a **suspended** computation. The `$( .. )` is required —
without it the body would run where it is written, not where the runner puts it.

| Name | Description |
|------|-------------|
| `case name body` | A case: `[name, body]`, where `body` is a thunk. |
| `case_name c` / `case_body c` | The two halves back out. |

#### Assertions

Each raises a message describing the difference. Raising is what ends a case at
its first failure, and what the runner catches.

| Name | Description |
|------|-------------|
| `eq! expected actual` | Equal by `==`. The most-used one. |
| `ne! unexpected actual` | Not equal. |
| `true! actual` / `false! actual` | Exactly `true` / exactly `false` — not merely truthy. |
| `near! tolerance expected actual` | Within `tolerance`, for floats that will not compare exactly. |
| `contains! x xs` | The list `xs` has an element equal to `x`. |
| `empty! xs` | The list is `[]`. |
| `raises! body` | The **thunk** `body` must raise. `test.raises! $( 1 / 0 )` — the `$( )` is what defers it, and the body runs in its own process, so an error that would kill the case cannot. |
| `fail! message` | Fail unconditionally, for a branch that should be unreachable. |
| `has_element x xs` | The predicate behind `contains!`. Returns a `bool` rather than raising, so it is usable in a condition. |

Note the argument order: **expected first, actual second**, so
`test.eq! 6 (list.sum [1, 2, 3])` reads as the claim being made.

#### Running

| Name | Description |
|------|-------------|
| `run_cases! passed failed cases` | Runs each case, printing a line per result. Returns `[passed, failed]`. |
| `suite! name cases` | Prints a heading, runs the cases, prints the tally. Returns the **number of failures**, so several suites can be summed. |
| `run_suites! failed suites` | Runs `[name, cases]` entries, returning the total failures, seeded with `failed`. |
| `main_of! suites` | The usual entry point: run everything, print a summary, and `raise! :tests_failed` if anything failed — which is what makes the process exit non-zero. |

```dream
let main! = test.main_of! [
    ["arith", [
        test.case "adds"    $( test.eq! 4 (2 + 2) ),
        test.case "divides" $( test.raises! $( 1 / 0 ) ),
    ]],
];
```

```
arith
  ok    adds
  ok    divides
  2 passed

all tests passed
```

A failing case prints the error beside its name and the run ends non-zero:

```
  FAIL  a failure on purpose
          <error :error expected 1, got 2>
  7 passed, 2 FAILED
```

#### The generated runner

Writing `main!` by hand as above is the explicit form. The usual way is to let
the compiler build it: a module opts in by defining a **parameterless `tests`
binding**, conventionally inside a `when test { .. }` so it costs nothing in an
ordinary build.

```dream
when test {
    import std.test;

    let tests = [
        test.case "sum" $( test.eq! 6 (list.sum [1, 2, 3]) ),
    ];
}
```

`dreams FILE --test` then scans **the modules it actually loaded** for that
binding and generates an entry point that runs every suite it found. There is
no registry to keep in step and no test that is silently never run; a program
with no `tests` anywhere still compiles, and reports that there was nothing to
run.

---

### `std.map`

```dream
import std.map;
```

Maps, and sets written as maps. Builtins provide the operations that have to
be primitive; this is the grain most code actually wants — "the value there, or
this one, updated" is one call rather than three and a conditional.

A map is a value: `put` answers a new map and leaves the old one alone, and the
trie shares every branch the change did not touch. **Order is not part of a
map** — `pairs`, `keys` and `values` answer in whatever order the trie walks,
which depends on the hashes and not on the program. `sorted_keys` is the only
thing here that imposes one, and it costs a sort every time.

| Name | Description |
|------|-------------|
| `empty` | The empty map. |
| `from_pairs ps` | A map from `[key, value]` pairs. A later pair wins. |
| `merge a b` | Both, with `b` winning where they overlap. |
| `get default m k` · `has m k` | Look up, with a default; membership. |
| `pairs m` · `keys m` · `values m` · `size m` · `is_empty m` | What is in it. |
| `sorted_keys m` | The keys in order, for output that has to be stable. |
| `put m k v` · `remove m k` | A new map with that key set or gone. |
| `update default f m k` | Apply `f` to what is there, or to `default`. Counting is `update 0 (fn n -> n + 1)`. |
| `put_strict k v m` · `update_strict default f k m` | `put` and `update` that force the value before storing it, so it does not hold the frame that computed it -- or the map, when the frame held that. |
| `put_new m k v` | Add only if absent, so a fold keeps the first. |
| `map_values f m` · `filter keep m` · `fold f init m` | Over the pairs. `fold` sees `acc k v`. |
| `without m other` | Every key of `m` that `other` does not have. |
| `set_of xs` · `member s x` · `add s x` · `members s` | A set is a map whose values say nothing. |

### `std.result`

```dream
import std.result;
```

`Result.ok value` and `Result.error reason` -- the prelude's `Result`, which
every module has without an import -- and what to do with them. It is how the
library reports a failure -- `json.parse`, `cli.parse`, `toml.parse` -- for one
reason: **raising is an effect**, so a pure parser cannot report a failure by
raising and has to answer one instead.

| Name | Description |
|------|-------------|
| `ok v` · `error r` | Make one. |
| `is_ok r` · `is_err r` | Which it is. |
| `unwrap_or default r` · `unwrap_or_else f r` | The value, or a fallback for an error. |
| `unwrap! r` | The value, raising the reason. Impure, because raising is. |
| `map f r` · `map_err f r` | Change one side, leave the other. |
| `and_then f r` | The next step, which may itself fail. This is what makes a pipeline read as one. |
| `or_else f r` · `otherwise other r` | Recover with a callback receiving the reason, or use a fallback result. |
| `all rs` | `Result.ok values` when every one is ok, or the first error — and it stops there, so a lazy list of a thousand checks costs one when the first is wrong. |
| `and other self` · `or other self` | Continue with `other` on success, or use it as a fallback on failure. An unused `other` stays lazy. |
| `oks rs` · `errors rs` | The values, or the reasons, of the ones that were. |
| `flatten r` · `to_option r` | Unnest a result, or turn its success into an option. |

### `std.error`

```dream
import std.error;
```

The **kind** and the **payload** of a failure. `try! .. catch e` binds the error
itself, and before the `_error_kind` and `_error_payload` primitives there was no
way into one: a caught error could be printed and nothing else. With them a
failure is an ordinary value to `match` on, and a program can raise failures as
distinguishable as the runtime's own.

```dream
match error.kind e {
    :divide_by_zero => ..,
    :not_found      => ..,
    _               => raise! e,        // not ours; pass it on
}
```

| Name | Description |
|------|-------------|
| `kind e` · `payload e` | The atom and the value, or `()` for anything that is not an error — so a `match` needs no type test first. |
| `is_error e` · `is_kind k e` · `is_any kinds e` | The questions a `catch` is usually asking. |
| `message e` | The payload as text; a string payload is used as it is, so a message does not gain quotes. |
| `describe e` | `kind: message`, the line to print when there is nothing better to say. |
| `new k p` | An error as a value, without raising it — how a pure function *returns* a typed failure. |
| `raise_as! k p` | Raise one of a named kind. The typed form of `raise!`, which otherwise wraps everything in `:error`. |
| `rethrow! e` | Re-raise unchanged, which is what keeps the original kind readable. |
| `trace! e` · `trace_lines! e` | Where it was first raised: the functions running, innermost first, and the ones that made each value being forced -- in a lazy program usually the line that matters, since a value fails where it is needed and not where it was written. The lines are what an uncaught error is printed with, and what `std.test` prints under a case that failed. A place is a function's, not a line within it. |

### `std.lazy`

| Name | What it does |
|------|-------------|
| `settled v` | `true`, once `v` has been forced all the way down: `strict!` for a pure program, which cannot say `!`. Asked in a condition (`if lazy.settled v { .. }`), because an argument is not evaluated and a call that hands the value back would be suspended with it. For a value about to be kept in a persistent table, where a suspension would hold every version of the table it was made from. |
| `forced v` | `true`, once `v` has been forced to its outermost constructor and no further. For a value carried through a pure loop -- a table threaded through every step -- so each step is made rather than suspended on the one before, without walking what the table holds. |

### `std.proc`

```dream
import std.proc;
```

The shapes the process builtins leave to the caller. Three properties of the
runtime decide what is here: **a thunk runs once** (so anything restartable
takes a function of unit, not a thunk), **spawning is lazy** (so anything
starting more than one process forces the list, and that `strict!` is the
difference between parallel and sequential), and **`recv!` takes the next
message, whatever it is** -- selective receive, timeouts, monitors and kill are
built here on the `std.vm` natives below it.

| Name | Description |
|------|-------------|
| `start_all! thunks` | Start every one *now* and answer the processes. |
| `start_each! start! xs` | Start `start! x` for each `x`, in parallel. |
| `wait_all! ps` | Join every one, in list order. |
| `outcome! p` · `outcomes! ps` | `Result.ok v` or `Result.error e` — what `join!` would raise, as a value. |
| `parallel! thunks` | Run these at once and answer their values in order. |
| `map! f xs` · `try_map! f xs` | `list.map` with one process per element; the second keeps failures as values. |
| `call! target body` · `reply! m value` | Request and reply. The request carries the process to answer and a reference (`vm.unique!`); the reply is `[:reply, ref, value]`, picked out of the mailbox with everything else left in place. The target is monitored for the call, so one that dies first raises an error of kind `:down` whose payload is its outcome. |
| `call_within! target body ms` | The same, giving up after `ms` milliseconds: `Result.ok reply` or `Result.error :timeout`. A reply that comes later stays in the mailbox and is never taken for another call's answer. `std.server` has the same `call_within!`. |
| `recv_where! wanted` | The first message `wanted` accepts, taken out of the mailbox; the rest stay where they were, in order. Waits as long as it takes. |
| `recv_within! ms` · `recv_where_within! wanted ms` | The same with a limit: `Result.ok message`, or `Result.error :timeout`. |
| `recv_until! wanted deadline` | What the three above are: `deadline` is a reading of `vm.now_ns!` (`deadline_in! ms` makes one), `forever` waits for ever, and `0` only looks. |
| `sleep! ms` | Wait. Messages that arrive meanwhile wait too. |
| `monitor! p` · `demonitor! p` · `is_down_of p m` | Be sent `[:down, p, outcome]` when `p` ends (at once if it has), `outcome` as `outcome!` answers it; a failure reported this way counts as handled. `demonitor!` also takes back a report already sent. |
| `kill! p reason` | End `p` as a failure of kind `:killed`, payload `reason`. It cannot be caught, and takes effect at the start of `p`'s next slice. For the process that will not stop when asked. |
| `is_call m` · `call_from m` · `call_body m` | Reading a request. |
| `serve! state handle!` | A message loop. `handle!` answers `[:go, state]` or `[:stop, value]`. |
| `shutdown` · `is_shutdown m` | The conventional stop message, so a worker and its supervisor need not agree on a spelling. |

A process parked on `recv!` is a process the runtime is still waiting for: a
program that ends with one of those reports "every process is waiting for a
message that cannot arrive" rather than its answer. **Whoever starts a server
owns stopping it.**

### `std.supervisor`

```dream
import std.supervisor;
```

Children started together and restarted when they die. Failure here is local —
a process that raises takes itself down and nothing else — so a program made of
processes needs someone whose job is noticing.

```dream
let worker! () = proc.serve! 0 handle!;

let main! = {
    let sup = supervisor.start! :one_for_one [supervisor.child "worker" worker!];
    ..
    supervisor.stop! sup
};
```

| Name | Description |
|------|-------------|
| `child name start` · `child_with name start restart` | A child: a name, a **function of unit** to start it, and when to restart it. A thunk would not do — it runs once. |
| `start! strategy children` · `start_with! strategy children limits` | Start a supervisor and its children; answers its process. |
| `limits n window_ms` · `default_limits` | More than `n` restarts inside the window and the supervisor gives up. Default: five in five seconds. |
| `which! sup` | The names running, in order. |
| `restart! sup name` | Start a child that is not running. |
| `notify! sup` | Hear about every child that goes down from now on, as `[:down, name, outcome]`. |
| `stop! sup` | Stop every child, then the supervisor. |

Restart policies: `:permanent` (always), `:transient` (only after a failure),
`:temporary` (never). Strategies: `:one_for_one` (just that child),
`:one_for_all` (stop the rest and start them all again), `:rest_for_one` (it and
everything started after it).

How it watches and stops its children:

- **Each child is monitored** (`proc.monitor!`), so a death arrives as
  `[:down, pid, outcome]` in the one mailbox the supervisor reads. A report for
  a process it no longer counts as running -- one it stopped itself, or has
  already restarted -- is ignored, so one death never restarts a child twice.
- **A child is asked to stop, then killed.** The supervisor sends
  `proc.shutdown` and waits for the child to go down for the limits'
  `:shutdown` milliseconds (`with_shutdown ms limits`; five seconds by
  default), then kills it with reason `:shutdown`. Anything meant to be
  supervised should still be written around `proc.serve!`, so that being asked
  is enough.

---

### `std.sql`

```dream
import std.sql;
import std.sql.sqlite;
```

SQL for whichever database is on the other end. A statement is built as a
value and rendered only when the dialect is known: the same query is `$1` and
`"name"` for Postgres, `?` and `` `name` `` for MySQL. Dialects are maps, so a
database not listed here is `sql.extend` away. The module's head comment is the
design.

```dream
let db = sqlite.open! "notes.db";
sql.query! db (sql.select ["name", sql.alias (sql.count "id") "n"]
    |> sql.from "users" |> sql.where (sql.gt "age" 18)
    |> sql.group_by ["name"] |> sql.order_by [sql.desc "n"] |> sql.limit 10)
sql.query! db (sql.q "select * from users where id = ? and ?" [7, sql.eq "active" true])
```

A string means one of three things, depending on where it is written: SQL text
(`raw`, `concat`, a whole statement), a column (the left of a comparison, a
select list, `order_by`), or a value (the right of a comparison, a hole in `q`,
an argument to `call`). A fragment is a fragment in all three places, and a
query used inside another is a parenthesized subquery.

| Name | Description |
|------|-------------|
| `raw` · `param` · `literal` · `ident` · `col` | The pieces: text, a bound value, a value written into the text, a quoted name, a qualified name (`"t.c"`). |
| `expand sql.query "SELECT .. {name} .."` · `expand sql.fragment` · `expand sql.script` | Templates checked at compile time. `{name}` binds a value, `{..name}` splices SQL. Plain literal statements are checked too; the shared lexer is `std.sql.lint`. |
| `positional text values` | Turn `$1`-style SQL into a composable fragment, renumbered for the target dialect. |
| `q text args` | SQL with `?` holes, each filled by a value (as a parameter) or a fragment/query (as SQL). `??` is a literal `?`. |
| `concat` · `join sep` · `parens` | Put fragments together. |
| `per_dialect f` | A fragment written differently for each dialect. |
| `eq` · `ne` · `lt` · `le` · `gt` · `ge` · `like` · `ilike` · `op sym` | Comparisons; `eq c ()` is `IS NULL`. |
| `in_list` · `not_in` · `in_query` · `between` · `exists` · `is_null` · `is_not_null` | More conditions. An empty `IN` is false. |
| `all_of` · `any_of` · `negate` | Combine conditions, each parenthesized. |
| `call` · `count` · `count_all` · `sum` · `avg` · `min` · `max` · `alias` · `asc` · `desc` · `case_when` | Expressions. |
| `select` · `from` · `join_on` · `left_join` · `right_join` · `full_join` · `cross_join` · `where` · `group_by` · `having` · `order_by` · `limit` · `offset` · `distinct` · `with_cte` · `with_recursive` | Queries, chained with `\|>`. |
| `union_all` · `union_distinct` · `intersect` · `except` | Compound queries. |
| `insert_into` · `values` · `values_list` · `values_in` · `insert_select` · `on_conflict_nothing` · `on_conflict_update` · `excluded` · `returning` · `update` · `set` · `delete_from` | Writing. `values` takes a map, and atom keys are allowed, so a record's `to_map` can be inserted as it is. |
| `create_table` · `column` · `not_null` · `unique` · `primary_key` · `auto_increment` · `default` · `references` · `check` · `primary_key_on` · `unique_on` · `foreign_key` · `check_that` · `drop_table` · `create_index` · `unique_index` · `drop_index` · `add_column` · `if_not_exists` · `if_exists` | Schema. A type is an atom the dialect knows (`:integer`, `:text`, `:bool`, ..), `[:varchar, 80]`, or a string written as given. |
| `ansi` · `sqlite` · `postgres` · `mysql` · `mssql` · `extend` | Dialects. |
| `render d x` · `inline d x` | `%{ :text, :params, :problems }` for a dialect, or the text with values written in. |
| `exec! c x` · `query! c x` · `run! c x` · `first!` · `one!` · `scalar!` · `column!` · `query_as! description` | Run a statement: its effect, rows as maps by column name, the raw outcome, or one piece of it. |
| `exec_many! c xs` | Many statements; a run of the same text is prepared once. |
| `script! c text` · `transaction! c f` · `using! c f` · `close! c` · `traced f c` | Scripts, transactions (nested ones are savepoints), scoped connections, and a hook that sees each statement. |
| `transaction_with! c options f` · `retrying! c attempts options f` · `serializable! c attempts f` | Isolation and read-only options, with bounded retries for serialization failures, deadlocks and lock contention. Retried callbacks may run more than once. |
| `fold! c x batch f init` · `each! c x batch f` | Stream rows through a fold or effectful callback. SQLite steps natively; PostgreSQL reads portal batches. |
| `migrate! c migrations` | Apply `[id, name, statements]` migrations the database has not seen, each in its own transaction. |
| `fail!` · `fail_with!` · `is_sql_error` · `error_code` · `error_field` · `is_retryable` · `is_connection_lost` | A database's refusal: kind `:sql_error`, with a code that is `:unique`, `:foreign_key`, `:not_null`, `:check`, `:constraint`, `:busy`, ... wherever the driver can say so. |
| `connection d state effects` · `no_rows` | For writing a driver: the effects are `:run`, `:close`, and optionally `:run_many`, `:script` and `:fold`. |

Rows use atom keys for column names already present as atoms in the program,
and string keys otherwise; `mapping` records can read them directly. In an
insert, `()` asks for the column default; `sql.null` requests an explicit NULL.
Other parameter positions still bind `()` as NULL. `sql.blob bytes` distinguishes
binary data from text. PostgreSQL additionally accepts lists as arrays, maps as
JSON, and atoms as their names; each dialect validates parameter kinds.

`std.sql.pg.driver` connects PostgreSQL to this API: `driver.open! target` takes a
`std.sql.pg.config` map, URL or keyword string, and `driver.wrap! conn` adapts an existing
`std.sql.pg.db` connection. It preserves SQLSTATE as `:state` and the original error as
`:cause`, returns `bytea` as blobs, streams through portals, and prepares each
run of equal SQL once for `exec_many!`. Use `RETURNING` for generated keys;
PostgreSQL outcomes have `:last_id` set to `()`. See [pg](https://github.com/dream-mindv2/std/blob/main/src/sql/pg/README.md).

`std.sql.sqlite` is the SQLite driver. It binds the system's
libsqlite3 through `std.foreign` as `from system "sqlite3" "0"`, which searches
the loader's path and the directories package managers use; where it is
somewhere else, `$DREAM_LIB_SQLITE3` names the file and `$DREAM_LIBRARY_PATH`
adds a directory.
`sqlite.open! path`, `sqlite.memory! ()`, and `sqlite.open_with! path
%{ :readonly, :busy_timeout, :foreign_keys, :place }`, where `:place` may be a
`foreign.start!` server to own the database. `sqlite.available! ()` says
whether the library can be loaded at all.

### `std.all`

```dream
import std.all;
```

Every module in the standard library, imported in one place. Two things use it.

`dreams mind/std/src/all.dr --test` builds a runner from the `tests` each of those
modules exports, so **adding a module here is all it takes for its tests to
run**. Importing them is also a check in itself: a module that no longer
compiles fails the build rather than being quietly skipped.

| Name | Description |
|------|-------------|
| `version` | The library's own version as a string, so a program can report what it was built against. Currently `"0.1.0"`. |
| `modules` | The module names this build provides, as a list of strings. |

```dream
all.version     // "0.1.0"
all.modules     // ["std.array", "std.cli", "std.error", "std.json",
                //  "std.list", "std.map", "std.proc", "std.result", "std.seq",
                //  "std.str", "std.streams", "std.supervisor", "std.test",
                //  "std.toml"]
```

`modules` is a hand-maintained list, so it names what the library intends to
ship rather than what happens to be on disk — a module that exists but is not
listed here is not part of the library's interface.

---

## Runtime error atoms

These atoms are raised by the built-in operations. A `match` on the error kind of a caught exception can distinguish them.

| Atom | Raised by |
|------|-----------|
| `:divide_by_zero` | Integer `/` or `%` with a zero divisor |
| `:type_error` | Wrong type passed to an operation |
| `:not_a_function` | Applying a non-callable value |
| `:no_such_member` | `module.name` where `name` is not exported |
| `:match_error` | A `match` with no arm that matched, or a destructuring the value does not fit. The payload is `[value, path, line, col]` |
| `:killed` | `vm.kill!`: the process was stopped from outside. The payload is the reason |
| `:down` | `proc.call!` and `server.call!`: the process called ended before it answered. The payload is its outcome |
| `:out_of_bounds` | Array index outside the valid range |
| `:shape_error` | Tensors whose shapes do not fit the operation |
| `:device_error` | A host tensor and a GPU tensor in one operation |
| `:no_gpu` | `tensor.gpu` where no GPU can be used |
| `:gpu_error` | The GPU refused an operation (out of device memory, say) |
| `:loop` | A thunk that depends on itself |
| `:stack_overflow` | Recursion exceeded the process's stack limit |
| `:out_of_memory` | Process heap exceeded its allocation limit |
| `:killed` | Process was killed externally |
| `:timeout` | Process timed out |
| `:not_found` | File or path does not exist |
| `:io_closed` | Operation on a closed or stale handle |
| `:io_error` | Generic I/O failure |
| `:permission_denied` | Filesystem permission denied |
| `:already_exists` | Tried to create something that already exists |
| `:wrong_kind` | Operation on the wrong kind of handle |
| `:connection_refused` | TCP connect failed |
| `:connection_lost` | Connection closed unexpectedly |
| `:address_in_use` | Port is already bound |
| `:os_error` | Unclassified OS error |
