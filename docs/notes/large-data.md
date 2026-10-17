# Large data in an image

The payload: data past the 4 GiB line, reached as views rather than copies. Moved out of `CLAUDE.md` on 2026-09-25; the prose is as it was, so a date or number in it is as of when it was written.

An image addresses everything with `u32` — a section offset, a `KSTR` record,
`StrObj::len` — so a `.dream` file and every string in it stop at 4 GiB. That is
plenty for a program and nothing for the datum a program is *about*. The payload
is the one region past that line:

```
dreams --payload FILE [--payload FILE ...] -o out.dream main.dr
```

Each file becomes one entry of the `LDAT` table, in the order given, and the
bytes go in `PAYL` at the very end of the file. A program reaches them with
`core.data_count ()` and `core.data_at i`.

What comes back is **not a string**. It is a `bigstr`: a length and a pointer
into the mapped image, built in constant time and copied by nobody — not on
materialization, not by the collector, not by `spawn!`, which all share the view
because the bytes belong to the runtime's image and outlive every process in it.
So `len`, `str.byte`, `str.slice` (another view), `==`, the ordering operators,
use as a map key, and `io.write!` all work without a copy; `+`, `to_string`,
`str.concat_all`, `str.find` and `str.chars` refuse, because each would have to
build a `StrObj` and so would undo both halves of the point. `type_of` says
`:bigstr` rather than `:string` precisely so that no code path written for one
is handed the other.

A `bigstr` and a `string` of the same bytes are `==` and hash alike, which is
what makes `str.slice 0 4 data == "%PDF"` mean what it looks like.

Two consequences worth knowing before changing any of it. The compiler never
holds a payload — a Dream string is capped at the same 4 GiB — so `build!`
measures each file with `io.size!`, lays the table out from the sizes, writes
the ordinary image, and only then streams the files through a bounded buffer
(`copy_payload!` in [dreams/src/main.dr](https://github.com/dream-mindv2/dreams/blob/main/src/main.dr)). And a build with no
`--payload` emits neither section, so it writes the bytes it wrote before any of
this existed — which is what the bootstrap compares.
