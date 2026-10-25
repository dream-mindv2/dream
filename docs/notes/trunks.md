# Trunks

`std.trunk`: one container whose representation the VM chooses. Begun
2026-10-18. The code is `TrunkObj`/`TrunkStoreObj` in `src/value.hpp`,
`src/trunk.cpp`, and `mind/std/src/trunk.dr`.

## Why one container

Every program in this repository that turned out quadratic did so by
picking the wrong container: a list searched more often than it was built
(found five times), `xs + [x]` in an accumulator, an association list where a
map belonged. The supercompiler can recognise some of those shapes in list
code and rewrite them (`dreams/src/accum.dr`), but recognising is guessing
at what the author meant from what they wrote, and every shape needs its own
proof that the rewrite is exact.

A trunk turns that around. The program says what it does -- push,
prepend, read a position, search, look a key up, put one -- and the meaning
never changes: a trunk is a finite sequence, and every operation answers
what it would of a list of the same elements. How the elements are kept is
the VM's, chosen from what it sees done, and every representation is a cache
of the sequence rather than a different meaning, so none needs proving.

## The representation

A store of slots, addressed by positions that never change. A new
trunk's first element is at `TRUNK_ORIGIN`, the middle of the position
space, and a push writes the next slot above, a prepend the next below, so
one store serves a stack, a queue, an array and a buffer. Reading a position
is a subtraction and a load.

Trunks are values, and the old version must still read as it was after
a new one is made from it. The store is shared by every version made from
another, with exactly one version, the **root**, reading the slots directly.
When the root makes a new version -- a push, a set, a `rest` -- the new one
becomes the root and the old becomes a version "like the next one, except
that slot `pos` holds `value`" (or just with other bounds). That is Baker's
version tree, the scheme behind Haskell's diff arrays:

- Used linearly, every change is made in place. Each version is used once to
  make the next, which is how a loop threads an accumulator, so the VM never
  has to be told which uses are linear.
- An old version still read reads through its chain, exactly what it held.
  When the chain is nine long, the old version gets a store of its own
  (`reroot_if_deep`), so a version kept and read for ever costs one copy, not
  a walk per read.
- A write from a version that is not the root is a branch, and copies.
- An old version's diff lives only as long as the old version: chains point
  from old to new, so nothing keeps a dead version's values.

A store in the shared area is never written; a version crossing to another
process crosses as its elements, in a store of their own, without the other
versions' slots or the indexes.

## Indexes, and forcing nothing extra

`contains`, `index_of` and `occurrences` search for an element;
`lookup` and `put` for a key of an association list, the first item of
an element that is a list. Each kind of search has an index: flat key to the
positions holding it. Two rules make them exact:

- An index covers only a run of positions a search already evaluated. A
  search scans what lies before that run, asks the index about the run, and
  scans what lies after it, filing elements as the scan passes them. So it
  evaluates exactly the elements a scan of the sequence would -- in the same
  order, stopping at the same one -- and a list holding `1 / 0` after the
  match is searched as safely as the list itself.
- A key is what `==` agrees with. Numbers equal across representations, so
  an integral float that fits a fixnum is filed as that fixnum and NaN under
  nothing. A big integer, or a float that only a big integer could equal, is
  not filed, and a search for one scans.

An index is made once searching has cost as much as the trunk is long,
since one search is cheaper as a scan and the second pays for the index. It
describes the root's slots; a write by the root keeps it true (a value
written in that is not yet evaluated shortens the run, since filing it would
evaluate it), and a search of an old version scans.

## What it bought

The naive programs, list against trunk, at n and 2n, milliseconds:

| program | list | trunk |
|---|---|---|
| dedupe: `contains`, then push | 388 → 1475 | 21 → 36 |
| word counts: `lookup`, then `put` | out of memory | 34 → 70 |
| a queue kept n/2 long: push, `rest` | 1856 → 6506 | 21 → 35 |
| reads by position | 257 → 896 | 18 → 27 |

The first version of the word counts was 244 → 832 ms, 95% of it in minor
trunks. The store is old and written once per turn, and the remembered
set kept an entry per write, so each minor trunk scanned the store once
per write since the last. `Heap::remember_once` keeps one entry per store per
trunk.

## A queue kept everything it was given

Found by sleep, whose scrollback is a trunk pushed at the back and dropped
from the front: `seq 1 200000` ended in `out_of_memory` with the history
capped at 5,000 lines -- the live set grew with the output (116 MB at 50,000
lines, 605 MB at 200,000), 333,000 row stores alive for 5,500 rows kept.

A `rest`, `drop` or `take` of the root moves the root and narrows what it
reads, and nothing more: the store's written range, `[lo, hi)`, stays as it
was, since an older version may still read the slots outside the new bounds.
`grow_to` carries that whole range into the bigger store. So a queue's store
held every slot ever pushed, the collector marked every one, and a push past
the end copied them all again. A slice of the root now takes a store of its
own (`copy_out`) once the slots outside it outnumber those in it; the old
version keeps the old store and reads as it did, and the store dies with it.
The copy is the slice's length and comes only after as many slots were let
go, so a queue's push and `rest` stay constant time, amortized. The same
emulator then holds 87 MB at 200,000 lines, flat in the length of the run,
and runs it in 3.8 s instead of 6.6.

The same workload showed the other cost: `concat` advanced the root once per
element, and every advance is a version object, because a version records
the one slot it overwrote. Past the root's end, where no version has written
(`end >= hi`), there is nothing to record, so the store grows once and the
elements go in behind one version. Writing a line of terminal output had
been a hundred versions; allocation for the run fell from 8.2 GB to 5.5 GB.

## What is not done

- The JIT compiles a function calling a search or `put`, but calls them
  through `dream_rt_apply`, the generic path, rather than as a native it
  knows: they may evaluate elements, and a native called directly from
  compiled code pins the heap while it does (`NativeDef::vouches`), where
  the generic path lets a collection run. Compiled, a queue went 274 ms to
  112, positional reads 171 to 66 and dedupe 316 to 267; word counts went
  550 to 626, the generic call costing more than the interpreter's direct
  one. A search that called directly and handed an unevaluated element back
  to be forced outside the native would have neither cost.
- A trunk does not cross between images, and cannot be ordered by
  `compare`. It does cross `std.wire` (tag 14, its elements), so the unit
  cache can keep one, and `std.codec`'s formats write it as a list.
- `c.[i => v]` is not yet `set`.
- The supercompiler does nothing with trunks yet. What it can add is
  the second half of the design: where it can see every use of a trunk,
  choose the representation when it is made, and drop the versions a linear
  use never reads.

## Where the library uses them

A queue that grows under load: joined at the back with `xs + [x]`, which
copies the queue per caller, and served from the front.

- `std.sql.pool` and `std.sql.pg.pool`: the callers waiting for a
  connection.
- `std.http.h2`: a stream's data waiting for its flow-control window. A
  piece sent in part is replaced in place (`trunk.set 0`).

The compiler has none: `dreams/tests/scale.py` is linear on both its axes,
and the lists it still searches are short.
