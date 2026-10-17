# Wrapping a C library

Dream reaches C through two modules and one declaration. `std.ffi` is the VM's
half: it binds a symbol to a signature, owns what C hands back, and checks
every pointer that crosses. A `foreign` declaration is how a program says what
it calls, and `std.foreign` is the rest of what a wrapper is written against.
This page is about the last two, and [builtins.md](builtins.md#stdffi) is the
reference for the first.

## A first binding

```dream
import std.console;

foreign libm from "libm.so.6" {
    sqrt : :double -> :double
}

foreign libc {
    puts! : :cstr -> :int
}

let main! = libc.puts! (to_string (libm.sqrt 2.0));
```

Each line is a C function, written as its C signature and bound to a name.
The block becomes a module, so `libm.sqrt` is an ordinary Dream function, and
every tool that works on Dream works on it. When the C name is not the Dream
name without its `!`, say it: `add! : Counter -> :long -> :long = counter_add`.

Purity is the name's, as it is everywhere in Dream: a function bound to a `!`
name is an effect, and one that is not may be called from pure code. Only the
program can know whether a C function is pure. Leave the `!` off one that is
not, and it will be run lazily, perhaps twice, perhaps never.

Nothing is loaded until a name is first used, and a library is opened once per
VM however many processes use it. `from` is a path the platform's loader
understands, `system "name"` for a library installed on the system (below),
`embedded "name"` for a library the image carries (further below), or an
expression in parentheses; without it, the library is the running program,
which is the C library on any normal system.
An expression is compiled inside the module the block makes, where only
`std.ffi` and `std.foreign` are in scope.

## Finding a library on the system

```dream
foreign sqlite3 from system "sqlite3" "0" {
    libversion : :void -> :cstr = sqlite3_libversion
}
```

A library installed on the system is a different file on every platform, and in
a different place on every distribution. It is `libsqlite3.so.0` where the
loader looks on Debian, under Homebrew's prefix on macOS (which the loader
never searches), and in a store path on NixOS. So `from system "sqlite3"` names
the library rather than a file, and the VM searches for it the first time it is
used:

1. **`$DREAM_LIB_SQLITE3`**, the file itself: `DREAM_LIB_` and the name in
   capitals, anything but a letter or digit made `_`. Whoever sets it is obeyed
   exactly. If that file cannot be opened, that is the error, and nothing else
   is tried.
2. **`$DREAM_LIBRARY_PATH`**, a list of directories, joined as `PATH` is; then
   each installation directory's `lib` (`std.install`).
3. **The platform's loader**, by each exact name: rpath, `ld.so.cache`,
   `LD_LIBRARY_PATH` and whatever else the system already knows.
4. **The directories package managers use**: `/usr/local/lib`, Debian's
   multiarch directories, `/usr/lib64`, `/usr/lib`, Linuxbrew; on macOS
   `/opt/homebrew/lib`, `/usr/local/lib` and MacPorts; on both, the Nix
   profiles.

The names it looks for are the platform's: `libsqlite3.so` then any
`libsqlite3.so.N`, newest first; `libsqlite3.dylib` and `libsqlite3.N.dylib`;
`sqlite3.dll`, `libsqlite3.dll` and MinGW's `libsqlite3-N.dll`. A second string
holds the library to one ABI version, the number in its soname:
`system "sqlite3" "0"` is `libsqlite3.so.0` or `libsqlite3.so.0.*`, and never a
`libsqlite3.so.1`, which would load and then not mean the same thing.

When nothing is found, the error says what was looked for, where, and what to
set:

```
cannot find the C library `sqlite3` (as libsqlite3.so.0, libsqlite3.so.0.*)
  the system's loader: libsqlite3.so.0: cannot open shared object file: No such file or directory
  nothing in: /usr/local/lib, /usr/lib/x86_64-linux-gnu, /usr/lib64, /usr/lib, ...
Set $DREAM_LIB_SQLITE3 to the library's file, or add its directory to $DREAM_LIBRARY_PATH.
```

The same search is a value outside a declaration: `foreign.system "sqlite3"`,
`foreign.system_version "sqlite3" "0"`. `foreign.any_of [foreign.embedded
"sqlite", foreign.system "sqlite3"]` is the first of several that opens, so a
program can carry its own copy for the machines that lack one. `foreign.locate!
lib` answers `Result.ok file` or `Result.error why` without raising, for a program or
tool that wants to report where a library came from. The search is described
in Dream (`foreign.library_files`, `foreign.library_dirs`) and carried out by
the VM ("Finding a library" in dream/src/ffi.cpp).

## Signatures are types

A C signature is written in the type grammar, and means two things at once:
what `std.ffi` passes, and the Dream type of the function on this side. So the
checker knows `add! : Counter -> :long -> :long` as `Counter -> :integer ->
:integer`:

```
error: argument 1 of `sample.add!` should be `Counter`, but this is `Cursor`
error: argument 2 of `sample.add!` should be `:integer`, but this is `"one"`
```

| Written | In C | On this side |
|---|---|---|
| `:int`, `:long`, `:size`, `:i32`, `:u8`, .. | that number | `:integer` |
| `:f32`, `:f64`, `:double` | that float | `:float` |
| `:bool` | `bool` | `:bool` |
| `:cstr` | `const char *`, copied | `:string` |
| `:bytes` | `const char *` with no promise of a terminator; also a payload view | a string |
| `:ptr` | a raw address, the escape hatch | an integer, or `()` for `NULL` |
| `:void -> t` | a function of no arguments | `:unit -> t` |
| a `resource` `R` | `R *` passed in | the handle type `R` |
| `R` as a result | `R *` handed over, freed by `R`'s destructor | `R` |
| `borrow R` | `R *` C keeps | `R` |
| `out t` | `t *` C writes through | leaves the arguments, and joins the result: `[result, out, ..]` |
| `taken free` | a `char *` the caller must free, with `free` | `:string` |
| `(a -> b)` | a function pointer | a pure Dream function |
| a `struct` `P`, or `:buffer` | `P *` | `foreign.Buffer` |
| `t \| ()` | a result that may be `NULL` | `t \| :unit` |

The C-named scalars (`:int`, `:long`, `:size`, ...) have the platform's widths,
settled by the VM, so a binding written from a header is right on each platform
it runs on. There is no `:float`: in Dream that is a double, and a signature is
exactly where the two readings would get confused. Write `:f32`.

A C type that means nothing is reported where it was written, before anything
runs:

```
error: there is no `:float` in C here, because in Dream that is a double: write `:f32` or `:double`
error: `Nope` is not a resource or a struct of `sample`
error: `out` is a parameter C writes through, so it is only an argument
```

## Pointers are owned

A raw pointer is the source of almost every way a C binding goes wrong: kept
after it is freed, freed twice, passed where another kind was wanted. So a
pointer C hands back is not given to Dream at all:

```dream
foreign sqlite from embedded "sqlite" {
    resource Db = sqlite3_close
    resource Stmt = sqlite3_finalize

    open! : :cstr -> out Db -> :int = sqlite3_open
    prepare! : Db -> :cstr -> :int -> out Stmt -> :ptr -> :int = sqlite3_prepare_v2
}
```

`resource Db = sqlite3_close` declares a kind of pointer and the C function
that lets one go. Where C hands one back, the answer is a handle,
`[:foreign, "sqlite.Db", 17]`, and the pointer stays in a table the calling
process owns. A handle is only a name for an entry, so:

- a handle that has been released is an `:ffi_error`, not a use-after-free;
- a `Db` parameter refuses a statement handle -- at compile time where the
  checker can see it, and at run time where it cannot;
- a handle names nothing in any process but the one that owns it;
- releasing twice is an error, not a double free.

The tag is the library's name and the resource's, so two libraries that each
declare a `Db` do not accept each other's.

**When the destructor runs.** When the program calls `foreign.release!`; or,
for whatever it did not release, when the owning process ends, whether it
returned, raised or called `os.exit!`. Resources are destroyed newest first.

**What was made from what.** A resource made by a call that took handles
records them as its parents, and releasing a parent releases its children
first. Above, `prepare!` took the database, so releasing the database finalizes
its statements before it closes, rather than leaving them pointing into freed
memory.

`foreign.using! h f` is the scoped form: `f h`, then `h` released, whether `f`
answered or raised. The answer is forced before the release, because a lazy
answer that read through `h` would otherwise be read later, from a handle that
is gone.

Out-parameters are not passed. C is given somewhere to write, and the call
answers `[result, out1, out2, ..]`, so the C idiom of a status code plus a
pointer through `**` reads as `let [status, db] = sqlite.open! "notes.db";`.

A resource with no destructor, `resource Env`, is one C only ever lends:
`borrow Env` as a result. C cannot hand one over, and saying it does is an
error.

## A library as a process

Most C libraries were not written for a program with many threads in it, and
most own things that must be let go. Both are what a process is for here:

```dream
let db_server = foreign.start! ();
let db = foreign.run! db_server (fn () -> {
    let [status, db] = sqlite.open! "notes.db";
    sqlite.exec! db "create table t (x)"
    db
});
foreign.stop! db_server        // every destructor runs
```

`run!` sends a function of unit to the server, which calls it and answers
what it answered, forced. It is a function and not a `$( .. )` because a
message is evaluated before it leaves: a thunk would run in the process that
sent it. A function travels as it stands, with what it captured. So any amount
of work with a library's resources is one message and one answer.

The server owns every resource made through it. Work reaches the library one
piece at a time, so a library with global state needs no lock. Work that raises
comes back to the caller as the caller's error, and the server carries on. When
the server stops (asked, crashed, or restarted by a supervisor, via
`foreign.child!`), everything it owned is released. Its clients hold handles
that name its resources and mean nothing anywhere else, so a handle from a
server that has been restarted raises rather than reaching freed memory.

`run!` takes either a server or `foreign.here`, so a wrapper written over a
place serves both uses:

```dream
let open_db! place path = foreign.run! place (fn () -> sqlite.open! path);

open_db! foreign.here "a.db"      // in this process
open_db! db_server "b.db"         // in the library's
```

## Memory and structs

`foreign.buffer! n` is `n` zeroed bytes this process owns, of a known size, so
every `peek!`, `poke!`, `read!` and `write!` is checked against it. A pointer C
hands back has no size, and Dream does not read one directly, because that
would mean trusting C about where it ends. A buffer is what a `:buffer` or a
struct parameter passes.

A struct is declared in the block, and laid out as C lays it out: each field
at the next offset its alignment allows, and the whole padded to its widest
alignment. The layout is settled while compiling:

```dream
foreign sample from embedded "sample" {
    struct Point { x : :i32, y : :double, tag : :u8 }
    fill! : Point -> :i32 -> :double -> :void = point_fill
}

let b = sample.Point.new! ();
sample.fill! b 3 1.5
sample.Point.read! b                        // %{ :x => 3, :y => 1.5, :tag => 7 }
sample.Point.write! b %{ :x => 10 }         // the fields it names, and no others
sample.Point.offset :y                      // 8
```

`Point.layout` is the layout itself, for `foreign.read_struct! b at layout`
over an array of them. `array_of! t xs` and `read_array! b at t n` do the same
for a C array of numbers.

## Callbacks

A parenthesised arrow passes a Dream function where C wants a function
pointer, so `qsort` and `sqlite3_exec`-style iteration work:

```dream
foreign sample from embedded "sample" {
    fold_range : :long -> (:long -> :long -> :long) -> :long -> :long
}

sample.fold_range 10 (fn acc i -> acc + i * i) 0      // 385
```

A callback runs Dream while C is on the stack, which is why it has rules of
its own. Each rule is checked rather than trusted:

- **It must be pure.** An impure one is refused before C is called. Nothing a
  callback does can be ordered against the C code around it, and a callback
  that waited on a message would have to be resumed with C's frames still on
  the stack, which cannot be done.
- **It lives for one call.** A library that keeps a callback and calls it
  later (a signal handler, a completion callback) cannot be given one. That
  later call would jump into freed memory, and this is the one rule nothing
  can detect.
- **It runs on the calling thread.** A call from another thread is refused and
  answered with zero.
- **An error does not unwind through C.** The callback answers zero from then
  on, and the error is raised once the C function has returned.

## Carrying the library in the image

```
dreams --payload sqlite=libsqlite3.so -o notes.dream main.dr
```

puts the file's bytes at the end of the image, and `foreign sqlite from embedded
"sqlite" { .. }` names them. On Linux the library is loaded straight from memory
(`memfd_create`), so nothing is written to disk. Elsewhere it is written to
the temporary directory and loaded from there, and removed at once where the
platform allows that, which is everywhere but Windows. A program that needs a
particular build of a library can carry exactly that build. The name is stored
in the image's `LNAM` section ([bytecode-format.md](bytecode-format.md)).

## Signatures built at run time

A declaration compiles to calls of `std.ffi`, and `foreign.function` and
`foreign.pure_function` are those calls, for a program whose signature is data
it computed:

```dream
let low_byte = foreign.pure_function (foreign.embedded "sample") "low_byte" [:u32] :u8;
```

The vocabulary is `std.foreign`'s `Arg` and `Return` types: the same words as
a declaration's, spelled as data -- `[:out, t]`, `[:own, tag, destructor]`,
`[:handle, tag]`, `[:callback, [args], result]`, `[:cstr, destructor]`.
`function` is signed with them, so a literal signature is still checked while
compiling; what is lost is the Dream type of the answer, which is `:any`.

## What this cannot make safe

A signature that does not match the C function corrupts the call, exactly as
it would in C, and nothing can see it. `:ptr` is a raw address and checks
nothing; it is the escape hatch, and so are `ffi.address!` and the
original `alloc!`/`read_u8!` members of `std.ffi`. Everything else is checked
at the boundary: argument kinds, handle liveness, ownership and tag, and buffer
bounds.

[dream/tests/ffi](../tests/ffi) is a C library written to be wrapped,
together with a program that exercises all of the above, under both the
interpreter and the JIT, and `system.dr`, the same library found as a system
library each way a person can point at it (`just test-ffi`).
