# Calling another image

A `.dream` image is a whole program. `std.image`, and the `image` declaration
written against it, let one program use another as a library. The other image
can be found by path, by the name an installation gives it, or carried inside
the program's own image. It is opened the first time something in it is
called, and each function in it is handed over as an ordinary Dream function.
It is to an image what [`foreign`](ffi.md) is to a C library, and it is built
the same way.

## A first binding

The library is any program. It needs no `main!`, and nothing in it knows it
will be called from elsewhere:

```dream
// shapes.dr, compiled with `dreams shapes.dr -o shapes.dream`
let area : [:atom, :float] -> :float;
let area shape = match shape {
    [:square, side] => side * side,
    [:circle, r] => 3.0 * r * r,
};

mod units {
    let cm inches = inches * 2.5;
}
```

The program that uses it declares what it calls:

```dream
import std.console;

image shapes from "shapes.dream" {
    area : [:atom, :float] -> :float
    cm : :float -> :float = shapes.units.cm
}

let main! = console.print! (to_string (shapes.area [:square, 3.0]));
```

Each line is a name, its Dream type and, when the function is not the image's
top-level `name`, `= module.member` for where it is. The block becomes a
module, so `shapes.area` is an ordinary function that every tool understands,
and the checker holds every caller to the type written for it:

```
error: argument 1 of `shapes.area` should be `[:atom, :float]`, but this is `[:square, :integer | :float]`
```

Purity is the name's, as it is everywhere: a line with `!` is bound as an
effect, one without as a pure function. A line with no arrow, `version :
:string`, is a value of the image, computed there once. A value cannot be an
effect, so an effect that takes nothing is `now! : :unit -> :integer`.

## Where an image is

| `from` | Means |
|---|---|
| `"shapes.dream"` | A path, relative to where the program runs. |
| `installed "shapes"` | The installed image `dream shapes` would run: `shapes` or `shapes.dream` in the first directory of `$MINDV2_PATH` (or `~/.mindv2`) that has one. `std.install` is the same lookup. |
| `embedded "shapes"` | An image this one carries: `dreams --payload shapes=shapes.dream`. |
| `(expression)` | Any `std.image` value. `image.any_of [image.installed "shapes", image.embedded "shapes"]` prefers the installed copy and falls back to the one it carries. |

An image that cannot be found raises an `:image_error` that says where it
looked. `image.locate! img` answers `Result.ok file` or `Result.error why` instead,
for a program that wants to check first.

## What crosses

The image runs in a runtime of its own, with its own heap, atom table and
scheduler. So what passes between the two is **data**: integers, floats,
strings, atoms, lists, arrays, maps and errors, copied across. A function or a
process names things in a heap the other side has never seen, so it cannot be
passed. A parameter declared as a function is a compile error, and one passed
anyway, through a binding made at run time, raises.

An error the image raises is raised again in the caller, with its kind and
payload, as the caller's error. A library's errors are its caller's to handle.

Each call runs to completion on the worker that made it, in a scheduler of
its own, and an image runs one call at a time. Processes can call one image
freely, and their calls take turns. A call that blocks for a long time holds
its worker while it does.

Lifting these limits, so that a closure can cross and a call can be
interleaved with the caller's processes, means merging the two images into
one index space. That is the plan in [dynamic-linking.md](dynamic-linking.md),
and this API is written so that it can arrive without the declaration
changing.

## Bindings made at run time

The declaration compiles to calls of `std.image`, which are there for a
program whose targets are data it computed:

```dream
let render! = image.function (image.installed "renderer") "render!" 2;
let pi = image.value (image.file "consts.dream") "pi";
```

`function` and `pure_function` take the image, the target and an arity. When
the binding is made, the image is opened, the target found, and the arity
checked against the function's. What is lost is the type, which is `:any`.

[dream/tests/image](../tests/image) is a library and a program that uses
it all three ways, run under both the interpreter and the JIT (`just
test-image`). "Images as libraries" in
[dream/src/builtins.cpp](../src/builtins.cpp) is the VM's half.
