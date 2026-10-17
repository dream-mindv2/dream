#!/bin/sh
# std.image: one Dream image calling another.
#
# `shapes.dr` is a library -- a program with no `main!` -- and `main.dr` uses
# it through `image` blocks, reaching the same image by path, as an installed
# image (a `$MINDV2_PATH` made here) and carried in its own payload. The
# program runs under the interpreter and the JIT, and both must print exactly
# `main.expected`. It runs in a directory of its own, because a path is
# relative to where the program runs.
set -u

dream=${dream:-build-dream/bin/dream}
dreams=${dreams:-build/dreams.dream}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
case "$dream" in /*) ;; *) dream="$root/$dream" ;; esac
case "$dreams" in /*) ;; *) dreams="$root/$dreams" ;; esac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/installed"
timeout 300 "$dream" "$dreams" -L "$root/mind" "$here/shapes.dr" -o "$tmp/shapes.dream" >"$tmp/build" 2>&1 \
    || { cat "$tmp/build"; exit 1; }
cp "$tmp/shapes.dream" "$tmp/installed/shapes.dream"
timeout 300 "$dream" "$dreams" -L "$root/mind" "$here/main.dr" --payload "shapes=$tmp/shapes.dream" \
    -o "$tmp/main.dream" >"$tmp/build" 2>&1 \
    || { cat "$tmp/build"; exit 1; }

fail=0
for tier in "" "--no-jit"; do
    # shellcheck disable=SC2086
    (cd "$tmp" && MINDV2_PATH="$tmp/installed" timeout 30 "$dream" $tier main.dream) >"$tmp/out" 2>&1
    if ! diff -u "$here/main.expected" "$tmp/out" >"$tmp/diff"; then
        echo "FAIL image ${tier:-(jit)}"
        head -40 "$tmp/diff"
        fail=1
    fi
done
[ "$fail" = 0 ] && echo "image: ok"
exit "$fail"
