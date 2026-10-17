#!/bin/sh
# std.ffi and std.foreign against a C library built for them.
#
# This directory is a package, and its `build.dr` makes the library from
# `sample.c` with `std.build.cc` and embeds it as a payload -- so this checks
# `mind` running a build script and the embedding both, and needs nothing
# installed but a C compiler. It is built from a copy, so that nothing is
# written into the source tree. The program runs under the interpreter and
# under the JIT, and both must print exactly `ffi.expected`: a callback is
# Dream running underneath C, which is precisely the kind of re-entry the two
# tiers could disagree about.
#
# A machine with no C compiler skips rather than fails, since the library
# cannot be built. There is no VM without libffi to skip for: `std.ffi` is a
# required part of the VM, and the build refuses to make one that lacks it.
set -u

dream=${dream:-build-dream/bin/dream}
dreams=${dreams:-build/dreams.dream}
here=$(dirname "$0")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cc=${CC:-cc}
if ! command -v "$cc" >/dev/null 2>&1; then
    echo "skip ffi: no C compiler"
    exit 0
fi

root=$(cd "$here/../../.." && pwd)
case "$dream" in /*) ;; *) dream="$root/$dream" ;; esac
case "$dreams" in /*) ;; *) dreams="$root/$dreams" ;; esac
mkdir -p "$tmp/pkg"
cp -r "$here/mind.toml" "$here/build.dr" "$here/src" "$here/sample.c" "$tmp/pkg/"
timeout 300 "$dream" "$dreams" -L "$root/mind/std" "$root/mind/tool/src/main.dr" -o "$tmp/mind.dream" >"$tmp/build" 2>&1 \
    || { cat "$tmp/build"; exit 1; }
(cd "$tmp/pkg" && CC="$cc" DREAM="$dream" MIND_STDLIB="$root/mind" MIND_HOME="$tmp/home" \
    timeout 300 "$dream" "$tmp/mind.dream" build --compiler "$dreams" -o "$tmp/ffi.dream") >"$tmp/build" 2>&1 \
    || { cat "$tmp/build"; exit 1; }

fail=0
for tier in "" "--no-jit"; do
    # shellcheck disable=SC2086
    timeout 20 "$dream" $tier "$tmp/ffi.dream" >"$tmp/out" 2>&1
    if ! diff -u "$here/ffi.expected" "$tmp/out" >"$tmp/diff"; then
        echo "FAIL ffi ${tier:-(jit)}"
        head -40 "$tmp/diff"
        fail=1
    fi
done
# The same library as one installed on the system: a plain shared object in a
# directory of its own, found by `from system "sample"` through each way a
# person can point at it, and not found -- with the reason -- when they do not.
mkdir -p "$tmp/lib"
case "$(uname -s)" in
    Darwin) lib="$tmp/lib/libsample.dylib"; shared=-dynamiclib ;;
    *) lib="$tmp/lib/libsample.so"; shared=-shared ;;
esac
"$cc" $shared -fPIC -O2 -o "$lib" "$here/sample.c" >"$tmp/build" 2>&1 || { cat "$tmp/build"; exit 1; }
timeout 300 "$dream" "$dreams" -L "$root/mind" "$here/src/system.dr" -o "$tmp/system.dream" >"$tmp/build" 2>&1 \
    || { cat "$tmp/build"; exit 1; }
found="found libsample.so
1
3
any_of takes the first that opens"
for how in "DREAM_LIBRARY_PATH=$tmp/lib" "DREAM_LIB_SAMPLE=$lib" ""; do
    if [ -n "$how" ]; then want="$found"; else want="not found
says what to set
any_of takes the first that opens"; fi
    got=$(env -u DREAM_LIBRARY_PATH -u DREAM_LIB_SAMPLE $how timeout 20 "$dream" "$tmp/system.dream" 2>&1 \
          | sed "s|$tmp/lib/libsample.dylib|libsample.so|")
    if [ "$got" != "$want" ]; then
        echo "FAIL ffi system library (${how:-nothing set})"
        echo "$got" | head -20
        fail=1
    fi
done
[ "$fail" = 0 ] && echo "ffi: ok"
exit "$fail"
