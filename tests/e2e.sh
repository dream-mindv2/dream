#!/usr/bin/env bash
# End-to-end tests: compile each program with `dreams`, run it under the VM, and
# compare against the recorded output.
#
# Every program runs twice, once with the JIT and once without. Both tiers must
# agree exactly -- a JIT that changes what a program prints is a broken JIT,
# and for a lazy language that is an easy mistake to make.

set -uo pipefail

# `DREAM` is the conventional spelling; the lowercase `dream` is accepted too
# because it is what this script used first, and a lowercase environment
# variable is easy to set by accident from a shell where `dream` is also a
# path or an alias.
dream="${DREAM:-${dream:-}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [[ -z "$dream" ]]; then
    for c in "$ROOT/build-dream/bin/dream" "$HERE/../build-dream/bin/dream" \
           "$ROOT/build/bin/dream"; do
    [[ -x "$c" ]] && dream="$c" && break
  done
fi

if [[ ! -x "${dream:-}" ]]; then
  echo "e2e: cannot find the dream VM; build it or set dream" >&2
  exit 1
fi

# The compiler is `dreams`, an image rather than a native program: a freshly
# built one when there is one, and otherwise the checked-in seed, which needs
# nothing but the VM this script is testing. `$DREAMS` overrides the choice,
# and a name that does not end in `.dream` is executed directly.
DREAMS="${DREAMS:-}"
if [[ -z "$DREAMS" ]]; then
  for c in "$ROOT/build/dreams.dream" "$ROOT/dreams/bootstrap/dreams.dream"; do
    [[ -f "$c" ]] && DREAMS="$c" && break
  done
fi
if [[ -z "$DREAMS" ]]; then
  echo "e2e: cannot find a compiler; run \`just dreams\` or set DREAMS" >&2
  exit 1
fi
if [[ "$DREAMS" == *.dream ]]; then compile=("$dream" "$DREAMS"); else compile=("$DREAMS"); fi
# The checkout's own `std`, named rather than found. A program under
# `programs/` finds it by walking up to `mind/`, but the shebang, payload and
# session programs are written to a temporary directory, which has no `mind`
# above it -- and since `std.console` became Dream, even a one-line program
# needs a `std` to compile.
compile+=(-L "$ROOT/mind")

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

# A stack regression may ask for a smaller OS-thread stack, in KiB. Apply it
# only to the VM running the image, not to the compiler or later test cases.
# Shells without this resource limit still run the tier-agreement check.
run_image() {
  if [[ -f "$HERE/programs/$name.stack" ]]; then
    local stack_kb
    read -r stack_kb < "$HERE/programs/$name.stack"
    ulimit -S -s "$stack_kb" 2>/dev/null || :
  fi
  exec "$@"
}

# Policy parsing is a VM option, including in an interpreter-only build.
bad_policy="$("$dream" --jit-policy invalid 2>&1)"
bad_policy_rc=$?
if [[ "$bad_policy_rc" != 2 || "$bad_policy" != *'unknown JIT policy'* ]]; then
  echo "FAIL jit_policy (unknown policy was not rejected)"
  fail=$((fail + 1))
fi
missing_policy="$("$dream" --jit-policy 2>&1)"
missing_policy_rc=$?
if [[ "$missing_policy_rc" != 2 || "$missing_policy" != *'needs a value'* ]]; then
  echo "FAIL jit_policy (missing policy was not rejected)"
  fail=$((fail + 1))
fi

# Each program is a case of its own, run as a job beside the others: they
# share nothing but the compiler, which keeps what it builds (`--cache`), and
# a case is mostly waiting on a VM. What each prints goes to a log of its own,
# and the logs are shown in order once every case is done, so the output reads
# as it did when they ran one after another.
e2e_case() {
  src="$1"

  name="$(basename "$src" .dr)"
  expected="$HERE/programs/$name.expected"
  image="$WORK/$name.dream"

  if ! "${compile[@]}" "$src" -o "$image" >"$WORK/$name.compile" 2>&1; then
    echo "FAIL $name (compile)"
    sed 's/^/    /' "$WORK/$name.compile"
    return 1
  fi

  # A program may set VM environment variables in `<name>.env`, one
  # `NAME=value` per line. It is how a test that has to trip a runtime limit
  # asks for a small one, rather than spending the seconds it would take to
  # reach the real one.
  env_args=()
  if [[ -f "$HERE/programs/$name.env" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" || "$line" == \#* ]] && continue
      env_args+=("$line")
    done < "$HERE/programs/$name.env"
  fi

  jit_args=()
  if [[ -f "$HERE/programs/$name.jit" ]]; then
    # These regressions must exercise compiled code even on their first call.
    jit_args=(--jit-threshold 1)
    compiled_ok=true
    while IFS= read -r fn; do
      ir="$("$dream" --dump-jit "$fn" "$image" 2>&1)"
      if [[ "$ir" == "dream: this build has no JIT" ]]; then break; fi
      if [[ "$ir" != *"define "* ]]; then
        echo "FAIL $name ($fn did not compile)"
        echo "$ir"
        compiled_ok=false
      fi
      if [[ "$name" == jit_integer && "$fn" == calc && "$ir" != *branch_weights* ]]; then
        echo "FAIL $name (integer hint missing from IR)"
        compiled_ok=false
      fi
      if [[ "$name" == jit_recursion_loops && ( "$ir" == *'.rec('* || "$ir" != *'br label %loop'* ) ]]; then
        echo "FAIL $name ($fn did not lower to a loop)"
        compiled_ok=false
      fi
    done < "$HERE/programs/$name.jit"
    if [[ "$compiled_ok" == false ]]; then
      return 1
    fi
  fi
  jit_out="$(run_image env "${env_args[@]}" "$dream" "${jit_args[@]}" "$image" 2>&1)"
  jit_rc=$?
  int_out="$(run_image env "${env_args[@]}" "$dream" --no-jit "$image" 2>&1)"
  int_rc=$?

  if [[ "$jit_out" != "$int_out" || "$jit_rc" != "$int_rc" ]]; then
    echo "FAIL $name (the two tiers disagree)"
    diff <(printf '%s\n' "$int_out") <(printf '%s\n' "$jit_out") | sed 's/^/    /'
    return 1
  fi

  if ! diff -q <(printf '%s\n' "$jit_out") "$expected" >/dev/null; then
    echo "FAIL $name (output)"
    diff "$expected" <(printf '%s\n' "$jit_out") | sed 's/^/    /'
    return 1
  fi

  if [[ "$name" == jit_recursion_loops ]]; then
    policy_out="$(run_image env DREAM_JIT_RANK=adaptive "$dream" --jit-policy density \
                  --stats --jit-threshold 1 "$image" 2>&1)"
    if [[ "$policy_out" == *'jit on'* && "$policy_out" != *'; jit policy density'* ]]; then
      echo "FAIL jit_policy (CLI did not override the environment)"
      return 1
    fi
  fi

  echo "ok   $name"
  return 0
}

jobs_max="${E2E_JOBS:-$(( ($(nproc 2>/dev/null || echo 2) + 1) / 2 ))}"
for src in "$HERE"/programs/*.dr; do
  name="$(basename "$src" .dr)"
  ( e2e_case "$src" >"$WORK/$name.log" 2>&1; echo $? >"$WORK/$name.rc" ) &
  while [[ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]]; do wait -n; done
done
wait
for src in "$HERE"/programs/*.dr; do
  name="$(basename "$src" .dr)"
  cat "$WORK/$name.log"
  if [[ "$(cat "$WORK/$name.rc" 2>/dev/null)" == 0 ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done

# --- shebang images ---------------------------------------------------------
#
# `--shebang` writes an interpreter line before the image and sets the
# execute bit. The VM skips such a line, so the file has to work both ways: run
# as a command, and loaded as an ordinary image. Testing only the first would
# miss an image the VM can no longer read.

shebang_src="$(mktemp -d)"
trap 'rm -rf "$shebang_src"' EXIT
cat > "$shebang_src/hello.dr" <<'EOF'
import std.console;
let main! = console.print! "shebang";
EOF

if "${compile[@]}" "$shebang_src/hello.dr" --shebang "$(cd "$(dirname "$dream")" && pwd)/$(basename "$dream")" \
     -o "$shebang_src/hello" >/dev/null 2>&1; then
  direct_out="$("$shebang_src/hello" 2>&1)"
  loaded_out="$("$dream" "$shebang_src/hello" 2>&1)"
  first_line="$(head -c 2 "$shebang_src/hello")"
  if [[ ! -x "$shebang_src/hello" ]]; then
    echo "FAIL shebang (the image was not made executable)"
    fail=$((fail + 1))
  elif [[ "$first_line" != "#!" ]]; then
    echo "FAIL shebang (no interpreter line was written)"
    fail=$((fail + 1))
  elif [[ "$direct_out" != "shebang" ]]; then
    echo "FAIL shebang (running it directly printed: $direct_out)"
    fail=$((fail + 1))
  elif [[ "$loaded_out" != "shebang" ]]; then
    echo "FAIL shebang (loading it as an image printed: $loaded_out)"
    fail=$((fail + 1))
  else
    echo "ok   shebang"
    pass=$((pass + 1))
  fi
else
  echo "FAIL shebang (could not compile with --shebang)"
  fail=$((fail + 1))
fi

# --- payloads ---------------------------------------------------------------
#
# `--payload` puts a file's bytes in the image past everything the rest of the
# container can address, and `data_at` hands them back as a view rather
# than a string. Its own block rather than a program in the loop above, because
# it is the compile that differs: the flag, and the files it names.
#
# What this is really checking is that nothing copies. There is no way to assert
# that from inside the language, so it is checked from the outside instead: the
# bytes must survive the round trip through the image exactly, including the
# ones that are not valid UTF-8 and would not survive being decoded.

payload_dir="$(mktemp -d)"
trap 'rm -rf "$WORK" "$shebang_src" "$payload_dir"' EXIT

printf 'the quick brown fox' > "$payload_dir/a.bin"
# Deliberately not text: a byte no decoder would let through unchanged.
printf 'head\x00\x80\xfftail' > "$payload_dir/b.bin"

cat > "$payload_dir/payload.dr" <<'EOF'
// `std.core` and no more: the programs here are compiled without a package
// path, so `std.str` -- which is Dream source in `mind/std` -- is not reachable.
// `str.byte` and `str.slice` are one-line wrappers over these two anyway.
import std.console;

import std.io;

let main! = {
    let a = _data_at 0;
    let b = _data_at 1;
    console.print! ("count " + to_string (_data_count ()))
    console.print! ("type " + to_string (type_of a))
    console.print! ("lens " + to_string (len a) + " " + to_string (len b))
    console.print! ("byte " + to_string (_str_byte a 4))
    console.print! ("slice " + to_string (_str_slice a 4 5 == "quick"))
    console.print! ("whole " + to_string (a == "the quick brown fox"))
    console.print! ("order " + to_string (b < a))
    // A slice is a view too, so slicing one again must land in the same place.
    console.print! ("again " + to_string (_str_slice (_str_slice a 4 9) 0 5 == "quick"))
    // The bytes cross a process boundary as a view, not as a copy.
    console.print! ("child " + to_string (join! (spawn! $( len b ))))
    // Concatenating one is refused, and the message says what to do instead.
    console.print! ("refused " + to_string (type_of (try! { a + "x" } catch e { e })))
    // Naming an atom is a read and builds nothing, so a view is a string for
    // it the way it is for `len` and `==`. `:quick` exists because this line
    // writes it; `:the` is written nowhere, so the same call over the other
    // slice has nothing to find.
    console.print! ("atom " + to_string (_to_existing_atom (_str_slice a 4 5) == :quick))
    console.print! ("noatom " + to_string (_to_existing_atom (_str_slice a 0 3)))
    io.write! (io.stdout! ()) b
};
EOF

{
  printf 'count 2\ntype :bigstr\nlens 19 11\nbyte 113\nslice true\nwhole true\n'
  printf 'order true\nagain true\nchild 11\nrefused :error\natom true\nnoatom ()\n'
  cat "$payload_dir/b.bin"
} > "$payload_dir/expected"

if "${compile[@]}" "$payload_dir/payload.dr" \
     --payload "$payload_dir/a.bin" --payload "$payload_dir/b.bin" \
     -o "$payload_dir/payload.dream" >"$payload_dir/compile" 2>&1; then
  # Through files rather than `$(...)`: the payload this program writes out
  # contains a NUL, and a command substitution drops it -- which would quietly
  # turn the one check that proves the bytes are untouched into a check that
  # they are untouched apart from the interesting one.
  "$dream" "$payload_dir/payload.dream" >"$payload_dir/jit.out" 2>&1
  "$dream" --no-jit "$payload_dir/payload.dream" >"$payload_dir/int.out" 2>&1
  # The payload is the last thing in the file, unpadded, in the order given.
  tail_bytes="$(tail -c 30 "$payload_dir/payload.dream" | od -An -tx1 | tr -d ' \n')"
  want_bytes="$(cat "$payload_dir/a.bin" "$payload_dir/b.bin" | od -An -tx1 | tr -d ' \n')"
  if ! cmp -s "$payload_dir/jit.out" "$payload_dir/int.out"; then
    echo "FAIL payload (the two tiers disagree)"
    diff <(od -c "$payload_dir/int.out") <(od -c "$payload_dir/jit.out") | sed 's/^/    /'
    fail=$((fail + 1))
  elif ! cmp -s "$payload_dir/jit.out" "$payload_dir/expected"; then
    echo "FAIL payload (output)"
    diff <(od -c "$payload_dir/expected") <(od -c "$payload_dir/jit.out") | sed 's/^/    /'
    fail=$((fail + 1))
  elif [[ "$tail_bytes" != "$want_bytes" ]]; then
    echo "FAIL payload (the image does not end with the payload bytes)"
    echo "    want $want_bytes"
    echo "    got  $tail_bytes"
    fail=$((fail + 1))
  else
    echo "ok   payload"
    pass=$((pass + 1))
  fi
else
  echo "FAIL payload (could not compile with --payload)"
  sed 's/^/    /' "$payload_dir/compile"
  fail=$((fail + 1))
fi

# A build with no payload must write the bytes it wrote before the feature
# existed. That is what the bootstrap's byte-equality rests on, so it is worth
# asserting directly rather than only noticing when the seed stops reproducing.
"${compile[@]}" "$shebang_src/hello.dr" -o "$payload_dir/plain1.dream" >/dev/null 2>&1
"${compile[@]}" "$shebang_src/hello.dr" -o "$payload_dir/plain2.dream" >/dev/null 2>&1
if cmp -s "$payload_dir/plain1.dream" "$payload_dir/plain2.dream" &&
   ! grep -qa 'PAYL' "$payload_dir/plain1.dream"; then
  echo "ok   no-payload build carries no payload sections"
  pass=$((pass + 1))
else
  echo "FAIL no-payload build (a PAYL section appeared, or the build is not deterministic)"
  fail=$((fail + 1))
fi

# --- compile-time sessions --------------------------------------------------
#
# `vm.open_image!` loads an image and keeps it; `vm.call_image!` calls a named
# global in it with ordinary values; `vm.close_image!` frees it. Its own block
# rather than a program in the loop above because it takes *two* compiles: the
# point of the feature is one image calling into another.
#
# What it is for is macro expansion, which until now answered a transformer by
# compiling the whole program with the call quoted into it. An image that can
# be called is an image that can be kept, which is what makes a transformer a
# library rather than a program written for the occasion.
#
# Everything here imports host modules only. These programs are compiled with
# no package path, so `std.list` and `std.str` -- which are Dream source in
# `mind/std` -- are not reachable, and the library deliberately imports nothing
# at all.

session_dir="$(mktemp -d)"
trap 'rm -rf "$WORK" "$shebang_src" "$payload_dir" "$session_dir"' EXIT

cat > "$session_dir/lib.dr" <<'EOF'
// A library: no `main!`, nothing imported, nothing to run. What a caller
// reaches is a global, by the name of its module and its own.
let twice x = x + x;
let greet name = "hello " + name;
let pair a b = [a, b];
let boom x = x / 0;
// A pure nullary global is a value, so naming this one hands back `twice`
// and the arguments apply to what comes back. A transformer reached through
// an alias is exactly this shape.
let alias = twice;

mod inner {
    // Same name as a global of the outer module, so a lookup that ignored
    // which module it was asked about could answer with the wrong one.
    let twice x = x + x + 1;
}
EOF

cat > "$session_dir/caller.dr" <<'EOF'
import std.console;
import std.io;
import std.vm;

let rec read_all! h acc = {
    let chunk = io.read! h 65536;
    if len chunk == 0 { acc } else { read_all! h (acc + chunk) }
};

let main! = {
    let bytes = read_all! (io.stdin! ()) "";
    let s = vm.open_image! bytes;
    // The same session, asked four times: what an image costs is loading it,
    // and a session is that cost paid once.
    console.print! (vm.call_image! s "lib" "twice" [21])
    console.print! (vm.call_image! s "lib" "greet" ["world"])
    console.print! (vm.call_image! s "lib" "twice" [1000])
    // Nested data crosses in both directions, by the same rule that governs
    // what `eval_image!` brings back.
    console.print! (vm.call_image! s "lib" "pair" [[:a, 1], %{ :k => "v" }])
    // A nullary global naming a function, applied to an argument anyway.
    console.print! (vm.call_image! s "lib" "alias" [7])
    // A module is part of the name. `inner.twice` is a different function.
    console.print! (vm.call_image! s "lib.inner" "twice" [21])
    // A transformer that raises is a failure of the call, not of the caller.
    console.print! (try! { vm.call_image! s "lib" "boom" [1] } catch e { type_of e })
    // A name the image does not have says so rather than calling something else.
    console.print! (try! { vm.call_image! s "lib" "absent" [1] } catch e { type_of e })
    // Closing answers whether the handle named anything, so closing twice is
    // not an error -- a compile that failed partway is entitled to close what
    // it thinks it opened without working out how far it got.
    console.print! (vm.close_image! s)
    console.print! (vm.close_image! s)
    // A handle that is no longer open names nothing, rather than someone else.
    console.print! (try! { vm.call_image! s "lib" "twice" [1] } catch e { type_of e })
};
EOF

cat > "$session_dir/expected" <<'EOF'
42
hello world
2000
[[:a, 1], %{:k => "v"}]
14
43
:error
:error
true
false
:error
EOF

if ! "${compile[@]}" "$session_dir/lib.dr" -o "$session_dir/lib.dream" \
       >"$session_dir/compile" 2>&1; then
  echo "FAIL session (could not compile the library)"
  sed 's/^/    /' "$session_dir/compile"
  fail=$((fail + 1))
elif ! "${compile[@]}" "$session_dir/caller.dr" -o "$session_dir/caller.dream" \
       >"$session_dir/compile" 2>&1; then
  echo "FAIL session (could not compile the caller)"
  sed 's/^/    /' "$session_dir/compile"
  fail=$((fail + 1))
else
  # The library is handed over on stdin rather than by a path the program
  # spells, so the test does not depend on where the temporary directory is.
  jit_out="$("$dream" "$session_dir/caller.dream" < "$session_dir/lib.dream" 2>&1)"
  int_out="$("$dream" --no-jit "$session_dir/caller.dream" < "$session_dir/lib.dream" 2>&1)"
  if [[ "$jit_out" != "$int_out" ]]; then
    echo "FAIL session (the two tiers disagree)"
    diff <(printf '%s\n' "$int_out") <(printf '%s\n' "$jit_out") | sed 's/^/    /'
    fail=$((fail + 1))
  elif ! diff -q <(printf '%s\n' "$jit_out") "$session_dir/expected" >/dev/null; then
    echo "FAIL session (output)"
    diff "$session_dir/expected" <(printf '%s\n' "$jit_out") | sed 's/^/    /'
    fail=$((fail + 1))
  else
    echo "ok   session"
    pass=$((pass + 1))
  fi
fi

echo
echo "$pass passed, $fail failed"
[[ $fail -eq 0 ]]
