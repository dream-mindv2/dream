#!/usr/bin/env bash
# Mutated images must be rejected, not crashed on.
#
# A `.dream` file is untrusted input: it can arrive from anywhere, and the
# interpreter indexes its arrays without further checks on the strength of the
# loader's validation. So the property under test is narrow and absolute --
# whatever bytes it is handed, the VM exits normally or reports an error, and
# never dies on a signal.
#
# Two different invariants are checked, because they are not the same:
#
#   --dump  loads and walks the image but evaluates nothing, so it must always
#           terminate. A hang here means the loader or the tree walker can be
#           driven in circles, which is a bug.
#   run     evaluates. A mutated image can easily be a valid program that
#           loops forever -- flipping the `1` in `n - 1` to `0` is enough --
#           so a hang here is expected and only a crash is a failure.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMS="${DREAMS:-}"
dream="${dream:-}"
ITERATIONS="${ITERATIONS:-400}"

if [[ -z "$DREAMS" ]]; then
  for c in "$HERE/../../build/dreams.dream" "$HERE/../../dreams/bootstrap/dreams.dream"; do
    [[ -f "$c" ]] && DREAMS="$c" && break
  done
fi
if [[ -z "$dream" ]]; then
  for c in "$HERE/../../build-dream/bin/dream" "$HERE/../build-dream/bin/dream" \
           "$HERE/../build/bin/dream"; do
    [[ -x "$c" ]] && dream="$c" && break
  done
fi
if [[ ! -x "${dream:-}" ]]; then
  echo "fuzz: need the dream VM; build it or set dream" >&2
  exit 1
fi
if [[ -z "$DREAMS" ]]; then
  echo "fuzz: cannot find a compiler; run \`just dreams\` or set DREAMS" >&2
  exit 1
fi
if [[ "$DREAMS" == *.dream ]]; then compile=("$dream" "$DREAMS"); else compile=("$DREAMS"); fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SEED_SRC="$WORK/seed.dr"
cat > "$SEED_SRC" <<'DREAM'
import std.console;

let rec fac n = if n <= 1 { 1 } else { n * fac (n - 1) };
let table = %{ :a => [1, 2, 3], :b => #[4.5] };
let pair a b = fn x -> a + b + x;
let main! = {
    console.print! "fac: " (fac 10)
    console.print! "tab: " (len table)
    console.print! "fun: " (pair 1 2 3)
    console.print! "dat: " (len (_data_at 0))
}
DREAM

# The seed carries a payload, so that `LDAT` and `PAYL` are among the sections
# being mutated. They are the one part of the container addressed in 64 bits,
# which makes them the one part where a length can be made to wrap rather than
# merely to be too large -- exactly the arithmetic an attacker reaches for, and
# not something the other sections can exercise on its behalf.
printf 'fuzz payload bytes' > "$WORK/payload.bin"

if ! "${compile[@]}" "$SEED_SRC" -L "$HERE/../../mind" \
       --payload "$WORK/payload.bin" -o "$WORK/seed.dream" >/dev/null 2>&1; then
  echo "fuzz: could not compile the seed program" >&2
  exit 1
fi

SIZE=$(stat -c %s "$WORK/seed.dream")
failures=0
checked=0
loops=0

# A signal shows as 128+n; 124 is the timeout tool.
#
# $3 holds the extra flags, and also decides whether a hang counts: with
# `--dump` nothing is evaluated, so failing to finish is a bug.
run_and_check() {
  local file="$1" label="$2" flags="${3:-}"
  timeout 10 "$dream" $flags "$file" >/dev/null 2>&1
  local rc=$?
  checked=$((checked + 1))
  if [[ $rc -ge 128 ]]; then
    failures=$((failures + 1))
    echo "CRASH ($label, signal $((rc - 128)))"
    cp "$file" "$HERE/fuzz-crash-$failures.dream" 2>/dev/null || true
  elif [[ $rc -eq 124 ]]; then
    if [[ "$flags" == *--dump* ]]; then
      failures=$((failures + 1))
      echo "HANG while loading ($label) -- dumping evaluates nothing and must finish"
      cp "$file" "$HERE/fuzz-hang-$failures.dream" 2>/dev/null || true
    else
      # A mutated program that loops forever is still a valid program.
      loops=$((loops + 1))
    fi
  fi
}

echo "fuzzing a $SIZE byte image, $ITERATIONS mutations"

# 1. Single-byte mutations at random offsets.
for ((i = 0; i < ITERATIONS; i++)); do
  cp "$WORK/seed.dream" "$WORK/m.dream"
  off=$((RANDOM * 32768 + RANDOM))
  off=$((off % SIZE))
  val=$((RANDOM % 256))
  printf "$(printf '\\x%02x' "$val")" |
    dd of="$WORK/m.dream" bs=1 seek="$off" count=1 conv=notrunc status=none
  run_and_check "$WORK/m.dream" "byte $off = $val" ""
  run_and_check "$WORK/m.dream" "dump, byte $off = $val" "--dump"
done

# 2. Truncation at every 16-byte boundary: a short section table or a section
#    that runs off the end is the easiest way to walk past the mapping.
for ((len = 0; len < SIZE; len += 16)); do
  head -c "$len" "$WORK/seed.dream" > "$WORK/t.dream"
  run_and_check "$WORK/t.dream" "truncated to $len" ""
  run_and_check "$WORK/t.dream" "dump, truncated to $len" "--dump"
done

# 3. Deliberately hostile counts in the header, which is what a real attacker
#    would reach for first.
for field in 28 20 24; do
  cp "$WORK/seed.dream" "$WORK/h.dream"
  printf '\xff\xff\xff\x7f' |
    dd of="$WORK/h.dream" bs=1 seek="$field" count=4 conv=notrunc status=none
  run_and_check "$WORK/h.dream" "header field $field = huge" ""
  run_and_check "$WORK/h.dream" "dump, header field $field = huge" "--dump"
done

echo
echo "$checked runs, $failures failures, $loops mutations that ran forever (expected)"
[[ $failures -eq 0 ]]
