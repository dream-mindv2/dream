#!/usr/bin/env bash
# Run with bash inside Termux. Builds locally; no root or desktop NDK needed.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash termux-install.sh [options]
  --source DIR     Build this checkout (default: this script's checkout,
                   otherwise clone into ~/.cache/dream/source)
  --prefix DIR     Install location (default: Termux's $PREFIX)
  --jobs N         Parallel compiler jobs (default: 2, to limit phone RAM use)
  --no-packages    Use dependencies already installed; do not run pkg
  --help           Show this help

Requires 64-bit Termux on Android 14 / API 34 or newer.
Standalone downloads clone https://github.com/Custards1/dream (main).
Existing checkouts are used as-is; this script never pulls or resets them.
EOF
}

die() { echo "dream installer: $*" >&2; exit 1; }
source_dir=
install_prefix=${PREFIX:-}
jobs=2
packages=yes
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h) usage; exit 0 ;;
        --source|--prefix|--jobs)
            [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"
            case $1 in
                --source) source_dir=$2 ;;
                --prefix) install_prefix=$2 ;;
                --jobs) jobs=$2 ;;
            esac
            shift 2 ;;
        --no-packages) packages=no; shift ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ -n ${PREFIX:-} && -x $PREFIX/bin/pkg && -x /system/bin/getprop ]] ||
    die "run this script inside Termux on an Android device"
api=$(/system/bin/getprop ro.build.version.sdk)
[[ $api =~ ^[0-9]+$ && $api -ge 34 ]] || die "Android 14 / API 34+ is required (device reports $api)"
case $(uname -m) in
    aarch64|arm64) triple=aarch64-linux-android34 ;;
    x86_64) triple=x86_64-linux-android34 ;;
    *) die "only 64-bit ARM and x86-64 Termux installations are supported" ;;
esac
[[ $install_prefix == /* ]] || die "--prefix must be an absolute path"
case $install_prefix in
    /sdcard*|/storage/*) die "install in Termux's private storage, not shared storage" ;;
esac

if [[ $packages == yes ]]; then
    pkg update -y
    pkg install -y git clang cmake ninja pkg-config libffi openssl ca-certificates libandroid-spawn
fi
for tool in git clang clang++ cmake ninja; do
    command -v "$tool" >/dev/null || die "$tool is missing; rerun without --no-packages"
done
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z $source_dir ]]; then
    if [[ -f $script_root/dreams/bootstrap/dreams.dream ]]; then
        source_dir=$script_root
    else
        source_dir="$HOME/.cache/dream/source"
        if [[ ! -e $source_dir ]]; then
            mkdir -p "$(dirname "$source_dir")"
            git clone --depth 1 --branch main https://github.com/Custards1/dream "$source_dir"
        fi
    fi
fi
[[ -f $source_dir/dreams/bootstrap/dreams.dream && -f $source_dir/dream/android/smoke.dr ]] ||
    die "--source must contain a Dream checkout with Android support"
source_dir=$(cd "$source_dir" && pwd)
case $source_dir in
    /sdcard*|/storage/*) die "keep the checkout in Termux's private storage (for example ~/dream)" ;;
esac
build_dir="$source_dir/build-termux"
export TMPDIR=${TMPDIR:-$PREFIX/tmp}
mkdir -p "$TMPDIR"

echo "Building Dream for $triple ($jobs jobs)..."
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_TARGET="$triple" -DCMAKE_CXX_COMPILER_TARGET="$triple" \
    -DCMAKE_BUILD_TYPE=Release -DDREAM_TERMUX=ON -DDREAM_ENABLE_JIT=OFF \
    -DDREAM_BUILD_TESTS=ON -DDREAM_LINK_DEPS=SHARED \
    -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DDREAM_FFI_INCLUDE="$PREFIX/include" -DDREAM_FFI_LIB="$PREFIX/lib/libffi.so" \
    -DOPENSSL_ROOT_DIR="$PREFIX"
cmake --build "$build_dir" --parallel "$jobs"
ctest --test-dir "$build_dir" -R '^dream_tests$' --output-on-failure

vm="$build_dir/bin/dream"
compiler="$build_dir/dreams.dream"
"$vm" "$source_dir/dreams/bootstrap/dreams.dream" \
    -L "$source_dir/mind" -L "$source_dir" "$source_dir/dreams/src/main.dr" -o "$compiler"
"$vm" "$compiler" -L "$source_dir/mind" "$source_dir/dream/android/smoke.dr" \
    -o "$build_dir/android-smoke.dream"
[[ $("$vm" "$build_dir/android-smoke.dream") == $'true\ntrue\ntrue' ]] ||
    die "Android platform, FFI, or subprocess smoke test failed"

# Finish all compilation and testing before replacing an existing install.
cmake --install "$build_dir"
share="$install_prefix/share/dream"
mkdir -p "$share/mind" "$share/examples" "$install_prefix/bin"
install -m 644 "$compiler" "$share/dreams.dream"
cp -R "$source_dir/mind/std" "$share/mind/"
cat > "$share/examples/hello.dr" <<'DR'
import std.console;
let main! = console.print! "Hello from Dream on Android!";
DR
{
    printf '#!%s/bin/bash\n' "$PREFIX"
    printf 'exec %q %q -L %q "$@"\n' \
        "$install_prefix/bin/dream" "$share/dreams.dream" "$share/mind"
} > "$install_prefix/bin/dreams"
chmod 755 "$install_prefix/bin/dreams"
"$install_prefix/bin/dreams" "$share/examples/hello.dr" -o "$build_dir/hello.dream"
"$install_prefix/bin/dream" "$build_dir/hello.dream"
printf '\nInstalled dream and dreams in %s/bin\n' "$install_prefix"
if [[ $install_prefix != "$PREFIX" ]]; then
    printf 'Add to your shell: export PATH=%q/bin:$PATH\n' "$install_prefix"
fi
printf '\nCompile and run your own program:\n  dreams hello.dr -o hello.dream\n  dream hello.dream\n'
