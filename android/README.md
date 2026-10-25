# Dream on Android

## Build and install in Termux

Use 64-bit Termux on **Android 14 or newer**, on an arm64 (aarch64) or x86_64
device. In a checkout of the workspace with its submodules, run:

```sh
git clone --recurse-submodules https://github.com/Custards1/dream ~/dream
cd ~/dream
bash dream/android/termux-install.sh
```

The VM, compiler and standard library are submodules of the workspace; a
checkout cloned without them has `git submodule update --init` to fetch them.

The installer installs dependencies with `pkg`, builds the interpreter and
self-hosted compiler on your device, runs VM and Android smoke tests, and
installs `dream` and `dreams` into `$PREFIX/bin`. The standard library and
compiler image go under `$PREFIX/share/dream`. No root, desktop computer,
NDK download, `just`, or LLVM JIT build is needed. Termux's OpenSSL uses the
CA roots supplied by its `ca-certificates` package.

Keep the checkout in Termux's home directory, such as `~/dream`. Shared
storage (`/sdcard`, `/storage/emulated/0`) cannot provide the executable files
and filesystem behavior required by the build. The default uses two compiler
jobs to limit memory use; use `--jobs 1` if Android kills the compiler.

```sh
bash dream/android/termux-install.sh --jobs 1
# Or use dependencies you already installed:
bash dream/android/termux-install.sh --no-packages
# Choose another checkout or installation directory:
bash dream/android/termux-install.sh --source "$HOME/dream" --prefix "$HOME/.local"
```

A standalone copy of the installer clones the repository into
`~/.cache/dream/source`. Existing checkouts are used as-is. To update, update
your checkout yourself and rerun the installer. It does not pull, reset, or
discard local work. `--help` lists the options.

## Compile and run

```sh
cat > hello.dr <<'EOF'
import std.console;
let main! = console.print! "Hello from Android!";
EOF

dreams hello.dr -o hello.dream
dream hello.dream
```

`dreams` automatically finds the installed standard library. You can also
copy a portable `.dream` image built on another OS and run it directly.
Android-specific code can use `when os == "android"`; runtime detection is
`os.platform ()`, which returns `:android`.

For desktop NDK builds, device/emulator tests, native payloads, and application
embedding limits, see [platform support](../docs/platforms.md#android).
