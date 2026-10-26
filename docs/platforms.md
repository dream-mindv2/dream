# Platform and image portability

Dream targets native 64-bit Linux, Windows, macOS, and Android. A compiled `.dream`
image is portable bytecode: copy the **same file**, without recompiling or
rewriting it, to a compatible Dream VM on any of these systems. This includes
the compiler image, `dreams.dream`.

The on-disk format uses fixed-width little-endian integers, IEEE 754 binary64
floats, and indices rather than host pointers. The current mapped reader
requires a 64-bit little-endian host (including x86-64 and ARM64). The VM checks
these host properties at build time. LLVM machine code is generated locally
at runtime and is not part of the portable image. The bytecode format and VM
version must be compatible; portability is not a promise of compatibility
with arbitrary future format versions.

Ordinary language semantics, binary strings, concurrency, and standard file,
process, and network operations belong to the portable interface. Native OS
calls live in the VM. Windows paths and arguments are converted between
Dream's UTF-8 and Windows' UTF-16, and binary file I/O does not translate CRLF
or treat byte 26 as end of file. Windows sockets are kept distinct from CRT
file descriptors. Linux and Android use epoll; macOS and Windows use a readiness poller.
The portable poller checks readiness every 5 ms and is a correctness baseline,
not a claim of identical throughput or latency to epoll.

Machine stacks are the VM's, not the platform's. Every Dream process runs on a
scheduler worker started with a 16 MB stack (`kWorkerStack` in
`dream/src/stack_thread.hpp`), where a plain thread would get `ulimit -s` from
glibc and 1 MB from Android's Bionic and from Windows -- on which the compiler,
compiling itself, needed about 2 MB and segfaulted. The stack is reserved, not
committed, so the size costs a shallow program nothing. A force nested deeper
than the stack holds raises `:stack_overflow`, as a recursion past the heap's
limits does.

`-DDREAM_SELF_CONTAINED=ON` (Linux, macOS) links everything but the C library
into the VM -- libffi, OpenSSL, the C++ runtime and, with a JIT, LLVM and its
zlib and zstd -- so one binary runs on any distribution with a glibc as new as
the one it was built against. It needs those as archives (PIC libffi, an
OpenSSL built without zlib, `libstdc++-static`; zlib and zstd only for LLVM's
sake, with the JIT). It is how the published toolchains are
built (dream-mindv2/toolchain, `ci/linux.sh`); without LLVM it stops, like
any build, unless the JIT is turned off with `-DDREAM_ENABLE_JIT=OFF`.

Platform-specific work remains platform-specific:

- FFI library names, exported symbols, and C ABIs require matching libraries
  on the receiving system. `std.ffi` itself is present in every VM; the
  library it opens is what has to exist there.
- Executable names, shell commands, environment variables, permissions, and
  filesystem case sensitivity depend on the host. `mind` dependency fetching
  requires the external programs it invokes, such as Git, curl, and tar.
- A shebang is optional metadata for Unix launchers. Use `dream program.dream`
  on every OS; Windows does not execute a `.dream` file as a native executable.
- An image records where it may run when it cannot run everywhere. A payload
  that is a native library (ELF, Mach-O or PE) pins it to that library's
  system and machine, and so does a `when os == ..` it was compiled through.
  `dreams --target os=linux,arch=x86_64` states it outright, which is also
  how to cross-build: a single `os` sets the `os` that `when` compares
  against. `--target any` records nothing, for an image that carries one
  library per system and chooses at run time. The VM refuses a mismatched
  image before it runs; `dream --any-target` runs it anyway. The header bits
  are in [the image format](bytecode-format.md#the-images-target).
- `os.platform ()` reports the **running** VM's OS. Use it for runtime selection
  when one image needs several implementations. `when os == "..."` selects
  code at **compile time** and therefore intentionally specializes an image.
- Windows `os.replace!` launches a child with the terminal attached, waits,
  and exits with its status. Windows has no POSIX process-image replacement,
  so the PID is not preserved.

## Building

Use a C++20 compiler and CMake 3.20 or newer. GCC, Clang, and MSVC are supported
by the build configuration. libffi is required -- `std.ffi` is part of every
VM -- and so is TLS for `std.tls`: OpenSSL 3 on Linux, macOS and Android (`libssl-dev`,
or Homebrew's `openssl@3`, which CMake finds by itself), and on Windows
SChannel, which is part of the system and needs nothing installed. LLVM is
optional. Start with the interpreter to avoid LLVM:

```text
cmake -S . -B build -DDREAM_ENABLE_JIT=OFF
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -R "^dream_tests$" --output-on-failure
```

Single-configuration builds put the VM in `build/bin/dream` (or `dream.exe`).
Visual Studio puts it in `build/bin/Release/dream.exe`. Keep the Windows VM DLL
beside the executable. Enable LLVM with `DREAM_ENABLE_JIT`; without it the
build produces an interpreter. Without libffi the configure stops: point
`DREAM_FFI_INCLUDE` and `DREAM_FFI_LIB` at an unusual install, or on Windows use
vcpkg's `libffi` through its toolchain file. Without OpenSSL the configure stops
too; `-DOPENSSL_ROOT_DIR` points it at an unusual install. On Windows, use
libraries built for the same compiler ABI.

### Android

To build and install directly on a phone, use the
[Termux installer](../android/README.md). It installs the VM, compiler,
and standard library using Termux's own dependencies.

Android is an interpreter target for **arm64-v8a (aarch64) and x86_64, API 34+**
(Android 14); the build takes either name for 64-bit ARM, the NDK's ABI or the
processor's. It uses the NDK, libffi, OpenSSL 3, and the existing epoll IO
backend. API 34 is required by `posix_spawn_file_actions_addchdir_np`, used
by `os.exec_in!` and `os.exec_with!`. JIT builds and 32-bit ABIs are rejected
at configure time. Android libraries and executables use 16 KB ELF alignment.

With NDK r28 or newer and unpacked libffi and OpenSSL 3 source releases:

```sh
export ANDROID_NDK_ROOT=/path/to/android-sdk/ndk/29.0.14206865
bash dream/android/build.sh arm64-v8a build/android-arm64 /path/to/libffi /path/to/openssl
```

The helper builds PIC static dependencies, then builds the VM through the NDK
CMake toolchain. `build/android-arm64/dist` contains `dream`, `dream_tests`,
`libdream.so`, and `libc++_shared.so`. Use `x86_64` for the emulator. The build
requires CMake, make, Perl, and a host C/C++ toolchain on Linux or macOS.
Existing Android dependency builds can instead be passed directly to CMake
using the NDK toolchain, `ANDROID_ABI`, `ANDROID_PLATFORM=android-34`,
`DREAM_FFI_INCLUDE`, `DREAM_FFI_LIB`, and the OpenSSL cache variables.

`os.platform ()` returns `:android`; its compile-time family is `"unix"`.
Use `dreams --target os=android,arch=aarch64` for Android-specific images.
Android and desktop Linux are separate image targets even though both use
ELF: libraries linked against glibc cannot run against Android's Bionic libc.
ELF payload sniffing defaults to Linux, so **explicitly select Android when
embedding Android native libraries**. Portable bytecode without native
payloads or OS-specific conditions needs no target flag.

For a shell deployment, push the four files and your image to a directory
under `/data/local/tmp`, set `LD_LIBRARY_PATH` and `TMPDIR` to that directory,
and run `dream program.dream`. Android has no general `/tmp` directory.
For application embedding, package `libdream.so` and `libc++_shared.so` in
the APK's native library directory and call the C API through your JNI layer.
The app must supply writable paths and any required Android permissions
(including `INTERNET` for sockets). Android's app sandbox and executable
loading restrictions still apply; the shell test harness does not establish
APK/JNI integration or permission handling.

Default TLS trust reads Android's system CA certificates from Conscrypt's
APEX store, falling back to `/system/etc/security/cacerts`. It does not apply
Java Network Security Configuration or user-installed roots. Supply `:ca`
to select application-specific PEM roots. Dependencies are bundled; Dream
does not link Android's private platform TLS libraries.
Native Termux builds instead use their packaged OpenSSL's default CA paths.

To verify a device or emulator using the same Linux-compiled bytecode bundle:

```sh
python3 dream/tests/portable.py compile --vm build/bin/dream --images build/images
python3 dream/android/test.py --serial emulator-5554 \
  --dist build/android-x86_64/dist --images build/images
```

The runner verifies hashes on both sides of `adb push`, runs VM unit tests,
language and IO fixtures, loopback TLS, FFI, subprocess working directories,
and the portable compiler on Android. CI cross-builds both ABIs and runs the
x86_64 suite on an API 34 emulator. ARM64 device execution and APK integration
require separate validation.

### TLS on each platform

`std.tls` is one interface over two libraries, and a program cannot tell which
it got: OpenSSL on Linux, macOS and Android, SChannel on Windows. Each is driven from
memory by the same socket code in the VM, so parking, non-blocking IO and
closing behave identically; the options are the ones every backend honours
(an identity is a PKCS#12 bundle, trusted roots are PEM); and a failure is one
of the same few kinds. What legitimately differs is the system's own trust
store, consulted when a program gives no `:ca` -- a program that names its
roots trusts the same thing everywhere. The version and cipher `tls.info!`
reports are each library's spelling, and a revocation list given as `:crl`
is honoured by both. On Windows a PKCS#12 identity's key is imported into the
user's key store for the life of the VM, because SChannel works outside the
process; each key is written down in `%LOCALAPPDATA%\dream\tls-keys`, so one
a killed VM left behind is deleted by the next VM to import an identity.

## Verification

`.github/workflows/portable.yml` compiles a bundle on Linux and sends those exact
artifacts to Linux, Windows, and macOS runners. Each runner verifies SHA-256
hashes before execution. The bundle covers language operations, binary I/O,
Unicode filenames, subprocess quoting, deadlines, TCP I/O with one worker, and
using the same compiler image to compile a new program on the receiving OS.
The initial matrix exercises the interpreter, with `std.ffi` built in as it
always is; the JIT, and calls into C libraries, need separate validation on
each host.

To reproduce locally:

```text
python3 dream/tests/portable.py compile --vm build/bin/dream --images build/images
python3 dream/tests/portable.py run --vm build/bin/dream --images build/images
```

Use `python` and the appropriate `.exe` path on Windows. Copy `build/images`
from the compiling machine to test another OS; do not regenerate it there.
`DREAM_PORTABLE_POLLER=ON` also exercises the non-epoll readiness backend on
Linux. Cross-compilation verifies Windows compilation/linking, but native CI
results remain necessary to establish platform support.
