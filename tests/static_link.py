#!/usr/bin/env python3
"""Exercise dependency policies with local fixtures, without installed LLVM/ffi."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def said(result, text):
    # CMake wraps a `message` to the terminal's width, so a phrase can be split
    # across lines at any space; compare with the whitespace folded.
    return " ".join(text.split()) in " ".join(result.stdout.split())


def main():
    if not all(shutil.which(t) for t in ("cmake", "cc", "ar")) or os.name == "nt":
        print("static CMake dependency tests: skipped (requires Unix C toolchain)")
        return
    with tempfile.TemporaryDirectory(prefix="dream-static-policy-") as tmp:
        root = Path(tmp)
        include = root / "include"
        include.mkdir()
        (include / "ffi.h").write_text("""
typedef struct { int value; } ffi_cif;
typedef struct { int value; } ffi_type;
extern ffi_type ffi_type_void;
#define FFI_DEFAULT_ABI 0
#define FFI_OK 0
int ffi_prep_cif(ffi_cif *, int, unsigned, ffi_type *, ffi_type **);
void ffi_call(ffi_cif *, void (*)(void), void *, void **);
""")
        (root / "ffi.c").write_text("""
#include <ffi.h>
ffi_type ffi_type_void = { 0 };
int ffi_prep_cif(ffi_cif *c, int abi, unsigned n, ffi_type *t, ffi_type **a) {
    return ffi_type_void.value;
}
void ffi_call(ffi_cif *c, void (*fn)(void), void *r, void **a) {
    ffi_type_void.value++;
}
""")
        for kind, flags in (("pic", ["-fPIC"]), ("nonpic", ["-fno-PIC", "-fno-PIE"])):
            obj = root / f"{kind}.o"
            result = run(["cc", *flags, "-I", str(include), "-c", str(root / "ffi.c"), "-o", str(obj)])
            assert result.returncode == 0, result.stdout
            result = run(["ar", "rcs", str(root / f"{kind}.a"), str(obj)])
            assert result.returncode == 0, result.stdout
        # Discovery needs OpenSSL targets, but these configure-only tests never
        # build the VM and need no real OpenSSL installation.
        modules = root / "modules"
        modules.mkdir()
        (modules / "FindOpenSSL.cmake").write_text("""
set(OPENSSL_FOUND TRUE)
set(OPENSSL_VERSION 3.0.0)
add_library(OpenSSL::SSL INTERFACE IMPORTED)
add_library(OpenSSL::Crypto INTERFACE IMPORTED)
""")
        tree = root / "build"

        def configure(archive="pic", mode="STATIC", jit=False, extra=()):
            return run(["cmake", "-S", str(ROOT), "-B", str(tree),
                        f"-DCMAKE_MODULE_PATH={modules}", f"-DDREAM_FFI_INCLUDE={include}",
                        f"-DDREAM_FFI_LIB={root / (archive + '.a')}",
                        f"-DDREAM_LINK_DEPS={mode}",
                        f"-DDREAM_ENABLE_JIT={'ON' if jit else 'OFF'}", *extra])

        result = configure()
        assert result.returncode == 0, result.stdout
        result = configure(mode="SHARED")
        assert result.returncode == 0, result.stdout
        result = configure(mode="invalid")
        assert result.returncode != 0 and said(result, "DREAM_LINK_DEPS must be"), result.stdout
        # Linux rejects text relocations in a shared object. Other Unix
        # linkers/architectures can legitimately support this fixture.
        if os.uname().sysname == "Linux":
            result = configure(archive="nonpic")
            assert result.returncode != 0 and said(result, "cannot be linked into a shared library"), result.stdout

        # The explicit CMake LLVM package must honor the policy too. A missing
        # llvm-config disables the host's automatic discovery for this fixture.
        package = root / "llvm"
        package.mkdir()
        (package / "LLVMConfig.cmake").write_text(f"""
set(LLVM_FOUND TRUE)
set(LLVM_PACKAGE_VERSION 21.0.0)
set(LLVM_INCLUDE_DIRS "{include}")
set(LLVM_LIBRARY_DIRS "{root}")
set(LLVM_ENABLE_RTTI TRUE)
add_library(LLVM STATIC IMPORTED)
set_target_properties(LLVM PROPERTIES IMPORTED_LOCATION "{root / 'pic.a'}")
add_library(LLVMCore SHARED IMPORTED)
set_target_properties(LLVMCore PROPERTIES IMPORTED_LOCATION "{root / 'pic.a'}")
function(llvm_map_components_to_libnames out)
  set(${{out}} LLVMCore PARENT_SCOPE)
endfunction()
""")
        extra = [f"-DLLVM_DIR={package}", f"-DDREAM_LLVM_CONFIG={root / 'missing-llvm-config'}"]
        result = configure(mode="STATIC", jit=True, extra=extra)
        assert result.returncode != 0 and said(result, "does not supply static component targets"), result.stdout
        result = configure(mode="SHARED", jit=True, extra=extra)
        assert result.returncode != 0 and said(result, "LLVM target is not a shared library"), result.stdout
        result = configure(mode="PREFER_STATIC", jit=True, extra=extra)
        assert result.returncode == 0, result.stdout
    print("CMake static dependency policies passed")


if __name__ == "__main__":
    main()
