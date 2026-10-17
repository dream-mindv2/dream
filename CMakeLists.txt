# dream, the Dream VM: a lazy graph-reduction interpreter with BEAM-style green
# processes, and an optional LLVM JIT tier.

cmake_minimum_required(VERSION 3.20)
project(dream CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Android is an interpreter target. Never discover a host llvm-config while
# cross-compiling: its libraries cannot be linked into an Android VM.
option(DREAM_TERMUX "Native on-device build using Termux's Android toolchain" OFF)
if(ANDROID OR DREAM_TERMUX)
  set(DREAM_JIT_DEFAULT OFF)
  if(ANDROID AND NOT CMAKE_ANDROID_ARCH_ABI MATCHES "^(arm64-v8a|x86_64)$")
    message(FATAL_ERROR "Dream Android requires arm64-v8a or x86_64")
  endif()
  set(DREAM_ANDROID_API "${CMAKE_SYSTEM_VERSION}")
  if(DEFINED ANDROID_PLATFORM_LEVEL)
    set(DREAM_ANDROID_API "${ANDROID_PLATFORM_LEVEL}")
  endif()
  if(ANDROID AND DREAM_ANDROID_API LESS 34)
    message(FATAL_ERROR "Dream Android requires API 34 or newer (posix_spawn chdir)")
  endif()
  add_link_options("-Wl,-z,max-page-size=16384")
  if(DREAM_TERMUX)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
      #if !defined(__ANDROID__) || __ANDROID_API__ < 34
      #error Dream Termux requires an Android API 34+ compiler target
      #endif
      static_assert(sizeof(void*) == 8);
      int main() { return 0; }" DREAM_TERMUX_TOOLCHAIN_OK)
    if(NOT DREAM_TERMUX_TOOLCHAIN_OK)
      message(FATAL_ERROR "Use the Termux installer with a 64-bit Android API 34+ compiler target")
    endif()
  endif()
else()
  set(DREAM_JIT_DEFAULT ON)
endif()
option(DREAM_ENABLE_JIT "Build the LLVM JIT tier" ${DREAM_JIT_DEFAULT})
if((ANDROID OR DREAM_TERMUX) AND DREAM_ENABLE_JIT)
  message(FATAL_ERROR "Dream Android currently supports the interpreter; use DREAM_ENABLE_JIT=OFF")
endif()
option(DREAM_PORTABLE_POLLER "Use the portable readiness poller even on Linux" OFF)
option(DREAM_BUILD_TESTS "Build the VM test suite" ON)

# ---------------------------------------------------------------------------
# LLVM discovery
#
# The JIT is optional: without LLVM the VM still builds and runs, interpreter
# only. `llvm-config` is preferred over LLVM's CMake package because the
# package's exported targets name transitive dependencies (ZLIB, zstd) that are
# frequently not importable on their own, and a missing one is a hard error
# rather than a graceful "not found". llvm-config just prints flags.
#
# Override either way with -DDREAM_LLVM_CONFIG=<path> or -DLLVM_DIR=<cmakedir>.
# ---------------------------------------------------------------------------
set(DREAM_HAVE_LLVM OFF)
set(DREAM_LLVM_STATIC OFF)
set(DREAM_LLVM_RTTI "NO")

# ---------------------------------------------------------------------------
# How LLVM and libffi are linked
#
# A VM that names `libLLVM.so` cannot be copied to a machine that lacks the very
# same LLVM: the soname carries the version, and distributions ship one LLVM at
# a time. The whole library is also ~130 MB to load at every start, for the
# dozen components the JIT uses. LLVM's per-component static archives fix both:
# the linker pulls only the objects something refers to, so what ends up in
# libdream is the part of LLVM the JIT reaches and nothing else. libffi is the
# same argument at a smaller size -- its soname moved from 7 to 8 within living
# memory, and a VM that refuses to start over it is a poor thing to ship.
#
#   PREFER_STATIC  static archives where they exist, the shared library where
#                  not; the configure says which it got. The default, so an
#                  ordinary `just vm` is already the shippable build.
#   STATIC         the archives or a failed configure. For a release, where
#                  a quiet fallback would ship the very dependency being avoided.
#   SHARED         the shared libraries, as before.
#
# libdream is itself a shared library, so the archives have to be position-
# independent: LLVM's are (it builds with LLVM_ENABLE_PIC), libffi's only when
# configured `--with-pic`. shell.nix and nix/dream-vm.nix build one that is.
# ---------------------------------------------------------------------------
set(DREAM_LINK_DEPS "PREFER_STATIC" CACHE STRING
    "How to link LLVM and libffi: PREFER_STATIC, STATIC or SHARED")
set_property(CACHE DREAM_LINK_DEPS PROPERTY STRINGS PREFER_STATIC STATIC SHARED)
if(NOT DREAM_LINK_DEPS MATCHES "^(PREFER_STATIC|STATIC|SHARED)$")
  message(FATAL_ERROR "DREAM_LINK_DEPS must be PREFER_STATIC, STATIC or SHARED")
endif()

# The JIT's LLVM components. Over-asking costs nothing when linking archives --
# an object nothing refers to is never pulled in -- so this is the list of what
# jit.cpp may touch, not a measurement. build.dr's `llvm_parts` is the same list.
set(DREAM_LLVM_COMPONENTS core orcjit native support target analysis passes ipo
                          instcombine scalaropts transformutils)

# `--system-libs` is the link line LLVM *itself* was built with, so it names the
# dependencies of the whole installation rather than of the components above.
# Exactly one of them is dead weight here: libxml2 is for WindowsManifest, which
# none of `DREAM_LLVM_COMPONENTS` links, and asking for it would make libxml2 a
# build dependency of a VM that cannot reach the code that uses it. It is
# excluded by name rather than by rule because it is the only such library --
# every other entry below is one a component may genuinely need.
set(DREAM_LLVM_SYSLIB_EXCLUDED xml2)

# Split an llvm-config `--system-libs` string into a link-line list, and report
# separately the names on it this machine cannot supply.
#
# Whether an unresolvable `-l` is fatal is the caller's decision -- for the
# archives it is, because they were built against it -- so this only reports.
#
# Each name is asked of the loader's own search path rather than assumed, since
# a `-l` that cannot be found stops the link even when nothing refers to it, and
# `--as-needed` does not save it: it declines to write a DT_NEEDED entry, but the
# search still happens and still fails.
function(dream_llvm_syslibs raw out_libs out_missing)
  separate_arguments(tokens NATIVE_COMMAND "${raw}")

  set(libs)
  set(dirs)
  set(missing)

  foreach(token IN LISTS tokens)
    if(token MATCHES "^-L(.+)$")
      # Kept on the link line as written, and remembered so the names after it
      # are looked for where the linker will look, not only where CMake's own
      # search path reaches.
      list(APPEND libs "${token}")
      list(APPEND dirs "${CMAKE_MATCH_1}")

    elseif(token MATCHES "^-l(.+)$")
      if("${CMAKE_MATCH_1}" IN_LIST DREAM_LLVM_SYSLIB_EXCLUDED)
        continue()
      endif()

      string(MAKE_C_IDENTIFIER "DREAM_LLVM_SYSLIB_${CMAKE_MATCH_1}" probe_var)
      find_library(${probe_var} NAMES ${CMAKE_MATCH_1} HINTS ${dirs})

      if(${probe_var})
        # The resolved path, not `-l${name}`. Having asked where the library is,
        # forgetting it would put the linker's own search path back in charge,
        # and that is the search that was failing: CMake reads NIX_LDFLAGS and
        # CMAKE_PREFIX_PATH, so on Nix it finds a store path ld is never told
        # about -- which is how a name CMake had just resolved came back as
        # `cannot find -l` one step later.
        list(APPEND libs "${${probe_var}}")
      else()
        list(APPEND missing "-l${CMAKE_MATCH_1}")
      endif()

    else()
      # An absolute path, or a flag llvm-config prints we pass straight on.
      list(APPEND libs "${token}")
    endif()
  endforeach()

  set(${out_libs} "${libs}" PARENT_SCOPE)
  set(${out_missing} "${missing}" PARENT_SCOPE)
endfunction()

if(DREAM_ENABLE_JIT)
  if(NOT DREAM_LLVM_CONFIG)
    find_program(DREAM_LLVM_CONFIG
      NAMES llvm-config llvm-config-21 llvm-config-20 llvm-config-19 llvm-config-18
            llvm-config-17 llvm-config-16 llvm-config-15)
  endif()

  if(NOT DREAM_LLVM_CONFIG AND UNIX AND NOT CMAKE_CROSSCOMPILING)
    # Nix keeps llvm-config in the -dev output, which is usually not in PATH.
    file(GLOB DREAM_NIX_LLVM_CONFIGS "/nix/store/*llvm-*-dev/bin/llvm-config")
    if(DREAM_NIX_LLVM_CONFIGS)
      list(SORT DREAM_NIX_LLVM_CONFIGS)
      list(REVERSE DREAM_NIX_LLVM_CONFIGS)
      list(GET DREAM_NIX_LLVM_CONFIGS 0 DREAM_LLVM_CONFIG)
    endif()
  endif()

  if(DREAM_LLVM_CONFIG)
    execute_process(COMMAND ${DREAM_LLVM_CONFIG} --version
                    OUTPUT_VARIABLE DREAM_LLVM_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(DREAM_LLVM_VERSION VERSION_LESS 15)
      message(WARNING "Dream: llvm-config reports ${DREAM_LLVM_VERSION}; the JIT needs 15 or newer")
    else()
      execute_process(COMMAND ${DREAM_LLVM_CONFIG} --includedir
                      OUTPUT_VARIABLE DREAM_LLVM_INCLUDEDIR
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      execute_process(COMMAND ${DREAM_LLVM_CONFIG} --libdir
                      OUTPUT_VARIABLE DREAM_LLVM_LIBDIR
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

      # `--link-static` fails outright when the archives are not installed
      # (Debian's llvm-dev ships them, some distributions do not), which is
      # what falling back on the shared library is decided by.
      set(DREAM_LLVM_STATIC OFF)
      if(NOT DREAM_LINK_DEPS STREQUAL "SHARED")
        execute_process(
          COMMAND ${DREAM_LLVM_CONFIG}
                  --link-static --libs ${DREAM_LLVM_COMPONENTS}
          RESULT_VARIABLE DREAM_LLVM_STATIC_RC
          OUTPUT_VARIABLE DREAM_LLVM_LIBS_RAW
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)

        if(DREAM_LLVM_STATIC_RC EQUAL 0 AND DREAM_LLVM_LIBS_RAW)
          execute_process(
            COMMAND ${DREAM_LLVM_CONFIG} --link-static --system-libs
            RESULT_VARIABLE DREAM_LLVM_SYSTEM_RC
            OUTPUT_VARIABLE DREAM_LLVM_SYSLIBS_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)

          dream_llvm_syslibs("${DREAM_LLVM_SYSLIBS_RAW}"
                             DREAM_LLVM_SYSLIBS DREAM_LLVM_SYSLIB_MISSING)

          # A missing name is not a library to drop. The archives were built
          # against these, and libdream being a shared object means an
          # unresolved one does not fail here: it is carried out of this link
          # as an undefined symbol and stops the *next* one, as a wall of
          # `crc32`/`compress2` with nothing to say which archive asked. So the
          # static configuration is given up on here, where the reason can
          # still be printed and acted on, and PREFER_STATIC takes the shared
          # library that satisfies the same dependencies from its own
          # DT_NEEDED rather than from the command line.
          if(NOT DREAM_LLVM_SYSTEM_RC EQUAL 0)
            list(APPEND DREAM_LLVM_SYSLIB_MISSING "llvm-config --system-libs failed")
          endif()

          if(DREAM_LLVM_SYSLIB_MISSING)
            string(REPLACE ";" " " DREAM_LLVM_SYSLIB_MISSING_TEXT
                           "${DREAM_LLVM_SYSLIB_MISSING}")

            if(DREAM_LINK_DEPS STREQUAL "STATIC")
              message(
                FATAL_ERROR
                "Dream: DREAM_LINK_DEPS=STATIC, but LLVM's static archives "
                "were built against ${DREAM_LLVM_SYSLIB_MISSING_TEXT}, which is "
                "not installed here, so they cannot be linked. Install LLVM's "
                "own build dependencies, or name them with "
                "-DCMAKE_LIBRARY_PATH=<dir>.")
            endif()

            message(
              STATUS
              "Dream: LLVM's static archives were built against "
              "${DREAM_LLVM_SYSLIB_MISSING_TEXT}, which is not installed here; "
              "linking the shared library instead.")
          else()
            set(DREAM_LLVM_STATIC ON)
          endif()
        elseif(DREAM_LINK_DEPS STREQUAL "STATIC")
          message(
            FATAL_ERROR
            "Dream: DREAM_LINK_DEPS=STATIC, but ${DREAM_LLVM_CONFIG} "
            "has no static archives for the JIT's components.")
        endif()
      endif()

      if(NOT DREAM_LLVM_STATIC)
        execute_process(
          COMMAND ${DREAM_LLVM_CONFIG} --link-shared --libs ${DREAM_LLVM_COMPONENTS}
          RESULT_VARIABLE DREAM_LLVM_SHARED_RC
          OUTPUT_VARIABLE DREAM_LLVM_LIBS_RAW
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)

        execute_process(
          COMMAND ${DREAM_LLVM_CONFIG} --link-shared --system-libs
          OUTPUT_VARIABLE DREAM_LLVM_SYSLIBS_RAW
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)

        if(NOT DREAM_LLVM_SHARED_RC EQUAL 0 OR NOT DREAM_LLVM_LIBS_RAW)
          message(FATAL_ERROR "Dream: llvm-config cannot supply shared LLVM libraries; choose STATIC or install shared LLVM")
        endif()

        # Usually an empty list, since libLLVM.so records its own DT_NEEDED.
        # Whatever is on it is filtered by the same rule for the same reason: an
        # unresolvable `-l` stops the link whether or not a component wanted it.
        dream_llvm_syslibs("${DREAM_LLVM_SYSLIBS_RAW}"
                           DREAM_LLVM_SYSLIBS DREAM_LLVM_SYSLIB_MISSING)

        if(DREAM_LLVM_SYSLIB_MISSING)
          string(REPLACE ";" " " DREAM_LLVM_SYSLIB_MISSING_TEXT
                         "${DREAM_LLVM_SYSLIB_MISSING}")

          message(
            STATUS
            "Dream: LLVM asked for ${DREAM_LLVM_SYSLIB_MISSING_TEXT}, which is "
            "not installed here; leaving it to libLLVM.so's own dependencies.")
        endif()
      endif()

      execute_process(
        COMMAND ${DREAM_LLVM_CONFIG} --has-rtti
        OUTPUT_VARIABLE DREAM_LLVM_RTTI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)

      separate_arguments(
        DREAM_LLVM_LIBS NATIVE_COMMAND "${DREAM_LLVM_LIBS_RAW}")

      set(DREAM_HAVE_LLVM ON)

      if(DREAM_LLVM_STATIC)
        set(DREAM_LLVM_HOW "static components")
      else()
        set(DREAM_LLVM_HOW "shared library")
      endif()

      message(
        STATUS
        "Dream: JIT enabled against LLVM ${DREAM_LLVM_VERSION}, "
        "${DREAM_LLVM_HOW} (${DREAM_LLVM_CONFIG})")
    endif()
  endif()

  if(NOT DREAM_HAVE_LLVM AND LLVM_DIR)
    # Explicit CMake package, for platforms where that is the better route.
    find_package(LLVM CONFIG QUIET)

    if(LLVM_FOUND)
      if(DREAM_LINK_DEPS STREQUAL "SHARED")
        if(NOT TARGET LLVM)
          message(FATAL_ERROR "Dream: LLVM_DIR has no shared LLVM target; use llvm-config or another LLVM installation")
        endif()
        get_target_property(DREAM_LLVM_TARGET_TYPE LLVM TYPE)
        if(NOT DREAM_LLVM_TARGET_TYPE STREQUAL "SHARED_LIBRARY")
          message(FATAL_ERROR "Dream: LLVM_DIR's LLVM target is not a shared library")
        endif()
        set(DREAM_LLVM_LIBS LLVM)
      else()
        llvm_map_components_to_libnames(DREAM_LLVM_LIBS ${DREAM_LLVM_COMPONENTS})
        set(DREAM_LLVM_STATIC ON)
        foreach(component IN LISTS DREAM_LLVM_LIBS)
          if(TARGET ${component})
            get_target_property(component_type ${component} TYPE)
            if(NOT component_type STREQUAL "STATIC_LIBRARY")
              set(DREAM_LLVM_STATIC OFF)
            endif()
          else()
            set(DREAM_LLVM_STATIC OFF)
          endif()
        endforeach()
        if(DREAM_LINK_DEPS STREQUAL "STATIC" AND NOT DREAM_LLVM_STATIC)
          message(FATAL_ERROR "Dream: LLVM_DIR does not supply static component targets")
        endif()
      endif()

      set(DREAM_LLVM_INCLUDEDIR ${LLVM_INCLUDE_DIRS})
      set(DREAM_LLVM_LIBDIR ${LLVM_LIBRARY_DIRS})

      if(LLVM_ENABLE_RTTI)
        set(DREAM_LLVM_RTTI "YES")
      endif()

      set(DREAM_HAVE_LLVM ON)

      message(
        STATUS
        "Dream: JIT enabled against LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
    endif()
  endif()

  if(NOT DREAM_HAVE_LLVM)
    message(
      WARNING
      "Dream: LLVM not found; building interpreter-only. "
      "Pass -DDREAM_LLVM_CONFIG=<path to llvm-config> to enable the JIT.")
  endif()
endif()

# ---------------------------------------------------------------------------
# libffi, for std.ffi
#
# Required, unlike LLVM. `std.ffi` is part of what a VM is: `std.foreign` and
# every `foreign` declaration compile to calls into it, so a VM that cannot
# reach C would run some valid images and fail others at the call site. There
# is no such build; without libffi the configure stops here.
#
# Static libffi deserves more care than merely finding libffi.a. Distribution
# archives are not necessarily built with -fPIC, and on Nix a store glob may
# also find an archive belonging to another libc/toolchain. Since libdream is a
# shared library, an automatically discovered static candidate is accepted only
# after proving that it can itself be linked into a shared library.
# ---------------------------------------------------------------------------

# An explicitly supplied value wins over all automatic library discovery.
if(DREAM_FFI_LIB)
  set(DREAM_FFI_LIB_EXPLICIT ON)
else()
  set(DREAM_FFI_LIB_EXPLICIT OFF)
endif()

find_path(DREAM_FFI_INCLUDE ffi.h PATH_SUFFIXES ffi)

# Nix normally keeps ffi.h in libffi's -dev output, which is not necessarily on
# CMake's default include search path. Resolve the header before testing static
# archives because the test program itself includes ffi.h.
if(UNIX AND NOT CMAKE_CROSSCOMPILING AND NOT DREAM_FFI_INCLUDE)
  file(GLOB DREAM_NIX_FFI_INC "/nix/store/*libffi-*-dev/include")

  if(DREAM_NIX_FFI_INC)
    list(SORT DREAM_NIX_FFI_INC)
    list(REVERSE DREAM_NIX_FFI_INC)
    list(GET DREAM_NIX_FFI_INC 0 DREAM_FFI_INCLUDE)
  endif()
endif()

# Test the property we actually need: whether this archive can become part of
# a shared object. A normal CMake compile check links an executable, usually a
# PIE on Linux. An archive built with -fPIE may work there while still being
# unusable in libdream.so.
#
# Therefore this creates a tiny nested project whose output is genuinely
# SHARED. ffi_call is referenced deliberately so the architecture-dependent
# call machinery is pulled out of libffi.a rather than allowing the linker to
# succeed without extracting meaningful objects from the archive.
function(dream_ffi_archive_works archive include_dir result_var)
  set(probe_src_dir
      "${CMAKE_BINARY_DIR}/CMakeFiles/dream-ffi-probe-src")
  set(probe_bin_dir
      "${CMAKE_BINARY_DIR}/CMakeFiles/dream-ffi-probe-build")

  file(MAKE_DIRECTORY "${probe_src_dir}")

  file(
    WRITE
    "${probe_src_dir}/CMakeLists.txt"
[=[
cmake_minimum_required(VERSION 3.20)
project(dream_ffi_probe C)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(dream_ffi_probe SHARED probe.c)
if(WIN32)
  target_compile_definitions(dream_ffi_probe PRIVATE FFI_STATIC_BUILD)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(dream_ffi_probe PRIVATE "-Wl,-z,defs")
endif()

target_include_directories(
  dream_ffi_probe
  PRIVATE
  "${DREAM_FFI_PROBE_INCLUDE}")

target_link_libraries(
  dream_ffi_probe
  PRIVATE
  "${DREAM_FFI_PROBE_ARCHIVE}")
]=])

  file(
    WRITE
    "${probe_src_dir}/probe.c"
[=[
#include <ffi.h>

static void dream_ffi_target(void)
{
}

void dream_ffi_probe(void)
{
    ffi_cif cif;

    if (ffi_prep_cif(
            &cif,
            FFI_DEFAULT_ABI,
            0,
            &ffi_type_void,
            0) != FFI_OK) {
        return;
    }

    ffi_call(
        &cif,
        (void (*)(void))dream_ffi_target,
        0,
        0);
}
]=])

  # Reuse the directory, but clear CMake's result from a previous candidate so
  # every archive really gets tested.
  unset(archive_works CACHE)

  # Whole-project try_compile on CMake 3.20 does not propagate the NDK's ABI
  # and API settings. Without these the probe silently targets 32-bit ARM.
  set(probe_platform_flags)
  if(ANDROID)
    list(APPEND probe_platform_flags
      "-DANDROID_ABI:STRING=${CMAKE_ANDROID_ARCH_ABI}"
      "-DANDROID_PLATFORM:STRING=android-${DREAM_ANDROID_API}"
      "-DANDROID_STL:STRING=${ANDROID_STL}")
  endif()

  try_compile(
    archive_works
    "${probe_bin_dir}"
    "${probe_src_dir}"
    dream_ffi_probe
    dream_ffi_probe
    CMAKE_FLAGS
      ${probe_platform_flags}
      "-DDREAM_FFI_PROBE_ARCHIVE:FILEPATH=${archive}"
      "-DDREAM_FFI_PROBE_INCLUDE:PATH=${include_dir}"
    OUTPUT_VARIABLE probe_output)

  set(${result_var} "${archive_works}" PARENT_SCOPE)

  if(NOT archive_works)
    message(
      VERBOSE
      "Dream: rejecting static libffi candidate ${archive}:\n"
      "${probe_output}")
  endif()
endfunction()

set(DREAM_FFI_STATIC OFF)

# ---------------------------------------------------------------------------
# Explicit library
# ---------------------------------------------------------------------------
if(DREAM_FFI_LIB_EXPLICIT)

  if((DREAM_FFI_LIB MATCHES "\\.a$" OR DREAM_FFI_LIB MATCHES "\\.lib$")
     AND NOT DREAM_FFI_LIB MATCHES "\\.dll\\.a$")

    if(NOT DREAM_FFI_INCLUDE)
      message(
        FATAL_ERROR
        "Dream: ${DREAM_FFI_LIB} was supplied as libffi, but ffi.h "
        "could not be found. Point "
        "-DDREAM_FFI_INCLUDE=<dir with ffi.h> at the matching headers.")
    endif()

    dream_ffi_archive_works(
      "${DREAM_FFI_LIB}"
      "${DREAM_FFI_INCLUDE}"
      DREAM_FFI_EXPLICIT_ARCHIVE_WORKS)

    if(NOT DREAM_FFI_EXPLICIT_ARCHIVE_WORKS)
      message(
        FATAL_ERROR
        "Dream: ${DREAM_FFI_LIB} cannot be linked into a shared library. "
        "It may not have been built with -fPIC, or it may belong to a "
        "different toolchain/ABI.")
    endif()

    set(DREAM_FFI_STATIC ON)

  elseif(DREAM_LINK_DEPS STREQUAL "STATIC")
    message(
      FATAL_ERROR
      "Dream: DREAM_LINK_DEPS=STATIC, but the explicitly supplied "
      "DREAM_FFI_LIB (${DREAM_FFI_LIB}) is not a static archive.")
  endif()

# ---------------------------------------------------------------------------
# Automatic library discovery
# ---------------------------------------------------------------------------
else()

  # First perform the normal lookup. PREFER_STATIC retains this as the safe
  # fallback while static candidates are investigated.
  find_library(DREAM_FFI_NORMAL_LIB NAMES ffi)

  if(DREAM_FFI_NORMAL_LIB)
    set(DREAM_FFI_LIB "${DREAM_FFI_NORMAL_LIB}")
  endif()

  if(NOT DREAM_LINK_DEPS STREQUAL "SHARED")
    set(DREAM_FFI_STATIC_CANDIDATES)

    # Ask specifically for archive names rather than "ffi": otherwise the
    # platform's normal library suffix order tends to select libffi.so.
    find_library(
      DREAM_FFI_STATIC_SEARCH_RESULT
      NAMES libffi.a ffi_static libffi_static)

    if(DREAM_FFI_STATIC_SEARCH_RESULT)
      list(
        APPEND DREAM_FFI_STATIC_CANDIDATES
        "${DREAM_FFI_STATIC_SEARCH_RESULT}")
    endif()

    if(UNIX AND NOT CMAKE_CROSSCOMPILING)
      # The Nix store is deliberately not an ordinary library search path.
      # There may be several libffi archives in it, and lexical order does not
      # tell us which one belongs to this compiler/libc. Try candidates until
      # one proves that it can participate in a shared-library link.
      file(
        GLOB DREAM_NIX_FFI_STATIC
        "/nix/store/*libffi-*/lib/libffi.a")

      if(DREAM_NIX_FFI_STATIC)
        list(SORT DREAM_NIX_FFI_STATIC)
        list(REVERSE DREAM_NIX_FFI_STATIC)

        list(
          APPEND DREAM_FFI_STATIC_CANDIDATES
          ${DREAM_NIX_FFI_STATIC})
      endif()
    endif()

    if(DREAM_FFI_STATIC_CANDIDATES)
      list(REMOVE_DUPLICATES DREAM_FFI_STATIC_CANDIDATES)

      foreach(candidate IN LISTS DREAM_FFI_STATIC_CANDIDATES)
        if(NOT candidate MATCHES "\\.(a|lib)$" OR candidate MATCHES "\\.dll\\.a$")
          continue()
        endif()
        if(NOT DREAM_FFI_INCLUDE)
          break()
        endif()

        dream_ffi_archive_works(
          "${candidate}"
          "${DREAM_FFI_INCLUDE}"
          DREAM_FFI_CANDIDATE_WORKS)

        if(DREAM_FFI_CANDIDATE_WORKS)
          set(DREAM_FFI_LIB "${candidate}")
          set(DREAM_FFI_STATIC_LIB "${candidate}")
          set(DREAM_FFI_STATIC ON)
          break()
        endif()
      endforeach()
    endif()

    if(NOT DREAM_FFI_STATIC
       AND DREAM_LINK_DEPS STREQUAL "STATIC")
      message(
        FATAL_ERROR
        "Dream: DREAM_LINK_DEPS=STATIC, but no usable PIC libffi.a "
        "was found. Build libffi with --enable-static --with-pic, or "
        "point -DDREAM_FFI_LIB=<path to a PIC libffi.a> at one.")
    endif()
  endif()

  # find_library() may still fail on Nix because /nix/store is not a normal
  # system library path. Only use this shared-library fallback after normal
  # lookup and static probing have both failed to provide a library.
  if(UNIX
     AND NOT CMAKE_CROSSCOMPILING
     AND NOT DREAM_FFI_LIB
     AND NOT DREAM_LINK_DEPS STREQUAL "STATIC")

    file(
      GLOB DREAM_NIX_FFI_LIB
      "/nix/store/*libffi-*/lib/libffi.so")

    if(DREAM_NIX_FFI_LIB)
      list(SORT DREAM_NIX_FFI_LIB)
      list(REVERSE DREAM_NIX_FFI_LIB)
      list(GET DREAM_NIX_FFI_LIB 0 DREAM_FFI_LIB)
    endif()
  endif()
endif()

if(NOT DREAM_FFI_INCLUDE OR NOT DREAM_FFI_LIB)
  message(
    FATAL_ERROR
    "Dream: libffi not found. std.ffi is a required part of the VM; "
    "install libffi, or point "
    "-DDREAM_FFI_INCLUDE=<dir with ffi.h> and "
    "-DDREAM_FFI_LIB=<path to libffi> at it.")
endif()

if(DREAM_FFI_STATIC)
  message(
    STATUS
    "Dream: FFI enabled, static (${DREAM_FFI_LIB})")
else()
  message(
    STATUS
    "Dream: FFI enabled, shared library (${DREAM_FFI_LIB})")
endif()

# ---------------------------------------------------------------------------
# TLS, for std.tls
#
# Required, for the reason libffi is: `std.tls` is part of what a VM is, and a
# program that speaks TLS must run on every VM, so there is no build without
# it. The library is the platform's choice and invisible to Dream: OpenSSL 3
# on Linux and macOS, linked here; SChannel on Windows, which is part of the
# system. See src/tls.hpp.
# ---------------------------------------------------------------------------
if(WIN32)
  set(DREAM_TLS_SOURCE src/tls_schannel.cpp)
  message(STATUS "Dream: TLS through SChannel")
else()
  if(APPLE AND NOT OPENSSL_ROOT_DIR)
    # Homebrew's OpenSSL is keg-only, so it is not on the default search path.
    execute_process(
      COMMAND brew --prefix openssl@3
      OUTPUT_VARIABLE DREAM_BREW_OPENSSL
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)

    if(DREAM_BREW_OPENSSL)
      set(OPENSSL_ROOT_DIR "${DREAM_BREW_OPENSSL}")
    endif()
  endif()

  find_package(OpenSSL 3.0)

  if(NOT OPENSSL_FOUND)
    message(
      FATAL_ERROR
      "Dream: OpenSSL 3 not found. std.tls is a required part of the VM; "
      "install OpenSSL's development files (libssl-dev, openssl@3), or "
      "point -DOPENSSL_ROOT_DIR=<prefix> at them.")
  endif()

  set(DREAM_TLS_SOURCE src/tls_openssl.cpp)

  message(
    STATUS
    "Dream: TLS through OpenSSL ${OPENSSL_VERSION}")
endif()

# ---------------------------------------------------------------------------
# The VM library
# ---------------------------------------------------------------------------
set(DREAM_SOURCES
  src/value.cpp
  src/image.cpp
  src/heap.cpp
  src/cores.cpp
  src/gc_pool.cpp
  src/process.cpp
  src/interp.cpp
  src/builtins.cpp
  src/ffi.cpp
  src/io.cpp
  src/digest.cpp
  src/crypto.cpp
  src/regex.cpp
  src/bigint.cpp
  src/os.cpp
  src/scheduler.cpp
  src/runtime.cpp
  src/capi.cpp
  src/tensor.cpp
  src/tensor_kernels.cpp
  src/gpu.cpp
  src/tls.cpp
  ${DREAM_TLS_SOURCE}
)

# The tensor kernels a second time, for CPUs with AVX2 and FMA, chosen at run
# time by `tensor_kernels()`. A second translation unit rather than
# `target_clones` because the matrix product's inner kernel has to be inlined
# into its loops to be fast, and GCC will not inline across instruction sets --
# the head of src/tensor_kernels.inc has the measurement. Only where the
# compiler can target x86-64 with GCC-style flags; elsewhere the baseline table
# is the only one.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT MSVC)
  list(APPEND DREAM_SOURCES src/tensor_kernels_avx2.cpp)

  set_source_files_properties(
    src/tensor_kernels_avx2.cpp
    PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")

  set(DREAM_TENSOR_AVX2 ON)
endif()

if(DREAM_HAVE_LLVM)
  list(APPEND DREAM_SOURCES src/jit.cpp src/jit_rt.cpp)
else()
  list(APPEND DREAM_SOURCES src/jit_stub.cpp src/jit_rt.cpp)
endif()

add_library(dream SHARED ${DREAM_SOURCES})
if(DREAM_TERMUX)
  target_compile_definitions(dream PRIVATE DREAM_TERMUX=1)
endif()

set_target_properties(
  dream
  PROPERTIES
  OUTPUT_NAME dream
  VERSION 0.1
  SOVERSION 0
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(
  dream
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(DREAM_TENSOR_AVX2)
  target_compile_definitions(
    dream
    PRIVATE
    DREAM_TENSOR_AVX2=1)
endif()

if(DREAM_PORTABLE_POLLER)
  target_compile_definitions(
    dream
    PRIVATE
    DREAM_PORTABLE_POLLER=1)
endif()

find_package(Threads REQUIRED)

target_link_libraries(
  dream
  PUBLIC
  Threads::Threads)

target_link_libraries(
  dream
  PRIVATE
  ${CMAKE_DL_LIBS})

target_include_directories(
  dream
  SYSTEM
  PRIVATE
  ${DREAM_FFI_INCLUDE})

target_link_libraries(
  dream
  PRIVATE
  ${DREAM_FFI_LIB})

if(DREAM_FFI_STATIC AND WIN32)
  # ffi.h marks its functions dllimport unless told the library is static.
  target_compile_definitions(
    dream
    PRIVATE
    FFI_STATIC_BUILD)
endif()

# What an archive brings into libdream stays inside it. Without this every
# LLVM and libffi symbol pulled in is exported again from libdream.so, which
# bloats its dynamic symbol table by tens of thousands of names and lets a
# program that embeds the VM and has its own LLVM bind one LLVM's calls to the
# other's code. Nothing needs them visible: the JIT hands its generated code
# the runtime's addresses as absolute symbols (`jd.define` in jit.cpp), not by
# looking names up in the process. `--as-needed` then drops a system library
# the static LLVM names but nothing pulled in uses -- libxml2, for the Windows
# manifest code the JIT never reaches.
if((DREAM_LLVM_STATIC OR DREAM_FFI_STATIC) AND NOT MSVC AND NOT APPLE)
  include(CheckLinkerFlag)

  check_linker_flag(
    CXX
    "-Wl,--exclude-libs,ALL"
    DREAM_HAVE_EXCLUDE_LIBS)

  if(DREAM_HAVE_EXCLUDE_LIBS)
    target_link_options(
      dream
      PRIVATE
      "-Wl,--exclude-libs,ALL"
      "-Wl,--as-needed"
      "-Wl,--gc-sections")
  endif()
endif()

if(MSVC)
  # C4100 is unused-parameter, which the GCC line below turns off too: a
  # native's signature is fixed by the table whether it reads `p` or not.
  # _CRT_SECURE_NO_WARNINGS silences MSVC's deprecation of `getenv`,
  # `strerror` and `_wopen` in favour of its own `_s` variants; the calls are
  # the portable ones every other platform makes, and a warning is a failure.
  target_compile_options(
    dream
    PRIVATE
    /W4
    /permissive-
    /utf-8
    /wd4100)

  target_compile_definitions(
    dream
    PRIVATE
    _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(
    dream
    PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter)
endif()

if(WIN32)
  target_compile_definitions(
    dream
    PUBLIC
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _WIN32_WINNT=0x0601)

  target_link_libraries(
    dream
    PRIVATE
    ws2_32
    secur32
    crypt32
    ncrypt
    bcrypt)
else()
  target_link_libraries(
    dream
    PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto)
endif()

# ---------------------------------------------------------------------------
# Codegen for the interpreter
#
# The VM is a shared library, and that costs more than it looks. By default a
# compiler must assume any global function in a `.so` can be interposed at load
# time by a definition from somewhere else, so every cross-TU call inside the
# library goes through the PLT and none of them can be inlined -- and the
# interpreter is nothing but cross-TU calls on its hot path (`force_whnf`,
# `well_known`, `builtin_def`, every `Heap::` constructor). Nothing here is
# meant to be interposed, so say so: `-fno-semantic-interposition` lets the
# calls bind directly and inline.
#
# `-O3` over the `-O2` a RelWithDebInfo build would pick, because the reduction
# loop is exactly the shape -O3's extra inlining and unswitching are for, and
# `just` builds RelWithDebInfo -- so the default build is the one that has to be
# fast.
# ---------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

foreach(flag -fno-semantic-interposition -fno-plt)
  string(
    MAKE_C_IDENTIFIER
    "HAVE${flag}"
    flag_var)

  check_cxx_compiler_flag(
    ${flag}
    ${flag_var})

  if(${flag_var})
    target_compile_options(
      dream
      PRIVATE
      ${flag})
  endif()
endforeach()

if(NOT MSVC)
  target_compile_options(
    dream
    PRIVATE
    $<$<CONFIG:RelWithDebInfo>:-O3>
    $<$<CONFIG:Release>:-O3>)
endif()

# The inliner's default ceilings are wrong for this interpreter, and knowing
# why is worth more than the numbers.
#
# `step_eval` is one switch over every opcode, so it is enormous -- far past
# `large-function-insns`, after which GCC will not grow a function further.
# Everything it calls is therefore outlined, however small: pushing a
# continuation is a compare and two stores, and it was **6% of a self-compile**
# as an out-of-line call. It is not that the helpers are expensive; it is that
# the one function they serve is too big to be allowed to absorb them, so the
# hot path pays a call per operand read, per continuation push, per operand
# decode. Raising the ceilings is worth about 5% on a self-compile, measured
# against the same source built without them, interleaved, seven rounds.
#
# The one helper that still needed `always_inline` even with these is
# `ContStack::push_back` -- see the note on it in process.hpp.
#
# The `SHELL:` prefix is load-bearing: without it CMake sees five options that
# all begin `--param`, deduplicates them down to one, and passes the other four
# values as bare words -- which the driver reports as unused linker inputs, one
# warning per file per param. Twelve files, four params, forty-eight warnings,
# and the flags not applied.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  foreach(
    param
    large-function-growth=400
    large-function-insns=10000
    inline-unit-growth=120
    max-inline-insns-auto=120
    max-inline-insns-single=800)

    target_compile_options(
      dream
      PRIVATE
      "SHELL:--param ${param}")
  endforeach()
endif()

# Train and rebuild in the same build directory. Instrument only the VM;
# ordinary builds have no profiling counters or profile-file dependency.
set(
  DREAM_PGO
  "OFF"
  CACHE STRING
  "GCC profile-guided optimization: OFF, GENERATE, USE")

set_property(
  CACHE DREAM_PGO
  PROPERTY STRINGS
  OFF GENERATE USE)

set(
  DREAM_PGO_DIR
  "${CMAKE_BINARY_DIR}/pgo"
  CACHE PATH
  "GCC runtime profile directory")

if(NOT DREAM_PGO STREQUAL "OFF")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(
      FATAL_ERROR
      "DREAM_PGO requires GCC")
  endif()

  if(DREAM_PGO STREQUAL "GENERATE")
    target_compile_options(
      dream
      PRIVATE
      "-fprofile-generate=${DREAM_PGO_DIR}"
      -fprofile-update=atomic)

    target_link_options(
      dream
      PRIVATE
      "-fprofile-generate=${DREAM_PGO_DIR}")

    target_compile_definitions(
      dream
      PRIVATE
      DREAM_PROFILE_GENERATE=1)

  elseif(DREAM_PGO STREQUAL "USE")
    target_compile_options(
      dream
      PRIVATE
      "-fprofile-use=${DREAM_PGO_DIR}"
      -Werror=missing-profile)

  else()
    message(
      FATAL_ERROR
      "DREAM_PGO must be OFF, GENERATE or USE")
  endif()
endif()

if(DREAM_HAVE_LLVM)
  target_compile_definitions(
    dream
    PRIVATE
    DREAM_HAVE_LLVM=1
    __STDC_CONSTANT_MACROS
    __STDC_FORMAT_MACROS
    __STDC_LIMIT_MACROS)

  target_include_directories(
    dream
    SYSTEM
    PRIVATE
    ${DREAM_LLVM_INCLUDEDIR})

  target_link_directories(
    dream
    PRIVATE
    ${DREAM_LLVM_LIBDIR})

  target_link_libraries(
    dream
    PRIVATE
    ${DREAM_LLVM_LIBS}
    ${DREAM_LLVM_SYSLIBS})

  # LLVM is normally built without RTTI; matching it avoids undefined vtable
  # references when we inherit from or dynamic_cast its types.
  if(NOT DREAM_LLVM_RTTI STREQUAL "YES")
    if(MSVC)
      set_source_files_properties(
        src/jit.cpp
        PROPERTIES
        COMPILE_OPTIONS "/GR-")
    else()
      set_source_files_properties(
        src/jit.cpp
        PROPERTIES
        COMPILE_OPTIONS "-fno-rtti")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# The `dream` executable
# ---------------------------------------------------------------------------
add_executable(
  dream_cli
  src/cli.cpp)

set_target_properties(
  dream_cli
  PROPERTIES
  OUTPUT_NAME dream)

target_link_libraries(
  dream_cli
  PRIVATE
  dream)

if(WIN32)
  target_link_libraries(
    dream_cli
    PRIVATE
    shell32)
endif()

target_include_directories(
  dream_cli
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DREAM_BUILD_TESTS)
  enable_testing()

  add_executable(
    dream_tests
    tests/test_main.cpp)

  target_link_libraries(
    dream_tests
    PRIVATE
    dream)

  target_include_directories(
    dream_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  if(DREAM_TENSOR_AVX2)
    target_compile_definitions(
      dream_tests
      PRIVATE
      DREAM_TENSOR_AVX2=1)
  endif()

  add_test(
    NAME dream_tests
    COMMAND dream_tests)

  if(UNIX AND NOT CMAKE_CROSSCOMPILING)
    # End-to-end tests need the compiler, so they are skipped when it is absent
    # rather than failing a VM-only build.
    add_test(
      NAME dream_e2e
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e.sh)

    set_tests_properties(
      dream_e2e
      PROPERTIES
      ENVIRONMENT "dream=$<TARGET_FILE:dream_cli>"
      SKIP_RETURN_CODE 1)

    # A malformed image must be rejected, never crashed on. Kept short here;
    # run `just fuzz 5000` for a real sweep.
    add_test(
      NAME dream_fuzz_image
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/fuzz_image.sh)

    set_tests_properties(
      dream_fuzz_image
      PROPERTIES
      ENVIRONMENT "dream=$<TARGET_FILE:dream_cli>;ITERATIONS=120"
      SKIP_RETURN_CODE 1
      TIMEOUT 600)

    # The same programs with the heap checked after every collection.
    add_test(
      NAME dream_e2e_verified_heap
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e.sh)

    set_tests_properties(
      dream_e2e_verified_heap
      PROPERTIES
      ENVIRONMENT "dream=$<TARGET_FILE:dream_cli>;DREAM_VERIFY_HEAP=1"
      SKIP_RETURN_CODE 1)
  endif()
endif()

# A worked example of embedding the VM from C. Building it keeps the public
# header honest: it must compile as C, not just as C++.
add_executable(
  dream_embed_example
  examples/embed.c)

target_link_libraries(
  dream_embed_example
  PRIVATE
  dream)

set_target_properties(
  dream_embed_example
  PROPERTIES
  OUTPUT_NAME embed
  C_STANDARD 11)

include(GNUInstallDirs)

# An installed `dream` finds `libdream` in `../lib` beside it, wherever the
# installation is put -- a package's /usr, a Homebrew keg, a tarball unpacked
# into ~/.local -- as `mind build vm` already links it (std.build.cc, gnu.dr).
# Only the installed copy: the one in the build tree keeps CMake's absolute
# path, and nothing that runs from there changes.
if(APPLE)
  set_target_properties(dream_cli PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  set_target_properties(dream_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

install(
  TARGETS dream dream_cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
