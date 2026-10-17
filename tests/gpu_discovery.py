#!/usr/bin/env python3
"""Linux OpenCL discovery regressions; no GPU or OpenCL SDK required.

Run: python3 dream/tests/gpu_discovery.py
Uses a mock system loader and vendor ICDs in an isolated temporary directory.
"""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

MOCK = r'''
#include <cstdint>
#include <cstring>
using Fn = void(*)();
static Fn dispatch[106];
static struct { Fn* table; } platform{dispatch};
extern "C" int info(void*, unsigned, size_t n, void* out, size_t*) {
    const char* value = OLD ? "OpenCL 1.1 Mock" : "OpenCL 1.2 Mock";
    if (n < strlen(value) + 1) return -30;
    strcpy(static_cast<char*>(out), value); return 0;
}
extern "C" int devices(void* p, uint64_t type, unsigned, void** out, unsigned*) {
    if (p != &platform || (CPU && type == 4)) return -1;
    *out = &platform; return 0;
}
extern "C" int device_info(void*, unsigned, size_t, void* out, size_t*) {
    *static_cast<int*>(out) = CPU ? 7 : 42; return 0;
}
static void unused() {}
static int platforms(unsigned, void** out, unsigned* count) {
    if (EMPTY) { if (count) *count = 0; return -1001; }
    // Slot zero intentionally null: direct vendors enumerate via the extension.
    for (int i = 1; i < 106; ++i) dispatch[i] = unused;
    dispatch[1] = reinterpret_cast<Fn>(info);
    dispatch[2] = reinterpret_cast<Fn>(devices);
    dispatch[3] = reinterpret_cast<Fn>(device_info);
    if (BROKEN) dispatch[105] = nullptr;
    if (count) *count = 1;
    if (out) *out = &platform;
    return 0;
}
#if VENDOR
extern "C" void* clGetExtensionFunctionAddress(const char* name) {
    return strcmp(name, "clIcdGetPlatformIDsKHR") ? nullptr : reinterpret_cast<void*>(platforms);
}
#else
extern "C" int clGetPlatformIDs(unsigned n, void** out, unsigned* count) {
    return platforms(n, out, count);
}
extern "C" int clGetDeviceIDs(void* p, uint64_t t, unsigned n, void** d, unsigned* c) {
    return devices(p, t, n, d, c);
}
extern "C" int clGetDeviceInfo(void* p, unsigned t, size_t n, void* o, size_t* s) {
    return device_info(p, t, n, o, s);
}
#define STUB(name) extern "C" void name() {}
STUB(clCreateContext) STUB(clCreateCommandQueue) STUB(clCreateBuffer)
STUB(clReleaseMemObject) STUB(clEnqueueWriteBuffer) STUB(clEnqueueReadBuffer)
STUB(clEnqueueCopyBuffer) STUB(clCreateProgramWithSource) STUB(clBuildProgram)
STUB(clGetProgramBuildInfo) STUB(clCreateKernel) STUB(clSetKernelArg)
STUB(clEnqueueNDRangeKernel) STUB(clFinish) STUB(clFlush)
STUB(clEnqueueMarkerWithWaitList) STUB(clWaitForEvents) STUB(clReleaseEvent)
#endif
'''
PROBE = r'''
#include "gpu.cpp"
#include <iostream>
int main() {
    dream::gpu::Device d;
    std::string why;
    if (!dream::gpu::pick_device(d, &why)) { std::cout << why; return 1; }
    int value = 0;
    d.cl.GetDeviceInfo(d.device, 0, sizeof value, &value, nullptr);
    std::cout << value;
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix="dream-gpu-discovery-") as tmp:
        work = Path(tmp)
        compiler = os.environ.get("CXX", "c++")
        (work / "mock.cpp").write_text(MOCK)
        (work / "probe.cpp").write_text(PROBE)
        subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "dream/src"),
                        str(work / "probe.cpp"), "-ldl", "-pthread", "-o", str(work / "probe")], check=True)

        def mock(name, *, vendor=0, cpu=0, empty=0, broken=0, old=0):
            path = work / name
            subprocess.run([compiler, "-shared", "-fPIC", f"-DVENDOR={vendor}",
                            f"-DCPU={cpu}", f"-DEMPTY={empty}", f"-DBROKEN={broken}", f"-DOLD={old}",
                            str(work / "mock.cpp"), "-o", str(path)], check=True)
            return path

        loader = mock("loader.so")
        vendor = mock("vendor.so", vendor=1)
        cpu = mock("cpu.so", vendor=1, cpu=1)
        broken = mock("broken.so", vendor=1, broken=1)
        old = mock("old.so", vendor=1, old=1)
        empty = mock("libOpenCL.so.1", empty=1)
        mock("libOpenCL.so", empty=1)
        registry = work / "vendors"
        registry.mkdir()
        env = dict(os.environ, LD_LIBRARY_PATH=str(work), OCL_ICD_VENDORS=str(registry))
        env.pop("DREAM_OPENCL_LIB", None)

        def check(label, expected, forced=None, error=None):
            run_env = dict(env)
            if forced is not None:
                run_env["DREAM_OPENCL_LIB"] = str(forced)
            result = subprocess.run([str(work / "probe")], env=run_env, capture_output=True, text=True)
            assert result.returncode == (1 if error else 0), (label, result.stdout, result.stderr)
            assert (error in result.stdout if error else result.stdout == expected), (label, result.stdout)
            print(f"PASS {label}")

        check("system loader API", "42", loader)
        check("vendor dispatch without core exports", "42", vendor)
        check("CPU fallback", "7", cpu)
        check("empty loader", "", empty, "no platforms")
        check("old vendor version", "", old, "OpenCL 1.2 or later")
        check("missing required dispatch entry", "", broken, "clEnqueueMarkerWithWaitList")
        (registry / "a-cpu.icd").write_text(str(cpu) + "\n")
        (registry / "b-broken.icd").write_text(str(broken) + "\n")
        (registry / "c-gpu.icd").write_text("  " + str(vendor) + "\r\n")
        (registry / "ignored.txt").write_text("not a driver")
        check("registered GPU beats CPU and broken providers", "42")
        check("explicit missing library never falls back", "", work / "absent.so", "no usable OpenCL")
        check("explicit CPU library overrides registered GPU", "7", cpu)


if __name__ == "__main__":
    main()
