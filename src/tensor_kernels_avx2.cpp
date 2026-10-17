// The CPU kernels again, compiled with AVX2 and FMA (see CMakeLists.txt).
// Only ever called once `tensor_kernels()` has asked the CPU whether it can.
#define DREAM_KERNEL_NS kernels_avx2
#include "tensor_kernels.inc"
