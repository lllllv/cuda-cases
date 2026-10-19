#include <cuda.h>

#include <cstdio>
#include <cstdlib>

void Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s: %s\n", operation, message ? message : "unknown");
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s file.cubin A EXPECTED0 EXPECTED1\n",
                 argv[0]);
    return 2;
  }
  int a = std::atoi(argv[2]);
  int index = 0;
  const int expected[] = {std::atoi(argv[3]), std::atoi(argv[4])};
  Check(cuInit(0), "cuInit");
  CUdevice device;
  Check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context;
  Check(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  CUmodule module;
  Check(cuModuleLoad(&module, argv[1]), "cuModuleLoad");
  CUfunction function;
  Check(cuModuleGetFunction(&function, module, "kernel"),
        "cuModuleGetFunction");
  int shared_bytes = -1;
  Check(cuFuncGetAttribute(&shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                           function),
        "cuFuncGetAttribute");
  std::printf("static_shared_bytes=%d\n", shared_bytes);
  CUdeviceptr output;
  Check(cuMemAlloc(&output, sizeof(expected)), "cuMemAlloc");
  Check(cuMemsetD32(output, 0, 2), "cuMemsetD32");
  void* args[] = {&output, &a, &index};
  Check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
        "cuLaunchKernel");
  Check(cuCtxSynchronize(), "cuCtxSynchronize");
  int actual[2];
  Check(cuMemcpyDtoH(actual, output, sizeof(actual)), "cuMemcpyDtoH");
  bool pass = true;
  for (int i = 0; i < 2; ++i) {
    std::printf("out[%d]=%d expected=%d\n", i, actual[i], expected[i]);
    pass &= actual[i] == expected[i];
  }
  Check(cuMemFree(output), "cuMemFree");
  Check(cuModuleUnload(module), "cuModuleUnload");
  Check(cuCtxDestroy(context), "cuCtxDestroy");
  std::puts(pass ? "KERNEL_PASS" : "KERNEL_FAIL");
  return pass ? 0 : 1;
}
