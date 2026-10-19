#include <cuda_runtime.h>

struct Pair {
  int lo;
  int hi;
};

extern "C" __global__ void kernel(int* out, int a, int index) {
  __shared__ Pair x;
#ifdef USE_X
  x.lo = a + 1;
  x.hi = a + 2;
  out[0] = x.lo;
  out[1] = x.hi;
#else
  out[0] = a;
#endif
}
