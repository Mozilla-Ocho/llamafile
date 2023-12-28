// https://stackoverflow.com/a/40695640/1653720
#include <cstdio>
#include <cstdlib>
#if !defined(USE_HIP)
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#if defined(USE_HIP)
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaError_t hipError_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#endif

int main(int argc, char *argv[]) {
  cudaDeviceProp prop;
  cudaError_t status;
  int device_count;
  int device_index = 0;
  if (argc > 1) {
    device_index = atoi(argv[1]);
  }
  status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status));
    return -1;
  }
  if (device_index >= device_count) {
    fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", device_index, device_count);
    return -1;
  }
  status = cudaGetDeviceProperties(&prop, device_index);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceProperties() for device device_index failed: %s\n", cudaGetErrorString(status));
    return -1;
  }

#if !defined(USE_HIP)
  int v = prop.major * 10 + prop.minor;
  printf("%d", v);
#else
  printf("%s", prop.gcnArchName);
#endif
}
