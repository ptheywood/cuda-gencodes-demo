#include <stdio.h>
#include <cuda_runtime.h>

// Macros to enable printf of a string macro values
#define str(s...) #s
#define xstr(s) (str(s))

/**
 * Kernel which just print's a message from thread 0, just to demonstrate device code executed,.
 */
__global__ void helloWorld() {
    if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
        printf("Hello from thread %d\n", threadIdx.x + blockDim.x * blockIdx.x);
    }
}

/**
 * Main method, 
 */
int main(int argc, const char * argv[]) {

#if defined(__NVCC__)
    fprintf(stdout, "Compiled with nvcc %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
#endif
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12  || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 5)) 
	printf("__CUDA_ARCH_LIST__ %s\n", xstr(__CUDA_ARCH_LIST__));
#endif 

    cudaError_t status = cudaSuccess;
    // Get properties for the 0th device, print the name and compute capability
    int deviceIdx = 0;
    int deviceCount = 0;
    status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaGetDeviceCount, %d CUDA devices found.\n  %s: %s\n", deviceCount, cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    if (deviceIdx >= deviceCount) {
        fprintf(stderr, "Requested device %d is not valid, %d devices found.\n", deviceIdx, deviceCount);
        return EXIT_FAILURE;
    }

    // Get and print device properties
    cudaDeviceProp deviceProps = {};
    status = cudaGetDeviceProperties(&deviceProps, deviceIdx);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaGetDeviceProperties(&deviceProps, %d) %s: %s\n", deviceIdx, cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    fprintf(stdout, "GPU %d: sm_%d%d %s\n", deviceIdx, deviceProps.major, deviceProps.minor, deviceProps.name);

    // Set the device
    status = cudaSetDevice(deviceIdx);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaSetDevice(%d) %s: %s\n", deviceIdx, cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }

    // Initialise a cuda context, reporting any errors if failure occurs
    status = cudaFree(nullptr);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaFree(nullptr).\n  %s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    // Launch the kernel with a single thread.
    helloWorld<<<1, 1>>>();
    // Check if the kernel launch reported any errors, bue as this is pre-sync the kernel may not have executed yet 
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaGetLastError() after helloWorld launch.\n  %s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    // Sync the device 
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "Error reported by cudaDeviceSynchronize().\n  %s: %s\n", cudaGetErrorName(status), cudaGetErrorString(status));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
