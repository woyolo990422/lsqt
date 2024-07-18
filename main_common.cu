
#include "error.cuh"
#include "main_common.cuh"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

void print_compile_information(void)
{
    print_line_1();
    printf("Compiling options:\n");
    print_line_2();

#ifdef DEBUG
    printf("DEBUG is on: Use a fixed PRNG seed for different runs.\n");
#else
    srand(std::chrono::system_clock::now().time_since_epoch().count());
    printf("DEBUG is off: Use different PRNG seeds for different runs.\n");
#endif
}

void print_gpu_information(void)
{
    print_line_1();
    printf("GPU information:\n");
    print_line_2();

    int num_gpus;
    CHECK(cudaGetDeviceCount(&num_gpus));
    printf("number of GPUs = %d\n", num_gpus);

    for (int device_id = 0; device_id < num_gpus; ++device_id) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, device_id));

        printf("Device id:                   %d\n", device_id);
        printf("    Device name:             %s\n", prop.name);
        printf("    Compute capability:      %d.%d\n", prop.major, prop.minor);
        printf("    Amount of global memory: %g GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("    Number of SMs:           %d\n", prop.multiProcessorCount);
    }

    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            int can_access;
            if (i != j) {
                CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    CHECK(cudaDeviceEnablePeerAccess(j, 0));
                    printf("GPU-%d can access GPU-%d.\n", i, j);
                }
                else {
                    printf("GPU-%d cannot access GPU-%d.\n", i, j);
                }
            }
        }
    }

    cudaSetDevice(0); // normally use GPU-0
}
