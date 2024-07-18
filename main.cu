#include <iostream>
#include "error.cuh"
#include "main_common.cuh"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include "Run.cuh"

void print_welcome_information(void)
{
    printf("\n");
    printf("---------------------------------------------------------------\n");
    printf("|                   Welcome to use NEPTB                      |\n");
    printf("|                      Version beta                           |\n");
    printf("|                   This is the executable                    |\n");
    printf("---------------------------------------------------------------\n");
    printf("\n");
}

int main(int argc, char* argv[])
{
    print_welcome_information();
    print_compile_information();
    print_gpu_information();

    print_line_1();
    printf("Started running NEPTB.\n");
    print_line_2();

    CHECK(cudaDeviceSynchronize());
    clock_t time_begin = clock();

    Run run;

    CHECK(cudaDeviceSynchronize());
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);

    print_line_1();
    printf("Time used = %f s.\n", time_used);
    print_line_2();

    print_line_1();
    printf("Finished running NEPTB.\n");
    print_line_2();

    return EXIT_SUCCESS;



}



