#include "ramon.h"

__global__ void add_numbers(int* result, int first, int second, Ramon& ramon)
{
    // We can call ramon's add_second_to_first method in GPU because it is decorated with __device__ and __host__
    ramon.add_second_to_first(&first, &second);
    result[0] = first;
}

__host__ void add_second_to_first(int* first, int* second, Ramon& ramon)
{
    int* dev_result = NULL;

    cudaMalloc(&dev_result, 1 * sizeof(int));
    cudaMemcpy(dev_result, first, 1 * sizeof(int), cudaMemcpyHostToDevice);

    add_numbers<<<1, 1>>>(dev_result, *first, *second, ramon);

    cudaMemcpy(first, dev_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);

}
