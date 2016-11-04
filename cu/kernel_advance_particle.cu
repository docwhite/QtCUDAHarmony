#include <iostream>
#include "particle.h"

/* This is the kernel. A kernel is a program that CPU launches to GPU.*/
__global__ void advanceParticles(float dt, particle * pArray, int nParticles)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nParticles)
    {
        pArray[idx].advance(dt);  // advance particle by dt distance
    }
}

/* We run the code parallely. For that we need to perform some STEPS:
 *   1. Allocate memory for the current array of particles --> cudaMalloc
 *   2. Copy the array of particles from CPU (host) to GPU (device) --> cudaMemcpy
 *   3. Launch kernel for each step. A kernel is a program that runs on each
        mall unit cell on the graphics card.
 *   4. Copy back the array of particles that now is in GPU (device) back to 
 *      CPU (host) --> cudaMemcpy
 */
__host__ void perform_on_gpu(particle* host_pArray, int n)
{
    particle * device_pArray = NULL;  
    cudaMalloc(&device_pArray, n*sizeof(particle));                                         // step 1
    cudaMemcpy(device_pArray, host_pArray, n*sizeof(particle), cudaMemcpyHostToDevice);     // step 2
    for(int i=0; i<100; i++)
    {
        float dt = (float)rand()/(float) RAND_MAX; // Random distance each step
        advanceParticles<<< 1 +  n/256, 256>>>(dt, device_pArray, n);                       // step 3
        cudaDeviceSynchronize();
    }
    cudaMemcpy(host_pArray, device_pArray, n*sizeof(particle), cudaMemcpyDeviceToHost);     // step 4
}

