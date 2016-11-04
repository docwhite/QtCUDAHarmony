#ifndef __v3_h__
#define __v3_h__

// v3.H IS AN HYBRID CLASS

/* __CUDACC__ is set to true when nvcc compiles code. We are
 * saying that if g++ is handling this header file it should
 * not have __host__ __device__ decorators because g++ would
 * not understand them. But when nvcc is using this header
 * file it will be placing __host__ and __device__ in the
 * method and that will be generating CPU and GPU code on the
 * particle.o object file.
 */

#ifdef __CUDACC__
#define CUDA_DECORATORS __host__ __device__
#else
#define CUDA_DECORATORS
#endif

class v3
{
public:
    float x;
    float y;
    float z;
    
    v3();
    v3(float _x, float _y, float _z);
    void randomize();    // sets x, y and z to random numbers between 0 and 1
    CUDA_DECORATORS void normalize();
    CUDA_DECORATORS void scramble();

};

#endif
