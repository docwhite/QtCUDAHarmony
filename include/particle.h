#ifndef __particle_h__
#define __particle_h__

// PARTICLE.H IS AN HYBRID CLASS

/* __CUDACC__ is set to true when nvcc compiles code. We are
 * saying that if g++ is handling this header file it should
 * not have __host__ __device__ decorators because g++ would
 * not understand them. But when nvcc is using this header
 * file it will be placing __host__ and __device__ in the
 * method and that will be generating CPU and GPU code on the
 * particle.o object file.
 */

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_AND_DEVICE __host__ __device__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_HOST_AND_DEVICE
#endif

#include <v3.h>

class particle
{
    private:
        v3 position;
        v3 velocity;
        v3 totalDistance;

    public:
        particle();
        CUDA_DEVICE void advance(float dist);
        const v3& getTotalDistance() const;
};

#endif
