#include <iostream>
#include <cstdlib>
#include <cmath>
#include "particle.h"
#include "ramon.h"

/* Pre-declaring the functions so that main knows they exist. They are defined
 * below the main function. At link time they are put together (read STEP 5 in .pro file)*/
void perform_on_cpu(particle* pArray, int n);
void perform_on_gpu(particle* pArray, int n);
void add_second_to_first(int* first, int* second, Ramon& ramon);

/* Entry point for the program. We can pass argument to the program as follows:
 *   
 *  $ ./app [runOnGPU] [n] [seed]
 *
 * Where:
 *   runOnGPU: can be either 0 (run on CPU) or 1 (run on GPU)
 *   n: can be any positive integer number
 *   seed: a seed for the random functions */
int main(int argc, char ** argv)
{
    Ramon ramon = Ramon();    // hybrid class with __host__ sayHi() method
    ramon.say_hi();           // calling a __host__ function from within host code WORKS

    int* first = new int(3);
    int* second = new int(5);

    std::cout << "CPU: Adding " << *(second) << " (SECOND) to " << *(first) << " (FIRST)" << std::endl;
    // We can call ramon's add_second_to_first method in CPU because it is decorated with __device__ and __host__
    ramon.add_second_to_first(first, second);
    std::cout << "CPU: Now FIRST is " << *(first) << '\n' << std::endl;

    std::cout << "GPU: Adding " << *(second) << " (SECOND) to " << *(first) << " (FIRST)" << std::endl;
    // The ramon.add_second_to_first(...) gets called on GPU (on a separate kernel)! It will work as we decorated with __device__ and __host__
    add_second_to_first(first, second, ramon);
    std::cout << "GPU: Now FIRST is " << *(first) << '\n' <<  std::endl;

    int n = 5000000;         // default number of particles
    bool runOnGPU = true;    // default runs GPU code
    if(argc > 1)    { runOnGPU = (bool)atoi(argv[1]); }
    if(argc > 2)    { n = atoi(argv[2]);}
    if(argc > 3)    { srand(atoi(argv[3]));}

    particle * pArray = new particle[n];    // array or particles

    clock_t startTime = clock();

    if (runOnGPU) {
        perform_on_gpu(pArray, n);
    } else {
        perform_on_cpu(pArray, n);
    }

    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;

    // calculate how much time it took
    double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;

    // calculate total distance and average distance that particles travelled
    v3 totalDistance(0,0,0);
    v3 temp;
    for(int i=0; i<n; i++)
    {
        temp = pArray[i].getTotalDistance();
        totalDistance.x += temp.x;
        totalDistance.y += temp.y;
        totalDistance.z += temp.z;
    }

    float avgX = totalDistance.x /(float)n;
    float avgY = totalDistance.y /(float)n;
    float avgZ = totalDistance.z /(float)n;
    float avgNorm = sqrt(avgX*avgX + avgY*avgY + avgZ*avgZ);

    // print results to standard output
    std::cout << "Moved " << n << " particles along 100 steps." << '\n'
              << "Average distance traveled is |("<< avgX << ", " << avgY 
              << ", " << avgZ <<")|="<< avgNorm << '\n'
              << "Time: " << timeInSeconds << " seconds" << std::endl;

    return 0;

}

/* We need to perfom the operations in a serial way, so we need use of for loop
 * to do that. Linearly stepping each particle one at a time. */
void perform_on_cpu(particle* pArray, int n) {
    for (int step = 0; step < 100; step++)
    {
        for (int i = 0; i < n; i++) {
            float dt = (float)rand()/(float) RAND_MAX;    // random distance each step
            pArray[i].advance(dt);
        }
    }
}
