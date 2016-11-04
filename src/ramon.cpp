#include <iostream>
#include "ramon.h"

Ramon::Ramon()
{
    std::cout << "Ramon has been created!" << std::endl;
};

__host__
void Ramon::say_hi()
{
    std::cout << "Ramon says HI!" << std::endl;
}

__host__ __device__
void Ramon::add_second_to_first(int* first, int* second)
{
    *(first) = *(first) + *(second);
}

Ramon::~Ramon()
{
    std::cout << "Ramon has been destructed!" << std::endl;
}
