#include <iostream>
#include <random>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template<typename Device>
void initMeshFunction (const int xSize, const int ySize, Vector<double, Device> &v, const double &c)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(100.0, 200.0);

    auto view = v.getView();
    auto init = [=] __cuda_callable__ (const StaticArray<2, int> &i) mutable
    {
        view[i.y()  * xSize + i.x()] = dis(gen)*dis(gen)*dis(gen)/dis(gen);
    };
    StaticArray<2, int> begin{0, 0};
    StaticArray<2, int> end{xSize, ySize};
    parallelFor<Device>(begin, end, init);
}

int main(int argc, char* argv[])
{
    const int xSize(10000), ySize(10000);
    const int size = xSize * ySize;

    Vector<double, Devices::Host> host_v(size);
    initMeshFunction(xSize, ySize, host_v, 1.0);
    //std::cout << "host_v = " << host_v << std::endl;
    std::cout << "DONE!!" << std::endl;

#ifdef __CUDACC__
    Vector< double, Devices::Cuda > cuda_v( size );
    initMeshFunction( xSize, ySize, cuda_v, 1.0 );
#endif

    return EXIT_SUCCESS;
}