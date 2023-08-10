#include <iostream>
#include <TNL/Containers/Array.h>

int main( int argc, char* argv[] )
{
    TNL::Containers::Array< int, TNL::Devices::Cuda > device_array{ 1, 2, 3 };
    std::cout << "device_array = " << device_array << std::endl;
    return EXIT_SUCCESS;
}
