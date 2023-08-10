#include <iostream>
#include <TNL/Containers/Array.h>

int main( int argc, char* argv[] )
{
    TNL::Containers::Array< int > host_array{ 1, 2, 3 };
    std::cout << "host_array = " << host_array << std::endl;
    return EXIT_SUCCESS;
}
