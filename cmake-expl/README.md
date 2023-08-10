# Example Project with CMake

Basic example how to set up a C++/CUDA project using TNL and CMake as the build
system

Note that this example assumes that TNL has been installed on the system and
can be found by CMake using `find_package(TNL)`.

See the [TNL documentation](https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/) for
details on using the library.

## CMake usage

To build the project using CMake, you must first generate a build system.
For example, using the [Ninja generator](https://ninja-build.org/):

    cmake -S . -B build -G Ninja

Then build the project in the `build` directory:

    cmake --build build

See [cmake(1)](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for
details.
