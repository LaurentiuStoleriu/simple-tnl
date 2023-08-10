#include <iostream>
#include <TNL/FileName.h>
#include <TNL/Containers/Vector.h>
//#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/ODE/Merson.h>
////////////////////
//#include "write.h"
#include <fstream>

template<typename Vector>
void write( std::fstream& file, const Vector& u, const int n, const double& h, const double& time )
{
    std::cout/*file*/ << "# time = " << time << std::endl;
    for( int i = 0; i < n; i++ )
        std::cout/*file*/ << i*h << " " << u.getElement( i ) << std::endl;
    std::cout/*file*/ << std::endl;
}
/////////////////////

//using Real = double;
//using Index = int;

template< typename Device >
void solveHeatEquation( const char* file_name )
{
    using Vector = TNL::Containers::Vector< double, Device, int >;
    using VectorView = typename Vector::ViewType;
    //using ODESolver = TNL::Solvers::ODE::Euler< Vector >;
    using ODESolver = TNL::Solvers::ODE::Merson< Vector >;

/***
* Parameters of the discretisation
*/
    const double final_t = 0.05;
    const double output_time_step = 0.005;
    const int n = 41;
    const double h = 1.0 / ( n - 1 );
    const double tau = 0.1 * h * h;
    const double h_sqr_inv = 1.0 / ( h * h );

/***
* Initial condition
*/
    Vector u( n );
    u.forAllElements( [=] __cuda_callable__ ( int i, double& value ) {
    const double x = i * h;
    if( x >= 0.4 && x <= 0.6 )
        value = 1.0;
    else 
        value = 0.0;
    } );

    std::fstream file;
    file.open( file_name, std::ios::out );
    write( file, u, n, h, (double) 0.0 );

/***
* Setup of the solver
*/
    ODESolver solver;
    solver.setTau(  tau );
    solver.setTime( 0.0 );

/***
* Time loop
*/
    while( solver.getTime() < final_t )
    {
        solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
        auto f = [=] __cuda_callable__ ( int i, const VectorView& u, VectorView& fu ) mutable {
         if( i == 0 || i == n-1 )                // boundary nodes -> boundary conditions
            fu[ i ] = 0.0;
         else                                    // interior nodes -> approximation of the second derivative
            fu[ i ] = h_sqr_inv * (  u[ i - 1 ] - 2.0 * u[ i ] + u[ i + 1 ] );
        };
        auto time_stepping = [=] ( const double& t, const double& tau, const VectorView& u, VectorView& fu ) {
            TNL::Algorithms::parallelFor< Device >( 0, n, f, u, fu ); };
            solver.solve( u, time_stepping );
        write( file, u, n, h, solver.getTime() ); // write the current state to a file
    }
}

int main( int argc, char* argv[] )
{
    //TNL::String file_name( argv[ 1 ] );
    TNL::String file_name("/home/lali/TITAN-ROG-sync/CUDA/simple-tnl/ODE/ODESolver-HeatEquationExample-result.out");

    std::cout << file_name.getString();

    //solveHeatEquation< TNL::Devices::Host >( file_name.getString() );
#ifdef __CUDACC__
    solveHeatEquation< TNL::Devices::Cuda >( file_name.getString() );
#endif
}