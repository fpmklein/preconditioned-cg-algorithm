#include "operations.hpp"
#include "cg_solver_jacobi.hpp"
#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

#include <cmath>

// Main program that solves the 3D Poisson equation
// on a unit cube. The grid size (nx,ny,nz) can be
// passed to the executable like this:
//
// ./main_cg_poisson <nx> <ny> <nz>
//
// or simply ./main_cg_poisson <nx> for ny=nz=nx.
// If no arguments are given, the default nx=ny=nz=128 is used.
//
// Boundary conditions and forcing term f(x,y,z) are
// hard-coded in this file. See README.md for details
// on the PDE and boundary conditions.


stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
  if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
  stencil3d L;
  L.nx=nx; L.ny=ny; L.nz=nz;
  double dx=1.0/(nx - 1), dy=1.0/(ny - 1), dz=1.0/(nz - 1);
  L.value_c = 2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
  L.value_n =  -1.0/(dy*dy);
  L.value_e = -1.0/(dx*dx);
  L.value_s =  -1.0/(dy*dy);
  L.value_w = -1.0/(dx*dx);
  L.value_t = -1.0/(dz*dz);
  L.value_b = -1.0/(dz*dz);
  return L;
}

int main(int argc, char* argv[])
{
  {
    Timer t("main_cg_solver_pre");
    int nx, ny, nz;

    if      (argc==1) {nx=128;           ny=128;           nz=128;}
    else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;}
    else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]);}
    else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)"<<std::endl; exit(-1);}
    if (ny<0) ny=nx;
    if (nz<0) nz=nx;

    // total number of unknowns
    int n=nx*ny*nz;

    double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

    // Laplace operator
    stencil3d L = laplace3d_stencil(nx,ny,nz);

    // solution vector
    double *x = new double[n];
    init(n, x, 1.0);

    // right-hand side
    double *b = new double[n];
    init(n, b, 0.0);

    // Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
    for (int j=0; j<ny; j++)
      for (int i=0; i<nx; i++)
      {
        apply_stencil3d(&L,x,b);
      }

    //initial guess solution vector
    init(n, x, 0.0);
    
    // solve the linear system of equations using CG
    int numIter, maxIter=600;
    double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());
    std::cout<<"tol = "<<tol<<std::endl;
    try
    {
      {
        // use preconditioned cg solver
        Timer t("cg_solver_pre");
        double sum = 0.0;
        jacobi_cg_solver(&L, n, x, b, tol, maxIter, &resNorm, &numIter);
        for (int i = 0; i<n; i++) sum += std::pow(x[i],2);
        sum = std::sqrt(sum);
        std::cout<<"||x||_2 = "<<sum<<std::endl;
      }
    } catch(std::exception e)
    {
      std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
      exit(-1);
    }
    delete [] x;
    delete [] b;
  }
  Timer::summarize();

  return 0;
}
