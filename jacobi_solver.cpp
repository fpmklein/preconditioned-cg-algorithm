#include "cg_solver.hpp"
#include "operations.hpp"
#include "timer.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

//preconditioned cg solver
void cg_solver(stencil3d const* op, int n, double* x, double const* b, //jacobi_solver
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to jacobi_solver");
  }
    
    double *sigma = new double[n];
    double *r = new double[n];
    
    double rho=1.0; 
    
    int iter = - 1;
    //x_0
    axy(n, 1.0/(op->value_c), b, x);
    for (int k=0; k<maxIter; k++)
    {
        iter++;
        if (verbose)
        {
          double sum = 0.0;
          for (int i = 0; i<n; i++) sum += std::pow(x[i],2);
          sum = std::sqrt(sum);
          
          std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << std::sqrt(rho) 
                    << "\t" << std::setw(8) << std::setprecision(4) << sum << std::endl;
        }
        apply_stencil3d(op, x, r);
        
        #pragma omp parallel for schedule(static)
        for (int i=0; i<n; i++)
        {
            x[i] = x[i] + (b[i] - r[i])/op->value_c;   
        }
        
        axpby(n, 1.0, b, - 1.0, r);
        rho = dot(n,r,r);
        if ((std::sqrt(rho) < tol) || (iter > maxIter) )//|| (std::sqrt(rho_x) < tol) )
        {
           break;
        }
        *resNorm = rho;
        *numIter = iter;
    }
    delete [] sigma;
    delete [] r;

  return;
}
