#include "cg_solver.hpp"
#include "operations.hpp"
#include "timer.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

//preconditioned cg solver
void cg_solver(stencil3d const* op, int n, double* x, double const* b, //jacobi_cg_solver
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to cg_solver");
  }
  double *p = new double[n];
  double *q = new double[n];
  double *r = new double[n];
  double *z = new double[n];
  
  double alpha;
  double beta;
  double rho=1.0;
  double rho_old=0.0; 
  double rho_r=1.0;
  double rho_x=1.0;

  
  {
      Timer t("init", op->nx, op->ny, op->nz);
      init(n, r, 0.0);
  }

  // r = op * x
  {
    Timer t("apply_stencil3d", op->nx, op->ny, op->nz);
    apply_stencil3d(op, x, r);
  }

  // r = b - r;
  {
      Timer t("axpby", op->nx, op->ny, op->nz);
      axpby(n, 1.0, b, -1.0, r);
  }

  {
      Timer t("init", op->nx, op->ny, op->nz);
      init(n, p, 0.0);
  }
  {
      Timer t("init", op->nx, op->ny, op->nz);
      init(n, q, 0.0);
  }
  {
      Timer t("init", op->nx, op->ny, op->nz);
      init(n, z, 0.0);
  }
   
  //auto [aa, bb] = extremal_eigenvalues(op, n);
  auto [aa, bb] = explicit_eigenvalues(op);
  //auto [aa, bb] = interval_eigenvalues(op);
  std::cout << "alpha = " << std::setw(8) << aa << "\t" << "beta = " << std::setw(8) << bb << std::endl;
  
  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;
    
    // rho_r = <r, r>
    {
        Timer t("dot", op->nx, op->ny, op->nz);
        rho_r = dot(n,r,r);
    }
    
    if (verbose)
    {
      double sum = 0.0;
      for (int i = 0; i<n; i++) sum += std::pow(x[i],2);
      sum = std::sqrt(sum);
      //checking whether ||r||_2 ~= ||Ax-b||_2
      /* apply_stencil3d(op, x, q);
      axpby(n, 1.0, b, -1.0, q);
      rho = dot(n,q,q);*/
      
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << std::sqrt(rho_r)
                //<< "\t" << std::setw(8) << std::setprecision(4) << std::sqrt(rho)
                << "\t" << std::setw(8) << std::setprecision(4) << sum << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho_r) < tol) || (iter > maxIter) )
    {
      break;
    }
    
    //solve Az=r with jacobi iterations
    
    {
        Timer t("apply_preconditioning", op->nx, op->ny, op->nz);
        apply_cheb(op, r, z, 20, aa, bb);
    }
    
    // rho = <r, z>
    {
        Timer t("dot", op->nx, op->ny, op->nz);
        rho = dot(n,r,z);
    }
      
    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    
    // p = z + alpha * p
    {
      Timer t("axpby", op->nx, op->ny, op->nz);
      axpby(n, 1.0, z, alpha, p);
    }

    // q = op * p
    {
        Timer t("apply_stencil3d", op->nx, op->ny, op->nz);
        apply_stencil3d(op, p, q);
    }

    // beta = <p,q>
    {
        Timer t("dot", op->nx, op->ny, op->nz);
        beta = dot(n,p,q);
    }

    alpha = rho / beta;

    // x = x + alpha * p
    {
      Timer t("axpby", op->nx, op->ny, op->nz);
      axpby(n,alpha,p,1.0,x);
    }
    
    // r = r - alpha * q
    {
      Timer t("axpby", op->nx, op->ny, op->nz);
      axpby(n,-alpha, q, 1.0, r);
    }
    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  delete [] z;
  
  // return number of iterations and achieved residual
  *resNorm = rho_r;
  *numIter = iter;
  return;
}
