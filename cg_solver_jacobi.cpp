#include "cg_solver_jacobi.hpp"
#include "operations.hpp"
#include "timer.hpp"

#include <cmath>
#include <stdexcept>

#include <iostream>
#include <iomanip>

//preconditioned cg solver
void jacobi_cg_solver(stencil3d const* op, int n, double* x, double const* b,
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
  //double *sigma = new double[n];
  double alpha, beta, rho=1.0, rho_old=0.0, rho_r=1.0;
  

  // r = op * x
  {
    Timer t("apply_stencil3d");
    t.m = (6 + 7)*(op->nx - 2)*(op->ny - 2)*(op->nz - 2) + (5 + 6)*2*((op->nx - 2)*(op->ny - 2)+(op->nx - 2)*(op->nz - 2)+(op->nz - 2)*(op->ny - 2)) + (4 + 5)*4*((op->nx - 2) + (op->ny - 2) + (op->nz - 2)) + (3 + 4)*8;
    t.b = 1.0 * sizeof(op) + 2.0 * n * sizeof(double);
    apply_stencil3d(op, x, r);
  }

  // r = b - r;
  {
      Timer t("axpby");
      t.m = 3.0 * n;
      t.b = 3.0 * sizeof(double) * n ;
      axpby(n, 1.0, b, -1.0, r);
  }

  {
      Timer t("init");
      t.m = 0.0; 
      t.b = 1.0 * sizeof(double) * n;
      init(n, p, 0.0);
  }
  {
      Timer t("init");
      t.m = 0.0; 
      t.b = 1.0 * sizeof(double) * n;
      init(n, q, 0.0);
  }
  {
      Timer t("init");
      t.m = 0.0; 
      t.b = 1.0 * sizeof(double) * n;
      init(n, z, 0.0);
  }

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;
    
    // rho_r = <r, r>
    {
        Timer t("dot");
        t.m = 2.0 * n;
        t.b = 2.0 * sizeof(double) * n;
        rho_r = dot(n,r,r);
    }
    
    if (verbose)
    {
      double sum = 0.0;
      for (int i = 0; i<n; i++) sum += std::pow(x[i],2);
      sum = std::sqrt(sum);
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho_r
                << "\t" << std::setw(8) << std::setprecision(4) << sum << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho_r) < tol) || (iter > maxIter))
    {
      break;
    }
    
    //solve Az=r with jacobi iterations
    
    {
        Timer t("apply_preconditioning");
        t.m = 0.0;
        t.b = 0.0;
        apply_gauss_seidel(op,r,z,5000);
        //apply_jacobi_iterations(op, r, z, 500);
    }
    
    // rho = <r, z>
    {
        Timer t("dot");
        t.m = 2.0 * n;
        t.b = 2.0 * sizeof(double) * n;
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
      Timer t("axpby");
      t.m = 3.0 * n;
      t.b = 3.0 * sizeof(double) * n ;
      axpby(n, 1.0, z, alpha, p);
    }

    // q = op * p
    {
        Timer t("apply_stencil3d");
        t.m = (6 + 7)*(op->nx - 2)*(op->ny - 2)*(op->nz - 2) + (5 + 6)*2*((op->nx - 2)*(op->ny - 2)+(op->nx - 2)*(op->nz - 2)+(op->nz - 2)*(op->ny - 2)) + (4 + 5)*4*((op->nx - 2) + (op->ny - 2) + (op->nz - 2)) + (3 + 4)*8;
        t.b = 1.0 * sizeof(op) + 2.0 * n * sizeof(double);
        apply_stencil3d(op, p, q);
    }

    // beta = <p,q>
    {
        Timer t("dot");
        t.m = 2.0 * n;
        t.b = 2.0 * sizeof(double) * n;
        beta = dot(n,p,q);
    }

    alpha = rho / beta;

    // x = x + alpha * p
    {
      Timer t("axpby");
      t.m = 3.0 * n;
      t.b = 3.0 * sizeof(double) * n ;
      axpby(n,alpha,p,1.0,x);
    }
    
    // r = r - alpha * q
    {
      Timer t("axpby");
      t.m = 3.0 * n;
      t.b = 3.0 * sizeof(double) * n ;
      axpby(n,-alpha, q, 1.0, r);
    }

    std::swap(rho_old, rho);
  }// end of while-loop

  // clean up
  delete [] p;
  delete [] q;
  delete [] r;
  delete [] z;
  //delete [] sigma;
  
  // return number of iterations and achieved residual
  *resNorm = rho_r;
  *numIter = iter;
  return;
}
