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
  double *sigma = new double[n];
  double *x_iter = new double[n];
  double alpha, beta, rho=1.0, rho_old=0.0;

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

  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;
    // rho = <r, r>
    {
        Timer t("dot");
        t.m = 2.0 * n;
        t.b = 2.0 * sizeof(double) * n;
        rho = dot(n,r,r);
    }
    if (verbose)
    {
      double sum = 0.0;
      for (int i = 0; i<n; i++) sum += std::pow(x[i],2);
      sum = std::sqrt(sum);
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << rho
                << "\t" << std::setw(8) << std::setprecision(4) << sum << std::endl;
    }

    // check for convergence or failure
    if ((std::sqrt(rho) < tol) || (iter > maxIter))
    {
      break;
    }
    
    if (rho_old==0.0)
    {
      alpha = 0.0;
    }
    else
    {
      alpha = rho / rho_old;
    }
    // p = r + alpha * p
    {
      Timer t("axpby");
      t.m = 3.0 * n;
      t.b = 3.0 * sizeof(double) * n ;
      axpby(n, 1.0, r, alpha, p);
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
    
    stencil3d op2;
    op2.nx = op->nx;
    op2.ny = op->ny;
    op2.nz = op->nz;
    op2.value_c = 0.0;
    op2.value_n = op->value_n;
    op2.value_s = op->value_s;
    op2.value_e = op->value_e;
    op2.value_w = op->value_w;
    op2.value_t = op->value_t;
    op2.value_b = op->value_b;
    
    for (int i=0; i<n; i++)
        x_iter[i] = x[i];
    for (int k=0; k<50; k++)
    {
        //sigma(i) = sum_i!=j A(i,j)*u(j)
        apply_stencil3d(&op2,x_iter,sigma);
        //sigma = 1/c(b-sigma)
        axpby(n,1.0/(op->value_c), b, -1.0/(op->value_c), sigma);
        for (int i=0; i<n; i++)
            x_iter[i] = sigma[i];
    }
    for (int i=0; i<n; i++)
        x[i] = x_iter[i];
    
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
  delete [] sigma;
  delete [] x_iter;
  
  // return number of iterations and achieved residual
  *resNorm = rho;
  *numIter = iter;
  return;
}
