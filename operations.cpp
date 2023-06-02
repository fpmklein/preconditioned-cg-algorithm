#include "operations.hpp"
#include <omp.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>

void init(int n, double* x, double value)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i<n; i++)
  {
    x[i] = value;
  }
  return;
}

double dot(int n, double const* x, double const* y)
{
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for (int i = 0; i<n; i++)
  {
    sum += x[i]*y[i];
  }
  return sum;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i<n; i++)
  {
    y[i] = a*x[i] + b*y[i];
  }
  return;
}

void axy(int n, double a, double const* x, double* y)
{
  #pragma omp parallel for schedule(static)
  for (int i = 0; i<n; i++)
  {
    y[i] = a*x[i];
  }
  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  //v=S*u: v,u vectors and S a 7-point stencil
  
  //interior points, k=1,...,nz-2, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(3) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
      for (int i = 1; i < S->nx - 1; i++)
      {
        v[S->index_c(i, j, k)]  = S->value_b * u[S->index_b(i, j, k)]
                                + S->value_s * u[S->index_s(i, j, k)]
                                + S->value_w * u[S->index_w(i, j, k)]
                                + S->value_c * u[S->index_c(i, j, k)]
                                + S->value_e * u[S->index_e(i, j, k)]
                                + S->value_n * u[S->index_n(i, j, k)]
                                + S->value_t * u[S->index_t(i, j, k)];
      }
    }
  }
  
  //boundary points
  //face: k = 0, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int j = 1; j < S->ny - 1; j++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, j, 0)]   = S->value_s * u[S->index_s(i, j, 0)]
                                + S->value_w * u[S->index_w(i, j, 0)]
                                + S->value_c * u[S->index_c(i, j, 0)]
                                + S->value_e * u[S->index_e(i, j, 0)]
                                + S->value_n * u[S->index_n(i, j, 0)]
                                + S->value_t * u[S->index_t(i, j, 0)];
    }
  }

  //face: k = nz-1, j=1,...,ny-2, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int j = 1; j < S->ny - 1; j++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, j, S->nz - 1)]  = S->value_b * u[S->index_b(i, j, S->nz - 1)]
                                        + S->value_s * u[S->index_s(i, j, S->nz - 1)]
                                        + S->value_w * u[S->index_w(i, j, S->nz - 1)]
                                        + S->value_c * u[S->index_c(i, j, S->nz - 1)]
                                        + S->value_e * u[S->index_e(i, j, S->nz - 1)]
                                        + S->value_n * u[S->index_n(i, j, S->nz - 1)];
    }
  }
  
  //face: k=1,...,nz-2, j=0, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, 0, k)]  = S->value_b * u[S->index_b(i, 0, k)]
                                + S->value_w * u[S->index_w(i, 0, k)]
                                + S->value_c * u[S->index_c(i, 0, k)]
                                + S->value_e * u[S->index_e(i, 0, k)]
                                + S->value_n * u[S->index_n(i, 0, k)]
                                + S->value_t * u[S->index_t(i, 0, k)];
    }
  }
        

  //face: k=1,...,nz-2, j=ny-1, i=1,...,nx-2
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int i = 1; i < S->nx - 1; i++)
    {
        v[S->index_c(i, S->ny - 1, k)]  = S->value_b * u[S->index_b(i, S->ny - 1, k)]
                                        + S->value_s * u[S->index_s(i, S->ny - 1, k)]
                                        + S->value_w * u[S->index_w(i, S->ny - 1, k)]
                                        + S->value_c * u[S->index_c(i, S->ny - 1, k)]
                                        + S->value_e * u[S->index_e(i, S->ny - 1, k)]
                                        + S->value_t * u[S->index_t(i, S->ny - 1, k)];
    }
  }

  //face: k=1,...,nz-2, j=1,...,ny-2, i=0
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
        v[S->index_c(0, j, k)]  = S->value_b * u[S->index_b(0, j, k)]
                                + S->value_s * u[S->index_s(0, j, k)]
                                + S->value_c * u[S->index_c(0, j, k)]
                                + S->value_e * u[S->index_e(0, j, k)]
                                + S->value_n * u[S->index_n(0, j, k)]
                                + S->value_t * u[S->index_t(0, j, k)];
    }
  }

  //face: k=1,...,nz-2, j=1,...,ny-2, i=nx-1
  #pragma omp parallel for schedule(static) //collapse(2) //<-for task 6 hw1, uncomment
  for (int k = 1; k < S->nz - 1; k++)
  {
    for (int j = 1; j < S->ny - 1; j++)
    {
        v[S->index_c(S->nx - 1, j, k)]  = S->value_b * u[S->index_b(S->nx - 1, j, k)]
                                        + S->value_s * u[S->index_s(S->nx - 1, j, k)]
                                        + S->value_w * u[S->index_w(S->nx - 1, j, k)]
                                        + S->value_c * u[S->index_c(S->nx - 1, j, k)]
                                        + S->value_n * u[S->index_n(S->nx - 1, j, k)]
                                        + S->value_t * u[S->index_t(S->nx - 1, j, k)];
    }
  }
  
  //lines: i=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < S->nx - 1; i++)
  {
    //k=0, j=0
    v[S->index_c(i, 0, 0)]  = S->value_w * u[S->index_w(i, 0, 0)]
                            + S->value_c * u[S->index_c(i, 0, 0)]
                            + S->value_e * u[S->index_e(i, 0, 0)]
                            + S->value_n * u[S->index_n(i, 0, 0)]
                            + S->value_t * u[S->index_t(i, 0, 0)];
    //k=0, j=ny-1
    v[S->index_c(i, S->ny - 1, 0)]  = S->value_s * u[S->index_s(i, S->ny - 1, 0)]
                                    + S->value_w * u[S->index_w(i, S->ny - 1, 0)]
                                    + S->value_c * u[S->index_c(i, S->ny - 1, 0)]
                                    + S->value_e * u[S->index_e(i, S->ny - 1, 0)]
                                    + S->value_t * u[S->index_t(i, S->ny - 1, 0)];
    
    //k=nz-1, j=0
    v[S->index_c(i, 0, S->nz - 1)]  = S->value_b * u[S->index_b(i, 0, S->nz - 1)]
                                    + S->value_w * u[S->index_w(i, 0, S->nz - 1)]
                                    + S->value_c * u[S->index_c(i, 0, S->nz - 1)]
                                    + S->value_e * u[S->index_e(i, 0, S->nz - 1)]
                                    + S->value_n * u[S->index_n(i, 0, S->nz - 1)];
    
    //k=nz-1, j=ny-1
    v[S->index_c(i, S->ny - 1, S->nz - 1)]  = S->value_b * u[S->index_b(i, S->ny - 1, S->nz - 1)]
                                            + S->value_s * u[S->index_s(i, S->ny - 1, S->nz - 1)]
                                            + S->value_w * u[S->index_w(i, S->ny - 1, S->nz - 1)]
                                            + S->value_c * u[S->index_c(i, S->ny - 1, S->nz - 1)]
                                            + S->value_e * u[S->index_e(i, S->ny - 1, S->nz - 1)];
  }
  
  //lines: j=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int j = 1; j < S->ny - 1; j++)
  {
    //k=0, i=0
    v[S->index_c(0, j, 0)]  = S->value_s * u[S->index_s(0, j, 0)]
                            + S->value_c * u[S->index_c(0, j, 0)]
                            + S->value_e * u[S->index_e(0, j, 0)]
                            + S->value_n * u[S->index_n(0, j, 0)]
                            + S->value_t * u[S->index_t(0, j, 0)];
    
    //k=0, i=ny-1
    v[S->index_c(S->nx - 1, j, 0)]  = S->value_s * u[S->index_s(S->nx - 1, j, 0)]
                                    + S->value_w * u[S->index_w(S->nx - 1, j, 0)]
                                    + S->value_c * u[S->index_c(S->nx - 1, j, 0)]
                                    + S->value_n * u[S->index_n(S->nx - 1, j, 0)]
                                    + S->value_t * u[S->index_t(S->nx - 1, j, 0)];
    
    //k=nz-1, i=0
    v[S->index_c(0, j, S->nz - 1)]  = S->value_b * u[S->index_b(0, j, S->nz - 1)]
                                    + S->value_s * u[S->index_s(0, j, S->nz - 1)]
                                    + S->value_c * u[S->index_c(0, j, S->nz - 1)]
                                    + S->value_e * u[S->index_e(0, j, S->nz - 1)]
                                    + S->value_n * u[S->index_n(0, j, S->nz - 1)];
    
    //k=nz-1, i=ny-1
    v[S->index_c(S->nx - 1, j, S->nz - 1)]  = S->value_b * u[S->index_b(S->nx - 1, j, S->nz - 1)]
                                            + S->value_s * u[S->index_s(S->nx - 1, j, S->nz - 1)]
                                            + S->value_w * u[S->index_w(S->nx - 1, j, S->nz - 1)]
                                            + S->value_c * u[S->index_c(S->nx - 1, j, S->nz - 1)]
                                            + S->value_n * u[S->index_n(S->nx - 1, j, S->nz - 1)];
  }
  
  //lines: k=1,..nx-2
  #pragma omp parallel for schedule(static)
  for (int k = 1; k < S->nz - 1; k++)
  {
    //j=0, i=0
    v[S->index_c(0, 0, k)]  = S->value_b * u[S->index_b(0, 0, k)]
                            + S->value_c * u[S->index_c(0, 0, k)]
                            + S->value_e * u[S->index_e(0, 0, k)]
                            + S->value_n * u[S->index_n(0, 0, k)]
                            + S->value_t * u[S->index_t(0, 0, k)];
    
    //j=0, i=nx-1
    v[S->index_c(S->nx - 1, 0, k)]  = S->value_b * u[S->index_b(S->nx - 1, 0, k)]
                                    + S->value_w * u[S->index_w(S->nx - 1, 0, k)]
                                    + S->value_c * u[S->index_c(S->nx - 1, 0, k)]
                                    + S->value_n * u[S->index_n(S->nx - 1, 0, k)]
                                    + S->value_t * u[S->index_t(S->nx - 1, 0, k)];
    
    //j=ny-1, i=0
    v[S->index_c(0, S->ny - 1, k)]  = S->value_b * u[S->index_b(0, S->ny - 1, k)]
                                    + S->value_s * u[S->index_s(0, S->ny - 1, k)]
                                    + S->value_c * u[S->index_c(0, S->ny - 1, k)]
                                    + S->value_e * u[S->index_e(0, S->ny - 1, k)]
                                    + S->value_t * u[S->index_t(0, S->ny - 1, k)];
    
    //j=ny-1, i=nx-1
    v[S->index_c(S->nx - 1, S->ny - 1, k)]  = S->value_b * u[S->index_b(S->nx - 1, S->ny - 1, k)]
                                            + S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, k)]
                                            + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, k)]
                                            + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, k)]
                                            + S->value_t * u[S->index_t(S->nx - 1, S->ny - 1, k)];
    
  }

  //corner: k=0,j=0,i=0
  v[S->index_c(0, 0, 0)]    = S->value_c * u[S->index_c(0, 0, 0)] 
                            + S->value_e * u[S->index_e(0, 0, 0)] 
                            + S->value_n * u[S->index_n(0, 0, 0)] 
                            + S->value_t * u[S->index_t(0, 0, 0)];
      
  //corner: k=0,j=0,i=nx-1
  v[S->index_c(S->nx - 1, 0, 0)]    = S->value_w * u[S->index_w(S->nx - 1, 0, 0)] 
                                    + S->value_c * u[S->index_c(S->nx - 1, 0, 0)] 
                                    + S->value_n * u[S->index_n(S->nx - 1, 0, 0)] 
                                    + S->value_t * u[S->index_t(S->nx - 1, 0, 0)];

 //corner: k=0,j=ny-1,i=0
  v[S->index_c(0, S->ny - 1, 0)]    =  S->value_s * u[S->index_s(0, S->ny - 1, 0)] 
                                    + S->value_c * u[S->index_c(0, S->ny - 1, 0)] 
                                    +  S->value_e * u[S->index_e(0, S->ny - 1, 0)] 
                                    + S->value_t * u[S->index_t(0, S->ny - 1, 0)];
      
  //corner: k=0,j=ny-1,i=nx-1
  v[S->index_c(S->nx - 1, S->ny - 1, 0)]    = S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, 0)] 
                                            + S->value_t * u[S->index_t(S->nx - 1, S->ny - 1, 0)];
      
  //corner: k=nz-1,j=0,i=0
  v[S->index_c(0, 0, S->nz - 1)]    = S->value_b * u[S->index_b(0, 0, S->nz - 1)] 
                                    + S->value_c * u[S->index_c(0, 0, S->nz - 1)] 
                                    + S->value_e * u[S->index_e(0, 0, S->nz - 1)] 
                                    + S->value_n * u[S->index_n(0, 0, S->nz - 1)];
      
  //corner: k=nz-1,j=0,i=nx-1
  v[S->index_c(S->nx - 1, 0, S->nz - 1)]    = S->value_b * u[S->index_b(S->nx - 1, 0, S->nz - 1)] 
                                            + S->value_w * u[S->index_w(S->nx - 1, 0, S->nz - 1)]
                                            + S->value_c * u[S->index_c(S->nx - 1, 0, S->nz - 1)] 
                                            + S->value_n * u[S->index_n(S->nx - 1, 0, S->nz - 1)];
      

  //corner: k=nz-1,j=ny-1,i=0
  v[S->index_c(0, S->ny - 1, S->nz - 1)]    = S->value_b * u[S->index_b(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_s * u[S->index_s(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_c * u[S->index_c(0, S->ny - 1, S->nz - 1)] 
                                            + S->value_e * u[S->index_e(0, S->ny - 1, S->nz - 1)];

  //corner: k=nz-1,j=ny-1,i=nx-1
  v[S->index_c(S->nx - 1, S->ny - 1, S->nz - 1)]    = S->value_b * u[S->index_b(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_s * u[S->index_s(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_w * u[S->index_w(S->nx - 1, S->ny - 1, S->nz - 1)] 
                                                    + S->value_c * u[S->index_c(S->nx - 1, S->ny - 1, S->nz - 1)];
  
  return;
}

void copy(int n, double const* u, double* v)
{
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; i++)
    {
        v[i] = u[i];   
    }
}

void apply_jacobi_pre(stencil3d const* S,
        double const* r, double* z)
{
  int n = S->nx * S->ny * S->nz;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
  {
      z[i] = r[i] / S->value_c;
  }
  return;
}


void identity(stencil3d const* S, double const* r, double* z)
{
    stencil3d op2;
    op2.nx = S->nx;
    op2.ny = S->ny;
    op2.nz = S->nz;
    op2.value_c = 1.0;
    op2.value_n = 0.0;
    op2.value_s = 0.0;
    op2.value_e = 0.0;
    op2.value_w = 0.0;
    op2.value_t = 0.0;
    op2.value_b = 0.0;
    apply_stencil3d(&op2, r, z);
    return;
}

void apply_jacobi_iterations(stencil3d const* S,
        double const* r, double* z, int iter_max)
{
    int n = S->nx * S->ny * S->nz;
    double *sigma = new double[n];

    //z_0 = 1/c * r   initial guess
    axy(n, 1.0/(S->value_c), r, z);

    for (int k=0; k<iter_max; k++)
    {
        //sigma = Az
        apply_stencil3d(S,z,sigma);
        
        //z = D^{-1} (r - (A-D)z ) = z + (r - Az)/c, c = a_ii (constant)
        #pragma omp parallel for schedule(static)
        for (int i=0; i<n; i++)
        {
            z[i] += (r[i] - sigma[i])/S->value_c;   
        }
    }
    delete [] sigma;
    return;
}

void apply_gauss_seidel(stencil3d const* S, 
double const* r, double* z, int iter_max)
{

int n = S->nx * S->ny * S->nz;
double *sigma_L = new double[n];
double *sigma_U = new double[n];
double *phi = new double[n];

    stencil3d L;
    L.nx = S->nx;
    L.ny = S->ny;
    L.nz = S->nz;
    L.value_c = 0.0;
    L.value_n = 0.0;
    L.value_s = S->value_s;
    L.value_e = 0.0;
    L.value_w = S->value_w;
    L.value_t = 0.0;
    L.value_b = S->value_b;
    
    stencil3d U;
    U.nx = S->nx;
    U.ny = S->ny;
    U.nz = S->nz;
    U.value_c = 0.0;
    U.value_n = S->value_n;
    U.value_s = 0.0;
    U.value_e = S->value_e;
    U.value_w = 0.0;
    U.value_t = S->value_t;
    U.value_b = 0.0;

init(n, phi, 0.0);

for (int k=0; k<iter_max; k++)
{

    apply_stencil3d(&U,phi,sigma_U);
    
    apply_stencil3d(&L,z,sigma_L);
    
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; i++)
    {
        phi[i] = (r[i] - sigma_L[i] - sigma_U[i])/S->value_c;   
    }
        
    copy(n,phi,z);
}

delete [] sigma_L;
delete [] sigma_U;
delete [] phi;
return;
}


std::pair<double,double> explicit_eigenvalues(stencil3d const* S)
{

// this only holds for the 3D Poisson
    
    int n = S->nx * S->ny * S->nz;
    
    double dx=1.0/(S->nx-1), dy=1.0/(S->ny-1), dz=1.0/(S->nz-1);
    
    double *eigenval = new double[n];
    
    #pragma omp parallel 
    #pragma omp for ordered
    for (int k = 1; k < S->nz - 1; k++)
    {
        for (int j = 1; j < S->ny - 1; j++)
        {
            for (int i = 1; i < S->nx - 1; i++)
            {
                #pragma omp ordered
                eigenval[S->index_c(i,j,k)] = 4*(sin(M_PI*i/(2*S->nx))*sin(M_PI*i/(2*S->nx)) + sin(M_PI*j/(2*S->ny))*sin(M_PI*j/(2*S->ny)) + sin(M_PI*k/(2*S->nz))*sin(M_PI*k/(2*S->nz)))/(dx*dx);
            }
        }
    }
    
    // Find the minimum and maximum eigenvalues in a simple way
    double alpha = eigenval[0];
    double beta = eigenval[0];

    #pragma omp parallel for reduction(min:alpha) reduction(max:beta)
    for (int l = 1; l < n; l++) {
        if (eigenval[l] < alpha) {
            alpha = eigenval[l];
        }
        if (eigenval[l] > beta) {
            beta = eigenval[l];
        }
    }


    delete[] eigenval;
    
    return {alpha, beta};
}



std::pair<double,double> extremal_eigenvalues(stencil3d const* S, int iter_max)
{
    int n = S->nx * S->ny * S->nz;
    double *q = new double[n];
    double *p = new double[n];
    double kappa;
    double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    double alpha = 0.0;
    double beta = 0.0;
    double alpha_old = -100.0;
    double beta_old = -100.0;
    //initialize q such that ||q||_2 = 1
    init(n, q, 1.0/std::sqrt(n));
    
    //p = S*q
    apply_stencil3d(S, q, p);
    
    //matrix S is SPD-> positive eigenvalues
    //power method computes largest eigenvalue in absolute value, note //max |eigenvalue| = max eigenvalue
    //since matrix is SPD, shift B=A-lambda_max I computes smallest eigenvalue in absolute value, note //min |eigenvalue| = min eigenvalue
    
    //power method to find largest eigenvalue
    double iter; 
    iter = -1;
    while (abs(beta - beta_old) > tol || iter < iter_max) 
    {
        iter++;
        //kappa = ||p||_2
        kappa = dot(n,p,p);
        kappa = std::sqrt(kappa);
        //q = p / kappa
        axy(n, 1.0/kappa, p, q);
        //beta_old = beta
        beta_old = beta;
        //lambda_max = <p, q>
        beta = dot(n,p,q);
        //p = S*q
        apply_stencil3d(S, q, p);
    }
    
    //shifted power method to find smallest eigenvalue
    //op = S - lambda_max I
    stencil3d op;
    op.nx = S->nx;
    op.ny = S->ny;
    op.nz = S->nz;
    op.value_c = S->value_c;// - beta;
    op.value_n = S->value_n;
    op.value_s = S->value_s;
    op.value_e = S->value_e;
    op.value_w = S->value_w;
    op.value_t = S->value_t;
    op.value_b = S->value_b;
    
    //initialize q such that ||q||_2 = 1
    init(n,q, 1.0/std::sqrt(n));
    
    //p = op*q
    apply_stencil3d(&op, q, p);
    iter = -1;
    while (abs(alpha - alpha_old) > tol || iter < iter_max) 
    {
        iter++;
        //kappa = ||p||_2
        kappa = dot(n,p,p);
        kappa = std::sqrt(kappa);
        //q = p / kappa
        axy(n, 1.0/kappa, p, q);
        //alpha_old = alpha
        alpha_old = alpha;
        //lambda_max = <p, q>
        alpha = dot(n,p,q);
        //p = op*q
        apply_stencil3d(&op, q, p);
    }
    delete [] p;
    delete [] q;
    //return alpha := lambda_min(S), beta := lambda_max(S)
    return {alpha, beta};

}

void apply_cheb(stencil3d const* S, double const* r, double* z, int iter_max, double const alpha, double const beta)
{
    int n = S->nx * S->ny * S->nz;
    double *y = new double[n];
    double *d = new double[n];
    double *r_star = new double[n];
    
    //z_0 = 0;
    init(n,z,0.0);

    //r_star_0 = r - Az_0, z_0 = 0 -> Az_0 = 0 -> r_star_0 = r, (no apply_stencil3d needed)
    copy(n, r, r_star);
    
    //auto [alpha, beta] = extremal_eigenvalues(S, n);
    double delta = 0.5*(beta - alpha);
    double theta = 0.5*(beta + alpha);
    double sigma = theta / delta; //
    
    //kappa_0 = 1/sigma
    double kappa_old = 1.0 / sigma;
    double kappa;
    //d_{0} = 1/theta * r
    axy(n,  1.0/theta, r_star, d);
    
    for (int k=0; k<iter_max; k++)
    {
        //z_{k+1} = z_{k} + d_{k}
        axpby(n,1.0, d, 1.0, z);
        
        //y_{k} = A d_{k}
        apply_stencil3d(S,d,y);
        
        //r_{k+1} = r_{k} - y_{k}
        axpby(n, - 1.0, y, 1.0, r_star);
        
        //kappa_{k+1} = 1/(2*sigma - kappa_{k})
        kappa = 1.0 / (2.0*sigma - kappa_old);
        
        //d_{k+1} = kappa_{k+1} \kappa_{k} d_k - 2*kappa_{k+1}/delta * r_{k+1}
        axpby(n, - 2.0 * kappa / delta, r_star, kappa*kappa_old, d);
        
        //kappa_{k} = \kappa_{k+1}
        kappa_old = kappa;
    }
    
    delete [] y;
    delete [] d;
    delete [] r_star;
    return;
}
