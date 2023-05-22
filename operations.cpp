#include "operations.hpp"
#include <omp.h>

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

void apply_jacobi_pre(stencil3d const* S,
        double const* u, double* v)
{
  int n = S->nx * S->ny * S->nz;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
  {
      v[i] = u[i] / S->value_c;
  }
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

void identity(stencil3d const* S, double const* u, double* v)
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
    apply_stencil3d(&op2, u, v);
    return;
}

void apply_jacobi_iterations(stencil3d const* S,
        double const* u, double* v, int iter_max)
{
    
    stencil3d op2;
    op2.nx = S->nx;
    op2.ny = S->ny;
    op2.nz = S->nz;
    op2.value_c = 0.0;
    op2.value_n = S->value_n;
    op2.value_s = S->value_s;
    op2.value_e = S->value_e;
    op2.value_w = S->value_w;
    op2.value_t = S->value_t;
    op2.value_b = S->value_b;
    
    int n = S->nx * S->ny * S->nz;
    double *sigma = new double[n];
    for (int k=0; k<iter_max; k++)
    {
        //sigma(i) = sum_nodiag A(i,j,k)*v(loc(i,j,k))
        apply_stencil3d(&op2,v,sigma);
        
        //u = b - Ax
        //sigma =(u - sigma)/c, c = a_ii
        #pragma omp parallel for schedule(static)
        for (int i=0; i<n; i++)
        {
            v[i] = (u[i] - sigma[i])/S->value_c;   
        }
    }
    delete [] sigma;
    return;
}

void apply_gauss_seidel(stencil3d const* S, 
double const* u, double* v, int iter_max)
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
    
    apply_stencil3d(&L,v,sigma_L);
    
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; i++)
    {
        phi[i] = (u[i] - sigma_L[i] - sigma_U[i])/S->value_c;   
    }
        
    copy(n,phi,v);
}

delete [] sigma_L;
delete [] sigma_U;
delete [] phi;
return;
}