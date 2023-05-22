#!/bin/bash
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450 --reservation=wi4450

# run CG for a n^3 Poisson problem
n=32
export OMP_NUM_THREADS=48
export  OMP_PROC_BIND=close
export  OMP_PLACES=cores

srun ./main_cg_poisson_jacobi.x $n >> cg_poisson_jacobi_500it_$n$OMP_NUM_THREADS.out
srun ./main_cg_poisson.x $n >> cg_poisson_$n$OMP_NUM_THREADS.out
