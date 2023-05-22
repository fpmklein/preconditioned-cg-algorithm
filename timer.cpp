
#include "timer.hpp"

#include <iomanip>

// for timing routine
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <cmath>

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> Timer::counts_;
std::map<std::string, double> Timer::times_;
std::map<std::string, double> Timer::flops_;
std::map<std::string, double> Timer::bytes_;

  Timer::Timer(std::string label, int nx, int ny, int nz)
  : label_(label)
  {
    t_start_ = omp_get_wtime();
    nx_ = nx; 
    ny_ = ny; 
    nz_ = nz;
    n_ = nx_*ny_*nz_;
  }


  Timer::~Timer()
  {
    double t_end = omp_get_wtime();
    times_[label_] += t_end - t_start_;
    counts_[label_]++;
    if (label_ == "init")
    {
        flops_[label_] = 0.0; 
        bytes_[label_] = 1.0 * sizeof(double) * n_;
    }
    else if (label_ == "apply_stencil3d")
    {
        flops_[label_] = (6 + 7)*(nx_ - 2)*(ny_ - 2)*(nz_ - 2) + (5 + 6)*2*((nx_ - 2)*(ny_ - 2)+(nx_ - 2)*(nz_ - 2)+(nz_ - 2)*(ny_ - 2)) 
            + (4 + 5)*4*((nx_ - 2) + (ny_ - 2) + (nz_ - 2)) + (3 + 4)*8;
        bytes_[label_] = 1.0 * 8.0 + 2.0 * n_ * sizeof(double);
    }
    else if (label_ == "axpby")
    {
        flops_[label_] = 3.0 * n_;
        bytes_[label_] = 3.0 * sizeof(double) * n_;
    }
    else if (label_ == "dot")
    {
        flops_[label_] = 2.0 * n_;
        bytes_[label_] = 2.0 * sizeof(double) * n_;
    }
    else if (label_ == "copy")
    {
        flops_[label_] = 0.0;
        bytes_[label_] = 1.0 * n_;
    }
    else
    {
        flops_[label_] = 0.0;
        bytes_[label_] = 0.0;
    }
    //flops_[label_] = m; //m is double
    //bytes_[label_] = b;  //b is double
  }

void Timer::summarize(std::ostream& os)
{
  
  os << "==================== TIMER SUMMARY =========================================" << std::endl;
  os << "label               \tcalls     \ttotal time\tmean time\tcomp. intensity\t mean Gflop/s\tmean Gbyte/s"<<std::endl;
  os << "----------------------------------------------" << std::endl;
  double convert = std::pow(10.0,9);
  unsigned int n_threads = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 1);
  for (auto [label, time]: times_)
  {
    int count = counts_[label];
    double gflops = flops_[label] / convert; //convert flop to Gflop
    double gbytes = bytes_[label] / convert; //convert byte to Gbyte
    double mean_time = time/double(count);
    std::cout << std::setw(20) << label << "\t" << std::setw(10) << count << "\t" << std::setw(10) << time << "\t" << std::setw(10) 
        << mean_time << "\t" << std::setw(10) << gflops/gbytes << "\t" << std::setw(10) << gflops/mean_time <<"\t" << std::setw(10) 
        << gbytes/mean_time << std::endl;
  }
  os << "============================================================================" << std::endl;
}
