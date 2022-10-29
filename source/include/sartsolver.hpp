
#pragma once

#include "raytransfer.hpp"
#include "laplacian.hpp"

#include "mpi.h"


class BaseSARTSolverMPI {

protected:
    RayTransferMatrix &raytransfer;
    LaplacianMatrix &laplacian;
    std::vector<double> ray_density;
    std::vector<double> ray_length;
    double ray_dens_thres;
    double ray_length_thres;
    double conv_tol;
    double beta_laplace;
    double relaxation;
    int max_iterations;
    MPI_Comm mpi_comm;

public:    
    static const int SUCCESS;
    static const int MAX_ITERATIONS_EXCEEDED;

    double get_ray_density_threshold() const;
    double get_ray_length_threshold() const;
    double get_convolution_tolerance() const;
    double get_beta_laplace() const;
    double get_relaxation() const;
    int get_max_iterations() const;

    void set_ray_density_threshold(double value);
    void set_ray_length_threshold(double value);
    void set_convolution_tolerance(double value);
    void set_beta_laplace(double value);
    void set_relaxation(double value);
    void set_max_iterations(int value);

    virtual int solve(std::vector<double>& solution,
                      const std::vector<double>& measurement) const;

    BaseSARTSolverMPI(RayTransferMatrix& rtm, LaplacianMatrix &lm,
                      double ray_density_threshold=1.e-6, double ray_length_threshold=1.e-6,
                      double convolution_tolerance=1.e-5, double beta_laplace=1.e-2,
                      double relaxation=1., int max_iterations=2000,
                      MPI_Comm mpi_communicator=MPI_COMM_WORLD);

    virtual ~BaseSARTSolverMPI() {};

};


class SARTSolverMPI: public BaseSARTSolverMPI {

public:
    int solve(std::vector<double>& solution,
              const std::vector<double>& measurement) const;

    using BaseSARTSolverMPI::BaseSARTSolverMPI;
};


class LogSARTSolverMPI: public BaseSARTSolverMPI {

public:
    int solve(std::vector<double>& solution,
              const std::vector<double>& measurement) const;

    using BaseSARTSolverMPI::BaseSARTSolverMPI;
};
