// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include "sartsolver.hpp"
#include "cublas_v2.h"


class BaseSARTSolverMPICuda: public BaseSARTSolverMPI {

protected:
    cublasHandle_t cublas_handle;
    float* dev_rtm;
    float* dev_laplace_val;
    size_t* dev_laplace_idx;
    float* dev_ray_length;
    float* dev_ray_density;

    int pre_iteration_setup(std::vector<double>& solution, const std::vector<double>& measurement,
                            std::vector<float>& host_solution, double& norm, double& measurement_squared,
                            float** dev_measured, float** dev_solution, float** dev_fitted, float** dev_gradpen) const;

public:    

    BaseSARTSolverMPICuda(RayTransferMatrix& rtm, LaplacianMatrix &lm, int rank=0,
                          double ray_density_threshold=1.e-6, double ray_length_threshold=1.e-6,
                          double convolution_tolerance=1.e-5, double beta_laplace=1.e-2,
                          double relaxation=1., int max_iterations=2000,
                          MPI_Comm mpi_communicator=MPI_COMM_WORLD);

    ~BaseSARTSolverMPICuda();

};


class SARTSolverMPICuda: public BaseSARTSolverMPICuda {

public:
    int solve(std::vector<double>& solution,
              const std::vector<double>& measurement) const;

    using BaseSARTSolverMPICuda::BaseSARTSolverMPICuda;
};


class LogSARTSolverMPICuda: public BaseSARTSolverMPICuda {

public:
    int solve(std::vector<double>& solution,
              const std::vector<double>& measurement) const;

    using BaseSARTSolverMPICuda::BaseSARTSolverMPICuda;
};
