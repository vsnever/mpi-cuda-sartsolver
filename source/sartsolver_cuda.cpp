
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "sartsolver_cuda.hpp"

#ifndef EPSILON_LOG_CUDA
#define EPSILON_LOG_CUDA 0.0000001f
#endif

extern "C" 
{
    void CallInitialGuessKernel(float* const solution, const float* const rtm, const float* const measured, 
                                const float* const ray_density, const float ray_dens_thres, const size_t npixel, const size_t nvoxel);

    void CallGradPenaltyKernel(float* const grad_penalty, const float* const solution, const size_t* const laplace_idx,
                               const float* const laplace_val, float beta_laplace, size_t laplacian_size, size_t nvoxel);

    void CallLogGradPenaltyKernel(float* const grad_penalty, const float* const solution, const size_t* const laplace_idx,
                                  const float* const laplace_val, float beta_laplace, size_t laplacian_size, size_t nvoxel);

    void CallPropagateKernel(float* const diff, const float* const rtm, const float* const measured, const float* const fitted,
                             const float* const ray_density, const float* const ray_length, const float* const grad_penalty,
                             float relaxation, float ray_dens_thres, float ray_length_thres, size_t npixel, size_t nvoxel);

    void CallLogPropagateKernel(float* const ofs_fit, const float* const rtm, const float* const measured, const float* const fitted,
                                const float* const ray_density, const float* const ray_length,
                                float ray_dens_thres, float ray_length_thres, size_t npixel, size_t nvoxel);

    void CallUpdateSolutionKernel(float* const solution, float* const diff, size_t nvoxel);

    void CallUpdateLogSolutionKernel(float* const solution, float* const ofs_fit, const float* const grad_penalty,
                                     float relaxation, size_t nvoxel);
}


#define SafeCudaCall(call) do \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "Cuda error in file " <<  __FILE__ << " in line " << __LINE__ << ".\n"; \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
} while(0)


#define SafeCublasCall(call) do \
{ \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CuBLAS error in file " <<  __FILE__ << " in line " << __LINE__ << ".\n"; \
        std::cerr << cublasGetStatusString(status) << std::endl; \
        std::exit(1); \
    } \
} while(0)


BaseSARTSolverMPICuda::BaseSARTSolverMPICuda(RayTransferMatrix& rtm, LaplacianMatrix &lm, int rank,
                                             double ray_density_threshold, double ray_length_threshold,
                                             double convolution_tolerance, double beta_laplace,
                                             double relaxation, int max_iterations,
                                             MPI_Comm mpi_communicator) :
                       BaseSARTSolverMPI(rtm, lm, ray_density_threshold, ray_length_threshold, convolution_tolerance,
                                         beta_laplace, relaxation, max_iterations, mpi_communicator) {

    dev_rtm = NULL;
    dev_laplace_val = NULL;
    dev_laplace_idx = NULL;
    dev_ray_length = NULL;
    dev_ray_density = NULL;

    const size_t npixel = raytransfer.npixel();
    const size_t nvoxel = raytransfer.nvoxel();

    // device initialization
    int device_count = 0;
    SafeCudaCall(cudaGetDeviceCount(&device_count));
    SafeCudaCall(cudaSetDevice(rank % device_count));

    // cuBLAS initialization
    SafeCublasCall(cublasCreate(&cublas_handle));

    // copy raytransfer matrix to device
    const size_t rtm_size = nvoxel * npixel;
    SafeCudaCall(cudaMalloc(&dev_rtm, rtm_size * sizeof(float)));
    SafeCudaCall(cudaMemcpy(dev_rtm, raytransfer.matrix().data(), rtm_size * sizeof(float), cudaMemcpyHostToDevice));

    // copy laplacian matrix to device
    const size_t laplace_size = laplacian.size();
    if (laplace_size) {
        SafeCudaCall(cudaMalloc(&dev_laplace_val, laplace_size * sizeof(float)));
        SafeCudaCall(cudaMemcpy(dev_laplace_val, laplacian.value().data(), laplace_size * sizeof(float), cudaMemcpyHostToDevice));
        SafeCudaCall(cudaMalloc(&dev_laplace_idx, laplace_size * sizeof(size_t)));
        SafeCudaCall(cudaMemcpy(dev_laplace_idx, laplacian.index().data(), laplace_size * sizeof(size_t), cudaMemcpyHostToDevice));
    }

    // copy ray lengths and density to device
    std::vector<float> host_ray_length(ray_length.begin(), ray_length.end()); // double to float
    SafeCudaCall(cudaMalloc(&dev_ray_length, host_ray_length.size() * sizeof(float)));
    SafeCudaCall(cudaMemcpy(dev_ray_length, host_ray_length.data(), host_ray_length.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> host_ray_density(ray_density.begin(), ray_density.end()); // double to float
    SafeCudaCall(cudaMalloc(&dev_ray_density, host_ray_density.size() * sizeof(float)));
    SafeCudaCall(cudaMemcpy(dev_ray_density, host_ray_density.data(), host_ray_density.size() * sizeof(float), cudaMemcpyHostToDevice));

}

BaseSARTSolverMPICuda::~BaseSARTSolverMPICuda(){
    if (dev_rtm) cudaFree(dev_rtm);
    if (dev_laplace_val) cudaFree(dev_laplace_val);
    if (dev_laplace_idx) cudaFree(dev_laplace_idx);
    if (dev_ray_length) cudaFree(dev_ray_length);
    if (dev_ray_density) cudaFree(dev_ray_density);
    SafeCublasCall(cublasDestroy(cublas_handle));
}


int BaseSARTSolverMPICuda::pre_iteration_setup(std::vector<double>& solution, const std::vector<double>& measurement,
                                               std::vector<float>& host_solution, double& norm, double& measurement_squared,
                                               float** dev_measured, float** dev_solution, float** dev_fitted, float** dev_gradpen) const {

    const size_t nvoxel = raytransfer.nvoxel();
    const size_t npixel = raytransfer.npixel();

    std::vector<float> host_measured(npixel);
    // normalization to avoid float overflow when calculating fitted_squared
    const double norm_loc = *(std::max_element(measurement.begin(), measurement.end()));
    norm = 0;
    MPI_Allreduce(&norm_loc, &norm, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    std::transform(measurement.begin(), measurement.end(), host_measured.begin(), [&norm](auto &v){ return (float)(v / norm); });

    // calculating measurement squared and normalizing
    measurement_squared = 0;
    double measurement_squared_loc = 0;
    for (auto m : measurement) {measurement_squared_loc += m * m;}
    MPI_Allreduce(&measurement_squared_loc, &measurement_squared, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    measurement_squared /= norm * norm;

    // copy measured to device
    SafeCudaCall(cudaMalloc(dev_measured, npixel * sizeof(float)));
    SafeCudaCall(cudaMemcpy(*dev_measured, host_measured.data(), npixel * sizeof(float), cudaMemcpyHostToDevice));

    // copy solution to deivce
    SafeCudaCall(cudaMalloc(dev_solution, nvoxel * sizeof(float)));

    if (solution.empty()) { // fill solution with initial guess
        solution.resize(nvoxel);  // resize for later use

        cudaMemset(*dev_solution, 0, nvoxel * sizeof(float));
        CallInitialGuessKernel(*dev_solution, dev_rtm, *dev_measured, dev_ray_density, (float)ray_dens_thres, npixel, nvoxel);

        std::vector<float> host_solution_loc(nvoxel);
        SafeCudaCall(cudaMemcpy(host_solution_loc.data(), *dev_solution, nvoxel * sizeof(float), cudaMemcpyDeviceToHost));
        MPI_Allreduce(host_solution_loc.data(), host_solution.data(), nvoxel, MPI_FLOAT, MPI_SUM, mpi_comm);
    }
    else {
        std::transform(solution.begin(), solution.end(), host_solution.begin(), [norm](double &v){ return (float)(v / norm); });
    }

    for (auto &s : host_solution) {if (s < EPSILON_LOG_CUDA) s = EPSILON_LOG_CUDA;}

    SafeCudaCall(cudaMemcpy(*dev_solution, host_solution.data(), nvoxel * sizeof(float), cudaMemcpyHostToDevice));

    // calculate fitted image
    const float alpha_gemv = 1;
    const float beta_gemv = 0;
    SafeCudaCall(cudaMalloc(dev_fitted, npixel * sizeof(float)));
    SafeCublasCall(cublasSgemv(cublas_handle, CUBLAS_OP_T, (int)nvoxel, (int)npixel,
                               &alpha_gemv, dev_rtm, (int)nvoxel, *dev_solution, 1, &beta_gemv, *dev_fitted, 1));

    SafeCudaCall(cudaMalloc(dev_gradpen, nvoxel * sizeof(float)));

    return 0;
}
    

int SARTSolverMPICuda::solve(std::vector<double>& solution,
                             const std::vector<double>& measurement) const{

    float* dev_measured = NULL;
    float* dev_fitted = NULL;
    float* dev_solution = NULL;
    float* dev_gradpen = NULL;
    float* dev_diff = NULL;

    const size_t nvoxel = raytransfer.nvoxel();
    const size_t npixel = raytransfer.npixel();

    if (!solution.empty() && (solution.size() != nvoxel)) {
        std::cerr << "Solution vector must be empty or contain nvoxel elements." << std::endl;
        std::exit(1);
    }

    std::vector<float> host_solution(nvoxel);
    double norm = 0;
    double measurement_squared = 0;

    pre_iteration_setup(solution, measurement, host_solution, norm, measurement_squared,
                        &dev_measured, &dev_solution, &dev_fitted, &dev_gradpen);

    std::vector<float> host_diff_loc(nvoxel);
    std::vector<float> host_diff(nvoxel);
    SafeCudaCall(cudaMalloc(&dev_diff, nvoxel * sizeof(float)));
    cudaMemset(dev_diff, 0, nvoxel * sizeof(float));

    const float alpha_gemv = 1;
    const float beta_gemv = 0;

    double convergence_prev = 0;
    int iter = 0;
    for (iter=0; iter<max_iterations; ++iter) {

        std::cout << iter << std::endl;

        cudaMemset(dev_gradpen, 0, nvoxel * sizeof(float));
        if (laplacian.size()) {
            CallGradPenaltyKernel(dev_gradpen, dev_solution, dev_laplace_idx, dev_laplace_val, (float)beta_laplace, laplacian.size(), nvoxel);
        }

        CallPropagateKernel(dev_diff, dev_rtm, dev_measured, dev_fitted, dev_ray_density, dev_ray_length, dev_gradpen,
                            (float)relaxation, (float)ray_dens_thres, (float)ray_length_thres, npixel, nvoxel);

        SafeCudaCall(cudaMemcpy(host_diff_loc.data(), dev_diff, nvoxel * sizeof(float), cudaMemcpyDeviceToHost));
        MPI_Allreduce(host_diff_loc.data(), host_diff.data(), nvoxel, MPI_FLOAT, MPI_SUM, mpi_comm);
        SafeCudaCall(cudaMemcpy(dev_diff, host_diff.data(), nvoxel * sizeof(float), cudaMemcpyHostToDevice));

        CallUpdateSolutionKernel(dev_solution, dev_diff, nvoxel);

        SafeCublasCall(cublasSgemv(cublas_handle, CUBLAS_OP_T, (int)nvoxel, (int)npixel,
                                   &alpha_gemv, dev_rtm, (int)nvoxel, dev_solution, 1, &beta_gemv, dev_fitted, 1));

        float fitted_squared = 0;
        float fitted_squared_loc = 0;
        SafeCublasCall(cublasSdot(cublas_handle, (int)npixel, dev_fitted, 1, dev_fitted, 1, &fitted_squared_loc));

        MPI_Allreduce(&fitted_squared_loc, &fitted_squared, 1, MPI_FLOAT, MPI_SUM, mpi_comm);

        const double convergence = (measurement_squared - (double)fitted_squared) / measurement_squared;

        if ((iter) && (std::abs(convergence - convergence_prev) < conv_tol)) break;

        convergence_prev = convergence;
    }

    SafeCudaCall(cudaMemcpy(host_solution.data(), dev_solution, nvoxel * sizeof(float), cudaMemcpyDeviceToHost));
    std::transform(host_solution.begin(), host_solution.end(), solution.begin(), [&norm](auto &v){ return norm * (double)v; });

    if (dev_measured) cudaFree(dev_measured);
    if (dev_fitted) cudaFree(dev_fitted);
    if (dev_solution) cudaFree(dev_solution);
    if (dev_gradpen) cudaFree(dev_gradpen);
    if (dev_diff) cudaFree(dev_diff);

    return (iter < max_iterations) ? SUCCESS : MAX_ITERATIONS_EXCEEDED;
}


int LogSARTSolverMPICuda::solve(std::vector<double>& solution,
                                const std::vector<double>& measurement) const{

    float* dev_measured = NULL;
    float* dev_fitted = NULL;
    float* dev_solution = NULL;
    float* dev_gradpen = NULL;
    float* dev_obs_fit = NULL;

    const size_t nvoxel = raytransfer.nvoxel();
    const size_t npixel = raytransfer.npixel();

    if (!solution.empty() && (solution.size() != nvoxel)) {
        std::cerr << "Solution vector must be empty or contain nvoxel elements." << std::endl;
        std::exit(1);
    }

    std::vector<float> host_solution(nvoxel);
    double norm = 0;
    double measurement_squared = 0;

    pre_iteration_setup(solution, measurement, host_solution, norm, measurement_squared,
                        &dev_measured, &dev_solution, &dev_fitted, &dev_gradpen);

    std::vector<float> host_obs_fit_loc(2 * nvoxel);
    std::vector<float> host_obs_fit(2 * nvoxel);
    SafeCudaCall(cudaMalloc(&dev_obs_fit, 2 * nvoxel * sizeof(float)));
    cudaMemset(dev_obs_fit, 0, 2 * nvoxel * sizeof(float));

    const float alpha_gemv = 1;
    const float beta_gemv = 0;

    double convergence_prev = 0;
    int iter = 0;
    for (iter=0; iter<max_iterations; ++iter) {

        std::cout << iter << std::endl;

        cudaMemset(dev_gradpen, 0, nvoxel * sizeof(float));
        if (laplacian.size()) {
            CallLogGradPenaltyKernel(dev_gradpen, dev_solution, dev_laplace_idx, dev_laplace_val, (float)beta_laplace, laplacian.size(), nvoxel);
        }

        CallLogPropagateKernel(dev_obs_fit, dev_rtm, dev_measured, dev_fitted, dev_ray_density, dev_ray_length,
                               (float)ray_dens_thres, (float)ray_length_thres, npixel, nvoxel);

        SafeCudaCall(cudaMemcpy(host_obs_fit_loc.data(), dev_obs_fit, 2 * nvoxel * sizeof(float), cudaMemcpyDeviceToHost));
        MPI_Allreduce(host_obs_fit_loc.data(), host_obs_fit.data(), 2 * nvoxel, MPI_FLOAT, MPI_SUM, mpi_comm);
        SafeCudaCall(cudaMemcpy(dev_obs_fit, host_obs_fit.data(), 2 * nvoxel * sizeof(float), cudaMemcpyHostToDevice));

        CallUpdateLogSolutionKernel(dev_solution, dev_obs_fit, dev_gradpen, (float)relaxation, nvoxel);

        SafeCublasCall(cublasSgemv(cublas_handle, CUBLAS_OP_T, (int)nvoxel, (int)npixel,
                                   &alpha_gemv, dev_rtm, (int)nvoxel, dev_solution, 1, &beta_gemv, dev_fitted, 1));

        float fitted_squared = 0;
        float fitted_squared_loc = 0;
        SafeCublasCall(cublasSdot(cublas_handle, (int)npixel, dev_fitted, 1, dev_fitted, 1, &fitted_squared_loc));

        MPI_Allreduce(&fitted_squared_loc, &fitted_squared, 1, MPI_FLOAT, MPI_SUM, mpi_comm);

        const double convergence = (measurement_squared - (double)fitted_squared) / measurement_squared;

        if ((iter) && (std::abs(convergence - convergence_prev) < conv_tol)) break;

        convergence_prev = convergence;
    }

    SafeCudaCall(cudaMemcpy(host_solution.data(), dev_solution, nvoxel * sizeof(float), cudaMemcpyDeviceToHost));
    std::transform(host_solution.begin(), host_solution.end(), solution.begin(), [&norm](auto &v){ return norm * (double)v; });

    if (dev_measured) cudaFree(dev_measured);
    if (dev_fitted) cudaFree(dev_fitted);
    if (dev_solution) cudaFree(dev_solution);
    if (dev_gradpen) cudaFree(dev_gradpen);
    if (dev_obs_fit) cudaFree(dev_obs_fit);

    return (iter < max_iterations) ? SUCCESS : MAX_ITERATIONS_EXCEEDED;
}
