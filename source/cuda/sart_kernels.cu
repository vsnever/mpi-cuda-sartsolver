
#include <iostream>
#include <cuda_runtime.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef EPSILON_LOG_CUDA
#define EPSILON_LOG_CUDA 0.0000001f
#endif


__global__ void InitialGuessKernel(float* const __restrict__ solution, const float* const  __restrict__ rtm,
                                   const float* const  __restrict__ measured, const float* const  __restrict__  ray_density,
                                   const float ray_dens_thres, const size_t npixel, const size_t nvoxel){

    const size_t jvox = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ipix_offset = blockIdx.y * BLOCK_SIZE;

    __shared__ float measured_cache[BLOCK_SIZE];

    const size_t cache_size = min(npixel - ipix_offset, (size_t)BLOCK_SIZE);
    if (threadIdx.x < cache_size) {
        const float meas = measured[ipix_offset + threadIdx.x];
        measured_cache[threadIdx.x] = (meas > 0) ? meas : 0;  // exclude negative values (saturated pixels)
    }

    __syncthreads();

    const float ray_dens = (jvox < nvoxel) ? ray_density[jvox] : 0;
    if (ray_dens > ray_dens_thres) {
        float res = 0;
        if (cache_size % 4 == 0) {
            for (size_t ipix_cache = 0; ipix_cache < cache_size; ipix_cache += 4) {
                const size_t ipix = ipix_offset + ipix_cache;
                res += rtm[ ipix      * nvoxel + jvox] * measured_cache[ipix_cache    ] + \
                       rtm[(ipix + 1) * nvoxel + jvox] * measured_cache[ipix_cache + 1] + \
                       rtm[(ipix + 2) * nvoxel + jvox] * measured_cache[ipix_cache + 2] + \
                       rtm[(ipix + 3) * nvoxel + jvox] * measured_cache[ipix_cache + 3];
            }
        }
        else {
            for (size_t ipix_cache = 0; ipix_cache < cache_size; ++ipix_cache) {
                res += rtm[(ipix_offset + ipix_cache) * nvoxel + jvox] * measured_cache[ipix_cache];
            }
        }
        res /= ray_dens;
        
        atomicAdd(solution + jvox, res);
    }
}


__global__ void PropagateKernel(float* const __restrict__ diff, const float* const __restrict__ rtm,
                                const float* const __restrict__ measured, const float* const __restrict__ fitted,
                                const float* const __restrict__ ray_density, const float* const __restrict__ ray_length,
                                const float* const __restrict__ grad_penalty, float relaxation, float ray_dens_thres, float ray_length_thres,
                                size_t npixel, size_t nvoxel){

    const size_t jvox = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ipix_offset = blockIdx.y * blockDim.x;

    __shared__ float fit_diff_cache[BLOCK_SIZE];
    __shared__ float inv_length_cache[BLOCK_SIZE];

    const size_t cache_size = min(npixel - ipix_offset, (size_t)BLOCK_SIZE);
    if (threadIdx.x < cache_size) {
        const size_t ipix = ipix_offset + threadIdx.x;
        const float meas = measured[ipix];
        fit_diff_cache[threadIdx.x] = (meas >= 0) ? meas - fitted[ipix] : 0;  // exclude negative values (saturated pixels)
        const float length = ray_length[ipix];
        inv_length_cache[threadIdx.x] = (length > ray_length_thres) ? 1.f / length : 0;
    }

    __syncthreads();

    const float ray_dens = (jvox < nvoxel) ? ray_density[jvox] : 0;
    float res = 0;
    if (ray_dens > ray_dens_thres) {
        if (cache_size % 4 == 0) {
            for (unsigned int ipix_cache = 0; ipix_cache < cache_size; ipix_cache += 4) {
                const unsigned int ipix = ipix_offset + ipix_cache;
                res += rtm[ ipix      * nvoxel + jvox] * inv_length_cache[ipix_cache    ] * fit_diff_cache[ipix_cache    ] + \
                       rtm[(ipix + 1) * nvoxel + jvox] * inv_length_cache[ipix_cache + 1] * fit_diff_cache[ipix_cache + 1] + \
                       rtm[(ipix + 2) * nvoxel + jvox] * inv_length_cache[ipix_cache + 2] * fit_diff_cache[ipix_cache + 2] + \
                       rtm[(ipix + 3) * nvoxel + jvox] * inv_length_cache[ipix_cache + 3] * fit_diff_cache[ipix_cache + 3];
            }
        }
        else {
            for (unsigned int ipix_cache = 0; ipix_cache < cache_size; ++ipix_cache) {
                res += rtm[(ipix_offset + ipix_cache) * nvoxel + jvox] * inv_length_cache[ipix_cache] * fit_diff_cache[ipix_cache];
            }
        }
        res *= relaxation / ray_dens;
    }

    if (jvox < nvoxel) {
        if (!blockIdx.y) atomicAdd(diff + jvox, res - grad_penalty[jvox]);
        else atomicAdd(diff + jvox, res);
    }
}


__global__ void LogPropagateKernel(float* const __restrict__ obs_fit, const float* const __restrict__ rtm,
                                   const float* const __restrict__ measured, const float* const __restrict__ fitted,
                                   const float* const __restrict__ ray_density, const float* const __restrict__ ray_length,
                                   float ray_dens_thres, float ray_length_thres, size_t npixel, size_t nvoxel){

    const size_t jvox = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ipix_offset = blockIdx.y * blockDim.x;

    __shared__ float measured_cache[BLOCK_SIZE];
    __shared__ float fitted_cache[BLOCK_SIZE];
    __shared__ float inv_length_cache[BLOCK_SIZE];

    const size_t cache_size = min(npixel - ipix_offset, (size_t)BLOCK_SIZE);
    if (threadIdx.x < cache_size) {
        const size_t ipix = ipix_offset + threadIdx.x;
        const float meas = measured[ipix];
        if (meas >= 0) {
            measured_cache[threadIdx.x] = meas;
            fitted_cache[threadIdx.x] = fitted[ipix];
        }
        else {  // exclude negative values (saturated pixels)
            measured_cache[threadIdx.x] = 0;
            fitted_cache[threadIdx.x] = 0;
        }
        const float length = ray_length[ipix];
        inv_length_cache[threadIdx.x] = (length > ray_length_thres) ? 1.f / length : 0;
    }

    __syncthreads();

    const float ray_dens = (jvox < nvoxel) ? ray_density[jvox] : 0;
    float obs = 0;
    float fit = 0;
    if (ray_dens > ray_dens_thres) {
        if (cache_size % 4 == 0) {
            float4 prop_ray_length;
            for (unsigned int ipix_cache = 0; ipix_cache < cache_size; ipix_cache += 4) {
                const unsigned int ipix = ipix_offset + ipix_cache;
                prop_ray_length.x = rtm[ ipix      * nvoxel + jvox] * inv_length_cache[ipix_cache    ];
                prop_ray_length.y = rtm[(ipix + 1) * nvoxel + jvox] * inv_length_cache[ipix_cache + 1];
                prop_ray_length.z = rtm[(ipix + 2) * nvoxel + jvox] * inv_length_cache[ipix_cache + 2];
                prop_ray_length.w = rtm[(ipix + 3) * nvoxel + jvox] * inv_length_cache[ipix_cache + 3];
                obs += prop_ray_length.x * measured_cache[ipix_cache    ] + \
                       prop_ray_length.y * measured_cache[ipix_cache + 1] + \
                       prop_ray_length.z * measured_cache[ipix_cache + 2] + \
                       prop_ray_length.w * measured_cache[ipix_cache + 3];
                fit += prop_ray_length.x * fitted_cache[ipix_cache    ] + \
                       prop_ray_length.y * fitted_cache[ipix_cache + 1] + \
                       prop_ray_length.z * fitted_cache[ipix_cache + 2] + \
                       prop_ray_length.w * fitted_cache[ipix_cache + 3];
            }
        }
        else {
            float prop_ray_length;
            for (unsigned int ipix_cache = 0; ipix_cache < cache_size; ++ipix_cache) {
                prop_ray_length = rtm[(ipix_offset + ipix_cache) * nvoxel + jvox] * inv_length_cache[ipix_cache];
                obs += prop_ray_length * measured_cache[ipix_cache];
                fit += prop_ray_length * fitted_cache[ipix_cache];
            }
        }
        atomicAdd(obs_fit + jvox, obs);
        atomicAdd(obs_fit + nvoxel + jvox, fit);
    }
}


__global__ void GradPenaltyKernel(float* const __restrict__ grad_penalty,  const float* const __restrict__ solution,
                                  const size_t* const __restrict__ laplace_idx, const float* const __restrict__ laplace_val,
                                  float beta_laplace, size_t laplacian_size, size_t nvoxel) {

    size_t i_offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = i_offset; i < laplacian_size; i += blockDim.x * gridDim.x) {
        const size_t index = laplace_idx[i];
        atomicAdd(grad_penalty + index / nvoxel, beta_laplace * laplace_val[i] * solution[index % nvoxel]);
    }
}


__global__ void LogGradPenaltyKernel(float* const __restrict__ grad_penalty,  const float* const __restrict__ solution,
                                     const size_t* const __restrict__ laplace_idx, const float* const __restrict__ laplace_val,
                                     float beta_laplace, size_t laplacian_size, size_t nvoxel) {

    size_t i_offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = i_offset; i < laplacian_size; i += blockDim.x * gridDim.x) {
        const size_t index = laplace_idx[i];
        atomicAdd(grad_penalty + index / nvoxel, beta_laplace * laplace_val[i] * __logf(solution[index % nvoxel]));
    }
}


__global__ void UpdateSolutionKernel(float* const __restrict__ solution, float* const __restrict__ diff, size_t nvoxel){
    size_t jvox = blockIdx.x * blockDim.x + threadIdx.x;
    if (jvox < nvoxel) {
        float sol = solution[jvox] + diff[jvox];
        solution[jvox] = (sol > 0) ? sol : 0;
        diff[jvox] = 0;
    }
}


__global__ void UpdateLogSolutionKernel(float* const __restrict__ solution, float* const __restrict__ obs_fit,
                                        const float* const __restrict__ grad_penalty, float relaxation, size_t nvoxel){
    size_t jvox = blockIdx.x * blockDim.x + threadIdx.x;
    if (jvox < nvoxel) {
        const float ratio = __powf((obs_fit[jvox] + EPSILON_LOG_CUDA) / (obs_fit[nvoxel + jvox] + EPSILON_LOG_CUDA), relaxation);
        solution[jvox] *= ratio * __expf(-grad_penalty[jvox]);
        obs_fit[jvox] = 0;
        obs_fit[nvoxel + jvox] = 0;
    }
}


extern "C" void CallInitialGuessKernel(float* const solution, const float* const rtm, const float* const measured, 
                                       const float* const ray_density, const float ray_dens_thres, const size_t npixel, const size_t nvoxel) {

    dim3 dim_block(BLOCK_SIZE, 1);
    dim3 dim_grid(nvoxel / BLOCK_SIZE + (bool)(nvoxel % BLOCK_SIZE), npixel / BLOCK_SIZE + (bool)(npixel % BLOCK_SIZE));
    InitialGuessKernel<<<dim_grid, dim_block>>>(solution, rtm, measured, ray_density, ray_dens_thres, npixel, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "InitialGuessKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallGradPenaltyKernel(float* const grad_penalty, const float* const solution, const size_t* const laplace_idx,
                                      const float* const laplace_val, float beta_laplace, size_t laplacian_size, size_t nvoxel) {

    size_t dim_grid = laplacian_size / BLOCK_SIZE + (bool)(laplacian_size % BLOCK_SIZE);
    GradPenaltyKernel<<<dim_grid, BLOCK_SIZE>>>(grad_penalty, solution, laplace_idx, laplace_val,
                                                beta_laplace, laplacian_size, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "GradPenaltyKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallLogGradPenaltyKernel(float* const grad_penalty, const float* const solution, const size_t* const laplace_idx,
                                         const float* const laplace_val, float beta_laplace, size_t laplacian_size, size_t nvoxel) {

    size_t dim_grid = laplacian_size / BLOCK_SIZE + (bool)(laplacian_size % BLOCK_SIZE);
    LogGradPenaltyKernel<<<dim_grid, BLOCK_SIZE>>>(grad_penalty, solution, laplace_idx, laplace_val,
                                                   beta_laplace, laplacian_size, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "LogGradPenaltyKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallPropagateKernel(float* const diff, const float* const rtm, const float* const measured, const float* const fitted,
                                    const float* const ray_density, const float* const ray_length, const float* const grad_penalty,
                                    float relaxation, float ray_dens_thres, float ray_length_thres, size_t npixel, size_t nvoxel) {

    dim3 dim_block(BLOCK_SIZE, 1);
    dim3 dim_grid(nvoxel / BLOCK_SIZE + (bool)(nvoxel % BLOCK_SIZE), npixel / BLOCK_SIZE + (bool)(npixel % BLOCK_SIZE));
    PropagateKernel<<<dim_grid, dim_block>>>(diff, rtm, measured, fitted, ray_density, ray_length, grad_penalty,
                                             relaxation, ray_dens_thres, ray_length_thres, npixel, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "PropagateKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallLogPropagateKernel(float* const ofs_fit, const float* const rtm, const float* const measured, const float* const fitted,
                                       const float* const ray_density, const float* const ray_length,
                                       float ray_dens_thres, float ray_length_thres, size_t npixel, size_t nvoxel) {

    dim3 dim_block(BLOCK_SIZE, 1);
    dim3 dim_grid(nvoxel / BLOCK_SIZE + (bool)(nvoxel % BLOCK_SIZE), npixel / BLOCK_SIZE + (bool)(npixel % BLOCK_SIZE));
    LogPropagateKernel<<<dim_grid, dim_block>>>(ofs_fit, rtm, measured, fitted, ray_density, ray_length,
                                                ray_dens_thres, ray_length_thres, npixel, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "LogPropagateKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallUpdateSolutionKernel(float* const solution, float* const diff, size_t nvoxel) {

    size_t dim_grid = nvoxel / BLOCK_SIZE + (bool)(nvoxel % BLOCK_SIZE);
    UpdateSolutionKernel<<<dim_grid, BLOCK_SIZE>>>(solution, diff, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "UpdateSolutionKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

extern "C" void CallUpdateLogSolutionKernel(float* const solution, float* const ofs_fit, const float* const grad_penalty,
                                            float relaxation, size_t nvoxel) {

    size_t dim_grid = nvoxel / BLOCK_SIZE + (bool)(nvoxel % BLOCK_SIZE);
    UpdateLogSolutionKernel<<<dim_grid, BLOCK_SIZE>>>(solution, ofs_fit, grad_penalty, relaxation, nvoxel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "UpdateLogSolutionKernel<<<,>>>(...) failure:\n";
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}
