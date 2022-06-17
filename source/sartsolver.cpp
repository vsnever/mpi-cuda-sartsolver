
#include <algorithm>
#include <iostream>
#include <cmath>
#include "sartsolver.hpp"

#define EPSILON_LOG 1.e-100

const int BaseSARTSolverMPI::SUCCESS = 0;
const int BaseSARTSolverMPI::MAX_ITERATIONS_EXCEEDED = -1;


BaseSARTSolverMPI::BaseSARTSolverMPI(RayTransferMatrix& rtm, LaplacianMatrix &lm,
                                     double ray_density_threshold, double ray_length_threshold,
                                     double convolution_tolerance, double beta_laplace,
                                     double relaxation, int max_iterations,
                                     MPI_Comm mpi_communicator):
    raytransfer(rtm),
    laplacian(lm) {
        set_ray_density_threshold(ray_density_threshold);
        set_ray_length_threshold(ray_length_threshold);
        set_convolution_tolerance(convolution_tolerance);
        set_beta_laplace(beta_laplace);
        set_relaxation(relaxation);
        set_max_iterations(max_iterations);
        mpi_comm = mpi_communicator;

        const size_t nvoxel = raytransfer.nvoxel();
        const size_t npixel = raytransfer.npixel();

        std::vector<double> ray_density_loc(nvoxel, 0);
        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            double res = 0;
            for (size_t ipix=0; ipix<npixel; ++ipix) {
                res += (double)raytransfer.matrix(ipix, jvox);
            }
            ray_density_loc[jvox] = res;
        }
        ray_density.resize(nvoxel);
        MPI_Allreduce(ray_density_loc.data(), ray_density.data(), nvoxel, MPI_DOUBLE, MPI_SUM, mpi_comm);

        ray_length.resize(npixel, 0);
        for (size_t ipix=0; ipix<npixel; ++ipix) {
            double res = 0;
            for (size_t jvox=0; jvox<nvoxel; ++jvox) {
                res += (double)raytransfer.matrix(ipix, jvox);
            }
            ray_length[ipix] = res;
        }

    }


double BaseSARTSolverMPI::get_ray_density_threshold() const{return ray_dens_thres;}

void BaseSARTSolverMPI::set_ray_density_threshold(double value) {
    if (value < 0) {
        std::cerr << "Ray density threshold must be non-negative." << std::endl;
        std::exit(1);
    }
    ray_dens_thres = value;
}

double BaseSARTSolverMPI::get_ray_length_threshold() const{return ray_length_thres;}

void BaseSARTSolverMPI::set_ray_length_threshold(double value) {
    if (value < 0) {
        std::cerr << "Ray length threshold must be non-negative." << std::endl;
        std::exit(1);
    }
    ray_length_thres = value;
}


double BaseSARTSolverMPI::get_convolution_tolerance() const{return conv_tol;}

void BaseSARTSolverMPI::set_convolution_tolerance(double value) {
    if (value <= 0) {
        std::cerr << "Convolution tolerance must be positive." << std::endl;
        std::exit(1);
    }
    conv_tol = value;
}


double BaseSARTSolverMPI::get_beta_laplace() const{return beta_laplace;}

void BaseSARTSolverMPI::set_beta_laplace(double value) {
    if (value < 0) {
        std::cerr << "Attribute beta_laplace must be non-negative." << std::endl;
        std::exit(1);
    }
    beta_laplace = value;
}


double BaseSARTSolverMPI::get_relaxation() const{return relaxation;}

void BaseSARTSolverMPI::set_relaxation(double value) {
    if ((value <= 0) || (value > 1.)) {
        std::cerr << "Attribute relaxation must be within (0, 1] interval." << std::endl;
        std::exit(1);
    }
    relaxation = value;
}


int BaseSARTSolverMPI::get_max_iterations() const{return max_iterations;}

void BaseSARTSolverMPI::set_max_iterations(int value) {
    if (value <= 0) {
        std::cerr << "Attribute max_iterations must be positive." << std::endl;
        std::exit(1);
    }
    max_iterations = value;
}


int BaseSARTSolverMPI::solve(std::vector<double>& solution,
                             const std::vector<double>& measurement) const{
    std::cerr << "Method not implemented." << std::endl;
    std::exit(1);
}


int SARTSolverMPI::solve(std::vector<double>& solution,
                         const std::vector<double>& measurement) const{

    const size_t nvoxel = raytransfer.nvoxel();
    const size_t npixel = raytransfer.npixel();

    if (!solution.empty() && (solution.size() != nvoxel)) {
        std::cerr << "Solution vector must be empty or contain nvoxel elements." << std::endl;
        std::exit(1);
    }

    if (solution.empty()) {
        // default initial guess
        solution.resize(nvoxel, 0);
        std::vector<double> solution_loc(nvoxel, 0);

        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            if (ray_density[jvox] > ray_dens_thres) {
                double res = 0;
                for (size_t ipix=0; ipix<npixel; ++ipix) {
                    res += (double)raytransfer.matrix(ipix, jvox) * measurement[ipix];
                }
                solution_loc[jvox] = res / ray_density[jvox];
            }
        }
        MPI_Allreduce(solution_loc.data(), solution.data(), nvoxel, MPI_DOUBLE, MPI_SUM, mpi_comm);
    }

    double measurement_squared = 0;
    double measurement_squared_loc = 0;
    for (auto m : measurement) {measurement_squared_loc += (m > 0) ? m * m : 0;}  // exclude negative values (saturated pixels)
    MPI_Allreduce(&measurement_squared_loc, &measurement_squared, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    
    std::vector<double> fitted(npixel);
    for (size_t ipix=0; ipix<npixel; ++ipix) {
        double res = 0;
        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            res += (double)raytransfer.matrix(ipix, jvox) * solution[jvox];
        }
        fitted[ipix] = res;
    }

    std::vector<double> grad_penalty(nvoxel);
    std::vector<double> diff_loc(nvoxel);
    std::vector<double> diff(nvoxel);

    double convergence_prev = 0;
    for (int iter=0; iter<max_iterations; ++iter) {

        std::cout << iter << std::endl;

        std::fill(grad_penalty.begin(), grad_penalty.end(), 0);
        for (size_t i=0; i<laplacian.size(); ++i) {
            const size_t index = laplacian.index(i);
            const size_t ivox = index / nvoxel;
            const size_t jvox = index % nvoxel;
            grad_penalty[ivox] += beta_laplace * (double)laplacian.value(i) * solution[jvox];
        }

        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            double diff_j = 0;
            if (ray_density[jvox] > ray_dens_thres) {
                for (size_t ipix=0; ipix<npixel; ++ipix) {
                    const double meas = measurement[ipix];
                    if ((ray_length[ipix] > ray_length_thres) && (meas >= 0)) {  // exlude negative values (saturated pixels)
                        const double prop_ray_length = (double)raytransfer.matrix(ipix, jvox) / ray_length[ipix];
                        diff_j += prop_ray_length * (meas - fitted[ipix]);
                    }
                }
                diff_j *= relaxation / ray_density[jvox];
            }            

            diff_loc[jvox] = diff_j - grad_penalty[jvox];
        }
        MPI_Allreduce(diff_loc.data(), diff.data(), nvoxel, MPI_DOUBLE, MPI_SUM, mpi_comm);

        std::transform(solution.begin(), solution.end(), diff.begin(), solution.begin(), std::plus<double>());
        std::replace_if(solution.begin(), solution.end(), [](double v){return std::signbit(v);}, 0); 

        double fitted_squared = 0;
        double fitted_squared_loc = 0;
        for (size_t ipix=0; ipix<npixel; ++ipix){
            double res = 0;
            for (size_t jvox=0; jvox<nvoxel; ++jvox) {
                res += (double)raytransfer.matrix(ipix, jvox) * solution[jvox];
            }
            fitted[ipix] = res;
            fitted_squared_loc += res * res;
        }

        MPI_Allreduce(&fitted_squared_loc, &fitted_squared, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

        const double convergence = (measurement_squared - fitted_squared) / measurement_squared;

        if ((iter) && (std::abs(convergence - convergence_prev) < conv_tol)) return SUCCESS;

        convergence_prev = convergence;
    }

    return MAX_ITERATIONS_EXCEEDED;
}


int LogSARTSolverMPI::solve(std::vector<double>& solution,
                            const std::vector<double>& measurement) const{

    const size_t nvoxel = raytransfer.nvoxel();
    const size_t npixel = raytransfer.npixel();

    if (!solution.empty() && (solution.size() != nvoxel)) {
        std::cerr << "Solution vector must be empty or contain nvoxel elements." << std::endl;
        std::exit(1);
    }

    if (solution.empty()) {
        // default initial guess
        solution.resize(nvoxel, 0);
        std::vector<double> solution_loc(nvoxel, 0);

        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            if (ray_density[jvox] > ray_dens_thres) {
                double res = 0;
                for (size_t ipix=0; ipix<npixel; ++ipix) {
                    res += (double)raytransfer.matrix(ipix, jvox) * measurement[ipix];
                }
                solution_loc[jvox] = res / ray_density[jvox];
            }
        }
        MPI_Allreduce(solution_loc.data(), solution.data(), nvoxel, MPI_DOUBLE, MPI_SUM, mpi_comm);
    }
    
    for (auto &s : solution) {if (s < EPSILON_LOG) s = EPSILON_LOG;}

    double measurement_squared = 0;
    double measurement_squared_loc = 0;
    for (auto m : measurement) {measurement_squared_loc += (m > 0) ? m * m : 0;}  // exclude negative values (saturated pixels)
    MPI_Allreduce(&measurement_squared_loc, &measurement_squared, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    
    std::vector<double> fitted(npixel);
    for (size_t ipix=0; ipix<npixel; ++ipix) {
        double res = 0;
        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            res += (double)raytransfer.matrix(ipix, jvox) * solution[jvox];
        }
        fitted[ipix] = res;
    }

    std::vector<double> grad_penalty(nvoxel);
    std::vector<double> obs_fit_loc(2 * nvoxel);
    std::vector<double> obs_fit(2 * nvoxel);

    double convergence_prev = 0;
    for (int iter=0; iter<max_iterations; ++iter) {
        std::cout << iter << std::endl;

        std::fill(grad_penalty.begin(), grad_penalty.end(), 0);
        for (size_t i=0; i<laplacian.size(); ++i) {
            const size_t index = laplacian.index(i);
            const size_t ivox = index / nvoxel;
            const size_t jvox = index % nvoxel;
            grad_penalty[ivox] += beta_laplace * (double)laplacian.value(i) * log(solution[jvox]);
        }

        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            double obs_j = 0;
            double fit_j = 0;
            if (ray_density[jvox] > ray_dens_thres) {
                for (size_t ipix=0; ipix<npixel; ++ipix) {
                    const double meas = measurement[ipix];
                    if ((ray_length[ipix] > ray_length_thres) && (meas >= 0)) {  // exlude negative values (saturated pixels)
                        const double prop_ray_length = (double)raytransfer.matrix(ipix, jvox) / ray_length[ipix];
                        obs_j += prop_ray_length * measurement[ipix];
                        fit_j += prop_ray_length * fitted[ipix];
                    }
                }
            }
            obs_fit_loc[jvox] = obs_j;
            obs_fit_loc[nvoxel + jvox] = fit_j;
        }

        MPI_Allreduce(obs_fit_loc.data(), obs_fit.data(), 2 * nvoxel, MPI_DOUBLE, MPI_SUM, mpi_comm);

        for (size_t jvox=0; jvox<nvoxel; ++jvox) {
            solution[jvox] *= pow((obs_fit[jvox] + EPSILON_LOG) / (obs_fit[nvoxel + jvox] + EPSILON_LOG), relaxation) * exp(-grad_penalty[jvox]);
        }

        double fitted_squared = 0;
        double fitted_squared_loc = 0;
        for (size_t ipix=0; ipix<npixel; ++ipix){
            double res = 0;
            for (size_t jvox=0; jvox<nvoxel; ++jvox) {
                res += (double)raytransfer.matrix(ipix, jvox) * solution[jvox];
            }
            fitted[ipix] = res;
            fitted_squared_loc += res * res;
        }

        MPI_Allreduce(&fitted_squared_loc, &fitted_squared, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

        const double convergence = (measurement_squared - fitted_squared) / measurement_squared;

        if ((iter) && (std::abs(convergence - convergence_prev) < conv_tol)) return SUCCESS;

        convergence_prev = convergence;
    }

    return MAX_ITERATIONS_EXCEEDED;
}
