
#include <algorithm>

#include "mpi.h"

#include "arguments.hpp"
#include "raytransfer.hpp"
#include "laplacian.hpp"
#include "image.hpp"
#include "sartsolver.hpp"
#include "sartsolver_cuda.hpp"
#include "hdf5files.hpp"


int main(int argc, char *argv[]){

    auto program = parse_arguments(argc, argv);

    const auto time_intervals = parse_time_intervals(program.get<std::string>("--time_range"));

    std::vector<std::string> matrix_files;
    std::vector<std::string> image_files;

    categorize_input_files(program.get<std::vector<std::string>>("input_files"), matrix_files, image_files);

    const std::string rtm_name(program.get<std::string>("--raytransfer_name"));

    check_group_attribute_consistency<double>(matrix_files, "rtm/" + rtm_name, {"wavelength"},
                                              H5::PredType::NATIVE_DOUBLE);
    check_group_attribute_consistency<size_t>(matrix_files, "rtm/voxel_map", {"nx", "ny", "nz"},
                                              H5::PredType::NATIVE_HSIZE);

    const auto sorted_matrix_files = sort_rtm_files(matrix_files);

    check_rtm_frame_consistency(sorted_matrix_files);
    check_rtm_voxel_consistency(sorted_matrix_files);

    check_group_attribute_consistency<double>(image_files, "image", {"wavelength"}, H5::PredType::NATIVE_DOUBLE);

    auto sorted_image_files = sort_image_files(image_files);

    std::vector<std::string> camera_names;
    for (auto const& cam_name : sorted_image_files) camera_names.push_back(cam_name.first);


    check_rtm_image_consistency(sorted_matrix_files, sorted_image_files, rtm_name, program.get<double>("--wavelength_threshold"));

    size_t npixel, nvoxel;
    std::tie(npixel, nvoxel) = get_total_rtm_size(sorted_matrix_files);

    auto rtm_frame_masks = read_rtm_frame_masks(sorted_matrix_files);

    // parallel code
    int numproc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    const auto offset_pixel = (size_t)rank * (npixel / (size_t)numproc) + std::min((size_t)rank, (npixel % (size_t)numproc));
    const auto npixel_local = ((size_t)rank < npixel % (size_t)numproc) ? npixel / (size_t)numproc + 1 : npixel / (size_t)numproc;

    CompositeImage composite_image(sorted_image_files, rtm_frame_masks, time_intervals, npixel_local, offset_pixel);
    composite_image.set_max_cache_size((size_t)program.get<int>("--max_cached_frames"));

    RayTransferMatrix raytransfer(npixel_local, nvoxel, offset_pixel);
    raytransfer.read_hdf5(sorted_matrix_files, rtm_name);

    const bool use_logsolver = program.get<bool>("--logarithmic");
    const bool use_cpusolver = program.get<bool>("--use_cpu");

    const auto laplacian_file = program.get<std::string>("--laplacian_file");
    LaplacianMatrix laplacian((laplacian_file.empty() || (rank && !use_logsolver)) ? 0 : nvoxel);
    if (laplacian.nvoxel() > 0) laplacian.read_hdf5(laplacian_file); // only root process reads Laplacian matrix if solver is linear

    BaseSARTSolverMPI *solver;
    if (use_cpusolver) {
        if (use_logsolver) {
            solver = new LogSARTSolverMPI(raytransfer, laplacian);
        }
        else {
            solver = new SARTSolverMPI(raytransfer, laplacian);
        }
    }
    else {
        if (use_logsolver) {
            solver = new LogSARTSolverMPICuda(raytransfer, laplacian);
        }
        else {
            solver = new SARTSolverMPICuda(raytransfer, laplacian);
        }
    }
    solver->set_ray_density_threshold(program.get<double>("--ray_density_threshold"));
    solver->set_ray_length_threshold(program.get<double>("--ray_length_threshold"));
    solver->set_convolution_tolerance(program.get<double>("--conv_tolerance"));
    solver->set_beta_laplace(program.get<double>("--beta_laplace"));
    solver->set_relaxation(program.get<double>("--relaxation"));
    solver->set_max_iterations(program.get<int>("--max_iterations"));

    std::vector<std::vector<double>> solutions;
    std::vector<double> solution;
    std::vector<double> frame;

    while(composite_image.next_frame(frame)) {
        solver->solve(solution, frame);
        if (rank == 0) solutions.push_back(std::vector<double>(solution));
        if (program.get<bool>("--no_guess")) solution.clear();
    }

    delete solver;

    if (rank == 0) write_solutions(program.get<std::string>("--output_file"), solutions);

    MPI_Finalize();

	return 0;
}
