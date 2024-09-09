// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <algorithm>
#include <chrono>

#include "mpi.h"

#include "arguments.hpp"
#include "raytransfer.hpp"
#include "laplacian.hpp"
#include "image.hpp"
#include "solution.hpp"
#include "voxelgrid.hpp"
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

    const bool use_logsolver = program.get<bool>("--logarithmic");

    const auto laplacian_file = program.get<std::string>("--laplacian_file");
    LaplacianMatrix laplacian((laplacian_file.empty() || (rank && !use_logsolver)) ? 0 : nvoxel);
    if (laplacian.nvoxel() > 0) laplacian.read_hdf5(laplacian_file); // only root process reads Laplacian matrix if solver is linear

    RayTransferMatrix raytransfer(npixel_local, nvoxel, offset_pixel);

    if (program.get<bool>("--parallel_read")) { // works faster on high-IOPS storage
        raytransfer.read_hdf5(sorted_matrix_files, rtm_name);
    }
    else { // works faster on HDDs
        for (int id = 0; id < numproc; ++id) {
            if (rank == id) raytransfer.read_hdf5(sorted_matrix_files, rtm_name);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    BaseSARTSolverMPI *solver;
    if (program.get<bool>("--use_cpu")) {
        if (use_logsolver) {
            solver = new LogSARTSolverMPI(raytransfer, laplacian);
        }
        else {
            solver = new SARTSolverMPI(raytransfer, laplacian);
        }
    }
    else {
        if (use_logsolver) {
            solver = new LogSARTSolverMPICuda(raytransfer, laplacian, rank);
        }
        else {
            solver = new SARTSolverMPICuda(raytransfer, laplacian, rank);
        }
    }
    solver->set_ray_density_threshold(program.get<double>("--ray_density_threshold"));
    solver->set_ray_length_threshold(program.get<double>("--ray_length_threshold"));
    solver->set_convolution_tolerance(program.get<double>("--conv_tolerance"));
    solver->set_beta_laplace(program.get<double>("--beta_laplace"));
    solver->set_relaxation(program.get<double>("--relaxation"));
    solver->set_max_iterations(program.get<int>("--max_iterations"));

    Solution solution(program.get<std::string>("--output_file"), camera_names, nvoxel);
    solution.set_max_cache_size((size_t)program.get<int>("--max_cached_solutions"));

    const auto coordsys = BaseVoxelGrid::get_coordinate_system_hdf5(sorted_matrix_files.begin()->second[0], "rtm/voxel_map");
    BaseVoxelGrid *voxelgrid;

    if (coordsys == BaseVoxelGrid::CYLINDRICAL) {
        voxelgrid = new CylindricalVoxelGrid();
    }
    else {
        voxelgrid = new CartesianVoxelGrid();
    }

    if (rank == 0) voxelgrid->read_hdf5(sorted_matrix_files.begin()->second, "rtm/voxel_map");

    std::vector<double> solution_vec, frame;
    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> last;

    while(composite_image.next_frame(frame)) {
        last = clock.now();
        const auto status = solver->solve(solution_vec, frame);
        if (rank == 0) {
            solution.add(solution_vec, status, composite_image.frame_time(), composite_image.camera_frame_time());
            std::chrono::duration<double,std::milli> duration(clock.now() - last);
            std::cout << "Processed in: " << duration.count() << " ms" << std::endl;
        }
        if (program.get<bool>("--no_guess")) solution_vec.clear();
    }

    solution.flush_hdf5();
    if (rank == 0) voxelgrid->write_hdf5(program.get<std::string>("--output_file"), "voxel_map");

    delete solver;
    delete voxelgrid;

    MPI_Finalize();

	return 0;
}
