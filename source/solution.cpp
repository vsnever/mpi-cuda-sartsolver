// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <iostream>
#include <H5Cpp.h>
#include "solution.hpp"


Solution::Solution(const std::string& filename,
                   const std::vector<std::string>& camera_names,
                   size_t nvoxel, size_t cache_size):
          filename(filename) {

    if (nvoxel == 0) {
        std::cerr << "Argument nvoxel must be positive." << std::endl;
        std::exit(1);
    }
    nvox = nvoxel;
    first_flush = true;
    set_max_cache_size(cache_size);
    for (const auto &name : camera_names) { cached_camera_time[name] = std::vector<double>(); }
}


Solution::~Solution() {
    flush_hdf5();
}

size_t Solution::get_max_cache_size() const{return max_cache_size;}

void Solution::set_max_cache_size(size_t value) {
    if (value == 0) {
        std::cerr << "Attribute max_cache_size must be positive." << std::endl;
        std::exit(1);
    }
    max_cache_size = value;
}

int Solution::add(const std::vector<double>& solution, int status, double time, const std::vector<double>& camera_time) {

    cached_status.push_back(status);
    cached_solutions.push_back(solution);
    cached_time.push_back(time);
    auto it = camera_time.cbegin();
    for (auto it_pair=cached_camera_time.begin(); it_pair!=cached_camera_time.end(); ++it_pair) {
        it_pair->second.push_back(*it);
        ++it;
    }

    if (cached_solutions.size() >= max_cache_size) return flush_hdf5();

    return 0;
}

int Solution::create_hdf5() {
    try {
        // create new HDF5 file with extendible datasets
        H5::H5File file(filename, H5F_ACC_TRUNC);
        auto group = file.createGroup("solution");

        const hsize_t maxdims[] = {H5S_UNLIMITED, nvox};
        const hsize_t dims[] = {cached_solutions.size(), nvox};
        H5::DataSpace dataspace(2, dims, maxdims);
        const hsize_t count[] = {1, nvox};
        const H5::DataSpace memspace(1, count + 1);
        hsize_t dset_offset[] = {0, 0};

        H5::DSetCreatPropList cparams;
        double fill_value = 0;
        cparams.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);
        cparams.setChunk(2, count);

        auto dataset = group.createDataSet("value", H5::PredType::NATIVE_DOUBLE, dataspace, cparams);
        for (size_t it=0; it<cached_solutions.size(); ++it) {
            dset_offset[0] = it;
            dataspace.selectHyperslab(H5S_SELECT_SET, count, dset_offset);
            dataset.write(cached_solutions[it].data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);
        }

        hsize_t maxdims_1d = H5S_UNLIMITED;
        hsize_t dims_1d = cached_time.size();
        H5::DataSpace dspace_1d(1, &dims_1d, &maxdims_1d);

        H5::DSetCreatPropList cparams_1d;
        cparams_1d.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value);
        cparams_1d.setChunk(1, &dims_1d);

        group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, dspace_1d, cparams_1d)
             .write(cached_time.data(), H5::PredType::NATIVE_DOUBLE);

        for (const auto &pair : cached_camera_time) {
            group.createDataSet("time_" + pair.first, H5::PredType::NATIVE_DOUBLE, dspace_1d, cparams_1d)
                 .write(pair.second.data(), H5::PredType::NATIVE_DOUBLE);
        }

        int fill_value_int = 0;
        cparams_1d.setFillValue(H5::PredType::NATIVE_DOUBLE, &fill_value_int);
        group.createDataSet("status", H5::PredType::NATIVE_INT, dspace_1d, cparams_1d)
             .write(cached_status.data(), H5::PredType::NATIVE_INT);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}

int Solution::update_hdf5() {
    try {
        // update existing HDF5 file extending the datasets
        H5::H5File file(filename, H5F_ACC_RDWR);

        hsize_t time_offset = 0;
        auto time_dset = file.openDataSet("solution/time");
        time_dset.getSpace().getSimpleExtentDims(&time_offset);

        const hsize_t count_1d = cached_time.size();
        const H5::DataSpace mspace_1d(1, &count_1d);
        hsize_t new_size_1d = time_offset + count_1d; 

        time_dset.extend(&new_size_1d);
        auto time_dspace = time_dset.getSpace();
        time_dspace.selectHyperslab(H5S_SELECT_SET, &count_1d, &time_offset);
        time_dset.write(cached_time.data(), H5::PredType::NATIVE_DOUBLE, mspace_1d, time_dspace);

        auto status_dset = file.openDataSet("solution/status");
        status_dset.extend(&new_size_1d);
        auto status_dspace = status_dset.getSpace();
        status_dspace.selectHyperslab(H5S_SELECT_SET, &count_1d, &time_offset);
        status_dset.write(cached_status.data(), H5::PredType::NATIVE_INT, mspace_1d, status_dspace);

        for (const auto &pair : cached_camera_time) {
            auto camtime_dset = file.openDataSet("solution/time_" + pair.first);
            camtime_dset.extend(&new_size_1d);
            auto camtime_dspace = camtime_dset.getSpace();
            camtime_dspace.selectHyperslab(H5S_SELECT_SET, &count_1d, &time_offset);
            camtime_dset.write(pair.second.data(), H5::PredType::NATIVE_DOUBLE, mspace_1d, camtime_dspace);
        }

        const hsize_t new_size_2d[] = {new_size_1d, nvox};
        auto dataset = file.openDataSet("solution/value");
        dataset.extend(new_size_2d);
        hsize_t dset_offset[] = {time_offset, 0};
        const hsize_t count[] = {1, nvox};
        const H5::DataSpace memspace(1, count + 1);
        auto dataspace = dataset.getSpace();
        for (size_t it=0; it<cached_solutions.size(); ++it) {
            dset_offset[0] = time_offset + it;
            dataspace.selectHyperslab(H5S_SELECT_SET, count, dset_offset);
            dataset.write(cached_solutions[it].data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}

int Solution::flush_hdf5() {

    if (!cached_solutions.size()) return 0;

    int res = (first_flush) ? create_hdf5() : update_hdf5();

    first_flush = false;
    cached_time.clear();
    cached_status.clear();
    for (auto &pair : cached_camera_time) { pair.second.clear(); }
    cached_solutions.clear();

    return res;
}
