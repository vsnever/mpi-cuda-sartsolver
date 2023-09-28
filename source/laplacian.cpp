// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution and use in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <iostream>
#include <numeric>
#include <algorithm>
#include <H5Cpp.h>
#include "laplacian.hpp"


LaplacianMatrix::LaplacianMatrix(size_t nvoxel): nvox(nvoxel) {}

const std::vector<float>& LaplacianMatrix::value() const {return val;}

const std::vector<size_t>& LaplacianMatrix::index() const {return index1d;}

float LaplacianMatrix::matrix(size_t i, size_t j) const {
    if ((i >= nvox) || (j >= nvox)) {
        std::cerr << "Indices " << i << "," << j << " are out of range of (" << nvox << "," << nvox << ") matrix." << std::endl;
        std::exit(1);
    }
    auto i1d = i * nvox + j;
    auto lower_index = std::lower_bound(index1d.begin(), index1d.end(), i1d);
    if ((lower_index == index1d.end()) || (*lower_index != i1d)) return 0;

    return val[std::distance(index1d.begin(), lower_index)];
}

int LaplacianMatrix::read_hdf5(const std::string& filename){
    // loads a laplacian matrix.

    try {
        const H5::H5File file(filename, H5F_ACC_RDONLY);

        const auto group = file.openGroup("laplacian");
        
        size_t nvoxel_data;
        group.openAttribute("nvoxel")
             .read(H5::PredType::NATIVE_HSIZE, &nvoxel_data);

        if (nvoxel_data != nvox) {
            std::cerr << "Laplacian and ray-transfer matrices have different number of voxels." << std::endl;
            std::exit(1);
        }

        const auto dset = group.openDataSet("value"); 
        hsize_t dims;
        dset.getSpace().getSimpleExtentDims(&dims);
        index1d.resize(dims);
        val.resize(dims);

        std::vector<size_t> i_index(dims);
        std::vector<size_t> j_index(dims);
        std::vector<size_t> index1d_temp(dims);
        std::vector<float> val_temp(dims);


        dset.read(val_temp.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("i").read(i_index.data(), H5::PredType::NATIVE_HSIZE);
        group.openDataSet("j").read(j_index.data(), H5::PredType::NATIVE_HSIZE);

        for (size_t i=0; i < dims; ++i) {index1d_temp[i] = i_index[i] * nvox + j_index[i];};

        if (std::is_sorted(index1d_temp.begin(), index1d_temp.end())) {
            index1d = index1d_temp;
            val = val_temp;            
        }
        else{
            std::vector<size_t> indices(dims);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) -> bool {return index1d_temp[a] < index1d_temp[b];});

            for (size_t i=0; i < dims; ++i) {
                index1d[i] = index1d_temp[indices[i]];
                val[i] = val_temp[indices[i]];
            }
        }

    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}
