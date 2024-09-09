// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <iostream>
#include <algorithm>
#include <numeric>
#include <H5Cpp.h>
#include "raytransfer.hpp"

RayTransferMatrix::RayTransferMatrix():
    offset_pix(0),
    npix(0),
    nvox(0) {};

RayTransferMatrix::RayTransferMatrix(size_t npixel, size_t nvoxel, size_t offset_pixel):
    offset_pix(offset_pixel),
    npix(npixel),
    nvox(nvoxel) {mat.resize(npix * nvox);};

const std::vector<float>& RayTransferMatrix::matrix() const {return mat;};

int RayTransferMatrix::read_hdf5(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files,
                                 const std::string& group_name){
    // loads a part of the RTM according to offset_pixel and npixel.

    const hsize_t rtm_size = mat.size();
    if (rtm_size == 0) {
        std::cerr << "To read RayTransferMatrix, its size must be non-zero." << std::endl;
        std::exit(1);
    }

    H5::DataSpace memspace(1, &rtm_size);
    try {
        size_t start_pixel = 0;

        for (const auto& p : sorted_matrix_files) {
            const H5::H5File file0(p.second[0], H5F_ACC_RDONLY);

            size_t npixel_data;
            file0.openGroup("rtm")
                 .openAttribute("npixel")
                 .read(H5::PredType::NATIVE_HSIZE, &npixel_data);

            if (offset_pix < start_pixel + npixel_data) {
                size_t start_voxel = 0;

                for (const auto& filename : p.second) {
                    const H5::H5File file(filename, H5F_ACC_RDONLY);

                    const auto rtm_group = file.openGroup("rtm");

                    size_t nvoxel_data;
                    rtm_group.openAttribute("nvoxel")
                             .read(H5::PredType::NATIVE_HSIZE, &nvoxel_data);

                    const auto group = rtm_group.openGroup(group_name);

                    int is_sparse;
                    group.openAttribute("is_sparse")
                         .read(H5::PredType::NATIVE_INT, &is_sparse);

                    if (is_sparse) {
                        hsize_t dims;
                        group.openDataSet("value")
                             .getSpace()
                             .getSimpleExtentDims(&dims);

                        std::vector<size_t> pixel_index(dims);
                        std::vector<size_t> voxel_index(dims);
                        std::vector<float> value(dims);
                        group.openDataSet("pixel_index").read(pixel_index.data(), H5::PredType::NATIVE_HSIZE);
                        group.openDataSet("voxel_index").read(voxel_index.data(), H5::PredType::NATIVE_HSIZE);
                        group.openDataSet("value").read(value.data(), H5::PredType::NATIVE_FLOAT);

                        std::transform(pixel_index.begin(), pixel_index.end(), pixel_index.begin(), [&](size_t x){return(x + start_pixel);});
                        std::transform(voxel_index.begin(), voxel_index.end(), voxel_index.begin(), [&](size_t x){return(x + start_voxel);});

                        const auto last_pixel = offset_pix + npix;

                        for (size_t i=0; i<dims; ++i){
                            if ((pixel_index[i] >= offset_pix) && (pixel_index[i] < last_pixel)) {
                                mat[(pixel_index[i] - offset_pix) * nvox + voxel_index[i]] = value[i];
                            }
                        }

                    }
                    else {
                        const auto dset = group.openDataSet("value"); 
                        auto dataspace = dset.getSpace();

                        const size_t ipix_begin = (offset_pix > start_pixel) ? offset_pix - start_pixel : 0;
                        const size_t ipix_end = (offset_pix + npix > start_pixel + npixel_data) ? npixel_data : offset_pix + npix - start_pixel;
                        const size_t pix_offset = (offset_pix > start_pixel) ? 0 : start_pixel - offset_pix;
                        hsize_t dset_offset[2] = {0, 0};
                        const hsize_t count[2] = {1, nvoxel_data};

                        // reading slices of a 2D datasets to 1D buffer
                        for (size_t ipix=ipix_begin; ipix<ipix_end; ++ipix) {
                            dset_offset[0] = ipix;
                            const hsize_t mem_offset = (pix_offset + ipix - ipix_begin) * nvox + start_voxel;
                            memspace.selectHyperslab(H5S_SELECT_SET, &count[1], &mem_offset);
                            dataspace.selectHyperslab(H5S_SELECT_SET, count, dset_offset);
                            dset.read(mat.data(), H5::PredType::NATIVE_FLOAT, memspace, dataspace);
                        }
                    }

                    start_voxel += nvoxel_data;
                }
            }

            start_pixel += npixel_data;

            if (offset_pix + npix < start_pixel) break;
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}
