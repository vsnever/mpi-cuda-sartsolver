// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <vector>
#include <map>
#include <string>


class RayTransferMatrix {
    size_t offset_pix;
    size_t npix;
    size_t nvox;
    std::vector<float> mat;

public:

    const std::vector<float>& matrix() const;
    inline float matrix(size_t ipix, size_t jvox) const {return mat[ipix * nvox + jvox];};

    inline size_t offset_pixel() const {return offset_pix;};
    inline size_t npixel() const {return npix;};
    inline size_t nvoxel() const {return nvox;};

    int read_hdf5(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files,
                  const std::string& group_name);

    RayTransferMatrix();

    RayTransferMatrix(size_t npixel, size_t nvoxel, size_t offset_pixel=0);
};
