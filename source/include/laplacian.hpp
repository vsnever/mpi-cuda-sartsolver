// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <vector>
#include <string>


class LaplacianMatrix {
    // sparse flattened Laplacian matrix
    size_t nvox;
    std::vector<size_t> index1d;
    std::vector<float> val;

public:

    inline size_t nvoxel() const {return nvox;}
    inline size_t size() const {return val.size();}

    const std::vector<float>& value() const;
    inline float value(size_t i) const {return val[i];}

    const std::vector<size_t>& index() const;
    inline size_t index(size_t i) const {return index1d[i];}

    float matrix(size_t i, size_t j) const;

    int read_hdf5(const std::string& filename);

    LaplacianMatrix(size_t nvoxel=0);

};
