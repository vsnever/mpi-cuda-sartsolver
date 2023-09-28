// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution and use in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <vector>
#include <map>
#include <string>


class Solution {
    std::string filename;
    size_t nvox;
    std::map<std::string, std::vector<double>> cached_camera_time;
    std::vector<double> cached_time;
    std::vector<int> cached_status;
    std::vector<std::vector<double>> cached_solutions;
    
    size_t max_cache_size;
    bool first_flush;

    int create_hdf5();
    int update_hdf5();

public:

    size_t get_max_cache_size() const;
    void set_max_cache_size(size_t value);

    int add(const std::vector<double>& solution, int status, double time, const std::vector<double>& camera_time);

    int flush_hdf5();

    inline size_t nvoxel() const {return nvox;};

    Solution(const std::string& filename,
             const std::vector<std::string>& camera_names,
             size_t nvoxel, size_t cache_size=100);

    ~Solution();
};
