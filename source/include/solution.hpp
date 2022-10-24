
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
             size_t nvoxel);

    ~Solution();
};
