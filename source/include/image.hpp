// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution use in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <array>
#include <vector>
#include <map>
#include <string>
#include <utility>


class CompositeImage {
    std::map<std::string, std::string>& files;
    std::map<std::string, std::vector<int>>& rtm_frame_masks;
    size_t offset_pix;
    size_t npix;
    size_t cframe_index;
    size_t cache_offset;
    size_t max_cache_size;
    std::vector<double> time;
    std::vector<std::vector<double>> camera_time;
    std::vector<std::vector<size_t>> frame_indices;
    std::vector<double> cached_frames;

    bool is_cached(size_t i) const;

    void frame_indices_from_timepairs(const std::vector<std::vector<std::pair<double, size_t>>> &timepairs,
                                      double step, double threshold);

    int read_frame_indices_hdf5(const std::vector<std::array<double, 4>>& time_intervals);

    int cache_hdf5(size_t i);

public:

    size_t get_max_cache_size() const;
    void set_max_cache_size(size_t value);

    bool next_frame(std::vector<double>& frame);

    std::vector<double> frame(size_t i);
    std::vector<double> frame();

    double frame_time(size_t i) const;
    double frame_time() const;

    std::vector<double> camera_frame_time(size_t i) const;
    std::vector<double> camera_frame_time() const;

    inline size_t offset_pixel() const {return offset_pix;};
    inline size_t npixel() const {return npix;};
    inline size_t nframe() const {return time.size();};
    inline size_t current_frame_index() const {return cframe_index;};

    CompositeImage(std::map<std::string, std::string>& image_files,
                   std::map<std::string, std::vector<int>>& frame_masks,
                   const std::vector<std::array<double, 4>>& time_intervals,
                   size_t npixel, size_t offset_pixel=0);
};
