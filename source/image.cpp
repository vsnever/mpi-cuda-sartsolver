// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution and use in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <H5Cpp.h>
#include "image.hpp"


#define TIME_EPSILON 1.e-10


CompositeImage::CompositeImage(std::map<std::string, std::string>& image_files,
                               std::map<std::string, std::vector<int>>& frame_masks,
                               const std::vector<std::array<double, 4>>& time_intervals,
                               size_t npixel,
                               size_t offset_pixel):
                files(image_files),
                rtm_frame_masks(frame_masks) {
    if (npixel == 0) {
        std::cerr << "Argument npixel must be positive." << std::endl;
        std::exit(1);
    }
    npix = npixel;
    offset_pix = offset_pixel;
    cache_offset = 0;
    max_cache_size = 100;

    read_frame_indices_hdf5(time_intervals);

    cframe_index = time.size();
}


size_t CompositeImage::get_max_cache_size() const{return max_cache_size;}

void CompositeImage::set_max_cache_size(size_t value) {
    if (value == 0) {
        std::cerr << "Attribute max_cache_size must be positive." << std::endl;
        std::exit(1);
    }
    max_cache_size = value;
}


int CompositeImage::read_frame_indices_hdf5(const std::vector<std::array<double, 4>>& time_intervals) {

    std::vector<std::vector<double>> timelines;

    try {
        for (const auto& p : files) {
            const H5::H5File file(p.second, H5F_ACC_RDONLY);

            const auto time_dset = file.openDataSet("image/time");

            hsize_t dims;
            time_dset.getSpace().getSimpleExtentDims(&dims);

            std::vector<double> timeline(dims);
            time_dset.read(timeline.data(), H5::PredType::NATIVE_DOUBLE);

            if (!std::is_sorted(timeline.begin(), timeline.end(), [&](const double a, const double b) {return a < b;})) {
                std::cerr << "Image frames are not sorted by time in " << p.second << "." << std::endl;
                std::exit(1);
            }

            timelines.push_back(timeline);

        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    for (const auto& ti : time_intervals) {
        std::vector<std::vector<std::pair<double, size_t>>> timepairs;
        for (const auto &tline : timelines) {
            std::vector<std::pair<double, size_t>> ctimepairs(tline.size());
            size_t count=0;
            for (size_t i=0; i<tline.size(); ++i) {
                if ((tline[i] >= ti[0]) && (tline[i] <= ti[1])) {
                    ctimepairs[count].first = tline[i];
                    ctimepairs[count].second = i;
                    count++;
                }
            }
            ctimepairs.resize(count);
            timepairs.push_back(ctimepairs);
        }
        frame_indices_from_timepairs(timepairs, ti[2], ti[3]);
    }

    if (frame_indices.empty()) {
        std::cerr << "No composite images can be created for given time intervals." << std::endl;
        std::exit(1);
    }

    return 0;
}


void CompositeImage::frame_indices_from_timepairs(const std::vector<std::vector<std::pair<double, size_t>>> &timepairs, double step, double threshold){
    /*
    Obtains the frame indices for the composite images using the timelines of different cameras.

    The images captured by two different cameras will be combined into a single image if they are taken within time_threshold interval.

    */ 

    // obtaining the time range
    double min_time = timepairs.front().front().first;
    double max_time = timepairs.front().back().first;
    for (const auto &tpair : timepairs) {
        min_time = std::min(tpair.front().first, min_time);
        max_time = std::max(tpair.back().first, max_time);
    }

    // obtaining the time step if not provided
    if (step == 0) {
        if ((max_time - min_time) < TIME_EPSILON) step = 1.; // all timepairs contain only a single time moment
        else {
            for (const auto &tpair : timepairs) {
                double min_diff = tpair.back().first - tpair.front().first;
                for (auto iter=tpair.begin(); iter!=tpair.end() - 1; ++iter) {min_diff = std::min((iter + 1)->first - iter->first, min_diff);}
                step = std::max(min_diff, step);
            }
        }
    }

    if (threshold == 0) threshold = step;

    // increasing the time range in both sides on a step to avoid border checks in the future
    min_time -= step;
    max_time += step;

    const size_t max_num_frames = (size_t)std::round((max_time - min_time) / step) + 1;
    const size_t num_cam = timepairs.size();

    // composite frame grid is flatten
    std::vector<std::pair<double, size_t>> composite_frame_grid(max_num_frames * num_cam, std::make_pair<double, size_t>(1.01 * threshold, 0));

    // filling in the composite frame grid
    for (size_t icam=0; icam<num_cam; ++icam) {
        for (const auto &tpair : timepairs[icam]) {
            const size_t iframe = (size_t)std::round((tpair.first - min_time) / step);
            for (int i=-1; i<2; ++i) {  // updating also previous and the next frames
                const size_t index = num_cam * (iframe + i) + icam;
                const double delta = tpair.first - min_time - (iframe + i) * step;
                if (std::abs(delta) + TIME_EPSILON < std::abs(composite_frame_grid[index].first)) {
                    // "+ TIME_EPSILON" to prefer previous frame over the next one if the frames are equally-distant
                    composite_frame_grid[index].first = delta;
                    composite_frame_grid[index].second = tpair.second;
                }
            }
        }
    }

    // filling in the frame indices from the frame grid
    double last_time_delta = 0;  // the total distance of the individual frame time moments and the last composite frame time moment  
    for (size_t iframe=1; iframe<max_num_frames-1; ++iframe) {
        std::vector<size_t> iframe_indices;
        std::vector<double> icamera_time;
        const double ftime = min_time + iframe * step;
        double time_delta = 0;

        for (size_t icam=0; icam<num_cam; ++icam) {
            const size_t index = num_cam * iframe + icam;
            const double delta = composite_frame_grid[index].first;
            const double abs_delta = std::abs(delta);
            if (abs_delta > threshold + TIME_EPSILON) break;
            iframe_indices.push_back(composite_frame_grid[index].second);
            icamera_time.push_back(ftime + delta);
            time_delta += abs_delta;
        }

        if (iframe_indices.size() == num_cam) {
            // add, if not equal to the last frame
            if (frame_indices.empty() || (iframe_indices != frame_indices.back())) {
                frame_indices.push_back(iframe_indices);
                camera_time.push_back(icamera_time);
                time.push_back(ftime);
            }
            // or update the time if the frame is closer to this time mark than to the last one
            else if (time_delta + TIME_EPSILON < last_time_delta) *(time.end() - 1) = ftime;
            last_time_delta = time_delta;
        }
    }
}


bool CompositeImage::is_cached(size_t i) const {
    return ((i >= cache_offset) && (i < cache_offset + cached_frames.size() / npix)) ? true : false;
}


std::vector<double> CompositeImage::frame(size_t i) {
    if (i >= time.size()) {
        std::cerr << "Index " << i << " is out of bounds (" << time.size() << ")." << std::endl;
        std::exit(1);
    }

    if (!is_cached(i)) cache_hdf5(i);
    cframe_index = i;
    const size_t ioff = (i - cache_offset) * npix;

    return std::vector<double>(cached_frames.begin() + ioff, cached_frames.begin() + ioff + npix);
}


std::vector<double> CompositeImage::frame() {

    if (cframe_index == time.size()) return frame(0); //initial state

    return frame(cframe_index);
}


bool CompositeImage::next_frame(std::vector<double>& fr) {

    if (cframe_index + 1 == time.size()) return false;

    fr = (cframe_index == time.size()) ? frame(0) : frame(cframe_index + 1);

    return true;
}


double CompositeImage::frame_time(size_t i) const {
    if (i >= time.size()) {
        std::cerr << "Index " << i << " is out of bounds (" << time.size() << ")." << std::endl;
        std::exit(1);
    }

    return time[i];
}


double CompositeImage::frame_time() const {

    return time[cframe_index];
}


std::vector<double> CompositeImage::camera_frame_time(size_t i) const {
    if (i >= time.size()) {
        std::cerr << "Index " << i << " is out of bounds (" << time.size() << ")." << std::endl;
        std::exit(1);
    }

    return camera_time[i];
}


std::vector<double> CompositeImage::camera_frame_time() const {

    return camera_time[cframe_index];
}


int CompositeImage::cache_hdf5(size_t itime) {

    const auto cache_size_t = (itime + max_cache_size < time.size()) ? max_cache_size : time.size() - itime;

    cached_frames.resize(cache_size_t * npix);

    try {
        size_t start_pixel = 0;

        size_t icam = 0;
        for (const auto& p : rtm_frame_masks) {
            const auto& mask = p.second;
            auto npixel_masked = std::accumulate(mask.begin(), mask.end(), 0);

            if (offset_pix < start_pixel + npixel_masked) {

                const H5::H5File file(files[p.first], H5F_ACC_RDONLY);

                const auto dset = file.openDataSet("image/frame");
                auto dataspace = dset.getSpace();
                hsize_t dims[3];
                dataspace.getSimpleExtentDims(dims);
                hsize_t dset_offset[] = {frame_indices[itime][icam], 0, 0};
                const hsize_t count[] = {1, dims[1], dims[2]};
                const hsize_t frame_size_1d = dims[1] * dims[2];

                std::vector<double> full_frame(frame_size_1d);
                std::vector<double> masked_frame(npixel_masked);
                H5::DataSpace memspace(1, &frame_size_1d);

                const size_t ipix_begin = (offset_pix > start_pixel) ? offset_pix - start_pixel : 0;
                const size_t ipix_end = (offset_pix + npix > start_pixel + npixel_masked) ? npixel_masked : offset_pix + npix - start_pixel;
                const size_t pix_offset = (offset_pix > start_pixel) ? 0 : start_pixel - offset_pix;

                for (size_t it=0; it<cache_size_t; ++it) {
                    dset_offset[0] = frame_indices[itime + it][icam];
                    dataspace.selectHyperslab(H5S_SELECT_SET, count, dset_offset);
                    dset.read(full_frame.data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);

                    size_t ipix_m = 0;
                    for (size_t ipix=0; ipix<frame_size_1d; ++ipix) {
                        if (mask[ipix]) {
                            masked_frame[ipix_m] = full_frame[ipix];
                            ++ipix_m;
                        }
                    }
                    std::copy(masked_frame.begin() + ipix_begin, masked_frame.begin() + ipix_end, cached_frames.begin() + it * npix + pix_offset);
                }

            }
            start_pixel += (size_t)npixel_masked;
            icam++;

            if (offset_pix + npix < start_pixel) break;
        }
        cache_offset = itime;
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}
