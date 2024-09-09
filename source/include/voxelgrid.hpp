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


class BaseVoxelGrid {

protected:
    size_t nx;
    size_t ny;
    size_t nz;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
    double dx;
    double dy;
    double dz;
    int coordsys;
    size_t nvox;
    std::vector<int> voxmap;

public:
    static const int CARTESIAN;
    static const int CYLINDRICAL;

    static int get_coordinate_system_hdf5(std::string filename, std::string group_name);

    inline size_t x_size() const {return nx;};
    inline size_t y_size() const {return ny;};
    inline size_t z_size() const {return nz;};
    inline double x_min() const {return xmin;};
    inline double x_max() const {return xmax;};
    inline double y_min() const {return ymin;};
    inline double y_max() const {return ymax;};
    inline double z_min() const {return zmin;};
    inline double z_max() const {return zmax;};

    const std::vector<int>& voxel_map() const;

    inline int voxel_index(size_t i, size_t j, size_t k) const {return voxmap[i * ny * nz + j * nz + k];};
    virtual int voxel_index(double x, double y, double z) const;

    inline size_t nvoxel() const {return nvox;};

    int read_hdf5(const std::vector<std::string>& filenames, std::string group_name);
    int write_hdf5(std::string filename, std::string group_name) const;

    BaseVoxelGrid() {};
    virtual ~BaseVoxelGrid() {};
};


class CartesianVoxelGrid: public BaseVoxelGrid {

public:
    int voxel_index(double x, double y, double z) const;
    int read_hdf5(const std::vector<std::string>& filenames, std::string group_name);

    CartesianVoxelGrid();
    CartesianVoxelGrid(const std::vector<std::string>& filenames, std::string group_name);
};


class CylindricalVoxelGrid: public BaseVoxelGrid {

public:
    int voxel_index(double x, double y, double z) const;
    int read_hdf5(const std::vector<std::string>& filenames, std::string group_name);

    CylindricalVoxelGrid();
    CylindricalVoxelGrid(const std::vector<std::string>& filenames, std::string group_name);
};
