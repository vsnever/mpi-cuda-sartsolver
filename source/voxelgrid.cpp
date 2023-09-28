// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution and use in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#include <iostream>
#include <algorithm>
#include <H5Cpp.h>
#include <cmath>
#include "voxelgrid.hpp"


const int BaseVoxelGrid::CARTESIAN = 0;
const int BaseVoxelGrid::CYLINDRICAL = 1;

int BaseVoxelGrid::get_coordinate_system_hdf5(std::string filename, std::string group_name) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        auto group = file.openGroup(group_name);

        if (group.attrExists("coordinate_system")) {
            std::string coordinate_system;
            const auto coordsys_attr = group.openAttribute("coordinate_system");
            coordsys_attr.read(coordsys_attr.getStrType(), coordinate_system);
            transform(coordinate_system.begin(), coordinate_system.end(), coordinate_system.begin(), ::tolower);
            return (coordinate_system == "cylindrical") ? CYLINDRICAL : CARTESIAN;
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return CARTESIAN;
}

int BaseVoxelGrid::read_hdf5(const std::vector<std::string>& filenames, std::string group_name) {

    try {
        H5::H5File file(filenames[0], H5F_ACC_RDONLY);

        auto group = file.openGroup(group_name);

        group.openAttribute("nx").read(H5::PredType::NATIVE_HSIZE, &nx);
        group.openAttribute("ny").read(H5::PredType::NATIVE_HSIZE, &ny);
        group.openAttribute("nz").read(H5::PredType::NATIVE_HSIZE, &nz);

        if (group.attrExists("xmin")) group.openAttribute("xmin").read(H5::PredType::NATIVE_DOUBLE, &xmin);
        else xmin = 0;            
        if (group.attrExists("xmax")) group.openAttribute("xmax").read(H5::PredType::NATIVE_DOUBLE, &xmax);
        else xmax = 1;
        if (group.attrExists("ymin")) group.openAttribute("ymin").read(H5::PredType::NATIVE_DOUBLE, &ymin);
        else ymin = 0;
        if (group.attrExists("ymax")) group.openAttribute("ymax").read(H5::PredType::NATIVE_DOUBLE, &ymax);
        else ymax = 1;
        if (group.attrExists("zmin")) group.openAttribute("zmin").read(H5::PredType::NATIVE_DOUBLE, &zmin);
        else zmin = 0;
        if (group.attrExists("zmax")) group.openAttribute("zmax").read(H5::PredType::NATIVE_DOUBLE, &zmax);
        else zmax = 1;

        voxmap.resize(nx * ny * nz, -1);

        int nvoxel_prev = 0;

        for (const auto& filename : filenames) {
            file = H5::H5File(filename, H5F_ACC_RDONLY);

            int nvoxel = 0;   

            group = file.openGroup(group_name);

            hsize_t dims;
            const auto idset = group.openDataSet("i");
            idset.getSpace().getSimpleExtentDims(&dims);
            std::vector<size_t> i(dims);
            idset.read(i.data(), H5::PredType::NATIVE_HSIZE);

            std::vector<size_t> j(dims);
            group.openDataSet("j").read(j.data(), H5::PredType::NATIVE_HSIZE);

            std::vector<size_t> k(dims);
            group.openDataSet("k").read(k.data(), H5::PredType::NATIVE_HSIZE);

            std::vector<int> value(dims);
            group.openDataSet("value").read(value.data(), H5::PredType::NATIVE_INT);

            for (size_t indx=0; indx<dims; ++indx){
                const auto iflat = i[indx] * ny * nz + j[indx] * nz + k[indx];
                voxmap[iflat] = value[indx] + nvoxel_prev;
                nvoxel = (value[indx] > nvoxel) ? value[indx] : nvoxel;
            }
            nvoxel_prev += nvoxel + 1;
        }
        nvox = (size_t)nvoxel_prev;
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny;
    dz = (zmax - zmin) / nz;

    return 0;
}

int BaseVoxelGrid::write_hdf5(std::string filename, std::string group_name) const {

    try {
        H5::H5File file(filename, H5F_ACC_RDWR);

        auto group = file.createGroup(group_name);

        H5::DataSpace attspace(H5S_SCALAR);

        group.createAttribute("nx", H5::PredType::NATIVE_HSIZE, attspace)
             .write(H5::PredType::NATIVE_HSIZE, &nx);
        group.createAttribute("ny", H5::PredType::NATIVE_HSIZE, attspace)
             .write(H5::PredType::NATIVE_HSIZE, &ny);
        group.createAttribute("nz", H5::PredType::NATIVE_HSIZE, attspace)
             .write(H5::PredType::NATIVE_HSIZE, &nz);

        group.createAttribute("xmin", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &xmin);
        group.createAttribute("xmax", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &xmax);
        group.createAttribute("ymin", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &ymin);
        group.createAttribute("ymax", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &ymax);
        group.createAttribute("zmin", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &zmin);
        group.createAttribute("zmax", H5::PredType::NATIVE_DOUBLE, attspace)
             .write(H5::PredType::NATIVE_DOUBLE, &zmax);

        H5::StrType strtype(H5::PredType::C_S1, H5T_VARIABLE);

        if (coordsys == CARTESIAN) {
            group.createAttribute("coordinate_system", strtype, attspace)
                 .write(strtype, std::string("cartesian"));
        }
        if (coordsys == CYLINDRICAL) {
            group.createAttribute("coordinate_system", strtype, attspace)
                 .write(strtype, std::string("cylindrical"));
        }

        const hsize_t n = std::count_if(voxmap.begin(), voxmap.end(), [](int v){return (v > -1);});

        std::vector<int> i(n);
        std::vector<int> j(n);
        std::vector<int> k(n);
        std::vector<int> value(n);

        size_t count = 0;
        for (size_t iflat = 0; iflat < voxmap.size(); ++iflat) {
            if (voxmap[iflat] > -1) {
                i[count] = (int) (iflat / (ny * nz));
                const size_t irem = iflat % (ny * nz);
                j[count] = (int) (irem / nz);
                k[count] = (int) (irem % nz);
                value[count] = voxmap[iflat];
                count++;
            }
        }

        H5::DataSpace dataspace(1, &n);
        group.createDataSet("i", H5::PredType::NATIVE_INT, dataspace)
             .write(i.data(), H5::PredType::NATIVE_INT);
        group.createDataSet("j", H5::PredType::NATIVE_INT, dataspace)
             .write(j.data(), H5::PredType::NATIVE_INT);
        group.createDataSet("k", H5::PredType::NATIVE_INT, dataspace)
             .write(k.data(), H5::PredType::NATIVE_INT);
        group.createDataSet("value", H5::PredType::NATIVE_INT, dataspace)
             .write(value.data(), H5::PredType::NATIVE_INT);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}

const std::vector<int>& BaseVoxelGrid::voxel_map() const {return voxmap;};

int BaseVoxelGrid::voxel_index(double x, double y, double z) const{
    std::cerr << "Method not implemented." << std::endl;
    std::exit(1);
}


CartesianVoxelGrid::CartesianVoxelGrid():
    BaseVoxelGrid() {
        coordsys = CARTESIAN;
    }

CartesianVoxelGrid::CartesianVoxelGrid(const std::vector<std::string>& filenames, std::string group_name):
    BaseVoxelGrid() {
        coordsys = CARTESIAN;
        read_hdf5(filenames, group_name);
    }

int CartesianVoxelGrid::read_hdf5(const std::vector<std::string>& filenames, std::string group_name) {

    try {
        H5::H5File file(filenames[0], H5F_ACC_RDONLY);

        auto group = file.openGroup(group_name);

        if (group.attrExists("coordinate_system")) {
            std::string coordinate_system;
            const auto coordsys_attr = group.openAttribute("coordinate_system");
            coordsys_attr.read(coordsys_attr.getStrType(), coordinate_system);
            std::transform(coordinate_system.begin(), coordinate_system.end(), coordinate_system.begin(), ::tolower);
            if (coordinate_system == "cylindrical") {
                std::cerr << "CartesianVoxelGrid cannot read cylindrical voxel map." << std::endl;
                std::exit(1);
            }
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    BaseVoxelGrid::read_hdf5(filenames, group_name);

    return 0;
}

int CartesianVoxelGrid::voxel_index(double x, double y, double z) const {

    if (!voxmap.size()) {
        std::cerr << "Voxel map is not initialized." << std::endl;
        std::exit(1);
    }

    if ((x < xmin) || (x >= xmax) || (y < ymin) || (y >= ymax) || (z < zmin) || (z >= zmax)) return -1;

    size_t i = (size_t)((x - xmin) / dx);
    size_t j = (size_t)((y - ymin) / dy);
    size_t k = (size_t)((z - zmin) / dz);

    return voxmap[i * ny * nz + j * nz + k];
}


CylindricalVoxelGrid::CylindricalVoxelGrid():
    BaseVoxelGrid() {
        coordsys = CYLINDRICAL;
    }

CylindricalVoxelGrid::CylindricalVoxelGrid(const std::vector<std::string>& filenames, std::string group_name):
    BaseVoxelGrid() {
        coordsys = CYLINDRICAL;
        read_hdf5(filenames, group_name);
    }

int CylindricalVoxelGrid::read_hdf5(const std::vector<std::string>& filenames, std::string group_name) {

    try {
        H5::H5File file(filenames[0], H5F_ACC_RDONLY);

        auto group = file.openGroup(group_name);

        if (group.attrExists("coordinate_system")) {
            std::string coordinate_system;
            const auto coordsys_attr = group.openAttribute("coordinate_system");
            coordsys_attr.read(coordsys_attr.getStrType(), coordinate_system);
            std::transform(coordinate_system.begin(), coordinate_system.end(), coordinate_system.begin(), ::tolower);
            if (coordinate_system == "cartesian") {
                std::cerr << "CylindricalVoxelGrid cannot read Cartesian voxel map." << std::endl;
                std::exit(1);
            }
        }
        else {
            std::cerr << "CylindricalVoxelGrid cannot read Cartesian voxel map." << std::endl;
            std::exit(1);
        }

    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    BaseVoxelGrid::read_hdf5(filenames, group_name);

    if (std::fmod(360., (ymax - ymin)) > 0.001) {
        std::cerr << (ymax - ymin) << " is not a divisor of 360." << std::endl;
        std::exit(1);
    }

    return 0;
}

int CylindricalVoxelGrid::voxel_index(double x, double y, double z) const {

    if (!voxmap.size()) {
        std::cerr << "Voxel map is not initialized." << std::endl;
        std::exit(1);
    }

    double r = std::sqrt(x * x + y * y);

    if ((r < xmin) || (r >= xmax) || (z < zmin) || (z >= zmax)) return -1;

    double period = ymax - ymin;
    double phi = 180. / M_PI * std::atan2(y, x);
    if (phi < 0) phi += 360.;
    phi = std::fmod(phi, period);

    size_t i = (size_t)((r - xmin) / dx);
    size_t j = (size_t)((phi - ymin) / dy);
    size_t k = (size_t)((z - zmin) / dz);

    return voxmap[i * ny * nz + j * nz + k];
}
