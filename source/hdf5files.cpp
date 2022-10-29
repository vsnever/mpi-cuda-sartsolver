
#include <memory>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "hdf5files.hpp"


#define TIME_EPSILON 1.e-9


int categorize_input_files(const std::vector<std::string>& input_files,
                           std::vector<std::string>& matrix_files,
                           std::vector<std::string>& image_files){

    for (const auto& filename : input_files) {
        try {
            H5::H5File file(filename, H5F_ACC_RDONLY);

            if (file.nameExists("rtm")) matrix_files.push_back(filename);
            else if (file.nameExists("image")) image_files.push_back(filename);
            else {
                std::cerr << "The file " << filename << " is neither an RTM file nor an image file." << std::endl;
                std::exit(1);
            }

    	}
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
    	}
	}

	return 0;
}


std::map<std::string, std::vector<std::string>> sort_rtm_files(const std::vector<std::string>& files){
    // sorts RTM files in view names
    std::map<std::string, std::map<size_t, std::string>> sorted;

    for (const auto& filename : files) {
        try {
            const H5::H5File file(filename, H5F_ACC_RDONLY);

            std::string camera_name;
            const auto name_attr = file.openGroup("rtm").openAttribute("camera_name");
            name_attr.read(name_attr.getStrType(), camera_name);

            const auto vmap_group = file.openGroup("rtm/voxel_map");

            hsize_t dims;
            const auto idset = vmap_group.openDataSet("i");
            idset.getSpace().getSimpleExtentDims(&dims);
            std::vector<size_t> i(dims);
            idset.read(i.data(), H5::PredType::NATIVE_HSIZE);

            std::vector<size_t> j(dims);
            vmap_group.openDataSet("j").read(j.data(), H5::PredType::NATIVE_HSIZE);

            std::vector<size_t> k(dims);
            vmap_group.openDataSet("k").read(k.data(), H5::PredType::NATIVE_HSIZE);

            size_t nx, ny, nz;
            vmap_group.openAttribute("nx").read(H5::PredType::NATIVE_HSIZE, &nx);
            vmap_group.openAttribute("ny").read(H5::PredType::NATIVE_HSIZE, &ny);
            vmap_group.openAttribute("nz").read(H5::PredType::NATIVE_HSIZE, &nz);

            auto indx_min = nx * ny * nz;
            for (size_t indx=0; indx<dims; ++indx){
                auto indx_temp = i[indx] * ny * nz + j[indx] * nz + k[indx];
                indx_min = indx_temp < indx_min ? indx_temp : indx_min;
            }

            if ( sorted.find(camera_name) == sorted.end() ) {
                sorted[camera_name] = {{indx_min, filename}};
            } else {
                sorted[camera_name][indx_min] = filename;
            }
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
        }
    }

    std::map<std::string, std::vector<std::string>> sorted_matrix_files;
    for (const auto& filemap : sorted) {
        std::vector<std::string> filenames;
        for (const auto& p : filemap.second) filenames.push_back(p.second);
        sorted_matrix_files[filemap.first] = filenames;
    }

    return sorted_matrix_files;
}


int check_rtm_frame_consistency(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files) {
    // checks if the same views have identical frame properties in all RTM files

    for (const auto& p : sorted_matrix_files) {
        // no need to check for frame consistency in case of a single RTM file per view
        if (p.second.size() < 2) continue;

        try {
            std::vector<uint8_t> ref_mask;

            for (const auto& filename : p.second) {
                const H5::H5File file(filename, H5F_ACC_RDONLY);

                const auto mask_dset = file.openDataSet("rtm/frame_mask");

                hsize_t dims[2];
                mask_dset.getSpace().getSimpleExtentDims(dims);

                std::vector<uint8_t> mask(dims[0] * dims[1]);
                mask_dset.read(mask.data(), H5::PredType::NATIVE_UINT8);

                if (ref_mask.size()) {
                    if (mask != ref_mask) {
                        std::cerr << "RTM files for " << p.first << " view have different frame masks." << std::endl;
                        std::exit(1);
                    }
                }   
                else ref_mask = mask;
            }
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
        }
    }

    return 0;
}


int check_rtm_voxel_consistency(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files) {
    // checks if the voxel map is the same for different views

    std::vector<int> ref_voxel_map;

    for (const auto& p : sorted_matrix_files) {

        try {
            H5::H5File file(p.second[0], H5F_ACC_RDONLY);

            size_t nx, ny, nz;
            auto vmap_group = file.openGroup("rtm/voxel_map");
            vmap_group.openAttribute("nx").read(H5::PredType::NATIVE_HSIZE, &nx);
            vmap_group.openAttribute("ny").read(H5::PredType::NATIVE_HSIZE, &ny);
            vmap_group.openAttribute("nz").read(H5::PredType::NATIVE_HSIZE, &nz);

            std::vector<int> voxel_map(nx * ny * nz, -1);

            int nsource_prev = 0;

            for (const auto& filename : p.second) {
                file = H5::H5File(filename, H5F_ACC_RDONLY);

                int nvox;
                file.openGroup("rtm")
                    .openAttribute("nvoxel")
                    .read(H5::PredType::NATIVE_INT, &nvox);                

                vmap_group = file.openGroup("rtm/voxel_map");

                hsize_t dims;
                const auto idset = vmap_group.openDataSet("i");
                idset.getSpace().getSimpleExtentDims(&dims);
                std::vector<size_t> i(dims);
                idset.read(i.data(), H5::PredType::NATIVE_HSIZE);

                std::vector<size_t> j(dims);
                vmap_group.openDataSet("j").read(j.data(), H5::PredType::NATIVE_HSIZE);

                std::vector<size_t> k(dims);
                vmap_group.openDataSet("k").read(k.data(), H5::PredType::NATIVE_HSIZE);

                std::vector<int> value(dims);
                vmap_group.openDataSet("value").read(value.data(), H5::PredType::NATIVE_INT);

                for (size_t indx=0; indx<dims; ++indx){
                    const auto iflat = i[indx] * ny * nz + j[indx] * nz + k[indx];
                    if (voxel_map[iflat] < 0) voxel_map[iflat] = value[indx] + nsource_prev;
                    else {
                        std::cerr << "RTM segments for " << p.first << " view have overlapping voxel maps at element (";
                        std::cerr << i[indx] << "," << j[indx] << "," << k[indx] <<  ")." << std::endl;
                        std::exit(1);
                    }
                }
                nsource_prev += nvox;
            }
            if (ref_voxel_map.size()) {
                if (voxel_map != ref_voxel_map) {
                    std::cerr << "RTM files for " << p.first << " and " << sorted_matrix_files.begin()->first;
                    std::cerr  << " views have different voxel maps." << std::endl;
                    std::exit(1);
                }
            }   
            else ref_voxel_map = voxel_map;
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
        }
    }

    return 0;
}


std::map<std::string, std::vector<int>> read_rtm_frame_masks(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files){
    std::map<std::string, std::vector<int>> frame_masks;

    for (const auto& p : sorted_matrix_files) {
        try {
            const H5::H5File file(p.second[0], H5F_ACC_RDONLY);

            const auto mask_dset = file.openDataSet("rtm/frame_mask");

            hsize_t mask_dims[2];
            mask_dset.getSpace().getSimpleExtentDims(mask_dims);

            std::vector<int> mask(mask_dims[0] * mask_dims[1]);
            mask_dset.read(mask.data(), H5::PredType::NATIVE_INT);
            frame_masks[p.first] = mask;
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
        }
    }

    return frame_masks;
}


std::map<std::string, std::string> sort_image_files(const std::vector<std::string>& files){
    // sort image files according to the diagnostic view names

    std::map<std::string, std::string> sorted;

    for (const auto& filename : files) {
        try {
            const H5::H5File file(filename, H5F_ACC_RDONLY);

            std::string camera_name;
            const auto name_attr = file.openGroup("image").openAttribute("camera_name");
            name_attr.read(name_attr.getStrType(), camera_name);

            if ( sorted.find(camera_name) == sorted.end() ) {
                sorted[camera_name] = filename;
            } else {
                std::cerr << "Image files " << filename << " and " << sorted[camera_name];
                std::cerr  << " share the same diagnostic view: " << camera_name << "." << std::endl;
                std::exit(1);
            }
            
        }
        catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::exit(1);
        }
    }

    return sorted;
}


int check_rtm_image_consistency(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files,
                                const std::map<std::string, std::string>& sorted_image_files,
                                const std::string& rtm_name,
                                const double wvl_threshold){
    // checks that RTM files are consistent with image files

    for (const auto& p : sorted_matrix_files) {
        if (sorted_image_files.find(p.first) == sorted_image_files.end()) {
            std::cerr << "No image file for " << p.first << " camera." << std::endl;
            std::exit(1);
        }
    }
    for (const auto& p : sorted_image_files) {
        if (sorted_matrix_files.find(p.first) == sorted_matrix_files.end()) {
            std::cerr << "No RTM file for " << p.first << " camera." << std::endl;
            std::exit(1);
        }
    }

    try {
        // check the wavelength for the first pair of files only is enough
        // because we already know that wavelengths are the same in all RTM and all image files
        {
            double rtm_wavelength, image_wavelength;
            const H5::H5File rtm_file(sorted_matrix_files.begin()->second[0], H5F_ACC_RDONLY);
            rtm_file.openGroup("rtm/" + rtm_name)
                    .openAttribute("wavelength")
                    .read(H5::PredType::NATIVE_DOUBLE, &rtm_wavelength);
            const H5::H5File image_file(sorted_image_files.begin()->second, H5F_ACC_RDONLY);
            image_file.openGroup("image")
                      .openAttribute("wavelength")
                      .read(H5::PredType::NATIVE_DOUBLE, &image_wavelength);
            if (std::abs(rtm_wavelength - image_wavelength) > wvl_threshold) {
                std::cerr << "RTM wavelength (" << rtm_wavelength << " nm) is not within " << wvl_threshold;
                std::cerr << " nm threshold from image wavelength (" << image_wavelength << " nm)." << std::endl;
                std::exit(1);
            }
        }
        // check frame properties, checking only the first RTM file for
        // each view is enough becuase we already know that frame peoperties are the same in all RTM files for this view
        for (const auto& p : sorted_matrix_files) {

            const H5::H5File rtm_file(p.second[0], H5F_ACC_RDONLY);
            hsize_t rtm_dims[2];
            rtm_file.openDataSet("rtm/frame_mask")
                    .getSpace()
                    .getSimpleExtentDims(rtm_dims);

            const H5::H5File image_file(sorted_image_files.at(p.first), H5F_ACC_RDONLY);
            hsize_t image_dims[3];
            image_file.openDataSet("image/frame")
                      .getSpace()
                      .getSimpleExtentDims(image_dims);

            if ((image_dims[1] != rtm_dims[0]) || (image_dims[2] != rtm_dims[1])) {
                std::cerr << "RTM for " << p.first << " view was calculated for resolution " << rtm_dims[1] << "x" << rtm_dims[0];
                std::cerr << ", but the camera image has resolution " << image_dims[2] << "x" << image_dims[1] << "." << std::endl;
                std::exit(1);
            }
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}


std::pair<size_t, size_t> get_total_rtm_size(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files){
    // returns size (npixel, nvoxel) of total RTM

    // getting the total number of pixels by iterating over the views
    size_t npixel = 0;
    try {
        for (const auto& p : sorted_matrix_files) {
            const H5::H5File file(p.second[0], H5F_ACC_RDONLY);

            size_t npix;
            file.openGroup("rtm")
                .openAttribute("npixel")
                .read(H5::PredType::NATIVE_HSIZE, &npix);
            npixel += npix;
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    // getting the total number of voxels by iterating over the segments of the first view
    size_t nvoxel = 0;
    try {
        for (const auto& filename : sorted_matrix_files.begin()->second) {
            const H5::H5File file(filename, H5F_ACC_RDONLY);

            size_t nvox;
            file.openGroup("rtm")
                .openAttribute("nvoxel")
                .read(H5::PredType::NATIVE_HSIZE, &nvox);
            nvoxel += nvox;
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return std::pair(npixel, nvoxel);
}
