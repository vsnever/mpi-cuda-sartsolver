
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <H5Cpp.h>


template <typename TYPE, size_t SIZE>
int check_group_attribute_consistency(const std::vector<std::string>& files,
                                      const std::string& group_name,
                                      const std::string (&attr_names)[SIZE],
                                      const H5::DataType& h5dtype) {
    // checks if the attributes of the group are the same for all given files

    std::array<TYPE, SIZE> ref_attr_values;

    try {
        // getting reference values
        H5::H5File file(files.front(), H5F_ACC_RDONLY);
        auto group = file.openGroup(group_name);
        for (size_t i = 0; i<SIZE; i++) {
            const auto attr = group.openAttribute(attr_names[i]);
            attr.read(h5dtype, &ref_attr_values[i]);
        }

        // checking if the values in other files are equal to the reference ones.
        for (auto iter = files.begin() + 1; iter < files.end(); ++iter) {
            file = H5::H5File(*iter, H5F_ACC_RDONLY);
            for (size_t i = 0; i<SIZE; i++) {
                group = file.openGroup(group_name);
                TYPE attr_value;
                const auto attr = group.openAttribute(attr_names[i]);
                attr.read(h5dtype, &attr_value);

                if (attr_value != ref_attr_values[i]) {
                    std::cerr << "Different " << attr_names[i] << " values in " << *iter;
                    std::cerr << " (" << attr_value << ") and in ";
                    std::cerr << files.front() << " (" << ref_attr_values[i] << ")." << std::endl;
                    std::exit(1);
                }
            }
            
        }

    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }

    return 0;
}


int categorize_input_files(const std::vector<std::string>& input_files,
                           std::vector<std::string>& files,
                           std::vector<std::string>& image_files);

std::map<std::string, std::vector<std::string>> sort_rtm_files(const std::vector<std::string>& files);

int check_rtm_frame_consistency(const std::map<std::string, std::vector<std::string>>& sorted_files);

int check_rtm_voxel_consistency(const std::map<std::string, std::vector<std::string>>& sorted_files);

std::map<std::string, std::string> sort_image_files(const std::vector<std::string>& files);

int check_rtm_image_consistency(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files,
                                const std::map<std::string, std::string>& sorted_image_files,
                                const std::string& rtm_name,
                                const double wvl_threshold=0);

std::map<std::string, std::vector<int>> read_rtm_frame_masks(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files);

std::pair<size_t, size_t> get_total_rtm_size(const std::map<std::string, std::vector<std::string>>& sorted_matrix_files);
