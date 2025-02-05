// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <argparse/argparse.hpp>


struct Config {
	std::vector<std::string> input_files;
	std::string output_file;
	std::string time_range;
	std::string laplacian_file;
	std::string raytransfer_name;
	double wavelength_threshold;
	double ray_density_threshold;
	double ray_length_threshold;
	double conv_tolerance;
	double beta_laplace;
	double relaxation;
	int max_iterations;
	int max_cached_frames;
	int max_cached_solutions;
	bool logarithmic;
	bool no_guess;
	bool use_cpu;
	bool parallel_read;
};

Config parse_arguments(int argc, char *argv[]);

std::vector<std::array<double, 4>> parse_time_intervals(std::string time_string);
