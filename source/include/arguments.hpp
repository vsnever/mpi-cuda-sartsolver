// Copyright (c) 2022 - 2023, Project Center ITER, 123060, ul. Raspletina, 11 bld. 2, Moscow, Russia
// Author: Vladislav Neverov, neverov_vs@nrcki.ru (NRC "Kurchatov Institute")
// 
// All rights reserved.
// 
// Redistribution in source and binary form, with or without modifications,
// is prohibited without permission of the copyright holder.

#pragma once

#include <argparse/argparse.hpp>

argparse::ArgumentParser parse_arguments(int argc, char *argv[]);

std::vector<std::array<double, 4>> parse_time_intervals(std::string time_string);
