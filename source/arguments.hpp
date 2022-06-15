
#pragma once

#include <argparse/argparse.hpp>

argparse::ArgumentParser parse_arguments(int argc, char *argv[]);

std::vector<std::array<double, 4>> parse_time_intervals(std::string time_string);
