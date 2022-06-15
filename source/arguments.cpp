
#include "arguments.hpp"

std::vector<std::array<double, 4>> parse_time_intervals(std::string time_string) {
    std::vector<std::array<double, 4>> time_intervals;

    if (time_string.empty()) {
        time_intervals.push_back({0, std::numeric_limits<double>::infinity(), 0, 0});
        return time_intervals;
    }

    const std::string ranges_delimiter(",");
    const std::string interval_delimiter(":");

    size_t rpos = 0;
    while (rpos < time_string.size() - 1) {  // '- 1' because trailing ',' is allowed but not processed
        const size_t new_rpos = time_string.find(ranges_delimiter, rpos);
        const auto interval_string = time_string.substr(rpos, new_rpos - rpos);
        rpos = (new_rpos==std::string::npos) ? new_rpos : new_rpos + 1;

        std::vector<std::string> interval;
        size_t ipos = 0;
        while (ipos < interval_string.size()) {
            const size_t new_ipos = interval_string.find(interval_delimiter, ipos);
            interval.push_back(interval_string.substr(ipos, new_ipos - ipos));
            ipos = (new_ipos==std::string::npos) ? new_ipos : new_ipos + 1;
        }

        if (interval.size() < 2) {
            std::cerr << "Unable to recognize a time interval in " << interval_string << "." << std::endl;
            std::exit(1);
        }
        if (interval.size() > 4) {
            std::cerr << "Too many values in a time interval: " << interval_string << "." << std::endl;
            std::exit(1);
        }

        try {
            const auto start = std::stod(interval[0]);
            const auto end = std::stod(interval[1]);
            const double step = (interval.size() > 2) ? std::stod(interval[2]) : 0;
            const double threshold = (interval.size() > 3) ? std::stod(interval[3]) : 0;

            if (start < 0) {
                std::cerr << "Time limits must be positive." << std::endl;
                std::exit(1);
            }
            if (end <= start) {
                std::cerr << "The upper limit of the time interval must be higher than the lower one." << std::endl;
                std::exit(1);
            }
            if (step > (end - start)) {
                std::cerr << "Time step must be less or equal to the time interval." << std::endl;
                std::exit(1);
            }
            if (threshold > step) {
                std::cerr << "Synchronization threshold must be less or equal to the time step." << std::endl;
                std::exit(1);
            }
            time_intervals.push_back({start, end, step, threshold});
        }
        catch(const std::invalid_argument& err)
        {
            std::cerr << err.what();
            std::cerr << "\nUnable to convert " << interval_string << " to the time interval." << std::endl;
            std::exit(1);
        }
    }

    return time_intervals;
}


argparse::ArgumentParser parse_arguments(int argc, char *argv[]) {

    argparse::ArgumentParser program("Impurity flux reconstruction for ITER: emissivity");

    program.add_argument("-o", "--output_file")
        .help("Filename to save the solution.")
        .default_value(std::string("solution.h5"));

    program.add_argument("-t", "--time_range")
        .help("Time intervals in s to process in a form: start:stop:(step):(synch_threshold),"
              "e.g. '20.5:40.1, 45.2:51:15:0.05'.\n"
              "The step and the synchronization threshold are optional.")
        .default_value(std::string());

    program.add_argument("-w", "--wavelength_threshold")
        .help("An RTM is considered valid if its wavelength is within this threshold of the image wavelength (in nm).")
        .default_value(50.)
        .scan<'g', double>();

    program.add_argument("-d", "--ray_density_threshold")
        .help("Voxels with ray density lesser than this threshold are ignored.")
        .default_value(1.e-6)
        .scan<'g', double>();

    program.add_argument("-r", "--ray_length_threshold")
        .help("Pixels with ray length lesser than this threshold are ignored.")
        .default_value(1.e-6)
        .scan<'g', double>();

    program.add_argument("-m", "--max_iterations")
        .help("Maximum number of SART iterations.")
        .default_value(2000)
        .scan<'i', int>();

    program.add_argument("-c", "--conv_tolerance")
        .help("SART convolution relative tolerance.")
        .default_value(1.e-5)
        .scan<'g', double>();

    program.add_argument("-l", "--laplacian_file")
        .help("File with laplacian regularization matrix.")
        .default_value(std::string());

    program.add_argument("-b", "--beta_laplace")
        .help("Weight of the regularization factor.")
        .default_value(2.e-2)
        .scan<'g', double>();

    program.add_argument("-R", "--relaxation")
        .help("Relaxation parameter.")
        .default_value(1.)
        .scan<'g', double>();

    program.add_argument("-n", "--raytransfer_name")
        .help("Ray transfer matrix dataset name.")
        .default_value(std::string("with_reflections"));

    program.add_argument("-L", "--logarithmic")
        .help("Use logarithmic SART solver.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max_cached_frames")
        .help("Maximum number of cached image frames.")
        .default_value(100)
        .scan<'i', int>();

    program.add_argument("--no_guess")
        .help("Do not use solution found on previous time moment as initial guess for the next one.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--use_cpu")
        .help("Perform all calculations on CPUs.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("input_files")
        .help("List of ray transfer matrix and camera image hdf5 files.")
        .remaining();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto ray_density_threshold = program.get<double>("--ray_density_threshold");
    if (ray_density_threshold < 0) {
        std::cerr << "Argument ray_density_threshold must be >= 0, " << ray_density_threshold << " given." << std::endl;
        std::exit(1);
    }

    auto ray_length_threshold = program.get<double>("--ray_length_threshold");
    if (ray_length_threshold < 0) {
        std::cerr << "Argument ray_length_threshold must be >= 0, " << ray_length_threshold << " given." << std::endl;
        std::exit(1);
    }

    auto max_iterations = program.get<int>("--max_iterations");
    if (max_iterations < 1) {
        std::cerr << "Argument max_iterations must be >= 1, " << max_iterations << " given." << std::endl;
        std::exit(1);
    }

    auto conv_tolerance = program.get<double>("--conv_tolerance");
    if (conv_tolerance <= 0) {
        std::cerr << "Argument conv_tolerance must be > 0, " << conv_tolerance << " given." << std::endl;
        std::exit(1);
    }

    auto relaxation = program.get<double>("--relaxation");
    if ((relaxation <= 0) || (relaxation > 1.)) {
        std::cerr << "Argument relaxation must be within (0, 1] interval," << relaxation << " given." << std::endl;
        std::exit(1);
    }

    auto beta_laplace = program.get<double>("--beta_laplace");
    if (beta_laplace < 0) {
        std::cerr << "Argument beta_laplace must be positive." << std::endl;
        std::exit(1);
    }

    auto max_cached_frames = program.get<int>("--max_cached_frames");
    if (max_cached_frames <= 0) {
        std::cerr << "Argument max_cached_frames must be positive." << std::endl;
        std::exit(1);
    }

    auto input_files = program.get<std::vector<std::string>>("input_files");
    if (input_files.size() < 2) {
        std::cerr << "At least two input file, one with RTM and one with image, are required, " << input_files.size() << " given." << std::endl;
        std::exit(1);
    }

    return program;
}
