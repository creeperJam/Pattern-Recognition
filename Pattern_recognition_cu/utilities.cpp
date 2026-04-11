//
// Created by albi0 on 08/04/2026.
//

#include "utilities.h"

bool ReadFile(const std::string& filepath, TimeSeriesSoA& loaded_data) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        return false;
    }
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ';');

        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                loaded_data.timeseries_c1.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                loaded_data.timeseries_c2.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                loaded_data.timeseries_c3.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                loaded_data.timeseries_c4.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                loaded_data.timeseries_c5.push_back(std::stof(token));
            }
        }
    }
    file.close();

    return true;
}

bool GenerateQueries(
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c1, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c2, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c3,
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c4, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c5, const TimeSeriesSoA& loaded_data, const int total_elements)
{
    try {
        std::mt19937 generator{SEED_GENERATOR};

        std::array<std::size_t, NUM_QUERIES> ground_truth_indices;
        for (int q = 0; q < NUM_QUERIES; ++q) {
            std::size_t safe_idx = (q + 1) * (total_elements / (NUM_QUERIES + 2));

            if (safe_idx + QUERY_LENGTH > total_elements) {
                safe_idx = 0;
            }
            ground_truth_indices[q] = safe_idx;
            std::cout << "  -> Query " << q << " extracted starting at index " << safe_idx << "\n";
        }

        for (int q = 0; q < NUM_QUERIES; ++q) {
            std::size_t base_idx = ground_truth_indices[q];
            std::size_t offset = static_cast<std::size_t>(q) * static_cast<std::size_t>(QUERY_LENGTH);

            for (int i = 0; i < QUERY_LENGTH; ++i) {
                all_queries_c1[offset + i] = loaded_data.timeseries_c1[base_idx + i] + PortableUniformGenerator(generator);
                all_queries_c2[offset + i] = loaded_data.timeseries_c2[base_idx + i] + PortableUniformGenerator(generator);
                all_queries_c3[offset + i] = loaded_data.timeseries_c3[base_idx + i] + PortableUniformGenerator(generator);
                all_queries_c4[offset + i] = loaded_data.timeseries_c4[base_idx + i] + PortableUniformGenerator(generator);
                all_queries_c5[offset + i] = loaded_data.timeseries_c5[base_idx + i] + PortableUniformGenerator(generator);
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool PrintAndSaveResults(const std::array<int, NUM_QUERIES * CHANNEL_COUNT>& best_indices, const std::array<float, NUM_QUERIES * CHANNEL_COUNT>& min_sads) {
    try {
        std::ostringstream ss;
        std::filesystem::path output(std::string(PROJECT_SOURCE_DIR) + "/output");
        if (!std::filesystem::exists(output)) {
            std::filesystem::create_directory(output);
        }
        output /= "output_cuda.csv";
        ss << "QueryNum,QueryLength,ChannelNum,MinIndex,SAD\n";

        std::cout << "\n========================= SEARCH RESULTS =========================\n";

        for (int query = 0; query < NUM_QUERIES; ++query) {
            std::cout << "[ QUERY " << query + 1 << " ]\n";
            for (int channel = 0; channel < 5; ++channel) {
                const int query_channel = query * 5 + channel;
                std::cout << "  - Serie C" << channel + 1
                          << " -> indice: " << best_indices[query_channel]
                          << ", SAD: " << min_sads[query_channel] << "\n";
                ss << query+1 << "," << QUERY_LENGTH << "," << channel+1 << "," << best_indices[query_channel] << "," << min_sads[query_channel] << "\n";

            }
            std::cout << "------------------------------------------------------------------\n";
        }
        std::ofstream file_write = std::ofstream(output);
        file_write << ss.str();
        file_write.close();

        std::cout << "\nThe results were also saved at the following path: " << std::string(PROJECT_SOURCE_DIR) + "/output.csv" << "\n";
    } catch (...) {
        return false;
    }

    return true;
}

bool SaveStats(const std::array<double, NUM_RUNS>& wall_times, const std::array<double, NUM_RUNS>& gpu_times) {
    std::filesystem::path output(std::string(PROJECT_SOURCE_DIR) + "/output");
    if (!std::filesystem::exists(output)) {
        std::filesystem::create_directory(output);
    }

    output = output / ("time_results.csv");

    bool empty_file = true;
    std::ifstream file_read(output);
    if (file_read.is_open()) {
        empty_file = (file_read.peek() == std::ifstream::traits_type::eof());
        file_read.close();
    }

    std::ofstream file_write(output, std::ios::app);

    if (!file_write.is_open()) {
        std::cerr << "ERROR: unable to open file " << output << "\n";
        return false;
    }
    if (empty_file) {
        file_write << "BlockSize,UnrollSize,NumQueries,QueryLength,WallMean_ms,WalLMax_ms,WallMin_ms,WallStd_ms,GpuMean_ms,GpuMax_ms,GpuMin_ms,GpuStd_ms\n";
    }

    const double wall_max = *std::ranges::max_element(wall_times);
    const double wall_min = *std::ranges::min_element(wall_times);
    double wall_std = 0.0f;
    double wall_mean = 0.0f;
    for (auto& wall_time : wall_times) {
        wall_mean += wall_time;
    }
    wall_mean /= NUM_RUNS;
    for (auto& wall_time : wall_times) {
        wall_std += (wall_time - wall_mean) * (wall_time - wall_mean);
    }
    wall_std /= NUM_RUNS - 1;
    wall_std = std::sqrt(wall_std);

    const double gpu_max = *std::ranges::max_element(gpu_times);
    const double gpu_min = *std::ranges::min_element(gpu_times);
    double gpu_std = 0.0f;
    double gpu_mean = 0.0f;
    for (auto& gpu_time : gpu_times) {
        gpu_mean += gpu_time;
    }
    gpu_mean /= NUM_RUNS;
    for (auto& gpu_time : gpu_times) {
        gpu_std += (gpu_time - gpu_mean) * (gpu_time - gpu_mean);
    }
    gpu_std /= NUM_RUNS - 1;
    gpu_std = std::sqrt(gpu_std);

    file_write << BLOCK_SIZE << "," << UNROLL_SIZE << "," << NUM_QUERIES << "," << QUERY_LENGTH << "," << wall_mean << "," << wall_max << "," << wall_min << "," << wall_std << "," << gpu_mean << "," << gpu_max << "," << gpu_min << "," << gpu_std << "\n";

    file_write.close();
    return true;
}

float PortableUniformGenerator(std::mt19937& eng) {
    uint32_t raw_value = eng();

    uint32_t max_value = std::mt19937::max();
    float normalized = static_cast<float>(raw_value) / static_cast<float>(max_value);

    return (normalized * 30.0f) - 15.0f;
}