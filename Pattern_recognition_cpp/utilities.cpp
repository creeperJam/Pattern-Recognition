#include "utilities.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

TimeSeriesSoA LoadAndPrepareData(const std::string &filepath) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Cannot open " << filepath << std::endl;
    }

    TimeSeriesSoA data(2'000'000);

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ';');

        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                data.timeseries_c1.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                data.timeseries_c2.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                data.timeseries_c3.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                data.timeseries_c4.push_back(std::stof(token));
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                data.timeseries_c5.push_back(std::stof(token));
            }
        }
    }
    file.close();
    return data;
}

double Average(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

bool SameResults(const SADResults &a, const SADResults &b, float atol) {
    if (a.best_indices != b.best_indices) {
        return false;
    }

    if (a.best_sads.size() != b.best_sads.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.best_sads.size(); ++i) {
        if (std::fabs(a.best_sads[i] - b.best_sads[i]) > atol) {
            std::cerr << "SAD value mismatch at index " << i << ": " << a.best_sads[i] << " vs " << b.best_sads[i] << " (difference: " << std::fabs(a.best_sads[i] - b.best_sads[i])
                    << ", atol: " << atol << ")\n";
            return false;
        }
    }

    return true;
}

/**
 * Generates a random float in the range [-15.0, 15.0] using the provided random engine. This function is designed to be portable across different platforms and compilers, ensuring consistent random number generation.
 * @param eng The random engine to use.
 * @return A random float in the range [-15.0, 15.0].
 */
float PortableUniformDistribution(std::mt19937 &eng) {
    const uint32_t raw_value = eng();

    constexpr uint32_t max_value = std::mt19937::max(); // For mt19937 is 4294967295
    const float normalized = static_cast<float>(raw_value) / static_cast<float>(max_value);

    return normalized * 30.0f - 15.0f;
}

bool SaveStats(const std::string &algo, int query_size, int query_count, int num_threads, RunStats &stats, int RUNS) {
    try {
        std::ostringstream ss;
        std::filesystem::path output(std::string(PROJECT_SOURCE_DIR) + "/output");
        if (!std::filesystem::exists(output)) {
            std::filesystem::create_directory(output);
        }
        output /= "benchmark_cpp.csv";
        std::ifstream file_read(output);

        if (file_read.is_open()) {
            if (file_read.peek() == std::ifstream::traits_type::eof())
                ss << "QueryLength,QueryCount,Algo,NumThreads,WallMinMs,WallMaxMs,WallAvgMs,WallStdMs,CpuMinMs,CpuMaxMs,CpuAvgMs,CpuStdMs\n";
            file_read.close();
        } else {
            ss << "QueryLength,QueryCount,Algo,NumThreads,WallMinMs,WallMaxMs,WallAvgMs,WallStdMs,CpuMinMs,CpuMaxMs,CpuAvgMs,CpuStdMs\n";
        }

        // wall clock metric
        const double wall_min = *std::ranges::min_element(stats.wall_ms);
        const double wall_max = *std::ranges::max_element(stats.wall_ms);
        double wall_sum = 0.0;
        for (const double v: stats.wall_ms) wall_sum += v;
        const double wall_avg = wall_sum / RUNS;
        double wall_var = 0.0;
        for (const double v: stats.wall_ms) wall_var += (v - wall_avg) * (v - wall_avg);
        const double wall_std = sqrt(wall_var / (RUNS - 1));

        stats.wall_avg = wall_avg;
        stats.wall_max = wall_max;
        stats.wall_min = wall_min;
        stats.wall_std = wall_std;

        // cpu time metric
        const double cpu_min = *std::ranges::min_element(stats.cpu_ms);
        const double cpu_max = *std::ranges::max_element(stats.cpu_ms);
        double cpu_sum = 0.0;
        for (const double v: stats.cpu_ms) cpu_sum += v;
        const double cpu_avg = cpu_sum / RUNS;
        double cpu_var = 0.0;
        for (const double v: stats.cpu_ms) cpu_var += (v - cpu_avg) * (v - cpu_avg);
        const double cpu_std = sqrt(cpu_var / (RUNS - 1));

        stats.cpu_avg = cpu_avg;
        stats.cpu_max = cpu_max;
        stats.cpu_min = cpu_min;
        stats.cpu_std = cpu_std;

        // write to file
        ss << query_size << "," << query_count << "," << algo << "," << num_threads << "," << wall_min << "," << wall_max << "," << wall_avg << "," << wall_std << "," << cpu_min
                << "," << cpu_max << "," << cpu_avg << "," << cpu_std << "\n";

        std::ofstream file_write = std::ofstream(output, std::ios::app);
        file_write << ss.str();
        file_write.close();
    } catch (...) {
        return false;
    }
    return true;
}

bool SaveResults(std::vector<SADResults> &results, const int query_count, const int query_length, const int num_threads) {
    try {
        std::ostringstream ss;
        std::filesystem::path output(std::string(PROJECT_SOURCE_DIR) + "/output");
        if (!std::filesystem::exists(output)) {
            std::filesystem::create_directory(output);
        }
        output /= "results_cpp.csv";
        std::ifstream file_read(output);

        if (file_read.is_open()) {
            if (file_read.peek() == std::ifstream::traits_type::eof()) ss << "ThreadNum,QueryNum,QueryLength,ChannelNum,MinIndex,SAD\n";
            file_read.close();
        } else {
            ss << "ThreadNum,QueryNum,QueryLength,ChannelNum,MinIndex,SAD\n";
        }

        for (int qc = 0; qc < query_count; ++qc) {
            for (int column = 0; column < 5; ++column) {
                ss << num_threads << "," << qc + 1 << "," << query_length << "," << column + 1 << "," << results[qc].best_indices[column] << "," << results[
                    qc].best_sads[column] << "\n";
            }
        }
        std::ofstream file_write = std::ofstream(output, std::ios::app);
        file_write << ss.str();
        file_write.close();
    } catch (...) {
        return false;
    }

    return true;
}
