#include "utilities.h"

#include <algorithm>
#include <cmath>
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
                try { data.historical_data_c1.push_back(std::stof(token)); } catch (...) {
                }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c2.push_back(std::stof(token)); } catch (...) {
                }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c3.push_back(std::stof(token)); } catch (...) {
                }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c4.push_back(std::stof(token)); } catch (...) {
                }
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { data.historical_data_c5.push_back(std::stof(token)); } catch (...) {
                }
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
            return false;
        }
    }

    return true;
}

float PortableUniformDistribution(std::mt19937 &eng) {
    const uint32_t raw_value = eng();

    constexpr uint32_t max_value = std::mt19937::max(); // For mt19937 is 4294967295
    const float normalized = static_cast<float>(raw_value) / static_cast<float>(max_value);

    return normalized * 15.0f;
}

void SaveStats(std::ofstream &csv_file, const std::string &algo, int query_size, int query_count, int num_threads, RunStats &stats, int RUNS){
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
    csv_file << query_size << "," << query_count << "," << algo << "," << num_threads
            << "," << wall_min << "," << wall_max << "," << wall_avg << "," << wall_std << "," << cpu_min
            << "," << cpu_max << "," << cpu_avg << "," << cpu_std << "\n";

    csv_file.flush();
}

