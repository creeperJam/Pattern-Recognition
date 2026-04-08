//
// Created by albi0 on 08/04/2026.
//

#ifndef PATTERN_RECOGNITION_CU_COMMON_H
#define PATTERN_RECOGNITION_CU_COMMON_H

// My RTX 4060 has 64 KiB of constant memory, meaning that query length and number have to respect the following limit:
// QUERY_LENGTH * NUM_QUERIES * sizeof(float) * COLUMNS_NUMBER(5 in this case) <= 65536, so:
// QUERY_LENGTH * NUM_QUERIES <= 3276 (approx.)
constexpr int QUERY_LENGTH = 819;   // MAX: 3276 - Limited by constant and shared memory size (99 KiB shared = 101376 B => (4812+256) * 5 * 4 = 101376, with 4813 we'd go over this limit). ACTUAL LIMIT IS 3276 DUE TO CONSTANT MEMORY SIZE
constexpr int NUM_QUERIES = 1;      // MAX: 3276 / QUERY_LENGTH
// QUERIES TO TEST:
// - LENGTHS: {128, 256, 512, 819}
// - COUNT  : {1, 2, 3, 4}
constexpr int TOTAL_QUERY_ELEMENTS = QUERY_LENGTH * NUM_QUERIES;
constexpr int NUM_RUNS = 100;
constexpr int NUM_WARMUP = 10;
constexpr int CHANNEL_COUNT = 5;
constexpr int SEED_GENERATOR = 42;

#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <vector>
#include <limits>

#include <filesystem>

struct DataSoA {
    std::vector<float> c1;
    std::vector<float> c2;
    std::vector<float> c3;
    std::vector<float> c4;
    std::vector<float> c5;

    explicit DataSoA(int size) {
        c1.reserve(size);
        c2.reserve(size);
        c3.reserve(size);
        c4.reserve(size);
        c5.reserve(size);
    }
};

float PortableUniformGenerator(std::mt19937& eng);
bool ReadFile(const std::string& filepath, DataSoA& loaded_data);
bool GenerateQueries(
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c1, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c2, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c3,
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c4, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c5, const DataSoA& loaded_data, int total_elements);
bool PrintAndSaveResults(const std::array<int, NUM_QUERIES * CHANNEL_COUNT>& best_indices, const std::array<float, NUM_QUERIES * CHANNEL_COUNT>& min_sads);
bool SaveStats(const std::array<double, NUM_RUNS>& wall_times);

#endif //PATTERN_RECOGNITION_CU_COMMON_H