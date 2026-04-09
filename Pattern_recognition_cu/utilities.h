//
// Created by albi0 on 08/04/2026.
//

#ifndef PATTERN_RECOGNITION_CU_COMMON_H
#define PATTERN_RECOGNITION_CU_COMMON_H

// My RTX 4060 has 64 KiB of constant memory, meaning that query length and number have to respect the following limit:
// QUERY_LENGTH * NUM_QUERIES * sizeof(float) * COLUMNS_NUMBER(5 in this case) <= 65536, so:
// QUERY_LENGTH * NUM_QUERIES <= 3276 (approx.)
constexpr int QUERY_LENGTH = 819;   // MAX: 3276 - Limited by constant and shared memory size (99 KiB shared = 101376 B => (4812+256) * 5 * 4 = 101376, with 4813 we'd go over this limit). ACTUAL LIMIT IS 3276 DUE TO CONSTANT MEMORY SIZE
constexpr int NUM_QUERIES = 4;      // MAX: 3276 / QUERY_LENGTH
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

// Returns a random generated float between 0 and 15.
// This is not the same as 'std::uniform_real_distribution' because the generated
// numbers are always the same, even on different environments, if the seed is the same.
float PortableUniformGenerator(std::mt19937& eng);
// Returns a bool that either confirms that the reading went well or there was an error.
// Reads all the contents of a .csv file and saves each column into the loaded_data structure
bool ReadFile(const std::string& filepath, DataSoA& loaded_data);
// Returns a bool that either confirms that the generation went well or there was an error.
// Extracts a number of elements from the loaded data based on the amount of data and the number of queries.
// After extracting the indexes it extracts the value, adds a noise using the random float generator and saves it
// to the query array
bool GenerateQueries(
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c1, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c2, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c3,
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c4, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c5, const DataSoA& loaded_data, int total_elements);
// Returns a bool that either confirms that the print and save went well or there was an error.
// It takes as input the best indices and the corresponding SADs and prints the to console first and saves the same
// output to a .csv file
bool PrintAndSaveResults(const std::array<int, NUM_QUERIES * CHANNEL_COUNT>& best_indices, const std::array<float, NUM_QUERIES * CHANNEL_COUNT>& min_sads);
// Returns a bool that either confirms that the time log save went well or there was an error.
// It takes as input all the wall times, calculates the mean, min, max and std times and save all of them to a .csv file.
bool SaveStats(const std::array<double, NUM_RUNS>& wall_times, const std::array<double, NUM_RUNS>& gpu_times);

#endif //PATTERN_RECOGNITION_CU_COMMON_H