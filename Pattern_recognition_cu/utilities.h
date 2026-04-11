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

// The value of these two constants were chosen based on the performances measured after some tests.
// We tested the following size respectively:
// - {64, 128, 256, 512, 1024} (1024 is the maximum allowed)
// - {NONE, 4, 8, 16, 32}
constexpr int BLOCK_SIZE = 512;
constexpr int UNROLL_SIZE = 8;

#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <vector>
#include <limits>
#include <filesystem>

struct TimeSeriesSoA {
    std::vector<float> timeseries_c1;
    std::vector<float> timeseries_c2;
    std::vector<float> timeseries_c3;
    std::vector<float> timeseries_c4;
    std::vector<float> timeseries_c5;

    explicit TimeSeriesSoA(int size) {
        timeseries_c1.reserve(size);
        timeseries_c2.reserve(size);
        timeseries_c3.reserve(size);
        timeseries_c4.reserve(size);
        timeseries_c5.reserve(size);
    }
};

/**
 * @brief Generates a pseudo-random number from -15 to +15 based on a chosen seed.
 *
 * @details This was needed since all float RNG can generate different values on different environments. This makes it
 * harder to test the program on different machines or even compilator, since results vary.
 * To overcome this we use a standardized number generator like std::mt19937 to get an integer and with simple math get
 * always the same numbers in a range no matter the environment.
 *
 * @param eng The engine used to generate the numbers passed by reference
 * @returns The generated value of type float
 */
float PortableUniformGenerator(std::mt19937& eng);
/**
 * @brief Reads all the time series data from a specified file
 *
 * @param filepath The path of the .csv file to read data from
 * @param loaded_data The data structured where the data will be put (passed by reference)
 * @return A boolean value that indicates whether everything went well or there were any errors during the process.
 */
bool ReadFile(const std::string& filepath, TimeSeriesSoA& loaded_data);
/**
 * @brief Generates all the queries based on the input data for testing purposes
 *
 * @details While having random queries is a way to test, checking whether the results are correct or not can be hard.
 * To fix this problem we extract the exact values saved in the structure with the time series values and add a random
 * noise to each using the PortableUniformGenerator, getting values that are guaranteed to be close to the actual data
 * without having to check for averages, minimum or maximums.
 *
 * @param all_queries_c1 Array containing all the queries for the first channel
 * @param all_queries_c2 Array containing all the queries for the second channel
 * @param all_queries_c3 Array containing all the queries for the third channel
 * @param all_queries_c4 Array containing all the queries for the fourth channel
 * @param all_queries_c5 Array containing all the queries for the fifth channel
 * @param loaded_data Structure containing all the data of the time series
 * @param total_elements Amount of data extracted from the time series
 * @return A boolean value that indicates whether everything went well or there were any errors during the process.
 */
bool GenerateQueries(
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c1, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c2, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c3,
    std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c4, std::array<float, TOTAL_QUERY_ELEMENTS>& all_queries_c5, const TimeSeriesSoA& loaded_data, int total_elements);
/**
 * @brief Prints to console and saves to csv file the best SADs found for each query and the corresponding index
 *
 * @param best_indices Array containing the indices of the best SADs values of each query for each channel
 * @param min_sads Array containing the lowest SADs values of each query for each channel
 * @return A boolean value that indicates whether everything went well or there were any errors during the process.
 */
bool PrintAndSaveResults(const std::array<int, NUM_QUERIES * CHANNEL_COUNT>& best_indices, const std::array<float, NUM_QUERIES * CHANNEL_COUNT>& min_sads);
/**
 * @brief Calculates the mean, min, max and std times for both wall and gpu time and saves them to a csv file
 *
 * @param wall_times Array containing the wall time of each run
 * @param gpu_times Array containing the GPU time of each run
 * @return A boolean value that indicates whether everything went well or there were any errors during the process.
 */
bool SaveStats(const std::array<double, NUM_RUNS>& wall_times, const std::array<double, NUM_RUNS>& gpu_times);

#endif //PATTERN_RECOGNITION_CU_COMMON_H