#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel.h"
#include "sequential.h"
#include "utilities.h"

std::string filepath = std::string(PROJECT_SOURCE_DIR) + "/../realistic_data.csv";
std::string stats_path = std::string(PROJECT_SOURCE_DIR) + "/run_stats.csv";

constexpr int RUNS = 10;
constexpr int WARMUP_RUNS = 2;
constexpr int MAX_QUERIES_PER_TIME_SERIES = 4;
constexpr std::size_t NUM_SERIES = 5;
constexpr std::array QUERIES_LENGHTS = {128, 256, 512, 819};
constexpr std::array THREAD_COUNTS = {1, 2, 4, 8, 16};

int main() {
    try {
        std::cout << "Loading data from " << filepath << "...\n";
        TimeSeriesSoA loaded_data = LoadAndPrepareData(filepath);

        const std::size_t n_points = loaded_data.timeseries_c1.size();
        std::cout << "Loaded " << NUM_SERIES << " time series with " << n_points << " points each.\n";

        if (n_points == 0) {
            throw std::runtime_error("Loaded data is empty.");
        }

        // START OF QUERY GENERATION
        std::mt19937 generator{42};

        for (const int query_length: QUERIES_LENGHTS) {
            if (n_points < query_length) {
                std::cout << "Skipping query_length=" << query_length << " (series too short).\n";
                continue;
            }

            std::vector<float> all_queries_c1(query_length * MAX_QUERIES_PER_TIME_SERIES);
            std::vector<float> all_queries_c2(query_length * MAX_QUERIES_PER_TIME_SERIES);
            std::vector<float> all_queries_c3(query_length * MAX_QUERIES_PER_TIME_SERIES);
            std::vector<float> all_queries_c4(query_length * MAX_QUERIES_PER_TIME_SERIES);
            std::vector<float> all_queries_c5(query_length * MAX_QUERIES_PER_TIME_SERIES);

            std::vector<std::size_t> timeseries_real_indices(MAX_QUERIES_PER_TIME_SERIES);
            for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
                std::size_t safe_idx = (q + 1) * (n_points / (MAX_QUERIES_PER_TIME_SERIES + 2));
                if (safe_idx + query_length > n_points) {
                    safe_idx = 0;
                }
                timeseries_real_indices[q] = safe_idx;
            }

            for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
                const std::size_t base_idx = timeseries_real_indices[q];
                const std::size_t offset = q * query_length;

                for (int i = 0; i < query_length; ++i) {
                    all_queries_c1[offset + i] = loaded_data.timeseries_c1[base_idx + i] + PortableUniformDistribution(generator);
                    all_queries_c2[offset + i] = loaded_data.timeseries_c2[base_idx + i] + PortableUniformDistribution(generator);
                    all_queries_c3[offset + i] = loaded_data.timeseries_c3[base_idx + i] + PortableUniformDistribution(generator);
                    all_queries_c4[offset + i] = loaded_data.timeseries_c4[base_idx + i] + PortableUniformDistribution(generator);
                    all_queries_c5[offset + i] = loaded_data.timeseries_c5[base_idx + i] + PortableUniformDistribution(generator);
                }
            }
            // END OF QUERY GENERATION

            // lambda to extract queries for a given query index from the big query buffers
            auto get_query = [&](const int query_idx) {
                const std::size_t offset = query_idx * query_length;
                std::vector<std::vector<float> > queries(5);
                queries[0] = std::vector(all_queries_c1.begin() + static_cast<long long>(offset), all_queries_c1.begin() + static_cast<long long>(offset + query_length));
                queries[1] = std::vector(all_queries_c2.begin() + static_cast<long long>(offset), all_queries_c2.begin() + static_cast<long long>(offset + query_length));
                queries[2] = std::vector(all_queries_c3.begin() + static_cast<long long>(offset), all_queries_c3.begin() + static_cast<long long>(offset + query_length));
                queries[3] = std::vector(all_queries_c4.begin() + static_cast<long long>(offset), all_queries_c4.begin() + static_cast<long long>(offset + query_length));
                queries[4] = std::vector(all_queries_c5.begin() + static_cast<long long>(offset), all_queries_c5.begin() + static_cast<long long>(offset + query_length));
                return queries;
            };

            std::vector<std::vector<std::vector<float> > > queries_pool(MAX_QUERIES_PER_TIME_SERIES);
            for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
                queries_pool[q] = get_query(q);
            }

            for (int query_count = 1; query_count <= MAX_QUERIES_PER_TIME_SERIES; ++query_count) {
                std::cout << "Running sequential config: query_length=" << query_length << ", num_queries=" << query_count << " |";

                // 2 WARMUP RUNS
                for (int run = 0; run < WARMUP_RUNS; ++run) {
                    for (int q = 0; q < query_count; ++q) {
                        SequentialSADSearch(loaded_data, queries_pool[q]);
                    }
                }

                RunStats sequential_stats(RUNS);
                std::vector<SADResults> seq_results(query_count);
                for (int run_idx = 0; run_idx < RUNS; ++run_idx) {
                    const std::clock_t cpu_start = std::clock();
                    const auto wall_start = std::chrono::steady_clock::now();

                    for (int q = 0; q < query_count; ++q) {
                        seq_results[q] = SequentialSADSearch(loaded_data, queries_pool[q]);
                    }

                    const auto wall_end = std::chrono::steady_clock::now();
                    const std::clock_t cpu_end = std::clock();

                    sequential_stats.wall_ms[run_idx] = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
                    sequential_stats.cpu_ms[run_idx] = static_cast<double>(cpu_end - cpu_start) * 1000.0 / CLOCKS_PER_SEC;
                }
                SaveStats("sequential", query_length, query_count, 0, sequential_stats, RUNS);
                SaveResults(seq_results, query_count, query_length, 0);

                std::cout << " wall_min=" << sequential_stats.wall_min << " ms, wall_avg=" << sequential_stats.wall_avg << " ms, wall_max=" << sequential_stats.wall_max << " ms, wall_std=" << sequential_stats
                        .wall_std << " ms\n";

                for (const int thread_count: THREAD_COUNTS) {
                    // 2 WARMUP RUNS
                    std::cout << "Running parallel config: query_length=" << query_length << ", num_queries=" << query_count << ", threads=" << thread_count << " |";

                    for (int run = 0; run < WARMUP_RUNS; ++run) {
                        for (int q = 0; q < query_count; ++q) {
                            ParallelSADSearch(loaded_data, queries_pool[q], thread_count);
                        }
                    }

                    RunStats parallel_stats(RUNS);
                    std::vector<SADResults> par_results(query_count);
                    for (int run_idx = 0; run_idx < RUNS; ++run_idx) {
                        const std::clock_t cpu_start = std::clock();
                        const auto wall_start = std::chrono::steady_clock::now();

                        for (int q = 0; q < query_count; ++q) {
                            par_results[q] = ParallelSADSearch(loaded_data, queries_pool[q], thread_count);
                        }

                        const auto wall_end = std::chrono::steady_clock::now();
                        const std::clock_t cpu_end = std::clock();

                        parallel_stats.wall_ms[run_idx] = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
                        parallel_stats.cpu_ms[run_idx] = static_cast<double>(cpu_end - cpu_start) * 1000.0 / CLOCKS_PER_SEC;
                    }
                    SaveStats("parallel", query_length, query_count, thread_count, parallel_stats, RUNS);
                    SaveResults(par_results, query_count, query_length, thread_count);
                    std::cout << " wall_min=" << parallel_stats.wall_min << " ms, wall_avg=" << parallel_stats.wall_avg << " ms, wall_max=" << parallel_stats.wall_max << " ms, wall_std=" <<
                            parallel_stats.wall_std << " ms\n";

                    //correctness check
                    for (int q = 0; q < query_count; ++q) {
                        if (!SameResults(seq_results[q], par_results[q], 1e-3f)) {
                            std::cerr << "Error: mismatch for query " << q << " in config (query_length=" << query_length << ", num_queries=" << query_count << ", threads=" <<
                                    thread_count << ")\n";
                            return 1;
                        }
                    }
                }
                std::cout << "------------------------------------------------------------------------------------------\n";
            }
        }

        std::cout << "Benchmark completed. Stats saved to " << stats_path << "\n";

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
