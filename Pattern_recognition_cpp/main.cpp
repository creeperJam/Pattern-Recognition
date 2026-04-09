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

int main() {
  try {
    std::string filepath = std::string(PROJECT_SOURCE_DIR) + "/../realistic_data.csv";
    std::string stats_path = std::string(PROJECT_SOURCE_DIR) + "/run_stats.csv";

    constexpr int RUNS = 10;
    constexpr int MAX_QUERIES_PER_TIME_SERIES = 4;
    constexpr std::size_t n_series = 5;
    const std::array query_lengths = {128, 256, 512, 819};
    const std::array thread_counts = {1, 2, 4, 8, 16};

    std::cout << "Loading data from " << filepath << "...\n";
    TimeSeriesSoA loaded_data = LoadAndPrepareData(filepath);

    const std::size_t n_points = loaded_data.historical_data_c1.size();
    std::cout << "Loaded " << n_series << " time series with " << n_points << " points each.\n";

    if (n_points == 0) {
      throw std::runtime_error("Loaded data is empty.");
    }

    std::ofstream csv_file(stats_path);
    if (!csv_file.is_open()) {
      throw std::runtime_error("Cannot open output stats file: " + stats_path);
    }

    csv_file << "query_length,num_queries,algo,num_threads,wall_min_ms,wall_max_ms,wall_avg_ms,wall_std_ms,cpu_min_ms,cpu_max_ms,cpu_avg_ms,cpu_std_ms\n";

    std::mt19937 generator{42};

    for (const int query_length : query_lengths) {
      if (n_points < static_cast<std::size_t>(query_length)) {
        std::cout << "Skipping query_length=" << query_length << " (series too short).\n";
        continue;
      }

      // START OF QUERY GENERATION
      std::vector<float> all_queries_c1(static_cast<std::size_t>(query_length) * MAX_QUERIES_PER_TIME_SERIES);
      std::vector<float> all_queries_c2(static_cast<std::size_t>(query_length) * MAX_QUERIES_PER_TIME_SERIES);
      std::vector<float> all_queries_c3(static_cast<std::size_t>(query_length) * MAX_QUERIES_PER_TIME_SERIES);
      std::vector<float> all_queries_c4(static_cast<std::size_t>(query_length) * MAX_QUERIES_PER_TIME_SERIES);
      std::vector<float> all_queries_c5(static_cast<std::size_t>(query_length) * MAX_QUERIES_PER_TIME_SERIES);

      std::vector<std::size_t> ground_truth_indices(MAX_QUERIES_PER_TIME_SERIES);
      for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
        std::size_t safe_idx = static_cast<std::size_t>(q + 1) * (n_points / static_cast<std::size_t>(MAX_QUERIES_PER_TIME_SERIES + 2));
        if (safe_idx + static_cast<std::size_t>(query_length) > n_points) {
          safe_idx = 0;
        }
        ground_truth_indices[q] = safe_idx;
      }

      for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
        const std::size_t base_idx = ground_truth_indices[q];
        const std::size_t offset = static_cast<std::size_t>(q) * static_cast<std::size_t>(query_length);

        for (int i = 0; i < query_length; ++i) {
          all_queries_c1[offset + static_cast<std::size_t>(i)] = loaded_data.historical_data_c1[base_idx + static_cast<std::size_t>(i)] + PortableUniformDistribution(generator);
          all_queries_c2[offset + static_cast<std::size_t>(i)] = loaded_data.historical_data_c2[base_idx + static_cast<std::size_t>(i)] + PortableUniformDistribution(generator);
          all_queries_c3[offset + static_cast<std::size_t>(i)] = loaded_data.historical_data_c3[base_idx + static_cast<std::size_t>(i)] + PortableUniformDistribution(generator);
          all_queries_c4[offset + static_cast<std::size_t>(i)] = loaded_data.historical_data_c4[base_idx + static_cast<std::size_t>(i)] + PortableUniformDistribution(generator);
          all_queries_c5[offset + static_cast<std::size_t>(i)] = loaded_data.historical_data_c5[base_idx + static_cast<std::size_t>(i)] + PortableUniformDistribution(generator);
        }
      }
      // END OF QUERY GENERATION

      // lambda to extract queries for a given query index from the big query buffers
      auto get_query = [&](int query_idx) {
        const std::size_t offset = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(query_length);
        std::vector<std::vector<float>> queries(5);
        queries[0] = std::vector(all_queries_c1.begin() + static_cast<long long>(offset), all_queries_c1.begin() + static_cast<long long>(offset + static_cast<std::size_t>(query_length)));
        queries[1] = std::vector(all_queries_c2.begin() + static_cast<long long>(offset), all_queries_c2.begin() + static_cast<long long>(offset + static_cast<std::size_t>(query_length)));
        queries[2] = std::vector(all_queries_c3.begin() + static_cast<long long>(offset), all_queries_c3.begin() + static_cast<long long>(offset + static_cast<std::size_t>(query_length)));
        queries[3] = std::vector(all_queries_c4.begin() + static_cast<long long>(offset), all_queries_c4.begin() + static_cast<long long>(offset + static_cast<std::size_t>(query_length)));
        queries[4] = std::vector(all_queries_c5.begin() + static_cast<long long>(offset), all_queries_c5.begin() + static_cast<long long>(offset + static_cast<std::size_t>(query_length)));
        return queries;
      };

      std::vector<std::vector<std::vector<float>>> queries_pool(MAX_QUERIES_PER_TIME_SERIES);
      for (int q = 0; q < MAX_QUERIES_PER_TIME_SERIES; ++q) {
        queries_pool[q] = get_query(q);
      }

      for (int query_count = 1; query_count <= MAX_QUERIES_PER_TIME_SERIES; ++query_count) {
        std::cout << "Running sequential config: query_length=" << query_length << ", num_queries=" << query_count << "\n";

        // 2 WARMUP RUNS
        std::cout << "  -> Running warmup runs...\n";
        for (int run = 0; run < 2; ++run) {
          for (int q = 0; q < query_count; ++q) {
            SequentialSADSearch(loaded_data, queries_pool[q]);
          }
        }

        RunStats seq_stats(RUNS);
        for (int run_idx = 0; run_idx < RUNS; ++run_idx) {
          const std::clock_t cpu_start = std::clock();
          const auto wall_start = std::chrono::steady_clock::now();

          for (int q = 0; q < query_count; ++q) {
            SequentialSADSearch(loaded_data, queries_pool[q]);
          }

          const auto wall_end = std::chrono::steady_clock::now();
          const std::clock_t cpu_end = std::clock();

          seq_stats.wall_ms[run_idx] = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
          seq_stats.cpu_ms[run_idx] = 1000.0 * static_cast<double>(cpu_end - cpu_start) / static_cast<double>(CLOCKS_PER_SEC);
        }
        SaveStats(csv_file, "sequential", query_length, query_count, 0, seq_stats, RUNS);
        std::cout << "  -> wall_min=" << seq_stats.wall_min << " ms, wall_avg=" << seq_stats.wall_avg << " ms, wall_max=" << seq_stats.wall_max << " ms, wall_std=" << seq_stats.wall_std << " ms\n";

        for (const int thread_count : thread_counts) {
          // 2 WARMUP RUNS
          std::cout << "Running parallel config: query_length=" << query_length << ", num_queries=" << query_count << ", threads=" << thread_count << "\n";

          std::cout << "  -> Running warmup runs...\n";
          for (int run = 0; run < 2; ++run) {
            for (int q = 0; q < query_count; ++q) {
              ParallelSADSearch(loaded_data, queries_pool[q], thread_count);
            }
          }

          RunStats par_stats(RUNS);
          for (int run_idx = 0; run_idx < RUNS; ++run_idx) {
            const std::clock_t cpu_start = std::clock();
            const auto wall_start = std::chrono::steady_clock::now();

            for (int q = 0; q < query_count; ++q) {
              ParallelSADSearch(loaded_data, queries_pool[q], thread_count);
            }

            const auto wall_end = std::chrono::steady_clock::now();
            const std::clock_t cpu_end = std::clock();

            par_stats.wall_ms[run_idx] = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
            par_stats.cpu_ms[run_idx] = 1000.0 * static_cast<double>(cpu_end - cpu_start) / static_cast<double>(CLOCKS_PER_SEC);
          }
          SaveStats(csv_file, "parallel", query_length, query_count, thread_count, par_stats, RUNS);
          std::cout << "  -> wall_min=" << par_stats.wall_min << " ms, wall_avg=" << par_stats.wall_avg << " ms, wall_max=" << par_stats.wall_max << " ms, wall_std=" << par_stats.wall_std << " ms\n";
        }
      }
    }

    std::cout << "Benchmark completed. Stats saved to " << stats_path << "\n";

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
