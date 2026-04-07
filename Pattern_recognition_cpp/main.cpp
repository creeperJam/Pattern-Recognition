#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "utilities.h"
#include "parallel.h"
#include "sequential.h"

int main(int argc, char **argv) {
    try {
        std::string filepath = R"(C:\Users\Cosimo\Desktop\GitHub\Pattern-Recognition\Pattern_Recognition_cpp\data\realistic_data.csv)";
        constexpr int RUNS = 5;
        constexpr int THREADS = 16;
        constexpr int QUERY_LENGTH = 256;
        constexpr int QUERIES_PER_TIME_SERIES = 4;

        std::cout << "Loading data from " << filepath << "...\n";
        TimeSeriesSoA loaded_data = LoadAndPrepareData(filepath);

        constexpr std::size_t n_series = 5;
        const std::size_t n_points = loaded_data.historical_data_c1.size();
        std::cout << "Loaded " << n_series << " time series with " << n_points << " points each.\n";

        if (n_points < static_cast<std::size_t>(QUERY_LENGTH)) {
            throw std::runtime_error("Query length is greater than series length.");
        }

        // in each vector store all the queries that will be executed on that time series
        std::vector<float> all_queries_c1(QUERY_LENGTH * QUERIES_PER_TIME_SERIES);
        std::vector<float> all_queries_c2(QUERY_LENGTH * QUERIES_PER_TIME_SERIES);
        std::vector<float> all_queries_c3(QUERY_LENGTH * QUERIES_PER_TIME_SERIES);
        std::vector<float> all_queries_c4(QUERY_LENGTH * QUERIES_PER_TIME_SERIES);
        std::vector<float> all_queries_c5(QUERY_LENGTH * QUERIES_PER_TIME_SERIES);


        std::mt19937 generator{42};
        std::normal_distribution<float> noise{0.0f, 15.0f};

        std::cout << "Generating " << QUERIES_PER_TIME_SERIES * 5 << " queries from actual data...\n";
        //TODO: usare nomi variabili più chiari
        std::vector<std::size_t> ground_truth_indices(QUERIES_PER_TIME_SERIES);
        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            std::size_t safe_idx = (q + 1) * (n_points / (QUERIES_PER_TIME_SERIES + 2));

            if (safe_idx + QUERY_LENGTH > n_points) {
                safe_idx = 0;
            }
            ground_truth_indices[q] = safe_idx;
            std::cout << "  -> Query " << q << " extracted starting at index " << safe_idx << "\n";
        }

        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            std::size_t base_idx = ground_truth_indices[q];
            std::size_t offset = static_cast<std::size_t>(q) * static_cast<std::size_t>(QUERY_LENGTH);

            for (int i = 0; i < QUERY_LENGTH; ++i) {
                all_queries_c1[offset + i] = loaded_data.historical_data_c1[base_idx + i] + noise(generator);
                all_queries_c2[offset + i] = loaded_data.historical_data_c2[base_idx + i] + noise(generator);
                all_queries_c3[offset + i] = loaded_data.historical_data_c3[base_idx + i] + noise(generator);
                all_queries_c4[offset + i] = loaded_data.historical_data_c4[base_idx + i] + noise(generator);
                all_queries_c5[offset + i] = loaded_data.historical_data_c5[base_idx + i] + noise(generator);
            }
        }


        // lambda to get all queries with specific IDX
        auto get_all_queries = [&](int query_idx) {
            const std::size_t offset = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(QUERY_LENGTH);

            std::vector<std::vector<float> > queries(5);
            queries[0] = std::vector<float>(all_queries_c1.begin() + offset, all_queries_c1.begin() + offset + QUERY_LENGTH);
            queries[1] = std::vector<float>(all_queries_c2.begin() + offset, all_queries_c2.begin() + offset + QUERY_LENGTH);
            queries[2] = std::vector<float>(all_queries_c3.begin() + offset, all_queries_c3.begin() + offset + QUERY_LENGTH);
            queries[3] = std::vector<float>(all_queries_c4.begin() + offset, all_queries_c4.begin() + offset + QUERY_LENGTH);
            queries[4] = std::vector<float>(all_queries_c5.begin() + offset, all_queries_c5.begin() + offset + QUERY_LENGTH);

            return queries;
        };

        std::cout << "Performing sequential warm-up runs...\n";
        for (int i = 0; i < RUNS/2; ++i) {
            for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
                const auto queries = get_all_queries(q);
                SequentialSADSearch(loaded_data, queries);
            }
        }


        std::cout << "Running sequential tests (" << RUNS << " iterations)...\n";
        std::vector<double> seq_times;
        seq_times.reserve(RUNS);
        std::vector<SADResults> seq_results_by_query(QUERIES_PER_TIME_SERIES);

        for (int i = 0; i < RUNS; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
                const auto queries = get_all_queries(q);
                seq_results_by_query[q] = SequentialSADSearch(loaded_data, queries);
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            seq_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        std::cout << "Performing parallel warm-up runs...\n";
        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            const auto queries = get_all_queries(q);
            ParallelSADSearch(loaded_data, queries, THREADS);
        }

        std::cout << "Running parallel tests (" << RUNS << " iterations)...\n";
        std::vector<double> par_times;
        par_times.reserve(RUNS);
        std::vector<SADResults> par_results_by_query(QUERIES_PER_TIME_SERIES);

        for (int i = 0; i < RUNS; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
                const auto queries = get_all_queries(q);
                par_results_by_query[q] = ParallelSADSearch(loaded_data, queries, THREADS);
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            par_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        bool all_results_match = true;
        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            if (!SameResults(seq_results_by_query[q], par_results_by_query[q])) {
                all_results_match = false;
                break;
            }
        }

        const double avg_seq = Average(seq_times);
        const double avg_par = Average(par_times);
        const double speedup = (avg_par > 0.0) ? (avg_seq / avg_par) : 0.0;

        std::cout << "\n--- Results ---\n";
        std::cout << "Executed " << QUERIES_PER_TIME_SERIES << " queries on all time series.\n";
        std::cout << "Results match across implementations: " << (all_results_match ? "true" : "false") << "\n";
        std::cout << "\n========================= SEQUENTIAL SEARCH RESULTS =========================" << std::endl;
        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            std::cout << "[ QUERY " << q + 1 << " ]" << std::endl;
            for (std::size_t s = 0; s < n_series; ++s) {
                std::cout << "  - Serie C" << s + 1 << " -> indice: " << seq_results_by_query[q].best_indices[s] << ", SAD: " << seq_results_by_query[q].best_sads[s] << std::endl;
            }
            std::cout << "------------------------------------------------------------------" << std::endl;
        }

        std::cout << "\n========================= PARALLEL SEARCH RESULTS =========================" << std::endl;
        for (int q = 0; q < QUERIES_PER_TIME_SERIES; ++q) {
            std::cout << "[ QUERY " << q + 1 << " ]" << std::endl;
            for (std::size_t s = 0; s < n_series; ++s) {
                std::cout << "  - Serie C" << s + 1 << " -> indice: " << par_results_by_query[q].best_indices[s]
                          << ", SAD: " << par_results_by_query[q].best_sads[s] << std::endl;
            }
            std::cout << "------------------------------------------------------------------" << std::endl;
        }

        std::cout << "Average Sequential Time: " << avg_seq << " seconds\n";
        std::cout << "Average Parallel Time:   " << avg_par << " seconds\n";
        std::cout << "Calculated Speedup:      " << speedup << "x\n";

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
