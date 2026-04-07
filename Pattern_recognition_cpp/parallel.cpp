#include "parallel.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <omp.h>

SADResults ParallelSADSearch(const TimeSeriesSoA &time_series, const std::vector<std::vector<float> > &queries, int n_threads) {
    if (queries.empty()) {
        throw std::invalid_argument("queries vector is empty.");
    }

    constexpr std::size_t n_series = 5;

    if (queries.size() != n_series) {
        throw std::invalid_argument("must provide exactly one query per time series.");
    }

    const std::size_t n_points = time_series.historical_data_c1.size();
    const std::size_t query_len = queries[0].size();

    if (n_points == 0) {
        throw std::invalid_argument("time_series has zero points.");
    }

    if (time_series.historical_data_c2.size() != n_points ||
        time_series.historical_data_c3.size() != n_points ||
        time_series.historical_data_c4.size() != n_points ||
        time_series.historical_data_c5.size() != n_points) {
        throw std::invalid_argument("all time series columns must have the same length.");
    }

    if (query_len > n_points) {
        throw std::invalid_argument("query length must be <= time series length.");
    }

    const std::size_t total_windows = n_points - query_len + 1;

    SADResults result;
    result.best_indices.assign(n_series, 0);
    result.best_sads.assign(n_series, std::numeric_limits<float>::infinity());

    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    // this basically is a fixed array of fixed pointers. you cannot change anything of it
    const std::vector<float> *const series_ptrs[n_series] = {
        &time_series.historical_data_c1,
        &time_series.historical_data_c2,
        &time_series.historical_data_c3,
        &time_series.historical_data_c4,
        &time_series.historical_data_c5
    };

    for (std::size_t s = 0; s < n_series; ++s) {
        float global_best_sad = std::numeric_limits<float>::infinity();
        int global_best_idx = 0;

        const float *const series_data = series_ptrs[s]->data();
        const float *const query_data = queries[s].data();

        //initial fork, each thread will have 2 private variables
#pragma omp parallel
        {
            float local_best_sad = std::numeric_limits<float>::infinity();
            int local_best_idx = 0;

#pragma omp for schedule(dynamic, 4096)
            for (long long start = 0; start < static_cast<long long>(total_windows); ++start) {
                float sad = 0.0f;

                for (std::size_t j = 0; j < query_len; j += 64) {
                    float chunk_sad = 0.0f;
                    std::size_t limit = std::min(j + 64, query_len);

#pragma omp simd reduction(+:chunk_sad)
                    for (std::size_t k = j; k < limit; ++k) {
                        chunk_sad += std::fabs(series_data[start + k] - query_data[k]);
                    }

                    sad += chunk_sad;

                    // check if we exceed the local_best_sad ONLY after calculating the chunk.
                    if (sad >= local_best_sad) {
                        break;
                    }
                }

                if (sad < local_best_sad) {
                    local_best_sad = sad;
                    local_best_idx = static_cast<int>(start);
                }
            }

            // Safely merge thread-local results into the global trackers
#pragma omp critical
            {
                if (local_best_sad < global_best_sad) {
                    global_best_sad = local_best_sad;
                    global_best_idx = local_best_idx;
                }
            }
        } // end parallel region

        result.best_indices[s] = global_best_idx;
        result.best_sads[s] = global_best_sad;
    }

    return result;
}
