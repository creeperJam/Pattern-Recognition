#include "parallel.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <omp.h>

SADResults ParallelSADSearch(const TimeSeriesSoA &time_series, const std::vector<std::vector<float> > &queries, int n_threads) {
    constexpr std::size_t TIME_SERIES_NUMBER = 5;
    const std::size_t n_points = time_series.timeseries_c1.size();
    const std::size_t query_len = queries[0].size();

    const auto total_windows = static_cast<long long>(n_points - query_len + 1);

    SADResults result;
    result.best_indices.assign(TIME_SERIES_NUMBER, 0);
    result.best_sads.assign(TIME_SERIES_NUMBER, std::numeric_limits<float>::max());

    //we use pointers so we don't load millions of floats every time
    const std::vector<float> *const series_ptrs[TIME_SERIES_NUMBER] = {
        &time_series.timeseries_c1,
        &time_series.timeseries_c2,
        &time_series.timeseries_c3,
        &time_series.timeseries_c4,
        &time_series.timeseries_c5
    };

    for (std::size_t s = 0; s < TIME_SERIES_NUMBER; ++s) {
        float global_best_sad = std::numeric_limits<float>::max();
        int global_best_idx = 0;

        const float *const series_data = series_ptrs[s]->data();
        const float *const query_data = queries[s].data();

        // initial fork, a number of threads is spawned and each will have 2 private variables
#pragma omp parallel default(none) num_threads(n_threads) shared(global_best_sad, global_best_idx, series_data, query_data, total_windows, query_len)
        {
            float local_best_sad = std::numeric_limits<float>::max();
            int local_best_idx = 0;

            // each thread will pickup a chunk of windows
#pragma omp for schedule(dynamic, 256)
            // iterate over all possible starting windows in the current time series
            for (long long start = 0; start < total_windows; ++start) {
                float sad = 0.0f;

                // we calculate SAD using chunks to still be able to use SIMD
                for (std::size_t j = 0; j < query_len; j += 64) {
                    float chunk_sad = 0.0f;

                    // ensure we don't read out of bounds on the final chunk
                    std::size_t limit = std::min(j + 64, query_len);

                    // force compiler to vectorize this loop and accumulate the result
#pragma omp simd reduction(+:chunk_sad)
                    for (std::size_t k = j; k < limit; ++k) {
                        chunk_sad += std::fabs(series_data[start + k] - query_data[k]);
                    }

                    // add the chunk's error to the total window error
                    sad += chunk_sad;

                    // check if we exceed the local_best_sad ONLY after calculating the chunk.
                    if (sad >= local_best_sad) {
                        break;
                    }
                }

                // if this window is the best one found so far by this thread, save it
                if (sad < local_best_sad) {
                    local_best_sad = sad;
                    local_best_idx = static_cast<int>(start);
                }
            }

            // safely merge thread-local results into the global trackers (one by one)
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
