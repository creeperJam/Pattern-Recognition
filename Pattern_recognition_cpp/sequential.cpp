#include "sequential.h"

#include <cmath>
#include <limits>
#include <vector>

SADResults SequentialSADSearch(const TimeSeriesSoA &time_series, const std::vector<std::vector<float> > &queries) {
    const std::size_t SERIES_NUMBER = 5;
    const std::size_t n_points = time_series.historical_data_c1.size();
    const std::size_t query_len = queries[0].size();

    const std::size_t total_windows = n_points - query_len + 1;

    SADResults result;
    result.best_indices.assign(SERIES_NUMBER, 0);
    result.best_sads.assign(SERIES_NUMBER, std::numeric_limits<float>::infinity());

    const std::vector<float> *const series_ptrs[5] = {
        &time_series.historical_data_c1,
        &time_series.historical_data_c2,
        &time_series.historical_data_c3,
        &time_series.historical_data_c4,
        &time_series.historical_data_c5
    };

    for (std::size_t s = 0; s < SERIES_NUMBER; ++s) {
        int best_idx = 0;
        float best_sad = std::numeric_limits<float>::infinity();

        const std::vector<float> &current_series = *(series_ptrs[s]);
        const std::vector<float> &current_query = queries[s];

        for (std::size_t start = 0; start < total_windows; ++start) {
            float sad = 0.0f;
            for (std::size_t j = 0; j < query_len; ++j) {
                sad += std::fabs(current_series[start + j] - current_query[j]);
            }

            if (sad < best_sad) {
                best_sad = sad;
                best_idx = static_cast<int>(start);
            }
        }

        result.best_indices[s] = best_idx;
        result.best_sads[s] = best_sad;
    }

    return result;
}
