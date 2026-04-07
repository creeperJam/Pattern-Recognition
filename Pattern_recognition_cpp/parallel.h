#ifndef TIMESERIES_PARALLEL_HPP
#define TIMESERIES_PARALLEL_HPP

#include <vector>

#include "sequential.h"

SADResults ParallelSADSearch(const TimeSeriesSoA& time_series, const std::vector<std::vector<float>>& queries, int n_threads=16);



#endif  // TIMESERIES_PARALLEL_HPP
