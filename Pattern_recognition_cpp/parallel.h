#ifndef TIMESERIES_PARALLEL_H
#define TIMESERIES_PARALLEL_H
#include "utilities.h"


SADResults ParallelSADSearch(const TimeSeriesSoA& time_series, const std::vector<std::vector<float>>& queries, int n_threads=16);

#endif  // TIMESERIES_PARALLEL_H
