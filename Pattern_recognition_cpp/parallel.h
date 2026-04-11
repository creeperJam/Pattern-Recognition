#ifndef PARALLEL_H
#define PARALLEL_H
#include "utilities.h"

/**
 * Function that by using SAD metric searches a multidimensional time series given per-channel specific queries. Parallel version.
 * @param time_series TimeSeriesSoA structure containing the time series data.
 * @param queries Vector of vectors containing the queries subdivided for channel.
 * @param n_threads Number of threads to use to run this function.
 * @return SADResults structure containing best SADs and best indexes.
 */
SADResults ParallelSADSearch(const TimeSeriesSoA& time_series, const std::vector<std::vector<float>>& queries, int n_threads=16);

#endif  // PARALLEL_H