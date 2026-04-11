#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "utilities.h"

/**
 * Function that by using SAD metric searches a multidimensional time series given per-channel specific queries. Sequential version.
 * @param time_series TimeSeriesSoA structure containing the time series data.
 * @param queries Vector of vectors containing the queries subdivided for channel.
 * @return SADResults structure containing best SADs and best indexes.
 */
SADResults SequentialSADSearch(const TimeSeriesSoA &time_series, const std::vector<std::vector<float> > &queries);

#endif  // SEQUENTIAL_H
