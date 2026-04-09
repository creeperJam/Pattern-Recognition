#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "utilities.h"

SADResults SequentialSADSearch(const TimeSeriesSoA &time_series, const std::vector<std::vector<float> > &queries);

#endif  // SEQUENTIAL_H
