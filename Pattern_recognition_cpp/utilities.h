#ifndef UTILITIES_H
#define UTILITIES_H
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

struct TimeSeriesSoA {
    std::vector<float> historical_data_c1;
    std::vector<float> historical_data_c2;
    std::vector<float> historical_data_c3;
    std::vector<float> historical_data_c4;
    std::vector<float> historical_data_c5;

    TimeSeriesSoA(const int size) {
        historical_data_c1.reserve(size);
        historical_data_c2.reserve(size);
        historical_data_c3.reserve(size);
        historical_data_c4.reserve(size);
        historical_data_c5.reserve(size);
    }
};

struct SADResults {
    // each vector contains QUERIES_PER_TIME_SERIES elements
    // element 0 -> best index for query 1 on column 1
    // element 1 -> best index for query 1 on column 2
    // etc...
    // same logic applies to best_sads
    std::vector<int> best_indices;
    std::vector<float> best_sads;
};

TimeSeriesSoA LoadAndPrepareData(const std::string& filepath);

double Average(const std::vector<double> &values);
bool SameResults(const SADResults &a, const SADResults &b, float atol = 1e-4f);

#endif // UTILITIES_H