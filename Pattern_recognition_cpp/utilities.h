#ifndef UTILITIES_H
#define UTILITIES_H
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct TimeSeriesSoA {
    std::vector<float> timeseries_c1;
    std::vector<float> timeseries_c2;
    std::vector<float> timeseries_c3;
    std::vector<float> timeseries_c4;
    std::vector<float> timeseries_c5;

    TimeSeriesSoA(const int size) {
        timeseries_c1.reserve(size);
        timeseries_c2.reserve(size);
        timeseries_c3.reserve(size);
        timeseries_c4.reserve(size);
        timeseries_c5.reserve(size);
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

struct RunStats {
    std::vector<double> wall_ms;
    std::vector<double> cpu_ms;

    double wall_avg;
    double cpu_avg;

    double wall_min;
    double cpu_min;
    double wall_max;
    double cpu_max;

    double wall_std;
    double cpu_std;

    RunStats(const int RUNS) : wall_ms(RUNS), cpu_ms(RUNS) {
    }
};

TimeSeriesSoA LoadAndPrepareData(const std::string &filepath);

double Average(const std::vector<double> &values);

bool SameResults(const SADResults &a, const SADResults &b, float atol = 1e-3f);

float PortableUniformDistribution(std::mt19937 &eng);

bool SaveStats(const std::string &algo, int query_size, int query_count, int num_threads, RunStats &stats, int RUNS);

bool SaveResults(std::vector<SADResults> &results, int NUM_QUERIES, int QUERY_LENGTH, int num_threads);

#endif  // UTILITIES_H
