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

/**
 * Function that given the path to a dataset, loads its contents into an appropriate structure.
 * @param filepath Path to CSV dataset containing time series data.
 * @return TimeSeriesSoA structure populated from CSV data.
 */
TimeSeriesSoA LoadAndPrepareData(const std::string &filepath);

/**
 * Function that calculates the average of the values inside a double vector.
 * @param values Vector of double values of which calculate the average.
 * @return The average of the values inside the vector.
 */
double Average(const std::vector<double> &values);

/**
 * Function that evaluates whether two SADResults structures are approximately equal, considering a specified absolute tolerance for the SAD values.
 * @param a First structure to compare.
 * @param b Second structure to compare.
 * @param atol Absolute tolerance for comparing values.
 * @return True if the results are approximately equal within the specified tolerance, false otherwise.
 */
bool SameResults(const SADResults &a, const SADResults &b, float atol = 1e-3f);

/**
 * Generates a random float in the range [-15.0, 15.0] using the provided random engine. This function is designed to be portable across different platforms and compilers, ensuring consistent random number generation.
 * @param eng The random engine to use.
 * @return A random float in the range [-15.0, 15.0].
 */
float PortableUniformDistribution(std::mt19937 &eng);

/**
 * Function to calculate run stats such as wall-time average, min, max, std... It also saves the results to a CSV file inside a project subfolder.
 * @param algo Whether sequential or parallel version.
 * @param query_size The length of the query time series.
 * @param query_count The number of queries performed on each data channel.
 * @param num_threads The number of threads used to execute the queries. 0 if sequential.
 * @param stats RunStats structure containing raw values that will be used to calculate the metrics and save them to file.
 * @param RUNS Number of runs executed to calculate the metrics. It is used to calculate the average and std, and also to know how many values are inside the stats structure.
 * @return True if saving to CSV was successful, false otherwise.
 */
bool SaveStats(const std::string &algo, int query_size, int query_count, int num_threads, RunStats &stats, int RUNS);

/**
 * Similar to SaveStats but this function actually saves the SADResults (best indixes and best SADs) to a CSV file.
 * @param results SADResults structure containing results.
 * @param NUM_QUERIES Number of queries executed on each data channel.
 * @param QUERY_LENGTH Length of each query.
 * @param num_threads Number of threads used for this research.
 * @return True if saving to CSV was successful, false otherwise.
 */
bool SaveResults(std::vector<SADResults> &results, int NUM_QUERIES, int QUERY_LENGTH, int num_threads);

#endif  // UTILITIES_H
