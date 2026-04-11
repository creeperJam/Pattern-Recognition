#ifndef MAIN_CUH
#define MAIN_CUH

#include <cuda_runtime.h>
#include "utilities.h"

/**
 * @brief Computes the Sum of Absolute Differences (SAD) between multiple queries and a 5-channel dataset.
 *
 * This kernel relies on a 2D grid architecture:
 * - The X-axis (tid_x) corresponds to the starting index of the sliding window within the main dataset.
 * - The Y-axis (tid_y) corresponds to the specific query pattern currently being evaluated.
 *
 * @param d_c1 Pointer to the global memory array containing Channel 1 of the dataset.
 * @param d_c2 Pointer to the global memory array containing Channel 2 of the dataset.
 * @param d_c3 Pointer to the global memory array containing Channel 3 of the dataset.
 * @param d_c4 Pointer to the global memory array containing Channel 4 of the dataset.
 * @param d_c5 Pointer to the global memory array containing Channel 5 of the dataset.
 * @param d_results_c1 Pointer to the output array for Channel 1 SAD scores. Flattened as [NUM_QUERIES][total_elements].
 * @param d_results_c2 Pointer to the output array for Channel 2 SAD scores. Flattened as [NUM_QUERIES][total_elements].
 * @param d_results_c3 Pointer to the output array for Channel 3 SAD scores. Flattened as [NUM_QUERIES][total_elements].
 * @param d_results_c4 Pointer to the output array for Channel 4 SAD scores. Flattened as [NUM_QUERIES][total_elements].
 * @param d_results_c5 Pointer to the output array for Channel 5 SAD scores. Flattened as [NUM_QUERIES][total_elements].
 * @param total_elements The total number of elements in the main dataset (length of the time-series).
 */
__global__ void SearchMultiplePatternsKernel(
    const float* __restrict__ d_c1, const float* __restrict__ d_c2, const float* __restrict__ d_c3, const float* __restrict__ d_c4, const float* __restrict__ d_c5,
    float* __restrict__ d_results_c1, float* __restrict__ d_results_c2, float* __restrict__ d_results_c3, float* __restrict__ d_results_c4, float* __restrict__ d_results_c5,
    int total_elements);

#endif