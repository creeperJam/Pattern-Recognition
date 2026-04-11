//
// Created by albi0 on 08/04/2026.
//

#ifndef PATTERN_RECOGNITION_CU_MAIN_CUH
#define PATTERN_RECOGNITION_CU_MAIN_CUH

#include <cuda_runtime.h>
#include "utilities.h"

/**
 * @brief Computes the Sum of Absolute Differences (SAD) between multiple queries and a 5-channel dataset.
 *
 * This kernel relies on a 2D grid architecture:
 * - The X-axis (tid_x) corresponds to the starting index of the sliding window within the main dataset.
 * - The Y-axis (tid_y) corresponds to the specific query pattern currently being evaluated.
 *
 * @details
 * To overcome the global memory bottleneck (Memory Bound), the kernel utilizes a Collaborative Shared Memory
 * loading strategy. Each block loads a data chunk of size (BLOCK_SIZE + QUERY_LENGTH) into shared memory.
 * This "halo" (or ghost cells) approach allows all threads within the block to compute their full sliding
 * window locally without performing redundant out-of-bounds reads to the global memory (VRAM).
 * After synchronization, the core loop computes the SAD using loop unrolling to optimize register usage
 * and instruction cache hits. Output values are written to a flattened 1D array partitioned by query ID.
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
    const float* __restrict__ d_h1, const float* __restrict__ d_h2, const float* __restrict__ d_h3, const float* __restrict__ d_h4, const float* __restrict__ d_h5,
    float* __restrict__ d_results_c1, float* __restrict__ d_results_c2, float* __restrict__ d_results_c3, float* __restrict__ d_results_c4, float* __restrict__ d_results_c5,
    int total_elements);

#endif //PATTERN_RECOGNITION_CU_MAIN_CUH