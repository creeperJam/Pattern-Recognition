//
// Created by albi0 on 08/04/2026.
//

#ifndef PATTERN_RECOGNITION_CU_MAIN_CUH
#define PATTERN_RECOGNITION_CU_MAIN_CUH

#include <cuda_runtime.h>
#include "utilities.h"

constexpr int BLOCK_SIZE = 256;

__global__ void SearchMultiplePatternsKernel(
    const float* __restrict__ d_h1, const float* __restrict__ d_h2, const float* __restrict__ d_h3, const float* __restrict__ d_h4, const float* __restrict__ d_h5,
    float* __restrict__ d_results_c1, float* __restrict__ d_results_c2, float* __restrict__ d_results_c3, float* __restrict__ d_results_c4, float* __restrict__ d_results_c5,
    int total_elements);

#endif //PATTERN_RECOGNITION_CU_MAIN_CUH