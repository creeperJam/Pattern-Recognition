#include "main.cuh"

__constant__ float c_q1[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q2[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q3[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q4[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q5[TOTAL_QUERY_ELEMENTS];

int main() {
    DataSoA loaded_data(2'000'000);
    std::string filepath = std::string(PROJECT_SOURCE_DIR) + "/../realistic_data.csv";

    std::cout << "Loading data from: " << filepath << "\n";
    if (!ReadFile(filepath, loaded_data)) {
        std::cerr << "ERROR: unable to open file " << filepath << "\n";
        return -1;
    }
    int total_elements = loaded_data.c1.size();
    std::cout << "Loaded " << total_elements << " data.\n";

    if (total_elements == 0) return 1;

    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c1{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c2{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c3{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c4{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c5{};

    std::cout << "Generating " << NUM_QUERIES << " realistic noisy queries from ground-truth data...\n";

    if (!GenerateQueries(all_queries_c1, all_queries_c2, all_queries_c3, all_queries_c4, all_queries_c5, loaded_data, total_elements)) {
        std::cerr << "ERROR: There was an error during the generation of the random queries.\n";
        return 1;
    }

    float *d_c1, *d_c2, *d_c3, *d_c4, *d_c5;
    float *d_results_c1, *d_results_c2, *d_results_c3, *d_results_c4, *d_results_c5;

    cudaMalloc(&d_c1, total_elements * sizeof(float));
    cudaMalloc(&d_c2, total_elements * sizeof(float));
    cudaMalloc(&d_c3, total_elements * sizeof(float));
    cudaMalloc(&d_c4, total_elements * sizeof(float));
    cudaMalloc(&d_c5, total_elements * sizeof(float));

    cudaMalloc(&d_results_c1, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c2, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c3, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c4, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c5, total_elements * sizeof(float) * NUM_QUERIES);

    cudaMemcpy(d_c1, loaded_data.c1.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c2, loaded_data.c2.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c3, loaded_data.c3.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c4, loaded_data.c4.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c5, loaded_data.c5.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);


    cudaMemcpyToSymbol(c_q1, all_queries_c1.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q2, all_queries_c2.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q3, all_queries_c3.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q4, all_queries_c4.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q5, all_queries_c5.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));

    for (int i = 0; i < NUM_WARMUP; i++) {
        dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
        dim3 blocksPerGrid(
            (total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NUM_QUERIES + threadsPerBlock.y - 1) / threadsPerBlock.y,
            1
        );
        int shared_bytes_needed = 5 * (BLOCK_SIZE + QUERY_LENGTH) * sizeof(float);

        cudaFuncSetAttribute(
            SearchMultiplePatternsKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes_needed
        );

        SearchMultiplePatternsKernel<<<blocksPerGrid, threadsPerBlock, shared_bytes_needed>>>(
            d_c1, d_c2, d_c3, d_c4, d_c5,
            d_results_c1, d_results_c2, d_results_c3, d_results_c4, d_results_c5,
            total_elements);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << "\n";
            return -1;
        }

        cudaDeviceSynchronize();
    }

    std::vector<float> h_results_c1(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c2(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c3(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c4(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c5(total_elements * NUM_QUERIES);

    std::array<int, NUM_QUERIES * CHANNEL_COUNT> last_best_indices;
    std::array<float, NUM_QUERIES * CHANNEL_COUNT> min_sads;
    std::array<int, NUM_QUERIES * CHANNEL_COUNT> best_indices;
    std::array<double, NUM_RUNS> wall_times;
    last_best_indices.fill(-1);

    for (int i = 0; i < NUM_RUNS; ++i) {
        best_indices.fill(-1);
        min_sads.fill(std::numeric_limits<float>::max());


        dim3 threadsPerBlock(BLOCK_SIZE, 1, 1);
        dim3 blocksPerGrid(
            (total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NUM_QUERIES + threadsPerBlock.y - 1) / threadsPerBlock.y,
            1
            );
        int shared_bytes_needed = 5 * (BLOCK_SIZE + QUERY_LENGTH) * sizeof(float);

        cudaFuncSetAttribute(
            SearchMultiplePatternsKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes_needed
        );

        auto time_start = std::chrono::high_resolution_clock::now();
        SearchMultiplePatternsKernel<<<blocksPerGrid, threadsPerBlock, shared_bytes_needed>>>(
            d_c1, d_c2, d_c3, d_c4, d_c5,
            d_results_c1, d_results_c2, d_results_c3, d_results_c4, d_results_c5,
            total_elements
        );
        cudaDeviceSynchronize();
        auto time_end = std::chrono::high_resolution_clock::now();
        wall_times[i] = std::chrono::duration<double>(time_end - time_start).count();

        cudaMemcpy(h_results_c1.data(), d_results_c1, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_results_c2.data(), d_results_c2, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_results_c3.data(), d_results_c3, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_results_c4.data(), d_results_c4, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_results_c5.data(), d_results_c5, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);

        const std::vector<float>* per_channel_results[5] = {
            &h_results_c1,
            &h_results_c2,
            &h_results_c3,
            &h_results_c4,
            &h_results_c5
        };

        for (int query = 0; query < NUM_QUERIES; query++) {
            int offset = query * total_elements;
            for (int c = 0; c < 5; ++c) {
                const int qc = query * 5 + c;
                const std::vector<float>& channel_results = *per_channel_results[c];

                for (int row = 0; row <= total_elements - QUERY_LENGTH; row++) {
                    if (channel_results[offset + row] < min_sads[qc]) {
                        min_sads[qc] = channel_results[offset + row];
                        best_indices[qc] = row;
                    }
                }
            }
        }

        for (int i = 0; i < NUM_QUERIES * 5; ++i) {
            if (best_indices[i] != last_best_indices[i] && last_best_indices[i] != -1) {
                const int query_id = i / 5;
                const int channel_id = i % 5;
                std::cout << "ERROR: index for query " << query_id + 1
                          << " and channel C" << channel_id + 1
                          << " is different from previous run\n";
            }
            last_best_indices[i] = best_indices[i];
        }
    }

    // std::cout << "Average time in " << NUM_RUNS << " runs: " << total_time / NUM_RUNS << "s\n";
    if (!PrintAndSaveResults(best_indices, min_sads)) {
        std::cout << "ERROR: There was an error during the printing or saving of the results.\n";
        return -1;
    }

    if (!SaveStats(wall_times)) {
        std::cout << "ERROR: There was an error during the saving of the time statistics.\n";
        return -1;
    }

    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_c3); cudaFree(d_c4); cudaFree(d_c5);
    cudaFree(d_results_c1); cudaFree(d_results_c2); cudaFree(d_results_c3); cudaFree(d_results_c4); cudaFree(d_results_c5);

    return 0;
}

__global__ void SearchMultiplePatternsKernel(
    const float* __restrict__ d_c1, const float* __restrict__ d_c2, const float* __restrict__ d_c3, const float* __restrict__ d_c4, const float* __restrict__ d_c5,
    float* __restrict__ d_results_c1, float* __restrict__ d_results_c2, float* __restrict__ d_results_c3, float* __restrict__ d_results_c4, float* __restrict__ d_results_c5,
    int total_elements)
{
    extern __shared__ float s_data[];
    constexpr int BLOCK_SIZE = 256;
    constexpr int SHARED_SIZE = BLOCK_SIZE + QUERY_LENGTH;

    float* s_h1 = s_data;
    float* s_h2 = &s_data[SHARED_SIZE];
    float* s_h3 = &s_data[SHARED_SIZE * 2];
    float* s_h4 = &s_data[SHARED_SIZE * 3];
    float* s_h5 = &s_data[SHARED_SIZE * 4];

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) {
        int global_idx = blockIdx.x * blockDim.x + i;

        if (global_idx < total_elements) {
            s_h1[i] = d_c1[global_idx];
            s_h2[i] = d_c2[global_idx];
            s_h3[i] = d_c3[global_idx];
            s_h4[i] = d_c4[global_idx];
            s_h5[i] = d_c5[global_idx];
        } else {
            s_h1[i] = 0.0f;
            s_h2[i] = 0.0f;
            s_h3[i] = 0.0f;
            s_h4[i] = 0.0f;
            s_h5[i] = 0.0f;
        }
    }

    __syncthreads();

    if (tid_x <= total_elements - QUERY_LENGTH && tid_y < NUM_QUERIES) {
        float sad_c1 = 0.0f;
        float sad_c2 = 0.0f;
        float sad_c3 = 0.0f;
        float sad_c4 = 0.0f;
        float sad_c5 = 0.0f;
        int q_start = tid_y * QUERY_LENGTH;

        #pragma unroll
        for (int i = 0; i < QUERY_LENGTH; i++) {
            sad_c1 += fabs(s_h1[threadIdx.x + i] - c_q1[q_start + i]);
            sad_c2 += fabs(s_h2[threadIdx.x + i] - c_q2[q_start + i]);
            sad_c3 += fabs(s_h3[threadIdx.x + i] - c_q3[q_start + i]);
            sad_c4 += fabs(s_h4[threadIdx.x + i] - c_q4[q_start + i]);
            sad_c5 += fabs(s_h5[threadIdx.x + i] - c_q5[q_start + i]);
        }

        const int out_idx = tid_y * total_elements + tid_x;
        d_results_c1[out_idx] = sad_c1;
        d_results_c2[out_idx] = sad_c2;
        d_results_c3[out_idx] = sad_c3;
        d_results_c4[out_idx] = sad_c4;
        d_results_c5[out_idx] = sad_c5;
    }
}