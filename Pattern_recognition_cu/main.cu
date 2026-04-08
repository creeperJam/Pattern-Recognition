#include <array>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include <random>

// TODO: Split every part into function for readability

// My RTX 4060 has 64 KiB of constant memory, meaning that query length and number have to respect the following limit:
// QUERY_LENGTH * NUM_QUERIES * sizeof(float) * COLUMNS_NUMBER(5 in this case) <= 65536, so:
// QUERY_LENGTH * NUM_QUERIES <= 3276 (approx.)
constexpr int QUERY_LENGTH = 819; // MAX: 3276 - Limited by constant and shared memory size (99 KiB shared = 101376 B => (4812+256) * 5 * 4 = 101376, with 4813 we'd go over this limit). ACTUAL LIMIT IS 3276 DUE TO CONSTANT MEMORY SIZE
constexpr int NUM_QUERIES = 4;     // MAX: 3276 / QUERY_LENGTH
constexpr int NUM_RUNS = 100.0f;
constexpr int NUM_WARMUP = 10;
constexpr int TOTAL_QUERY_ELEMENTS = QUERY_LENGTH * NUM_QUERIES;

__constant__ float c_q1[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q2[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q3[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q4[TOTAL_QUERY_ELEMENTS];
__constant__ float c_q5[TOTAL_QUERY_ELEMENTS];

struct data_SoA {
    std::vector<float> c1;
    std::vector<float> c2;
    std::vector<float> c3;
    std::vector<float> c4;
    std::vector<float> c5;

    data_SoA(int size) {
        c1.reserve(size);
        c2.reserve(size);
        c3.reserve(size);
        c4.reserve(size);
        c5.reserve(size);
    }
};

__global__ void search_multiple_patterns_kernel(
    const float* __restrict__ d_h1, const float* __restrict__ d_h2, const float* __restrict__ d_h3, const float* __restrict__ d_h4, const float* __restrict__ d_h5,
    float* __restrict__ d_results_c1,
    float* __restrict__ d_results_c2,
    float* __restrict__ d_results_c3,
    float* __restrict__ d_results_c4,
    float* __restrict__ d_results_c5,
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

    // __shared__ float s_h1[SHARED_SIZE];
    // __shared__ float s_h2[SHARED_SIZE];
    // __shared__ float s_h3[SHARED_SIZE];
    // __shared__ float s_h4[SHARED_SIZE];
    // __shared__ float s_h5[SHARED_SIZE];

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) {
        int global_idx = blockIdx.x * blockDim.x + i;

        if (global_idx < total_elements) {
            s_h1[i] = d_h1[global_idx];
            s_h2[i] = d_h2[global_idx];
            s_h3[i] = d_h3[global_idx];
            s_h4[i] = d_h4[global_idx];
            s_h5[i] = d_h5[global_idx];
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

float portable_uniform(std::mt19937& eng) {
    uint32_t raw_value = eng();

    uint32_t max_value = std::mt19937::max(); // Per mt19937 è 4294967295
    float normalized = static_cast<float>(raw_value) / static_cast<float>(max_value);

    return normalized * 15.0f;
}

int main() {
    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/../realistic_data.csv";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "ERROR: unable to open file " << filename << std::endl;
        return 1;
    }

    data_SoA loaded_data(2'000'000);

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ';');

        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { loaded_data.c1.push_back(std::stof(token)); }
                catch (...) {}
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { loaded_data.c2.push_back(std::stof(token)); }
                catch (...) {}
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { loaded_data.c3.push_back(std::stof(token)); }
                catch (...) {}
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { loaded_data.c4.push_back(std::stof(token)); }
                catch (...) {}
            }
        }
        if (std::getline(ss, token, ';')) {
            if (!token.empty()) {
                try { loaded_data.c5.push_back(std::stof(token)); }
                catch (...) {}
            }
        }
    }
    file.close();

    int total_elements = loaded_data.c1.size();
    std::cout << "Loaded " << total_elements << " data." << std::endl;

    if (total_elements == 0) return 1;

    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c1{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c2{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c3{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c4{};
    std::array<float, TOTAL_QUERY_ELEMENTS> all_queries_c5{};

    std::mt19937 generator{42};
    std::normal_distribution<float> noise{0.0f, 15.0f};
    std::cout << "Generating " << NUM_QUERIES << " realistic noisy queries from ground-truth data...\n";

    std::vector<std::size_t> ground_truth_indices(NUM_QUERIES);
    for (int q = 0; q < NUM_QUERIES; ++q) {
        std::size_t safe_idx = (q + 1) * (total_elements / (NUM_QUERIES + 2));

        if (safe_idx + QUERY_LENGTH > total_elements) {
            safe_idx = 0;
        }
        ground_truth_indices[q] = safe_idx;
        std::cout << "  -> Query " << q << " extracted starting at index " << safe_idx << "\n";
    }

    for (int q = 0; q < NUM_QUERIES; ++q) {
        std::size_t base_idx = ground_truth_indices[q];
        std::size_t offset = static_cast<std::size_t>(q) * static_cast<std::size_t>(QUERY_LENGTH);

        for (int i = 0; i < QUERY_LENGTH; ++i) {
            all_queries_c1[offset + i] = loaded_data.c1[base_idx + i] + portable_uniform(generator);
            all_queries_c2[offset + i] = loaded_data.c2[base_idx + i] + portable_uniform(generator);
            all_queries_c3[offset + i] = loaded_data.c3[base_idx + i] + portable_uniform(generator);
            all_queries_c4[offset + i] = loaded_data.c4[base_idx + i] + portable_uniform(generator);
            all_queries_c5[offset + i] = loaded_data.c5[base_idx + i] + portable_uniform(generator);
            if (q == 0) {
                std::cout << all_queries_c1[i] << " ==> ";
            }
        }
    }

    float *d_h1;
    float *d_h2;
    float *d_h3;
    float *d_h4;
    float *d_h5;
    float *d_results_c1;
    float *d_results_c2;
    float *d_results_c3;
    float *d_results_c4;
    float *d_results_c5;

    cudaMalloc(&d_h1, total_elements * sizeof(float));
    cudaMalloc(&d_h2, total_elements * sizeof(float));
    cudaMalloc(&d_h3, total_elements * sizeof(float));
    cudaMalloc(&d_h4, total_elements * sizeof(float));
    cudaMalloc(&d_h5, total_elements * sizeof(float));

    cudaMalloc(&d_results_c1, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c2, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c3, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c4, total_elements * sizeof(float) * NUM_QUERIES);
    cudaMalloc(&d_results_c5, total_elements * sizeof(float) * NUM_QUERIES);

    cudaMemcpy(d_h1, loaded_data.c1.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h2, loaded_data.c2.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h3, loaded_data.c3.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h4, loaded_data.c4.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h5, loaded_data.c5.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(c_q1, all_queries_c1.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q2, all_queries_c2.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q3, all_queries_c3.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q4, all_queries_c4.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));
    cudaMemcpyToSymbol(c_q5, all_queries_c5.data(), TOTAL_QUERY_ELEMENTS * sizeof(float));

    double total_time = 0.0f;
    for (int i = 0; i < NUM_WARMUP; i++) {
        dim3 threadsPerBlock(256, 1, 1);
        dim3 blocksPerGrid(
            (total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NUM_QUERIES + threadsPerBlock.y - 1) / threadsPerBlock.y,
            1
        );
        int shared_bytes_needed = 5 * (256 + QUERY_LENGTH) * sizeof(float);

        cudaFuncSetAttribute(
            search_multiple_patterns_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes_needed
        );

        search_multiple_patterns_kernel<<<blocksPerGrid, threadsPerBlock, shared_bytes_needed>>>(
            d_h1, d_h2, d_h3, d_h4, d_h5,
            d_results_c1, d_results_c2, d_results_c3, d_results_c4, d_results_c5,
            total_elements);
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        //     return -1;
        // }

        cudaDeviceSynchronize();
    }

    std::vector<float> h_results_c1(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c2(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c3(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c4(total_elements * NUM_QUERIES);
    std::vector<float> h_results_c5(total_elements * NUM_QUERIES);
    std::vector<int> last_best_indices(NUM_QUERIES * 5, -1);
    std::vector<float> min_sads(NUM_QUERIES * 5);
    std::vector<int> best_indices(NUM_QUERIES * 5);

    for (int i = 0; i < NUM_RUNS; ++i) {
        std::fill(min_sads.begin(), min_sads.end(), std::numeric_limits<float>::max());
        std::fill(best_indices.begin(), best_indices.end(), -1);

        dim3 threadsPerBlock(256, 1, 1);
        dim3 blocksPerGrid(
            (total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NUM_QUERIES + threadsPerBlock.y - 1) / threadsPerBlock.y,
            1
            );
        int shared_bytes_needed = 5 * (256 + QUERY_LENGTH) * sizeof(float);

        cudaFuncSetAttribute(
            search_multiple_patterns_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes_needed
        );

        auto time_start = std::chrono::high_resolution_clock::now();
        search_multiple_patterns_kernel<<<blocksPerGrid, threadsPerBlock, shared_bytes_needed>>>(
            d_h1, d_h2, d_h3, d_h4, d_h5,
            d_results_c1, d_results_c2, d_results_c3, d_results_c4, d_results_c5,
            total_elements
        );
        cudaDeviceSynchronize();
        auto time_end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(time_end - time_start).count();
        total_time += time;

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

        for (int q = 0; q < NUM_QUERIES; q++) {
            int offset = q * total_elements;
            for (int c = 0; c < 5; ++c) {
                const int qc = q * 5 + c;
                const std::vector<float>& channel_results = *per_channel_results[c];

                for (int i = 0; i <= total_elements - QUERY_LENGTH; i++) {
                    if (channel_results[offset + i] < min_sads[qc]) {
                        min_sads[qc] = channel_results[offset + i];
                        best_indices[qc] = i;
                    }
                }
            }
        }

        for (int i = 0; i < NUM_QUERIES * 5; ++i) {
            if (best_indices[i] != last_best_indices[i] && last_best_indices[i] != -1) {
                const int query_id = i / 5;
                const int channel_id = i % 5;
                std::cout << "ERRORE: index for query " << query_id + 1
                          << " e channel C" << channel_id + 1
                          << " is different from previous run\n";
            }
            last_best_indices[i] = best_indices[i];
        }
    }

    std::cout << "Average time in " << NUM_RUNS << " runs: " << total_time / NUM_RUNS << "s" << std::endl;
    std::cout << "\n========================= SEARCH RESULTS =========================" << std::endl;

    for (int q = 0; q < NUM_QUERIES; q++) {
        std::cout << "[ QUERY " << q + 1 << " ]" << std::endl;
        for (int c = 0; c < 5; ++c) {
            const int qc = q * 5 + c;
            std::cout << "  - Serie C" << c + 1
                      << " -> indice: " << best_indices[qc]
                      << ", SAD: " << min_sads[qc] << std::endl;
        }
        std::cout << "------------------------------------------------------------------" << std::endl;
    }

    cudaFree(d_h1);    cudaFree(d_h2);    cudaFree(d_h3);    cudaFree(d_h4);    cudaFree(d_h5);
    cudaFree(d_results_c1); cudaFree(d_results_c2); cudaFree(d_results_c3); cudaFree(d_results_c4); cudaFree(d_results_c5);

    return 0;
}