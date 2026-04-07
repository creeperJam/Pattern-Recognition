#include <array>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <random>

// La mia RTX 4060 ha 64 KiB di memoria costante, quindi i limiti sulle dimensioni e numero di query devono rispettare questi limiti:
// QUERY_LENGTH * NUM_QUERIES * sizeof(float) * numero_dati(5 in questo caso) <= 65536, QUINDI:
// QUERY_LENGTH * NUM_QUERIES <= 3276 (approssimato per difetto)
constexpr int QUERY_LENGTH = 256;
constexpr int NUM_QUERIES = 4;
constexpr double NUM_RUNS = 100.0f;
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
    float* __restrict__ d_results, int total_elements)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int SHARED_SIZE = BLOCK_SIZE + QUERY_LENGTH;

    __shared__ float s_h1[SHARED_SIZE];
    __shared__ float s_h2[SHARED_SIZE];
    __shared__ float s_h3[SHARED_SIZE];
    __shared__ float s_h4[SHARED_SIZE];
    __shared__ float s_h5[SHARED_SIZE];

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < total_elements) {
        s_h1[threadIdx.x] = d_h1[tid_x];
        s_h2[threadIdx.x] = d_h2[tid_x];
        s_h3[threadIdx.x] = d_h3[tid_x];
        s_h4[threadIdx.x] = d_h4[tid_x];
        s_h5[threadIdx.x] = d_h5[tid_x];
    }

    if (threadIdx.x < QUERY_LENGTH - 1) {
        int halo_index_global = tid_x + BLOCK_SIZE;

        if (halo_index_global < total_elements) {
            s_h1[threadIdx.x + BLOCK_SIZE] = d_h1[halo_index_global];
            s_h2[threadIdx.x + BLOCK_SIZE] = d_h2[halo_index_global];
            s_h3[threadIdx.x + BLOCK_SIZE] = d_h3[halo_index_global];
            s_h4[threadIdx.x + BLOCK_SIZE] = d_h4[halo_index_global];
            s_h5[threadIdx.x + BLOCK_SIZE] = d_h5[halo_index_global];
        }
    }

    __syncthreads();

    if (tid_x <= total_elements - QUERY_LENGTH && tid_y < NUM_QUERIES) {
        float total_sad = 0.0f;
        int q_start = tid_y * QUERY_LENGTH;

        #pragma unroll
        for (int i = 0; i < QUERY_LENGTH; i++) {
            total_sad += fabs(s_h1[threadIdx.x + i] - c_q1[q_start + i]);
            total_sad += fabs(s_h2[threadIdx.x + i] - c_q2[q_start + i]);
            total_sad += fabs(s_h3[threadIdx.x + i] - c_q3[q_start + i]);
            total_sad += fabs(s_h4[threadIdx.x + i] - c_q4[q_start + i]);
            total_sad += fabs(s_h5[threadIdx.x + i] - c_q5[q_start + i]);
        }

        d_results[tid_y * total_elements + tid_x] = total_sad;
    }
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
            all_queries_c1[offset + i] = loaded_data.c1[base_idx + i] + noise(generator);
            all_queries_c2[offset + i] = loaded_data.c2[base_idx + i] + noise(generator);
            all_queries_c3[offset + i] = loaded_data.c3[base_idx + i] + noise(generator);
            all_queries_c4[offset + i] = loaded_data.c4[base_idx + i] + noise(generator);
            all_queries_c5[offset + i] = loaded_data.c5[base_idx + i] + noise(generator);
        }
    }

    float *d_h1;
    float *d_h2;
    float *d_h3;
    float *d_h4;
    float *d_h5;
    float *d_results;

    cudaMalloc(&d_h1, total_elements * sizeof(float));
    cudaMalloc(&d_h2, total_elements * sizeof(float));
    cudaMalloc(&d_h3, total_elements * sizeof(float));
    cudaMalloc(&d_h4, total_elements * sizeof(float));
    cudaMalloc(&d_h5, total_elements * sizeof(float));

    cudaMalloc(&d_results, total_elements * sizeof(float) * NUM_QUERIES);

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

        search_multiple_patterns_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_h1, d_h2, d_h3, d_h4, d_h5,
            d_results,
            total_elements);
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        //     return -1;
        // }

        cudaDeviceSynchronize();
    }

    std::vector<float> h_results(total_elements * NUM_QUERIES);
    std::vector<int> last_best_indices(NUM_QUERIES, -1);
    std::vector<float> min_sads(NUM_QUERIES);
    std::vector<int> best_indices(NUM_QUERIES);

    for (int i = 0; i < NUM_RUNS; ++i) {
        std::fill(min_sads.begin(), min_sads.end(), std::numeric_limits<float>::max());
        std::fill(best_indices.begin(), best_indices.end(), -1);

        dim3 threadsPerBlock(256, 1, 1);
        dim3 blocksPerGrid(
            (total_elements + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NUM_QUERIES + threadsPerBlock.y - 1) / threadsPerBlock.y,
            1
        );

        auto time_start = std::chrono::high_resolution_clock::now();
        search_multiple_patterns_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_h1, d_h2, d_h3, d_h4, d_h5,
            d_results,
            total_elements
        );
        cudaDeviceSynchronize();
        auto time_end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(time_end - time_start).count();
        total_time += time;

        cudaMemcpy(h_results.data(), d_results, total_elements * NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);

        for (int q = 0; q < NUM_QUERIES; q++) {
            int offset = q * total_elements;

            for (int i = 0; i <= total_elements - QUERY_LENGTH; i++) {
                if (h_results[offset + i] < min_sads[q]) {
                    min_sads[q] = h_results[offset + i];
                    best_indices[q] = i;
                }
            }
        }

        for (int i = 0; i < NUM_QUERIES; ++i) {
            if (best_indices[i] != last_best_indices[i] && last_best_indices[i] != -1) {
                std::cout << "ERRORE: index for the " << i+1 << "-th query is different. Different result from the expected\n";
                last_best_indices[i] = best_indices[i];
            }
        }
    }

    std::cout << "Average time in " << NUM_RUNS << " runs: " << total_time / NUM_RUNS << "ms" << std::endl;
    std::cout << "\n========================= SEARCH RESULTS =========================" << std::endl;

    for (int q = 0; q < NUM_QUERIES; q++) {
        std::cout << "[ QUERY " << q + 1 << " ]" << std::endl;
        std::cout << "  -> Match migliore all'indice: " << best_indices[q] << std::endl;
        std::cout << "  -> Valore di SAD (errore)   : " << min_sads[q] << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
    }

    cudaFree(d_h1);    cudaFree(d_h2);    cudaFree(d_h3);    cudaFree(d_h4);    cudaFree(d_h5);
    cudaFree(d_results);

    return 0;
}