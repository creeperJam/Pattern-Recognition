cmake_minimum_required(VERSION 4.0)
project(Pattern_recognition_cu CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_89")

add_compile_definitions("PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_executable(Pattern_recognition_cu_sad main.cu)
add_executable(Pattern_recognition_cu_fft main-FFT.cu)

target_link_libraries(Pattern_recognition_cu_fft PRIVATE cufft)

set_target_properties(Pattern_recognition_cu_sad PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Pattern_recognition_cu_fft PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


