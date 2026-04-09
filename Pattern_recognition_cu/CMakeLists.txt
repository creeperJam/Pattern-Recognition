cmake_minimum_required(VERSION 4.0)
project(Pattern_recognition_cu LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /arch:AVX2")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_89")
set(CMAKE_CUDA_ARCHITECTURES 89)

add_compile_definitions("PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_executable(Pattern_recognition_cu_sad
        main.cu main.cuh
        utilities.cpp utilities.h)

target_compile_options(Pattern_recognition_cu_sad PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xcompiler /O2
        --ptxas-options=-v
        >
)
#add_executable(Pattern_recognition_cu_fft main-FFT.cu)

#target_link_libraries(Pattern_recognition_cu_fft PRIVATE cufft)
