cmake_minimum_required(VERSION 4.0)
project(pattern_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native -fopenmp")

add_compile_definitions("PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

add_executable(pattern_recognition
        main.cpp
        sequential.cpp
        parallel.cpp
        utilities.cpp
        utilities.h
)