cmake_minimum_required(VERSION 3.16)
project(Reduction CUDA C)

set(CMAKE_CUDA_STANDARD 14)
add_compile_options(-O3)
add_executable(Reduction main.cu reductionGPU.h reductionGPU.cu Mytime.h myTime.c)

set_target_properties(
        Reduction
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)