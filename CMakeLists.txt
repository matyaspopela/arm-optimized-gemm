cmake_minimum_required(VERSION 3.28)
project(gemm_kernel_arm)

set(CMAKE_CXX_STANDARD 20)

add_executable(gemm_kernel_arm main.cpp
        include/mlib/gemm.hpp
        kernels/_kern.cpp
        eval/problem.hpp
        eval/buffer.hpp)
