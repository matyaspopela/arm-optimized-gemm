cmake_minimum_required(VERSION 3.28)
project(gemm_kernel_arm)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_executable(eval
        eval/problem.hpp
        eval/buffer.hpp
        eval/validate.cpp
        kernels/gemm_naive.cpp
        kernels/gemm_naive_optim.cpp
        kernels/gemm_register_optim.cpp
        kernels/scalar_micro_4x4.h
        kernels/simd_micro_4x4.h
        kernels/packed_simd_micro.h
        kernels/gemm_packed_register.cpp
)


add_executable(microtests           scripts/microtests.cpp)


target_link_libraries(eval PRIVATE Eigen3::Eigen)