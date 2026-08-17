#ifndef GEMM_KERNEL_ARM_GEMM_HPP
#define GEMM_KERNEL_ARM_GEMM_HPP

#pragma once


void gemm_naive(	int M, int K, int N,
					float alpha,
					float* A, int lda,
					float* B, int ldb,
					float beta,
					float* C, int ldc);

void gemm_naive_optim(	int M, int K, int N,
					float alpha,
					float* A, int lda,
					float* B, int ldb,
					float beta,
					float* C, int ldc);

void gemm_register_optim(	int M, int K, int N,
							float alpha,
							float* A, int lda,
							float* B, int ldb,
							float beta,
							float* C, int ldc);

void gemm_packed_register_optim(	int M, int K, int N,
									float alpha,
									float* A, int lda,
									float* B, int ldb,
									float beta,
									float* C, int ldc);

void gemm_cache_tiling(				int M, int K, int N,
									float alpha,
									float* A, int lda,
									float* B, int ldb,
									float beta,
									float* C, int ldc);

using GemmFn = void (*) (	int M, int K, int N,
							float alpha,
							float* A, int lda,
							float* B, int ldb,
							float beta,
							float* C, int ldc);

inline constexpr GemmFn gemm = gemm_cache_tiling;
inline constexpr const char* gemm_name = "cache_tiling_optim";


#endif //GEMM_KERNEL_ARM_GEMM_HPP
