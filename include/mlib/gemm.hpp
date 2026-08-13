#ifndef GEMM_KERNEL_ARM_GEMM_HPP
#define GEMM_KERNEL_ARM_GEMM_HPP

#pragma once

//C = alpha * (A@B) + beta*C
void gemm(	int M, int K, int N,
			float alpha,
			float* A, int lda,
			float* B, int ldb,
			float beta,
			float* C, int ldc);
// A.shape = M,K : blended in per alpha
// B.shape = K,N ^^

// C.shape = M,N : blended in per beta

#endif //GEMM_KERNEL_ARM_GEMM_HPP
