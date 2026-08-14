#include "../include/mlib/gemm.hpp"
#include <cstddef>



void gemm_naive(	int M, int K, int N,
			float alpha,
			float* A, int lda,
			float* B, int ldb,
			float beta,
			float* C, int ldc) {

	// A is M x K
	// B is K x N
	// C is M x N

	for (size_t i = 0; i < M; i++ ) {
		for (size_t j = 0; j < N; j++ ) {
			float sum = 0;
			for (size_t k = 0; k < K; k++ ) { // version one - deliberately missing CPU cache
				sum += (A[i * lda + k] * B[k * ldb + j]);
			}
			if (beta == 0) {
				C[i * ldc + j] = alpha * sum;
			} else {
				C[i * ldc + j] = alpha * sum + beta*C[i * ldc + j];
			}
		}
	}
}
