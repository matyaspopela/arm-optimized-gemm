#include "../include/mlib/gemm.hpp"
#include <cstddef>



void gemm_naive_optim(	int M, int K, int N,
						float alpha,
						float* A, int lda,
						float* B, int ldb,
						float beta,
						float* C, int ldc) {

	// A is M x K
	// B is K x N
	// C is M x N

	for (size_t i = 0; i < M; ++i) {

		//seed row first (initilized to NaN's)
		if (beta == 0.0) {
			for (size_t j = 0; j < N; ++j) {
				C[i * ldc + j] = 0.0;
			}
		} else {
			for (size_t j = 0; j < N; ++j) {
				C[i * ldc + j] = beta * C[i * ldc + j];
			}
		}

		for (size_t k = 0; k < K; ++k) {
			for (size_t j = 0; j < N; ++j) {
				C[i * ldc + j] += alpha * (A[i * lda + j] * B[i * ldb + k]);
			}
		}
	}






	for (size_t i = 0; i < M; i++ ) {
		for (size_t j = 0; j < N; j++ ) {
			float sum = 0;
			for (size_t k = 0; k < K; k++ ) {
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

