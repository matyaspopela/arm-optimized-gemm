#include "micro_4x4.h"
#include "../include/mlib/gemm.hpp"

//input has to be padded with zeros to be divisible by 4 on both dimensions.

void gemm_register_optim(	int M, int K, int N,
							float alpha,
							float* A, int lda,
							float* B, int ldb,
							float beta,
							float* C, int ldc) {

	for (int i = 0; i < M; i+= 4) {
		for (int j = 0; j < N; j+= 4) {

			micro_4x4(			K, alpha, beta,
								A + i * lda,			lda,
								B + j,					ldb,
								C + i * ldc + j,		ldc);

		}
	}

}