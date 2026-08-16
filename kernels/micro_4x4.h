//
// Created by matya on 8/14/2026.
//

#ifndef GEMM_KERNEL_ARM_MICRO_4X4_H

static inline void micro_4x4(	int K, float alpha, float beta,
								const float* __restrict A, int lda,
								const float* __restrict B, int ldb,
								float* __restrict C, int ldc		) {


	float c00=0, c01=0, c02=0, c03=0;
	float c10=0, c11=0, c12=0, c13=0;
	float c20=0, c21=0, c22=0, c23=0;
	float c30=0, c31=0, c32=0, c33=0;

	for (int k = 0; k < K; ++k) {

		float a0 = A[0*lda + k];
		float a1 = A[1*lda + k];
		float a2 = A[2*lda + k];
		float a3 = A[3*lda + k];

		float b0 = B[k*ldb + 0];
		float b1 = B[k*ldb + 1];
		float b2 = B[k*ldb + 2];
		float b3 = B[k*ldb + 3];

		c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
		c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
		c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
		c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
	}

	C[0 * ldc + 0] = c00; C[0 * ldc + 1] = c01; C[0 * ldc + 2] = c02; C[0 * ldc + 3] = c03;
	C[1 * ldc + 0] = c10; C[1 * ldc + 1] = c11; C[1 * ldc + 2] = c12; C[1 * ldc + 3] = c13;
	C[2 * ldc + 0] = c20; C[2 * ldc + 1] = c21; C[2 * ldc + 2] = c22; C[2 * ldc + 3] = c23;
	C[3 * ldc + 0] = c30; C[3 * ldc + 1] = c31; C[3 * ldc + 2] = c32; C[3 * ldc + 3] = c33;

}








#define GEMM_KERNEL_ARM_MICRO_4X4_H

#endif //GEMM_KERNEL_ARM_MICRO_4X4_H
