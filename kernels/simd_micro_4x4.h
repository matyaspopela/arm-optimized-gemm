//
// Created by matya on 8/14/2026.
//

#ifndef GEMM_KERNEL_ARM_MICRO_4X4_H
#define GEMM_KERNEL_ARM_MICRO_4X4_H

static inline void simd_micro_4x4(	int K, float alpha, float beta,
								const float* __restrict A, int lda,
								const float* __restrict B, int ldb,
								float* __restrict C, int ldc		) {


}


#endif //GEMM_KERNEL_ARM_MICRO_4X4_H
