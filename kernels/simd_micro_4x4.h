#ifndef GEMM_KERNEL_ARM_MICRO_4X4_H
#define GEMM_KERNEL_ARM_MICRO_4X4_H
#include <arm_neon.h>

static inline void simd_micro_4x4(	int K, float alpha, float beta,
								const float* __restrict A, int lda,
								const float* __restrict B, int ldb,
								float* __restrict C, int ldc		) {

	float32x4_t c0;
	float32x4_t c1;
	float32x4_t c2;
	float32x4_t c3;

	// fill registry
	c0 = vdupq_n_f32(0.0);
	c1 = vdupq_n_f32(0.0);
	c2 = vdupq_n_f32(0.0);
	c3 = vdupq_n_f32(0.0);


	for (int k=0; k<K; ++k) {

		float32x4_t a0 = vdupq_n_f32(A[0*lda + k]);
		float32x4_t a1 = vdupq_n_f32(A[1*lda + k]);
		float32x4_t a2 = vdupq_n_f32(A[2*lda + k]);
		float32x4_t a3 = vdupq_n_f32(A[3*lda + k]);

		float32x4_t b = vld1q_f32(&B[k*ldb]);

		c0 = vfmaq_f32(c0, a0, b);
		c1 = vfmaq_f32(c1, a1, b);
		c2 = vfmaq_f32(c2, a2, b);
		c3 = vfmaq_f32(c3, a3, b);

	}
	float32x4_t vec_alpha = vdupq_n_f32(alpha);

	c0 = vmulq_f32(c0, vec_alpha);
	c1 = vmulq_f32(c1, vec_alpha);
	c2 = vmulq_f32(c2, vec_alpha);
	c3 = vmulq_f32(c3, vec_alpha);

	// fill back to C
	if (beta == 0.0) {
		vst1q_f32(&C[0*ldc], c0);
		vst1q_f32(&C[1*ldc], c1);
		vst1q_f32(&C[2*ldc], c2);
		vst1q_f32(&C[3*ldc], c3);
	} else {
		float32x4_t vec_beta = vdupq_n_f32(beta);

		float32x4_t c0_orig = vld1q_f32(&C[0*ldc]);
		float32x4_t c1_orig = vld1q_f32(&C[1*ldc]);
		float32x4_t c2_orig = vld1q_f32(&C[2*ldc]);
		float32x4_t c3_orig = vld1q_f32(&C[3*ldc]);

		c0 = vfmaq_f32(c0, c0_orig, vec_beta);
		c1 = vfmaq_f32(c1, c1_orig, vec_beta);
		c2 = vfmaq_f32(c2, c2_orig, vec_beta);
		c3 = vfmaq_f32(c3, c3_orig, vec_beta);

		vst1q_f32(&C[0*ldc], c0);
		vst1q_f32(&C[1*ldc], c1);
		vst1q_f32(&C[2*ldc], c2);
		vst1q_f32(&C[3*ldc], c3);
	}
}


#endif //GEMM_KERNEL_ARM_MICRO_4X4_H
