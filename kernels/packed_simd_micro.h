#ifndef GEMM_KERNEL_ARM_MICRO_4X4_H
#define GEMM_KERNEL_ARM_MICRO_4X4_H
#include <arm_neon.h>


// note: has to accept lda, ldb to save me time in validation (same format as other gemms)
// this way i can just use the GemmFn function pointer
static inline void packed_simd_micro_4x4(	int K, float alpha, float beta,
								const float* __restrict packed_A, int lda,
								const float* __restrict packed_B, int ldb,
								float* __restrict C, int ldc		) {

	// fill registry
	float32x4_t c0 = vdupq_n_f32(0.0);
	float32x4_t c1 = vdupq_n_f32(0.0);
	float32x4_t c2 = vdupq_n_f32(0.0);
	float32x4_t c3 = vdupq_n_f32(0.0);

	for (int k=0; k<K; ++k) {

		float32x4_t a0 = vdupq_n_f32(packed_A[0]);
		float32x4_t a1 = vdupq_n_f32(packed_A[1]);
		float32x4_t a2 = vdupq_n_f32(packed_A[2]);
		float32x4_t a3 = vdupq_n_f32(packed_A[3]);
		packed_A += 4;

		float32x4_t b = vld1q_f32(packed_B);
		packed_B += 4;

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
