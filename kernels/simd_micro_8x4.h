#ifndef GEMM_KERNEL_ARM_SIMD_MICRO_8X4_H
#define GEMM_KERNEL_ARM_SIMD_MICRO_8X4_H

#include <arm_neon.h>

// A, B are pre packed into contiguous buffers
void simd_micro_8x4(	int Nr, int Kc, int Mr, float* B,
						float* A, float* C, int ldc, bool is_first_k_block) {

	float32x4_t c0 = vdupq_n_f32(0.f);
	float32x4_t c1 = vdupq_n_f32(0.f);
	float32x4_t c2 = vdupq_n_f32(0.f);
	float32x4_t c3 = vdupq_n_f32(0.f);
	float32x4_t c4 = vdupq_n_f32(0.f);
	float32x4_t c5 = vdupq_n_f32(0.f);
	float32x4_t c6 = vdupq_n_f32(0.f);
	float32x4_t c7 = vdupq_n_f32(0.f);

	for (int i = 0; i < Kc; i ++) {

		float32x4_t a0 = vdupq_n_f32(A[0 + i*Mr]);
		float32x4_t a1 = vdupq_n_f32(A[1 + i*Mr]);
		float32x4_t a2 = vdupq_n_f32(A[2 + i*Mr]);
		float32x4_t a3 = vdupq_n_f32(A[3 + i*Mr]);
		float32x4_t a4 = vdupq_n_f32(A[4 + i*Mr]);
		float32x4_t a5 = vdupq_n_f32(A[5 + i*Mr]);
		float32x4_t a6 = vdupq_n_f32(A[6 + i*Mr]);
		float32x4_t a7 = vdupq_n_f32(A[7 + i*Mr]);

		float32x4_t b = vld1q_f32(B);
		B+=Nr;

		c0 = vfmaq_f32(c0, a0, b);
		c1 = vfmaq_f32(c1, a1, b);
		c2 = vfmaq_f32(c2, a2, b);
		c3 = vfmaq_f32(c3, a3, b);
		c4 = vfmaq_f32(c4, a4, b);
		c5 = vfmaq_f32(c5, a5, b);
		c6 = vfmaq_f32(c6, a6, b);
		c7 = vfmaq_f32(c7, a7, b);
	}

	if (is_first_k_block) {
		vst1q_f32(&C[0 * ldc], c0);
		vst1q_f32(&C[1 * ldc], c1);
		vst1q_f32(&C[2 * ldc], c2);
		vst1q_f32(&C[3 * ldc], c3);
		vst1q_f32(&C[4 * ldc], c4);
		vst1q_f32(&C[5 * ldc], c5);
		vst1q_f32(&C[6 * ldc], c6);
		vst1q_f32(&C[7 * ldc], c7);
	} else {
		float32x4_t c0orig = vld1q_f32(&C[0 * ldc]);
		float32x4_t c1orig = vld1q_f32(&C[1 * ldc]);
		float32x4_t c2orig = vld1q_f32(&C[2 * ldc]);
		float32x4_t c3orig = vld1q_f32(&C[3 * ldc]);
		float32x4_t c4orig = vld1q_f32(&C[4 * ldc]);
		float32x4_t c5orig = vld1q_f32(&C[5 * ldc]);
		float32x4_t c6orig = vld1q_f32(&C[6 * ldc]);
		float32x4_t c7orig = vld1q_f32(&C[7 * ldc]);

		c0 = vaddq_f32(c0, c0orig);
		c1 = vaddq_f32(c1, c1orig);
		c2 = vaddq_f32(c2, c2orig);
		c3 = vaddq_f32(c3, c3orig);
		c4 = vaddq_f32(c4, c4orig);
		c5 = vaddq_f32(c5, c5orig);
		c6 = vaddq_f32(c6, c6orig);
		c7 = vaddq_f32(c7, c7orig);

		vst1q_f32(&C[0 * ldc], c0);
		vst1q_f32(&C[1 * ldc], c1);
		vst1q_f32(&C[2 * ldc], c2);
		vst1q_f32(&C[3 * ldc], c3);
		vst1q_f32(&C[4 * ldc], c4);
		vst1q_f32(&C[5 * ldc], c5);
		vst1q_f32(&C[6 * ldc], c6);
		vst1q_f32(&C[7 * ldc], c7);
	}
}

#endif //GEMM_KERNEL_ARM_SIMD_MICRO_8X4_H
