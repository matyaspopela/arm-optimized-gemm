#ifndef GEMM_KERNEL_ARM_SIMD_MICRO_8X4_H
#define GEMM_KERNEL_ARM_SIMD_MICRO_8X4_H

#include <arm_neon.h>

// A, B are pre packed into contiguous buffers
void simd_micro_8x4(	int Nr, int Kc, int Mr, float* __restrict B,
						float* __restrict A, float* __restrict C, int ldc, bool is_first_k_block) {

	float32x4_t c0 = vdupq_n_f32(0.f);
	float32x4_t c1 = vdupq_n_f32(0.f);
	float32x4_t c2 = vdupq_n_f32(0.f);
	float32x4_t c3 = vdupq_n_f32(0.f);
	float32x4_t c4 = vdupq_n_f32(0.f);
	float32x4_t c5 = vdupq_n_f32(0.f);
	float32x4_t c6 = vdupq_n_f32(0.f);
	float32x4_t c7 = vdupq_n_f32(0.f);

	for (int i = 0; i < Kc; i += 4) {

		float32x4_t a1 = vld1q_f32(A + 0);
		float32x4_t a2 = vld1q_f32(A + 4);

		float32x4_t b = vld1q_f32(B + 0);

		c0 = vfmaq_laneq_f32(c0, b, a1, 0);
		c1 = vfmaq_laneq_f32(c1, b, a1, 1);
		c2 = vfmaq_laneq_f32(c2, b, a1, 2);
		c3 = vfmaq_laneq_f32(c3, b, a1, 3);
		c4 = vfmaq_laneq_f32(c4, b, a2, 0);
		c5 = vfmaq_laneq_f32(c5, b, a2, 1);
		c6 = vfmaq_laneq_f32(c6, b, a2, 2);
		c7 = vfmaq_laneq_f32(c7, b, a2, 3);

		a1 = vld1q_f32(A + 8);
		a2 = vld1q_f32(A + 12);

		b = vld1q_f32(B + 4);


		c0 = vfmaq_laneq_f32(c0, b, a1, 0);
		c1 = vfmaq_laneq_f32(c1, b, a1, 1);
		c2 = vfmaq_laneq_f32(c2, b, a1, 2);
		c3 = vfmaq_laneq_f32(c3, b, a1, 3);
		c4 = vfmaq_laneq_f32(c4, b, a2, 0);
		c5 = vfmaq_laneq_f32(c5, b, a2, 1);
		c6 = vfmaq_laneq_f32(c6, b, a2, 2);
		c7 = vfmaq_laneq_f32(c7, b, a2, 3);

		a1 = vld1q_f32(A + 16);
		a2 = vld1q_f32(A + 20);

		b = vld1q_f32(B + 8);

		c0 = vfmaq_laneq_f32(c0, b, a1, 0);
		c1 = vfmaq_laneq_f32(c1, b, a1, 1);
		c2 = vfmaq_laneq_f32(c2, b, a1, 2);
		c3 = vfmaq_laneq_f32(c3, b, a1, 3);
		c4 = vfmaq_laneq_f32(c4, b, a2, 0);
		c5 = vfmaq_laneq_f32(c5, b, a2, 1);
		c6 = vfmaq_laneq_f32(c6, b, a2, 2);
		c7 = vfmaq_laneq_f32(c7, b, a2, 3);

		a1 = vld1q_f32(A + 24);
		a2 = vld1q_f32(A + 28);

		b = vld1q_f32(B + 12);

		c0 = vfmaq_laneq_f32(c0, b, a1, 0);
		c1 = vfmaq_laneq_f32(c1, b, a1, 1);
		c2 = vfmaq_laneq_f32(c2, b, a1, 2);
		c3 = vfmaq_laneq_f32(c3, b, a1, 3);
		c4 = vfmaq_laneq_f32(c4, b, a2, 0);
		c5 = vfmaq_laneq_f32(c5, b, a2, 1);
		c6 = vfmaq_laneq_f32(c6, b, a2, 2);
		c7 = vfmaq_laneq_f32(c7, b, a2, 3);

		A += 32;
		B += 16;

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

