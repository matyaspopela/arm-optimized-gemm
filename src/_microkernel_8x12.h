#ifndef GEMM_KERNEL_ARM_SIMD_MICROKERNEL_8X12_H
#define GEMM_KERNEL_ARM_SIMD_MICROKERNEL_8X12_H

#include <arm_neon.h>


#ifndef GEMM_KERNEL_ARM_SIMD_MICRO_8X12_H
#define GEMM_KERNEL_ARM_SIMD_MICRO_8X12_H

#include <arm_neon.h>


static inline void _microkernel_8x12(		int Nr, int Kc, int Mr, float* __restrict B,
										float* __restrict A, float* __restrict C, int ldc, bool is_first_k_block) {

	float32x4_t c00 = vdupq_n_f32(0.0); float32x4_t c01 = vdupq_n_f32(0.0); float32x4_t c02 = vdupq_n_f32(0.0);
	float32x4_t c10 = vdupq_n_f32(0.0); float32x4_t c11 = vdupq_n_f32(0.0); float32x4_t c12 = vdupq_n_f32(0.0);
	float32x4_t c20 = vdupq_n_f32(0.0); float32x4_t c21 = vdupq_n_f32(0.0); float32x4_t c22 = vdupq_n_f32(0.0);
	float32x4_t c30 = vdupq_n_f32(0.0); float32x4_t c31 = vdupq_n_f32(0.0); float32x4_t c32 = vdupq_n_f32(0.0);
	float32x4_t c40 = vdupq_n_f32(0.0); float32x4_t c41 = vdupq_n_f32(0.0); float32x4_t c42 = vdupq_n_f32(0.0);
	float32x4_t c50 = vdupq_n_f32(0.0); float32x4_t c51 = vdupq_n_f32(0.0); float32x4_t c52 = vdupq_n_f32(0.0);
	float32x4_t c60 = vdupq_n_f32(0.0); float32x4_t c61 = vdupq_n_f32(0.0); float32x4_t c62 = vdupq_n_f32(0.0);
	float32x4_t c70 = vdupq_n_f32(0.0); float32x4_t c71 = vdupq_n_f32(0.0); float32x4_t c72 = vdupq_n_f32(0.0);

	int i = 0;

	for (; i < Kc; i++) {

		float32x4_t a1 = vld1q_f32(A + 0);
		float32x4_t a2 = vld1q_f32(A + 4);

		float32x4_t b1 = vld1q_f32(B + 0);
		float32x4_t b2 = vld1q_f32(B + 4);
		float32x4_t b3 = vld1q_f32(B + 8);

		c00 = vfmaq_laneq_f32(c00, b1, a1, 0); c01 = vfmaq_laneq_f32(c01, b2, a1, 0); c02 = vfmaq_laneq_f32(c02, b3, a1, 0);
		c10 = vfmaq_laneq_f32(c10, b1, a1, 1); c11 = vfmaq_laneq_f32(c11, b2, a1, 1); c12 = vfmaq_laneq_f32(c12, b3, a1, 1);
		c20 = vfmaq_laneq_f32(c20, b1, a1, 2); c21 = vfmaq_laneq_f32(c21, b2, a1, 2); c22 = vfmaq_laneq_f32(c22, b3, a1, 2);
		c30 = vfmaq_laneq_f32(c30, b1, a1, 3); c31 = vfmaq_laneq_f32(c31, b2, a1, 3); c32 = vfmaq_laneq_f32(c32, b3, a1, 3);
		c40 = vfmaq_laneq_f32(c40, b1, a2, 0); c41 = vfmaq_laneq_f32(c41, b2, a2, 0); c42 = vfmaq_laneq_f32(c42, b3, a2, 0);
		c50 = vfmaq_laneq_f32(c50, b1, a2, 1); c51 = vfmaq_laneq_f32(c51, b2, a2, 1); c52 = vfmaq_laneq_f32(c52, b3, a2, 1);
		c60 = vfmaq_laneq_f32(c60, b1, a2, 2); c61 = vfmaq_laneq_f32(c61, b2, a2, 2); c62 = vfmaq_laneq_f32(c62, b3, a2, 2);
		c70 = vfmaq_laneq_f32(c70, b1, a2, 3); c71 = vfmaq_laneq_f32(c71, b2, a2, 3); c72 = vfmaq_laneq_f32(c72, b3, a2, 3);

		A += 8;
		B += 12;
	}

	float tmp[8][12];

	if (is_first_k_block) {

		vst1q_f32(&tmp[0][0], c00); vst1q_f32(&tmp[0][4], c01); vst1q_f32(&tmp[0][8], c02);
		vst1q_f32(&tmp[1][0], c10); vst1q_f32(&tmp[1][4], c11); vst1q_f32(&tmp[1][8], c12);
		vst1q_f32(&tmp[2][0], c20); vst1q_f32(&tmp[2][4], c21); vst1q_f32(&tmp[2][8], c22);
		vst1q_f32(&tmp[3][0], c30); vst1q_f32(&tmp[3][4], c31); vst1q_f32(&tmp[3][8], c32);
		vst1q_f32(&tmp[4][0], c40); vst1q_f32(&tmp[4][4], c41); vst1q_f32(&tmp[4][8], c42);
		vst1q_f32(&tmp[5][0], c50); vst1q_f32(&tmp[5][4], c51); vst1q_f32(&tmp[5][8], c52);
		vst1q_f32(&tmp[6][0], c60); vst1q_f32(&tmp[6][4], c61); vst1q_f32(&tmp[6][8], c62);
		vst1q_f32(&tmp[7][0], c70); vst1q_f32(&tmp[7][4], c71); vst1q_f32(&tmp[7][8], c72);

		for (int r = 0; r < Mr; ++r) {
			for (int c = 0; c < Nr; ++c) {
				C[r*ldc + c] = tmp[r][c];
			}
		}

	} else {

		float32x4_t c00orig = vld1q_f32(&C[0 * ldc]); float32x4_t c01orig = vld1q_f32(&C[0 * ldc + 4]); float32x4_t c02orig = vld1q_f32(&C[0 * ldc + 8]);
		float32x4_t c10orig = vld1q_f32(&C[1 * ldc]); float32x4_t c11orig = vld1q_f32(&C[1 * ldc + 4]); float32x4_t c12orig = vld1q_f32(&C[1 * ldc + 8]);
		float32x4_t c20orig = vld1q_f32(&C[2 * ldc]); float32x4_t c21orig = vld1q_f32(&C[2 * ldc + 4]); float32x4_t c22orig = vld1q_f32(&C[2 * ldc + 8]);
		float32x4_t c30orig = vld1q_f32(&C[3 * ldc]); float32x4_t c31orig = vld1q_f32(&C[3 * ldc + 4]); float32x4_t c32orig = vld1q_f32(&C[3 * ldc + 8]);
		float32x4_t c40orig = vld1q_f32(&C[4 * ldc]); float32x4_t c41orig = vld1q_f32(&C[4 * ldc + 4]); float32x4_t c42orig = vld1q_f32(&C[4 * ldc + 8]);
		float32x4_t c50orig = vld1q_f32(&C[5 * ldc]); float32x4_t c51orig = vld1q_f32(&C[5 * ldc + 4]); float32x4_t c52orig = vld1q_f32(&C[5 * ldc + 8]);
		float32x4_t c60orig = vld1q_f32(&C[6 * ldc]); float32x4_t c61orig = vld1q_f32(&C[6 * ldc + 4]); float32x4_t c62orig = vld1q_f32(&C[6 * ldc + 8]);
		float32x4_t c70orig = vld1q_f32(&C[7 * ldc]); float32x4_t c71orig = vld1q_f32(&C[7 * ldc + 4]); float32x4_t c72orig = vld1q_f32(&C[7 * ldc + 8]);

		c00 = vaddq_f32(c00orig, c00); c01 = vaddq_f32(c01orig, c01); c02 = vaddq_f32(c02orig, c02);
		c10 = vaddq_f32(c10orig, c10); c11 = vaddq_f32(c11orig, c11); c12 = vaddq_f32(c12orig, c12);
		c20 = vaddq_f32(c20orig, c20); c21 = vaddq_f32(c21orig, c21); c22 = vaddq_f32(c22orig, c22);
		c30 = vaddq_f32(c30orig, c30); c31 = vaddq_f32(c31orig, c31); c32 = vaddq_f32(c32orig, c32);
		c40 = vaddq_f32(c40orig, c40); c41 = vaddq_f32(c41orig, c41); c42 = vaddq_f32(c42orig, c42);
		c50 = vaddq_f32(c50orig, c50); c51 = vaddq_f32(c51orig, c51); c52 = vaddq_f32(c52orig, c52);
		c60 = vaddq_f32(c60orig, c60); c61 = vaddq_f32(c61orig, c61); c62 = vaddq_f32(c62orig, c62);
		c70 = vaddq_f32(c70orig, c70); c71 = vaddq_f32(c71orig, c71); c72 = vaddq_f32(c72orig, c72);

		vst1q_f32(&tmp[0][0], c00); vst1q_f32(&tmp[0][4], c01); vst1q_f32(&tmp[0][8], c02);
		vst1q_f32(&tmp[1][0], c10); vst1q_f32(&tmp[1][4], c11); vst1q_f32(&tmp[1][8], c12);
		vst1q_f32(&tmp[2][0], c20); vst1q_f32(&tmp[2][4], c21); vst1q_f32(&tmp[2][8], c22);
		vst1q_f32(&tmp[3][0], c30); vst1q_f32(&tmp[3][4], c31); vst1q_f32(&tmp[3][8], c32);
		vst1q_f32(&tmp[4][0], c40); vst1q_f32(&tmp[4][4], c41); vst1q_f32(&tmp[4][8], c42);
		vst1q_f32(&tmp[5][0], c50); vst1q_f32(&tmp[5][4], c51); vst1q_f32(&tmp[5][8], c52);
		vst1q_f32(&tmp[6][0], c60); vst1q_f32(&tmp[6][4], c61); vst1q_f32(&tmp[6][8], c62);
		vst1q_f32(&tmp[7][0], c70); vst1q_f32(&tmp[7][4], c71); vst1q_f32(&tmp[7][8], c72);

		for (int r = 0; r < Mr; ++r) {
			for (int c = 0; c < Nr; ++c) {
				C[r*ldc + c] = tmp[r][c];
			}
		}

	}
}



#endif //GEMM_KERNEL_ARM_SIMD_MICRO_8X12_H




#endif //GEMM_KERNEL_ARM_SIMD_MICROKERNEL_8X12_H
