#include "packed_simd_micro.h"
#include <stdexcept>
#include <arm_neon.h>

// all functions assume pre-padded input so that it is divisible by 4!

static inline void pack_A_4x4(int K, float* A, int lda, float* packed_A) {

	if (K%4 != 0) {throw std::runtime_error("pack_A_4x4: K must be multiple of 4");}

	float* out = packed_A;

	float* a0 = A + 0*lda;
	float* a1 = A + 1*lda;
	float* a2 = A + 2*lda;
	float* a3 = A + 3*lda;

	for (int i = 0; i < K; ++i) { //marching pointers
		*out++ = *a0++;
		*out++ = *a1++;
		*out++ = *a2++;
		*out++ = *a3++;
	}

};

static inline void pack_B_4x4(int K, float* B, int ldb, float* packed_B) { //simd for this one

	if (K%4 != 0) {throw std::runtime_error("pack_B_4x4: K must be multiple of 4");}

	float* out = packed_B;

	for (int k = 0; k < K; ++k) {
		float32x4_t row = vld1q_f32(&B[k*ldb]); //vld1q_f32 reads 4 floats exactly.
		vst1q_f32(out, row);
		out += 4;
	}
};

void gemm_packed_register_optim(	int M, int K, int N,
									float alpha,
									float* A, int lda,
									float* B, int ldb,
									float beta,
									float* C, int ldc) {

	if (K%4 != 0) {throw std::runtime_error("gemm_packed_register_optim: K must be multiple of 4");}

	for (int i = 0; i < M; i+=4) {

		float packed_A[4*K];
		pack_A_4x4(K, A + i*lda,	lda, packed_A);

		for (int j = 0; j < N; j+=4) {

			float packed_B[4*K];
			pack_B_4x4(K, B + j,		ldb, packed_B);

			packed_simd_micro_4x4(		K, alpha, beta,
										packed_A, lda,
										packed_B, ldb,
										C + i * ldc + j, ldc);

		}
	}



}




