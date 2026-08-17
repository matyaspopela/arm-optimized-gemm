#include <stdexcept>
#include <arm_neon.h>
#include <cstddef>
#include "simd_micro_8x4.h"
#include <vector>
#include <algorithm>
#include "simd_micro_8x12.h"


// at packing point Bp is (Kc, Nc)
static inline void pack_B(int kc, int nc, const float* __restrict B, int ldb,
						  float* __restrict out)
{
	for (int j0 = 0; j0 < nc; j0 += NR) {
		for (int k = 0; k < kc; ++k) {
			const float* src = &B[k*ldb + j0];
			vst1q_f32(out + 0, vld1q_f32(src + 0));
			vst1q_f32(out + 4, vld1q_f32(src + 4));
			vst1q_f32(out + 8, vld1q_f32(src + 8));
			out += NR;
		}
	}
}

// at packing point A is (Mc, Kc)
static inline void pack_A(int Mc, int Kc, int Mr, float* A, int lda, float* out) {

	for (size_t i = 0; i < Mc; i += Mr) {
		// declare ptrs 0..7
		float* a0 = A + i * lda;
		float* a1 = A + (i+1) * lda;
		float* a2 = A + (i+2) * lda;
		float* a3 = A + (i+3) * lda;
		float* a4 = A + (i+4) * lda;
		float* a5 = A + (i+5) * lda;
		float* a6 = A + (i+6) * lda;
		float* a7 = A + (i+7) * lda;

		for (size_t k = 0; k < Kc; ++k) {
			// march ptrs crazy style
			*out++ = *a0++;
			*out++ = *a1++;
			*out++ = *a2++;
			*out++ = *a3++;
			*out++ = *a4++;
			*out++ = *a5++;
			*out++ = *a6++;
			*out++ = *a7++;
		}
	}
}

void gemm_cache_tiling(int M, int K, int N,
					   float alpha,
					   float* A, int lda,
					   float* B, int ldb,
					   float beta,
					   float* C, int ldc)
{
	// Nc MUST be a multiple of NR (12) and Mc a multiple of MR (8),
	// otherwise the packing loops read past the end of a panel.
	const int Nc = 1536;    // was 2048 -- 2048 is not divisible by 12
	const int Kc = 1024;    // left alone for now, so the A/B test isolates the kernel
	const int Mc = 512;

	std::vector<float> packed_A_vec((size_t)Mc * Kc);
	std::vector<float> packed_B_vec((size_t)Kc * Nc);
	float* packed_A = packed_A_vec.data();
	float* packed_B = packed_B_vec.data();

	for (int i = 0; i < N; i += Nc) {
		int curNc = std::min(Nc, N - i);

		for (int j = 0; j < K; j += Kc) {
			int curKc = std::min(Kc, K - j);
			bool is_first_k_block = (j == 0);

			pack_B(curKc, curNc, &B[j*ldb + i], ldb, packed_B);   // Nr argument removed

			for (int k = 0; k < M; k += Mc) {
				int curMc = std::min(Mc, M - k);

				pack_A(curMc, curKc, MR, &A[k*lda + j], lda, packed_A);

				for (int l = 0; l < curNc; l += NR) {            // step by 12 now
					const float* ptr_B = packed_B + (size_t)l * curKc;

					for (int m = 0; m < curMc; m += MR) {        // step by 8
						const float* ptr_A = packed_A + (size_t)m * curKc;
						float* ptr_C = &C[(size_t)(k + m)*ldc + (i + l)];

						// A before B -- opposite order from the old 8x4 kernel
						simd_micro_8x12(curKc, ptr_A, ptr_B, ptr_C, ldc, is_first_k_block);
					}
				}
			}
		}
	}
}


