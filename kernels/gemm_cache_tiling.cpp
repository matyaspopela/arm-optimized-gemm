#include <stdexcept>
#include <arm_neon.h>
#include <cstddef>
#include "simd_micro_8x4.h"
#include <vector>
#include <algorithm>


// at packing point Bp is (Kc, Nc)
static inline void pack_B(int Kc, int Nc, int Nr, const float* B, int ldb, float* out) {

    for (size_t i = 0; i < Nc; i += Nr) {
       for (size_t j = 0; j < Kc; ++j) {
          float32x4_t row = vld1q_f32(&B[j * ldb + i]);
          vst1q_f32(out, row);
          out += Nr;
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

void gemm_cache_tiling( int M, int K, int N,
                   float alpha,
                   float* A, int lda,
                   float* B, int ldb,
                   float beta,
                   float* C, int ldc) {

    // block size params
    size_t Nc = 2048;
    size_t Kc = 1024;
    size_t Mc = 512;
    size_t Nr = 4;
    size_t Mr = 8;

    // config
    bool is_first_k_block = true;

    // allocate arrays on heap otherwise stack overflow happens
    std::vector<float> packed_A_vec(Mc * Kc);
    std::vector<float> packed_B_vec(Kc * Nc);
    float* packed_B = packed_B_vec.data();
    float* packed_A = packed_A_vec.data();

    for (size_t i = 0; i < N; i += Nc) {
       size_t curNc = std::min(Nc, N-i);

       for (size_t j = 0; j < K; j += Kc) {
          size_t curKc = std::min(Kc, K-j);

          if (j != 0) { is_first_k_block = false; } else { is_first_k_block = true; }

          pack_B(curKc, curNc, Nr, &B[j*ldb + i], ldb, packed_B);

          for (size_t k = 0; k < M; k += Mc) {
             size_t curMc = std::min(Mc, M-k);

             pack_A(curMc, curKc, Mr, &A[k*lda+j], lda, packed_A);

             for (size_t l = 0; l < curNc; l += Nr) {

                float* ptr_B = packed_B + l * curKc;

                for (size_t m = 0; m < curMc; m += Mr) {

                   float* ptr_A = packed_A + m * curKc;
                   float* ptr_C = &C[m*ldc + k*ldc + l + i];

                   simd_micro_8x4(Nr, curKc, Mr, ptr_B, ptr_A, ptr_C, ldc, is_first_k_block);

                }
             }
          }
       }
    }
}
