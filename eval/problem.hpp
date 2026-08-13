//
// Created by matya on 8/13/2026.
//

#ifndef GEMM_KERNEL_ARM_PROBLEM_HPP
#define GEMM_KERNEL_ARM_PROBLEM_HPP

#pragma once


struct Problem {
	//mat defs
	int M, N, K;
	int lda, ldb, ldc;

	//blending
	float alpha = 1.0f;
	float beta = 0.0f;
};


















#endif //GEMM_KERNEL_ARM_PROBLEM_HPP
