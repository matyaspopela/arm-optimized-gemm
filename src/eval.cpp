#include "Matrix.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <vector>

struct TestCase {
	size_t M;
	size_t N;
	size_t K;
};

int main() {
	std::vector<TestCase> test_cases = {
		{2048, 2048, 2048}
	};

	uint64_t seed = 157982134905234718ULL;

	for (const auto& tc : test_cases) {
		std::mt19937_64 rng(seed);
		std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

		size_t M = tc.M;
		size_t N = tc.N;
		size_t K = tc.K;
		double flops = 2.0 * M * N * K;

		Matrix A(M, K);
		Matrix B(K, N);
		Matrix C(M, N);

		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eA(M, K);
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eB(K, N);
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eC(M, N);

		for (size_t i = 0; i < M * K; ++i) {
			float val = distribution(rng);
			A.buffer().data()[i] = val;
			eA.data()[i] = val;
		}

		for (size_t i = 0; i < K * N; ++i) {
			float val = distribution(rng);
			B.buffer().data()[i] = val;
			eB.data()[i] = val;
		}

		auto start = std::chrono::steady_clock::now();

		C = A * B;

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		double gflops = (flops / elapsed_seconds.count()) / 1e9;

		std::cout << "Elapsed time (own gemm): " << elapsed_seconds.count() << std::endl;
		std::cout << "GFLOPS (own gemm): " << gflops << std::endl;

		start = std::chrono::steady_clock::now();

		eC = eA * eB;

		end = std::chrono::steady_clock::now();
		elapsed_seconds = end - start;
		gflops = (flops / elapsed_seconds.count()) / 1e9;

		std::cout << "Elapsed time (Eigen): " << elapsed_seconds.count() << std::endl;
		std::cout << "GFLOPS (Eigen): " << gflops << std::endl;
	}

	return 0;
}