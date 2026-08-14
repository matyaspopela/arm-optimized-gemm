#include <cstdint>
#include <random>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#include "../include/mlib/gemm.hpp"
#include "buffer.hpp"
#include "problem.hpp"

constexpr const char* output_folder = "../outputs/";


// eigen buffer loading setup
using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using StridedMap      = Eigen::Map<      RowMajorF, Eigen::Unaligned, Eigen::OuterStride<>>;
using ConstStridedMap = Eigen::Map<const RowMajorF, Eigen::Unaligned, Eigen::OuterStride<>>;

struct Performance {
	double time	= 0.0;
	double flops;

	double gflops() { return (flops/time)/1e9; }
	const double gflops() const { return (flops/time)/1e9; }
};

void fill_random(float* p, size_t size, std::mt19937& gen) {

	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	for (size_t i = 0; i < size; ++i) {
		p[i] = distribution(gen);
	}
}

Buffers assign_buffers(const Problem& problem, size_t alignement) {
	Buffers buffers;

	buffers.A = AlignedBuffer(problem.M * problem.K, alignement);
	buffers.B = AlignedBuffer(problem.K * problem.N, alignement);
	buffers.C = AlignedBuffer(problem.M * problem.N, alignement);
	buffers.Cref = AlignedBuffer(problem.M * problem.N, alignement);

	return buffers;
}

void fill_buffers(const Problem& problem, Buffers& buffers, uint64_t seed) {
	std::mt19937 gen(seed);

	fill_random(buffers.A.data(), buffers.A.size(), gen);
	fill_random(buffers.B.data(), buffers.B.size(), gen);

	std::fill(buffers.C.data(), buffers.C.data() + buffers.C.size(), std::numeric_limits<float>::quiet_NaN());

	StridedMap A(buffers.A.data(), problem.M, problem.K, Eigen::OuterStride<>(problem.lda));
	StridedMap B(buffers.B.data(), problem.K, problem.N, Eigen::OuterStride<>(problem.ldb));
	StridedMap Cref(buffers.Cref.data(), problem.M, problem.N, Eigen::OuterStride<>(problem.ldc));

	Cref = A * B;
}

bool compare(const Problem& problem, const float* C, const float* Cref,
	double& max_abs, double& max_rel) {

	max_abs = 0.0f; max_rel = 0.0f;
	for (size_t i = 0; i < problem.M; ++i) {
		for (size_t j = 0; j < problem.N; ++j) {

			double c = C[i * problem.ldc + j];
			double cref = Cref[i * problem.ldc + j];

			double diff = std::fabs(c - cref);
			max_abs = std::max(max_abs, diff);

			double denom = std::max({1.0, std::fabs(cref), std::fabs(c)});
			max_rel = std::max(max_rel, diff/denom);
		}
	}

	double tolerance = std::numeric_limits<float>::epsilon() * problem.K * 8.0;
	return max_abs <= tolerance;
}

std::vector<Performance> benchmark(const Problem& problem, Buffers& buffers, size_t n_runs, size_t warmup=3) {
	std::vector<Performance> performances(n_runs - warmup);
	double flops = 2.0 * problem.M * problem.K * problem.N;


	fill_buffers(problem, buffers, 913469136491123); // reset matrices before every gemm

	for (size_t i = 0; i < n_runs; ++i) {
		size_t rel_idx;

		if (i < warmup) {
			gemm(	problem.M, problem.K, problem.N,
					problem.alpha,
					buffers.A.data(), problem.lda,
					buffers.B.data(), problem.ldb,
					problem.beta,
					buffers.C.data(), problem.ldc);
		} else {
			rel_idx = i - warmup;
			auto start = std::chrono::steady_clock::now();

			gemm(	problem.M, problem.K, problem.N,
					problem.alpha,
					buffers.A.data(), problem.lda,
					buffers.B.data(), problem.ldb,
					problem.beta,
					buffers.C.data(), problem.ldc);

			auto end = std::chrono::steady_clock::now();

			std::chrono::duration<double> elapsed_seconds = end - start;

			performances[rel_idx].time = elapsed_seconds.count();
			performances[rel_idx].flops = flops;
		};
	}
	return performances;
}

std::vector<Performance> benchmark_eigen(const Problem& problem, Buffers& buffers, size_t n_runs, size_t warmup=3) {
	std::vector<Performance> performances(n_runs-warmup);
	double flops = 2.0 * problem.M * problem.N * problem.K;

	StridedMap A(buffers.A.data(), problem.M, problem.K, Eigen::OuterStride<>(problem.lda));
	StridedMap B(buffers.B.data(), problem.K, problem.N, Eigen::OuterStride<>(problem.ldb));
	StridedMap Cref(buffers.Cref.data(), problem.M, problem.N, Eigen::OuterStride<>(problem.ldc));

	fill_buffers(problem, buffers, 913469136491123); // reset matrices before every gemm



	for (size_t i = 0; i < n_runs; ++i) {
		size_t rel_idx;

		if (i < warmup) {
			Cref = A*B;
		} else {
			rel_idx = i - warmup;

			auto start = std::chrono::steady_clock::now();
			Cref = A * B;
			auto end = std::chrono::steady_clock::now();

			std::chrono::duration<double> elapsed_seconds = end - start;

			performances[rel_idx].time = elapsed_seconds.count();
			performances[rel_idx].flops = flops;

		}
	}
	return performances;
}

void write_results(std::string path, const std::vector<Performance>& performances, const std::vector<Performance>& comparisons) {
	std::ofstream file(path);

	if (!file.is_open()) {
		throw std::runtime_error("File not open");
	}

	//header
	file << "our_gflops, our_flops, our_time, eigen_gflops, eigen_flops, eigen_time\n";

	for (size_t i = 0; i < performances.size(); ++i) {
		auto p = performances[i];
		auto c = comparisons[i];

		file	<< p.gflops() << ","
				<< p.flops << ","
				<< p.time << ","
				<< c.gflops() << ","
				<< c.flops << ","
				<< c.time << "\n";
	}
	file.close();
};


int main() {

	const std::vector<Problem> cases = {
		{1,1,1,   1,1,1},        // M,N,K, lda,ldb,ldc
		{7,7,7,   7,7,7},
		{16,16,16, 16,16,16},
		{15,17,15, 15,15,15},
		{33,33,33, 33,33,33},
	};

	bool passed = true;

	for (auto& problem : cases) {
		double max_abs, max_rel;
		Buffers buffers = assign_buffers(problem, 64);
		fill_buffers(problem, buffers, 913469136491123);

		gemm(	problem.M, problem.K, problem.N,
				problem.alpha, buffers.A.data(), problem.lda,
				buffers.B.data(), problem.ldb, problem.beta,
				buffers.C.data(), problem.ldc);

		bool ok = compare(problem, buffers.C.data(), buffers.Cref.data(), max_abs, max_rel);
		std::cout<< "Maximum absolute error: " << max_abs << std::endl;
		std::cout<< "Maximum relative error: " << max_rel << std::endl;
		std::cout<< "Passed: " << ok << std::endl;
		passed &= ok;
	}
	std::cout<<"Passed: "<< passed <<std::endl;


	Problem time_benchmark = {2048, 2048, 2048, 2048, 2048, 2048};

	Buffers buffers = assign_buffers(time_benchmark, 64);
	size_t warmup = 2;
	size_t n_runs = 6;

	std::vector<Performance> ourperf = benchmark(time_benchmark, buffers, n_runs, warmup);
	std::vector<Performance> eigenperf = benchmark_eigen(time_benchmark, buffers, n_runs, warmup);

	std::string output_path = std::string(output_folder) + gemm_name;
	output_path += std::string(".csv");

	write_results(output_path, ourperf, eigenperf);

	return 0;
	}