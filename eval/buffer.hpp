//
// Created by matya on 8/13/2026.
//

#ifndef GEMM_KERNEL_ARM_BUFFER_HPP
#define GEMM_KERNEL_ARM_BUFFER_HPP
#include <cstddef>
#include <cmath>
#include <cstdlib>

#pragma once

class AlignedBuffer {
public:
	AlignedBuffer() = default;
	AlignedBuffer(size_t count, size_t alignment) {
		size_t bytes = count * sizeof(float);
		bytes = ((bytes + alignment - 1) / alignment) * alignment; //integer division makes it correct
		p_ = static_cast<float*>(std::aligned_alloc(bytes, alignment));
	}

	~AlignedBuffer();

	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer operator=(const AlignedBuffer&) = delete;

	AlignedBuffer(AlignedBuffer&&) noexcept;
	AlignedBuffer& operator=(AlignedBuffer&&) noexcept;

private:
	float* p_ = nullptr;
	size_t n_ = 0;
};

struct Buffers {
	AlignedBuffer A, B, C, Cref;
};



#endif //GEMM_KERNEL_ARM_BUFFER_HPP
