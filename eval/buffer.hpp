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
		p_ = static_cast<float*>(std::aligned_alloc(alignment, bytes));
		n_ = count;
	}

	~AlignedBuffer() { std::free(p_); }

	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer operator=(const AlignedBuffer&) = delete;

	AlignedBuffer(AlignedBuffer&& obj) noexcept : p_(obj.p_), n_(obj.n_) {
		obj.p_ = nullptr; obj.n_ = 0; //without obj.p_ being nullified we woud wipe the new object!
	}
	AlignedBuffer& operator=(AlignedBuffer&& obj) noexcept {
		if (&obj != this) {
			std::free(p_); //free before reallocating
			p_ = obj.p_;
			n_ = obj.n_;
			obj.p_ = nullptr;
			obj.n_ = 0;
		}
		return *this; //dereference and return the object, not the ptr
	}

	float* data() { return p_; }
	const float* data() const { return p_; } //read only
	const size_t size() const { return n_; } //read only

private:
	float* p_ = nullptr;
	size_t n_ = 0;
};

struct Buffers {
	AlignedBuffer A, B, C, Cref;
};

#endif //GEMM_KERNEL_ARM_BUFFER_HPP
