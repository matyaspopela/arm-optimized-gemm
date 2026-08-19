#ifndef GEMM_KERNEL_ARM_BUFFER_H
#define GEMM_KERNEL_ARM_BUFFER_H

#include <cstdlib>


class AlignedBuffer {
public:
	AlignedBuffer() = default;
	AlignedBuffer(const size_t& size, const size_t& alignment) {
		size_t bytes = size * sizeof(float);
		bytes = (( bytes + alignment - 1 ) / alignment ) * alignment; // integer division rounds up
		_p = static_cast<float*>(std::aligned_alloc(alignment, bytes));
		_n = size;
	}

	~AlignedBuffer() { std::free(_p); }

	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer operator=(const AlignedBuffer&) = delete;

	AlignedBuffer(AlignedBuffer&& obj) noexcept : _p(obj._p), _n(obj._n) { // creating a new object
		obj._p = nullptr;
		obj._n = 0;
	}

	AlignedBuffer& operator=(AlignedBuffer&& obj) noexcept { // overriding an existing object
		if (this != &obj) {
			_p = obj._p;
			_n = obj._n;
			obj._p = nullptr;
			obj._n = 0;
		}
		return *this;
	}

	float* data() {
		return _p;
	}

	const float* data() const {
		return _p;
	}

private:
	float* _p = nullptr;
	size_t _n = 0;
};




#endif //GEMM_KERNEL_ARM_BUFFER_H
