#ifndef GEMM_KERNEL_ARM_MATRIX_H
#define GEMM_KERNEL_ARM_MATRIX_H

#include <stdexcept>
#include "buffer.h"



class Matrix {
public:
	Matrix() = default;
	Matrix(size_t rows, size_t cols) {

		_buf = AlignedBuffer(rows * cols, 64);
		_rows = rows;
		_cols = cols;

	}

	~Matrix() = default;

	Matrix(const Matrix&) = delete;
	Matrix operator=(const Matrix&) = delete;

	Matrix(Matrix&&) = default;
	Matrix& operator=(Matrix&&) = default;

	AlignedBuffer& buffer() {
		return _buf;
	}

	const AlignedBuffer& buffer() const {
		return _buf;
	}

	float& operator()(size_t row, size_t col) {
		return _buf[row * _cols + col];
	}

	const float& operator()(size_t row, size_t col) const {
		return _buf[row * _cols + col];
	}

private:
	AlignedBuffer _buf;
	size_t _rows = 0;
	size_t _cols = 0;
};



#endif //GEMM_KERNEL_ARM_MATRIX_H
