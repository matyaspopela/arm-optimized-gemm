#ifndef GEMM_KERNEL_ARM_MATRIX_H
#define GEMM_KERNEL_ARM_MATRIX_H

#include <stdexcept>
#include "buffer.h"
#include "gemm.h"



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
		return _buf.data()[row * _cols + col];
	}

	const float& operator()(size_t row, size_t col) const {
		return _buf.data()[row * _cols + col];
	}

	size_t rows() {
		return _rows;
	}

	const size_t rows() const{
		return _rows;
	}

	size_t cols() {
		return _cols;
	}

	const size_t cols() const {
		return _cols;
	}

private:
	AlignedBuffer _buf;
	size_t _rows = 0;
	size_t _cols = 0;
};



inline Matrix operator*(Matrix& A, Matrix& B) {

	if (A.cols() != B.rows()) {
		throw std::runtime_error("Inner dimensions of matrices must match in gemm");
	}

	Matrix C(A.rows(), B.cols());

	gemm(		A.rows(), A.cols(), B.cols(),
				A.buffer().data(), A.cols(),
				B.buffer().data(), B.cols(),
				C.buffer().data(), C.cols());

	return C;
}



#endif //GEMM_KERNEL_ARM_MATRIX_H
