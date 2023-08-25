#include "../include/sparse_matrix.h"
#include "../include/matrix.h"
#include <algorithm>

namespace atg_scs {

#if ATG_ENABLE_SPM_HACK

void SparseMatrixBase::initialize(int width, int height) {
	resize(width, height);
	memset(m_blockData, 0xFFFFFF, sizeof(uint8_t) * T_Entries * m_height);
}

void SparseMatrixBase::resize(int width, int height) {
	if (width == m_width && height == m_height) return;
	else if (height > m_capacityHeight) {
		destroy();

		m_capacityHeight = (height > m_capacityHeight)
			? height
			: m_capacityHeight;

		#if ATG_MATRIX_ALIGN
		m_data = (double*)_aligned_malloc((sizeof(double) * ((size_t)T_Stride * T_Entries * m_capacityHeight)), ATG_MATRIX_ALIGN);
		m_matrix = (double**)_aligned_malloc((sizeof(double*) * ((size_t)m_capacityHeight)), ATG_MATRIX_ALIGN);
		m_blockData = (uint8_t*)_aligned_malloc(sizeof(uint8_t) * ((size_t)m_capacityHeight * T_Entries), ATG_MATRIX_ALIGN);
		#else
		m_data = new double[(size_t)T_Stride * T_Entries * m_capacityHeight];
		m_matrix = new double* [m_capacityHeight];
		m_blockData = new uint8_t[(size_t)m_capacityHeight * T_Entries];
		#endif
	}

	m_height = height;
	m_width = width;

	for (int i = 0; i < height; ++i) {
		m_matrix[i] = &m_data[i * T_Entries * T_Stride];
	}
}

void SparseMatrixBase::destroy() {
	if (m_matrix == nullptr) {
		return;
	}

	#if ATG_MATRIX_ALIGN
	_aligned_free(m_matrix);
	_aligned_free(m_data);
	_aligned_free(m_blockData);
	#else
	delete[] m_matrix;
	delete[] m_data;
	delete[] m_blockData;
	#endif

	m_matrix = nullptr;
	m_data = nullptr;
	m_blockData = nullptr;

	m_width = m_height = 0;
}

void SparseMatrixBase::expand(Matrix* matrix) {
	matrix->initialize(m_width, m_height);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < T_Entries; ++j) {
			const uint8_t block = m_blockData[i * T_Entries + j];
			if (block == 0xFF) continue;
			else {
				for (int k = 0; k < T_Stride; ++k) {
					matrix->set(block * T_Stride + k, i, m_matrix[i][j * T_Stride + k]);
				}
			}
		}
	}
}

void SparseMatrixBase::expandTransposed(Matrix* matrix) {
	matrix->initialize(m_height, m_width);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < T_Entries; ++j) {
			const uint8_t block = m_blockData[i * T_Entries + j];
			if (block == 0xFF) continue;
			else {
				for (int k = 0; k < T_Stride; ++k) {
					matrix->set(i, block * T_Stride + k, m_matrix[i][j * T_Stride + k]);
				}
			}
		}
	}
}

void SparseMatrixBase::multiplyTranspose(const SparseMatrixBase& b_T, Matrix* target) const {
	assert(m_width == b_T.m_width);

	target->initialize(b_T.m_height, m_height);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < b_T.m_height; ++j) {
			double dot = 0;
			for (int k = 0; k < T_Entries; ++k) {
				const uint8_t block0 = m_blockData[i * T_Entries + k];
				if (block0 == 0xFF) continue;

				for (int l = 0; l < T_Entries; ++l) {
					const uint8_t block1 = b_T.m_blockData[j * T_Entries + l];
					if (block0 == block1) {
						for (int m = 0; m < T_Stride; ++m) {
							dot +=
								m_matrix[i][k * T_Stride + m]
								* b_T.m_matrix[j][l * T_Stride + m];
						}
					}
				}
			}

			target->set(j, i, dot);
		}
	}
}

void SparseMatrixBase::transposeMultiplyVector(Matrix& b, Matrix* target) const {
	const int b_w = b.getWidth();
	const int b_h = b.getHeight();

	assert(b_w == 1);
	assert(m_height == b_h);

	target->initialize(1, m_width);

	for (int i = 0; i < m_height; ++i) {
		double v = 0.0;
		for (int k = 0; k < T_Entries; ++k) {
			const int offset = k * T_Stride;
			const uint8_t block = m_blockData[i * T_Entries + k];
			if (block == 0xFF) continue;

			for (int l = 0; l < T_Stride; ++l) {
				const int j = block * T_Stride + l;
				target->add(0, j, m_matrix[i][offset + l] * b.get(0, i));
			}
		}
	}
}

void SparseMatrixBase::multiply(Matrix& b, Matrix* target) const {
	const int b_w = b.getWidth();
	const int b_h = b.getHeight();

	assert(m_width == b_h);

	target->initialize(b.getWidth(), m_height);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < b_w; ++j) {
			double v = 0.0;
			for (int k = 0; k < T_Entries; ++k) {
				const int offset = k * T_Stride;
				const uint8_t block = m_blockData[i * T_Entries + k];
				if (block == 0xFF) continue;

				for (int l = 0; l < T_Stride; ++l) {
					v += m_matrix[i][offset + l] * b.get(j, block * T_Stride + l);
				}
			}

			target->set(j, i, v);
		}
	}
}

void SparseMatrixBase::rightScale(Matrix& scale, SparseMatrixBase* target) {
	assert(scale.getWidth() == 1);
	assert(scale.getHeight() == m_width);

	target->initialize(m_width, m_height);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < T_Entries; ++j) {
			const uint8_t index = m_blockData[i * T_Entries + j];
			if (index == 0xFF) continue;

			target->setBlock(i, j, index);

			for (int k = 0; k < T_Stride; ++k) {
				target->set(
					i,
					j,
					k,
					scale.get(0, index * T_Stride + k) * m_matrix[i][j * T_Stride + k]);
			}
		}
	}
}

void SparseMatrixBase::leftScale(Matrix& scale, SparseMatrixBase* target) {
	assert(scale.getWidth() == 1 || m_height == 0);
	assert(scale.getHeight() == m_height);

	target->initialize(m_width, m_height);

	for (int i = 0; i < m_height; ++i) {
		for (int j = 0; j < T_Entries; ++j) {
			const uint8_t index = m_blockData[i * T_Entries + j];
			if (index == 0xFF) continue;

			target->setBlock(i, j, index);

			for (int k = 0; k < T_Stride; ++k) {
				target->set(
					i,
					j,
					k,
					scale.get(0, i) * m_matrix[i][j * T_Stride + k]);
			}
		}
	}
}

#endif

}


