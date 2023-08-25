#ifndef ATG_SIMPLE_2D_CONSTRAINT_SOLVER_MATRIX_H
#define ATG_SIMPLE_2D_CONSTRAINT_SOLVER_MATRIX_H

#include <assert.h>

#include <memory>
#include <cstdlib>
#include <immintrin.h>

#define ATG_MATRIX_ALIGN 16

namespace atg_scs {
    class Matrix {
        public:
            Matrix();
            Matrix(int width, int height, double value = 0.0);
            ~Matrix();

            void initialize(int width, int height, double value);
            void initialize(int width, int height);
            void resize(int width, int height);
            void destroy();

            void set(const double *data);

            __forceinline void set(int column, int row, double value) {
                assert(column >= 0 && column < m_width);
                assert(row >= 0 && row < m_height);

                m_matrix[row][column] = value;
            }

            __forceinline void add(int column, int row, double value) {
                assert(column >= 0 && column < m_width);
                assert(row >= 0 && row < m_height);

                m_matrix[row][column] += value;
            }

            __forceinline double get(int column, int row) {
                assert(column >= 0 && column < m_width);
                assert(row >= 0 && row < m_height);

                return m_matrix[row][column];
            }

            void set(Matrix *reference);

            void multiply(Matrix &b, Matrix *target);
            void componentMultiply(Matrix &b, Matrix *target);
            void transposeMultiply(Matrix &b, Matrix *target);
            void leftScale(Matrix &scale, Matrix *target);
            void rightScale(Matrix &scale, Matrix *target);
            void scale(double s, Matrix *target);
            void subtract(Matrix &b, Matrix *target);
            void add(Matrix &b, Matrix *target);
            void negate(Matrix *target);
            bool equals(Matrix &b, double err = 1e-6);
            double vectorMagnitudeSquared() const;
            double dot(Matrix &b) const;

            void madd(Matrix &b, double s);
            void pmadd(Matrix &b, double s);

            void transpose(Matrix *target);

            __forceinline int getWidth() const { return m_width; }
            __forceinline int getHeight() const { return m_height; }

            __forceinline void fastRowSwap(int a, int b) {
                double *temp = m_matrix[a];
                m_matrix[a] = m_matrix[b];
                m_matrix[b] = temp;
            }

            __forceinline double** getRawMatrix() const { return m_matrix; }
            __forceinline double* getRawData() const { return m_data; }

        protected:
            double **m_matrix;
            double *m_data;
            int m_width;
            int m_height;
            int m_capacityWidth;
            int m_capacityHeight;
    };
} /* namespace atg_scs */

#endif /* ATG_SIMPLE_2D_CONSTRAINT_SOLVER_MATRIX_H */
