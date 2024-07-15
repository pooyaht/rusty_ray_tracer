use std::ops::Mul;

use crate::vector::Vector3;

#[derive(Clone, Debug)]
struct Matrix<const ROWS: usize, const COLS: usize> {
    data: [[f32; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    pub fn new() -> Self {
        Self {
            data: [[0.0; COLS]; ROWS],
        }
    }

    pub fn with_data(data: [[f32; COLS]; ROWS]) -> Self {
        Self { data }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row][col] = value;
    }

    pub fn identity() -> Result<Self, &'static str> {
        if ROWS != COLS {
            return Err("Matrix must be square");
        }

        let mut matrix = Self::new();
        for i in 0..ROWS {
            matrix.data[i][i] = 1.0;
        }

        Ok(matrix)
    }

    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut matrix = Matrix::<COLS, ROWS>::new();

        for i in 0..ROWS {
            for j in 0..COLS {
                matrix.data[j][i] = self.data[i][j];
            }
        }

        matrix
    }
}

impl Matrix<2, 2> {
    pub fn inverse(&self) -> Result<Matrix<2, 2>, &'static str> {
        if self.determinant() == 0.0 {
            return Err("Matrix is not invertible");
        }

        let mut matrix = Matrix::<2, 2>::new();
        matrix.data[0][0] = self.data[1][1] / self.determinant();
        matrix.data[0][1] = -self.data[0][1] / self.determinant();
        matrix.data[1][0] = -self.data[1][0] / self.determinant();
        matrix.data[1][1] = self.data[0][0] / self.determinant();

        Ok(matrix)
    }
    pub fn determinant(&self) -> f32 {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }
}

impl Matrix<3, 3> {
    pub fn inverse(&self) -> Result<Matrix<3, 3>, &'static str> {
        if self.determinant() == 0.0 {
            return Err("Matrix is not invertible");
        }

        let mut matrix = Matrix::<3, 3>::new();
        let determinant = self.determinant();
        for i in 0..3 {
            for j in 0..3 {
                matrix.data[j][i] = self.cofactor(i, j) / determinant;
            }
        }

        Ok(matrix)
    }

    pub fn determinant(&self) -> f32 {
        self.data[0][0] * self.cofactor(0, 0)
            + self.data[0][1] * self.cofactor(0, 1)
            + self.data[0][2] * self.cofactor(0, 2)
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        let minor = self.minor(row, col);
        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix<2, 2> {
        let submatrix_data: Vec<Vec<f32>> = (0..3)
            .filter(|&i| i != row)
            .map(|i| {
                (0..3)
                    .filter(|&j| j != col)
                    .map(|j| self.data[i][j])
                    .collect()
            })
            .collect();

        Matrix::<2, 2>::with_data([
            [submatrix_data[0][0], submatrix_data[0][1]],
            [submatrix_data[1][0], submatrix_data[1][1]],
        ])
    }
}

impl Matrix<4, 4> {
    pub fn inverse(&self) -> Result<Matrix<4, 4>, &'static str> {
        if self.determinant() == 0.0 {
            return Err("Matrix is not invertible");
        }

        let mut matrix = Matrix::<4, 4>::new();
        let determinant = self.determinant();
        for i in 0..4 {
            for j in 0..4 {
                matrix.data[j][i] = self.cofactor(i, j) / determinant;
            }
        }

        Ok(matrix)
    }

    pub fn determinant(&self) -> f32 {
        self.data[0][0] * self.cofactor(0, 0)
            + self.data[0][1] * self.cofactor(0, 1)
            + self.data[0][2] * self.cofactor(0, 2)
            + self.data[0][3] * self.cofactor(0, 3)
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        let minor = self.minor(row, col);
        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix<3, 3> {
        let submatrix_data: Vec<Vec<f32>> = (0..4)
            .filter(|&i| i != row)
            .map(|i| {
                (0..4)
                    .filter(|&j| j != col)
                    .map(|j| self.data[i][j])
                    .collect()
            })
            .collect();

        Matrix::<3, 3>::with_data([
            [
                submatrix_data[0][0],
                submatrix_data[0][1],
                submatrix_data[0][2],
            ],
            [
                submatrix_data[1][0],
                submatrix_data[1][1],
                submatrix_data[1][2],
            ],
            [
                submatrix_data[2][0],
                submatrix_data[2][1],
                submatrix_data[2][2],
            ],
        ])
    }
}

impl From<Vector3> for Matrix<3, 1> {
    fn from(value: Vector3) -> Self {
        Self {
            data: [[value.x], [value.y], [value.z]],
        }
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;

    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut result = Matrix::<M, P>::new();
        for i in 0..M {
            for j in 0..P {
                for k in 0..N {
                    result.data[i][j] += self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_delta, vector::Vector3};

    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::<3, 3>::new();

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(matrix.data[i][j], 0.0);
            }
        }
    }

    #[test]
    fn test_matrix_comparison() {
        let matrix_a = Matrix::<3, 3>::new();
        let matrix_b = matrix_a.clone();
        let mut matrix_c = Matrix::<3, 3>::new();
        matrix_c.data[0][0] = 1.0;

        assert!(matrix_a.data == matrix_b.data);
        assert!(matrix_a.data != matrix_c.data);
    }

    #[test]
    fn test_3_by_3_matrix_multiplication() {
        let matrix_a = Matrix::<3, 3>::new();
        let matrix_b = Matrix::<3, 3>::new();

        let matrix_c = matrix_a * matrix_b;

        assert!(matrix_c.data == [[0.0; 3]; 3]);
    }

    #[test]
    fn test_4_by_4_matrix_multiplication() {
        let matrix_a = Matrix::<4, 4>::with_data([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let matrix_b = Matrix::<4, 4>::with_data([
            [-2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, -1.0],
            [4.0, 3.0, 6.0, 5.0],
            [1.0, 2.0, 7.0, 8.0],
        ]);

        let matrix_c = matrix_a * matrix_b;

        assert!(
            matrix_c.data
                == [
                    [20.0, 22.0, 50.0, 48.0],
                    [44.0, 54.0, 114.0, 108.0],
                    [40.0, 58.0, 110.0, 102.0],
                    [16.0, 26.0, 46.0, 42.0]
                ]
        );
    }

    #[test]
    fn test_2_by_3_3_by_2_matrix_multiplication() {
        let matrix_a = Matrix::<2, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let matrix_b = Matrix::<3, 2>::with_data([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

        let matrix_c = matrix_a * matrix_b;

        assert!(matrix_c.data == [[58.0, 64.0], [139.0, 154.0],]);
    }

    #[test]
    fn test_3_by_3_vector3_multiplication() {
        let matrix = Matrix::<3, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let vector = Vector3::new(1.0, 2.0, 3.0).into();

        let result = matrix * vector;

        assert!(result.get(0, 0) == 14.0);
        assert!(result.get(1, 0) == 32.0);
        assert!(result.get(2, 0) == 50.0);
    }

    #[test]
    fn test_3_by_3_identity_matrix_multiplication() {
        let matrix = Matrix::<3, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let identity = Matrix::<3, 3>::identity();

        let result = matrix.clone() * identity.unwrap();

        assert!(result.data == matrix.data);
    }

    #[test]
    fn test_vector3_identity_matrix_multiplication() {
        let vector: Matrix<3, 1> = Vector3::new(1.0, 2.0, 3.0).into();
        let identity = Matrix::<3, 3>::identity();

        let result = identity.unwrap() * vector.clone();

        assert!(result.data == vector.data);
    }

    #[test]
    fn test_2_by_3_matrix_transpose() {
        let matrix = Matrix::<2, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let result = matrix.transpose();

        assert!(result.data == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn test_3_by_3_identity_matrix_transpose() {
        let matrix = Matrix::<3, 3>::identity();

        let result = matrix.unwrap().transpose();

        assert!(result.data == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_3_by_3_submatrix() {
        let matrix = Matrix::<3, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let result = matrix.submatrix(1, 1);

        assert!(result.data == [[1.0, 3.0], [7.0, 9.0]]);
    }

    #[test]
    fn test_3_by_3_submatrix_2() {
        let matrix = Matrix::<3, 3>::with_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = matrix.submatrix(0, 0);

        assert!(result.data == [[5.0, 6.0], [8.0, 9.0]]);
    }

    #[test]
    fn test_3_by_3_minor() {
        let matrix =
            Matrix::<3, 3>::with_data([[3.0, 5.0, 0.0], [2.0, -1.0, -7.0], [6.0, -1.0, 5.0]]);

        let result = matrix.minor(1, 0);

        assert!(result == 25.0);
    }

    #[test]
    fn test_3_by_3_cofactor() {
        let matrix =
            Matrix::<3, 3>::with_data([[3.0, 5.0, 0.0], [2.0, -1.0, -7.0], [6.0, -1.0, 5.0]]);

        let result1 = matrix.cofactor(0, 0);
        let result2 = matrix.cofactor(1, 0);

        assert!(result1 == -12.0);
        assert!(result2 == -25.0);
    }

    #[test]
    fn test_3_by_3_determinant() {
        let matrix =
            Matrix::<3, 3>::with_data([[1.0, 2.0, 6.0], [-5.0, 8.0, -4.0], [2.0, 6.0, 4.0]]);

        let result = matrix.determinant();

        assert!(result == -196.0);
    }

    #[test]
    fn test_4_by_4_determinant() {
        let matrix = Matrix::<4, 4>::with_data([
            [-2.0, -8.0, 3.0, 5.0],
            [-3.0, 1.0, 7.0, 3.0],
            [1.0, 2.0, -9.0, 6.0],
            [-6.0, 7.0, 7.0, -9.0],
        ]);

        let result = matrix.determinant();

        assert_eq!(result, -4071.0);
    }

    #[test]
    fn test_4_by_4_inverse() {
        let matrix = Matrix::<4, 4>::with_data([
            [-5.0, 2.0, 6.0, -8.0],
            [1.0, -5.0, 1.0, 8.0],
            [7.0, 7.0, -6.0, -7.0],
            [1.0, -3.0, 7.0, 4.0],
        ]);

        let result = matrix.inverse().unwrap();
        let expected_result = Matrix::<4, 4>::with_data([
            [0.21805, 0.45113, 0.24060, -0.04511],
            [-0.80827, -1.45677, -0.44361, 0.52068],
            [-0.07895, -0.22368, -0.05263, 0.19737],
            [-0.52256, -0.81391, -0.30075, 0.30639],
        ]);

        for i in 0..4 {
            for j in 0..4 {
                assert_delta!(result.data[i][j], expected_result.data[i][j], 0.001);
            }
        }
    }

    #[test]
    fn test_4_by_4_multiply_by_inverse() {
        let matrix_a = Matrix::<4, 4>::with_data([
            [3.0, -9.0, 7.0, 3.0],
            [3.0, -8.0, 2.0, -9.0],
            [-4.0, 4.0, 4.0, 1.0],
            [-6.0, 5.0, -1.0, 1.0],
        ]);
        let matrix_b = Matrix::<4, 4>::with_data([
            [8.0, 2.0, 2.0, 2.0],
            [3.0, -1.0, 7.0, 0.0],
            [7.0, 0.0, 5.0, 4.0],
            [6.0, -2.0, 0.0, 5.0],
        ]);

        let expected_result = matrix_a.clone() * matrix_b.clone() * matrix_b.inverse().unwrap();

        for i in 0..4 {
            for j in 0..4 {
                assert_delta!(matrix_a.data[i][j], expected_result.data[i][j], 0.001);
            }
        }
    }
}
