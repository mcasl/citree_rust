//! Minimal dense linear algebra helpers.
//!
//! The algorithm only needs to invert small, well-conditioned (precision) matrices,
//! so a pure-Rust Gauss-Jordan elimination with partial pivoting is enough and avoids
//! depending on a system BLAS/LAPACK (previously `ndarray-linalg` + OpenBLAS).

use ndarray::Array2;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum LinalgError {
    #[error("matrix is not square: shape {0}x{1}")]
    NotSquare(usize, usize),
    #[error("matrix is singular (pivot {pivot:e} in column {column})")]
    Singular { column: usize, pivot: f64 },
}

/// Returns the inverse of a square matrix using Gauss-Jordan elimination
/// with partial pivoting.
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>, LinalgError> {
    let (rows, cols) = a.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare(rows, cols));
    }
    let n = rows;
    let mut m = a.to_owned();
    let mut inv = Array2::<f64>::eye(n);

    // Tolerance relative to the magnitude of the matrix entries.
    let scale = m.iter().fold(0.0_f64, |acc, x| acc.max(x.abs())).max(1.0);
    let tol = f64::EPSILON * scale * n as f64;

    for col in 0..n {
        // Partial pivoting: pick the row with the largest absolute value in this column.
        let pivot_row = (col..n)
            .max_by(|&i, &j| m[[i, col]].abs().total_cmp(&m[[j, col]].abs()))
            .expect("non-empty range");
        let pivot = m[[pivot_row, col]];
        if pivot.abs() <= tol {
            return Err(LinalgError::Singular { column: col, pivot });
        }
        if pivot_row != col {
            for k in 0..n {
                m.swap([pivot_row, k], [col, k]);
                inv.swap([pivot_row, k], [col, k]);
            }
        }

        // Normalise the pivot row.
        let inv_pivot = 1.0 / m[[col, col]];
        for k in 0..n {
            m[[col, k]] *= inv_pivot;
            inv[[col, k]] *= inv_pivot;
        }

        // Eliminate the column in every other row.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[[row, col]];
            if factor == 0.0 {
                continue;
            }
            for k in 0..n {
                m[[row, k]] -= factor * m[[col, k]];
                inv[[row, k]] -= factor * inv[[col, k]];
            }
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn assert_close(a: &Array2<f64>, b: &Array2<f64>) {
        assert_eq!(a.dim(), b.dim());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn inverse_of_identity_is_identity() {
        let i = Array2::<f64>::eye(3);
        assert_close(&inv(&i).unwrap(), &i);
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let a = arr2(&[[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]);
        let a_inv = inv(&a).unwrap();
        assert_close(&a.dot(&a_inv), &Array2::eye(3));
        assert_close(&a_inv.dot(&a), &Array2::eye(3));
    }

    #[test]
    fn needs_pivoting() {
        let a = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        assert_close(&inv(&a).unwrap(), &a);
    }

    #[test]
    fn known_2x2_inverse() {
        let a = arr2(&[[2.0, 2.0], [2.0, 4.0]]);
        let expected = arr2(&[[1.0, -0.5], [-0.5, 0.5]]);
        assert_close(&inv(&a).unwrap(), &expected);
    }

    #[test]
    fn singular_matrix_is_an_error() {
        let a = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        assert!(matches!(inv(&a), Err(LinalgError::Singular { .. })));
    }

    #[test]
    fn non_square_matrix_is_an_error() {
        let a = Array2::<f64>::zeros((2, 3));
        assert_eq!(inv(&a), Err(LinalgError::NotSquare(2, 3)));
    }
}
