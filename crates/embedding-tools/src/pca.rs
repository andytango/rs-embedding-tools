//! Principal Component Analysis (PCA) for dimensionality reduction.
//!
//! This module provides a simple implementation of PCA for reducing the
//! dimensionality of data while preserving as much variance as possible.

use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

/// Errors that can occur during PCA
#[derive(Error, Debug)]
pub enum PcaError {
    #[error("Invalid number of components: {msg}")]
    InvalidComponents { msg: String },
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Computation error: {msg}")]
    ComputationError { msg: String },
}

/// Principal Component Analysis
pub struct Pca {
    n_components: usize,
}

impl Pca {
    /// Create a new PCA instance
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to keep
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit the PCA model and transform the data
    ///
    /// This combines fitting and transformation in one step for convenience.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Transformed data with reduced dimensionality (n_samples × n_components)
    pub fn fit_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PcaError> {
        if data.nrows() == 0 {
            return Err(PcaError::EmptyDataset);
        }

        if self.n_components == 0 || self.n_components > data.ncols() {
            return Err(PcaError::InvalidComponents {
                msg: format!(
                    "n_components must be between 1 and {} (number of features)",
                    data.ncols()
                ),
            });
        }

        // Center the data (subtract mean)
        let mean = data.mean_axis(Axis(0)).ok_or_else(|| PcaError::ComputationError {
            msg: "Failed to compute mean".to_string(),
        })?;

        let centered = data - &mean.insert_axis(Axis(0));

        // Compute covariance matrix
        let n_samples = data.nrows() as f64;
        let covariance = centered.t().dot(&centered) / (n_samples - 1.0);

        // Compute eigenvalues and eigenvectors
        // For simplicity, we'll use a power iteration method for the top k components
        let components = self.compute_top_eigenvectors(&covariance, self.n_components)?;

        // Project data onto principal components
        let transformed = centered.dot(&components);

        Ok(transformed)
    }

    /// Compute the top k eigenvectors using power iteration
    ///
    /// This is a simplified approach suitable for this use case.
    fn compute_top_eigenvectors(
        &self,
        matrix: &Array2<f64>,
        k: usize,
    ) -> Result<Array2<f64>, PcaError> {
        let n = matrix.nrows();
        let mut components = Array2::zeros((n, k));
        let mut working_matrix = matrix.clone();

        for i in 0..k {
            // Power iteration to find dominant eigenvector
            let mut vec = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

            for _ in 0..100 {
                // Iterate to converge
                let new_vec = working_matrix.dot(&vec);
                let norm = new_vec.dot(&new_vec).sqrt();

                if norm < 1e-10 {
                    break;
                }

                vec = &new_vec / norm;
            }

            // Store the eigenvector
            for j in 0..n {
                components[[j, i]] = vec[j];
            }

            // Deflate the matrix (remove this component's contribution)
            let eigenvalue = vec.dot(&working_matrix.dot(&vec));
            let outer_product = Self::outer_product(&vec, &vec);
            working_matrix -= &(outer_product * eigenvalue);
        }

        Ok(components)
    }

    /// Compute outer product of two vectors
    fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let n = a.len();
        let m = b.len();
        let mut result = Array2::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                result[[i, j]] = a[i] * b[j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0,
            ],
        )
        .unwrap();

        let pca = Pca::new(2);
        let transformed = pca.fit_transform(&data).unwrap();

        assert_eq!(transformed.nrows(), 5);
        assert_eq!(transformed.ncols(), 2);

        // Check that all values are finite
        for &val in transformed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_pca_empty_dataset() {
        let data = Array2::from_shape_vec((0, 3), vec![]).unwrap();
        let pca = Pca::new(2);
        let result = pca.fit_transform(&data);
        assert!(matches!(result, Err(PcaError::EmptyDataset)));
    }

    #[test]
    fn test_pca_invalid_components() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();

        let pca = Pca::new(3); // More components than features
        let result = pca.fit_transform(&data);
        assert!(matches!(result, Err(PcaError::InvalidComponents { .. })));
    }

    #[test]
    fn test_pca_preserves_samples() {
        let data = Array2::from_shape_vec(
            (10, 5),
            (0..50).map(|x| x as f64).collect(),
        )
        .unwrap();

        let pca = Pca::new(2);
        let transformed = pca.fit_transform(&data).unwrap();

        assert_eq!(transformed.nrows(), data.nrows());
        assert_eq!(transformed.ncols(), 2);
    }
}
