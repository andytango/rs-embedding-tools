//! Pure Rust implementation of PACMAP (Pairwise Controlled Manifold Approximation Projection).
//!
//! PACMAP is a dimensionality reduction technique that preserves both local and global
//! structure better than PCA and is faster than t-SNE or UMAP.
//!
//! # Algorithm Overview
//!
//! PACMAP uses three types of point pairs:
//! 1. **Near pairs**: k-nearest neighbors (preserve local structure)
//! 2. **Mid-near pairs**: Medium-distance points (preserve intermediate structure)
//! 3. **Far pairs**: Distant points (preserve global structure)
//!
//! The embedding is optimized to:
//! - Pull near pairs close together
//! - Keep mid-near pairs at moderate distance
//! - Push far pairs apart
//!
//! # Example
//!
//! ```
//! use pacmap::PacmapBuilder;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((100, 50), (0..5000).map(|x| x as f64).collect()).unwrap();
//!
//! let pacmap = PacmapBuilder::new()
//!     .n_components(2)
//!     .n_neighbors(10)
//!     .build();
//!
//! let embedding = pacmap.fit_transform(&data).unwrap();
//! assert_eq!(embedding.dim(), (100, 2));
//! ```
//!
//! # References
//!
//! Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).
//! "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization"
//! Journal of Machine Learning Research, 22(201), 1-73.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashSet;
use thiserror::Error;

/// Errors that can occur during PACMAP computation
#[derive(Error, Debug)]
pub enum PacmapError {
    #[error("Invalid parameters: {msg}")]
    InvalidParameters { msg: String },
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Computation error: {msg}")]
    ComputationError { msg: String },
}

/// Builder for PACMAP dimensionality reduction
#[derive(Debug, Clone)]
pub struct PacmapBuilder {
    n_components: usize,
    n_neighbors: usize,
    n_mid_near: usize,
    n_far: usize,
    n_iterations: usize,
    learning_rate: f64,
    seed: Option<u64>,
}

impl Default for PacmapBuilder {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 10,
            n_mid_near: 5,
            n_far: 2,
            n_iterations: 450,
            learning_rate: 1.0,
            seed: Some(42),
        }
    }
}

impl PacmapBuilder {
    /// Create a new PACMAP builder with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of components (dimensions) in the embedding
    ///
    /// Default: 2
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set the number of nearest neighbors to use
    ///
    /// Default: 10
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set the number of mid-near pairs per point
    ///
    /// Default: 5
    pub fn n_mid_near(mut self, n: usize) -> Self {
        self.n_mid_near = n;
        self
    }

    /// Set the number of far pairs per point
    ///
    /// Default: 2
    pub fn n_far(mut self, n: usize) -> Self {
        self.n_far = n;
        self
    }

    /// Set the number of optimization iterations
    ///
    /// Default: 450
    pub fn n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set the learning rate for optimization
    ///
    /// Default: 1.0
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set random seed for reproducibility
    ///
    /// Default: Some(42)
    pub fn seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Build the PACMAP instance
    pub fn build(self) -> Pacmap {
        Pacmap {
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            n_mid_near: self.n_mid_near,
            n_far: self.n_far,
            n_iterations: self.n_iterations,
            learning_rate: self.learning_rate,
            seed: self.seed,
        }
    }
}

/// PACMAP dimensionality reduction
#[derive(Debug, Clone)]
pub struct Pacmap {
    n_components: usize,
    n_neighbors: usize,
    n_mid_near: usize,
    n_far: usize,
    n_iterations: usize,
    learning_rate: f64,
    seed: Option<u64>,
}

impl Pacmap {
    /// Fit the model and transform the data
    ///
    /// # Arguments
    ///
    /// * `data` - Input data matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Transformed data with reduced dimensionality (n_samples × n_components)
    pub fn fit_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PacmapError> {
        if data.nrows() == 0 {
            return Err(PacmapError::EmptyDataset);
        }

        if self.n_components == 0 || self.n_components > data.ncols() {
            return Err(PacmapError::InvalidParameters {
                msg: format!(
                    "n_components must be between 1 and {} (number of features)",
                    data.ncols()
                ),
            });
        }

        let n_samples = data.nrows();

        if self.n_neighbors >= n_samples {
            return Err(PacmapError::InvalidParameters {
                msg: format!("n_neighbors ({}) must be less than n_samples ({})", self.n_neighbors, n_samples),
            });
        }

        // Step 1: Find k-nearest neighbors
        let neighbors = self.find_k_nearest_neighbors(data)?;

        // Step 2: Generate pair sets
        let near_pairs = self.generate_near_pairs(&neighbors)?;
        let mid_near_pairs = self.generate_mid_near_pairs(&neighbors, n_samples)?;
        let far_pairs = self.generate_far_pairs(&neighbors, n_samples)?;

        // Step 3: Initialize embedding
        let mut embedding = self.initialize_embedding(data)?;

        // Step 4: Optimize embedding
        self.optimize_embedding(
            &mut embedding,
            &near_pairs,
            &mid_near_pairs,
            &far_pairs,
        )?;

        Ok(embedding)
    }

    /// Find k-nearest neighbors for each point using brute force
    fn find_k_nearest_neighbors(&self, data: &Array2<f64>) -> Result<Vec<Vec<usize>>, PacmapError> {
        let n_samples = data.nrows();
        let mut neighbors = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let point = data.row(i);
            let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n_samples);

            for j in 0..n_samples {
                if i != j {
                    let other = data.row(j);
                    let dist = self.euclidean_distance(&point, &other);
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let k_neighbors: Vec<usize> = distances
                .iter()
                .take(self.n_neighbors)
                .map(|(idx, _)| *idx)
                .collect();

            neighbors.push(k_neighbors);
        }

        Ok(neighbors)
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Generate near pairs from k-nearest neighbors
    fn generate_near_pairs(&self, neighbors: &[Vec<usize>]) -> Result<Vec<(usize, usize)>, PacmapError> {
        let mut pairs = Vec::new();

        for (i, neighbor_list) in neighbors.iter().enumerate() {
            for &j in neighbor_list {
                if i < j {
                    pairs.push((i, j));
                }
            }
        }

        Ok(pairs)
    }

    /// Generate mid-near pairs (neighbors of neighbors that aren't direct neighbors)
    fn generate_mid_near_pairs(
        &self,
        neighbors: &[Vec<usize>],
        n_samples: usize,
    ) -> Result<Vec<(usize, usize)>, PacmapError> {
        let mut pairs = Vec::new();
        let mut rng = SimpleRng::new(self.seed.unwrap_or(42));

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let direct_neighbors: HashSet<usize> = neighbors[i].iter().copied().collect();
            let mut mid_near_candidates = HashSet::new();

            // Find neighbors of neighbors
            for &neighbor in &neighbors[i] {
                for &neighbor_of_neighbor in &neighbors[neighbor] {
                    if neighbor_of_neighbor != i && !direct_neighbors.contains(&neighbor_of_neighbor) {
                        mid_near_candidates.insert(neighbor_of_neighbor);
                    }
                }
            }

            // Sample n_mid_near pairs
            let mut candidates: Vec<usize> = mid_near_candidates.into_iter().collect();
            if candidates.len() > self.n_mid_near {
                // Simple random sampling
                for _ in 0..self.n_mid_near {
                    if !candidates.is_empty() {
                        let idx = rng.next_usize() % candidates.len();
                        let j = candidates.swap_remove(idx);
                        if i < j {
                            pairs.push((i, j));
                        }
                    }
                }
            } else {
                for j in candidates {
                    if i < j {
                        pairs.push((i, j));
                    }
                }
            }
        }

        Ok(pairs)
    }

    /// Generate far pairs (random distant points)
    fn generate_far_pairs(
        &self,
        neighbors: &[Vec<usize>],
        n_samples: usize,
    ) -> Result<Vec<(usize, usize)>, PacmapError> {
        let mut pairs = Vec::new();
        let mut rng = SimpleRng::new(self.seed.unwrap_or(42) + 1);

        for (i, neighbor_list) in neighbors.iter().enumerate().take(n_samples) {
            let neighbor_set: HashSet<usize> = neighbor_list.iter().copied().collect();

            for _ in 0..self.n_far {
                // Sample a random point that's not a neighbor
                for _ in 0..100 {
                    // Try up to 100 times
                    let j = rng.next_usize() % n_samples;
                    if j != i && !neighbor_set.contains(&j) {
                        if i < j {
                            pairs.push((i, j));
                        }
                        break;
                    }
                }
            }
        }

        Ok(pairs)
    }

    /// Initialize embedding using simple PCA-like initialization
    fn initialize_embedding(&self, data: &Array2<f64>) -> Result<Array2<f64>, PacmapError> {
        let n_samples = data.nrows();
        let mut rng = SimpleRng::new(self.seed.unwrap_or(42) + 2);

        // Simple random initialization scaled by data range
        let mean = data.mean_axis(Axis(0)).ok_or_else(|| PacmapError::ComputationError {
            msg: "Failed to compute mean".to_string(),
        })?;

        let mut std = Array1::<f64>::zeros(data.ncols());
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let diff = data[[i, j]] - mean[j];
                std[j] += diff * diff;
            }
        }
        std.mapv_inplace(|x: f64| (x / n_samples as f64).sqrt());
        let scale = std.iter().sum::<f64>() / std.len() as f64;

        let mut embedding = Array2::<f64>::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = (rng.next_f64() - 0.5) * scale * 0.0001;
            }
        }

        Ok(embedding)
    }

    /// Optimize the embedding using gradient descent
    fn optimize_embedding(
        &self,
        embedding: &mut Array2<f64>,
        near_pairs: &[(usize, usize)],
        mid_near_pairs: &[(usize, usize)],
        far_pairs: &[(usize, usize)],
    ) -> Result<(), PacmapError> {
        let n_samples = embedding.nrows();

        // Weights for different phases (inspired by PACMAP paper)
        let phases = [
            // Phase 1: Focus on global structure
            (100, 2.0, 0.0, 1.0),
            // Phase 2: Balance all
            (200, 3.0, 3.0, 0.1),
            // Phase 3: Refine local structure
            (150, 1.0, 0.5, 0.001),
        ];

        let mut iteration = 0;

        for (phase_iters, w_near, w_mid, w_far) in phases.iter() {
            for _ in 0..*phase_iters {
                if iteration >= self.n_iterations {
                    break;
                }

                let lr = self.learning_rate * (1.0 - iteration as f64 / self.n_iterations as f64);

                // Compute gradients
                let mut gradients = Array2::<f64>::zeros((n_samples, self.n_components));

                // Near pairs: attract
                for &(i, j) in near_pairs {
                    let diff = &embedding.row(i).to_owned() - &embedding.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt().max(1e-10);

                    let grad_scale = w_near * 2.0 / (1.0 + dist * dist);
                    for d in 0..self.n_components {
                        let grad = grad_scale * diff[d];
                        gradients[[i, d]] += grad;
                        gradients[[j, d]] -= grad;
                    }
                }

                // Mid-near pairs: moderate distance
                for &(i, j) in mid_near_pairs {
                    let diff = &embedding.row(i).to_owned() - &embedding.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt().max(1e-10);

                    // Want distance around 1.0
                    let grad_scale = w_mid * 2.0 * (dist - 1.0) / (1.0 + dist * dist);
                    for d in 0..self.n_components {
                        let grad = grad_scale * diff[d] / dist;
                        gradients[[i, d]] += grad;
                        gradients[[j, d]] -= grad;
                    }
                }

                // Far pairs: repel
                for &(i, j) in far_pairs {
                    let diff = &embedding.row(i).to_owned() - &embedding.row(j);
                    let dist = diff.mapv(|x| x * x).sum().sqrt().max(1e-10);

                    let grad_scale = -w_far * 2.0 / ((1.0 + dist * dist) * dist);
                    for d in 0..self.n_components {
                        let grad = grad_scale * diff[d];
                        gradients[[i, d]] += grad;
                        gradients[[j, d]] -= grad;
                    }
                }

                // Update embedding
                for i in 0..n_samples {
                    for d in 0..self.n_components {
                        embedding[[i, d]] -= lr * gradients[[i, d]];
                    }
                }

                iteration += 1;
            }
        }

        Ok(())
    }
}

/// Simple pseudo-random number generator for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pacmap_basic() {
        let data = Array2::from_shape_vec(
            (10, 5),
            (0..50).map(|x| x as f64).collect(),
        )
        .unwrap();

        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(3)
            .n_iterations(50)
            .build();

        let embedding = pacmap.fit_transform(&data).unwrap();

        assert_eq!(embedding.dim(), (10, 2));
        // All values should be finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_empty_dataset() {
        let data = Array2::from_shape_vec((0, 5), vec![]).unwrap();
        let pacmap = PacmapBuilder::new().build();
        let result = pacmap.fit_transform(&data);
        assert!(matches!(result, Err(PacmapError::EmptyDataset)));
    }

    #[test]
    fn test_invalid_n_components() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();
        let pacmap = PacmapBuilder::new().n_components(10).build();
        let result = pacmap.fit_transform(&data);
        assert!(matches!(result, Err(PacmapError::InvalidParameters { .. })));
    }

    #[test]
    fn test_reproducibility() {
        let data = Array2::from_shape_vec((20, 5), (0..100).map(|x| x as f64).collect()).unwrap();

        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5)
            .n_iterations(100)
            .seed(Some(42))
            .build();

        let embedding1 = pacmap.fit_transform(&data).unwrap();
        let embedding2 = pacmap.fit_transform(&data).unwrap();

        // Should be identical with same seed
        for i in 0..embedding1.nrows() {
            for j in 0..embedding1.ncols() {
                assert!((embedding1[[i, j]] - embedding2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_preserves_sample_count() {
        let data = Array2::from_shape_vec((15, 8), (0..120).map(|x| x as f64).collect()).unwrap();

        let pacmap = PacmapBuilder::new().n_components(3).build();
        let embedding = pacmap.fit_transform(&data).unwrap();

        assert_eq!(embedding.nrows(), data.nrows());
        assert_eq!(embedding.ncols(), 3);
    }
}
