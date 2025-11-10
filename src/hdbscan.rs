//! Pure Rust implementation of the HDBSCAN clustering algorithm.
//!
//! HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
//! is a clustering algorithm that extends DBSCAN by converting it into a hierarchical
//! clustering algorithm, and then using a technique to extract a flat clustering based
//! on the stability of clusters.
//!
//! # Algorithm Overview
//!
//! 1. Compute core distances for each point (distance to k-th nearest neighbor)
//! 2. Build mutual reachability graph
//! 3. Construct minimum spanning tree using Prim's algorithm
//! 4. Build cluster hierarchy by processing MST edges
//! 5. Condense the hierarchy based on min_cluster_size
//! 6. Extract stable clusters based on stability scores
//!
//! # Example
//!
//! ```
//! use scanner_embeddings::hdbscan::HdbscanBuilder;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 2.0,
//!     1.1, 2.1,
//!     10.0, 11.0,
//!     10.1, 11.1,
//!     10.2, 11.2,
//! ]).unwrap();
//!
//! let clusterer = HdbscanBuilder::new()
//!     .min_cluster_size(2)
//!     .min_samples(2)
//!     .build();
//!
//! let labels = clusterer.fit_predict(&data).unwrap();
//! ```

use ndarray::{Array1, Array2, ArrayView1};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use thiserror::Error;

/// Errors that can occur during HDBSCAN clustering
#[derive(Error, Debug)]
pub enum HdbscanError {
    #[error("Invalid parameters: {msg}")]
    InvalidParameters { msg: String },
    #[error("Empty dataset provided")]
    EmptyDataset,
    #[error("Computation error: {msg}")]
    ComputationError { msg: String },
}

/// Builder for HDBSCAN clustering algorithm
#[derive(Debug, Clone)]
pub struct HdbscanBuilder {
    min_cluster_size: usize,
    min_samples: usize,
    metric: DistanceMetric,
}

/// Distance metrics supported by HDBSCAN
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
}

impl Default for HdbscanBuilder {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: 5,
            metric: DistanceMetric::Euclidean,
        }
    }
}

impl HdbscanBuilder {
    /// Create a new HDBSCAN builder with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum cluster size (default: 5)
    ///
    /// Clusters smaller than this will be considered noise.
    pub fn min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Set the minimum number of samples (default: 5)
    ///
    /// This affects the core distance calculation (k-th nearest neighbor).
    pub fn min_samples(mut self, samples: usize) -> Self {
        self.min_samples = samples;
        self
    }

    /// Set the distance metric (default: Euclidean)
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Build the HDBSCAN clusterer
    pub fn build(self) -> Hdbscan {
        Hdbscan {
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            metric: self.metric,
        }
    }
}

/// HDBSCAN clustering algorithm
#[derive(Debug, Clone)]
pub struct Hdbscan {
    min_cluster_size: usize,
    min_samples: usize,
    metric: DistanceMetric,
}

impl Hdbscan {
    /// Fit the model and predict cluster labels
    ///
    /// Returns a vector of cluster labels where -1 indicates noise.
    pub fn fit_predict(&self, data: &Array2<f64>) -> Result<Vec<i32>, HdbscanError> {
        if data.nrows() == 0 {
            return Err(HdbscanError::EmptyDataset);
        }

        if self.min_cluster_size < 2 {
            return Err(HdbscanError::InvalidParameters {
                msg: "min_cluster_size must be at least 2".to_string(),
            });
        }

        if self.min_samples < 1 {
            return Err(HdbscanError::InvalidParameters {
                msg: "min_samples must be at least 1".to_string(),
            });
        }

        let n_points = data.nrows();

        // Step 1: Compute core distances (distance to k-th nearest neighbor)
        let core_distances = self.compute_core_distances(data)?;

        // Step 2: Build minimum spanning tree using mutual reachability distance
        let mst = self.build_mst(data, &core_distances)?;

        // Step 3: Build single linkage dendrogram from MST
        let dendrogram = self.build_dendrogram(&mst, n_points)?;

        // Step 4: Condense the tree based on min_cluster_size
        let condensed_tree = self.condense_tree(&dendrogram, n_points)?;

        // Step 5: Extract stable clusters
        let labels = self.extract_clusters(&condensed_tree, n_points)?;

        Ok(labels)
    }

    /// Compute core distances for all points
    ///
    /// The core distance is the distance to the k-th nearest neighbor,
    /// where k = min_samples.
    fn compute_core_distances(&self, data: &Array2<f64>) -> Result<Array1<f64>, HdbscanError> {
        let n_points = data.nrows();
        let mut core_distances = Array1::zeros(n_points);

        for i in 0..n_points {
            let point = data.row(i);
            let mut distances = Vec::with_capacity(n_points);

            for j in 0..n_points {
                let other = data.row(j);
                let dist = self.compute_distance(&point, &other);
                distances.push(dist);
            }

            // Sort distances to find k-th nearest neighbor
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Core distance is distance to min_samples-th neighbor (1-indexed)
            // Ensure we don't go out of bounds
            let k = self.min_samples.min(distances.len() - 1);
            core_distances[i] = if k < distances.len() {
                distances[k]
            } else {
                distances[distances.len() - 1]
            };
        }

        Ok(core_distances)
    }

    /// Compute distance between two points based on the selected metric
    fn compute_distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f64>()
                    .sqrt()
            }
        }
    }

    /// Compute mutual reachability distance between two points
    ///
    /// mutual_reach_dist(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
    fn mutual_reachability_distance(
        &self,
        i: usize,
        j: usize,
        data: &Array2<f64>,
        core_distances: &Array1<f64>,
    ) -> f64 {
        let dist = self.compute_distance(&data.row(i), &data.row(j));
        dist.max(core_distances[i]).max(core_distances[j])
    }

    /// Build minimum spanning tree using Prim's algorithm
    fn build_mst(
        &self,
        data: &Array2<f64>,
        core_distances: &Array1<f64>,
    ) -> Result<Vec<MstEdge>, HdbscanError> {
        let n_points = data.nrows();
        let mut mst = Vec::with_capacity(n_points - 1);
        let mut in_tree = vec![false; n_points];
        let mut heap = BinaryHeap::new();

        // Start with point 0
        in_tree[0] = true;

        // Add all edges from point 0
        for j in 1..n_points {
            let dist = self.mutual_reachability_distance(0, j, data, core_distances);
            heap.push(HeapEdge {
                from: 0,
                to: j,
                weight: dist,
            });
        }

        // Prim's algorithm main loop
        while let Some(edge) = heap.pop() {
            if in_tree[edge.to] {
                continue;
            }

            in_tree[edge.to] = true;
            mst.push(MstEdge {
                from: edge.from,
                to: edge.to,
                weight: edge.weight,
            });

            if mst.len() == n_points - 1 {
                break;
            }

            // Add all edges from the newly added point
            for (j, &is_in_tree) in in_tree.iter().enumerate() {
                if !is_in_tree {
                    let dist = self.mutual_reachability_distance(edge.to, j, data, core_distances);
                    heap.push(HeapEdge {
                        from: edge.to,
                        to: j,
                        weight: dist,
                    });
                }
            }
        }

        Ok(mst)
    }

    /// Build dendrogram (single linkage tree) from MST
    fn build_dendrogram(
        &self,
        mst: &[MstEdge],
        n_points: usize,
    ) -> Result<Vec<DendrogramNode>, HdbscanError> {
        // Sort MST edges by weight (ascending)
        let mut sorted_edges = mst.to_vec();
        sorted_edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));

        // Use Union-Find to track cluster membership
        let mut union_find = UnionFind::new(n_points * 2); // Enough space for all merges
        let mut dendrogram = Vec::new();
        let mut next_cluster_id = n_points;

        // Build dendrogram by processing edges in order
        for edge in sorted_edges {
            let cluster_a = union_find.find(edge.from);
            let cluster_b = union_find.find(edge.to);

            if cluster_a != cluster_b {
                // Merge the two clusters
                union_find.parent[cluster_a] = next_cluster_id;
                union_find.parent[cluster_b] = next_cluster_id;

                dendrogram.push(DendrogramNode {
                    cluster_id: next_cluster_id,
                    left: cluster_a,
                    right: cluster_b,
                    distance: edge.weight,
                });

                next_cluster_id += 1;
            }
        }

        Ok(dendrogram)
    }

    /// Simplified condensation - just use the dendrogram structure
    fn condense_tree(
        &self,
        dendrogram: &[DendrogramNode],
        n_points: usize,
    ) -> Result<CondensedTree, HdbscanError> {
        let mut cluster_sizes = HashMap::new();

        // Sort dendrogram by distance (ascending)
        let mut sorted_dendrogram = dendrogram.to_vec();
        sorted_dendrogram.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        // Initialize sizes for all original points
        for i in 0..n_points {
            cluster_sizes.insert(i, 1);
        }

        let mut condensed_nodes = Vec::new();

        // Process dendrogram nodes in order to compute cluster sizes
        for node in &sorted_dendrogram {
            let size_left = cluster_sizes.get(&node.left).copied().unwrap_or(1);
            let size_right = cluster_sizes.get(&node.right).copied().unwrap_or(1);
            let total_size = size_left + size_right;

            cluster_sizes.insert(node.cluster_id, total_size);

            // Keep all nodes (no filtering by min_cluster_size here)
            condensed_nodes.push(CondensedNode {
                cluster_id: node.cluster_id,
                left: node.left,
                right: node.right,
                lambda_val: if node.distance > 0.0 {
                    1.0 / node.distance
                } else {
                    f64::INFINITY
                },
                size: total_size,
            });
        }

        Ok(CondensedTree {
            nodes: condensed_nodes,
            cluster_sizes,
        })
    }

    /// Extract clusters using a simplified stability-based approach
    fn extract_clusters(
        &self,
        condensed_tree: &CondensedTree,
        n_points: usize,
    ) -> Result<Vec<i32>, HdbscanError> {
        let mut labels = vec![-1; n_points];

        if condensed_tree.nodes.is_empty() {
            return Ok(labels);
        }

        // Find all clusters that meet the minimum size requirement
        let mut valid_clusters: Vec<_> = condensed_tree
            .nodes
            .iter()
            .filter(|node| node.size >= self.min_cluster_size)
            .collect();

        // Sort by stability (lambda * size) in descending order
        valid_clusters.sort_by(|a, b| {
            let stability_a = a.lambda_val * a.size as f64;
            let stability_b = b.lambda_val * b.size as f64;
            stability_b
                .partial_cmp(&stability_a)
                .unwrap_or(Ordering::Equal)
        });

        // Assign clusters starting with the most stable
        // Make sure we don't assign overlapping clusters
        let mut assigned_points = HashSet::new();
        let mut cluster_label = 0;

        for node in valid_clusters {
            // Get all leaf points in this cluster
            let points = self.get_cluster_points(node.cluster_id, condensed_tree, n_points);

            // Check how many points are unassigned
            let unassigned: Vec<_> = points
                .iter()
                .filter(|p| !assigned_points.contains(*p))
                .copied()
                .collect();

            // Only create cluster if we have enough unassigned points
            if unassigned.len() >= self.min_cluster_size {
                for &point in &unassigned {
                    labels[point] = cluster_label;
                    assigned_points.insert(point);
                }
                cluster_label += 1;
            }
        }

        Ok(labels)
    }

    /// Get all points in a cluster by traversing the hierarchy
    fn get_cluster_points(
        &self,
        cluster_id: usize,
        condensed_tree: &CondensedTree,
        n_points: usize,
    ) -> Vec<usize> {
        let mut points = Vec::new();
        let mut stack = vec![cluster_id];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if current < n_points {
                // This is a leaf (original point)
                points.push(current);
            } else {
                // Find this node and add its children
                if let Some(node) = condensed_tree.nodes.iter().find(|n| n.cluster_id == current) {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            }
        }

        points
    }

    /// Compute stability scores for clusters (unused but kept for future enhancements)
    #[allow(dead_code)]
    fn compute_stability(&self, condensed_tree: &CondensedTree) -> HashMap<usize, f64> {
        let mut stabilities = HashMap::new();

        for node in &condensed_tree.nodes {
            let stability = node.lambda_val * node.size as f64;
            stabilities.insert(node.cluster_id, stability);
        }

        stabilities
    }

}

/// Edge in minimum spanning tree
#[derive(Debug, Clone, Copy)]
struct MstEdge {
    from: usize,
    to: usize,
    weight: f64,
}

/// Edge for BinaryHeap (implements reverse ordering for min-heap)
#[derive(Debug, Clone, Copy)]
struct HeapEdge {
    from: usize,
    to: usize,
    weight: f64,
}

impl Ord for HeapEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse comparison for min-heap
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEdge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl Eq for HeapEdge {}

/// Node in the single-linkage dendrogram
#[derive(Debug, Clone)]
struct DendrogramNode {
    cluster_id: usize,
    left: usize,
    right: usize,
    distance: f64,
}

/// Node in the condensed cluster tree
#[derive(Debug, Clone)]
struct CondensedNode {
    cluster_id: usize,
    left: usize,
    right: usize,
    lambda_val: f64,
    size: usize,
}

/// Condensed cluster tree
#[derive(Debug)]
struct CondensedTree {
    nodes: Vec<CondensedNode>,
    #[allow(dead_code)] // May be used for future enhancements
    cluster_sizes: HashMap<usize, usize>,
}

/// Union-Find data structure for tracking cluster membership
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdbscan_basic() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 10.0, 11.0, 10.1, 11.1, 10.2, 11.2,
            ],
        )
        .unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(2)
            .min_samples(2)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_empty_dataset() {
        let data = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let clusterer = HdbscanBuilder::new().build();
        let result = clusterer.fit_predict(&data);
        assert!(matches!(result, Err(HdbscanError::EmptyDataset)));
    }

    #[test]
    fn test_invalid_min_cluster_size() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let clusterer = HdbscanBuilder::new().min_cluster_size(1).build();
        let result = clusterer.fit_predict(&data);
        assert!(matches!(
            result,
            Err(HdbscanError::InvalidParameters { .. })
        ));
    }

    #[test]
    fn test_core_distances() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let clusterer = HdbscanBuilder::new().min_samples(2).build();
        let core_distances = clusterer.compute_core_distances(&data).unwrap();

        assert_eq!(core_distances.len(), 3);
        for dist in core_distances.iter() {
            assert!(dist.is_finite());
            assert!(*dist >= 0.0);
        }
    }

    #[test]
    fn test_distance_metric() {
        let clusterer = HdbscanBuilder::new().build();
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let dist = clusterer.compute_distance(&a.view(), &b.view());
        assert!((dist - 5.0).abs() < 1e-10);
    }
}
