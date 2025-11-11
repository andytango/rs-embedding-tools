# HDBSCAN - Pure Rust Implementation

Pure Rust implementation of HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), a clustering algorithm that finds clusters of varying densities.

## Features

- **Pure Rust**: No external dependencies for the core algorithm
- **Automatic Cluster Detection**: No need to specify number of clusters
- **Noise Handling**: Identifies outliers and noise points
- **Varying Densities**: Handles clusters with different densities
- **Robust**: Stable results across parameter variations

## Usage

```rust
use hdbscan::HdbscanBuilder;
use ndarray::Array2;

// Create sample data with two clusters
let data = Array2::from_shape_vec((10, 2), vec![
    // Cluster 1
    0.0, 0.0,
    0.1, 0.1,
    0.2, 0.0,
    0.1, 0.2,
    0.0, 0.1,
    // Cluster 2
    10.0, 10.0,
    10.1, 10.1,
    10.2, 10.0,
    10.1, 10.2,
    10.0, 10.1,
]).unwrap();

// Configure and run HDBSCAN
let clusterer = HdbscanBuilder::new()
    .min_cluster_size(3)  // Minimum points to form a cluster
    .min_samples(2)       // Core distance parameter
    .build();

let labels = clusterer.fit_predict(&data).unwrap();
// Result: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
// -1 indicates noise points
```

## Parameters

- **`min_cluster_size`** (default: 5): Minimum number of points required to form a cluster
  - Smaller values: Find more, smaller clusters
  - Larger values: More conservative, fewer clusters

- **`min_samples`** (default: 5): Number of neighbors for core distance calculation
  - Controls how conservative the clustering is
  - Usually set equal to or slightly less than `min_cluster_size`

- **`metric`** (default: Euclidean): Distance metric for clustering
  - Currently only supports Euclidean distance

## Algorithm Overview

1. **Core Distance Calculation**: Find distance to k-th nearest neighbor for each point
2. **Mutual Reachability**: Transform distances to handle varying densities
3. **Minimum Spanning Tree**: Build MST using mutual reachability distances
4. **Hierarchy Construction**: Create dendrogram from MST
5. **Condensation**: Simplify hierarchy based on `min_cluster_size`
6. **Stability Extraction**: Select most stable clusters

## Performance

- **Time Complexity**: O(n² log n) for n points
- **Space Complexity**: O(n²) for distance matrix
- Optimized for datasets with 10-10,000 points

## Output

- Returns a vector of cluster labels (i32)
- Cluster IDs start from 0 and are sequential
- Noise/outlier points are marked as -1

## References

1. Campello, R. J., Moulavi, D., & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates". PAKDD.

2. McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering". JOSS.