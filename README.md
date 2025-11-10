# Scanner Embeddings - Pure Rust HDBSCAN Implementation

A pure Rust implementation of the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering algorithm, along with PACMAP for dimensionality reduction.

## Features

- **Pure Rust HDBSCAN**: No external dependencies for the core algorithm
- **PACMAP Implementation**: Pure Rust Pairwise Controlled Manifold Approximation Projection for dimensionality reduction
- **WASM Support**: Compile to WebAssembly for use in web applications
- **Extensively Tested**: 119 comprehensive tests including property-based testing
- **Zero Clippy Warnings**: Clean, idiomatic Rust code

## Algorithm Overview

HDBSCAN is a density-based clustering algorithm that:
1. Finds clusters of varying densities
2. Identifies noise points (outliers)
3. Doesn't require specifying the number of clusters
4. Is robust to parameter changes

### How It Works

1. **Core Distance Calculation**: For each point, compute the distance to its k-th nearest neighbor
2. **Mutual Reachability Distance**: Transform the distance space to handle varying densities
3. **Minimum Spanning Tree**: Build an MST using mutual reachability distances
4. **Hierarchy Construction**: Create a dendrogram from the MST
5. **Cluster Condensation**: Simplify the hierarchy based on `min_cluster_size`
6. **Stability Extraction**: Select the most stable clusters based on persistence

## Usage

### Basic Clustering

```rust
use scanner_embeddings::hdbscan::HdbscanBuilder;
use ndarray::Array2;

// Create sample data (10 points in 2D)
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

// labels: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
// -1 indicates noise points
```

### With PACMAP Dimensionality Reduction

```rust
use scanner_embeddings::pacmap::PacmapBuilder;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use ndarray::Array2;

// High-dimensional data (100 points × 50 dimensions)
let data = Array2::zeros((100, 50));

// Reduce to 2 dimensions using PACMAP
let pacmap = PacmapBuilder::new()
    .n_components(2)
    .n_neighbors(10)
    .n_iterations(450)
    .build();
let reduced = pacmap.fit_transform(&data).unwrap();

// Cluster the reduced data
let clusterer = HdbscanBuilder::new()
    .min_cluster_size(5)
    .min_samples(3)
    .build();

let labels = clusterer.fit_predict(&reduced).unwrap();
```

### WASM Integration

```rust
use wasm_bindgen::prelude::*;
use scanner_embeddings::{process_embeddings, InputData};

#[wasm_bindgen]
pub fn cluster_data(data: JsValue) -> Result<JsValue, JsValue> {
    // data should be: { data: [[x, y, z], ...] }
    process_embeddings(data)
    // returns: { clusters: [0, 1, -1, ...], reduced_data: [[x, y], ...] }
}
```

## Parameters

### HDBSCAN Parameters

- **`min_cluster_size`** (default: 5)
  - Minimum number of points to form a cluster
  - Smaller values find more, smaller clusters
  - Larger values are more conservative

- **`min_samples`** (default: 5)
  - Number of neighbors to consider for core distance
  - Controls how conservative the clustering is
  - Generally keep close to `min_cluster_size`

- **`metric`** (default: Euclidean)
  - Distance metric to use
  - Currently supports: Euclidean

### PACMAP Parameters

- **`n_components`** (default: 2)
  - Number of dimensions in the output embedding
  - Must be less than the number of input features
  - Common values: 2 or 3 for visualization

- **`n_neighbors`** (default: 10)
  - Number of nearest neighbors to consider for local structure
  - Smaller values preserve more local structure
  - Larger values preserve more global structure
  - Typical range: 5-50

- **`n_mid_near`** (default: 5)
  - Number of mid-near pairs per point
  - These pairs help bridge local and global structure
  - Usually set to half of n_neighbors

- **`n_far`** (default: 5)
  - Number of far pairs per point
  - These pairs help preserve global structure
  - Usually set equal to n_mid_near

- **`n_iterations`** (default: 450)
  - Number of optimization iterations
  - More iterations may improve quality but increase runtime
  - Typical range: 250-500

- **`learning_rate`** (default: 1.0)
  - Step size for gradient descent
  - Higher values converge faster but may be unstable
  - Typical range: 0.5-2.0

- **`seed`** (optional)
  - Random seed for reproducibility
  - If not set, uses default seed of 42

## Performance Characteristics

### HDBSCAN
- **Time Complexity**: O(n² log n) for n points
  - Core distance: O(n²)
  - MST construction: O(n² log n)
  - Hierarchy building: O(n)
- **Space Complexity**: O(n²) for distance computations

### PACMAP
- **Time Complexity**: O(n² + k·n·d) where k is iterations, d is dimensions
  - k-NN construction: O(n²)
  - Per iteration: O(n·d)
- **Space Complexity**: O(n·k + n·d) for neighbors and embedding

## Implementation Details

### Core Distance

The core distance of a point is the distance to its k-th nearest neighbor, where k = `min_samples`. This helps normalize density across the dataset.

```rust
// For point i with k neighbors
core_dist[i] = distance_to_kth_neighbor(i, k)
```

### Mutual Reachability Distance

Transforms distances to handle varying densities:

```rust
mutual_reach_dist(a, b) = max(
    core_dist(a),
    core_dist(b),
    euclidean_dist(a, b)
)
```

### Minimum Spanning Tree

Uses Prim's algorithm to construct an MST with O(n² log n) complexity:
1. Start from an arbitrary node
2. Greedily add the minimum-weight edge connecting tree to non-tree nodes
3. Repeat until all nodes are included

### Cluster Extraction

Selects stable clusters based on:
- Size (must be ≥ `min_cluster_size`)
- Stability score (λ × cluster_size, where λ = 1/distance)
- Non-overlapping constraint (points assigned to most stable cluster)

### PACMAP Algorithm

PACMAP (Pairwise Controlled Manifold Approximation Projection) preserves both local and global structure:

1. **k-Nearest Neighbors**: Find k nearest neighbors for each point
2. **Pair Selection**: Create three types of pairs:
   - **Near pairs**: From k-nearest neighbors (preserve local structure)
   - **Mid-near pairs**: From 2nd-order neighbors (bridge local/global)
   - **Far pairs**: Random distant points (preserve global structure)
3. **Initialization**: Random initialization scaled by data variance
4. **Optimization**: Gradient descent with three loss terms:
   - Pull near pairs together (weight: 2.0)
   - Pull mid-near pairs moderately (weight: 1.0)
   - Push far pairs apart (weight: 1.0)

```rust
// Loss for different pair types
loss_near = weight_near * (distance - target_near)²
loss_mid = weight_mid * (distance - target_mid)²
loss_far = weight_far * max(0, target_far - distance)²
```

The algorithm balances these three objectives through gradient descent, producing embeddings that preserve multi-scale structure.

## Testing

The implementation includes 119 comprehensive tests across 9 test suites:

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --lib                       # Unit tests (14 tests)
cargo test --test hdbscan_validation   # HDBSCAN validation (11 tests)
cargo test --test hdbscan_edge_cases   # HDBSCAN edge cases (16 tests)
cargo test --test pacmap_validation    # PACMAP validation (13 tests)
cargo test --test integration_tests    # Pipeline integration (9 tests)
cargo test --test numerical_accuracy   # Numerical stability (14 tests)
cargo test --test stress_tests         # Performance/scale (12 tests)
cargo test --test pathological_cases   # Edge patterns (18 tests)
cargo test --test property_tests       # Property-based testing (10 tests)

# Run with output
cargo test -- --nocapture
```

### Property-Based Testing

The test suite includes property-based tests using proptest:
- Validates invariants across randomly generated inputs
- Tests HDBSCAN determinism, label validity, cluster size enforcement
- Tests PACMAP sample preservation, output finiteness, reproducibility
- Validates the complete PACMAP → HDBSCAN pipeline
- Runs 256 random test cases per property by default

### Test Coverage

**HDBSCAN Tests (49 tests):**
- ✅ Empty datasets & parameter validation
- ✅ Single/two points & minimum datasets
- ✅ Identical & collinear points
- ✅ Well-separated & varying density clusters
- ✅ High dimensions (up to 100D)
- ✅ Outlier & noise detection
- ✅ Nested & concentric structures
- ✅ Parameter sensitivity analysis
- ✅ Numerical stability (10^-10 to 10^10)
- ✅ Pathological patterns (grids, spirals, moons)
- ✅ Reproducibility & determinism
- ✅ Label validity & sequential IDs

**PACMAP Tests (42 tests):**
- ✅ Basic transformation & shape preservation
- ✅ Blob clusters structure retention
- ✅ Manifold unrolling (swiss roll, S-curve)
- ✅ High-dimensional reduction (100D → 2D)
- ✅ N-dimensional outputs (2D, 3D, 5D, etc.)
- ✅ Parameter variations & sensitivity
- ✅ Local structure preservation (k-NN)
- ✅ Global structure preservation (distances)
- ✅ Numerical stability across scales
- ✅ Reproducibility with seeds
- ✅ Topological patterns (rings, spirals)
- ✅ Edge cases & small datasets

**Integration Tests (9 tests):**
- ✅ Full pipeline (PACMAP → HDBSCAN)
- ✅ Cluster separation preservation
- ✅ Noise handling in pipeline
- ✅ Parameter sensitivity end-to-end
- ✅ Multi-dimensional workflows
- ✅ Output validation (no NaN/Inf)

**Stress Tests (12 tests):**
- ✅ Scaling behavior (20-200 points)
- ✅ High-dimensional data (up to 100D)
- ✅ Varying cluster sizes
- ✅ Elongated & sparse clusters
- ✅ Performance benchmarks

## Linting

Zero clippy warnings in strict mode:

```bash
cargo clippy --all-targets
```

## Building for WASM

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Build for nodejs
wasm-pack build --target nodejs
```

## Comparison with Other Implementations

### vs. scikit-learn HDBSCAN

**Similarities:**
- Same core algorithm (MST-based hierarchy)
- Similar parameter interface
- Comparable results on standard datasets

**Differences:**
- Pure Rust (no Python/NumPy dependencies)
- Simplified stability calculation
- No hierarchical cluster tree visualization
- Smaller feature set (focused on core functionality)

### vs. linfa-clustering

**Why not use linfa?**
- linfa 0.7 doesn't include HDBSCAN
- linfa 0.8+ has dependency conflicts
- This implementation is self-contained
- Better integration with WASM targets

### vs. Python PACMAP

**Similarities:**
- Same core algorithm (pairwise distance preservation)
- Similar parameter interface (n_neighbors, n_components)
- Comparable embedding quality

**Differences:**
- Pure Rust (no Python dependencies)
- Simplified pair selection (no advanced sampling strategies)
- Fixed weights (not adaptive)
- No GPU acceleration
- Focused on core functionality

## Known Limitations

1. **Distance Metrics**: Currently only Euclidean
   - Manhattan, Cosine, etc. not yet implemented

2. **Memory Usage**: O(n²) for distance matrix
   - Could be optimized for sparse data

3. **Visualization**: No built-in plotting
   - Export labels for external visualization

4. **Incremental Updates**: Requires full recomputation
   - No support for adding new points

## Future Enhancements

- [ ] Additional distance metrics (Manhattan, Cosine)
- [ ] Approximate nearest neighbors for large datasets
- [ ] Hierarchical cluster tree output
- [ ] Soft clustering (cluster membership probabilities)
- [ ] Parallel MST construction
- [ ] GPU acceleration via wgpu

## References

### HDBSCAN
1. Campello, R. J., Moulavi, D., & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates". In Pacific-Asia Conference on Knowledge Discovery and Data Mining
2. McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering". Journal of Open Source Software

### PACMAP
3. Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization". Journal of Machine Learning Research, 22(201), 1-73

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Ensure `cargo test` passes
3. Run `cargo clippy` and fix warnings
4. Update documentation

## Author

Generated with [Claude Code](https://claude.com/claude-code)
