# Scanner Embeddings - Pure Rust ML Algorithms for Embeddings

A Cargo workspace providing pure Rust implementations of HDBSCAN clustering and PACMAP dimensionality reduction, with WebAssembly support for browser and Node.js environments.

## Features

- **Pure Rust Implementations**: No external ML framework dependencies
- **Modular Design**: Three separate crates for different use cases
- **WASM Support**: Ready for web deployment via `embedding-tools` crate
- **Extensively Tested**: 119+ comprehensive tests including property-based testing
- **Zero Clippy Warnings**: Clean, idiomatic Rust code
- **High Performance**: Optimized implementations with configurable parameters

## Workspace Structure

This project is organized as a Cargo workspace with three crates:

### 1. `pacmap` - Pure Rust PACMAP Implementation
Located in `crates/pacmap/`

PACMAP (Pairwise Controlled Manifold Approximation Projection) for dimensionality reduction that preserves both local and global structure.

**Features:**
- Preserves multi-scale structure better than PCA
- Faster than t-SNE or UMAP
- Configurable components, neighbors, and iterations
- Reproducible with seed support

### 2. `hdbscan` - Pure Rust HDBSCAN Implementation
Located in `crates/hdbscan/`

HDBSCAN (Hierarchical Density-Based Spatial Clustering) for finding clusters of varying densities.

**Features:**
- Finds clusters without specifying count
- Identifies noise points/outliers
- Robust to parameter changes
- Handles varying cluster densities

### 3. `embedding-tools` - WASM-Compatible Pipeline
Located in `crates/embedding-tools/`

WebAssembly bindings combining both algorithms into an easy-to-use pipeline.

**Features:**
- Full pipeline API (reduction + clustering)
- Separate functions for each step
- Browser and Node.js compatible
- Configurable processing options

## Quick Start

### Rust Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
pacmap = { path = "crates/pacmap" }
hdbscan = { path = "crates/hdbscan" }
# Or for the full pipeline:
embedding-tools = { path = "crates/embedding-tools" }
```

#### Basic HDBSCAN Clustering

```rust
use hdbscan::HdbscanBuilder;
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

#### PACMAP Dimensionality Reduction

```rust
use pacmap::PacmapBuilder;
use ndarray::Array2;

// High-dimensional data (100 points × 50 dimensions)
let data = Array2::zeros((100, 50));

// Reduce to 2 dimensions using PACMAP
let pacmap = PacmapBuilder::new()
    .n_components(2)
    .n_neighbors(10)
    .n_iterations(450)
    .build();

let embedding = pacmap.fit_transform(&data).unwrap();
// embedding: 100 × 2 array
```

#### Complete Pipeline

```rust
use pacmap::PacmapBuilder;
use hdbscan::HdbscanBuilder;
use ndarray::Array2;

// High-dimensional data
let data = Array2::zeros((100, 50));

// Step 1: Reduce dimensions
let pacmap = PacmapBuilder::new()
    .n_components(2)
    .n_neighbors(10)
    .build();
let reduced = pacmap.fit_transform(&data).unwrap();

// Step 2: Cluster reduced data
let clusterer = HdbscanBuilder::new()
    .min_cluster_size(5)
    .min_samples(3)
    .build();
let labels = clusterer.fit_predict(&reduced).unwrap();
```

### WebAssembly Usage

Build the WASM module:

```bash
# Install wasm-pack if needed
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
cd crates/embedding-tools
wasm-pack build --target web
```

JavaScript usage:

```javascript
import { process_embeddings, reduce_dimensions, cluster_data } from 'embedding-tools';

// Full pipeline
const input = {
  data: [[1, 2, 3], [4, 5, 6], ...],  // Your embeddings
  config: {
    n_components: 2,
    min_cluster_size: 5,
    n_neighbors: 10,
    n_iterations: 450
  }
};

const result = process_embeddings(input);
// result.reduced_data - 2D coordinates
// result.clusters - cluster assignments
// result.metadata - processing info

// Or use individual functions
const reduced = reduce_dimensions(data, 2);
const clusters = cluster_data(reduced, 5);
```

## Algorithm Parameters

### HDBSCAN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 5 | Minimum points to form a cluster. Smaller = more clusters |
| `min_samples` | 5 | Core distance k-neighbors. Controls conservativeness |
| `metric` | Euclidean | Distance metric (currently only Euclidean) |

### PACMAP Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output dimensions (typically 2-3 for visualization) |
| `n_neighbors` | 10 | Local structure preservation (5-50 typical) |
| `n_mid_near` | 5 | Mid-range pairs (usually n_neighbors/2) |
| `n_far` | 5 | Global structure pairs |
| `n_iterations` | 450 | Optimization iterations (250-500 typical) |
| `learning_rate` | 1.0 | Gradient descent step size (0.5-2.0) |
| `seed` | 42 | Random seed for reproducibility |

## Building & Testing

```bash
# Build all crates
cargo build

# Build release version
cargo build --release

# Run all tests (119+ tests)
cargo test

# Run specific test suites
cargo test --lib                       # Unit tests
cargo test --test hdbscan_validation   # HDBSCAN validation
cargo test --test pacmap_validation    # PACMAP validation
cargo test --test integration_tests    # Pipeline integration
cargo test --test property_tests       # Property-based testing

# Check code quality
cargo clippy -- -W clippy::all
cargo fmt --check
```

## Performance Characteristics

### HDBSCAN
- **Time Complexity**: O(n² log n) for n points
- **Space Complexity**: O(n²) for distance matrix
- Optimized for datasets with 10-10,000 points

### PACMAP
- **Time Complexity**: O(n² + k·n·d) where k=iterations, d=dimensions
- **Space Complexity**: O(n·k + n·d)
- Efficient for high-dimensional reduction

## Implementation Details

### HDBSCAN Algorithm
1. **Core Distance**: Distance to k-th nearest neighbor
2. **Mutual Reachability**: max(core_dist(a), core_dist(b), dist(a,b))
3. **MST Construction**: Prim's algorithm with mutual reachability distances
4. **Hierarchy Building**: Process MST edges by weight
5. **Condensation**: Simplify based on min_cluster_size
6. **Extraction**: Select stable clusters by persistence

### PACMAP Algorithm
1. **k-NN Graph**: Find nearest neighbors
2. **Pair Selection**: Near (local), mid (bridge), far (global) pairs
3. **Initialization**: Random scaled by data variance
4. **Optimization**: Gradient descent balancing three objectives
5. **Convergence**: Typically 250-500 iterations

## Test Coverage

The implementation includes comprehensive test coverage:

- ✅ **Edge Cases**: Empty datasets, single points, identical values
- ✅ **Clustering Patterns**: Well-separated, varying density, nested structures
- ✅ **Manifolds**: Swiss roll, S-curve, concentric circles
- ✅ **High Dimensions**: Up to 100D data
- ✅ **Numerical Stability**: Scale invariance (10^-10 to 10^10)
- ✅ **Pathological Cases**: Grids, spirals, moons
- ✅ **Property-Based**: 256 random cases per property
- ✅ **Integration**: Full pipeline validation

## Comparison with Other Implementations

### vs. scikit-learn
- Same core algorithms with comparable results
- Pure Rust (no Python/NumPy dependencies)
- Simplified API focused on core functionality
- WASM-compatible for browser deployment

### vs. linfa-clustering
- Self-contained implementation
- Better WASM integration
- No dependency conflicts
- More extensive test coverage

## Known Limitations

1. **Distance Metrics**: Currently only Euclidean
2. **Memory Usage**: O(n²) for distance matrix
3. **Incremental Updates**: Requires full recomputation
4. **Visualization**: No built-in plotting

## Future Enhancements

- [ ] Additional distance metrics (Manhattan, Cosine)
- [ ] Approximate nearest neighbors for large datasets
- [ ] Soft clustering probabilities
- [ ] GPU acceleration via wgpu
- [ ] Incremental clustering updates

## References

### HDBSCAN
1. Campello, R. J., Moulavi, D., & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates". Pacific-Asia Conference on Knowledge Discovery and Data Mining

2. McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering". Journal of Open Source Software

### PACMAP
3. Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization". Journal of Machine Learning Research, 22(201), 1-73

## License

MIT OR Apache-2.0 (dual licensed)

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Ensure `cargo test` passes
3. Run `cargo clippy` and fix warnings
4. Update documentation
