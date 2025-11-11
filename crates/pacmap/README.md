# PACMAP - Pure Rust Implementation

Pure Rust implementation of PACMAP (Pairwise Controlled Manifold Approximation Projection), a dimensionality reduction algorithm that preserves both local and global structure.

## Features

- **Pure Rust**: No external ML framework dependencies
- **Fast**: Optimized implementation, faster than t-SNE or UMAP
- **Structure Preservation**: Maintains both local neighborhoods and global topology
- **Configurable**: Flexible parameters for different use cases
- **Reproducible**: Seed support for deterministic results

## Usage

```rust
use pacmap::PacmapBuilder;
use ndarray::Array2;

// Create high-dimensional data
let data = Array2::from_shape_vec((100, 50),
    (0..5000).map(|x| x as f64).collect()
).unwrap();

// Configure PACMAP
let pacmap = PacmapBuilder::new()
    .n_components(2)      // Output dimensions
    .n_neighbors(10)      // Local structure preservation
    .n_iterations(450)    // Optimization iterations
    .seed(Some(42))       // For reproducibility
    .build();

// Transform data
let embedding = pacmap.fit_transform(&data).unwrap();
assert_eq!(embedding.dim(), (100, 2));
```

## Parameters

- **`n_components`** (default: 2): Number of output dimensions
- **`n_neighbors`** (default: 10): Number of nearest neighbors for local structure
- **`n_mid_near`** (default: 5): Mid-range pairs for bridging local/global
- **`n_far`** (default: 5): Far pairs for global structure
- **`n_iterations`** (default: 450): Optimization iterations
- **`learning_rate`** (default: 1.0): Gradient descent step size
- **`seed`** (optional): Random seed for reproducibility

## Algorithm

PACMAP balances three objectives through pairwise distances:

1. **Near pairs** (k-NN): Preserve local neighborhoods
2. **Mid-near pairs**: Bridge between local and global structure
3. **Far pairs**: Maintain global topology

The algorithm optimizes these objectives using gradient descent, producing embeddings that preserve multi-scale structure better than traditional methods like PCA.

## Performance

- **Time Complexity**: O(n² + k·n·d) where k=iterations, d=dimensions
- **Space Complexity**: O(n·k + n·d)
- Efficient for datasets with 100-100,000 points

## References

Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). "Understanding How Dimension Reduction Tools Work". JMLR, 22(201), 1-73.