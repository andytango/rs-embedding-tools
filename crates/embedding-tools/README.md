# Embedding Tools - WASM-Compatible Pipeline

WebAssembly-compatible crate providing a simplified pipeline for processing high-dimensional embeddings with dimensionality reduction and clustering.

## Features

- **WASM Support**: Ready for browser and Node.js deployment
- **Complete Pipeline**: Combines PACMAP and HDBSCAN in one easy API
- **Flexible**: Use the full pipeline or individual components
- **Configurable**: Extensive options for fine-tuning
- **Type-Safe**: Full TypeScript definitions when built with wasm-pack

## Installation

### For Rust Projects

```toml
[dependencies]
embedding-tools = { path = "crates/embedding-tools" }
```

### For JavaScript/TypeScript

```bash
# Build WASM module
wasm-pack build --target web

# Or for Node.js
wasm-pack build --target nodejs
```

## Usage

### Rust API

```rust
use embedding_tools::{EmbeddingPipeline, PipelineConfig, EmbeddingInput};
use ndarray::Array2;

// Configure pipeline
let config = PipelineConfig {
    n_components: 2,
    min_cluster_size: 5,
    n_neighbors: Some(10),
    n_iterations: 450,
    skip_clustering: false,
    ..Default::default()
};

// Create pipeline
let pipeline = EmbeddingPipeline::new(config);

// Process data
let data = Array2::zeros((100, 50));
let output = pipeline.process(data).unwrap();

// Access results
println!("Reduced data: {:?}", output.reduced_data);
println!("Clusters: {:?}", output.clusters);
println!("Metadata: {:?}", output.metadata);
```

### JavaScript API

```javascript
import { process_embeddings, reduce_dimensions, cluster_data } from './pkg';

// Full pipeline with configuration
const input = {
  data: [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    // ... more embeddings
  ],
  config: {
    n_components: 2,
    n_neighbors: 10,
    n_iterations: 450,
    min_cluster_size: 5,
    min_samples: 5,
    skip_clustering: false
  }
};

const result = await process_embeddings(input);
console.log('Reduced:', result.reduced_data);
console.log('Clusters:', result.clusters);
console.log('Metadata:', result.metadata);

// Or use individual functions
const reduced = await reduce_dimensions(data, 2);
const clusters = await cluster_data(reduced, 5);
```

## API Functions

### `process_embeddings(input)`
Full pipeline: dimensionality reduction + clustering

**Input:**
- `data`: 2D array of embeddings (samples × features)
- `config`: Optional configuration object

**Output:**
- `reduced_data`: 2D coordinates after reduction
- `clusters`: Cluster assignments (null for noise)
- `metadata`: Processing information

### `reduce_dimensions(data, n_components)`
Only perform dimensionality reduction

**Input:**
- `data`: 2D array of embeddings
- `n_components`: Target dimensions (default: 2)

**Output:**
- 2D array of reduced coordinates

### `cluster_data(data, min_cluster_size)`
Only perform clustering on already-reduced data

**Input:**
- `data`: 2D array (typically already reduced)
- `min_cluster_size`: Minimum cluster size (default: 5)

**Output:**
- Array of cluster assignments

## Configuration Options

```typescript
interface PipelineConfig {
  // Dimensionality reduction
  n_components?: number;      // Output dimensions (default: 2)
  n_neighbors?: number;        // k-NN for PACMAP (default: auto)
  n_iterations?: number;       // PACMAP iterations (default: 450)

  // Clustering
  min_cluster_size?: number;   // Min cluster size (default: 5)
  min_samples?: number;        // Core distance k (default: 5)
  skip_clustering?: boolean;   // Skip clustering step (default: false)
}
```

## Building for Production

```bash
# Optimize for size
wasm-pack build --release -- --features wee_alloc

# Include debug symbols
wasm-pack build --dev

# Custom output directory
wasm-pack build --out-dir ./wasm-dist
```

## Performance Tips

1. **Batch Processing**: Process multiple embeddings at once
2. **Pre-reduce**: For very high dimensions, consider PCA first
3. **Parameter Tuning**: Start with defaults, then adjust based on results
4. **Memory**: WASM has memory limits; consider chunking large datasets

## Error Handling

All functions return Results with descriptive error messages:

```javascript
try {
  const result = await process_embeddings(input);
} catch (error) {
  console.error('Processing failed:', error);
}
```

## License

MIT OR Apache-2.0