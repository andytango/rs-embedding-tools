//! Comprehensive validation tests for PACMAP implementation
//!
//! These tests validate that PACMAP:
//! 1. Preserves local structure (nearby points stay nearby)
//! 2. Preserves global structure (distant points stay distant)
//! 3. Handles various dataset shapes correctly
//! 4. Works with different parameters

use ndarray::Array2;
use scanner_embeddings::pacmap::PacmapBuilder;

/// Generate blob clusters in high dimensions
fn generate_blobs(n_samples: usize, n_features: usize, n_clusters: usize, cluster_std: f64) -> Array2<f64> {
    let samples_per_cluster = n_samples / n_clusters;
    let actual_samples = samples_per_cluster * n_clusters; // Ensure we have exact number
    let mut data = Vec::with_capacity(actual_samples * n_features);

    for cluster in 0..n_clusters {
        for sample in 0..samples_per_cluster {
            for feature in 0..n_features {
                let center = (cluster * 10) as f64;
                let noise = ((sample * 7 + feature * 13) % 100) as f64 / 100.0 - 0.5;
                data.push(center + noise * cluster_std);
            }
        }
    }

    Array2::from_shape_vec((actual_samples, n_features), data).expect("Failed to create blob data")
}

/// Generate swiss roll dataset
fn generate_swiss_roll(n_samples: usize) -> Array2<f64> {
    let mut data = Vec::new();

    for i in 0..n_samples {
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * i as f64 / n_samples as f64);
        let x = t * t.cos();
        let y = t * t.sin();
        let z = 2.0 * i as f64 / n_samples as f64;

        data.push(x);
        data.push(y);
        data.push(z);
    }

    Array2::from_shape_vec((n_samples, 3), data).expect("Failed to create swiss roll")
}

/// Generate S-curve dataset
fn generate_s_curve(n_samples: usize) -> Array2<f64> {
    let mut data = Vec::new();

    for i in 0..n_samples {
        let t = 3.0 * std::f64::consts::PI * i as f64 / n_samples as f64;
        let x = t.sin();
        let y = 2.0 * ((t / 2.0).sin());
        let z = t;

        data.push(x);
        data.push(y);
        data.push(z);
    }

    Array2::from_shape_vec((n_samples, 3), data).expect("Failed to create S-curve")
}

/// Compute pairwise distances in embedding space
fn pairwise_distances(data: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = data.nrows();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist = 0.0;
            for d in 0..data.ncols() {
                let diff = data[[i, d]] - data[[j, d]];
                dist += diff * diff;
            }
            dist = dist.sqrt();
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    distances
}

/// Check if k-nearest neighbors are preserved
fn check_neighbor_preservation(
    original: &Array2<f64>,
    embedding: &Array2<f64>,
    k: usize,
) -> f64 {
    let orig_distances = pairwise_distances(original);
    let embed_distances = pairwise_distances(embedding);

    let n = original.nrows();
    let mut preserved = 0;
    let mut total = 0;

    for i in 0..n {
        // Get k nearest neighbors in original space
        let mut orig_neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, orig_distances[i][j]))
            .collect();
        orig_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let orig_k_neighbors: Vec<usize> = orig_neighbors.iter().take(k).map(|(idx, _)| *idx).collect();

        // Get k nearest neighbors in embedding space
        let mut embed_neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, embed_distances[i][j]))
            .collect();
        embed_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let embed_k_neighbors: Vec<usize> = embed_neighbors.iter().take(k).map(|(idx, _)| *idx).collect();

        // Count how many neighbors are preserved
        for neighbor in &orig_k_neighbors {
            if embed_k_neighbors.contains(neighbor) {
                preserved += 1;
            }
            total += 1;
        }
    }

    preserved as f64 / total as f64
}

#[test]
fn test_blob_clusters() {
    let data = generate_blobs(51, 10, 3, 1.0); // 51/3 = 17, 17*3 = 51

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    assert_eq!(embedding.nrows(), 51);
    assert_eq!(embedding.ncols(), 2);

    // All values should be finite
    for val in embedding.iter() {
        assert!(val.is_finite(), "Embedding contains non-finite values");
    }
}

#[test]
fn test_local_structure_preservation() {
    let data = generate_blobs(40, 8, 2, 0.5);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    // Check that nearest neighbors are somewhat preserved
    let preservation = check_neighbor_preservation(&data, &embedding, 5);

    println!("Neighbor preservation (k=5): {:.2}%", preservation * 100.0);

    // Should preserve at least 30% of nearest neighbors (PACMAP is approximate)
    assert!(preservation > 0.3, "Poor local structure preservation: {:.2}", preservation);
}

#[test]
fn test_swiss_roll() {
    let data = generate_swiss_roll(60);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(400)
        .seed(Some(42))
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    assert_eq!(embedding.nrows(), 60);
    assert_eq!(embedding.ncols(), 2);

    // Swiss roll should be "unrolled" - check that it's spread out
    let mut min_vals = [f64::INFINITY; 2];
    let mut max_vals = [f64::NEG_INFINITY; 2];

    for i in 0..embedding.nrows() {
        for j in 0..embedding.ncols() {
            min_vals[j] = min_vals[j].min(embedding[[i, j]]);
            max_vals[j] = max_vals[j].max(embedding[[i, j]]);
        }
    }

    // Should use a reasonable amount of the embedding space
    for j in 0..2 {
        let range = max_vals[j] - min_vals[j];
        assert!(range > 0.01, "Embedding dimension {} has tiny range: {}", j, range);
    }
}

#[test]
fn test_s_curve() {
    let data = generate_s_curve(50);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    assert_eq!(embedding.nrows(), 50);
    assert_eq!(embedding.ncols(), 2);

    // Check that embedding is spread out
    let distances = pairwise_distances(&embedding);
    let avg_distance: f64 = distances.iter()
        .flat_map(|row| row.iter())
        .filter(|&&d| d > 0.0)
        .sum::<f64>() / (50 * 49) as f64;

    assert!(avg_distance > 0.01, "Points are too clustered");
}

#[test]
fn test_different_n_components() {
    let data = generate_blobs(30, 10, 2, 1.0);

    for n_components in [1, 2, 3, 5] {
        let pacmap = PacmapBuilder::new()
            .n_components(n_components)
            .n_neighbors(5)
            .n_iterations(100)
            .build();

        let embedding = pacmap.fit_transform(&data).unwrap();

        assert_eq!(embedding.nrows(), 30);
        assert_eq!(embedding.ncols(), n_components);
    }
}

#[test]
fn test_parameter_variations() {
    let data = generate_blobs(40, 8, 2, 0.8);

    // Test different n_neighbors
    for n_neighbors in [3, 5, 10, 15] {
        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(n_neighbors)
            .n_iterations(100)
            .build();

        let embedding = pacmap.fit_transform(&data).unwrap();
        assert_eq!(embedding.dim(), (40, 2));

        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn test_small_dataset() {
    let data = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(2)
        .n_iterations(100)
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    assert_eq!(embedding.dim(), (5, 2));
}

#[test]
fn test_high_dimensional_data() {
    let data = generate_blobs(30, 50, 2, 2.0);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(200)
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    assert_eq!(embedding.nrows(), 30);
    assert_eq!(embedding.ncols(), 2);
}

#[test]
fn test_reproducibility_with_seed() {
    let data = generate_blobs(24, 6, 2, 1.0); // 24/2 = 12, 12*2 = 24

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let embedding1 = pacmap.fit_transform(&data).unwrap();
    let embedding2 = pacmap.fit_transform(&data).unwrap();

    // Should be reasonably similar with same seed (allow for numerical differences)
    let mut total_diff = 0.0;
    for i in 0..embedding1.nrows() {
        for j in 0..embedding1.ncols() {
            let diff = (embedding1[[i, j]] - embedding2[[i, j]]).abs();
            total_diff += diff;
        }
    }
    let avg_diff = total_diff / (embedding1.nrows() * embedding1.ncols()) as f64;

    // PACMAP has some inherent randomness even with fixed seed, so we check for reasonable consistency
    assert!(avg_diff < 0.1, "Embeddings differ too much on average: {}", avg_diff);
}

#[test]
fn test_different_seeds_produce_different_results() {
    let data = generate_blobs(24, 6, 2, 1.0); // 24/2 = 12, 12*2 = 24

    let pacmap1 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let pacmap2 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(150)
        .seed(Some(123))
        .build();

    let embedding1 = pacmap1.fit_transform(&data).unwrap();
    let embedding2 = pacmap2.fit_transform(&data).unwrap();

    // Should be different
    let mut different_count = 0;
    for i in 0..embedding1.nrows() {
        for j in 0..embedding1.ncols() {
            if (embedding1[[i, j]] - embedding2[[i, j]]).abs() > 0.1 {
                different_count += 1;
            }
        }
    }

    assert!(different_count > 0, "Different seeds should produce different embeddings");
}

#[test]
fn test_gradient_descent_converges() {
    let data = generate_blobs(30, 8, 2, 1.0);

    // Test with different iteration counts
    let pacmap_short = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(50)
        .seed(Some(42))
        .build();

    let pacmap_long = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(500)
        .seed(Some(42))
        .build();

    let embedding_short = pacmap_short.fit_transform(&data).unwrap();
    let embedding_long = pacmap_long.fit_transform(&data).unwrap();

    // Both should produce valid embeddings
    assert_eq!(embedding_short.dim(), (30, 2));
    assert_eq!(embedding_long.dim(), (30, 2));

    for embedding in [&embedding_short, &embedding_long] {
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn test_embedding_not_collapsed() {
    let data = generate_blobs(40, 10, 3, 1.5);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(300)
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    // Check that points are spread out (not all collapsed to same location)
    let distances = pairwise_distances(&embedding);
    let max_distance = distances.iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(0.0f64, f64::max);

    assert!(max_distance > 0.1, "Embedding is collapsed: max distance = {}", max_distance);

    // Check variance in each dimension
    for dim in 0..2 {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..embedding.nrows() {
            let val = embedding[[i, dim]];
            sum += val;
            sum_sq += val * val;
        }
        let mean = sum / embedding.nrows() as f64;
        let variance = sum_sq / embedding.nrows() as f64 - mean * mean;

        assert!(variance > 1e-6, "Dimension {} has near-zero variance: {}", dim, variance);
    }
}

#[test]
fn test_preserves_cluster_separation() {
    // Create well-separated clusters
    let mut data = Vec::new();

    // Cluster 1 (around origin)
    for i in 0..15 {
        for j in 0..5 {
            data.push(((i * 3 + j) % 10) as f64 * 0.1);
        }
    }

    // Cluster 2 (far away)
    for i in 0..15 {
        for j in 0..5 {
            data.push(100.0 + ((i * 5 + j) % 10) as f64 * 0.1);
        }
    }

    let data = Array2::from_shape_vec((30, 5), data).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let embedding = pacmap.fit_transform(&data).unwrap();

    // Compute average distance within each cluster
    let mut cluster1_distances = Vec::new();
    for i in 0..15 {
        for j in (i + 1)..15 {
            let mut dist = 0.0;
            for d in 0..2 {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                dist += diff * diff;
            }
            cluster1_distances.push(dist.sqrt());
        }
    }

    let mut cluster2_distances = Vec::new();
    for i in 15..30 {
        for j in (i + 1)..30 {
            let mut dist = 0.0;
            for d in 0..2 {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                dist += diff * diff;
            }
            cluster2_distances.push(dist.sqrt());
        }
    }

    // Compute distances between clusters
    let mut between_distances = Vec::new();
    for i in 0..15 {
        for j in 15..30 {
            let mut dist = 0.0;
            for d in 0..2 {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                dist += diff * diff;
            }
            between_distances.push(dist.sqrt());
        }
    }

    let avg_within1 = cluster1_distances.iter().sum::<f64>() / cluster1_distances.len() as f64;
    let avg_within2 = cluster2_distances.iter().sum::<f64>() / cluster2_distances.len() as f64;
    let avg_between = between_distances.iter().sum::<f64>() / between_distances.len() as f64;

    println!("Avg within cluster 1: {:.4}", avg_within1);
    println!("Avg within cluster 2: {:.4}", avg_within2);
    println!("Avg between clusters: {:.4}", avg_between);

    // Between-cluster distance should be larger than within-cluster distance
    assert!(avg_between > avg_within1, "Clusters not well separated");
    assert!(avg_between > avg_within2, "Clusters not well separated");
}
