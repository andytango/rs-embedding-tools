use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use scanner_embeddings::pacmap::PacmapBuilder;

/// Helper function to generate blob clusters
fn generate_blobs(n_samples: usize, n_features: usize, n_clusters: usize, cluster_std: f64, seed: u64) -> Array2<f64> {
    let samples_per_cluster = n_samples / n_clusters;
    let mut data = Vec::new();

    // Simple LCG for reproducibility
    let mut rng = seed;
    let next = |r: &mut u64| {
        *r = r.wrapping_mul(1103515245).wrapping_add(12345);
        (*r / 65536) % 32768
    };

    for cluster in 0..n_clusters {
        let center_offset = cluster as f64 * 10.0;
        for _ in 0..samples_per_cluster {
            for feature in 0..n_features {
                let noise = ((next(&mut rng) as f64 / 32768.0) - 0.5) * 2.0 * cluster_std;
                data.push(center_offset + (feature as f64) + noise);
            }
        }
    }

    Array2::from_shape_vec((n_samples / n_clusters * n_clusters, n_features), data).unwrap()
}

/// Test HDBSCAN with 100 points in 10 dimensions
#[test]
fn test_hdbscan_100_points_10d() {
    let data = generate_blobs(100, 10, 3, 1.0, 42);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(10)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 99); // 100 points / 3 clusters * 3 = 99

    // Should find some clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(!unique_labels.is_empty(), "Should find at least 1 cluster with 100 points");
}

/// Test HDBSCAN with 200 points in 5 dimensions
#[test]
fn test_hdbscan_200_points_5d() {
    let data = generate_blobs(200, 5, 4, 1.0, 123);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(15)
        .min_samples(10)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 200);

    // Should find multiple clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(unique_labels.len() >= 2, "Should find at least 2 clusters with 200 points");
}

/// Test PACMAP with 100 points reducing from 20D to 2D
#[test]
fn test_pacmap_100_points_20d_to_2d() {
    let data = generate_blobs(100, 20, 3, 2.0, 42);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 99);
    assert_eq!(result.ncols(), 2);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite(), "Non-finite value at [{}, {}]", i, j);
        }
    }
}

/// Test PACMAP with 150 points reducing from 30D to 3D
#[test]
fn test_pacmap_150_points_30d_to_3d() {
    let data = generate_blobs(150, 30, 5, 2.0, 999);

    let pacmap = PacmapBuilder::new()
        .n_components(3)
        .n_neighbors(15)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 150);
    assert_eq!(result.ncols(), 3);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with varying cluster sizes
#[test]
fn test_hdbscan_varying_cluster_sizes() {
    let mut data_vec = Vec::new();

    // Large cluster (50 points)
    for i in 0..50 {
        data_vec.push((i as f64 % 10.0) * 0.1);
        data_vec.push((i as f64 / 10.0) * 0.1);
    }

    // Medium cluster (20 points)
    for i in 0..20 {
        data_vec.push(10.0 + (i as f64 % 5.0) * 0.1);
        data_vec.push(10.0 + (i as f64 / 5.0) * 0.1);
    }

    // Small cluster (10 points)
    for i in 0..10 {
        data_vec.push(20.0 + (i as f64 % 3.0) * 0.1);
        data_vec.push(20.0 + (i as f64 / 3.0) * 0.1);
    }

    let data = Array2::from_shape_vec((80, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 80);

    // Should find multiple clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(unique_labels.len() >= 2, "Should find clusters of varying sizes");
}

/// Test PACMAP maintains relative distances
#[test]
fn test_pacmap_relative_distances() {
    // Create three well-separated clusters
    let mut data_vec = Vec::new();

    // Cluster 1: near origin
    for i in 0..20 {
        data_vec.push((i as f64 % 5.0) * 0.1);
        data_vec.push((i as f64 / 5.0) * 0.1);
        data_vec.push(0.0);
    }

    // Cluster 2: far away
    for i in 0..20 {
        data_vec.push(100.0 + (i as f64 % 5.0) * 0.1);
        data_vec.push(100.0 + (i as f64 / 5.0) * 0.1);
        data_vec.push(100.0);
    }

    // Cluster 3: very far away
    for i in 0..20 {
        data_vec.push(1000.0 + (i as f64 % 5.0) * 0.1);
        data_vec.push(1000.0 + (i as f64 / 5.0) * 0.1);
        data_vec.push(1000.0);
    }

    let data = Array2::from_shape_vec((60, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 60);
    assert_eq!(result.ncols(), 2);

    // Calculate average position of each cluster in embedding
    let mut cluster1_center = [0.0, 0.0];
    let mut cluster2_center = [0.0, 0.0];
    let mut cluster3_center = [0.0, 0.0];

    for i in 0..20 {
        cluster1_center[0] += result[[i, 0]];
        cluster1_center[1] += result[[i, 1]];
        cluster2_center[0] += result[[i + 20, 0]];
        cluster2_center[1] += result[[i + 20, 1]];
        cluster3_center[0] += result[[i + 40, 0]];
        cluster3_center[1] += result[[i + 40, 1]];
    }

    cluster1_center[0] /= 20.0;
    cluster1_center[1] /= 20.0;
    cluster2_center[0] /= 20.0;
    cluster2_center[1] /= 20.0;
    cluster3_center[0] /= 20.0;
    cluster3_center[1] /= 20.0;

    // Distance between cluster centers
    let dist_1_2 = ((cluster1_center[0] - cluster2_center[0]).powi(2)
                  + (cluster1_center[1] - cluster2_center[1]).powi(2)).sqrt();
    let dist_1_3 = ((cluster1_center[0] - cluster3_center[0]).powi(2)
                  + (cluster1_center[1] - cluster3_center[1]).powi(2)).sqrt();

    // Cluster 3 should be farther from cluster 1 than cluster 2 is
    // (preserving global structure)
    assert!(dist_1_3 > dist_1_2 * 0.8,
            "Global structure should be roughly preserved: dist_1_3={}, dist_1_2={}",
            dist_1_3, dist_1_2);
}

/// Test HDBSCAN with elongated clusters
#[test]
fn test_hdbscan_elongated_clusters() {
    let mut data_vec = Vec::new();

    // Elongated cluster 1 (horizontal)
    for i in 0..30 {
        data_vec.push(i as f64 * 0.5);
        data_vec.push(0.0 + ((i as f64).sin() * 0.2));
    }

    // Elongated cluster 2 (vertical)
    for i in 0..30 {
        data_vec.push(20.0 + ((i as f64).sin() * 0.2));
        data_vec.push(i as f64 * 0.5);
    }

    let data = Array2::from_shape_vec((60, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(10)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 60);

    // Should find at least one cluster
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(!unique_labels.is_empty(), "Should handle elongated clusters");
}

/// Test PACMAP with sparse high-dimensional data
#[test]
fn test_pacmap_sparse_high_dimensional() {
    let mut data_vec = Vec::new();

    // Create sparse data where most features are zero
    for i in 0..50 {
        for j in 0..20 {
            if j < 3 {
                // Only first 3 dimensions have signal
                data_vec.push((i as f64) + (j as f64 * 10.0));
            } else {
                data_vec.push(0.0);
            }
        }
    }

    let data = Array2::from_shape_vec((50, 20), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 50);
    assert_eq!(result.ncols(), 2);

    // Should still produce valid embedding
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN performance degradation with increasing points
#[test]
fn test_hdbscan_scaling() {
    for n_points in [20, 40, 60] {
        let data = generate_blobs(n_points, 3, 2, 1.0, 42);

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(5)
            .min_samples(3)
            .build();

        let start = std::time::Instant::now();
        let labels = clusterer.fit_predict(&data).unwrap();
        let duration = start.elapsed();

        assert_eq!(labels.len(), (n_points / 2) * 2);
        println!("HDBSCAN with {} points took {:?}", n_points, duration);
    }
}

/// Test PACMAP performance degradation with increasing points
#[test]
fn test_pacmap_scaling() {
    for n_points in [20, 40, 60] {
        let data = generate_blobs(n_points, 5, 2, 1.0, 42);

        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5)
            .n_iterations(100)
            .seed(Some(42))
            .build();

        let start = std::time::Instant::now();
        let result = pacmap.fit_transform(&data).unwrap();
        let duration = start.elapsed();

        assert_eq!(result.nrows(), (n_points / 2) * 2);
        println!("PACMAP with {} points took {:?}", n_points, duration);
    }
}

/// Test HDBSCAN with maximum dimension (100D)
#[test]
fn test_hdbscan_max_dimensions() {
    let data = generate_blobs(30, 100, 2, 5.0, 42);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 30);

    // Should still produce valid labels
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with maximum reduction ratio (100D to 2D)
#[test]
fn test_pacmap_max_reduction_ratio() {
    let data = generate_blobs(30, 100, 3, 5.0, 42);

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 30);
    assert_eq!(result.ncols(), 2);

    // Should handle extreme reduction
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}
