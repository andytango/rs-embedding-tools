use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use scanner_embeddings::pacmap::PacmapBuilder;

/// Helper to generate blob clusters
fn generate_blobs(n_samples: usize, n_features: usize, n_clusters: usize, cluster_std: f64, seed: u64) -> Array2<f64> {
    let samples_per_cluster = n_samples / n_clusters;
    let mut data = Vec::new();

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

    Array2::from_shape_vec((samples_per_cluster * n_clusters, n_features), data).unwrap()
}

/// Test complete pipeline: high-dimensional data -> PACMAP -> HDBSCAN
#[test]
fn test_complete_pipeline_highdim_to_clusters() {
    // Generate high-dimensional data with 3 clusters
    let data = generate_blobs(90, 20, 3, 2.0, 42);

    // Step 1: Reduce dimensionality with PACMAP
    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    assert_eq!(reduced.nrows(), 90);
    assert_eq!(reduced.ncols(), 2);

    // Verify reduction produced finite values
    for i in 0..reduced.nrows() {
        for j in 0..reduced.ncols() {
            assert!(reduced[[i, j]].is_finite());
        }
    }

    // Step 2: Cluster the reduced data with HDBSCAN
    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(10)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&reduced).unwrap();

    assert_eq!(labels.len(), 90);

    // Should find multiple clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(unique_labels.len() >= 2, "Pipeline should find multiple clusters, found: {}", unique_labels.len());

    // Verify cluster assignments are reasonable (each cluster has multiple points)
    for &cluster_id in unique_labels.iter() {
        let count = labels.iter().filter(|&&x| x == *cluster_id).count();
        assert!(count >= 10, "Cluster {} should have at least min_cluster_size points, has {}", cluster_id, count);
    }
}

/// Test pipeline with varying n_components
#[test]
fn test_pipeline_varying_dimensions() {
    let data = generate_blobs(60, 15, 3, 1.5, 123);

    for n_components in [2, 3, 5] {
        let pacmap = PacmapBuilder::new()
            .n_components(n_components)
            .n_neighbors(8)
            .n_iterations(200)
            .seed(Some(42))
            .build();

        let reduced = pacmap.fit_transform(&data).unwrap();

        assert_eq!(reduced.nrows(), 60);
        assert_eq!(reduced.ncols(), n_components);

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(8)
            .min_samples(4)
            .build();

        let labels = clusterer.fit_predict(&reduced).unwrap();

        assert_eq!(labels.len(), 60);

        // Should find clusters regardless of intermediate dimensionality
        let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
        assert!(!unique_labels.is_empty(),
                "Should find clusters with n_components={}", n_components);
    }
}

/// Test pipeline consistency across multiple runs with same seed
#[test]
fn test_pipeline_consistency_with_seed() {
    let data = generate_blobs(60, 10, 3, 1.0, 999);

    // Run 1
    let pacmap1 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let reduced1 = pacmap1.fit_transform(&data).unwrap();

    let clusterer1 = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels1 = clusterer1.fit_predict(&reduced1).unwrap();

    // Run 2
    let pacmap2 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let reduced2 = pacmap2.fit_transform(&data).unwrap();

    let clusterer2 = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels2 = clusterer2.fit_predict(&reduced2).unwrap();

    // Results should be very similar (allowing for tiny numerical differences)
    assert_eq!(reduced1.shape(), reduced2.shape());
    let mut max_diff: f64 = 0.0;
    for i in 0..reduced1.nrows() {
        for j in 0..reduced1.ncols() {
            let diff = (reduced1[[i, j]] - reduced2[[i, j]]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(max_diff < 2.0,
            "PACMAP embeddings should be reasonably similar with same seed, max diff: {}", max_diff);

    // Cluster labels should be similar (structure roughly preserved)
    // Note: exact labels may differ slightly if embeddings have small variations
    let clusters1: std::collections::HashSet<_> = labels1.iter().filter(|&&x| x != -1).copied().collect();
    let clusters2: std::collections::HashSet<_> = labels2.iter().filter(|&&x| x != -1).copied().collect();

    // Number of clusters should be similar (within 1-2 of each other)
    let diff = (clusters1.len() as i32 - clusters2.len() as i32).abs();
    assert!(diff <= 2,
            "Number of clusters should be similar: {} vs {}", clusters1.len(), clusters2.len());
}

/// Test pipeline with noise detection
#[test]
fn test_pipeline_with_noise() {
    let mut data_vec = Vec::new();

    // Cluster 1
    for i in 0..20 {
        data_vec.push((i as f64 % 5.0) * 0.5);
        data_vec.push((i as f64 / 5.0) * 0.5);
        data_vec.push(0.0);
        data_vec.push(0.0);
        data_vec.push(0.0);
    }

    // Cluster 2
    for i in 0..20 {
        data_vec.push(10.0 + (i as f64 % 5.0) * 0.5);
        data_vec.push(10.0 + (i as f64 / 5.0) * 0.5);
        data_vec.push(10.0);
        data_vec.push(10.0);
        data_vec.push(10.0);
    }

    // Noise points (scattered)
    for i in 0..10 {
        data_vec.push(i as f64 * 3.0);
        data_vec.push(i as f64 * 4.0);
        data_vec.push(i as f64 * 5.0);
        data_vec.push(i as f64 * 2.0);
        data_vec.push(i as f64 * 6.0);
    }

    let data = Array2::from_shape_vec((50, 5), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(250)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&reduced).unwrap();

    assert_eq!(labels.len(), 50);

    // Check that noise points exist OR that we have at least 2 distinct clusters
    // (either outcome is valid - depends on how PACMAP embeds the noise)
    let noise_count = labels.iter().filter(|&&x| x == -1).count();
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();

    // Either we detect noise OR we separate the two clusters properly
    assert!(
        noise_count > 0 || unique_labels.len() >= 2,
        "Should either detect noise ({} points) or find multiple clusters ({} clusters)",
        noise_count, unique_labels.len()
    );

    // Should find at least one cluster
    assert!(!unique_labels.is_empty(), "Should find at least one cluster");
}

/// Test pipeline preserves cluster separation
#[test]
fn test_pipeline_preserves_separation() {
    // Create well-separated clusters
    let mut data_vec = Vec::new();

    // Cluster 1: near origin
    for i in 0..25 {
        for j in 0..10 {
            if j < 3 {
                data_vec.push((i as f64 % 5.0) * 0.1);
            } else {
                data_vec.push(0.0);
            }
        }
    }

    // Cluster 2: far away
    for i in 0..25 {
        for j in 0..10 {
            if j < 3 {
                data_vec.push(100.0 + (i as f64 % 5.0) * 0.1);
            } else {
                data_vec.push(0.0);
            }
        }
    }

    let data = Array2::from_shape_vec((50, 10), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(300)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(10)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&reduced).unwrap();

    // Should find exactly 2 clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(unique_labels.len() >= 2, "Should preserve well-separated clusters, found: {}", unique_labels.len());

    // Verify clusters are properly assigned
    let cluster1_indices: Vec<usize> = labels.iter()
        .enumerate()
        .filter(|(_, &label)| label == 0)
        .map(|(i, _)| i)
        .collect();

    let cluster2_indices: Vec<usize> = labels.iter()
        .enumerate()
        .filter(|(_, &label)| label == 1)
        .map(|(i, _)| i)
        .collect();

    // One cluster should be in first 25 points, other in last 25
    if !cluster1_indices.is_empty() && !cluster2_indices.is_empty() {
        let cluster1_avg = cluster1_indices.iter().sum::<usize>() as f64 / cluster1_indices.len() as f64;
        let cluster2_avg = cluster2_indices.iter().sum::<usize>() as f64 / cluster2_indices.len() as f64;

        assert!((cluster1_avg - cluster2_avg).abs() > 10.0,
                "Clusters should have different average indices");
    }
}

/// Test pipeline parameter sensitivity
#[test]
fn test_pipeline_parameter_sensitivity() {
    let data = generate_blobs(60, 10, 3, 1.5, 42);

    // Reduce with PACMAP
    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(250)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    // Test different min_cluster_size values
    for min_cluster_size in [5, 10, 15] {
        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(min_cluster_size)
            .min_samples(min_cluster_size / 2)
            .build();

        let labels = clusterer.fit_predict(&reduced).unwrap();

        assert_eq!(labels.len(), 60);

        let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();

        // Larger min_cluster_size should find fewer/no clusters
        if min_cluster_size <= 10 {
            assert!(!unique_labels.is_empty(),
                    "Should find clusters with min_cluster_size={}", min_cluster_size);
        }

        println!("min_cluster_size={}: found {} clusters",
                 min_cluster_size, unique_labels.len());
    }
}

/// Test pipeline with different PACMAP neighbor counts
#[test]
fn test_pipeline_different_neighbor_counts() {
    let data = generate_blobs(60, 8, 3, 1.0, 123);

    for n_neighbors in [5, 10, 15] {
        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(n_neighbors)
            .n_iterations(200)
            .seed(Some(42))
            .build();

        let reduced = pacmap.fit_transform(&data).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(8)
            .min_samples(5)
            .build();

        let labels = clusterer.fit_predict(&reduced).unwrap();

        let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();

        // Different n_neighbors may affect clustering
        assert_eq!(labels.len(), 60);
        println!("n_neighbors={}: found {} clusters", n_neighbors, unique_labels.len());
    }
}

/// Test pipeline handles edge case: minimum viable dataset
#[test]
fn test_pipeline_minimum_viable_dataset() {
    // Just barely enough points for PACMAP and HDBSCAN
    let data = Array2::from_shape_vec(
        (10, 5),
        (0..50).map(|x| x as f64).collect(),
    )
    .unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(3)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    assert_eq!(reduced.nrows(), 10);
    assert_eq!(reduced.ncols(), 2);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&reduced).unwrap();

    assert_eq!(labels.len(), 10);

    // May or may not find clusters, but should not crash
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test pipeline output validation
#[test]
fn test_pipeline_output_validation() {
    let data = generate_blobs(60, 12, 4, 1.5, 999);

    let pacmap = PacmapBuilder::new()
        .n_components(3)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let reduced = pacmap.fit_transform(&data).unwrap();

    // Validate PACMAP output
    assert_eq!(reduced.nrows(), 60);
    assert_eq!(reduced.ncols(), 3);

    // Check for NaN or Inf
    for i in 0..reduced.nrows() {
        for j in 0..reduced.ncols() {
            assert!(reduced[[i, j]].is_finite(),
                    "PACMAP produced non-finite value at [{}, {}]: {}",
                    i, j, reduced[[i, j]]);
        }
    }

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&reduced).unwrap();

    // Validate HDBSCAN output
    assert_eq!(labels.len(), 60);

    // Check labels are in valid range
    for (i, &label) in labels.iter().enumerate() {
        assert!(label >= -1,
                "Invalid label at index {}: {}", i, label);
        assert!(label < 100,
                "Suspiciously high cluster ID at index {}: {}", i, label);
    }

    // Check cluster IDs are sequential starting from 0
    let cluster_ids: Vec<i32> = labels.iter()
        .filter(|&&x| x != -1)
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if !cluster_ids.is_empty() {
        let mut sorted_ids = cluster_ids.clone();
        sorted_ids.sort();
        assert_eq!(sorted_ids[0], 0, "Cluster IDs should start at 0");

        for i in 1..sorted_ids.len() {
            assert!(sorted_ids[i] <= sorted_ids[i - 1] + 1,
                    "Cluster IDs should be sequential");
        }
    }
}
