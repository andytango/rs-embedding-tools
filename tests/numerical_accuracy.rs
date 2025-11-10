use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use scanner_embeddings::pacmap::PacmapBuilder;

/// Test numerical stability with very small values
#[test]
fn test_hdbscan_very_small_values() {
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            1e-10, 2e-10,
            1.1e-10, 2.1e-10,
            1.2e-10, 2.0e-10,
            1.0e-10, 2.2e-10,
            1.1e-10, 2.0e-10,
            10e-10, 20e-10,
            10.1e-10, 20.1e-10,
            10.2e-10, 20.0e-10,
            10.0e-10, 20.2e-10,
            10.1e-10, 20.0e-10,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should still identify two clusters
    assert_eq!(labels.len(), 10);
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert!(unique_labels.len() >= 2, "Should find at least 2 clusters");

    // All values should be finite
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test numerical stability with very large values
#[test]
fn test_hdbscan_very_large_values() {
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            1e10, 2e10,
            1.1e10, 2.1e10,
            1.2e10, 2.0e10,
            1.0e10, 2.2e10,
            1.1e10, 2.0e10,
            10e10, 20e10,
            10.1e10, 20.1e10,
            10.2e10, 20.0e10,
            10.0e10, 20.2e10,
            10.1e10, 20.0e10,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should still identify two clusters
    assert_eq!(labels.len(), 10);
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert!(unique_labels.len() >= 2, "Should find at least 2 clusters");
}

/// Test PACMAP numerical stability with very small values
#[test]
fn test_pacmap_very_small_values() {
    let mut data_vec = Vec::new();
    for i in 0..20 {
        data_vec.push((i as f64 * 1e-10) + 1e-10);
        data_vec.push((i as f64 * 2e-10) + 1e-10);
        data_vec.push((i as f64 * 1.5e-10) + 1e-10);
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite(), "Non-finite value at [{}, {}]", i, j);
        }
    }
}

/// Test PACMAP numerical stability with very large values
#[test]
fn test_pacmap_very_large_values() {
    let mut data_vec = Vec::new();
    for i in 0..20 {
        data_vec.push((i as f64 * 1e10) + 1e10);
        data_vec.push((i as f64 * 2e10) + 1e10);
        data_vec.push((i as f64 * 1.5e10) + 1e10);
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite(), "Non-finite value at [{}, {}]", i, j);
        }
    }
}

/// Test HDBSCAN with mixed scale data
#[test]
fn test_hdbscan_mixed_scale() {
    // One dimension is much larger than the other
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0,
            0.1, 100.0,
            0.2, 200.0,
            0.0, 50.0,
            0.1, 150.0,
            10.0, 1000.0,
            10.1, 1100.0,
            10.2, 1200.0,
            10.0, 1050.0,
            10.1, 1150.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should handle mixed scales
    assert_eq!(labels.len(), 10);
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with mixed scale data
#[test]
fn test_pacmap_mixed_scale() {
    let mut data_vec = Vec::new();
    for i in 0..20 {
        data_vec.push(i as f64 * 0.01); // Small scale
        data_vec.push(i as f64 * 1000.0); // Large scale
        data_vec.push(i as f64); // Medium scale
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with zero variance in one dimension
#[test]
fn test_hdbscan_zero_variance_dimension() {
    let data = Array2::from_shape_vec(
        (10, 3),
        vec![
            0.0, 5.0, 0.0,
            0.1, 5.0, 0.0,
            0.2, 5.0, 0.0,
            0.0, 5.0, 0.0,
            0.1, 5.0, 0.0,
            10.0, 5.0, 0.0,
            10.1, 5.0, 0.0,
            10.2, 5.0, 0.0,
            10.0, 5.0, 0.0,
            10.1, 5.0, 0.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should still cluster based on other dimensions
    assert_eq!(labels.len(), 10);
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert!(unique_labels.len() >= 2, "Should find clusters despite zero variance dimension");
}

/// Test PACMAP with zero variance in one dimension
#[test]
fn test_pacmap_zero_variance_dimension() {
    let mut data_vec = Vec::new();
    for i in 0..20 {
        data_vec.push(i as f64);
        data_vec.push(5.0); // Constant dimension
        data_vec.push(i as f64 * 2.0);
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Should handle zero variance dimension
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN determinism - same input should produce same output
#[test]
fn test_hdbscan_determinism() {
    let data = Array2::from_shape_vec(
        (20, 2),
        (0..40).map(|x| x as f64).collect(),
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels1 = clusterer.fit_predict(&data).unwrap();
    let labels2 = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels1, labels2, "HDBSCAN should be deterministic");
}

/// Test PACMAP with different seeds produces different results
#[test]
fn test_pacmap_different_seeds() {
    let data = Array2::from_shape_vec(
        (20, 3),
        (0..60).map(|x| x as f64).collect(),
    )
    .unwrap();

    let pacmap1 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let pacmap2 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(123))
        .build();

    let result1 = pacmap1.fit_transform(&data).unwrap();
    let result2 = pacmap2.fit_transform(&data).unwrap();

    // Different seeds should produce different embeddings
    let mut differs = false;
    for i in 0..result1.nrows() {
        for j in 0..result1.ncols() {
            if (result1[[i, j]] - result2[[i, j]]).abs() > 0.01 {
                differs = true;
                break;
            }
        }
        if differs {
            break;
        }
    }

    assert!(differs, "Different seeds should produce different embeddings");
}

/// Test PACMAP with same seed produces same results
#[test]
fn test_pacmap_same_seed_determinism() {
    let data = Array2::from_shape_vec(
        (20, 3),
        (0..60).map(|x| x as f64).collect(),
    )
    .unwrap();

    let pacmap1 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let pacmap2 = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result1 = pacmap1.fit_transform(&data).unwrap();
    let result2 = pacmap2.fit_transform(&data).unwrap();

    // Same seed should produce identical embeddings
    for i in 0..result1.nrows() {
        for j in 0..result1.ncols() {
            assert!(
                (result1[[i, j]] - result2[[i, j]]).abs() < 1e-10,
                "Same seed should produce identical results at [{}, {}]: {} vs {}",
                i, j, result1[[i, j]], result2[[i, j]]
            );
        }
    }
}

/// Test HDBSCAN with negative values
#[test]
fn test_hdbscan_negative_values() {
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            -10.0, -20.0,
            -10.1, -20.1,
            -10.2, -20.0,
            -10.0, -20.2,
            -10.1, -20.0,
            -1.0, -2.0,
            -1.1, -2.1,
            -1.2, -2.0,
            -1.0, -2.2,
            -1.1, -2.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 10);
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert!(unique_labels.len() >= 2, "Should find clusters with negative values");
}

/// Test PACMAP with negative values
#[test]
fn test_pacmap_negative_values() {
    let mut data_vec = Vec::new();
    for i in 0..20 {
        data_vec.push(-(i as f64));
        data_vec.push(-(i as f64 * 2.0));
        data_vec.push(-(i as f64 * 1.5));
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Check all values are finite
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with data centered at origin
#[test]
fn test_hdbscan_centered_at_origin() {
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            -0.1, -0.1,
            -0.05, -0.05,
            0.0, 0.0,
            0.05, 0.05,
            0.1, 0.1,
            -5.0, -5.0,
            -5.1, -5.1,
            -4.9, -4.9,
            -5.05, -5.05,
            -4.95, -4.95,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 10);
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    assert!(unique_labels.len() >= 2, "Should find clusters centered at origin");
}
