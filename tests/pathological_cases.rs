use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use scanner_embeddings::pacmap::PacmapBuilder;

/// Test HDBSCAN with all points at the same location (degenerate case)
#[test]
fn test_hdbscan_all_same_location() {
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![5.0; 20], // All points at (5.0, 5.0)
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 10);

    // With all identical points, HDBSCAN behavior is undefined since distance is always 0
    // It may form one cluster, mark all as noise, or have degenerate behavior
    // The important thing is that it doesn't crash and returns valid labels
    for &label in &labels {
        assert!(label >= -1, "All labels should be >= -1");
    }

    // Document that all points are assigned some label (not necessarily same cluster)
    let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
    println!("All identical points produced {} unique labels", unique_labels.len());
}

/// Test PACMAP with all points at the same location
#[test]
fn test_pacmap_all_same_location() {
    let data = Array2::from_shape_vec(
        (20, 3),
        vec![5.0; 60],
    )
    .unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(100)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // All values should be finite even with degenerate input
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with grid layout (regular spacing)
#[test]
fn test_hdbscan_perfect_grid() {
    let mut data_vec = Vec::new();

    // Create a 5x5 grid
    for i in 0..5 {
        for j in 0..5 {
            data_vec.push(i as f64);
            data_vec.push(j as f64);
        }
    }

    let data = Array2::from_shape_vec((25, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 25);

    // Grid should form clusters or be classified as noise uniformly
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with perfect grid
#[test]
fn test_pacmap_perfect_grid() {
    let mut data_vec = Vec::new();

    // Create a 6x6 grid in 3D
    for i in 0..6 {
        for j in 0..6 {
            data_vec.push(i as f64);
            data_vec.push(j as f64);
            data_vec.push((i + j) as f64);
        }
    }

    let data = Array2::from_shape_vec((36, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 36);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with linearly dependent features
#[test]
fn test_hdbscan_linearly_dependent_features() {
    let mut data_vec = Vec::new();

    for i in 0..20 {
        let x = i as f64;
        data_vec.push(x);
        data_vec.push(x * 2.0); // y = 2x
        data_vec.push(x * 3.0); // z = 3x
        data_vec.push(x * 4.0); // w = 4x
    }

    let data = Array2::from_shape_vec((20, 4), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 20);

    // Should handle linearly dependent features
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with linearly dependent features
#[test]
fn test_pacmap_linearly_dependent_features() {
    let mut data_vec = Vec::new();

    for i in 0..30 {
        let x = i as f64;
        data_vec.push(x);
        data_vec.push(x * 2.0);
        data_vec.push(x * 3.0);
        data_vec.push(x * 4.0);
        data_vec.push(x * 5.0);
    }

    let data = Array2::from_shape_vec((30, 5), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 30);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with ring/circle topology
#[test]
fn test_hdbscan_ring_topology() {
    let mut data_vec = Vec::new();
    let n_points = 30;

    // Create a circle
    for i in 0..n_points {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
        data_vec.push(angle.cos() * 5.0);
        data_vec.push(angle.sin() * 5.0);
    }

    let data = Array2::from_shape_vec((n_points, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), n_points);

    // Ring should form one cluster
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with ring topology
#[test]
fn test_pacmap_ring_topology() {
    let mut data_vec = Vec::new();
    let n_points = 40;

    // Create a circle in 3D
    for i in 0..n_points {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
        data_vec.push(angle.cos() * 5.0);
        data_vec.push(angle.sin() * 5.0);
        data_vec.push(0.0);
    }

    let data = Array2::from_shape_vec((n_points, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), n_points);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with spiral topology
#[test]
fn test_hdbscan_spiral_topology() {
    let mut data_vec = Vec::new();
    let n_points = 50;

    // Create a spiral
    for i in 0..n_points {
        let t = i as f64 * 0.2;
        let r = t;
        data_vec.push(r * t.cos());
        data_vec.push(r * t.sin());
    }

    let data = Array2::from_shape_vec((n_points, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), n_points);

    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with spiral topology
#[test]
fn test_pacmap_spiral_topology() {
    let mut data_vec = Vec::new();
    let n_points = 50;

    // Create a 3D spiral
    for i in 0..n_points {
        let t = i as f64 * 0.3;
        let r = t;
        data_vec.push(r * t.cos());
        data_vec.push(r * t.sin());
        data_vec.push(t);
    }

    let data = Array2::from_shape_vec((n_points, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(200)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), n_points);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with moon shapes (two interleaving crescents)
#[test]
fn test_hdbscan_moons() {
    let mut data_vec = Vec::new();
    let n_points_per_moon = 25;

    // First moon
    for i in 0..n_points_per_moon {
        let angle = std::f64::consts::PI * i as f64 / n_points_per_moon as f64;
        data_vec.push(angle.cos());
        data_vec.push(angle.sin());
    }

    // Second moon (shifted and flipped)
    for i in 0..n_points_per_moon {
        let angle = std::f64::consts::PI * i as f64 / n_points_per_moon as f64;
        data_vec.push(1.0 - angle.cos());
        data_vec.push(0.5 - angle.sin());
    }

    let data = Array2::from_shape_vec((n_points_per_moon * 2, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), n_points_per_moon * 2);

    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test HDBSCAN with one cluster inside another (concentric circles)
#[test]
fn test_hdbscan_concentric_circles() {
    let mut data_vec = Vec::new();

    // Inner circle
    for i in 0..20 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
        data_vec.push(angle.cos() * 2.0);
        data_vec.push(angle.sin() * 2.0);
    }

    // Outer circle
    for i in 0..30 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 30.0;
        data_vec.push(angle.cos() * 5.0);
        data_vec.push(angle.sin() * 5.0);
    }

    let data = Array2::from_shape_vec((50, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 50);

    // Should handle nested structure
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with concentric circles
#[test]
fn test_pacmap_concentric_circles() {
    let mut data_vec = Vec::new();

    // Inner circle in 3D
    for i in 0..25 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 25.0;
        data_vec.push(angle.cos() * 2.0);
        data_vec.push(angle.sin() * 2.0);
        data_vec.push(0.0);
    }

    // Outer circle
    for i in 0..35 {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / 35.0;
        data_vec.push(angle.cos() * 5.0);
        data_vec.push(angle.sin() * 5.0);
        data_vec.push(1.0);
    }

    let data = Array2::from_shape_vec((60, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(10)
        .n_iterations(250)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 60);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with alternating dense and sparse regions
#[test]
fn test_hdbscan_dense_sparse_alternating() {
    let mut data_vec = Vec::new();

    // Dense region 1
    for i in 0..20 {
        data_vec.push((i as f64 % 5.0) * 0.1);
        data_vec.push((i as f64 / 5.0) * 0.1);
    }

    // Sparse region (individual points far apart)
    data_vec.push(10.0);
    data_vec.push(10.0);
    data_vec.push(15.0);
    data_vec.push(15.0);
    data_vec.push(20.0);
    data_vec.push(20.0);

    // Dense region 2
    for i in 0..20 {
        data_vec.push(30.0 + (i as f64 % 5.0) * 0.1);
        data_vec.push(30.0 + (i as f64 / 5.0) * 0.1);
    }

    let data = Array2::from_shape_vec((43, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(5)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 43);

    // Sparse points should be noise
    let noise_count = labels.iter().filter(|&&x| x == -1).count();
    assert!(noise_count > 0, "Should identify sparse region points as noise");

    // Should find dense regions as clusters
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&x| x != -1).collect();
    assert!(!unique_labels.is_empty(), "Should find dense region clusters");
}

/// Test HDBSCAN with exponentially increasing distances
#[test]
fn test_hdbscan_exponential_distances() {
    let mut data_vec = Vec::new();

    // Points with exponentially increasing distances
    for i in 0..15 {
        let scale = 2.0_f64.powi(i);
        data_vec.push(scale);
        data_vec.push(0.0);
    }

    let data = Array2::from_shape_vec((15, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 15);

    // Should handle exponential scales
    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with exponentially increasing distances
#[test]
fn test_pacmap_exponential_distances() {
    let mut data_vec = Vec::new();

    for i in 0..20 {
        let scale = 1.5_f64.powi(i);
        data_vec.push(scale);
        data_vec.push(scale * 0.5);
        data_vec.push(scale * 0.25);
    }

    let data = Array2::from_shape_vec((20, 3), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 20);
    assert_eq!(result.ncols(), 2);

    // Should handle exponential scales
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}

/// Test HDBSCAN with binary/boolean-like features
#[test]
fn test_hdbscan_binary_features() {
    let mut data_vec = Vec::new();

    // Points with only 0 or 1 values
    for i in 0..40 {
        data_vec.push(if i % 2 == 0 { 0.0 } else { 1.0 });
        data_vec.push(if i % 3 == 0 { 0.0 } else { 1.0 });
        data_vec.push(if i % 5 == 0 { 0.0 } else { 1.0 });
    }

    let data = Array2::from_shape_vec((40, 3), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 40);

    for &label in &labels {
        assert!(label >= -1);
    }
}

/// Test PACMAP with binary features
#[test]
fn test_pacmap_binary_features() {
    let mut data_vec = Vec::new();

    for i in 0..30 {
        data_vec.push(if i % 2 == 0 { 0.0 } else { 1.0 });
        data_vec.push(if i % 3 == 0 { 0.0 } else { 1.0 });
        data_vec.push(if i % 5 == 0 { 0.0 } else { 1.0 });
        data_vec.push(if i % 7 == 0 { 0.0 } else { 1.0 });
    }

    let data = Array2::from_shape_vec((30, 4), data_vec).unwrap();

    let pacmap = PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(8)
        .n_iterations(150)
        .seed(Some(42))
        .build();

    let result = pacmap.fit_transform(&data).unwrap();

    assert_eq!(result.nrows(), 30);
    assert_eq!(result.ncols(), 2);

    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert!(result[[i, j]].is_finite());
        }
    }
}
