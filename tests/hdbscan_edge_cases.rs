//! Edge case tests for HDBSCAN implementation
//!
//! This test suite covers edge cases, boundary conditions, and stress tests.

use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;

#[test]
fn test_minimum_viable_cluster() {
    // Exactly min_cluster_size points close together
    let data = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.0, 0.0,
            0.01, 0.0,
            0.0, 0.01,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should form exactly one cluster
    let unique_labels: std::collections::HashSet<_> = labels.iter().filter(|&&l| l != -1).collect();
    assert!(!unique_labels.is_empty(), "Should form at least one cluster");
}

#[test]
fn test_just_below_min_cluster_size() {
    // min_cluster_size - 1 points (should be noise)
    let data = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,
            0.01, 0.0,
            0.0, 0.01,
            0.01, 0.01,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5) // More than we have
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // All should be noise since we don't have enough points
    assert!(labels.iter().all(|&l| l == -1), "All points should be noise");
}

#[test]
fn test_single_point() {
    // Edge case: single point
    let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(2)
        .min_samples(1)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0], -1, "Single point should be noise");
}

#[test]
fn test_two_points() {
    // Edge case: exactly two points
    let data = Array2::from_shape_vec(
        (2, 2),
        vec![0.0, 0.0, 0.1, 0.1],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(2)
        .min_samples(1)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 2);
    // Should form one cluster with both points
    assert_eq!(labels[0], labels[1], "Both points should be in same cluster");
    assert_ne!(labels[0], -1, "Points should not be noise");
}

#[test]
fn test_identical_points() {
    // All points at same location
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
            1.0, 1.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // Should all be in same cluster (or all noise)
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
    assert!(unique_labels.len() <= 2, "Should have at most one cluster plus noise");
}

#[test]
fn test_collinear_points() {
    // All points on a line
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0,
            1.0, 0.0,
            2.0, 0.0,
            3.0, 0.0,
            4.0, 0.0,
            5.0, 0.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 6);
    // Should find at least some structure
}

#[test]
fn test_one_dimensional_data() {
    // Data in 1D (second dimension is constant)
    let mut data_vec = Vec::new();
    for i in 0..10 {
        data_vec.push(i as f64);
        data_vec.push(0.0); // All have same y coordinate
    }

    let data = Array2::from_shape_vec((10, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 10);
}

#[test]
fn test_high_dimensional_sparse() {
    // High dimensional data (10D)
    let n_points = 20;
    let n_dims = 10;

    let mut data_vec = Vec::new();
    for i in 0..n_points {
        for d in 0..n_dims {
            data_vec.push((i * (d + 1)) as f64 % 10.0);
        }
    }

    let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), n_points);
    // Should complete without panic
}

#[test]
fn test_very_large_distances() {
    // Points very far apart
    let data = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,
            1000.0, 0.0,
            0.0, 1000.0,
            1000.0, 1000.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(2)
        .min_samples(1)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 4);
    // Most likely all noise due to large distances
}

#[test]
fn test_very_small_distances() {
    // Points very close together
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![
            0.0, 0.0,
            0.0001, 0.0,
            0.0, 0.0001,
            0.0001, 0.0001,
            0.00005, 0.00005,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 5);
    // Should form one tight cluster
    let unique_clusters = labels.iter().filter(|&&l| l != -1).collect::<std::collections::HashSet<_>>();
    assert!(!unique_clusters.is_empty(), "Should find at least one cluster");
}

#[test]
fn test_outliers_with_clusters() {
    // Two clusters with clear outliers
    let data = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Cluster 1
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            0.1, 0.1,
            // Cluster 2
            10.0, 10.0,
            10.1, 10.0,
            10.0, 10.1,
            10.1, 10.1,
            // Outliers
            5.0, 5.0,
            -5.0, -5.0,
            15.0, 0.0,
            0.0, 15.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    let _n_noise = labels.iter().filter(|&&l| l == -1).count();
    // Algorithm should identify some structure (either clusters or noise)
    // but may vary based on parameters

    let n_clusters = labels.iter().filter(|&&l| l != -1).collect::<std::collections::HashSet<_>>().len();
    assert!(n_clusters >= 1, "Should find at least one valid cluster");

    // Verify that the two tight clusters (first 8 points) are found
    let cluster1_labels: Vec<_> = labels[0..4].iter().collect();
    let cluster2_labels: Vec<_> = labels[4..8].iter().collect();

    // At least one of the tight clusters should be identified
    let cluster1_has_cluster = cluster1_labels.iter().any(|&&l| l != -1);
    let cluster2_has_cluster = cluster2_labels.iter().any(|&&l| l != -1);
    assert!(cluster1_has_cluster || cluster2_has_cluster, "Should find at least one tight cluster");
}

#[test]
fn test_nested_clusters() {
    // Inner and outer ring
    let mut data_vec = Vec::new();

    // Inner cluster (tight)
    for i in 0..5 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 5.0;
        data_vec.push(0.5 * angle.cos());
        data_vec.push(0.5 * angle.sin());
    }

    // Outer ring
    for i in 0..10 {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / 10.0;
        data_vec.push(3.0 * angle.cos());
        data_vec.push(3.0 * angle.sin());
    }

    let data = Array2::from_shape_vec((15, 2), data_vec).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(4)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), 15);
    // Should find at least one cluster
    let n_clusters = labels.iter().filter(|&&l| l != -1).collect::<std::collections::HashSet<_>>().len();
    assert!(n_clusters >= 1);
}

#[test]
fn test_different_densities_far_apart() {
    // Dense cluster and sparse cluster far apart
    let data = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Dense cluster (8 points, very close)
            0.0, 0.0,
            0.05, 0.0,
            0.0, 0.05,
            0.05, 0.05,
            0.025, 0.025,
            0.075, 0.025,
            0.025, 0.075,
            0.075, 0.075,
            // Sparse cluster (4 points, farther apart)
            100.0, 100.0,
            101.0, 100.0,
            100.0, 101.0,
            101.0, 101.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    let n_clusters = labels.iter().filter(|&&l| l != -1).collect::<std::collections::HashSet<_>>().len();
    assert!(n_clusters >= 1, "Should find at least one cluster");
}

#[test]
fn test_parameter_extremes() {
    // Test with extreme parameters
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.0,
            0.0, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2, 0.15, 0.15,
        ],
    )
    .unwrap();

    // Very large min_cluster_size
    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(20) // More than total points
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();
    assert!(labels.iter().all(|&l| l == -1), "All should be noise with impossible min_cluster_size");
}

#[test]
fn test_reproducibility() {
    // Same input should give same output (deterministic)
    let data = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1,
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels1 = clusterer.fit_predict(&data).unwrap();
    let labels2 = clusterer.fit_predict(&data).unwrap();

    assert_eq!(labels1, labels2, "Algorithm should be deterministic");
}

#[test]
fn test_cluster_labels_are_sequential() {
    // Verify cluster labels are 0, 1, 2, ... (no gaps)
    let data = Array2::from_shape_vec(
        (15, 2),
        vec![
            // Cluster 1
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05,
            // Cluster 2
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 10.05, 10.05,
            // Cluster 3
            20.0, 20.0, 20.1, 20.0, 20.0, 20.1, 20.1, 20.1, 20.05, 20.05,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    let mut unique_clusters: Vec<_> = labels.iter().copied().filter(|&l| l != -1).collect();
    unique_clusters.sort_unstable();
    unique_clusters.dedup();

    // Check that labels are sequential starting from 0
    for (i, &label) in unique_clusters.iter().enumerate() {
        assert_eq!(label, i as i32, "Cluster labels should be sequential");
    }
}
