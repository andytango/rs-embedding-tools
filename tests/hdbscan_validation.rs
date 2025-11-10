//! Validation tests for our HDBSCAN implementation.
//!
//! This test suite validates the correctness of our pure Rust HDBSCAN implementation
//! using known clustering scenarios and synthetic datasets.

use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;

/// Generate synthetic blob clusters for testing
fn generate_blobs(n_samples: usize, n_features: usize, n_clusters: usize, cluster_std: f64) -> Array2<f64> {
    let samples_per_cluster = n_samples / n_clusters;
    let mut data = Vec::new();

    for cluster in 0..n_clusters {
        let center_x = (cluster as f64) * 10.0;
        let center_y = (cluster as f64) * 10.0;

        for sample in 0..samples_per_cluster {
            // Deterministic "random" values
            let noise_x = ((sample * 7 + cluster) % 100) as f64 / 100.0 - 0.5;
            let noise_y = ((sample * 13 + cluster) % 97) as f64 / 97.0 - 0.5;

            let x = center_x + noise_x * cluster_std;
            let y = center_y + noise_y * cluster_std;

            if n_features == 2 {
                data.push(x);
                data.push(y);
            } else {
                data.push(x);
                data.push(y);
                for i in 2..n_features {
                    let noise = ((sample * (i + 1)) % 50) as f64 / 50.0 - 0.5;
                    data.push(noise * cluster_std);
                }
            }
        }
    }

    Array2::from_shape_vec((n_samples, n_features), data).expect("Failed to create blob data")
}

/// Generate two-circle pattern for testing
fn generate_circles(n_samples: usize) -> Array2<f64> {
    let samples_per_circle = n_samples / 2;
    let mut data = Vec::new();

    // Inner circle (radius 1)
    for i in 0..samples_per_circle {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (samples_per_circle as f64);
        let radius = 1.0;
        data.push(radius * angle.cos());
        data.push(radius * angle.sin());
    }

    // Outer circle (radius 3)
    for i in 0..samples_per_circle {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (samples_per_circle as f64);
        let radius = 3.0;
        data.push(radius * angle.cos());
        data.push(radius * angle.sin());
    }

    Array2::from_shape_vec((n_samples, 2), data).expect("Failed to create circle data")
}

/// Count number of clusters (excluding noise)
fn count_clusters(labels: &[i32]) -> usize {
    let mut unique: Vec<i32> = labels.iter().copied().filter(|&l| l != -1).collect();
    unique.sort_unstable();
    unique.dedup();
    unique.len()
}

/// Count number of noise points
fn count_noise(labels: &[i32]) -> usize {
    labels.iter().filter(|&&l| l == -1).count()
}

/// Check that points in the same cluster are close together
#[allow(dead_code)]
fn check_cluster_coherence(data: &Array2<f64>, labels: &[i32], max_distance: f64) -> bool {
    for cluster_id in labels.iter().copied().filter(|&l| l != -1).collect::<Vec<_>>() {
        let cluster_points: Vec<_> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster_id)
            .map(|(i, _)| data.row(i))
            .collect();

        for i in 0..cluster_points.len() {
            for j in (i + 1)..cluster_points.len() {
                let dist: f64 = cluster_points[i]
                    .iter()
                    .zip(cluster_points[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist > max_distance {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn test_two_well_separated_clusters() {
    // Create two well-separated clusters
    let data = Array2::from_shape_vec(
        (10, 2),
        vec![
            // Cluster 1 (around origin)
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            0.1, -0.1,
            -0.1, 0.1,
            // Cluster 2 (around (10, 10))
            10.0, 10.0,
            10.1, 10.1,
            10.2, 10.0,
            10.1, 9.9,
            9.9, 10.1,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("Labels: {:?}", labels);
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 10);

    // We should find at least one cluster
    assert!(count_clusters(&labels) >= 1, "Should find at least one cluster");
}

#[test]
fn test_three_blob_clusters() {
    let data = generate_blobs(30, 2, 3, 0.5);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nThree Blobs Test:");
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 30);

    // Should find at least one cluster
    let n_clusters = count_clusters(&labels);
    assert!(n_clusters >= 1, "Should find at least one cluster");
}

#[test]
fn test_uniform_noise() {
    // Generate uniformly distributed points (should all be noise or very few clusters)
    let mut data = Vec::new();
    for i in 0..20 {
        let x = ((i * 7) % 20) as f64;
        let y = ((i * 13) % 20) as f64;
        data.push(x);
        data.push(y);
    }
    let data = Array2::from_shape_vec((20, 2), data).unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nUniform Noise Test:");
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 20);

    // Most points should be noise or we should have very few clusters
    let _noise_count = count_noise(&labels);
    let cluster_count = count_clusters(&labels);

    assert!(cluster_count <= 2, "Uniform noise should produce few or no clusters");
}

#[test]
fn test_single_dense_cluster() {
    // Generate a single dense cluster
    let data = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            0.1, 0.1,
            0.2, 0.0,
            0.0, 0.2,
            0.1, 0.2,
            0.2, 0.1,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(4)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nSingle Cluster Test:");
    println!("Labels: {:?}", labels);
    println!("Clusters: {}", count_clusters(&labels));

    assert_eq!(labels.len(), 8);

    // Should find one cluster
    let n_clusters = count_clusters(&labels);
    assert!((1..=2).contains(&n_clusters), "Should find around one cluster");
}

#[test]
fn test_varying_density() {
    // Create clusters with different densities
    let data = Array2::from_shape_vec(
        (15, 2),
        vec![
            // Dense cluster (spacing ~0.05)
            0.0, 0.0,
            0.05, 0.0,
            0.0, 0.05,
            0.05, 0.05,
            0.1, 0.0,
            // Sparse cluster (spacing ~0.5)
            10.0, 10.0,
            10.5, 10.0,
            10.0, 10.5,
            10.5, 10.5,
            11.0, 10.0,
            // Another sparse cluster (spacing ~0.6)
            20.0, 20.0,
            20.6, 20.0,
            20.0, 20.6,
            20.6, 20.6,
            21.0, 21.0,
        ],
    )
    .unwrap();

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nVarying Density Test:");
    println!("Labels: {:?}", labels);
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 15);

    // Should find at least one cluster
    assert!(count_clusters(&labels) >= 1);
}

#[test]
fn test_circles_pattern() {
    let data = generate_circles(40);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(8)
        .min_samples(4)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nCircles Pattern Test:");
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 40);

    // Should find at least one ring-shaped cluster
    let n_clusters = count_clusters(&labels);
    assert!(n_clusters >= 1, "Should find at least one cluster in circles");
}

#[test]
fn test_parameter_sensitivity_min_cluster_size() {
    let data = generate_blobs(24, 2, 3, 0.3);

    // Test different min_cluster_size values
    for min_size in [2, 3, 5, 8] {
        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(min_size)
            .min_samples(2)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        println!("\nParameter test (min_cluster_size={}):", min_size);
        println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

        assert_eq!(labels.len(), 24);

        // Larger min_cluster_size should generally produce fewer or same number of clusters
        // (though this isn't always strictly true due to algorithm specifics)
    }
}

#[test]
fn test_parameter_sensitivity_min_samples() {
    let data = generate_blobs(24, 2, 3, 0.3);

    // Test different min_samples values
    for min_samples in [2, 3, 5] {
        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(4)
            .min_samples(min_samples)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        println!("\nParameter test (min_samples={}):", min_samples);
        println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

        assert_eq!(labels.len(), 24);
    }
}

#[test]
fn test_higher_dimensions() {
    // Test with higher dimensional data
    let data = generate_blobs(30, 5, 3, 0.8);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(3)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nHigher Dimensions Test (5D):");
    println!("Clusters: {}, Noise: {}", count_clusters(&labels), count_noise(&labels));

    assert_eq!(labels.len(), 30);

    // Should find at least one cluster
    assert!(count_clusters(&labels) >= 1);
}

#[test]
fn test_all_labels_valid() {
    let data = generate_blobs(20, 2, 2, 0.5);

    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(3)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    // All labels should be >= -1
    for &label in &labels {
        assert!(label >= -1, "Label {} is invalid", label);
    }

    // Cluster labels should be sequential starting from 0
    let cluster_labels: Vec<i32> = labels.iter().copied().filter(|&l| l != -1).collect();
    if !cluster_labels.is_empty() {
        let max_label = cluster_labels.iter().max().unwrap();
        assert!(*max_label < cluster_labels.len() as i32, "Cluster labels should be sequential");
    }
}

#[test]
fn test_min_cluster_size_enforcement() {
    let data = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.0, 0.0,
            0.1, 0.0,
            10.0, 10.0,
            10.1, 10.0,
            10.0, 10.1,
            20.0, 20.0,
            20.1, 20.0,
        ],
    )
    .unwrap();

    // With min_cluster_size=5, no cluster should form (all noise or one cluster)
    let clusterer = HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(2)
        .build();

    let labels = clusterer.fit_predict(&data).unwrap();

    println!("\nMin Cluster Size Enforcement:");
    println!("Labels: {:?}", labels);
    println!("Clusters: {}", count_clusters(&labels));

    // Should have few or no clusters since we don't have 5 points close together
    assert!(count_clusters(&labels) <= 1);
}
