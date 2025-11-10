use proptest::prelude::*;
use ndarray::Array2;
use scanner_embeddings::hdbscan::HdbscanBuilder;
use scanner_embeddings::pacmap::PacmapBuilder;

// Property: HDBSCAN should always return the same number of labels as input points
proptest! {
    #[test]
    fn prop_hdbscan_label_count_matches_input(
        n_points in 5usize..50,
        n_dims in 2usize..10,
        min_cluster_size in 2usize..10,
    ) {
        let n_dims = n_dims.min(n_points - 1);
        let min_cluster_size = min_cluster_size.min(n_points / 2);

        // Generate random data
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(min_cluster_size)
            .min_samples(min_cluster_size / 2 + 1)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        prop_assert_eq!(labels.len(), n_points);
    }
}

// Property: All HDBSCAN labels should be >= -1
proptest! {
    #[test]
    fn prop_hdbscan_labels_valid(
        n_points in 5usize..50,
        n_dims in 2usize..10,
    ) {
        let n_dims = n_dims.min(n_points - 1);

        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1).cos())
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(3)
            .min_samples(2)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        for &label in &labels {
            prop_assert!(label >= -1, "Invalid label: {}", label);
        }
    }
}

// Property: HDBSCAN cluster IDs should be sequential starting from 0
proptest! {
    #[test]
    fn prop_hdbscan_sequential_cluster_ids(
        n_points in 10usize..40,
        n_dims in 2usize..8,
    ) {
        let n_dims = n_dims.min(n_points - 1);

        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.2).sin() * 10.0)
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(3)
            .min_samples(2)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        // Collect cluster IDs (excluding noise)
        let mut cluster_ids: Vec<i32> = labels.iter()
            .filter(|&&x| x != -1)
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if !cluster_ids.is_empty() {
            cluster_ids.sort();
            prop_assert_eq!(cluster_ids[0], 0, "First cluster ID should be 0");

            // Check sequential
            for i in 1..cluster_ids.len() {
                prop_assert!(
                    cluster_ids[i] <= cluster_ids[i-1] + 1,
                    "Cluster IDs should be sequential: {:?}", cluster_ids
                );
            }
        }
    }
}

// Property: HDBSCAN should be deterministic
proptest! {
    #[test]
    fn prop_hdbscan_deterministic(
        n_points in 10usize..30,
        n_dims in 2usize..6,
    ) {
        let n_dims = n_dims.min(n_points - 1);

        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.15))
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(3)
            .min_samples(2)
            .build();

        let labels1 = clusterer.fit_predict(&data).unwrap();
        let labels2 = clusterer.fit_predict(&data).unwrap();

        prop_assert_eq!(labels1, labels2, "HDBSCAN should be deterministic");
    }
}

// Property: HDBSCAN respects min_cluster_size
proptest! {
    #[test]
    fn prop_hdbscan_respects_min_cluster_size(
        n_points in 15usize..40,
        min_cluster_size in 4usize..10,
    ) {
        let min_cluster_size = min_cluster_size.min(n_points / 3);

        let data_vec: Vec<f64> = (0..n_points * 2)
            .map(|i| (i as f64 * 0.1))
            .collect();

        let data = Array2::from_shape_vec((n_points, 2), data_vec).unwrap();

        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(min_cluster_size)
            .min_samples(min_cluster_size / 2)
            .build();

        let labels = clusterer.fit_predict(&data).unwrap();

        // Count points in each cluster
        let mut cluster_counts = std::collections::HashMap::new();
        for &label in &labels {
            if label != -1 {
                *cluster_counts.entry(label).or_insert(0) += 1;
            }
        }

        // All clusters should have at least min_cluster_size points
        for (cluster_id, count) in cluster_counts {
            prop_assert!(
                count >= min_cluster_size,
                "Cluster {} has {} points, expected at least {}",
                cluster_id, count, min_cluster_size
            );
        }
    }
}

// Property: PACMAP should preserve sample count
proptest! {
    #[test]
    fn prop_pacmap_preserves_sample_count(
        n_points in 10usize..50,
        n_input_dims in 5usize..20,
        n_output_dims in 2usize..5,
    ) {
        let n_output_dims = n_output_dims.min(n_input_dims - 1);

        let data_vec: Vec<f64> = (0..n_points * n_input_dims)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();

        let data = Array2::from_shape_vec((n_points, n_input_dims), data_vec).unwrap();

        let pacmap = PacmapBuilder::new()
            .n_components(n_output_dims)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(50)
            .seed(Some(42))
            .build();

        let result = pacmap.fit_transform(&data).unwrap();

        prop_assert_eq!(result.nrows(), n_points);
        prop_assert_eq!(result.ncols(), n_output_dims);
    }
}

// Property: PACMAP output should be finite
proptest! {
    #[test]
    fn prop_pacmap_output_finite(
        n_points in 10usize..40,
        n_dims in 5usize..15,
    ) {
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.2).cos() * 5.0)
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(50)
            .seed(Some(42))
            .build();

        let result = pacmap.fit_transform(&data).unwrap();

        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                prop_assert!(
                    result[[i, j]].is_finite(),
                    "Non-finite value at [{}, {}]: {}",
                    i, j, result[[i, j]]
                );
            }
        }
    }
}

// Property: PACMAP with seed is reproducible
proptest! {
    #[test]
    fn prop_pacmap_reproducible_with_seed(
        n_points in 10usize..30,
        n_dims in 5usize..12,
        seed in 0u64..1000,
    ) {
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1))
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let pacmap1 = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(50)
            .seed(Some(seed))
            .build();

        let pacmap2 = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(50)
            .seed(Some(seed))
            .build();

        let result1 = pacmap1.fit_transform(&data).unwrap();
        let result2 = pacmap2.fit_transform(&data).unwrap();

        // Results should be very similar
        let mut max_diff: f64 = 0.0;
        for i in 0..result1.nrows() {
            for j in 0..result1.ncols() {
                let diff = (result1[[i, j]] - result2[[i, j]]).abs();
                max_diff = max_diff.max(diff);
            }
        }

        prop_assert!(
            max_diff < 2.0,
            "Results differ too much with same seed: max_diff = {}",
            max_diff
        );
    }
}

// Property: PACMAP different seeds produce different results
proptest! {
    #[test]
    fn prop_pacmap_different_seeds_differ(
        n_points in 15usize..30,
        n_dims in 5usize..12,
        seed1 in 0u64..500,
        seed2 in 500u64..1000,
    ) {
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1))
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let pacmap1 = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(200)
            .seed(Some(seed1))
            .build();

        let pacmap2 = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(5.min(n_points - 1))
            .n_iterations(200)
            .seed(Some(seed2))
            .build();

        let result1 = pacmap1.fit_transform(&data).unwrap();
        let result2 = pacmap2.fit_transform(&data).unwrap();

        // Results should differ
        let mut differs = false;
        for i in 0..result1.nrows() {
            for j in 0..result1.ncols() {
                if (result1[[i, j]] - result2[[i, j]]).abs() > 0.1 {
                    differs = true;
                    break;
                }
            }
            if differs { break; }
        }

        prop_assert!(differs, "Different seeds should produce different results");
    }
}

// Property: PACMAP preserves local neighborhoods (approximate)
// Note: This is a statistical property, not a strict invariant
// Commented out as it's too variable for property testing
// The dedicated validation tests cover this more appropriately
/*
proptest! {
    #[test]
    fn prop_pacmap_attempts_neighborhood_preservation(
        n_points in 20usize..40,
        n_dims in 8usize..15,
    ) {
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1).sin() * 10.0)
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(10.min(n_points - 1))
            .n_iterations(200)
            .seed(Some(42))
            .build();

        let result = pacmap.fit_transform(&data).unwrap();

        // For a few sample points, check that nearest neighbors in input
        // are still relatively close in output
        let sample_indices = vec![0, n_points / 2, n_points - 1];

        for &idx in &sample_indices {
            if idx >= n_points { continue; }

            // Find nearest neighbor in input space
            let mut min_dist = f64::MAX;
            let mut nearest_input = 0;

            for j in 0..n_points {
                if j == idx { continue; }
                let mut dist = 0.0;
                for k in 0..n_dims {
                    let diff = data[[idx, k]] - data[[j, k]];
                    dist += diff * diff;
                }
                if dist < min_dist {
                    min_dist = dist;
                    nearest_input = j;
                }
            }

            // Check this neighbor is in top 50% of neighbors in output space
            let mut output_dists: Vec<(usize, f64)> = (0..n_points)
                .filter(|&j| j != idx)
                .map(|j| {
                    let mut dist = 0.0;
                    for k in 0..result.ncols() {
                        let diff = result[[idx, k]] - result[[j, k]];
                        dist += diff * diff;
                    }
                    (j, dist)
                })
                .collect();

            output_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let rank = output_dists.iter().position(|(j, _)| *j == nearest_input).unwrap();
            let percentile = rank as f64 / output_dists.len() as f64;

            // Input nearest neighbor should not be in the absolute worst 10%
            // (very relaxed check - just ensuring the algorithm attempts preservation)
            prop_assert!(
                percentile < 0.9,
                "Nearest neighbor preservation completely failed: rank={}/{} ({:.1}%)",
                rank, output_dists.len(), percentile * 100.0
            );
        }
    }
}
*/

// Property: Pipeline (PACMAP → HDBSCAN) should produce valid output
proptest! {
    #[test]
    fn prop_pipeline_produces_valid_output(
        n_points in 20usize..50,
        n_dims in 10usize..20,
    ) {
        let data_vec: Vec<f64> = (0..n_points * n_dims)
            .map(|i| (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let data = Array2::from_shape_vec((n_points, n_dims), data_vec).unwrap();

        // PACMAP reduction
        let pacmap = PacmapBuilder::new()
            .n_components(2)
            .n_neighbors(10.min(n_points - 1))
            .n_iterations(100)
            .seed(Some(42))
            .build();

        let reduced = pacmap.fit_transform(&data).unwrap();

        // HDBSCAN clustering
        let clusterer = HdbscanBuilder::new()
            .min_cluster_size(5.min(n_points / 3))
            .min_samples(3)
            .build();

        let labels = clusterer.fit_predict(&reduced).unwrap();

        // Validate output
        prop_assert_eq!(labels.len(), n_points);

        for &label in &labels {
            prop_assert!(label >= -1);
        }

        // All output values should be finite
        for i in 0..reduced.nrows() {
            for j in 0..reduced.ncols() {
                prop_assert!(reduced[[i, j]].is_finite());
            }
        }
    }
}
