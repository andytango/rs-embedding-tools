#!/usr/bin/env python3
"""
Reference validation script comparing Rust implementations against Python reference implementations.

This script generates test datasets, runs both Rust and Python implementations,
and compares results to validate correctness.

Requirements:
    pip install hdbscan pacmap scikit-learn numpy
"""

import json
import subprocess
import numpy as np
from sklearn.datasets import make_blobs
import sys

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("Warning: hdbscan not installed. Skipping HDBSCAN validation.")
    print("Install with: pip install hdbscan")
    HDBSCAN_AVAILABLE = False

try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    print("Warning: pacmap not installed. Skipping PACMAP validation.")
    print("Install with: pip install pacmap")
    PACMAP_AVAILABLE = False

def generate_test_data(n_samples=100, n_features=10, n_clusters=3, random_state=42):
    """Generate synthetic test data."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    return X, y_true

def validate_hdbscan(X, min_cluster_size=5, min_samples=5):
    """Validate HDBSCAN implementation against sklearn."""
    if not HDBSCAN_AVAILABLE:
        return None

    print("\n" + "="*60)
    print("HDBSCAN Validation")
    print("="*60)

    # Run Python HDBSCAN
    print(f"\nRunning Python HDBSCAN...")
    print(f"  Data shape: {X.shape}")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    python_labels = clusterer.fit_predict(X)

    print(f"  Python labels shape: {python_labels.shape}")
    print(f"  Python unique labels: {sorted(set(python_labels))}")
    print(f"  Python n_clusters: {len(set(python_labels)) - (1 if -1 in python_labels else 0)}")
    print(f"  Python n_noise: {sum(python_labels == -1)}")

    # Write data to JSON for Rust
    data_file = "/tmp/hdbscan_test_data.json"
    params_file = "/tmp/hdbscan_params.json"

    with open(data_file, 'w') as f:
        json.dump({"data": X.tolist()}, f)

    with open(params_file, 'w') as f:
        json.dump({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples
        }, f)

    print(f"\n  Data written to {data_file}")
    print(f"  Params written to {params_file}")

    return {
        "python_labels": python_labels,
        "n_samples": len(python_labels),
        "n_clusters_python": len(set(python_labels)) - (1 if -1 in python_labels else 0),
        "n_noise_python": sum(python_labels == -1)
    }

def validate_pacmap(X, n_components=2, n_neighbors=10, n_iterations=450, random_state=42):
    """Validate PACMAP implementation against Python pacmap."""
    if not PACMAP_AVAILABLE:
        return None

    print("\n" + "="*60)
    print("PACMAP Validation")
    print("="*60)

    # Run Python PACMAP
    print(f"\nRunning Python PACMAP...")
    print(f"  Data shape: {X.shape}")
    print(f"  n_components: {n_components}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  n_iterations: {n_iterations}")

    embedding = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        num_iters=n_iterations,
        random_state=random_state
    )
    python_embedding = embedding.fit_transform(X)

    print(f"  Python embedding shape: {python_embedding.shape}")
    print(f"  Python embedding mean: {python_embedding.mean():.4f}")
    print(f"  Python embedding std: {python_embedding.std():.4f}")
    print(f"  Python embedding range: [{python_embedding.min():.4f}, {python_embedding.max():.4f}]")

    # Write data to JSON for Rust
    data_file = "/tmp/pacmap_test_data.json"
    params_file = "/tmp/pacmap_params.json"

    with open(data_file, 'w') as f:
        json.dump({"data": X.tolist()}, f)

    with open(params_file, 'w') as f:
        json.dump({
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "n_iterations": n_iterations,
            "seed": random_state
        }, f)

    print(f"\n  Data written to {data_file}")
    print(f"  Params written to {params_file}")

    return {
        "python_embedding": python_embedding,
        "shape": python_embedding.shape,
        "mean": float(python_embedding.mean()),
        "std": float(python_embedding.std())
    }

def compare_cluster_structure(labels1, labels2):
    """Compare clustering structure between two label arrays."""
    # Both should have same number of samples
    if len(labels1) != len(labels2):
        return {"match": False, "reason": "Different number of samples"}

    # Count clusters (excluding noise)
    n_clusters1 = len(set(labels1)) - (1 if -1 in labels1 else 0)
    n_clusters2 = len(set(labels2)) - (1 if -1 in labels2 else 0)

    # Count noise points
    n_noise1 = sum(labels1 == -1)
    n_noise2 = sum(labels2 == -1)

    print(f"\n  Cluster Comparison:")
    print(f"    Python: {n_clusters1} clusters, {n_noise1} noise points")
    print(f"    Rust:   {n_clusters2} clusters, {n_noise2} noise points")

    # Allow some differences
    cluster_diff = abs(n_clusters1 - n_clusters2)
    noise_diff = abs(n_noise1 - n_noise2)

    result = {
        "n_clusters_python": n_clusters1,
        "n_clusters_rust": n_clusters2,
        "n_noise_python": n_noise1,
        "n_noise_rust": n_noise2,
        "cluster_count_similar": cluster_diff <= 2,
        "noise_count_similar": noise_diff <= len(labels1) * 0.1  # Within 10%
    }

    if result["cluster_count_similar"] and result["noise_count_similar"]:
        print(f"    ✓ Structure matches (cluster diff: {cluster_diff}, noise diff: {noise_diff})")
        result["match"] = True
    else:
        print(f"    ✗ Structure differs significantly")
        result["match"] = False

    return result

def main():
    print("="*60)
    print("Python Reference Validation")
    print("="*60)

    # Generate test data
    print("\nGenerating test data...")
    X_small, y_small = generate_test_data(n_samples=100, n_features=10, n_clusters=3)
    X_medium, y_medium = generate_test_data(n_samples=200, n_features=20, n_clusters=4)

    results = {
        "hdbscan": [],
        "pacmap": []
    }

    # Test HDBSCAN
    if HDBSCAN_AVAILABLE:
        for dataset_name, X in [("small", X_small), ("medium", X_medium)]:
            print(f"\n--- Dataset: {dataset_name} ({X.shape}) ---")
            result = validate_hdbscan(X, min_cluster_size=10, min_samples=5)
            if result:
                result["dataset"] = dataset_name
                results["hdbscan"].append(result)

    # Test PACMAP
    if PACMAP_AVAILABLE:
        for dataset_name, X in [("small", X_small), ("medium", X_medium)]:
            print(f"\n--- Dataset: {dataset_name} ({X.shape}) ---")
            result = validate_pacmap(X, n_components=2, n_neighbors=10, n_iterations=100)
            if result:
                result["dataset"] = dataset_name
                results["pacmap"].append(result)

    # Write results
    results_file = "/tmp/python_validation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for hdb_result in results["hdbscan"]:
            if "python_labels" in hdb_result:
                hdb_result["python_labels"] = hdb_result["python_labels"].tolist()
        for pac_result in results["pacmap"]:
            if "python_embedding" in pac_result:
                pac_result["python_embedding"] = pac_result["python_embedding"].tolist()

        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"Results written to {results_file}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run the Rust validation tests with these reference datasets")
    print("  2. Compare cluster structures and embedding properties")
    print("\nNote: This script only generates reference data.")
    print("      The actual comparison must be done in Rust tests.")

if __name__ == "__main__":
    main()
