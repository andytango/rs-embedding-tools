//! Node.js N-API bindings for embedding processing tools
//!
//! This module provides Node.js native bindings for the embedding processing pipeline,
//! including dimensionality reduction using PACMAP and clustering using HDBSCAN.

use napi_derive::napi;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{EmbeddingPipeline, PipelineConfig as RustPipelineConfig};

/// Input format for embedding data
#[derive(Debug, Deserialize)]
struct DataInput {
    data: Vec<Vec<f32>>,
}

/// Configuration for the embedding processing pipeline
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of components for dimensionality reduction (default: 3)
    pub n_components: Option<u32>,

    /// Number of neighbors for PACMAP (default: auto-determined)
    pub n_neighbors: Option<u32>,

    /// Number of iterations for PACMAP (default: 450)
    pub n_iterations: Option<u32>,

    /// Minimum cluster size for HDBSCAN (default: 5)
    pub min_cluster_size: Option<u32>,

    /// Minimum samples for HDBSCAN (default: 5)
    pub min_samples: Option<u32>,

    /// Whether to skip clustering (default: false)
    pub skip_clustering: Option<bool>,
}

impl From<PipelineConfig> for RustPipelineConfig {
    fn from(config: PipelineConfig) -> Self {
        RustPipelineConfig {
            n_components: config.n_components.map(|v| v as usize).unwrap_or(3),
            n_neighbors: config.n_neighbors.map(|v| v as usize),
            n_iterations: config.n_iterations.map(|v| v as usize).unwrap_or(450),
            min_cluster_size: config.min_cluster_size.map(|v| v as usize).unwrap_or(5),
            min_samples: config.min_samples.map(|v| v as usize).unwrap_or(5),
            skip_clustering: config.skip_clustering.unwrap_or(false),
        }
    }
}

/// Process embeddings through the full pipeline
///
/// This performs a two-stage process:
/// 1. Reduce high-dimensional embeddings to 3D (or specified dimensions) using PACMAP
/// 2. Cluster the reduced data using HDBSCAN
///
/// # Arguments
///
/// * `data_json` - Input embeddings as a JSON string representing a 2D array (samples x features)
/// * `config` - Optional configuration for the pipeline
///
/// # Returns
///
/// A JSON string containing cluster assignments, reduced data, and metadata
///
/// # Example
///
/// ```javascript
/// const { processEmbeddings } = require('embedding-tools');
///
/// const data = JSON.stringify([
///   [1.0, 2.0, 3.0],
///   [1.1, 2.1, 3.1],
///   [10.0, 11.0, 12.0],
/// ]);
///
/// const config = {
///   n_components: 3,
///   min_cluster_size: 2,
/// };
///
/// const resultJson = processEmbeddings(data, config);
/// const result = JSON.parse(resultJson);
/// console.log(result.clusters);
/// console.log(result.reduced_data);
/// ```
#[napi]
pub fn process_embeddings(
    data_json: String,
    config: Option<PipelineConfig>,
) -> napi::Result<String> {
    // Parse JSON input
    let input: DataInput = serde_json::from_str(&data_json)
        .map_err(|e| napi::Error::from_reason(format!("Failed to parse input JSON: {}", e)))?;

    let data = input.data;

    if data.is_empty() {
        return Err(napi::Error::from_reason("Input data is empty".to_string()));
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Check all rows have the same length
    for (i, row) in data.iter().enumerate() {
        if row.len() != n_features {
            return Err(napi::Error::from_reason(format!(
                "Inconsistent dimensions: row {} has {} features, expected {}",
                i,
                row.len(),
                n_features
            )));
        }
    }

    // Convert to ndarray
    let flat_data: Vec<f64> = data.into_iter().flatten().map(|x| x as f64).collect();

    let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| napi::Error::from_reason(format!("Failed to create array: {}", e)))?;

    // Create pipeline with config
    let rust_config: RustPipelineConfig = config
        .unwrap_or(PipelineConfig {
            n_components: Some(3),
            n_neighbors: None,
            n_iterations: Some(450),
            min_cluster_size: Some(5),
            min_samples: Some(5),
            skip_clustering: Some(false),
        })
        .into();

    let pipeline = EmbeddingPipeline::new(rust_config);

    // Process through pipeline
    let output = pipeline
        .process(data_array)
        .map_err(|e| napi::Error::from_reason(format!("Processing error: {}", e)))?;

    // Serialize output to JSON
    serde_json::to_string(&output)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize output: {}", e)))
}

/// Reduce the dimensionality of embeddings without clustering
///
/// Uses PACMAP for dimensionality reduction. Default is to reduce to 3D.
///
/// # Arguments
///
/// * `data_json` - Input embeddings as a JSON string representing a 2D array (samples x features)
/// * `n_components` - Optional number of dimensions to reduce to (default: 3)
///
/// # Returns
///
/// A JSON string containing the reduced 2D array of embeddings
///
/// # Example
///
/// ```javascript
/// const { reduceDimensions } = require('embedding-tools');
///
/// const data = JSON.stringify([
///   [1.0, 2.0, 3.0, 4.0, 5.0],
///   [2.0, 3.0, 4.0, 5.0, 6.0],
/// ]);
///
/// const reducedJson = reduceDimensions(data, 2);
/// const reduced = JSON.parse(reducedJson);
/// console.log(reduced); // [[x1, y1], [x2, y2]]
/// ```
#[napi]
pub fn reduce_dimensions(data_json: String, n_components: Option<u32>) -> napi::Result<String> {
    // Parse JSON input
    let input: DataInput = serde_json::from_str(&data_json)
        .map_err(|e| napi::Error::from_reason(format!("Failed to parse input JSON: {}", e)))?;

    let data = input.data;

    if data.is_empty() {
        return Err(napi::Error::from_reason("Input data is empty".to_string()));
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Convert to ndarray
    let flat_data: Vec<f64> = data.into_iter().flatten().map(|x| x as f64).collect();

    let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| napi::Error::from_reason(format!("Failed to create array: {}", e)))?;

    // Create configuration for reduction only
    let config = RustPipelineConfig {
        n_components: n_components.map(|v| v as usize).unwrap_or(3),
        skip_clustering: true,
        ..Default::default()
    };

    let pipeline = EmbeddingPipeline::new(config);

    // Process through pipeline
    let output = pipeline
        .process(data_array)
        .map_err(|e| napi::Error::from_reason(format!("Processing error: {}", e)))?;

    // Serialize just the reduced data
    serde_json::to_string(&output.reduced_data)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize output: {}", e)))
}

/// Cluster pre-reduced data using HDBSCAN
///
/// # Arguments
///
/// * `data_json` - Pre-reduced embeddings as a JSON string representing a 2D array (samples x features)
/// * `min_cluster_size` - Optional minimum cluster size (default: 5)
///
/// # Returns
///
/// A JSON string containing a vector of cluster assignments (null for noise/outliers)
///
/// # Example
///
/// ```javascript
/// const { clusterData } = require('embedding-tools');
///
/// const reducedData = JSON.stringify([
///   [1.0, 2.0, 3.0],
///   [1.1, 2.1, 3.1],
///   [10.0, 11.0, 12.0],
/// ]);
///
/// const clustersJson = clusterData(reducedData, 2);
/// const clusters = JSON.parse(clustersJson);
/// console.log(clusters); // [0, 0, null] (noise point)
/// ```
#[napi]
pub fn cluster_data(data_json: String, min_cluster_size: Option<u32>) -> napi::Result<String> {
    // Parse JSON input
    let input: DataInput = serde_json::from_str(&data_json)
        .map_err(|e| napi::Error::from_reason(format!("Failed to parse input JSON: {}", e)))?;

    let data: Vec<Vec<f64>> = input
        .data
        .into_iter()
        .map(|row| row.into_iter().map(|x| x as f64).collect())
        .collect();

    if data.is_empty() {
        return Err(napi::Error::from_reason("Input data is empty".to_string()));
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Convert to ndarray
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| napi::Error::from_reason(format!("Failed to create array: {}", e)))?;

    // Perform clustering
    let min_cluster_size = min_cluster_size.map(|v| v as usize).unwrap_or(5);
    let hdbscan = hdbscan::HdbscanBuilder::new()
        .min_cluster_size(min_cluster_size)
        .min_samples(min_cluster_size)
        .build();

    let cluster_labels = hdbscan
        .fit_predict(&data_array)
        .map_err(|e| napi::Error::from_reason(format!("Clustering error: {}", e)))?;

    // Convert to Option<u32> format
    let clusters: Vec<Option<u32>> = cluster_labels
        .into_iter()
        .map(|label| {
            if label == -1 {
                None
            } else {
                Some(label as u32)
            }
        })
        .collect();

    serde_json::to_string(&clusters)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize output: {}", e)))
}
