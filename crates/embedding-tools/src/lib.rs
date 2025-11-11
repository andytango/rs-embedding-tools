//! WASM-compatible embedding processing tools
//!
//! This crate provides a simplified pipeline for processing high-dimensional embeddings,
//! including dimensionality reduction using PACMAP and clustering using HDBSCAN.

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::Array2;
use thiserror::Error;

// Optional PCA module for additional dimensionality reduction
pub mod pca;

/// Error types for the embedding tools
#[derive(Error, Debug)]
pub enum EmbeddingToolsError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("PACMAP error: {0}")]
    Pacmap(#[from] pacmap::PacmapError),

    #[error("HDBSCAN error: {0}")]
    Hdbscan(#[from] hdbscan::HdbscanError),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

/// Input data structure for processing embeddings
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingInput {
    /// The embedding data as a 2D array (samples x features)
    pub data: Vec<Vec<f32>>,

    /// Optional configuration for the pipeline
    #[serde(default)]
    pub config: PipelineConfig,
}

/// Configuration for the embedding processing pipeline
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PipelineConfig {
    /// Number of components for dimensionality reduction (default: 2)
    #[serde(default = "default_n_components")]
    pub n_components: usize,

    /// Number of neighbors for PACMAP (default: auto-determined)
    pub n_neighbors: Option<usize>,

    /// Number of iterations for PACMAP (default: 450)
    #[serde(default = "default_n_iterations")]
    pub n_iterations: usize,

    /// Minimum cluster size for HDBSCAN (default: 5)
    #[serde(default = "default_min_cluster_size")]
    pub min_cluster_size: usize,

    /// Minimum samples for HDBSCAN (default: 5)
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,

    /// Whether to skip clustering (default: false)
    #[serde(default)]
    pub skip_clustering: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            n_components: default_n_components(),
            n_neighbors: None,
            n_iterations: default_n_iterations(),
            min_cluster_size: default_min_cluster_size(),
            min_samples: default_min_samples(),
            skip_clustering: false,
        }
    }
}

fn default_n_components() -> usize { 2 }
fn default_n_iterations() -> usize { 450 }
fn default_min_cluster_size() -> usize { 5 }
fn default_min_samples() -> usize { 5 }

/// Output data structure containing results from the pipeline
#[derive(Serialize, Deserialize, Debug)]
pub struct EmbeddingOutput {
    /// Cluster assignments for each point (-1 for noise/outliers)
    /// None if clustering was skipped
    pub clusters: Option<Vec<Option<usize>>>,

    /// Reduced dimensionality data
    pub reduced_data: Vec<Vec<f64>>,

    /// Metadata about the processing
    pub metadata: ProcessingMetadata,
}

/// Metadata about the processing pipeline
#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessingMetadata {
    /// Original number of dimensions
    pub original_dimensions: usize,

    /// Number of data points processed
    pub n_samples: usize,

    /// Number of clusters found (if clustering was performed)
    pub n_clusters: Option<usize>,

    /// Number of noise points (if clustering was performed)
    pub n_noise: Option<usize>,
}

/// Main processing pipeline for embeddings
pub struct EmbeddingPipeline {
    config: PipelineConfig,
}

impl Default for EmbeddingPipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

impl EmbeddingPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Process embeddings through the pipeline
    pub fn process(&self, data: Array2<f64>) -> Result<EmbeddingOutput, EmbeddingToolsError> {
        let n_samples = data.nrows();
        let original_dimensions = data.ncols();

        if n_samples == 0 {
            return Err(EmbeddingToolsError::InvalidInput("Empty data provided".into()));
        }

        // Dimensionality reduction with PACMAP
        let n_neighbors = self.config.n_neighbors
            .unwrap_or_else(|| (10.min(n_samples - 1)).max(1));

        let pacmap = pacmap::PacmapBuilder::new()
            .n_components(self.config.n_components)
            .n_neighbors(n_neighbors)
            .n_iterations(self.config.n_iterations)
            .build();

        let reduced_data = pacmap.fit_transform(&data)?;

        // Clustering with HDBSCAN (optional)
        let (clusters, n_clusters, n_noise) = if self.config.skip_clustering {
            (None, None, None)
        } else {
            let hdbscan = hdbscan::HdbscanBuilder::new()
                .min_cluster_size(self.config.min_cluster_size)
                .min_samples(self.config.min_samples)
                .build();

            let cluster_labels = hdbscan.fit_predict(&reduced_data)?;

            // Convert to Option<usize> format and calculate statistics
            let mut max_cluster = 0;
            let mut noise_count = 0;

            let clusters: Vec<Option<usize>> = cluster_labels
                .into_iter()
                .map(|label| {
                    if label == -1 {
                        noise_count += 1;
                        None
                    } else {
                        let cluster = label as usize;
                        if cluster > max_cluster {
                            max_cluster = cluster;
                        }
                        Some(cluster)
                    }
                })
                .collect();

            let n_clusters = if max_cluster > 0 || clusters.iter().any(|c| c.is_some()) {
                max_cluster + 1
            } else {
                0
            };

            (Some(clusters), Some(n_clusters), Some(noise_count))
        };

        // Convert reduced data to Vec<Vec<f64>> for serialization
        let reduced_data_vec: Vec<Vec<f64>> = reduced_data
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        Ok(EmbeddingOutput {
            clusters,
            reduced_data: reduced_data_vec,
            metadata: ProcessingMetadata {
                original_dimensions,
                n_samples,
                n_clusters,
                n_noise,
            },
        })
    }
}

/// WASM binding for processing embeddings
#[wasm_bindgen]
pub fn process_embeddings(input_js: JsValue) -> Result<JsValue, JsValue> {
    // Set up panic hook for better error messages in WASM
    console_error_panic_hook::set_once();

    // Deserialize input
    let input: EmbeddingInput = serde_wasm_bindgen::from_value(input_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse input: {}", e)))?;

    // Validate input
    if input.data.is_empty() {
        return Err(JsValue::from_str("Input data is empty"));
    }

    let n_samples = input.data.len();
    let n_features = input.data[0].len();

    // Check all rows have the same length
    for (i, row) in input.data.iter().enumerate() {
        if row.len() != n_features {
            return Err(JsValue::from_str(&format!(
                "Inconsistent dimensions: row {} has {} features, expected {}",
                i, row.len(), n_features
            )));
        }
    }

    // Convert to ndarray
    let flat_data: Vec<f64> = input.data
        .into_iter()
        .flatten()
        .map(|x| x as f64)
        .collect();

    let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| JsValue::from_str(&format!("Failed to create array: {}", e)))?;

    // Process through pipeline
    let pipeline = EmbeddingPipeline::new(input.config);
    let output = pipeline.process(data_array)
        .map_err(|e| JsValue::from_str(&format!("Processing error: {}", e)))?;

    // Serialize output
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
}

/// WASM binding for simple dimensionality reduction without clustering
#[wasm_bindgen]
pub fn reduce_dimensions(input_js: JsValue, n_components: Option<usize>) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    // Parse input as simple 2D array
    let data: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(input_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse input: {}", e)))?;

    if data.is_empty() {
        return Err(JsValue::from_str("Input data is empty"));
    }

    // Create configuration for reduction only
    let config = PipelineConfig {
        n_components: n_components.unwrap_or(2),
        skip_clustering: true,
        ..Default::default()
    };

    let input = EmbeddingInput { data, config };

    // Process and extract only reduced data
    let input_value = serde_wasm_bindgen::to_value(&input)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize input: {}", e)))?;

    let output_value = process_embeddings(input_value)?;
    let output: EmbeddingOutput = serde_wasm_bindgen::from_value(output_value)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse output: {}", e)))?;

    // Return just the reduced data
    serde_wasm_bindgen::to_value(&output.reduced_data)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize reduced data: {}", e)))
}

/// WASM binding for clustering pre-reduced data
#[wasm_bindgen]
pub fn cluster_data(input_js: JsValue, min_cluster_size: Option<usize>) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    // Parse input as 2D array
    let data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(input_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse input: {}", e)))?;

    if data.is_empty() {
        return Err(JsValue::from_str("Input data is empty"));
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Convert to ndarray
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();
    let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data)
        .map_err(|e| JsValue::from_str(&format!("Failed to create array: {}", e)))?;

    // Perform clustering
    let min_cluster_size = min_cluster_size.unwrap_or(5);
    let hdbscan = hdbscan::HdbscanBuilder::new()
        .min_cluster_size(min_cluster_size)
        .min_samples(min_cluster_size)
        .build();

    let cluster_labels = hdbscan.fit_predict(&data_array)
        .map_err(|e| JsValue::from_str(&format!("Clustering error: {}", e)))?;

    // Convert to Option<usize> format
    let clusters: Vec<Option<usize>> = cluster_labels
        .into_iter()
        .map(|label| {
            if label == -1 {
                None
            } else {
                Some(label as usize)
            }
        })
        .collect();

    serde_wasm_bindgen::to_value(&clusters)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize clusters: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
            vec![10.0, 11.0, 12.0],
            vec![10.1, 11.1, 12.1],
            vec![10.2, 11.2, 12.2],
            vec![100.0, 101.0, 102.0],
            vec![100.1, 101.1, 102.1],
            vec![100.2, 101.2, 102.2],
        ]
    }

    #[test]
    fn test_pipeline_with_default_config() {
        let data = generate_test_data();
        let input = EmbeddingInput {
            data,
            config: PipelineConfig::default(),
        };

        let flat_data: Vec<f64> = input.data
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .collect();

        let n_samples = input.data.len();
        let n_features = input.data[0].len();
        let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data).unwrap();

        let pipeline = EmbeddingPipeline::new(input.config);
        let output = pipeline.process(data_array).unwrap();

        assert_eq!(output.reduced_data.len(), n_samples);
        assert_eq!(output.reduced_data[0].len(), 2);
        assert!(output.clusters.is_some());
        assert_eq!(output.metadata.n_samples, n_samples);
        assert_eq!(output.metadata.original_dimensions, n_features);
    }

    #[test]
    fn test_pipeline_skip_clustering() {
        let data = generate_test_data();
        let config = PipelineConfig {
            skip_clustering: true,
            ..Default::default()
        };

        let input = EmbeddingInput { data, config };

        let flat_data: Vec<f64> = input.data
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .collect();

        let n_samples = input.data.len();
        let n_features = input.data[0].len();
        let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data).unwrap();

        let pipeline = EmbeddingPipeline::new(input.config);
        let output = pipeline.process(data_array).unwrap();

        assert!(output.clusters.is_none());
        assert!(output.metadata.n_clusters.is_none());
        assert!(output.metadata.n_noise.is_none());
    }

    #[test]
    fn test_custom_dimensions() {
        let data = generate_test_data();
        let config = PipelineConfig {
            n_components: 3,
            skip_clustering: true,
            ..Default::default()
        };

        let input = EmbeddingInput { data, config };

        let flat_data: Vec<f64> = input.data
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .collect();

        let n_samples = input.data.len();
        let n_features = input.data[0].len();
        let data_array = Array2::from_shape_vec((n_samples, n_features), flat_data).unwrap();

        let pipeline = EmbeddingPipeline::new(input.config);
        let output = pipeline.process(data_array).unwrap();

        assert_eq!(output.reduced_data[0].len(), 3);
    }
}

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_process_embeddings() {
        let input = EmbeddingInput {
            data: vec![
                vec![1.0, 2.0, 3.0],
                vec![1.1, 2.1, 3.1],
                vec![10.0, 11.0, 12.0],
                vec![10.1, 11.1, 12.1],
                vec![100.0, 101.0, 102.0],
                vec![100.1, 101.1, 102.1],
            ],
            config: PipelineConfig::default(),
        };

        let input_js = serde_wasm_bindgen::to_value(&input).unwrap();
        let output_js = process_embeddings(input_js).unwrap();
        let output: EmbeddingOutput = serde_wasm_bindgen::from_value(output_js).unwrap();

        assert_eq!(output.reduced_data.len(), 6);
        assert_eq!(output.reduced_data[0].len(), 2);
    }

    #[wasm_bindgen_test]
    fn test_wasm_reduce_dimensions() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        let input_js = serde_wasm_bindgen::to_value(&data).unwrap();
        let output_js = reduce_dimensions(input_js, Some(2)).unwrap();
        let reduced: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(output_js).unwrap();

        assert_eq!(reduced.len(), 3);
        assert_eq!(reduced[0].len(), 2);
    }
}