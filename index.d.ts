/**
 * High-performance embedding processing tools with dimensionality reduction (PACMAP)
 * and clustering (HDBSCAN) for Node.js
 */

/**
 * Configuration options for the embedding processing pipeline
 */
export interface PipelineConfig {
  /**
   * Number of components for dimensionality reduction (default: 3)
   */
  n_components?: number;

  /**
   * Number of neighbors for PACMAP (default: auto-determined based on data size)
   */
  n_neighbors?: number;

  /**
   * Number of iterations for PACMAP optimization (default: 450)
   */
  n_iterations?: number;

  /**
   * Minimum cluster size for HDBSCAN (default: 5)
   */
  min_cluster_size?: number;

  /**
   * Minimum samples for HDBSCAN core distance (default: 5)
   */
  min_samples?: number;

  /**
   * Whether to skip clustering step (default: false)
   */
  skip_clustering?: boolean;
}

/**
 * Metadata about the processing pipeline execution
 */
export interface ProcessingMetadata {
  /**
   * Original number of dimensions in input data
   */
  original_dimensions: number;

  /**
   * Number of data points processed
   */
  n_samples: number;

  /**
   * Number of clusters found (null if clustering was skipped)
   */
  n_clusters: number | null;

  /**
   * Number of noise points identified (null if clustering was skipped)
   */
  n_noise: number | null;
}

/**
 * Output from the embedding processing pipeline
 */
export interface EmbeddingOutput {
  /**
   * Cluster assignments for each point (null for noise/outliers)
   * null if clustering was skipped
   */
  clusters: Array<number | null> | null;

  /**
   * Reduced dimensionality data (samples × n_components)
   */
  reduced_data: number[][];

  /**
   * Metadata about the processing
   */
  metadata: ProcessingMetadata;
}

/**
 * Process embeddings through the full pipeline
 *
 * This performs a two-stage process:
 * 1. Reduce high-dimensional embeddings to n_components (default 3D) using PACMAP
 * 2. Cluster the reduced data using HDBSCAN
 *
 * @param data - Input embeddings as a 2D array (samples × features)
 * @param config - Optional configuration for the pipeline
 * @returns Object containing cluster assignments, reduced data, and metadata
 *
 * @example
 * ```typescript
 * const { processEmbeddings } = require('embedding-tools');
 *
 * const data = [
 *   [1.0, 2.0, 3.0],
 *   [1.1, 2.1, 3.1],
 *   [10.0, 11.0, 12.0],
 * ];
 *
 * const config = {
 *   n_components: 3,
 *   min_cluster_size: 2,
 * };
 *
 * const result = processEmbeddings(data, config);
 * console.log(result.clusters);
 * console.log(result.reduced_data);
 * console.log(result.metadata);
 * ```
 */
export function processEmbeddings(
  data: number[][],
  config?: PipelineConfig,
): EmbeddingOutput;

/**
 * Reduce the dimensionality of embeddings without clustering
 *
 * Uses PACMAP for dimensionality reduction. Default is to reduce to 3D.
 *
 * @param data - Input embeddings as a 2D array (samples × features)
 * @param nComponents - Optional number of dimensions to reduce to (default: 3)
 * @returns Reduced embeddings as a 2D array (samples × nComponents)
 *
 * @example
 * ```typescript
 * const { reduceDimensions } = require('embedding-tools');
 *
 * const data = [
 *   [1.0, 2.0, 3.0, 4.0, 5.0],
 *   [2.0, 3.0, 4.0, 5.0, 6.0],
 * ];
 *
 * const reduced = reduceDimensions(data, 2);
 * console.log(reduced); // [[x1, y1], [x2, y2]]
 * ```
 */
export function reduceDimensions(
  data: number[][],
  nComponents?: number,
): number[][];

/**
 * Cluster pre-reduced data using HDBSCAN
 *
 * @param data - Pre-reduced embeddings as a 2D array (samples × features)
 * @param minClusterSize - Optional minimum cluster size (default: 5)
 * @returns Array of cluster assignments (null for noise/outliers)
 *
 * @example
 * ```typescript
 * const { clusterData } = require('embedding-tools');
 *
 * const reducedData = [
 *   [1.0, 2.0, 3.0],
 *   [1.1, 2.1, 3.1],
 *   [10.0, 11.0, 12.0],
 * ];
 *
 * const clusters = clusterData(reducedData, 2);
 * console.log(clusters); // [0, 0, null] (null = noise)
 * ```
 */
export function clusterData(
  data: number[][],
  minClusterSize?: number,
): Array<number | null>;
