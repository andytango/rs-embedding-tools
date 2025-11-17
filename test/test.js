const { processEmbeddings, reduceDimensions, clusterData } = require('..');

console.log('Testing embedding-tools Node.js bindings...\n');

// Test data: 3 clusters of points
const testData = [
  // Cluster 1
  [1.0, 2.0, 3.0],
  [1.1, 2.1, 3.1],
  [1.2, 2.2, 3.2],
  [0.9, 1.9, 2.9],
  [1.0, 2.0, 3.0],
  // Cluster 2
  [10.0, 11.0, 12.0],
  [10.1, 11.1, 12.1],
  [10.2, 11.2, 12.2],
  [9.9, 10.9, 11.9],
  [10.0, 11.0, 12.0],
  // Cluster 3
  [100.0, 101.0, 102.0],
  [100.1, 101.1, 102.1],
  [100.2, 101.2, 102.2],
  [99.9, 100.9, 101.9],
  [100.0, 101.0, 102.0],
];

try {
  // Test 1: Full pipeline
  console.log('Test 1: processEmbeddings (full pipeline)');
  const config = {
    n_components: 3,
    min_cluster_size: 3,
    n_iterations: 100, // Fewer iterations for faster testing
  };

  const result = processEmbeddings(testData, config);

  console.log('  ✓ Processed', result.metadata.n_samples, 'samples');
  console.log('  ✓ Reduced from', result.metadata.original_dimensions, 'to 3 dimensions');
  console.log('  ✓ Found', result.metadata.n_clusters, 'clusters');
  console.log('  ✓ Noise points:', result.metadata.n_noise);
  console.log('  ✓ Sample reduced data:', result.reduced_data[0]);
  console.log('  ✓ Sample clusters:', result.clusters.slice(0, 5));
  console.log('');

  // Test 2: Dimensionality reduction only
  console.log('Test 2: reduceDimensions (reduction only)');
  const reduced = reduceDimensions(testData, 2);

  console.log('  ✓ Reduced to', reduced[0].length, 'dimensions');
  console.log('  ✓ Number of samples:', reduced.length);
  console.log('  ✓ Sample reduced point:', reduced[0]);
  console.log('');

  // Test 3: Clustering on pre-reduced data
  console.log('Test 3: clusterData (clustering only)');
  const clusters = clusterData(reduced, 3);

  console.log('  ✓ Cluster assignments:', clusters);
  const uniqueClusters = new Set(clusters.filter(c => c !== null));
  console.log('  ✓ Number of unique clusters:', uniqueClusters.size);
  console.log('');

  console.log('✅ All tests passed!');
  process.exit(0);
} catch (error) {
  console.error('❌ Test failed:', error.message);
  console.error(error.stack);
  process.exit(1);
}
