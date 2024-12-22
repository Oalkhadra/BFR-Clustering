# BFR Clustering Algorithm

## Overview
This project implements the Bradley-Fayyad-Reina (BFR) algorithm, a scalable variation of k-means designed for clustering large datasets. The implementation handles data in chunks, making it memory-efficient and suitable for large-scale applications.

## Algorithm Features
- Processes data in chunks (20% at a time)
- Maintains three sets of points:
  - Discard Set (DS): Points assigned to clusters
  - Compression Set (CS): Secondary clusters for potentially merging
  - Retained Set (RS): Outliers and points needing further processing
- Uses Mahalanobis distance for cluster assignment
- Implements cluster statistics maintenance (N, SUM, SUMSQ)
- Handles outlier detection and management

## Requirements
- Python 3.6+
- NumPy
- scikit-learn

## Installation
```bash
git clone https://github.com/yourusername/bfr-clustering.git
cd bfr-clustering
pip install -r requirements.txt
```

## Usage
The program can be run using the following command:
```bash
python task.py <input_file> <n_cluster> <output_file>
```

### Parameters
- `input_file`: Path to the input dataset
- `n_cluster`: Number of clusters to generate
- `output_file`: Path for the output file

### Input Format
The input file should be a CSV with the following format:
```
index,label,feature1,feature2,...,featureN
```
- First column: Point index
- Second column: Ground truth label (if available)
- Remaining columns: Feature values

### Output Format
The program generates a text file containing:
1. Intermediate results for each round showing:
   - Number of discard points
   - Number of compression set clusters
   - Number of compression points
   - Number of retained points
2. Final clustering results with point indices and assigned clusters

## Implementation Details
The implementation follows these key steps:

1. Initial Processing:
   - Loads first 20% of data randomly
   - Runs k-means with large K (5 * n_cluster)
   - Identifies and handles single-point clusters as potential outliers
   - Creates initial DS clusters

2. Subsequent Rounds:
   - Processes remaining data in 20% chunks
   - Assigns points to DS/CS/RS based on Mahalanobis distance
   - Merges CS clusters when possible
   - Handles outliers through RS management

3. Final Processing:
   - Merges qualifying CS clusters with DS clusters
   - Generates final cluster assignments

## Performance Notes
- Runtime target: < 600 seconds
- Memory efficient through chunk-based processing
- Scales well with large datasets due to statistical summarization

## Personal Notes
This implementation taught me valuable lessons about:
- Handling large-scale clustering problems
- Importance of statistical summarization in data mining
- Balancing accuracy with computational efficiency
- Real-world applications of clustering algorithms

## Future Improvements
- Implement parallel processing for larger datasets
- Add visualization capabilities
- Enhance outlier detection mechanisms
- Add support for different distance metrics
