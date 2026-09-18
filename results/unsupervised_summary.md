# Unsupervised Analysis Summary (K-Means, k=2)

- Adjusted Rand Index (cluster vs true label): **0.6536**
- Silhouette score (cluster cohesion/separation): **0.3434**
- PCA: PC1 explains 44.3% of variance, PC2 explains 19.0%, total 63.2%

## Cluster vs Actual Diagnosis (cross-tab)

|   KMeans Cluster |   benign |   malignant |
|-----------------:|---------:|------------:|
|                0 |      339 |          36 |
|                1 |       18 |         176 |
