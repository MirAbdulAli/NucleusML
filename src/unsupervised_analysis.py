"""
unsupervised_analysis.py
--------------------------
Applies unsupervised learning (ML-3) to the same dataset, WITHOUT using
the diagnosis label, then checks afterwards whether the discovered
structure relates to the true (supervised) target.

Technique: K-Means clustering (k=2) on standardized features, plus PCA
(2 components) purely for visualization of the cluster structure.

Why this is meaningful (not just a checkbox):
- We ask "if we didn't have diagnosis labels at all, would an unsupervised
  method rediscover roughly the same two groups a doctor's biopsy found?"
- This tests whether malignant/benign tumors are naturally separable in
  feature space, which is a genuinely different question from "can a
  classifier map features -> label" (ML-2). It's evaluated with different
  tools too (ARI/silhouette instead of accuracy/F1), which is why it
  belongs to ML-3, not ML-2.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix

from data_prep import load_data, RANDOM_STATE
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def run_clustering():
    os.makedirs(FIG_DIR, exist_ok=True)

    X, y, target_names = load_data()

    # Scale full feature set (KMeans is distance-based, so scaling matters a lot)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means clustering (unsupervised: never sees y) ---
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # How well do clusters align with the TRUE diagnosis label?
    # (We only use y here to *evaluate* the clustering after the fact,
    # exactly as the task instructions suggest: "checking if they relate
    # to your supervised target.")
    ari = adjusted_rand_score(y, cluster_labels)
    sil = silhouette_score(X_scaled, cluster_labels)

    # Cross-tab: cluster assignment vs actual diagnosis
    crosstab = pd.crosstab(
        pd.Series(cluster_labels, name="KMeans Cluster"),
        pd.Series(y.values, name="Actual Diagnosis").map(
            {0: target_names[0], 1: target_names[1]}
        ),
    )

    # --- PCA for 2D visualization ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    scatter0 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="coolwarm", alpha=0.7, s=20
    )
    axes[0].set_title("PCA Projection colored by K-Means Cluster")
    axes[0].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")

    scatter1 = axes[1].scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7, s=20
    )
    axes[1].set_title("PCA Projection colored by TRUE Diagnosis")
    axes[1].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pca_cluster_vs_true_label.png"), dpi=150)
    plt.close(fig)

    # Save a small text summary
    summary_path = os.path.join(RESULTS_DIR, "unsupervised_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Unsupervised Analysis Summary (K-Means, k=2)\n\n")
        f.write(f"- Adjusted Rand Index (cluster vs true label): **{ari:.4f}**\n")
        f.write(f"- Silhouette score (cluster cohesion/separation): **{sil:.4f}**\n")
        f.write(f"- PCA: PC1 explains {explained_var[0]*100:.1f}% of variance, "
                 f"PC2 explains {explained_var[1]*100:.1f}%, "
                 f"total {sum(explained_var)*100:.1f}%\n\n")
        f.write("## Cluster vs Actual Diagnosis (cross-tab)\n\n")
        f.write(crosstab.to_markdown())
        f.write("\n")

    return ari, sil, crosstab, explained_var


if __name__ == "__main__":
    ari, sil, crosstab, explained_var = run_clustering()
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print(crosstab)
