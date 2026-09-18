"""
main.py
-------
Single entry point that reproduces the entire analysis:
1. Loads data
2. Trains & evaluates supervised models (ML-2 + ML-4)
3. Runs unsupervised clustering analysis (ML-3)
4. Writes comparison table + figures to results/

Run with:  python src/main.py
"""

from evaluate import evaluate_all_models
from unsupervised_analysis import run_clustering


def main():
    print("=" * 60)
    print("1. Training & evaluating supervised models...")
    print("=" * 60)
    results_df, *_ = evaluate_all_models()
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("2. Running unsupervised clustering analysis...")
    print("=" * 60)
    ari, sil, crosstab, explained_var = run_clustering()
    print(f"Adjusted Rand Index vs true diagnosis: {ari:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print("\nCluster vs Actual Diagnosis:")
    print(crosstab)

    print("\nAll results, tables, and figures saved to ../results/")


if __name__ == "__main__":
    main()
