"""
data_prep.py
------------
Loads the Breast Cancer Wisconsin (Diagnostic) dataset and prepares it
for both supervised and unsupervised experiments.

Dataset: sklearn.datasets.load_breast_cancer
- 569 samples, 30 numeric features computed from digitized images of a
  fine needle aspirate (FNA) of a breast mass.
- Target: diagnosis, binary -> 0 = malignant, 1 = benign.

This dataset was chosen because:
1. It has a clear, clinically meaningful binary target variable (good for
   ML-2 supervised classification).
2. All features are continuous and on different scales, which makes
   scaling + PCA + clustering (ML-3 unsupervised) genuinely interesting.
3. It's small and dependency-free (ships with scikit-learn), so the
   project is fully reproducible with no external downloads.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data():
    """Load the dataset as a pandas DataFrame + target Series."""
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()  # 0 = malignant, 1 = benign
    target_names = list(data.target_names)
    return X, y, target_names


def get_train_test_split(test_size=0.25, random_state=RANDOM_STATE):
    """
    Standard, reproducible train/test split (ML-4 requirement).
    Stratified on y since classes are somewhat imbalanced (~63% / 37%).
    """
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names


def get_scaled_data(X_train, X_test):
    """
    Standardize features (mean=0, std=1). Fit ONLY on training data to
    avoid leakage, then apply the same transform to test data.
    Needed for Logistic Regression, KNN, PCA, and KMeans, all of which
    are distance/gradient based and sensitive to feature scale.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    X, y, names = load_data()
    print(f"Shape: {X.shape}")
    print(f"Target classes: {names}")
    print(f"Class balance:\n{y.value_counts(normalize=True)}")
