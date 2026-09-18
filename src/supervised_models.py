"""
supervised_models.py
---------------------
Trains multiple, genuinely different supervised classification algorithms
(ML-2) on the same target variable (malignant vs benign).

Algorithms used (deliberately different families, not the same algorithm
with different hyperparameters):
1. Logistic Regression  -> linear / parametric model
2. Random Forest        -> tree-based ensemble, non-linear, bagging
3. K-Nearest Neighbors   -> instance-based, non-parametric

All models are trained on the SAME train/test split so evaluation is
apples-to-apples.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_prep import get_train_test_split, get_scaled_data, RANDOM_STATE


def get_models():
    """Return a dict of {name: sklearn estimator} for the three models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def train_all_models():
    """
    Trains all models on the same split.
    Logistic Regression and KNN use SCALED features (they are
    distance/gradient sensitive). Random Forest uses the RAW (unscaled)
    features since tree splits are scale-invariant -- this is intentional
    and demonstrates understanding of when scaling matters (ML-1/ML-2).

    Returns:
        fitted_models: dict of {name: fitted estimator}
        splits: dict with X_train, X_test, y_train, y_test (raw, for RF)
        splits_scaled: dict with X_train_scaled, X_test_scaled (for LR/KNN)
        target_names: list of class names
    """
    X_train, X_test, y_train, y_test, target_names = get_train_test_split()
    X_train_scaled, X_test_scaled, scaler = get_scaled_data(X_train, X_test)

    models = get_models()
    fitted_models = {}

    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train_scaled, y_train)
        fitted_models[name] = model

    splits = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    splits_scaled = {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled}

    return fitted_models, splits, splits_scaled, target_names, scaler


if __name__ == "__main__":
    fitted_models, splits, splits_scaled, target_names, scaler = train_all_models()
    print("Trained models:", list(fitted_models.keys()))
