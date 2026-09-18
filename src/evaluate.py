"""
evaluate.py
-----------
Proper evaluation for every supervised model (ML-4):
- Uses the held-out test set (never seen during training)
- Computes multiple relevant metrics: accuracy, precision, recall, F1, ROC-AUC
- Produces a confusion matrix per model
- Saves a comparison table (CSV + Markdown) and confusion matrix plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from supervised_models import train_all_models

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def get_predictions(model, name, splits, splits_scaled):
    """Return (y_true, y_pred, y_proba) using the correct feature set per model."""
    y_test = splits["y_test"]
    if name == "Random Forest":
        X_test = splits["X_test"]
    else:
        X_test = splits_scaled["X_test_scaled"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "benign"=1
    return y_test, y_pred, y_proba


def evaluate_all_models():
    os.makedirs(FIG_DIR, exist_ok=True)

    fitted_models, splits, splits_scaled, target_names, scaler = train_all_models()

    rows = []
    for name, model in fitted_models.items():
        y_test, y_pred, y_proba = get_predictions(model, name, splits, splits_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        rows.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
            }
        )

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix: {name}")
        fig.tight_layout()
        fname = name.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
        plt.close(fig)

    results_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)

    # Save comparison table
    results_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "comparison_table.md"), "w") as f:
        f.write("# Supervised Model Comparison (Test Set)\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n")

    return results_df, fitted_models, splits, splits_scaled, target_names


if __name__ == "__main__":
    results_df, *_ = evaluate_all_models()
    print(results_df.to_string(index=False))
