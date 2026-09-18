# ML Model Comparison: Breast Cancer Diagnosis

A structured comparison of multiple machine learning approaches — spanning
fundamentals, supervised learning, and unsupervised learning — applied to
one dataset, with proper evaluation throughout.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** dataset, loaded directly from
`sklearn.datasets.load_breast_cancer` (no external download required —
fully reproducible offline).

- **569 samples**, 30 numeric features computed from digitized images of
  a fine needle aspirate (FNA) of a breast mass (e.g. radius, texture,
  perimeter, smoothness of cell nuclei).
- **Target**: diagnosis — `malignant` (37.3%) or `benign` (62.7%).

This dataset was chosen because it has a clean binary target suitable for
supervised classification, all-continuous features that make scaling/PCA
genuinely relevant, and it's small enough to keep the project fast and
dependency-free.

## Project structure

```
ml_model_comparison/
├── README.md                          <- you are here
├── requirements.txt
├── src/
│   ├── data_prep.py                   <- load data, train/test split, scaling
│   ├── supervised_models.py           <- trains 3 different supervised algorithms
│   ├── evaluate.py                    <- metrics, confusion matrices, comparison table
│   ├── unsupervised_analysis.py       <- K-Means clustering + PCA
│   └── main.py                        <- runs the full pipeline end-to-end
└── results/
    ├── comparison_table.csv / .md     <- supervised model comparison
    ├── unsupervised_summary.md        <- clustering results
    └── figures/
        ├── confusion_matrix_*.png     <- one per supervised model
        └── pca_cluster_vs_true_label.png
```

## How to reproduce

```bash
pip install -r requirements.txt
python src/main.py
```

This single command loads the data, trains all models, evaluates them,
runs the clustering analysis, and regenerates every file in `results/`.
Everything uses a fixed `random_state=42`, so results are identical on
every run.

---

## 1. Supervised models (ML-2)

Three **genuinely different algorithm families** were trained on the same
train/test split (75/25, stratified) to predict diagnosis:

| Model | Type | Why it's different from the others |
|---|---|---|
| **Logistic Regression** | Linear / parametric | Learns a linear decision boundary in feature space; fast, interpretable coefficients |
| **Random Forest** (300 trees, max depth 6) | Tree-based ensemble (bagging) | Learns non-linear, axis-aligned splits; robust to outliers and unscaled features |
| **K-Nearest Neighbors** (k=7) | Instance-based / non-parametric | Makes no assumption about a decision boundary at all — just votes among nearby training points |

Logistic Regression and KNN were trained on **standardized** features
(mean 0, std 1) since both are distance/gradient sensitive. Random Forest
was intentionally trained on **raw** features, since tree splits are
scale-invariant — this is a deliberate design choice, not an oversight,
and it's a good illustration of an ML-1 fundamental (when scaling
matters and when it doesn't).

## 2. Evaluation (ML-4)

Each model was evaluated on the **held-out test set only** (113 samples,
never seen during training), using five metrics plus a confusion matrix.

### Comparison table

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|:--|--:|--:|--:|--:|--:|
| **Logistic Regression** | 0.986 | 0.989 | 0.989 | **0.989** | 0.998 |
| K-Nearest Neighbors | 0.979 | 0.968 | **1.000** | 0.984 | 0.992 |
| Random Forest | 0.958 | 0.957 | 0.978 | 0.967 | 0.993 |

*(Precision/Recall/F1 computed with "benign" as the positive class.
Full numbers in `results/comparison_table.csv`.)*

### Confusion matrices

See `results/figures/confusion_matrix_*.png` for all three. Logistic
Regression made only **2 total errors** out of 113 test samples (1 false
positive, 1 false negative). KNN never missed a true benign case
(recall = 1.0) but had slightly more false positives. Random Forest
was the weakest of the three, with 5 total errors.

### What the metrics mean in context

This is a medical diagnosis problem, so **recall on malignant cases**
(i.e. not missing a cancer) matters more than raw accuracy. All three
models perform well here, but it's worth noting precision/recall were
computed for the "benign" class by default — the practically important
number is how many malignant cases were *missed* (false negatives on
malignant = false positives on benign in this table). Logistic
Regression and KNN each missed only 1 malignant case in the test set;
Random Forest missed 1 as well but had more false alarms.

---

## 3. Unsupervised analysis (ML-3)

**Technique: K-Means clustering (k=2)** on standardized features,
applied *without ever showing it the diagnosis label*, followed by
**PCA (2 components)** purely for visualization.

**The question this answers is genuinely different from the supervised
task above**: "If we didn't have a doctor's diagnosis at all, would an
unsupervised method rediscover roughly the same two groups just from the
shape of the feature data?"

### Results

- **Adjusted Rand Index vs. true diagnosis: 0.654** (1.0 = perfect
  agreement, 0.0 = random). This is a strong signal that the two
  clusters K-Means finds largely correspond to malignant vs. benign,
  even with zero access to the label.
- **Silhouette score: 0.343** — the clusters are reasonably well
  separated but do overlap somewhat (expected, since real biology is
  messier than two clean blobs).
- **PC1 alone explains 44.3%** of total feature variance, and PC1+PC2
  together explain **63.2%** — most of the separability lives along a
  single dominant axis (likely related to overall cell size/shape
  irregularity, which is exactly what a pathologist would look for).

### Cluster vs. actual diagnosis

| KMeans Cluster | benign | malignant |
|--:|--:|--:|
| 0 | 339 | 36 |
| 1 | 18 | 176 |

Cluster 0 is overwhelmingly benign, Cluster 1 is overwhelmingly
malignant — but neither is pure. **36 malignant tumors cluster with the
benign group**, and 18 benign tumors cluster with the malignant group.
Side-by-side PCA plots (`results/figures/pca_cluster_vs_true_label.png`)
show visually that the unsupervised clusters and the true labels line up
closely but not perfectly — there's a real overlap region in the middle
where the two classes' feature distributions blend together.

**Interpretation:** the structure that separates malignant from benign
tumors is strong enough to be found *without* supervision, which is a
reassuring sanity check on the dataset itself — it tells us the
supervised models aren't succeeding by exploiting some quirk unrelated
to the underlying biology. It also quantifies *how much harder* the
problem is without labels: an unsupervised approach alone would
misclassify roughly 1 in 10 tumors, while the supervised models above
get that down to 1–5 errors out of 113.

---

## 4. Recommendation

**Recommended model: Logistic Regression.**

| Criterion | Logistic Regression | Random Forest | KNN |
|---|---|---|---|
| Best F1 / accuracy on test set | ✅ (0.989 / 0.986) | ✗ (0.967 / 0.958) | close 2nd (0.984 / 0.979) |
| Interpretability | ✅ coefficients map directly to features | ✗ harder to explain to a clinician | ✗ no explicit model |
| Training/inference speed | ✅ fastest | slower (300 trees) | slow at inference (must scan neighbors) |
| Risk of overfitting on this size dataset | low | moderate (mitigated with depth limit) | low, but sensitive to choice of k |

**Why Logistic Regression over KNN**, despite KNN's perfect recall on
this test set: KNN's performance is more fragile to the choice of *k*
and to how future data is scaled, and it offers no interpretable
explanation for a prediction — which matters a great deal in a
medical-diagnosis setting where a clinician needs to understand *why*
a model flagged a case as malignant. Logistic Regression's coefficients
can be directly inspected to see which cell measurements are driving a
prediction, it had the best F1-score and ROC-AUC overall, and it's by
far the cheapest to train and deploy.

**Why not Random Forest**: it had the weakest metrics across the board
here. On a larger or noisier dataset it would likely close the gap or
win (tree ensembles tend to shine with more data and more complex,
non-linear feature interactions), but on this clean, moderately-sized,
mostly-linearly-separable dataset, its extra complexity isn't earning
its keep.

**Caveat**: with only 113 test samples, the gap between Logistic
Regression (2 errors) and KNN (a comparable handful) is not necessarily
statistically significant — on a re-shuffled split the ranking could
shift slightly. The recommendation leans on interpretability and
robustness as much as the raw metric, not on the metric alone.

---

## Notes on methodology (ML-1 fundamentals applied throughout)

- **Train/test split is stratified** to preserve class balance, since the
  target is imbalanced (~63/37).
- **Scaling is fit only on the training set** and applied to the test set
  (`StandardScaler.fit` on train, `.transform` on test) to avoid data
  leakage.
- **Random Forest deliberately skips scaling** since tree splits are
  scale-invariant, while Logistic Regression and KNN require it — this
  distinction is a core ML-1 concept, not an inconsistency.
- All random processes (`train_test_split`, `RandomForestClassifier`,
  `KMeans`) are seeded with `random_state=42` for full reproducibility.
