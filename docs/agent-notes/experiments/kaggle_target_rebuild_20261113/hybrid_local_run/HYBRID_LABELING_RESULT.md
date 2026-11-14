# Hybrid CNN + HOG + color train-label experiment ? 13 Nov 2026

## Method

Raw train images only: 1,920 images. Frozen EfficientNet-B0 ImageNet embedding (1,280 features), HOG (1,764), and RGB histogram (768) were evaluated as separate blocks and fused with RBF-SVM. Per-block StandardScaler and L2 normalization were fit inside grouped folds; block weights and SVM parameters were selected in inner folds. The 37 participant-reviewed train seeds were the only target labels. Exact-image and pHash source groups were kept together. No test pixels or labels entered this experiment.

## Grouped OOF result

| Feature set | OOF accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| hybrid_cnn_hog_color | 0.081 | 0.092 | 0.119 |
| cnn_only | 0.108 | 0.100 | 0.104 |
| hog_only | 0.054 | 0.050 | 0.045 |
| hog_color | 0.054 | 0.050 | 0.031 |
| color_only | 0.027 | 0.025 | 0.018 |

Hybrid CNN+HOG+color reached 3/37 OOF seed matches (8.1%) and macro-F1 0.119. CNN-only reached 4/37 (10.8%) and macro-F1 0.104. HOG-only, color-only, and HOG+color were weaker. This is too weak to accept the hybrid SVM as a train-labeling rule. The full-train candidate distribution was highly skewed (metal 3, foodwaste 67, other classes 281?386), another warning against using these pseudo-labels.

## Decision and artifacts

The model wrote a separate 1,920-row candidate CSV for audit only; it did not update processed labels or active assignments. The experiment is train-only and reproducible from the raw dataset plus pinned EfficientNet weights. Candidate predictions are not semantic truth, and OOF seed agreement is not competition accuracy. The active parity gate remains closed.

- `labels/hybrid_train_label_candidates.csv` ? per-ID candidates, margins, seed flags, provenance.
- `validation/hybrid_label_seed_oof*.csv` ? grouped OOF predictions, summary, class metrics, and confusion counts.
- `validation/hybrid_label_inner_selection.csv` ? fold-local model selection.
- `features/hybrid_labeling/train_cnn_hog_color.npz` ? raw-train feature cache with source hash and cache identity.
- `logs/hybrid_label_feature_provenance.json` ? settings, feature sizes, runtime, and boundaries.

Runtime on local CPU after feature-cache creation: 169.0 seconds (nested grouped OOF 158.3 s, refit/inference 7.5 s). The EfficientNet feature extraction was performed locally on CPU to generate the cache; Kaggle GPU should reduce feature extraction time, but SVM grid search is CPU-based.
