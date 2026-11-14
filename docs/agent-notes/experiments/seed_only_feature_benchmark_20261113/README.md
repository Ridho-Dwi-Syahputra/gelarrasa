# Train-only seed benchmark — 2026-11-13

The experiment reads only the 37 participant-confirmed seed rows and matching
raw images under the competition `train` directory. It extracts frozen
ResNet50/EfficientNetV2-S features plus color and texture descriptors, then
compares small classifiers in three source-group folds. It does not read test
images, processed assignments, archive annotations, or disagreement tables.

## Result

Best observed candidate among the tested feature/model combinations was
ResNet50 plus color histograms with ExtraTrees: 14/37 seed predictions correct,
balanced accuracy 0.390, macro-F1 0.390. The candidate was selected from many
comparisons on the same small seed set, so this score is exploratory and likely
optimistic. It is not enough evidence to propagate labels across the remaining
train images.

No processed labels, notebook, or training targets were changed. Detailed metrics
and fold predictions are in `metrics.csv` and `oof_seed_predictions.csv`;
`report.json` records input scope and source hashes.
