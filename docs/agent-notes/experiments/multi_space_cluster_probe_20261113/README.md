# Multi-space clustering probe — 2026-11-13

This train-only experiment clustered the 1,920 raw training images using
ResNet18, EfficientNet-B0, color histograms, HOG/LBP texture, and fixed feature
combinations. It compared MiniBatchKMeans and Birch at K=32, 64, 128, 256, and
512. In each of five folds, all images in held-out seed source groups were
excluded from clustering; cluster names were assigned only from participant
seed labels in the fold's fit portion. No test images, active assignments,
archive labels, or disagreement lists entered fitting or model selection.

## Result

There are 210 reported combinations (seven feature spaces, two algorithms,
five K values, and three predeclared cluster-purity cutoffs). The highest
number of correct held-out seeds was only 8/37, with 12/37 seeds covered
(66.7% accuracy among covered seeds). This came from EfficientNet-B0 plus
color histograms, Birch, K=128. The widest-coverage candidate covered 22/37
but got only 4 correct. Therefore no cluster mapping is reliable enough to
publish as labels for the full train set.

No active processed labels, Kaggle notebook, or training targets were changed.
See `report.json`, `cluster_metrics.csv`, and `oof_seed_cluster_predictions.csv`.
