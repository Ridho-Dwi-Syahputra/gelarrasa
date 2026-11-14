# Submission 010: DINOv3 OVR RBF SVM (C=1)

- Notebook: `final-submit/DSC2026_Notebook.ipynb`, final model cell.
- Model fit: One-vs-Rest RBF SVM on 1,879 eligible full-frame train descriptors (DINOv3, 3,072 dimensions); test only receives inference.
- Validation: 3-fold `StratifiedGroupKFold` by source. Grouped OOF accuracy 0.94838, balanced accuracy 0.94906, macro-F1 0.94896 against the pipeline's train targets. This is a model-selection diagnostic, not official accuracy.
- Output: `final-submit/output/submission_010_dinov3_ovr_svm_c1.csv`, 480 unique IDs in sample order; SHA-256 `21525dbdf50d5820e99972518333a692c3c7cbde18b5581b6d7c6c674be6d034`.
- Difference counts: 8 labels vs `submission_009.csv`; 15 vs the fixed comparison file `docs/submission_ (1).csv`.
- Kaggle score has not been measured. The OVR C=2 trial had a higher grouped OOF score but was one row farther from both comparison CSVs; C=1 is kept as the closer model candidate for the current request.
