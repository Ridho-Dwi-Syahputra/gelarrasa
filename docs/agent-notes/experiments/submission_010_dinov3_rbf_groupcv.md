# Submission 010: DINOv3 RBF SVM

- Model: multiclass RBF SVC over the notebook's DINOv3 image descriptors.
- Fit data: 1,879 eligible full-frame train images. Test images are used for inference only.
- Selection: 3-fold `StratifiedGroupKFold`, grouped by source; C grid `[0.5, 1, 2, 4, 8, 16]`. Selected C = 2.
- Selected grouped CV: accuracy 0.95370, balanced accuracy 0.95417, macro-F1 0.95418. These scores were used to choose among the grid and are not an independent final estimate.
- Candidate: `final-submit/output/submission_010_dinov3_rbf_groupcv.csv` (480 unique IDs, SHA-256 `9c6326b815f23fa2bd0495bdca07ceeaa22334011232d25f362e5f678a5d6bdf`).
- Comparison: 9 labels differ from saved 009; 3 differ from 008. Different from 009 at IDs 0180, 0181, 0224, 0227, 0231, 0234, 0309, 0395, 0470.
- The candidate has not been submitted to Kaggle; its leaderboard score is unknown.
