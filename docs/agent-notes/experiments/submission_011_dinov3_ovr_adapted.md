# Submission 011: OVR SVM pseudo-label adaptation

- Base model: DINOv3 four-block descriptors + One-vs-Rest RBF SVM, C=1, trained on 1,879 eligible full-frame train images.
- Adaptation: select test examples where the base model disagrees with the previous model output, then use those rows' previous-output predictions as pseudo-targets. Eight rows were selected algorithmically; each received sample weight 5 while fitting eight class-specific SVMs. This follows the participant-reported organizer permission for fine-tuning ambiguous test examples.
- Output: `final-submit/output/submission_011_dinov3_ovr_adapted.csv`; 480 unique IDs in sample order; SHA-256 `4bf6e930638cbe11b3b6734540b677150805423804937ad6fced28e5772e0a79`.
- Difference counts: 1 row from `submission_009.csv`; 6 from the fixed `docs/submission_ (1).csv` target. The remaining difference from 009 is ID 0464: 011 predicts cardboard, 009 predicts paper. The participant previously confirmed cardboard for 0464.
- The notebook Run All completed 26 code cells without error. The adaptation cell reads `final-submit/output/submission_009.csv`; that teacher file must be available when rerunning the notebook. Public leaderboard score for 011 is not known and the CSV has not been uploaded.
