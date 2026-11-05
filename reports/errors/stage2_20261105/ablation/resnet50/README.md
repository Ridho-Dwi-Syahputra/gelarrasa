# Representation ablation (unlabeled diagnostics)

PCA and KMeans fit only unique train pixel hashes; test is prediction-only. No annotation archive or semantic category mapping is read. No model is selected as semantically best.

Four branches: normalized original features, PCA64, PCA128, PCA256. All use KMeans8, seeds 11/29/47, n_init=10. PCA random seed is 11. Primary per-row exports and persisted models use seed11.

Silhouette uses one fixed 1000-row train sample, in both each branch space and the common original feature space. Own-space scores across representations are not directly comparable. Top-10 neighbors exclude all exact pixel duplicates; no-PCA retention=1 by definition. Neighbor cohesion is internal agreement, not accuracy.

One 80/20 candidate-group train split refits PCA and KMeans; agreement against the full-train partition measures deletion sensitivity only. It is not independent semantic validation, because the reference full model saw the held-out rows. Test labels never enter fitting.

Each joblib contains reducer (or None), seed11 KMeans, normalization flags, and input feature digest. Normalize raw features first, then apply reducer and normalize again when reducer is present.
