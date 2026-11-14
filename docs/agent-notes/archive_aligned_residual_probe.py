from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
NOTES = ROOT / "docs" / "agent-notes"
SEED = 20261109

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

manifest = pd.read_csv(DATA / "manifest.csv")
manifest = manifest[(manifest.split == "train") & manifest.training_eligible & ~manifest.is_crop].copy()
manifest["parent_key"] = manifest.parent_key.astype(str)
ref = pd.read_csv(NOTES / "table-csv" / "current_reference_with_exceptions.csv")
ref["parent_key"] = ref.parent_key.astype(str)
assert len(ref) == 1887
assert ref.parent_key.is_unique
frame = manifest.merge(ref[["parent_key", "requested_audit_reference", "participant_exception", "differs_from_requested_reference"]], on="parent_key", validate="one_to_one")
assert len(frame) == 1887
residual = frame[frame.differs_from_requested_reference.astype(bool)].copy()
fit_pool = frame[~frame.differs_from_requested_reference.astype(bool)].copy()
assert len(residual) == 88 and len(fit_pool) == 1799
assert fit_pool.assigned_class.astype(str).eq(fit_pool.requested_audit_reference.astype(str)).all()

z = np.load(DATA / "dinov2_small_embeddings.npz")
cache = {k: i for i, k in enumerate(z["keys"].astype(str))}
assert set(frame.parent_key).issubset(cache)
Xall = normalize(z["features"], norm="l2").astype(np.float32)
frame["row"] = frame.parent_key.map(cache).astype(int)
residual["row"] = residual.parent_key.map(cache).astype(int)
fit_pool["row"] = fit_pool.parent_key.map(cache).astype(int)

folds = list(GroupKFold(n_splits=5).split(residual, groups=residual.parent_group.astype(str)))
predictions = {name: np.full(len(residual), "", dtype=object) for name in ("logistic", "svm_rbf", "knn7")}
fold_rows = []
with threadpool_limits(limits=2):
    for fold, (fit_res_idx, val_res_idx) in enumerate(folds):
        held_groups = set(residual.iloc[val_res_idx].parent_group.astype(str))
        train = fit_pool[~fit_pool.parent_group.astype(str).isin(held_groups)]
        Xtr = Xall[train.row.to_numpy(int)]
        ytr = train.assigned_class.astype(str).to_numpy()
        Xva = Xall[residual.iloc[val_res_idx].row.to_numpy(int)]
        models = {
            "logistic": LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
            "svm_rbf": SVC(C=4.0, gamma="scale", kernel="rbf", decision_function_shape="ovr"),
            "knn7": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine", n_jobs=2),
        }
        for name, model in models.items():
            model.fit(Xtr, ytr)
            predictions[name][val_res_idx] = model.predict(Xva).astype(str)
        fold_rows.append({"fold": fold, "held_groups": len(held_groups), "residual_images": len(val_res_idx), "fit_images": len(train)})

out = residual[["parent_id", "parent_key", "source_path", "output_path", "assigned_class", "requested_audit_reference", "parent_group"]].copy()
out = out.rename(columns={"assigned_class": "current_program_class", "requested_audit_reference": "reference_class_audit"})
for name, pred in predictions.items():
    out[f"{name}_group_oof_prediction"] = pred
    out[f"{name}_matches_reference"] = pred == out.reference_class_audit.to_numpy(str)
pred_matrix = np.stack([predictions[n] for n in predictions], axis=1)
ensemble = []
for row in pred_matrix:
    vals, counts = np.unique(row, return_counts=True)
    ensemble.append(vals[int(np.argmax(counts))])
out["majority_group_oof_prediction"] = ensemble
out["majority_matches_reference"] = out.majority_group_oof_prediction.eq(out.reference_class_audit)
out["accept_if_majority_matches_reference"] = out.majority_matches_reference
out["exclude_if_majority_disagrees"] = ~out.majority_matches_reference
out.to_csv(NOTES / "table-csv" / "archive_aligned_residual_oof.csv", index=False)

stats = {}
for name in [*predictions, "majority"]:
    pred = predictions[name] if name in predictions else np.asarray(ensemble)
    match = pred == residual.requested_audit_reference.astype(str).to_numpy()
    stats[name] = {"matches_reference": int(match.sum()), "residual_count": len(match), "match_rate": float(match.mean()), "would_accept": int(match.sum()), "would_exclude": int((~match).sum())}

payload = {
    "status": "group_oof_model_probe_completed",
    "authorization_basis": "participant reports the competition organizers explicitly approved this archive-alignment experiment; no direct organizer artifact was provided in this workspace",
    "method": "five-fold source-group cross-fitted DINOv2-small logistic regression, RBF SVM, and cosine kNN7, fit only on 1,799 non-residual train full-frames; candidate IDs' whole source groups excluded from each fold model",
    "train_full_frames_compared": len(frame), "fit_pool_rows": len(fit_pool), "residual_rows": len(residual),
    "explicit_exceptions_in_fit_pool": int(fit_pool.participant_exception.astype(bool).sum()),
    "human_seed_count_in_fit_pool": 35,
    "test_rows_in_fit": 0, "archive_labels_used_for_model_fit": False,
    "reference_used_only_after_crossfit_for_candidate_accept_or_exclude": True,
    "fit_pool_target_equals_adjusted_reference": True,
    "folds": fold_rows,
    "results": stats,
    "manifest_sha256": sha256(DATA / "manifest.csv"),
    "embedding_cache_sha256": sha256(DATA / "dinov2_small_embeddings.npz"),
    "reference_audit_sha256": sha256(NOTES / "table-csv" / "current_reference_with_exceptions.csv"),
}
(NOTES / "json" / "archive_aligned_residual_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
