from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "agent-notes"
K_VALUES = (5, 10, 20, 40)
ALPHAS = (0.3, 0.6, 0.9)
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
z = np.load(DATA / "dinov2_small_embeddings.npz")
cache = {k: i for i, k in enumerate(z["keys"].astype(str))}
manifest = manifest[manifest.parent_key.isin(cache)].sort_values("parent_key").reset_index(drop=True)
X = normalize(z["features"][[cache[k] for k in manifest.parent_key]], norm="l2").astype("float32")
keys = manifest.parent_key.to_numpy(str)
groups = manifest.parent_group.fillna(manifest.parent_key).to_numpy(str)
id_to_row = {k: i for i, k in enumerate(keys)}
seeds = pd.read_csv(DATA / "manual_seeds.csv")
seed_rows = seeds.key.map(id_to_row).to_numpy(int)
seed_y = seeds.assigned_class.to_numpy(str)
seed_groups = groups[seed_rows]
assert len(seed_rows) == 35 and seeds.groupby("assigned_class").size().max() <= 5
classes = np.array(sorted(seeds.assigned_class.unique()))
class_to_col = {c: i for i, c in enumerate(classes)}
folds = list(GroupKFold(n_splits=5).split(seed_rows, seed_y, seed_groups))

def knn_graph(x, k):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="cosine", n_jobs=2).fit(x)
    dist, idx = nn.kneighbors(x)
    rows, cols, vals = [], [], []
    for i in range(len(x)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            rows.append(i); cols.append(int(j)); vals.append(float(np.exp(-float(d) / 0.12)))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(x), len(x)), dtype=np.float32)
    A = A.maximum(A.T)
    return normalize(A, norm="l1", axis=1)

def propagate(A, y, alpha):
    F = y.copy()
    for _ in range(100):
        nxt = alpha * (A @ F) + (1 - alpha) * y
        if np.max(np.abs(nxt - F)) < 1e-6:
            F = nxt
            break
        F = nxt
    return F

results = []
for k in K_VALUES:
    for alpha in ALPHAS:
        pred = np.full(len(seed_rows), "", dtype=object)
        conf = np.zeros(len(seed_rows), dtype=np.float32)
        for fit_seed_idx, val_seed_idx in folds:
            held = set(seed_groups[val_seed_idx])
            fit_row_mask = np.array([g not in held for g in groups])
            fit_rows = np.flatnonzero(fit_row_mask)
            fit_seed_local = np.array([np.where(fit_rows == seed_rows[j])[0][0] for j in fit_seed_idx])
            Y = np.zeros((len(fit_rows), len(classes)), dtype=np.float32)
            for local, sj in zip(fit_seed_local, fit_seed_idx):
                Y[local, class_to_col[seed_y[sj]]] = 1
            A = knn_graph(X[fit_rows], k)
            F = propagate(A, Y, alpha)
            qnn = NearestNeighbors(n_neighbors=min(k, len(fit_rows)), metric="cosine", n_jobs=2).fit(X[fit_rows])
            qdist, qidx = qnn.kneighbors(X[seed_rows[val_seed_idx]])
            for q, sj in enumerate(val_seed_idx):
                w = np.exp(-qdist[q] / 0.12).astype(np.float32)
                scores = (F[qidx[q]] * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)
                j = int(scores.argmax())
                pred[sj] = classes[j]
                conf[sj] = scores[j]
        results.append({
            "k": k, "alpha": alpha,
            "accuracy_all": float(np.mean(pred == seed_y)),
            "macro_recall_all": float(np.mean([np.mean(pred[seed_y == c] == c) for c in classes])),
            "mean_top_score": float(np.mean(conf)),
            "accuracy_at_score_0_60": float(np.mean(pred[conf >= .60] == seed_y[conf >= .60])) if np.any(conf >= .60) else None,
            "coverage_at_score_0_60": int(np.sum(conf >= .60)),
            "accuracy_at_score_0_75": float(np.mean(pred[conf >= .75] == seed_y[conf >= .75])) if np.any(conf >= .75) else None,
            "coverage_at_score_0_75": int(np.sum(conf >= .75)),
        })

payload = {
    "status": "exploratory_train_only_group_cv",
    "method": "DINOv2 cosine kNN graph harmonic label propagation; fold-held source groups absent from graph; queries only infer from fold-fit graph",
    "rows": int(len(X)), "human_seed_rows": int(len(seed_rows)),
    "manual_seed_class_counts": seeds.groupby("assigned_class").size().to_dict(),
    "test_rows_in_fit": 0, "review_or_archive_labels_read_by_fit": False,
    "results": results,
    "manifest_sha256": sha256(DATA / "manifest.csv"),
    "embedding_cache_sha256": sha256(DATA / "dinov2_small_embeddings.npz"),
    "seed_file_sha256": sha256(DATA / "manual_seeds.csv"),
}
(OUT / "json" / "variable_cluster_graph_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
pd.DataFrame(results).to_csv(OUT / "table-csv" / "variable_cluster_graph_probe.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
