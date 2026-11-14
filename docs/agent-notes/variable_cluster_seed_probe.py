from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "agent-notes"
DATA = ROOT / "data" / "processed"
SEED = 20261109
K_VALUES = (32, 64, 128, 256)
REPEATS = 2


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
cache_keys = z["keys"].astype(str)
cache = {key: i for i, key in enumerate(cache_keys)}
manifest = manifest[manifest.parent_key.isin(cache)].sort_values("parent_key").reset_index(drop=True)
X = normalize(z["features"][[cache[k] for k in manifest.parent_key]], norm="l2").astype("float32")
keys = manifest.parent_key.to_numpy(str)
groups = manifest.parent_group.fillna(manifest.parent_key).to_numpy(str)
id_to_row = {key: i for i, key in enumerate(keys)}
seeds = pd.read_csv(DATA / "manual_seeds.csv")
seeds["row"] = seeds.key.map(id_to_row)
assert seeds.row.notna().all(), "manual seed missing from eligible full frames"
seed_rows = seeds.row.to_numpy(int)
seed_y = seeds.assigned_class.to_numpy(str)
seed_groups = groups[seed_rows]
assert len(seeds) == 35
assert seeds.groupby("assigned_class").size().max() <= 5
folds = list(GroupKFold(n_splits=5).split(seed_rows, seed_y, seed_groups))

results = []
with threadpool_limits(limits=2):
 for k in K_VALUES:
    all_repeats = []
    for repeat in range(REPEATS):
        pred = np.full(len(seed_rows), "", dtype=object)
        for fold, (fit_seed_idx, val_seed_idx) in enumerate(folds):
            val_groups = set(seed_groups[val_seed_idx])
            # The entire held-out source groups are excluded from cluster fitting.
            fit_rows = np.array([i for i, g in enumerate(groups) if g not in val_groups], dtype=int)
            fit_seed_rows = seed_rows[fit_seed_idx]
            km = MiniBatchKMeans(
                n_clusters=min(k, max(2, len(fit_rows) // 3)),
                random_state=SEED + repeat * 100 + fold,
                n_init=2,
                batch_size=1024,
                max_iter=100,
                reassignment_ratio=0.01,
            ).fit(X[fit_rows])
            seed_cluster = km.predict(X[fit_seed_rows])
            val_cluster = km.predict(X[seed_rows[val_seed_idx]])
            classes = np.unique(seed_y[fit_seed_idx])
            for j, c in zip(val_seed_idx, val_cluster):
                members = seed_y[fit_seed_idx][seed_cluster == c]
                if len(members):
                    vals, counts = np.unique(members, return_counts=True)
                    winner_i = int(np.argmax(counts))
                    if counts[winner_i] / len(members) >= 0.80:
                        pred[j] = vals[winner_i]
        covered = pred != ""
        acc = float(np.mean(pred[covered] == seed_y[covered])) if covered.any() else None
        all_repeats.append({
            "repeat": repeat,
            "coverage": int(covered.sum()),
            "coverage_rate": float(covered.mean()),
            "accuracy_covered": acc,
            "correct_covered": int(np.sum((pred == seed_y) & covered)),
        })
    results.append({"k": k, "repeats": all_repeats})

summary = []
for r in results:
    cov = np.mean([x["coverage_rate"] for x in r["repeats"]])
    accs = [x["accuracy_covered"] for x in r["repeats"] if x["accuracy_covered"] is not None]
    summary.append({"k": r["k"], "mean_coverage": float(cov), "mean_accuracy_when_covered": float(np.mean(accs)) if accs else None})

payload = {
    "status": "exploratory_train_only_seed_cv",
    "feature": "DINOv2-small frozen embeddings, L2 normalized",
    "rows": int(len(X)),
    "human_seed_rows": int(len(seed_rows)),
    "manual_seed_class_counts": seeds.groupby("assigned_class").size().to_dict(),
    "test_rows_in_fit": 0,
    "review_or_archive_labels_read_by_fit": False,
    "method": "MiniBatchKMeans variable K; cluster mapped by >=0.80 majority of training-fold participant seeds; GroupKFold seed evaluation excludes held-out source groups from unsupervised fit",
    "k_values": list(K_VALUES),
    "repeats": REPEATS,
    "fold_results": results,
    "summary": summary,
    "manifest_sha256": sha256(DATA / "manifest.csv"),
    "embedding_cache_sha256": sha256(DATA / "dinov2_small_embeddings.npz"),
    "seed_file_sha256": sha256(DATA / "manual_seeds.csv"),
}
(OUT / "json" / "variable_cluster_seed_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
pd.DataFrame(summary).to_csv(OUT / "table-csv" / "variable_cluster_seed_probe_summary.csv", index=False)
print(json.dumps(payload["summary"], indent=2))
