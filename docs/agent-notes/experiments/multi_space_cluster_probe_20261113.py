"""Multi-feature, multi-granularity clustering probe; train images only."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from skimage.feature import hog, local_binary_pattern
from torchvision import models

ROOT = Path(__file__).resolve().parents[3]
RAW_TRAIN = ROOT / "data" / "Dataset DSC GelarRasa 2026" / "train"
SEEDS = ROOT / "data" / "processed" / "manual_seeds.csv"
OUT = Path(__file__).resolve().parent / "multi_space_cluster_probe_20261113"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261113
K_VALUES = (32, 64, 128, 256, 512)
THRESHOLDS = (0.50, 0.70, 0.90)
CLASSES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rgb(path: Path, size=None):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        return im.copy()


def p_hash(path: Path):
    im = load_rgb(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    low = dctn(np.asarray(im, dtype=np.float32), type=2, norm="ortho")[:8, :8]
    return low > np.median(low[1:, :])


def source_groups(paths):
    # Union only close visual duplicates so a held-out seed's near-duplicate
    # source cannot enter the unsupervised clustering fit.
    hashes = np.stack([p_hash(p).reshape(-1) for p in paths])
    parent = np.arange(len(paths))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(paths)):
        near = np.flatnonzero(np.count_nonzero(hashes[:i] != hashes[i], axis=1) <= 4)
        for j in near:
            a, b = find(i), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = [find(i) for i in range(len(paths))]
    return np.asarray([f"g{r:04d}" for r in roots])


def extract_cnn(paths):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [
        ("resnet18", models.ResNet18_Weights.IMAGENET1K_V1,
         lambda w: (models.resnet18(weights=w), lambda m: torch.nn.Sequential(*list(m.children())[:-1]), w.transforms())),
        ("efficientnet_b0", models.EfficientNet_B0_Weights.IMAGENET1K_V1,
         lambda w: (models.efficientnet_b0(weights=w), lambda m: torch.nn.Sequential(m.features, m.avgpool), w.transforms())),
    ]
    out = {}
    for name, weights, build in specs:
        base, head, tf = build(weights)
        model = head(base).eval().to(device)
        del base
        chunks, batch = [], []
        with torch.inference_mode():
            for i, path in enumerate(paths):
                batch.append(tf(load_rgb(path)))
                if len(batch) == 32 or i == len(paths) - 1:
                    x = torch.stack(batch).to(device)
                    chunks.append(model(x).flatten(1).float().cpu().numpy())
                    batch.clear()
                    print(f"features {name}: {i+1}/{len(paths)}", flush=True)
        out[name] = normalize(np.concatenate(chunks).astype(np.float32))
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    return out


def handcrafted(paths):
    color, textures = [], []
    for n, path in enumerate(paths, 1):
        rgb = np.asarray(load_rgb(path, 96), dtype=np.uint8)
        hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
        hist = np.concatenate([np.histogram(hsv[:, :, c], bins=b, range=(0, 256))[0] for c, b in ((0, 24), (1, 16), (2, 16))]).astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255
        h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
        lh = np.histogram(lbp, bins=np.arange(0, 11))[0].astype(np.float32)
        lh /= max(float(lh.sum()), 1.0)
        color.append(hist)
        textures.append(np.concatenate([h.astype(np.float32), 8 * lh]))
        if n % 400 == 0: print(f"handcrafted: {n}/{len(paths)}", flush=True)
    color = normalize(np.asarray(color, dtype=np.float32))
    texture = normalize(np.asarray(textures, dtype=np.float32))
    return {"color_hist": color, "texture_hog_lbp": texture,
            "color_texture": normalize(np.c_[color, texture]),
            "resnet18_color": normalize(np.c_[FEATURES["resnet18"], .25 * color]),
            "efficientnet_b0_color": normalize(np.c_[FEATURES["efficientnet_b0"], .25 * color])}


def map_clusters_to_seeds(cluster_fit, fit_seed_rows, y_fit, cluster_query, threshold):
    mapping = {}
    for c in np.unique(cluster_fit):
        labels = y_fit[cluster_fit == c]
        if len(labels) == 0: continue
        vals, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        if len(order) > 1 and counts[order[0]] == counts[order[1]]: continue
        purity = counts[order[0]] / len(labels)
        if purity >= threshold:
            mapping[int(c)] = (str(vals[order[0]]), float(purity), int(len(labels)))
    pred = np.array([mapping.get(int(c), ("", 0.0, 0))[0] for c in cluster_query], dtype=object)
    return pred, mapping


def main():
    global FEATURES
    started = time.time()
    seed_frame = pd.read_csv(SEEDS, dtype={"key": str, "assigned_class": str})
    assert seed_frame.origin.eq("participant_reviewed_train").all()
    assert seed_frame.key.is_unique and seed_frame.assigned_class.isin(CLASSES).all()
    assert seed_frame.groupby("assigned_class").size().max() <= 5
    paths = sorted([p for p in RAW_TRAIN.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.stem.zfill(4))
    ids = np.asarray([p.stem.zfill(4) for p in paths])
    id_to_row = {v: i for i, v in enumerate(ids)}
    seed_ids = seed_frame.key.str.split(":", regex=False).str[-1].str.zfill(4).to_numpy()
    assert set(seed_ids) <= set(id_to_row) and len(paths) == 1920
    seed_rows = np.asarray([id_to_row[i] for i in seed_ids])
    y = seed_frame.assigned_class.to_numpy(str)
    groups = source_groups(paths)
    seed_groups = groups[seed_rows]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(seed_rows, y, seed_groups))
    assert len(set(groups[seed_rows])) == len(seed_rows)
    for tr, _ in folds: assert set(y[tr]) == set(CLASSES)

    FEATURES = extract_cnn(paths)
    FEATURE_BANKS = {**FEATURES, **handcrafted(paths)}
    results, oof_records = [], []
    for bank, X in FEATURE_BANKS.items():
        print(f"cluster bank {bank}: {X.shape}", flush=True)
        for algorithm in ("minibatch_kmeans", "birch"):
            for k in K_VALUES:
                # Fit each fold once, then assess predeclared cluster-purity guards.
                predictions = {t: np.full(len(seed_rows), "", dtype=object) for t in THRESHOLDS}
                purity_scores = {t: np.zeros(len(seed_rows), dtype=np.float32) for t in THRESHOLDS}
                for fold, (fit_s, val_s) in enumerate(folds):
                    held_groups = set(seed_groups[val_s])
                    fit_rows = np.flatnonzero(~np.isin(groups, list(held_groups)))
                    n_clusters = min(k, max(2, len(fit_rows) // 3))
                    if algorithm == "minibatch_kmeans":
                        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED + fold,
                                               n_init=2, batch_size=512, max_iter=70,
                                               reassignment_ratio=0.01)
                        model.fit(X[fit_rows])
                        cf = model.predict(X[seed_rows[fit_s]])
                        cq = model.predict(X[seed_rows[val_s]])
                    else:
                        model = Birch(n_clusters=n_clusters, threshold=0.35, branching_factor=32)
                        model.fit(X[fit_rows])
                        cf = model.predict(X[seed_rows[fit_s]])
                        cq = model.predict(X[seed_rows[val_s]])
                    for threshold in THRESHOLDS:
                        p, mapping = map_clusters_to_seeds(cf, seed_rows[fit_s], y[fit_s], cq, threshold)
                        predictions[threshold][val_s] = p
                        purity_scores[threshold][val_s] = [mapping.get(int(c), ("", 0.0, 0))[1] for c in cq]
                for threshold in THRESHOLDS:
                    pred = predictions[threshold]
                    covered = pred != ""
                    n_correct = int(np.sum(pred[covered] == y[covered]))
                    result = {"feature_bank": bank, "algorithm": algorithm, "k": k,
                              "seed_purity_threshold": threshold, "n_seed": int(len(y)),
                              "covered": int(covered.sum()), "coverage": float(covered.mean()),
                              "correct_covered": n_correct,
                              "accuracy_when_covered": float(n_correct / covered.sum()) if covered.any() else None,
                              "correct_of_all_seeds": n_correct,
                              "accuracy_all_seeds_with_uncovered_wrong": float(n_correct / len(y)),
                              "macro_recall_covered": float(np.mean([np.mean(pred[covered & (y == c)] == c) for c in CLASSES if np.any(covered & (y == c))])) if covered.any() else None}
                    results.append(result)
                    for i in range(len(y)):
                        oof_records.append({"feature_bank": bank, "algorithm": algorithm, "k": k,
                                            "seed_purity_threshold": threshold, "image_id": seed_ids[i],
                                            "fold": next(f for f, (_, va) in enumerate(folds) if i in va),
                                            "seed_label": y[i], "oof_cluster_label": pred[i],
                                            "cluster_purity": float(purity_scores[threshold][i])})
    table = pd.DataFrame(results).sort_values(["correct_covered", "coverage", "accuracy_when_covered"], ascending=False, na_position="last")
    table.to_csv(OUT / "cluster_metrics.csv", index=False)
    pd.DataFrame(oof_records).to_csv(OUT / "oof_seed_cluster_predictions.csv", index=False)
    report = {"status": "completed_train_only_grouped_seed_cv",
              "raw_train_images_used_for_unsupervised_fit": len(paths),
              "eligible_seed_labels": len(seed_frame),
              "seed_class_counts": seed_frame.assigned_class.value_counts().to_dict(),
              "grouped_folds": 5, "seed_source_groups": int(len(set(seed_groups))),
              "test_images_read": 0, "processed_assignment_labels_read": False,
              "archive_annotations_read": False,
              "feature_banks": {name: int(x.shape[1]) for name, x in FEATURE_BANKS.items()},
              "algorithms": ["MiniBatchKMeans", "Birch"], "k_values": list(K_VALUES),
              "purity_thresholds": list(THRESHOLDS), "results_count": len(table),
              "top_candidates": table.head(20).replace({np.nan: None}).to_dict("records"),
              "raw_seed_hashes": {i: sha256(p) for i, p in zip(ids, paths) if i in set(seed_ids)},
              "seed_ledger_sha256": sha256(SEEDS), "runtime_seconds": round(time.time() - started, 2)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table.head(20).to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {"top_candidates", "raw_seed_hashes"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
