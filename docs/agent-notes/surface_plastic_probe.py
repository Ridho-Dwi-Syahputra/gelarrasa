"""Train-only pixel surface features for a grouped plastic material probe."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, unit


def describe(path: Path) -> np.ndarray:
    with Image.open(path) as src:
        im = np.asarray(ImageOps.exif_transpose(src).convert("RGB").resize((128, 128)), np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1].astype(np.float32) / 255
    val = hsv[:, :, 2].astype(np.float32) / 255
    edges = cv2.Canny(gray, 40, 120)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    channels = [im[:, :, i] for i in range(3)] + [hsv[:, :, i] for i in range(3)] + [lab[:, :, i] for i in range(3)]
    moments = [v for c in channels for v in (np.mean(c), np.std(c), np.percentile(c, 10), np.percentile(c, 90))]
    hist = np.histogram(hsv[:, :, 0], bins=12, range=(0, 180), density=True)[0]
    surface = [np.mean(sat < .12), np.mean(sat > .55), np.mean(val > .9),
               np.mean((val > .85) & (sat < .2)), np.mean(edges > 0),
               np.mean(np.abs(lap)), np.std(lap),
               np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0))),
               np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1)))]
    return np.asarray([*moments, *hist, *surface], np.float32)


def main():
    start = time.time()
    plan = {"method": "pixel_surface_plus_frozen_dinov3_extratrees_v1",
            "pixels": "RGB HSV LAB moments, hue histogram, highlight fraction, Canny edges, Laplacian texture",
            "feature_fit": "none on test; only outer-train PCA on DINOv3 features",
            "targets": "pre-human automatic anchors plus max5 human sources per class",
            "validation": "five geometry source-held-out folds",
            "test_rows": 0, "review_archive_read": False, "publication": False,
            "source_code_sha256": sha(__file__)}
    (NOTES / "json/surface_plastic_probe_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    m = pd.read_csv(ROOT / "data/processed/manifest.csv")
    f = m[(m.split == "train") & (~m.is_crop) & m.training_eligible].sort_values("parent_id").reset_index(drop=True)
    keys = f.parent_key.to_numpy(str)
    assert len(keys) == 1904
    s = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    idx = {k: i for i, k in enumerate(keys)}
    pos = np.array([idx[k] for k in s.key])
    gold = np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos) == 34 and s.assigned_class.value_counts().le(5).all()
    human = np.zeros(len(keys), bool)
    human[pos] = True
    a = pd.read_csv(NOTES / "table-csv/automatic_full_seed_targets.csv").set_index("key").loc[keys]
    g = pd.read_csv(NOTES / "table-csv/geometry_train_groups.csv").set_index("key").loc[keys]
    folds, groups = g.fold.to_numpy(), g.group.to_numpy()
    labels = a.automatic_class.eq("plastic").to_numpy().astype(int)
    labels[pos] = (gold == 4).astype(int)
    fitmask = a.automatic_anchor.to_numpy(bool) | human
    weight = np.clip(a.reliable_vote_count.to_numpy(float) / 4, .25, 1.)
    weight[pos] = 8.
    d = np.load(NOTES / ".runtime/dinov3_base_train_features.npz", allow_pickle=False)
    y = np.load(NOTES / ".runtime/litter_object_features.npz", allow_pickle=False)
    assert np.array_equal(d["keys"], keys) and np.array_equal(y["keys"], keys)
    raw = unit(np.c_[d["features"][:, :768], d["features"][:, 1536:2304]].astype(np.float32))
    det = np.sqrt(np.maximum(y["detection_stats"], 0)).astype(np.float32)
    paths = [ROOT / v for v in f.source_path.to_numpy(str)]
    assert all(p.is_file() for p in paths)
    surface = np.vstack([describe(p) for p in paths])
    np.savez_compressed(NOTES / ".runtime/surface_train_features.npz", keys=keys, features=surface)
    prior = np.load(NOTES / ".runtime/pseudo_plastic_material_probe_oof.npz", allow_pickle=False)
    assert np.array_equal(prior["keys"], s.key.to_numpy(str))
    base = prior["baseline"]
    rows = []
    with threadpool_limits(limits=4):
        score = np.zeros(len(pos), np.float32)
        for fold in range(5):
            tr = np.flatnonzero(folds != fold)
            fit = tr[fitmask[tr]]
            q = np.flatnonzero(folds[pos] == fold)
            assert not set(groups[fit]) & set(groups[pos[q]])
            pca = PCA(n_components=64, svd_solver="randomized", random_state=20261108).fit(raw[tr])
            xfit = np.c_[unit(pca.transform(raw[fit])), det[fit], surface[fit] / 255]
            xq = np.c_[unit(pca.transform(raw[pos[q]])), det[pos[q]], surface[pos[q]] / 255]
            model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                         class_weight="balanced", max_features=.7,
                                         n_jobs=4, random_state=20261108 + fold)
            model.fit(xfit, labels[fit], sample_weight=weight[fit])
            score[q] = model.predict_proba(xq)[:, 1]
        auc = float(roc_auc_score(gold == 4, score))
        for threshold in [.35, .4, .45, .5, .55, .6, .65, .7]:
            route = (score >= threshold) & (base != 4)
            pred = np.where(route, 4, base)
            count = int(route.sum())
            right = int((route & (gold == 4)).sum())
            row = {"threshold": threshold, "auc": auc, "matches": int((pred == gold).sum()),
                   "macro": float(balanced_accuracy_score(gold, pred)), "routed": count,
                   "correct_routes": right, "guard": bool((pred == gold).sum() >= 30 and
                             balanced_accuracy_score(gold, pred) >= .9 and count >= 2 and right == count)}
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(NOTES / "table-csv/surface_plastic_probe_development.csv", index=False)
    result = {"status": "diagnostic_only", "best": table.sort_values(["matches", "macro", "correct_routes"], ascending=False).iloc[0].to_dict(),
              "passing": int(table.guard.sum()), "seconds": round(time.time() - start, 1)}
    (NOTES / "json/surface_plastic_probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
