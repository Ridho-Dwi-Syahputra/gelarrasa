"""Train-only, grouped-CV benchmark from participant-confirmed seed images.

Inputs are the raw competition train images and the participant seed ledger.
No processed assignments, archive labels, review disagreement lists, test
images, or test predictions are read by this experiment.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[3]
RAW_TRAIN = ROOT / "data" / "Dataset DSC GelarRasa 2026" / "train"
SEEDS = ROOT / "data" / "processed" / "manual_seeds.csv"
OUT = Path(__file__).resolve().parent / "seed_only_feature_benchmark_20261113"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261113
CLASSES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def phash(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    x = np.asarray(im, dtype=np.float32)
    from scipy.fft import dctn
    low = dctn(x, type=2, norm="ortho")[:8, :8]
    return low > np.median(low[1:, :])


def paths_and_seeds():
    seeds = pd.read_csv(SEEDS, dtype={"key": str, "assigned_class": str})
    assert seeds.origin.eq("participant_reviewed_train").all()
    assert seeds.key.is_unique and seeds.assigned_class.isin(CLASSES).all()
    counts = seeds.assigned_class.value_counts()
    assert counts.max() <= 5 and set(counts.index) == set(CLASSES)
    lookup = {p.stem.zfill(4): p for p in RAW_TRAIN.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    ids = seeds.key.str.split(":", regex=False).str[-1].str.zfill(4)
    assert ids.isin(lookup).all(), f"Missing raw seed files: {ids[~ids.isin(lookup)].tolist()}"
    files = [lookup[i] for i in ids]
    for row, path in zip(seeds.itertuples(), files):
        assert sha256(path) == row.source_sha256, f"Raw source hash mismatch: {row.key}"
    return seeds, ids.to_numpy(), files


def make_groups(files):
    hashes = [phash(p) for p in files]
    parent = list(range(len(files)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(files)):
        for j in range(i):
            if np.count_nonzero(hashes[i] != hashes[j]) <= 4:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(files))])


def raw_features(files):
    # Frozen non-VLM feature extractors; classifiers are fit only on seed fold train rows.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        ("resnet50", models.ResNet50_Weights.IMAGENET1K_V2,
         lambda w: (models.resnet50(weights=w), lambda m: torch.nn.Sequential(*list(m.children())[:-1]), w.transforms())),
        ("efficientnet_v2_s", models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
         lambda w: (models.efficientnet_v2_s(weights=w), lambda m: torch.nn.Sequential(m.features, m.avgpool), w.transforms())),
    ]
    cnn = {}
    with torch.inference_mode():
        for name, weights, builder in configs:
            model, head, tf = builder(weights)
            model = head(model).eval().to(device)
            batch = []
            for path in files:
                with Image.open(path) as im:
                    batch.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            x = torch.stack(batch).to(device)
            feats = model(x).flatten(1).float().cpu().numpy()
            cnn[name] = feats.astype(np.float32)
            del model, x
            if device.type == "cuda": torch.cuda.empty_cache()
    color, hogs, lbps, pixels = [], [], [], []
    for path in files:
        with Image.open(path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        hsv = np.asarray(Image.fromarray(np.uint8(rgb * 255), "RGB").convert("HSV"), dtype=np.uint8)
        h = np.concatenate([np.histogram(hsv[:, :, c], bins=b, range=(0, 256))[0] for c, b in [(0, 24), (1, 16), (2, 16)]]).astype(np.float32)
        h /= max(h.sum(), 1)
        gray = np.asarray(Image.fromarray(np.uint8(rgb * 255), "RGB").convert("L").resize((96, 96)), dtype=np.float32) / 255
        color.append(h)
        hogs.append(hog(gray, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2), feature_vector=True))
        lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
        hist = np.histogram(lbp, bins=np.arange(0, 11))[0].astype(np.float32); hist /= max(hist.sum(), 1)
        lbps.append(hist)
        pixels.append(np.asarray(Image.fromarray(np.uint8(rgb * 255), "RGB").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32).reshape(-1))
    cnn["color_hist"] = np.asarray(color, np.float32)
    cnn["hog"] = np.asarray(hogs, np.float32)
    cnn["lbp"] = np.asarray(lbps, np.float32)
    cnn["cnn_color"] = np.concatenate([cnn["resnet50"], cnn["color_hist"] * 12], axis=1)
    cnn["effnet_color"] = np.concatenate([cnn["efficientnet_v2_s"], cnn["color_hist"] * 12], axis=1)
    cnn["texture_hog_lbp"] = np.concatenate([cnn["hog"], cnn["lbp"] * 8], axis=1)
    cnn["rgb_lowres"] = np.asarray(pixels, np.float32)
    return cnn


def candidates():
    return {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced")),
        "svm_linear": make_pipeline(StandardScaler(), SVC(C=0.1, kernel="linear", class_weight="balanced")),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)),
        "knn3": make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine")),
        "knn5": make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine")),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight="balanced_subsample", random_state=SEED),
    }


def main():
    started = time.time()
    seed_frame, ids, files = paths_and_seeds()
    y = seed_frame.assigned_class.to_numpy()
    groups = make_groups(files)
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(split.split(np.zeros(len(y)), y, groups))
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(groups[va])
        assert set(y[tr]) == set(CLASSES), "A fold lacks one or more classes in fit rows."
    X = raw_features(files)
    rows, prediction_rows = [], []
    for bank, x in X.items():
        for name, estimator in candidates().items():
            pred = np.empty(len(y), dtype=object)
            for fold, (tr, va) in enumerate(folds):
                model = clone(estimator)
                model.fit(x[tr], y[tr])
                pred[va] = model.predict(x[va])
            rows.append({"feature_bank": bank, "model": name,
                         "accuracy": accuracy_score(y, pred),
                         "balanced_accuracy": balanced_accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
                         "n": len(y), "folds": len(folds)})
            for i, p in enumerate(pred):
                prediction_rows.append({"feature_bank": bank, "model": name, "image_id": ids[i],
                                        "fold": next(k for k, (_, va) in enumerate(folds) if i in va),
                                        "truth_seed": y[i], "oof_prediction": p})
    result = pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1", "accuracy"], ascending=False)
    result.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "oof_seed_predictions.csv", index=False)
    best = result.iloc[0]
    best_pred = pd.DataFrame(prediction_rows).query("feature_bank == @best.feature_bank and model == @best.model").sort_values("image_id")
    cm = confusion_matrix(best_pred.truth_seed, best_pred.oof_prediction, labels=CLASSES)
    report = {
        "status": "completed_train_only_seed_cv",
        "raw_train_seed_images": len(files),
        "human_seed_class_counts": seed_frame.assigned_class.value_counts().to_dict(),
        "grouped_folds": 3,
        "source_group_count": int(len(set(groups))),
        "test_images_read": 0,
        "processed_assignment_labels_read": False,
        "archive_annotation_labels_read": False,
        "features": list(X), "models_per_feature": len(candidates()),
        "best_by_balanced_accuracy": best.to_dict(),
        "best_confusion_matrix_class_order": CLASSES,
        "best_confusion_matrix": cm.tolist(),
        "runtime_seconds": round(time.time() - started, 2),
        "raw_seed_file_hashes": {str(i): sha256(p) for i, p in zip(ids, files)},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.head(15).to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k not in {"raw_seed_file_hashes", "best_confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
