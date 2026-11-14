"""Train-only bag-of-visual-words diagnostic on frozen non-VLM patch tokens.

The 34 lawful human sources are held out by geometry group. This probe never
reads the post-freeze review, archive labels, or test images, and never
publishes assignments. Automatic targets predate the human review.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, unit


def main():
    started = time.time()
    plan = {
        "method": "source_heldout_train_only_patch_vocabulary_v1",
        "banks": ["ConvNeXt_coarse_7x7", "ConvNeXt_fine_7x7"],
        "codebook": "fixed Gaussian 64D projection then MiniBatchKMeans 48 on other-fold train patches",
        "classifier": "regularized multinomial logistic regression on patch histogram plus global DINOv3 CLS",
        "targets": "pre-human automatic anchors plus at most five team-confirmed sources per class",
        "query_handling": "whole geometry source group excluded from codebook and classifier",
        "test_rows": 0,
        "review_or_archive_read": False,
        "publication": False,
        "limitation": "same 34 sources already reused for prior development; diagnostic only",
        "source_code_sha256": sha(__file__),
    }
    (NOTES / "json/patch_vocabulary_probe_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    frame = pd.read_csv(ROOT / "data/processed/manifest.csv")
    frame = frame[(frame.split == "train") & (~frame.is_crop) & frame.training_eligible].sort_values("parent_id").reset_index(drop=True)
    keys = frame.parent_key.to_numpy(str)
    n = len(keys)
    assert n == 1904
    seeds = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    index = dict(zip(keys, range(n)))
    pos = np.array([index[k] for k in seeds.key])
    gold = np.array([CLASSES.index(c) for c in seeds.assigned_class])
    assert len(pos) == 34 and seeds.assigned_class.value_counts().le(5).all()
    human = np.zeros(n, bool)
    human[pos] = True
    auto = pd.read_csv(NOTES / "table-csv/automatic_full_seed_targets.csv").set_index("key").loc[keys]
    targets = np.array([CLASSES.index(c) for c in auto.automatic_class])
    targets[pos] = gold
    fitmask = auto.automatic_anchor.to_numpy(bool) | human
    weights = np.clip(auto.reliable_vote_count.to_numpy(float) / 4, .25, 1.)
    weights[pos] = 8.
    geo = pd.read_csv(NOTES / "table-csv/geometry_train_groups.csv").set_index("key").loc[keys]
    folds, groups = geo.fold.to_numpy(), geo.group.to_numpy()
    din = np.load(NOTES / ".runtime/dinov3_base_train_features.npz", allow_pickle=False)
    assert np.array_equal(din["keys"], keys)
    global_x = unit(din["features"][:, :768].astype(np.float32))
    file_index = json.loads((NOTES / ".runtime/convnext_patch_token_index.json").read_text())
    assert np.array_equal(file_index["keys"], keys)
    rng = np.random.default_rng(20261108)
    result = []
    with threadpool_limits(limits=3):
        for bank, dim in [("coarse", 768), ("fine", 192)]:
            record = file_index["files"][bank]
            path = ROOT / record["path"]
            assert sha(path) == record["sha256"]
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            if bank == "fine":
                raw = np.asarray(raw, np.float32).reshape(n, 7, 2, 7, 2, dim).mean((2, 4)).reshape(n, 49, dim)
            projection = rng.normal(0, 1 / np.sqrt(64), (dim, 64)).astype(np.float32)
            tokens = np.asarray(raw, np.float32) @ projection
            tokens = tokens / np.maximum(np.linalg.norm(tokens, axis=2, keepdims=True), 1e-12)
            predictions = np.full(len(pos), -1, int)
            for fold in range(5):
                tr = np.flatnonzero(folds != fold)
                query = np.flatnonzero(folds[pos] == fold)
                fit = tr[fitmask[tr]]
                assert not set(groups[fit]) & set(groups[pos[query]])
                # Only source images in the training fold form the vocabulary.
                codebook_rows = tr[::2]
                sampled = tokens[codebook_rows][:, ::6].reshape(-1, 64)
                km = MiniBatchKMeans(n_clusters=48, batch_size=512, n_init=2, max_iter=80, random_state=20261108 + fold).fit(sampled)
                needed = np.unique(np.r_[fit, pos[query]])
                nearest = km.predict(tokens[needed].reshape(-1, 64)).reshape(len(needed), 49)
                hist = np.zeros((n, 48), np.float32)
                for row, words in zip(needed, nearest):
                    hist[row] = np.bincount(words, minlength=48) / 49
                x = np.c_[global_x, hist * 4].astype(np.float32)
                model = LogisticRegression(C=.1, max_iter=350, class_weight=None)
                model.fit(x[fit], targets[fit], sample_weight=weights[fit])
                predictions[query] = model.predict(x[pos[query]])
            assert (predictions >= 0).all()
            row = {"bank": bank, "matches": int((predictions == gold).sum()),
                   "macro_recall": float(balanced_accuracy_score(gold, predictions)),
                   "plastic_recall": float((predictions[gold == 4] == 4).mean()),
                   "seconds": round(time.time() - started, 1)}
            result.append(row)
            print(row, flush=True)
    output = {"plan": plan, "result": result, "status": "diagnostic_only_no_active_change"}
    (NOTES / "json/patch_vocabulary_probe_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
