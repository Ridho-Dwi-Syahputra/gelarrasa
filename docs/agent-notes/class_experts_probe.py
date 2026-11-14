"""Eight source-held-out one-vs-rest experts from organizer TRAIN only.

Every expert uses pre-human automatic train anchors and the same 34 lawful
human sources. Neither test nor post-freeze review/LLM archive is read.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, unit


def main():
    started = time.time()
    plan = {
        "method": "eight_one_vs_rest_material_experts_v1",
        "fit": "pre-human automatic train anchors + at most five team-confirmed sources per class; 8x human weight",
        "validation": "five geometry source-held-out folds; PCA fit other-fold train only",
        "experts": {
            "glass": "DINOv3 object, YOLO source stats",
            "paper": "DINOv3 CLS, ConvNeXt, pixel surface",
            "electronic": "DINOv3 CLS, YOLO source stats",
            "metal": "DINOv3 object, YOLO source stats",
            "plastic": "DINOv3 CLS and object, YOLO source stats, pixel surface",
            "cardboard": "DINOv3 CLS, ConvNeXt, pixel surface",
            "textile": "DINOv3 CLS, ConvNeXt",
            "foodwaste": "DINOv3 CLS, ConvNeXt",
        },
        "model": "ExtraTrees240 minleaf3 balanced binary per expert",
        "decision": "highest expert score; selectively route only with fixed confidence and gap",
        "confidence_grid": [.55, .65, .75],
        "gap_grid": [0., .1, .2],
        "test_rows": 0,
        "review_archive_read": False,
        "publication": False,
        "limitation": "only 34 reused development anchors; one-vs-rest probabilities need not be calibrated across experts",
        "source_code_sha256": sha(__file__),
    }
    (NOTES / "json/class_experts_probe_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    m = pd.read_csv(ROOT / "data/processed/manifest.csv")
    f = m[(m.split == "train") & (~m.is_crop) & m.training_eligible].sort_values("parent_id").reset_index(drop=True)
    keys = f.parent_key.to_numpy(str)
    n = len(keys)
    assert n == 1904
    s = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    lookup = {key: i for i, key in enumerate(keys)}
    pos = np.array([lookup[key] for key in s.key])
    gold = np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos) == 34 and s.assigned_class.value_counts().le(5).all()
    human = np.zeros(n, bool)
    human[pos] = True
    a = pd.read_csv(NOTES / "table-csv/automatic_full_seed_targets.csv").set_index("key").loc[keys]
    geo = pd.read_csv(NOTES / "table-csv/geometry_train_groups.csv").set_index("key").loc[keys]
    folds, groups = geo.fold.to_numpy(), geo.group.to_numpy()
    target = np.array([CLASSES.index(c) for c in a.automatic_class])
    target[pos] = gold
    fitmask = a.automatic_anchor.to_numpy(bool) | human
    weight = np.clip(a.reliable_vote_count.to_numpy(float) / 4, .25, 1.)
    weight[pos] = 8.
    z = np.load(NOTES / ".runtime/dinov3_base_train_features.npz", allow_pickle=False)
    c = np.load(NOTES / ".runtime/convnext_train_features.npz", allow_pickle=False)
    d = np.load(NOTES / ".runtime/litter_object_features.npz", allow_pickle=False)
    surf = np.load(NOTES / ".runtime/surface_train_features.npz", allow_pickle=False)
    assert all(np.array_equal(x["keys"], keys) for x in [z, c, d, surf])
    cls = unit(z["features"][:, :768].astype(np.float32))
    obj = unit(z["features"][:, 1536:2304].astype(np.float32))
    cnn = unit(c["features"].astype(np.float32))
    detector = np.sqrt(np.maximum(d["detection_stats"], 0)).astype(np.float32)
    surface = surf["features"].astype(np.float32) / 255
    prior = np.load(NOTES / ".runtime/pseudo_plastic_material_probe_oof.npz", allow_pickle=False)
    assert np.array_equal(prior["keys"], s.key.to_numpy(str))
    baseline = prior["baseline"]
    score = np.zeros((len(pos), 8), np.float32)
    with threadpool_limits(limits=4):
        for fold in range(5):
            tr = np.flatnonzero(folds != fold)
            fit = tr[fitmask[tr]]
            query = np.flatnonzero(folds[pos] == fold)
            assert not set(groups[fit]) & set(groups[pos[query]])
            pca_cls = PCA(n_components=64, svd_solver="randomized", random_state=20261108).fit(cls[tr])
            pca_obj = PCA(n_components=64, svd_solver="randomized", random_state=20261108).fit(obj[tr])
            pca_cnn = PCA(n_components=48, svd_solver="randomized", random_state=20261108).fit(cnn[tr])
            qpos = pos[query]
            blocks_fit = {"cls": unit(pca_cls.transform(cls[fit])), "obj": unit(pca_obj.transform(obj[fit])),
                          "cnn": unit(pca_cnn.transform(cnn[fit])), "det": detector[fit], "surface": surface[fit]}
            blocks_q = {"cls": unit(pca_cls.transform(cls[qpos])), "obj": unit(pca_obj.transform(obj[qpos])),
                        "cnn": unit(pca_cnn.transform(cnn[qpos])), "det": detector[qpos], "surface": surface[qpos]}
            recipe = ["obj det", "cls cnn surface", "cls det", "obj det",
                      "cls obj det surface", "cls cnn surface", "cls cnn", "cls cnn"]
            for cat, spec in enumerate(recipe):
                names = spec.split()
                xfit = np.column_stack([blocks_fit[name] for name in names])
                xq = np.column_stack([blocks_q[name] for name in names])
                expert = ExtraTreesClassifier(n_estimators=240, min_samples_leaf=3,
                                              max_features=.7, class_weight="balanced",
                                              n_jobs=4, random_state=20261108 + fold * 8 + cat)
                expert.fit(xfit, (target[fit] == cat).astype(int), sample_weight=weight[fit])
                score[query, cat] = expert.predict_proba(xq)[:, 1]
            print("completed fold", fold, round(time.time() - started, 1), flush=True)
    assert np.isfinite(score).all()
    order = np.argsort(-score, axis=1, kind="stable")
    candidate = order[:, 0]
    top = score[np.arange(len(pos)), candidate]
    gap = top - score[np.arange(len(pos)), order[:, 1]]
    rows = []
    for minimum in plan["confidence_grid"]:
        for margin in plan["gap_grid"]:
            route = (top >= minimum) & (gap >= margin) & (candidate != baseline)
            pred = np.where(route, candidate, baseline)
            count = int(route.sum())
            right = int((route & (candidate == gold)).sum())
            rows.append({"confidence": minimum, "gap": margin,
                         "matches": int((pred == gold).sum()),
                         "macro": float(balanced_accuracy_score(gold, pred)),
                         "routes": count, "correct_routes": right,
                         "guard": bool((pred == gold).sum() >= 30 and
                             balanced_accuracy_score(gold, pred) >= .9 and count >= 2 and right == count)})
    development = pd.DataFrame(rows)
    development.to_csv(NOTES / "table-csv/class_experts_probe_development.csv", index=False)
    pd.DataFrame({"key": s.key.to_numpy(str), "fold": folds[pos],
                  "human_class": s.assigned_class.to_numpy(str),
                  "baseline_class": np.asarray(CLASSES)[baseline],
                  "expert_class": np.asarray(CLASSES)[candidate],
                  "top_score": top, "gap": gap,
                  **{f"score_{name}": score[:, i] for i, name in enumerate(CLASSES)}}).to_csv(
                      NOTES / "table-csv/class_experts_probe_oof.csv", index=False)
    result = {"status": "diagnostic_only", "unrouted_matches": int((candidate == gold).sum()),
              "unrouted_macro": float(balanced_accuracy_score(gold, candidate)),
              "best": development.sort_values(["matches", "macro", "correct_routes"], ascending=False).iloc[0].to_dict(),
              "passing": int(development.guard.sum()), "seconds": round(time.time() - started, 1)}
    (NOTES / "json/class_experts_probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
