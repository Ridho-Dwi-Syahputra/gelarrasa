"""Train-only nonlinear plastic specialist, grouped source-held-out diagnostic."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, unit


def main():
    started = time.time()
    plan = {
        "method": "pre_human_anchor_extratrees_plastic_v1",
        "features": "outer-train PCA64 of DINOv3 CLS and object mean, plus frozen YOLO source scores",
        "targets": "pre-human automatic anchors, supplemented by 34 lawful human seed sources",
        "fit": "only other-fold geometry groups; 8x weight per human source",
        "estimators": 240,
        "leaf_sizes": [2, 5],
        "routing_thresholds": [.4, .5, .6, .7, .8],
        "test_rows": 0,
        "review_archive_read": False,
        "publication": False,
        "limitation": "34 reused development anchors and noisy automatic target labels",
        "source_code_sha256": sha(__file__),
    }
    (NOTES / "json/tree_plastic_material_probe_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    m = pd.read_csv(ROOT / "data/processed/manifest.csv")
    f = m[(m.split == "train") & (~m.is_crop) & m.training_eligible].sort_values("parent_id").reset_index(drop=True)
    keys = f.parent_key.to_numpy(str)
    n = len(keys)
    assert n == 1904
    s = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    lookup = dict(zip(keys, range(n)))
    pos = np.array([lookup[k] for k in s.key])
    gold = np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos) == 34 and s.assigned_class.value_counts().le(5).all()
    human = np.zeros(n, bool)
    human[pos] = True
    a = pd.read_csv(NOTES / "table-csv/automatic_full_seed_targets.csv").set_index("key").loc[keys]
    g = pd.read_csv(NOTES / "table-csv/geometry_train_groups.csv").set_index("key").loc[keys]
    groups, folds = g.group.to_numpy(), g.fold.to_numpy()
    auto = a.automatic_class.eq("plastic").to_numpy()
    labels = auto.astype(int)
    labels[pos] = (gold == 4).astype(int)
    fitmask = a.automatic_anchor.to_numpy(bool) | human
    weights = np.clip(a.reliable_vote_count.to_numpy(float) / 4, .25, 1.)
    weights[pos] = 8.
    d = np.load(NOTES / ".runtime/dinov3_base_train_features.npz", allow_pickle=False)
    y = np.load(NOTES / ".runtime/litter_object_features.npz", allow_pickle=False)
    assert np.array_equal(d["keys"], keys) and np.array_equal(y["keys"], keys)
    raw = unit(np.c_[d["features"][:, :768], d["features"][:, 1536:2304]].astype(np.float32))
    det = np.sqrt(np.maximum(y["detection_stats"], 0)).astype(np.float32)
    prior = np.load(NOTES / ".runtime/pseudo_plastic_material_probe_oof.npz", allow_pickle=False)
    assert np.array_equal(prior["keys"], s.key.to_numpy(str))
    base = prior["baseline"]
    rows = []
    with threadpool_limits(limits=4):
        scores = np.zeros((len(plan["leaf_sizes"]), len(pos)), np.float32)
        for fold in range(5):
            tr = np.flatnonzero(folds != fold)
            fit = tr[fitmask[tr]]
            q = np.flatnonzero(folds[pos] == fold)
            assert not set(groups[fit]) & set(groups[pos[q]])
            pca = PCA(n_components=64, svd_solver="randomized", random_state=20261108).fit(raw[tr])
            xfit = np.c_[unit(pca.transform(raw[fit])), det[fit]]
            xq = np.c_[unit(pca.transform(raw[pos[q]])), det[pos[q]]]
            for j, leaf in enumerate(plan["leaf_sizes"]):
                model = ExtraTreesClassifier(n_estimators=240, min_samples_leaf=leaf,
                                             max_features=.6, class_weight="balanced",
                                             n_jobs=4, random_state=20261108 + fold)
                model.fit(xfit, labels[fit], sample_weight=weights[fit])
                scores[j, q] = model.predict_proba(xq)[:, 1]
        for j, leaf in enumerate(plan["leaf_sizes"]):
            auc = float(roc_auc_score(gold == 4, scores[j]))
            for threshold in plan["routing_thresholds"]:
                route = (scores[j] >= threshold) & (base != 4)
                pred = np.where(route, 4, base)
                count = int(route.sum())
                correct = int((route & (gold == 4)).sum())
                row = {"leaf": leaf, "threshold": threshold, "auc": auc,
                       "matches": int((pred == gold).sum()),
                       "macro": float(balanced_accuracy_score(gold, pred)),
                       "routed": count, "correct_routes": correct,
                       "guard": bool((pred == gold).sum() >= 30 and
                                     balanced_accuracy_score(gold, pred) >= .9 and
                                     count >= 2 and correct == count)}
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(NOTES / "table-csv/tree_plastic_material_probe_development.csv", index=False)
    pd.DataFrame({"key": s.key.to_numpy(str), "fold": folds[pos],
                  "human_class": s.assigned_class.to_numpy(str),
                  "baseline_class": np.asarray(CLASSES)[base],
                  "plastic_score_leaf2": scores[0],
                  "plastic_score_leaf5": scores[1]}).to_csv(
                      NOTES / "table-csv/tree_plastic_material_probe_oof.csv", index=False)
    result = {"status": "diagnostic_only", "best": table.sort_values(
        ["matches", "macro", "correct_routes"], ascending=False).iloc[0].to_dict(),
              "passing": int(table.guard.sum()), "seconds": round(time.time() - started, 1)}
    (NOTES / "json/tree_plastic_material_probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
