"""Plastic-vs-each-material experts trained on organizer TRAIN automatic anchors."""
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
    start = time.time()
    plan = {"method": "plastic_vs_each_auto_material_experts_v1",
            "fit": "pre-human automatic anchors in each pair, plus max five lawful human sources/class",
            "geometry": "outer fold excludes complete query source group from PCA and fit",
            "pairs": [f"plastic_vs_{c}" for c in CLASSES if c != "plastic"],
            "features": "DINOv3 CLS+object and ConvNeXt PCA, YOLO source scores, pixel surface",
            "model": "ExtraTrees240 minleaf2 class balanced, 8x human source weight",
            "router": "query evaluated only by specialist matching current baseline class; fixed probability thresholds",
            "thresholds": [.5, .6, .7, .8], "test_rows": 0,
            "review_archive_read": False, "publication": False,
            "limitation": "34 repeatedly used human development sources; automatic pair labels noisy",
            "source_code_sha256": sha(__file__)}
    (NOTES / "json/pairwise_auto_experts_probe_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    m = pd.read_csv(ROOT / "data/processed/manifest.csv")
    f = m[(m.split == "train") & (~m.is_crop) & m.training_eligible].sort_values("parent_id").reset_index(drop=True)
    keys = f.parent_key.to_numpy(str)
    n = len(keys)
    assert n == 1904
    s = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    pos = np.array([dict(zip(keys, range(n)))[k] for k in s.key])
    gold = np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos) == 34 and s.assigned_class.value_counts().le(5).all()
    human = np.zeros(n, bool)
    human[pos] = True
    a = pd.read_csv(NOTES / "table-csv/automatic_full_seed_targets.csv").set_index("key").loc[keys]
    g = pd.read_csv(NOTES / "table-csv/geometry_train_groups.csv").set_index("key").loc[keys]
    target = np.array([CLASSES.index(c) for c in a.automatic_class])
    target[pos] = gold
    fitmask = a.automatic_anchor.to_numpy(bool) | human
    weight = np.clip(a.reliable_vote_count.to_numpy(float) / 4, .25, 1.)
    weight[pos] = 8.
    folds, groups = g.fold.to_numpy(), g.group.to_numpy()
    z = np.load(NOTES / ".runtime/dinov3_base_train_features.npz", allow_pickle=False)
    c = np.load(NOTES / ".runtime/convnext_train_features.npz", allow_pickle=False)
    d = np.load(NOTES / ".runtime/litter_object_features.npz", allow_pickle=False)
    surf = np.load(NOTES / ".runtime/surface_train_features.npz", allow_pickle=False)
    assert all(np.array_equal(x["keys"], keys) for x in [z, c, d, surf])
    raw = unit(np.c_[z["features"][:, :768], z["features"][:, 1536:2304]].astype(np.float32))
    cnn = unit(c["features"].astype(np.float32))
    det = np.sqrt(np.maximum(d["detection_stats"], 0)).astype(np.float32)
    surface = surf["features"].astype(np.float32) / 255
    prior = np.load(NOTES / ".runtime/pseudo_plastic_material_probe_oof.npz", allow_pickle=False)
    assert np.array_equal(prior["keys"], s.key.to_numpy(str))
    baseline = prior["baseline"]
    scores = np.full((len(pos), 8), np.nan, np.float32)
    with threadpool_limits(limits=4):
        for fold in range(5):
            tr = np.flatnonzero(folds != fold)
            q = np.flatnonzero(folds[pos] == fold)
            pca = PCA(n_components=80, svd_solver="randomized", random_state=20261108).fit(raw[tr])
            pcac = PCA(n_components=32, svd_solver="randomized", random_state=20261108).fit(cnn[tr])
            def feature(indices):
                return np.c_[unit(pca.transform(raw[indices])), unit(pcac.transform(cnn[indices])), det[indices], surface[indices]]
            for other in range(8):
                if other == 4:
                    continue
                fit = tr[fitmask[tr] & np.isin(target[tr], [other, 4])]
                assert not set(groups[fit]) & set(groups[pos[q]])
                expert = ExtraTreesClassifier(n_estimators=240, min_samples_leaf=2,
                                              max_features=.7, class_weight="balanced",
                                              n_jobs=4, random_state=20261108 + fold * 8 + other)
                expert.fit(feature(fit), (target[fit] == 4).astype(int), sample_weight=weight[fit])
                scores[q, other] = expert.predict_proba(feature(pos[q]))[:, 1]
            print("fold", fold, "done", round(time.time() - start, 1), flush=True)
    assert np.isfinite(scores[:, [c for c in range(8) if c != 4]]).all()
    chosen_score = scores[np.arange(len(pos)), baseline]
    chosen_score[baseline == 4] = 0.
    rows = []
    for threshold in plan["thresholds"]:
        route = (baseline != 4) & (chosen_score >= threshold)
        pred = np.where(route, 4, baseline)
        count = int(route.sum())
        correct = int((route & (gold == 4)).sum())
        rows.append({"threshold": threshold, "matches": int((pred == gold).sum()),
                     "macro": float(balanced_accuracy_score(gold, pred)),
                     "routes": count, "correct_routes": correct,
                     "guard": bool((pred == gold).sum() >= 30 and
                         balanced_accuracy_score(gold, pred) >= .9 and count >= 2 and correct == count)})
    out = pd.DataFrame(rows)
    out.to_csv(NOTES / "table-csv/pairwise_auto_experts_probe_development.csv", index=False)
    pd.DataFrame({"key": s.key.to_numpy(str), "human_class": s.assigned_class.to_numpy(str),
                  "baseline_class": np.asarray(CLASSES)[baseline], "matching_expert_plastic_score": chosen_score,
                  **{f"plastic_vs_{CLASSES[i]}": scores[:, i] for i in range(8) if i != 4}}).to_csv(
                      NOTES / "table-csv/pairwise_auto_experts_probe_oof.csv", index=False)
    result = {"status": "diagnostic_only", "best": out.sort_values(["matches", "macro"], ascending=False).iloc[0].to_dict(),
              "passing": int(out.guard.sum()), "seconds": round(time.time() - start, 1)}
    (NOTES / "json/pairwise_auto_experts_probe_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
