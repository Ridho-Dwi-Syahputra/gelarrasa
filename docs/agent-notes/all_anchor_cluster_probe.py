"""Frozen train-only multi-anchor cluster candidate; no archive used in fit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "docs/agent-notes"


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"parent_id": str})
    frame = manifest.loc[manifest.split.eq("train") & ~manifest.is_crop & manifest.training_eligible].sort_values("parent_id").reset_index(drop=True)
    cached = np.load(ROOT / "data/processed/dinov2_small_embeddings.npz", allow_pickle=False)
    lookup = {key: i for i, key in enumerate(cached["keys"])}
    x = normalize(cached["features"][[lookup[key] for key in frame.parent_key]].astype("float64"))
    anchors = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    assert len(frame) == 1887 and len(anchors) == 35
    human = dict(zip(anchors.key, anchors.assigned_class))
    positions = {key: i for i, key in enumerate(frame.parent_key)}
    predictions = []
    with threadpool_limits(limits=4):
        runs = [KMeans(n_clusters=32, n_init=5, random_state=seed).fit_predict(x)
                for seed in range(20261109, 20261114)]
    for key, category in human.items():
        anchor_index = positions[key]
        support = np.stack([labels == labels[anchor_index] for labels in runs]).sum(0)
        stable = support >= 4
        conflicts = [other for other in frame.parent_key[stable] if other in human and human[other] != category]
        if conflicts:
            continue
        for index in np.flatnonzero(stable):
            predictions.append((index, key, category, int(support[index])))
    proposals = {}
    for index, anchor, category, support in predictions:
        proposals.setdefault(index, []).append((anchor, category, support))
    selected = {}
    for index, votes in proposals.items():
        if len({category for _, category, _ in votes}) == 1:
            selected[index] = (votes[0][1], len(votes), max(s for _, _, s in votes))
    result = frame[["parent_key", "parent_id", "assigned_class"]].copy()
    result["proposed_class"] = [selected[i][0] if i in selected else category
                                for i, category in enumerate(result.assigned_class)]
    result["changed"] = result.proposed_class.ne(result.assigned_class)
    result["supporting_anchors"] = [selected[i][1] if i in selected else 0 for i in range(len(result))]
    result["cluster_support"] = [selected[i][2] if i in selected else 0 for i in range(len(result))]
    path = NOTES / "table-csv/all_anchor_cluster_candidate.csv"
    result.to_csv(path, index=False)
    report = {"status": "frozen_candidate_not_published", "train_rows":len(frame), "test_rows_in_fit":0,
              "archive_labels_read_by_experiment":False, "human_anchors":len(human),
              "represented_train_rows":len(selected), "proposed_changes":int(result.changed.sum()),
              "changes_by_pair":result.loc[result.changed].groupby(["assigned_class","proposed_class"]).size().to_dict(),
              "candidate_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "active_processed_modified":False}
    report["changes_by_pair"] = {" -> ".join(pair):count for pair,count in report["changes_by_pair"].items()}
    (NOTES / "json/all_anchor_cluster_candidate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
