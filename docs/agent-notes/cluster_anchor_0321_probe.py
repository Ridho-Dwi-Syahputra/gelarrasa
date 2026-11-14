"""Train-only object-feature cluster anchored by previously reviewed plastic 0321."""
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
ANCHOR = "train:0321"
LABEL = "plastic"


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"parent_id": str})
    frame = manifest.loc[manifest.split.eq("train") & ~manifest.is_crop & manifest.training_eligible].sort_values("parent_id").reset_index(drop=True)
    cached = np.load(NOTES / ".runtime/dinov2_object_zoom_features.npz", allow_pickle=False)
    lookup = {key: i for i, key in enumerate(cached["keys"])}
    assert len(frame) == 1887 and set(frame.parent_key) <= set(lookup)
    x = normalize(cached["features"][[lookup[key] for key in frame.parent_key]].astype("float64"))
    anchors = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    human = dict(zip(anchors.key, anchors.assigned_class))
    assert human[ANCHOR] == LABEL and len(anchors) == 35
    pos = int(np.flatnonzero(frame.parent_key.eq(ANCHOR))[0])
    with threadpool_limits(limits=4):
        memberships = []
        for random_state in range(20261109, 20261114):
            cluster = KMeans(n_clusters=32, n_init=5, random_state=random_state).fit_predict(x)
            memberships.append(cluster == cluster[pos])
    support = np.stack(memberships).sum(0)
    stable = support >= 4
    conflicts = [(key, human[key]) for key in frame.parent_key[stable] if key in human and human[key] != LABEL]
    assert not conflicts
    candidate = frame[["parent_key", "parent_id", "assigned_class"]].copy()
    candidate["anchor_cluster_support"] = support.astype("uint8")
    candidate["proposed_class"] = np.where(stable, LABEL, candidate.assigned_class)
    candidate["is_proposed_change"] = candidate.proposed_class.ne(candidate.assigned_class)
    candidate["method"] = "five_seed_kmeans32_object_features_train_only_consensus"
    path = NOTES / "table-csv/cluster_anchor_0321_candidate.csv"
    candidate.to_csv(path, index=False)
    report = {"status":"frozen_candidate_not_published", "anchor":ANCHOR,"anchor_class":LABEL,
              "anchor_source":"existing_participant_reviewed_train_seed",
              "train_only_rows":len(frame),"test_rows_in_fit":0,"archive_labels_read_by_experiment":False,
              "cluster_count":32,"random_states":list(range(20261109,20261114)),"minimum_support":4,
              "anchor_cluster_sizes":[int(x.sum()) for x in memberships],"stable_cluster_size":int(stable.sum()),
              "proposed_changes":int(candidate.is_proposed_change.sum()),
              "current_classes_in_stable_cluster":frame.loc[stable,"assigned_class"].value_counts().to_dict(),
              "human_conflicts":conflicts,"candidate_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "active_processed_modified":False,"independent_semantic_validation":False}
    (NOTES / "json/cluster_anchor_0321_candidate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
