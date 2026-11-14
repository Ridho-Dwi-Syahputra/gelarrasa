"""Train-only, source-aware cluster probe anchored by a participant-reviewed image.

This writes a candidate and freezes its digest before any archive comparison.
It does not modify active preprocessing, notebook, or submissions.
"""
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
ANCHOR = "train:1039"
CLUSTER_COUNT = 32
SEEDS = tuple(range(20261109, 20261114))
MIN_SUPPORT = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"parent_id": str})
    frame = manifest.loc[
        manifest.split.eq("train") & ~manifest.is_crop & manifest.training_eligible
    ].sort_values("parent_id").reset_index(drop=True)
    assert len(frame) == 1887 and frame.parent_key.is_unique
    assert ANCHOR in set(frame.parent_key)

    cached = np.load(ROOT / "data/processed/dinov2_small_embeddings.npz", allow_pickle=False)
    lookup = {key: i for i, key in enumerate(cached["keys"])}
    assert set(frame.parent_key) <= set(lookup)
    x = normalize(cached["features"][[lookup[key] for key in frame.parent_key]].astype("float64"))
    anchor_pos = int(np.flatnonzero(frame.parent_key.eq(ANCHOR))[0])
    memberships = []
    with threadpool_limits(limits=4):
        for random_state in SEEDS:
            cluster = KMeans(n_clusters=CLUSTER_COUNT, n_init=5, random_state=random_state).fit_predict(x)
            memberships.append(cluster == cluster[anchor_pos])
    support = np.stack(memberships).sum(axis=0)
    stable = support >= MIN_SUPPORT

    human = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    assert len(human) == 34 and human.assigned_class.value_counts().le(5).all()
    reviewed = dict(zip(human.key, human.assigned_class))
    reviewed[ANCHOR] = "foodwaste"  # Confirmed directly by participant on 2026-11-09.
    conflicting = [(key, reviewed[key]) for key in frame.parent_key[stable] if key in reviewed and reviewed[key] != "foodwaste"]
    if conflicting:
        stable[:] = False

    output = frame[["parent_key", "parent_id", "parent_group", "assigned_class"]].copy()
    output["anchor_cluster_support"] = support.astype("uint8")
    output["proposed_class"] = np.where(stable, "foodwaste", output.assigned_class)
    output["is_proposed_change"] = output.proposed_class.ne(output.assigned_class)
    output["human_anchor"] = output.parent_key.eq(ANCHOR)
    output["method"] = "five_seed_kmeans32_dinov2_train_only_consensus"
    path = NOTES / "table-csv/cluster_anchor_1039_candidate.csv"
    output.to_csv(path, index=False)
    report = {
        "status": "frozen_candidate_not_published",
        "anchor": ANCHOR,
        "anchor_class_source": "participant_direct_visual_confirmation_2026-11-09",
        "human_foodwaste_used_after_confirmation": 2,
        "train_only_rows": len(frame),
        "test_rows_in_fit": 0,
        "archive_labels_read_by_experiment": False,
        "cluster_count": CLUSTER_COUNT,
        "random_states": list(SEEDS),
        "minimum_support": MIN_SUPPORT,
        "anchor_cluster_sizes": [int(m.sum()) for m in memberships],
        "stable_cluster_size": int(stable.sum()),
        "proposed_changes": int(output.is_proposed_change.sum()),
        "current_classes_in_stable_cluster": frame.loc[stable, "assigned_class"].value_counts().to_dict(),
        "human_conflicts": conflicting,
        "candidate_sha256": sha256(path),
        "active_processed_modified": False,
        "independent_semantic_validation": False,
    }
    report_path = NOTES / "json/cluster_anchor_1039_candidate.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
