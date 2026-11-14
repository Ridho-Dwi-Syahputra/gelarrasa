"""Rebuild the original v1 preprocessing bundle from archived train-only outputs.

This reproduces the initial bundle, not the later refined bundle currently used
by the main notebook. The source image data is checked separately by
``rebuild_initial_stage1.py``; this script then assembles the original target
and its train-only DINOv2 reference bank from the frozen upstream artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ARCHIVE = PROJECT / "arsip/sebelum_rapi_20261107"
BASELINE = PROJECT / "docs/agent-notes/.runtime/before_program_refinement_20261107T050743Z/preprocessing_inputs.npz"
STAGE17 = HERE / "output/named_assignments.csv"
OUTPUT = HERE / "preprocessing_inputs_v1_rebuilt.npz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    profile = pd.read_csv(
        ARCHIVE / "output/manifests/stage1_v1/image_profile.csv",
        dtype={"image_id": str},
    )
    groups_frame = pd.read_csv(
        ARCHIVE / "output/groups/stage1_v1/groups.csv", dtype={"image_id": str}
    )
    fit = pd.read_csv(
        ARCHIVE / "output/groups/stage4_self_training_pilot_v1/fit_seed_mask.csv",
        dtype=str,
    )
    assignments = pd.read_csv(STAGE17, dtype={"image_id": str})
    dino = np.load(
        ARCHIVE / "output/embeddings/stage2_repr_v1/dinov2_features.npz",
        allow_pickle=False,
    )
    keys = assignments.key.to_numpy(dtype=str)
    assert np.array_equal(keys, profile.key.to_numpy(dtype=str))
    assert np.array_equal(keys, groups_frame.key.to_numpy(dtype=str))
    assert np.array_equal(keys, dino["keys"].astype(str))
    assert assignments.split.eq("train").sum() == 1920
    assert assignments.split.eq("test").sum() == 480

    candidate_groups = groups_frame.candidate_group.to_numpy(dtype=str)
    ref_keys = fit.key.to_numpy(dtype=str)
    assignment_index = pd.Series(np.arange(len(keys)), index=keys)
    ref_assignment_idx = assignment_index.loc[ref_keys].to_numpy()
    group_index = pd.Series(candidate_groups, index=keys)
    features_index = pd.Series(np.arange(len(dino["keys"])), index=dino["keys"].astype(str))
    ref_feature_idx = features_index.loc[ref_keys].to_numpy()

    metadata = {
        "model": "vit_small_patch14_dinov2.lvd142m",
        "repository": "timm/vit_small_patch14_dinov2.lvd142m",
        "revision": "4610ca143709d58a633b6397a74412c2c3842454",
        "checkpoint_sha256": "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081",
        "version": 1,
        "assignment_method": "fixed_hierarchical_material_routing_v1",
        "assignment_sha256": sha(STAGE17),
        "class_order": ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"],
        "reference_selection": "historical_fit_train_only; crop parent group excluded during query",
        "crop_method": "DINOv2 cosine 7-neighbor vote; temperature .07; no parameter search",
        "official_ground_truth": False,
    }
    arrays = {
        "keys": keys,
        "source_sha256": profile.source_sha256.to_numpy(dtype=str),
        "groups": candidate_groups,
        "predicted_class": assignments.predicted_class.to_numpy(dtype=str),
        "confidence": assignments.uncalibrated_probability.to_numpy(dtype=np.float32),
        "margin": assignments.margin.to_numpy(dtype=np.float32),
        "needs_review": assignments.needs_review.to_numpy(dtype=bool),
        "cnn_prior_disagrees": assignments.cnn_prior_disagrees.to_numpy(dtype=bool),
        "reference_keys": ref_keys,
        "reference_features": dino["features"][ref_feature_idx].astype(np.float32, copy=False),
        "reference_labels": assignments.predicted_class.to_numpy(dtype=str)[ref_assignment_idx],
        "reference_groups": group_index.loc[ref_keys].to_numpy(dtype=str),
        "metadata": np.array(json.dumps(metadata)),
    }
    np.savez_compressed(OUTPUT, **arrays)

    expected = np.load(BASELINE, allow_pickle=False)
    rebuilt = np.load(OUTPUT, allow_pickle=False)
    assert expected.files == rebuilt.files
    mismatches = []
    for name in expected.files:
        if not np.array_equal(expected[name], rebuilt[name]):
            mismatches.append(name)
    assert not mismatches, f"Rebuilt v1 arrays differ: {mismatches}"
    print(json.dumps({
        "result": "all v1 arrays reproduced exactly",
        "output": str(OUTPUT),
        "output_sha256": sha(OUTPUT),
        "baseline_npz_sha256": sha(BASELINE),
        "assignment_sha256": sha(STAGE17),
        "arrays": expected.files,
        "reference_rows": len(ref_keys),
        "test_targets_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
