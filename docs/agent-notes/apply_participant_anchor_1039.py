"""Record the participant-confirmed train 1039 anchor within the five/class limit.

Preserves the previous operational snapshot as a local rollback artifact. This
script never reads the visual annotation archive or competition test labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "docs/agent-notes"
PATH = ROOT / "data/preprocessing_inputs.npz"
BACKUP = NOTES / ".runtime/preprocessing_inputs_before_anchor_1039.npz"
KEY = "train:1039"
LABEL = "foodwaste"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    with np.load(PATH, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    keys = data["keys"].tolist()
    pos = keys.index(KEY)
    seed_keys = data["human_seed_keys"].tolist()
    if KEY in seed_keys:
        assert data["human_seed_labels"][seed_keys.index(KEY)] == LABEL
        print("Anchor already recorded")
        return
    assert len(seed_keys) == 34
    assert sum(data["human_seed_labels"] == LABEL) == 1
    assert data["predicted_class"][pos] == "textile"
    assert data["source_sha256"][pos] == digest((ROOT / "data/Dataset DSC GelarRasa 2026/train/1039.jpg").read_bytes())
    ref = np.flatnonzero(data["reference_keys"] == KEY)
    assert len(ref) == 1 and data["reference_labels"][ref[0]] == "textile"

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(PATH, BACKUP)
    data["predicted_class"][pos] = LABEL
    data["confidence"][pos] = 1.0
    data["margin"][pos] = 1.0
    data["needs_review"][pos] = False
    data["category_source"][pos] = "participant_reviewed_seed"
    data["category_score_kind"][pos] = "human_reviewed_category"
    data["reference_labels"][ref[0]] = LABEL
    data["human_seed_keys"] = np.append(data["human_seed_keys"], KEY)
    data["human_seed_labels"] = np.append(data["human_seed_labels"], LABEL)
    data["human_seed_source_sha256"] = np.append(data["human_seed_source_sha256"], data["source_sha256"][pos])
    metadata = json.loads(str(data["metadata"].item()))
    metadata["participant_anchor_amendment"] = {
        "source_key": KEY,
        "assigned_class": LABEL,
        "confirmed_by_participant": "2026-11-09",
        "previous_class": "textile",
        "prior_snapshot_sha256": digest(BACKUP.read_bytes()),
    }
    metadata["refinement"]["manual_source_images"] = 35
    metadata["refinement"]["manual_category_counts"][LABEL] = 2
    metadata["refinement"]["human_seed_sha256"] = digest("|".join(data["human_seed_keys"]).encode())
    data["metadata"] = np.array(json.dumps(metadata, separators=(",", ":")))
    temporary = PATH.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    os.replace(temporary, PATH)

    ledger_path = NOTES / "json/manual_label_budget_status.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["manual_unique_source_images_ever_used"]) == 34
    ledger["manual_unique_source_images_ever_used"].append("1039")
    ledger["manual_unique_source_images_ever_used"].sort()
    ledger["manual_labels_consumed_by_current_experiments"] = 35
    ledger["manual_labels_active_in_processed"] = 35
    ledger["manual_class_counts"][LABEL] = 2
    ledger["remaining_by_class"][LABEL] = 3
    ledger["anchor_1039_provenance"] = "participant_direct_visual_confirmation_2026-11-09"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"anchor": KEY, "class": LABEL, "human_sources": 35,
                      "foodwaste_sources": 2, "snapshot_sha256": digest(PATH.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
