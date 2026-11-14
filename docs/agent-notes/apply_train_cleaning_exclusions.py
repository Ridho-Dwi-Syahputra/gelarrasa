"""Record participant-selected noisy train sources without assigning new labels."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/preprocessing_inputs.npz"
CONTRACT = ROOT / "docs/agent-notes/project_contract.json"
REVIEW = ROOT / "docs/agent-notes/table-csv/remaining_current_review_differences.csv"
IDS = (
    "0058", "0265", "0310", "0498", "0559", "0964", "0978", "1018", "1142",
    "1281", "1436", "1515", "1531", "1657", "1848", "1904", "1913",
)
KEYS = {f"train:{image_id}" for image_id in IDS}


def main() -> None:
    with REVIEW.open(newline="", encoding="utf-8") as file:
        review_ids = {row["parent_id"].zfill(4) for row in csv.DictReader(file)}
    assert review_ids == set(IDS), "Active 17-source audit changed; inspect before applying"

    with np.load(INPUT, allow_pickle=False) as bundle:
        arrays = {key: bundle[key].copy() for key in bundle.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    all_keys = set(map(str, arrays["keys"]))
    human_keys = set(map(str, arrays.get("human_seed_keys", [])))
    assert KEYS <= all_keys and not KEYS & human_keys
    assert all(key.startswith("train:") for key in KEYS)
    deferred = set(metadata.get("deferred_train_keys", []))
    assert set(metadata.get("train_cleaning_exclusion_keys", [])) <= KEYS
    metadata["deferred_train_keys"] = sorted(deferred | KEYS)
    metadata["train_cleaning_exclusion_keys"] = sorted(KEYS)
    metadata["train_cleaning_policy"] = "participant_noise_cleaning_exclude_from_classifier_fit_only"
    metadata["train_cleaning_rule_source"] = (
        "https://padlet.com/gelarrasa/"
        "qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/"
        "wish/PVKBQO0wN9pkZj5x"
    )
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    temporary = INPUT.with_name("preprocessing_inputs.cleaning_pending.npz")
    np.savez_compressed(temporary, **arrays)
    with np.load(temporary, allow_pickle=False) as check:
        assert json.loads(str(check["metadata"].item()))["train_cleaning_exclusion_keys"] == sorted(KEYS)
    temporary.replace(INPUT)

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["deferred_train_keys"] = sorted(deferred | KEYS)
    contract["train_cleaning_exclusion_keys"] = sorted(KEYS)
    CONTRACT.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Excluded {len(KEYS)} additional train sources; total deferred {len(deferred | KEYS)}")


if __name__ == "__main__":
    main()
