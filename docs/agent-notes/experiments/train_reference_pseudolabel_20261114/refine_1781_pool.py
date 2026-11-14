"""Grouped, train-only residual expansion from the participant-approved pool.

The 1,781-row starting pool is frozen from the preceding post-prediction
agreement audit plus the participant's 19 keep IDs. Features are regenerated
from raw organizer train images by DSC2026_Notebook.ipynb. For residuals, each
fold model freezes its predictions before reference matching/acceptance.
No test image or test label is loaded here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import SVC


PROJECT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT / "data" / "Dataset DSC GelarRasa 2026"
RUN_ROOT = Path(os.environ.get("GELARRASA_WORK_ROOT", str(Path.home() / "AppData/Local/Temp/gelarrasa_raw_only_rerun_20261114")))
OUT = Path(__file__).resolve().parent
REFERENCE_ROOT = PROJECT / "docs" / "label"
KEEP_IDS = {
    "1180", "1337", "1368", "1548", "1663", "1154", "0890", "0637", "0676",
    "0557", "0160", "0086", "0183", "0225", "0189", "0260", "0267", "0378", "0396",
}
CLASSES = ["cardboard", "electronic", "foodwaste", "glass", "metal", "paper", "plastic", "textile"]


def norm_id(value: object) -> str:
    return str(value).strip().split(":")[-1].split(".")[0].zfill(4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference() -> dict[str, str]:
    ref: dict[str, str] = {}
    for folder in sorted(REFERENCE_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in CLASSES:
            raise ValueError(f"Unknown reference class folder: {folder.name}")
        for path in folder.iterdir():
            if path.is_file():
                key = norm_id(path.stem)
                if key in ref:
                    raise ValueError(f"Duplicate reference image ID: {key}")
                ref[key] = folder.name
    return ref


def load_feature_bank(path: Path, id_key: str, feature_key: str, hash_key: str):
    z = np.load(path, allow_pickle=False)
    ids = [norm_id(x) for x in z[id_key].astype(str)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate feature IDs in {path}")
    feats = np.asarray(z[feature_key], dtype=np.float32)
    hashes = z[hash_key].astype(str).tolist()
    if len(ids) != len(feats) or len(ids) != len(hashes):
        raise ValueError(f"Feature bank length mismatch: {path}")
    return dict(zip(ids, feats)), dict(zip(ids, hashes))


def make_models(train_ids: list[str], labels: np.ndarray, banks: dict[str, dict[str, np.ndarray]]):
    models = {}
    for name in ("dinov2", "dinov3", "effnet"):
        x = np.stack([banks[name][sid] for sid in train_ids]).astype(np.float32)
        if name in ("dinov2", "dinov3"):
            x = normalize(x)
            estimator = SVC(C=10.0, kernel="rbf", gamma="scale", decision_function_shape="ovr")
        else:
            estimator = make_pipeline(StandardScaler(), SVC(C=10.0, kernel="rbf", gamma="scale", decision_function_shape="ovr"))
        estimator.fit(x, labels)
        models[name] = estimator
    return models


def main() -> None:
    raw_paths = {norm_id(p.stem): p for p in (DATA_ROOT / "train").iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    if len(raw_paths) != 1920:
        raise AssertionError(f"Expected 1920 raw train images, found {len(raw_paths)}")
    ids = sorted(raw_paths, key=int)
    reference = load_reference()
    if set(reference) != set(ids):
        raise AssertionError("Reference and raw train IDs differ")

    decisions = pd.read_csv(RUN_ROOT / "labels/all_train_pseudo_label_decisions.csv", dtype={"sample_id": str, "parent_id": str})
    decisions = decisions.loc[~decisions.is_crop.astype(bool)].copy()
    decisions["id"] = decisions.parent_id.map(norm_id)
    decision_by_id = decisions.set_index("id").to_dict("index")
    groups_df = pd.read_csv(RUN_ROOT / "labels/coarse_train_only/stage4_train_coarse_clusters.csv", dtype={"image_id": str})
    groups_df["id"] = groups_df.image_id.map(norm_id)
    group_map = dict(zip(groups_df.id, groups_df.source_group.astype(str)))

    # Verify every cached feature row still identifies the raw train image.
    raw_hashes = {sid: sha256(path) for sid, path in raw_paths.items()}
    group_rows = groups_df.set_index("id")
    for sid, digest in raw_hashes.items():
        if str(group_rows.loc[sid, "source_sha256"]) != digest:
            raise AssertionError(f"Raw train hash mismatch against fresh notebook feature run: {sid}")

    d2, d2h = load_feature_bank(RUN_ROOT / "features/assignment_rebuild/dinov2_small_train_features.npz", "keys", "features", "source_sha256")
    d3, d3h = load_feature_bank(RUN_ROOT / "features/assignment_rebuild/dinov3_small_train_descriptors.npz", "keys", "features", "source_sha256")
    z = np.load(RUN_ROOT / "features/hybrid_labeling/train_cnn_hog_color.npz", allow_pickle=False)
    eff_ids = [norm_id(x) for x in z["image_ids"].astype(str)]
    if len(eff_ids) != len(set(eff_ids)):
        raise AssertionError("Duplicate EfficientNet feature IDs")
    eff = dict(zip(eff_ids, np.asarray(z["cnn"], dtype=np.float32)))
    eff_hash = dict(zip(eff_ids, z["source_sha256"].astype(str).tolist()))
    for sid in ids:
        if any(sid not in bank for bank in (d2, d3, eff)):
            raise AssertionError(f"Missing raw-train feature for {sid}")
        if raw_hashes[sid] != d2h[sid] or raw_hashes[sid] != d3h[sid] or raw_hashes[sid] != eff_hash[sid]:
            raise AssertionError(f"Feature/source checksum mismatch for {sid}")
    banks = {"dinov2": d2, "dinov3": d3, "effnet": eff}

    # Freeze the initial pool: archive matches plus the participant's explicit
    # keep list. For keep IDs, retain the already-produced program label.
    trusted: dict[str, str] = {}
    source: dict[str, str] = {}
    for sid in ids:
        row = decision_by_id.get(sid)
        candidate = str(row["assigned_class"]) if row else None
        if sid in KEEP_IDS:
            if candidate not in CLASSES:
                raise AssertionError(f"User-approved keep ID has no program prediction: {sid}")
            trusted[sid] = candidate
            source[sid] = "participant_keep_existing_program_prediction"
        elif candidate in CLASSES and candidate == reference[sid]:
            trusted[sid] = candidate
            source[sid] = "frozen_program_prediction_matches_reference"
    if len(trusted) != 1781:
        raise AssertionError(f"Expected 1781 initial pool rows, got {len(trusted)}")
    if len(KEEP_IDS) != 19 or not KEEP_IDS <= set(trusted):
        raise AssertionError("The 19 participant-approved IDs are not all in the initial pool")

    initial_pool_size = len(trusted)
    all_round_rows: list[dict] = []
    round_summaries: list[dict] = []
    max_rounds = 10

    for round_id in range(1, max_rounds + 1):
        residual = [sid for sid in ids if sid not in trusted]
        if not residual:
            break
        current_train = sorted(trusted, key=int)
        y = np.asarray([trusted[sid] for sid in current_train], dtype=object)
        if set(y) != set(CLASSES):
            raise AssertionError(f"Round {round_id}: training pool misses class(es): {set(CLASSES)-set(y)}")

        votes = {sid: {} for sid in residual}
        splitter = GroupKFold(n_splits=5)
        # Split all train source groups; for each prediction fold, exclude every
        # trusted member of each held source group from the fit.
        for fold_id, (all_fit_idx, all_val_idx) in enumerate(splitter.split(ids, groups=[group_map[sid] for sid in ids])):
            val_ids = [ids[i] for i in all_val_idx]
            val_groups = {group_map[sid] for sid in val_ids}
            val_residual = [sid for sid in val_ids if sid in votes]
            train_ids = [sid for sid in current_train if group_map[sid] not in val_groups]
            if not val_residual:
                continue
            if not train_ids or val_groups & {group_map[sid] for sid in train_ids}:
                raise AssertionError("Source-group leakage or empty fold training set")
            fold_models = make_models(train_ids, np.asarray([trusted[sid] for sid in train_ids], dtype=object), banks)
            for model_name, model in fold_models.items():
                x_val = np.stack([banks[model_name][sid] for sid in val_residual]).astype(np.float32)
                if model_name in ("dinov2", "dinov3"):
                    x_val = normalize(x_val)
                preds = model.predict(x_val)
                for sid, pred in zip(val_residual, preds):
                    votes[sid][model_name] = str(pred)

        accepted_this_round = []
        for sid in residual:
            row_votes = votes[sid]
            if len(row_votes) != 3:
                raise AssertionError(f"Missing cross-fit model predictions for {sid}: {row_votes}")
            counts = pd.Series(list(row_votes.values())).value_counts()
            top_label = str(counts.index[0])
            consensus = int(counts.iloc[0])
            matched = top_label == reference[sid]
            accepted = consensus >= 2 and matched
            all_round_rows.append({
                "round": round_id, "image_id": sid,
                "previous_program_label": (decision_by_id.get(sid) or {}).get("assigned_class", ""),
                "reference_label": reference[sid],
                "dinov2_svm": row_votes["dinov2"], "dinov3_svm": row_votes["dinov3"], "effnet_svm": row_votes["effnet"],
                "majority_prediction": top_label, "model_consensus": consensus,
                "prediction_matches_reference": matched, "accepted_for_next_round": accepted,
                "source_group": group_map[sid], "raw_source_sha256": raw_hashes[sid],
            })
            if accepted:
                accepted_this_round.append(sid)
        round_summaries.append({
            "round": round_id, "pool_before": len(trusted), "residual_before": len(residual),
            "predicted": len(residual), "accepted": len(accepted_this_round),
            "remaining_after": len(residual) - len(accepted_this_round),
            "accept_rule": "at least 2/3 source-group-crossfit models agree and majority equals reference",
        })
        if not accepted_this_round:
            break
        for sid in accepted_this_round:
            trusted[sid] = reference[sid]
            source[sid] = f"round_{round_id}_crossfit_consensus_matches_reference"

    predictions = pd.DataFrame(all_round_rows)
    final_pool = pd.DataFrame([
        {"image_id": sid, "assigned_class": trusted[sid], "pool_source": source[sid],
         "source_group": group_map[sid], "raw_source_sha256": raw_hashes[sid]}
        for sid in sorted(trusted, key=int)
    ])
    residual_final = sorted(set(ids) - set(trusted), key=int)
    pred_path = OUT / "iteration_06_residual_predictions.csv"
    pool_path = OUT / "iteration_06_final_pool.csv"
    summary_path = OUT / "iteration_06_summary.json"
    predictions.to_csv(pred_path, index=False)
    final_pool.to_csv(pool_path, index=False)

    raw_candidate_labels = {sid: str(decision_by_id[sid]["assigned_class"]) for sid in decision_by_id}
    baseline_assigned = len(raw_candidate_labels)
    summary = {
        "method": "1781_user_approved_and_reference_agreeing_seed_pool_then_iterative_source_group_crossfit",
        "raw_train_images": len(ids), "raw_test_images_read": 0,
        "raw_dataset_root": str(DATA_ROOT), "notebook_generated_feature_run": str(RUN_ROOT),
        "initial_pool": initial_pool_size, "initial_user_keep_ids": len(KEEP_IDS),
        "initial_reference_matching_program_labels": initial_pool_size - len(KEEP_IDS & {sid for sid in ids if sid in decision_by_id and str(decision_by_id[sid]['assigned_class']) != reference[sid]}),
        "initial_residual": len(ids) - initial_pool_size,
        "residual_composition": {"existing_program_label_differs_from_reference": sum(sid in decision_by_id and str(decision_by_id[sid]['assigned_class']) != reference[sid] and sid not in KEEP_IDS for sid in ids),
                                 "no_program_assignment": sum(sid not in decision_by_id for sid in ids)},
        "feature_families": ["DINOv2-Small + RBF SVM", "DINOv3-Small + RBF SVM", "EfficientNet-B0 embedding + StandardScaler + RBF SVM"],
        "validation": "5-fold GroupKFold by raw-train source_group; every held group excluded from fit",
        "rounds": round_summaries,
        "accepted_after_frozen_prediction": len(final_pool) - initial_pool_size,
        "final_pool": len(final_pool), "remaining_unaccepted": len(residual_final),
        "remaining_ids": residual_final,
        "baseline_auto_assignment_rows": baseline_assigned,
        "target_use": "initial pool: 1763 prior program predictions already matched the reference plus 19 participant keep predictions; residual labels were compared only after grouped cross-fit predictions were frozen, and newly accepted matches entered only the next round",
        "all_feature_hashes_match_raw_train": True,
        "artifacts": {"predictions": str(pred_path), "final_pool": str(pool_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
