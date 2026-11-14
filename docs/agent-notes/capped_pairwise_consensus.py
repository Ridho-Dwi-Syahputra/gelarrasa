"""Apply a fixed small cap to already frozen train-only specialist proposals.

No review labels are read. All scores, thresholds, and source models come from
the previously frozen pairwise-detector candidate.
"""
import datetime
import json
import sys

import numpy as np
import pandas as pd

from fewshot_train_only_probe import ROOT, NOTES, sha, dump


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    assert cap in (5, 10)
    prefix = "capped_pairwise_consensus" if cap == 5 else "capped_pairwise_consensus_10"
    source = "pairwise_detector_consensus"
    protected = {p: sha(ROOT / p) for p in ["data/processed/manifest.csv",
                                          "data/preprocessing_inputs.npz",
                                          "final-submit/DSC2026_Notebook.ipynb"]}
    plan = {"method": f"frozen_train_only_pairwise_detector_top{cap}_v1",
            "source_candidate_sha256": sha(NOTES / f"table-csv/{source}_selected.csv"),
            "source_frozen_sha256": sha(NOTES / f"json/{source}_frozen.json"),
            "selection": f"top{cap} by minimum of matching pair, all-pair, detector source score; stable source key tie-break",
            "cap": cap, "reason": "small automatic batch to reduce category drift while preserving most active assignments",
            "train_only": True, "review_archive_read": False, "new_manual_labels": 0,
            "protected_sha256": protected, "source_code_sha256": sha(__file__),
            "evaluation_limit": "cap selected after observing previous 39-proposal audit; adaptive and not independent validation"}
    dump(NOTES / f"json/{prefix}_plan.json", plan)
    original = pd.read_csv(NOTES / f"table-csv/{source}_selected.csv", dtype={"image_id": str}).set_index("key")
    evidence = pd.read_csv(NOTES / f"table-csv/{source}_evidence.csv").set_index("key")
    assert original.index.is_unique and len(original) == 1906
    assert set(evidence.index).issubset(original.index)
    assert len(evidence) == 1904
    chosen = evidence.loc[evidence.accepted]
    assert len(chosen) == 39
    score = chosen[["matching_pair_score", "minimum_all_pair_score", "detector_plastic_score"]].min(axis=1)
    ranked = score.sort_values(ascending=False, kind="stable")
    take = ranked.head(cap).index
    base = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"parent_id": str})
    full = base[base.split.eq("train") & ~base.is_crop].sort_values("parent_id")
    result = full[["parent_key", "parent_id", "source_path", "source_sha256"]].rename(
        columns={"parent_key": "key", "parent_id": "image_id"}).copy()
    for dest, col in [("baseline_class", "assigned_class"), ("predicted_class", "assigned_class"),
                      ("category_source", "label_source"), ("category_score", "category_score"),
                      ("category_score_kind", "category_score_kind"),
                      ("human_reviewed", "human_reviewed"), ("needs_review", "needs_review")]:
        result[dest] = full[col]
    result = result.set_index("key")
    assert set(take).issubset(result.index)
    for col in ["predicted_class", "category_source", "category_score", "category_score_kind", "needs_review"]:
        result.loc[take, col] = original.loc[take, col]
    assert (result.predicted_class != result.baseline_class).sum() == cap
    dest = NOTES / f"table-csv/{prefix}_selected.csv"
    result.reset_index().to_csv(dest, index=False)
    pd.DataFrame({"key": evidence.index, "candidate": evidence.accepted,
                  "priority": evidence[["matching_pair_score", "minimum_all_pair_score", "detector_plastic_score"]].min(axis=1),
                  "selected": evidence.index.isin(take)}).to_csv(NOTES / f"table-csv/{prefix}_evidence.csv", index=False)
    source_report = json.loads((NOTES / f"json/{source}_frozen.json").read_text())
    active = json.loads((NOTES / "json/bounded_anchor_publication.json").read_text())
    assert active["status"] == "passed" and active["source_frozen_prefix"] == "frozen_residual_continuation"
    old = pd.read_csv(NOTES / "table-csv/bounded_consensus_selected.csv").set_index("key")
    source_report.update(status="frozen_before_small_batch_audit", method=plan["method"],
                         prediction_sha256=sha(dest), protected_sha256=protected,
                         primary_plan_sha256=sha(NOTES / f"json/{prefix}_plan.json"),
                         changed_rows_vs_baseline=cap,
                         cumulative_changes_vs_initial_program=int(result.predicted_class.ne(
                             old.loc[result.index, "baseline_class"]).sum()),
                         source_counts=result.category_source.value_counts().to_dict(),
                         source_code_sha256=sha(__file__),
                         validation_limit=plan["evaluation_limit"])
    dump(NOTES / f"json/{prefix}_frozen.json", source_report)
    dump(NOTES / f"json/{prefix}_publication_decision.json", {
        "decision": "publish", "accepted_changes": cap, "prediction_sha256": sha(dest),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    print(json.dumps({"prefix": prefix, "changes": cap,
                      "source_oof_matches": source_report["human_leave_source_group_out_matches"],
                      "source_oof_macro": source_report["development_balanced_recall"]}, indent=2))


if __name__ == "__main__":
    main()
