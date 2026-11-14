"""Restore the publication pointer after publisher already restored active files."""
import json
import sys

from publish_program_refinement import ROOT, NOTES, digest, write_json


def main():
    failed = json.loads((NOTES / "json/bounded_anchor_publication.json").read_text())
    active = json.loads((NOTES / "json/current_review_status.json").read_text())
    assert failed["status"] == "rolled_back"
    assert active["status"] == "passed"
    for path, expected in failed["original_active_sha256"].items():
        assert digest(ROOT / path) == expected
    previous = active["active_prefix"]
    frozen = json.loads((NOTES / f"json/{previous}_frozen.json").read_text())
    write_json(NOTES / f"json/{failed['source_frozen_prefix']}_initial_validation_failure.json", failed)
    restored = {"status": "passed", "rollback_of_failed_attempt": failed["source_frozen_prefix"],
                "backup": failed["backup"], "source_frozen_prefix": previous,
                "candidate_sha256": frozen["prediction_sha256"],
                "manual_unique_source_images": 34,
                "manual_class_counts": frozen["manual_class_counts"],
                "cumulative_changes_vs_initial_program": frozen["cumulative_changes_vs_initial_program"],
                "test_snapshot_unchanged": True, "raw_sources_unchanged": True,
                "input_sha256": digest(ROOT / "data/preprocessing_inputs.npz"),
                "notebook_sha256": digest(ROOT / "final-submit/DSC2026_Notebook.ipynb"),
                "manifest_sha256": digest(ROOT / "data/processed/manifest.csv")}
    write_json(NOTES / "json/bounded_anchor_publication.json", restored)
    print(json.dumps(restored, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 1
    main()
