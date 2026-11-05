"""Export already-reviewed manual visual labels; this script does not predict images."""
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    review = Path(__file__).resolve().parent
    project = review.parent.parent
    sample_path = project / "data" / "SampleSubmission_DSC.csv"
    output = project / "docs" / "submission_99.csv"
    decisions_path = review / "decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    session = json.loads((review / "session.json").read_text(encoding="utf-8"))
    with sample_path.open(encoding="utf-8-sig", newline="") as f:
        sample_reader = csv.DictReader(f)
        header = sample_reader.fieldnames
        sample = list(sample_reader)
    assert header == ["image_id", "waste_category"]
    expected_ids = [r["image_id"] for r in sample]
    assert len(expected_ids) == len(set(expected_ids)) == 480
    by_id = {r["image_id"]: r for r in decisions}
    assert len(by_id) == len(decisions) == 480
    assert set(by_id) == set(expected_ids)
    assert all(r["seen"] and r["class_gallery_reviewed"] for r in decisions)
    assert all(r["visual_evidence"].strip() for r in decisions)
    allowed = set(session["allowed_classes"])
    assert all(r["waste_category"] in allowed for r in decisions)
    assert all(Path(r["path"]).is_file() for r in decisions)
    assert all(Path(r["path"]).stem == r["image_id"] for r in decisions)
    train_overview = json.loads((review / "train_overview_manifest.json").read_text(encoding="utf-8"))
    gallery = json.loads((review / "test_class_review_manifest.json").read_text(encoding="utf-8"))
    assert len(train_overview) == 54 and all(r["viewed"] for r in train_overview)
    assert len(gallery) == 17 and all(r["viewed"] for r in gallery)
    train_ids = [i for sheet in train_overview for i in sheet["ids"]]
    gallery_ids = [i for sheet in gallery for i in sheet["ids"]]
    assert len(train_ids) == len(set(train_ids)) == 1920
    assert set(train_ids) == {f"{i:04d}" for i in range(1, 1921)}
    assert len(gallery_ids) == len(set(gallery_ids)) == 480
    assert set(gallery_ids) == set(expected_ids)
    assert all(by_id[i]["waste_category"] == sheet["label"] for sheet in gallery for i in sheet["ids"])

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows((i, by_id[i]["waste_category"]) for i in expected_ids)
    contents = buffer.getvalue().encode("utf-8")
    if output.exists():
        assert output.read_bytes() == contents, "An existing different submission will not be overwritten."
    else:
        with output.open("xb") as f:
            f.write(contents)

    with output.open(encoding="utf-8", newline="") as f:
        actual_reader = csv.DictReader(f)
        assert actual_reader.fieldnames == header
        actual = list(actual_reader)
    assert [r["image_id"] for r in actual] == expected_ids
    assert len(actual) == 480
    assert all(set(r) == set(header) and r["waste_category"] in allowed for r in actual)
    assert all(r["image_id"] == f"{int(r['image_id']):03d}" for r in actual)

    fields = ["image_id", "waste_category", "confidence", "seen", "class_gallery_reviewed", "visual_evidence", "targeted_original_review"]
    with (review / "label_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in expected_ids:
            r = by_id[i]
            writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), dict) else r.get(k, "") for k in fields})

    first = json.loads((review / "first_pass_decisions.json").read_text(encoding="utf-8"))
    first_by_id = {r["image_id"]: r for r in first}
    changes = [{"image_id": i, "before": first_by_id[i]["waste_category"], "after": by_id[i]["waste_category"], "review": by_id[i].get("second_review")} for i in expected_ids if first_by_id[i]["waste_category"] != by_id[i]["waste_category"]]
    (review / "label_changes.json").write_text(json.dumps(changes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = [{"image_id": i, "path": by_id[i]["path"], "sha256": hashlib.sha256(Path(by_id[i]["path"]).read_bytes()).hexdigest()} for i in expected_ids]
    (review / "test_source_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    validation = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "rows": len(actual), "header": header,
        "sample_ids_and_order_match": True, "three_digit_ids_preserved": True,
        "only_allowed_labels": True, "no_missing_predictions": True,
        "test_originals_individually_viewed": sum(r["seen"] for r in decisions),
        "test_class_gallery_reviewed": sum(r["class_gallery_reviewed"] for r in decisions),
        "targeted_original_rechecks_recorded": sum("targeted_original_review" in r for r in decisions),
        "train_images_in_viewed_overviews": len(train_ids),
        "train_overview_sheets_viewed": len(train_overview),
        "class_counts": dict(sorted(Counter(r["waste_category"] for r in actual).items())),
        "subjective_confidence_counts": dict(Counter(r["confidence"] for r in decisions)),
        "changes_after_first_pass": len(changes),
        "method": "Assistant direct visual classification, with manual second review. No computer-vision model was trained and no external prediction API was called.",
        "sample_labels_used_as_predictions": False,
        "known_test_accuracy": None,
        "limitations": "No train label file was found locally. Train images supplied visual context only. Confidence levels are subjective, not calibrated probabilities. Mixed-material and low-resolution cases remain uncertain."
    }
    (review / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
