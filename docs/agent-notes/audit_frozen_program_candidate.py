"""Read-only comparison after the program predictions have been frozen."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    args = parser.parse_args()
    frozen = json.loads((NOTES / f'json/{args.prefix}_frozen.json').read_text())
    source = NOTES / f'table-csv/{args.prefix}_selected.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen['prediction_sha256']
    frame = pd.read_csv(source, dtype={'image_id': str})
    reference = {}
    for path in (ROOT / 'data/Dataset DSC GelarRasa 2026/train/label').glob('*/*'):
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            assert path.stem not in reference
            reference[path.stem] = path.parent.name
    assert len(frame) == 1906 and frame.image_id.is_unique
    frame['archive_class'] = frame.image_id.map(reference)
    assert frame.archive_class.notna().all()
    old = frame.baseline_class.ne(frame.archive_class)
    new = frame.predicted_class.ne(frame.archive_class)
    changed = frame.baseline_class.ne(frame.predicted_class)
    report = {
        'scope': 'posthoc agreement with historical archive, not semantic accuracy',
        'rows': len(frame),
        'baseline_differences': int(old.sum()),
        'candidate_differences': int(new.sum()),
        'old_disagreements_resolved': int((old & ~new).sum()),
        'new_disagreements': int((~old & new).sum()),
        'changed_wrong_to_other_wrong': int((old & new & changed).sum()),
        'changes_vs_baseline': int(changed.sum()),
        'manual_labels_consumed': 0,
        'frozen_prediction_sha256': frozen['prediction_sha256'],
        'active_data_modified': False,
    }
    frame['differs_from_archive'] = new
    frame.sort_values('image_id').to_csv(NOTES / f'table-csv/{args.prefix}_archive_audit.csv', index=False)
    (NOTES / f'json/{args.prefix}_archive_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
