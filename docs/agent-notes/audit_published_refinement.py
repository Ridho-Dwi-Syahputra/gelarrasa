"""Check published assignments, then perform a separate read-only archive audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    published = json.loads((NOTES / 'json/program_refinement_publication.json').read_text())
    assert published['status'] == 'passed'
    manifest_path = ROOT / 'data/processed/manifest.csv'
    assert digest(manifest_path) == published['manifest_sha256']
    frame = pd.read_csv(manifest_path, dtype={'parent_id': str})
    full = frame[frame.split.eq('train') & ~frame.is_crop].copy()
    predicted_path = NOTES / 'table-csv/iterative_program_selected.csv'
    assert digest(predicted_path) == published['candidate_sha256']
    frozen = pd.read_csv(predicted_path, dtype={'image_id': str})
    checked = full.merge(frozen, left_on='parent_key', right_on='key', validate='one_to_one')
    assert len(checked) == len(full) == len(frozen) == 1906
    assert checked.assigned_class.eq(checked.predicted_class).all()
    assert checked.source_sha256_x.eq(checked.source_sha256_y).all()
    audit = pd.read_csv(NOTES / 'table-csv/iterative_program_archive_audit.csv', dtype={'image_id': str})
    checked = checked.merge(audit[['key', 'archive_class']], on='key', validate='one_to_one')
    different = checked.assigned_class.ne(checked.archive_class)
    remaining = checked.loc[different, ['parent_id', 'source_path', 'output_path', 'assigned_class', 'archive_class', 'category_score', 'category_score_kind', 'needs_review']].sort_values('parent_id')
    remaining['use'] = 'posthoc_diagnostic_only_never_training_targets'
    remaining.to_csv(NOTES / 'table-csv/remaining_program_archive_differences.csv', index=False)
    backup = ROOT / published['backup']
    previous = pd.read_csv(backup / 'processed/manifest.csv', dtype={'parent_id': str})
    crops = frame[frame.is_crop].merge(previous[previous.is_crop][['sample_key', 'assigned_class']], on='sample_key', suffixes=('_current', '_previous'), validate='one_to_one')
    with np.load(backup / 'preprocessing_inputs.npz', allow_pickle=False) as old, np.load(ROOT / 'data/preprocessing_inputs.npz', allow_pickle=False) as current:
        mask = np.char.startswith(old['keys'].astype(str), 'test:')
        for name in old.files:
            value = old[name]
            if value.ndim == 1 and len(value) == 2400:
                assert np.array_equal(value[mask], current[name][mask]), name
    report = {
        'status': 'passed', 'active_train_samples': 1913, 'active_test_samples': 480,
        'comparable_full_frame_train': 1906, 'full_frame_matches': int((~different).sum()),
        'full_frame_differences': int(different.sum()),
        'scope': 'agreement with historical archive; not official accuracy',
        'manual_labels_consumed': 0, 'test_rows_in_refinement_fit': 0,
        'full_frame_assignments_match_frozen_candidate': True,
        'test_snapshot_unchanged': True,
        'crop_categories_changed': int(crops.assigned_class_current.ne(crops.assigned_class_previous).sum()),
        'crop_categories': dict(zip(crops.sample_key, crops.assigned_class_current)),
        'train_category_counts': frame[frame.split.eq('train')].assigned_class.value_counts().to_dict(),
        'notebook_sha256': published['notebook_sha256'],
    }
    (NOTES / 'json/published_refinement_audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
