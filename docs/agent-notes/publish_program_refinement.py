"""Publish frozen automatic train assignments, with a reversible clean rebuild."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and path != ROOT
    return path


def move(source, destination):
    source, destination = inside(source), inside(destination)
    assert source.exists() and not destination.exists()
    shutil.move(str(source), str(destination))


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def main():
    frozen = json.loads((NOTES / 'json/iterative_program_frozen.json').read_text())
    candidate_path = NOTES / 'table-csv/iterative_program_selected.csv'
    assert digest(candidate_path) == frozen['prediction_sha256']
    for name, expected in frozen['protected_sha256'].items():
        assert digest(ROOT / name) == expected, f'Active input changed: {name}'
    candidate = pd.read_csv(candidate_path, dtype={'image_id': str}).set_index('key')
    assert len(candidate) == 1906 and candidate.index.is_unique and candidate.index.str.startswith('train:').all()
    bundle_path = ROOT / 'data/preprocessing_inputs.npz'
    with np.load(bundle_path, allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    keys = data['keys'].astype(str)
    index = {key: i for i, key in enumerate(keys)}
    positions = np.array([index[key] for key in candidate.index])
    assert np.array_equal(data['predicted_class'][positions], candidate.baseline_class.to_numpy())
    assert np.array_equal(data['source_sha256'][positions], candidate.source_sha256.to_numpy())
    test_mask = np.char.startswith(keys, 'test:')
    test_original = {name: value[test_mask].copy() for name, value in data.items()
                     if value.ndim == 1 and len(value) == len(keys)}
    changed = candidate.predicted_class.ne(candidate.baseline_class).to_numpy()
    changed_positions = positions[changed]
    data['predicted_class'] = data['predicted_class'].astype('U32')
    data['predicted_class'][positions] = candidate.predicted_class.to_numpy(str)
    # A conservative lower bound on the four-of-five vote, not calibrated confidence.
    data['confidence'][changed_positions] = .8
    data['margin'][changed_positions] = 0
    data['needs_review'][changed_positions] = True
    data['category_source'] = np.full(len(keys), 'historical_routing_snapshot', dtype='U48')
    data['category_source'][positions] = 'iterative_program_refinement_snapshot'
    data['category_score_kind'] = np.full(len(keys), 'historical_uncalibrated_program_score', dtype='U48')
    data['category_score_kind'][changed_positions] = 'minimum_family_vote_share'
    references_before = data['reference_labels'].copy()
    data['reference_labels'] = data['reference_labels'].astype('U32')
    for i, key in enumerate(data['reference_keys']):
        if key in candidate.index:
            data['reference_labels'][i] = candidate.at[key, 'predicted_class']
    assert all(np.array_equal(data[name][test_mask], value) for name, value in test_original.items())
    metadata = json.loads(str(data['metadata'].item()))
    metadata['version'] = 2
    metadata['prior_assignment_method'] = metadata['assignment_method']
    metadata['assignment_method'] = 'iterative_crossfit_program_denoising_v1'
    metadata['assignment_sha256'] = frozen['prediction_sha256']
    metadata['refinement'] = {
        'full_frame_train_rows': 1906,
        'changes_vs_historical_program': int(changed.sum()),
        'five_feature_families': ['DINOv2_global', 'DINOv2_local', 'DINOv3_global', 'DINOv3_foreground', 'DINOv3_with_eight_class_CNN'],
        'minimum_family_agreement': 4,
        'source_group_folds': 5,
        'selected_final_round': frozen['selected_final_round'],
        'stop_changed_fraction': .01,
        'stop_reason': frozen['stop_reason'],
        'strong_automatic_anchors': frozen['fixed_automatic_anchors'],
        'human_labels_consumed': 0,
        'test_rows_in_fit': 0,
        'target_origin': 'historical program categories, never independent ground truth',
        'semantic_accuracy': None,
        'refinement_repeated_by_preprocessing_run': False,
        'DINOv3_repository': 'timm/vit_small_patch16_dinov3.lvd1689m',
        'DINOv3_revision': '3bf4720a82ec2066db88137180ff1f83a675cef0',
        'eight_class_CNN_repository': 'C0mpl1cated3/trash_classifier_v2',
        'eight_class_CNN_revision': 'ac54e811e3a5d505b57bf6dc1495ed5cd695c036',
    }
    data['metadata'] = np.array(json.dumps(metadata))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = inside(NOTES / '.runtime' / f'before_program_refinement_{stamp}')
    backup.mkdir(parents=True)
    notebook = ROOT / 'final-submit/DSC2026_Notebook.ipynb'
    shutil.copy2(bundle_path, backup / 'preprocessing_inputs.npz')
    shutil.copy2(notebook, backup / notebook.name)
    proof_names = ['execution.json', 'validation.json', 'latest.json', 'notebook_figures.json', 'feature_recipe.json', 'policy.json', 'skills_validation.json']
    for name in proof_names:
        path = NOTES / name
        if path.exists(): shutil.copy2(path, backup / name)
    move(ROOT / 'data/processed', backup / 'processed')
    move(ROOT / 'final-submit/reports', backup / 'reports')
    report = {
        'status': 'rebuilding', 'backup': str(backup.relative_to(ROOT)),
        'manual_labels_consumed': 0, 'test_rows_in_fit': 0,
        'candidate_sha256': frozen['prediction_sha256'],
        'changes_from_initial_program': int(changed.sum()),
        'reference_categories_updated': int((references_before != data['reference_labels']).sum()),
        'test_snapshot_unchanged': True, 'original_active_sha256': frozen['protected_sha256'],
    }
    write_json(NOTES / 'json/program_refinement_publication.json', report)
    try:
        np.savez_compressed(bundle_path, **data)
        assert not (ROOT / 'data/processed').exists()
        with (backup / 'notebook_execution.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(ROOT / 'docs/skills/tools/build_notebook.py')],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (backup / 'validation.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(ROOT / 'docs/skills/tools/validate_collection.py'), '--verify-images'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report.update(status='passed', input_sha256=digest(bundle_path), notebook_sha256=digest(notebook),
                      manifest_sha256=digest(ROOT / 'data/processed/manifest.csv'))
    except BaseException as exc:
        # Preserve failed artifacts, then restore the complete previous working state.
        for path, name in [(ROOT / 'data/processed', 'failed_processed'), (ROOT / 'final-submit/reports', 'failed_reports')]:
            if path.exists(): move(path, backup / name)
        move(backup / 'processed', ROOT / 'data/processed')
        move(backup / 'reports', ROOT / 'final-submit/reports')
        shutil.copy2(backup / 'preprocessing_inputs.npz', bundle_path)
        shutil.copy2(backup / notebook.name, notebook)
        for name in proof_names:
            if (backup / name).exists(): shutil.copy2(backup / name, NOTES / name)
        report.update(status='rolled_back', error=repr(exc))
        write_json(NOTES / 'json/program_refinement_publication.json', report)
        raise
    write_json(NOTES / 'json/program_refinement_publication.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
