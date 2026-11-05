"""Verify the completed notebook, source provenance and preserved raw inputs."""
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/reproducibility'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    notebook_path = ROOT / 'notebook/01_EDA_Preprocessing_Clustering.ipynb'
    run = json.loads((REPORT / 'notebook_v2_execution.json').read_text())
    kernel = json.loads((REPORT / 'notebook_v2_in_kernel_checks.json').read_text())
    assert run['status'] == 'passed' and run['errors'] == []
    assert run['code_cells'] == run['executed_cells'] == 35
    assert run['notebook_sha256'] == sha(notebook_path)
    nb = nbformat.read(notebook_path, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in cells] == list(range(1, 36))
    assert not [o for c in cells for o in c.outputs if o.output_type == 'error']
    source = '\n'.join(c.source for c in cells)
    assert hashlib.sha256(source.encode()).hexdigest() == run['notebook_code_sha256']
    assert nb.metadata.kernelspec.name == run['kernel'] == 'gelar-rasa-venv'
    assert Path(run['interpreter']).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    assert kernel['stage1_complete'] and kernel['stage2_complete']
    assert kernel['head_probability_rows_verified'] == 2400 and kernel['group_overlap'] == 0
    records = kernel['stage2_runtime_records']
    assert len(records) == 6
    for record in records:
        assert record['status'] == 'completed'
        assert sha(ROOT / 'eksperimen/skrip' / record['script']) == record['source_sha256']
    assignment = ROOT / 'output/groups/stage2_neighbor_v1/assignments.csv'
    assert sha(assignment) == kernel['assignment_sha256']
    materialization = json.loads((ROOT / 'data/processed/stage2_neighbor_v1/provenance.json').read_text())
    assert materialization['all_copies_hash_verified']
    assert materialization['assignment_sha256'] == sha(assignment)
    assert materialization['script_sha256'] == sha(ROOT / 'eksperimen/skrip/materialize_stage2_clusters.py')
    backup = REPORT / 'backups/20261105_before_notebook_v2/previous_analysis_artifacts.zip'
    with zipfile.ZipFile(backup) as archive:
        original = pd.read_csv(io.BytesIO(archive.read('output/manifests/stage1_v1/image_profile.csv')), dtype={'image_id': str})
    current = pd.read_csv(ROOT / 'output/manifests/stage1_v1/image_profile.csv', dtype={'image_id': str})
    assert len(original) == len(current) == 2400 and original.key.is_unique and current.key.is_unique
    columns = ['source_path', 'source_sha256']
    pd.testing.assert_frame_equal(original.set_index('key')[columns].sort_index(), current.set_index('key')[columns].sort_index())

    def verify_raw(row):
        path = ROOT / 'data' / row['source_path']
        assert sha(path) == row['source_sha256'], str(path)
        return 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_count = sum(pool.map(verify_raw, original.to_dict('records')))
    old = json.loads((ROOT / 'reports/errors/stage2_20261105/diagnostics/audit_summary.json').read_text())
    preserved = ['docs\\secret\\submission_.csv', 'data\\processed\\stage1_named_review\\review_alias_mapping.csv']
    for relative in preserved:
        assert sha(ROOT / relative) == old['source_hashes_before_and_after'][relative]
    plan = json.loads((ROOT / 'reports/errors/stage2_20261105/representation_plan.json').read_text())
    contract = 'configs\\data_contract.json'
    assert sha(ROOT / contract) == plan['frozen_inputs'][contract]
    results = dict(status='passed', checked_at_wib=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
                   executed_code_cells=35, cell_errors=0, stage2_module_executions_verified=len(records),
                   raw_hashes_verified_against_before_run_backup=raw_count,
                   folder_copy_hash_verification='2400 copies checked by materializer during this Run All',
                   in_kernel_head_rows_reproduced=2400, fit_holdout_group_overlap=0,
                   archives_and_active_contract_preserved=True,
                   notebook_sha256=sha(notebook_path), category_accuracy=None,
                   scope='Notebook execution, provenance and integrity; semantic category Accuracy is unmeasured.')
    (REPORT / 'notebook_v2_delivery_validation.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
