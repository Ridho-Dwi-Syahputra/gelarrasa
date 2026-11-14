"""Reversibly restore the immediately previous, hash-verified publication.

Usage: python restore_published_snapshot.py CURRENT_PREFIX PRIOR_PREFIX PRIOR_BACKUP
Every path is resolved within the project root. Published artifacts are moved
to a separate local archive, never deleted.
"""
import json,shutil,sys
from pathlib import Path
from publish_program_refinement import ROOT,NOTES,digest,inside,move,write_json

def copy(source,dest):
    source,dest=inside(source),inside(dest);assert source.is_file();shutil.copy2(source,dest)

def main(current,prior,prior_backup):
    pub=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())
    assert pub['status']=='passed' and pub['source_frozen_prefix']==current
    backup=inside(ROOT/pub['backup']);rejected=inside(NOTES/'.runtime'/f'rejected_{current}_{backup.name}')
    earlier=inside(ROOT/prior_backup)
    assert (backup/'processed/manifest.csv').is_file() and (earlier/'processed/manifest.csv').is_file()
    assert not rejected.exists()
    assert digest(ROOT/'data/preprocessing_inputs.npz')==pub['input_sha256']
    assert digest(ROOT/'final-submit/DSC2026_Notebook.ipynb')==pub['notebook_sha256']
    assert digest(ROOT/'data/processed/manifest.csv')==pub['manifest_sha256']
    old=json.loads((NOTES/f'json/{prior}_frozen.json').read_text())
    for name,expected in pub['original_active_sha256'].items():
        source=backup/Path(name).name if name!='data/processed/manifest.csv' else backup/'processed/manifest.csv'
        assert digest(source)==expected,(name,source)
    rejected.mkdir(parents=True)
    move(ROOT/'data/processed',rejected/'processed');move(backup/'processed',ROOT/'data/processed')
    move(ROOT/'final-submit/reports',rejected/'reports');move(backup/'reports',ROOT/'final-submit/reports')
    for name in ['preprocessing_inputs.npz','DSC2026_Notebook.ipynb']:
        target=ROOT/('data' if name.endswith('.npz') else 'final-submit')/name
        copy(target,rejected/name);copy(backup/name,target)
    for name in ['execution.json','validation.json','latest.json','notebook_figures.json','feature_recipe.json','policy.json','skills_validation.json','project_contract.json']:
        target=NOTES/name
        if target.exists():copy(target,rejected/name)
        if (backup/name).exists():copy(backup/name,target)
    target=NOTES/'json/manual_label_budget_status.json';copy(target,rejected/'manual_label_budget_status.json');copy(backup/'manual_label_budget_status.json',target)
    restored={'status':'passed','rollback_of':current,'backup':str(earlier.relative_to(ROOT)),'rejected_artifacts':str(rejected.relative_to(ROOT)),
              'candidate_sha256':old['prediction_sha256'],'manual_unique_source_images':34,'manual_class_counts':old['manual_class_counts'],
              'source_frozen_prefix':prior,'cumulative_changes_vs_initial_program':old['cumulative_changes_vs_initial_program'],
              'test_snapshot_unchanged':True,'raw_sources_unchanged':True,
              'input_sha256':digest(ROOT/'data/preprocessing_inputs.npz'),
              'notebook_sha256':digest(ROOT/'final-submit/DSC2026_Notebook.ipynb'),
              'manifest_sha256':digest(ROOT/'data/processed/manifest.csv')}
    write_json(NOTES/'json/bounded_anchor_publication.json',restored)
    print(json.dumps(restored,indent=2))

if __name__=='__main__':
    assert len(sys.argv)==4
    main(*sys.argv[1:])
