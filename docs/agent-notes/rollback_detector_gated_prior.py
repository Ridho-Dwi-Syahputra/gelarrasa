"""Restore the verified pre-publication snapshot after detector-prior regression.

This moves the newly published files into an experiment archive. It never
deletes raw data, processed images, notebook outputs, or experiment evidence.
"""
import json
import shutil
from pathlib import Path
from publish_program_refinement import ROOT, NOTES, digest, inside, move, write_json

BACKUP = NOTES/'.runtime/before_bounded_anchor_20261108T090422Z'
REJECTED = NOTES/'.runtime/rejected_detector_gated_prior_20261108T090422Z'

def copy(source, dest):
    source,dest=inside(source),inside(dest)
    assert source.is_file()
    shutil.copy2(source,dest)

def main():
    pub=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())
    assert pub['status']=='passed' and pub['source_frozen_prefix']=='detector_gated_prior'
    assert digest(ROOT/'data/preprocessing_inputs.npz')==pub['input_sha256']
    assert digest(ROOT/'final-submit/DSC2026_Notebook.ipynb')==pub['notebook_sha256']
    assert digest(ROOT/'data/processed/manifest.csv')==pub['manifest_sha256']
    assert not REJECTED.exists()
    old=json.loads((NOTES/'json/frozen_residual_continuation_frozen.json').read_text())
    for name,digest_expected in pub['original_active_sha256'].items():
        source=BACKUP/Path(name).name if name!='data/processed/manifest.csv' else BACKUP/'processed/manifest.csv'
        assert digest(source)==digest_expected,(name,source)
    REJECTED.mkdir(parents=True)
    move(ROOT/'data/processed',REJECTED/'processed')
    move(BACKUP/'processed',ROOT/'data/processed')
    move(ROOT/'final-submit/reports',REJECTED/'reports')
    move(BACKUP/'reports',ROOT/'final-submit/reports')
    for name in ['preprocessing_inputs.npz','DSC2026_Notebook.ipynb']:
        target=ROOT/('data' if name.endswith('.npz') else 'final-submit')/name
        copy(target,REJECTED/name)
        copy(BACKUP/name,target)
    for name in ['execution.json','validation.json','latest.json','notebook_figures.json','feature_recipe.json','policy.json','skills_validation.json','project_contract.json']:
        target=NOTES/name
        if target.exists():copy(target,REJECTED/name)
        if (BACKUP/name).exists():copy(BACKUP/name,target)
    target=NOTES/'json/manual_label_budget_status.json'
    copy(target,REJECTED/'manual_label_budget_status.json')
    copy(BACKUP/'manual_label_budget_status.json',target)
    new={'status':'passed','rollback_of':'detector_gated_prior','rollback_reason':'post-freeze audit disclosed three regressions versus one recovery; previous completed snapshot restored',
         'backup':str(BACKUP.relative_to(ROOT)),'rejected_artifacts':str(REJECTED.relative_to(ROOT)),
         'candidate_sha256':old['prediction_sha256'],'manual_unique_source_images':34,
         'manual_class_counts':old['manual_class_counts'],'changed_full_frame_categories':7,
         'cumulative_changes_vs_initial_program':old['cumulative_changes_vs_initial_program'],
         'source_frozen_prefix':'frozen_residual_continuation','test_snapshot_unchanged':True,'raw_sources_unchanged':True,
         'input_sha256':digest(ROOT/'data/preprocessing_inputs.npz'),
         'notebook_sha256':digest(ROOT/'final-submit/DSC2026_Notebook.ipynb'),
         'manifest_sha256':digest(ROOT/'data/processed/manifest.csv')}
    write_json(NOTES/'json/bounded_anchor_publication.json',new)
    write_json(NOTES/'json/detector_gated_prior_postfreeze_rejection.json',
               {'decision':'rollback_after_audit','before_differences':17,'after_differences':19,
                'recovered':['0964'],'regressed':['0597','0969','1176'],'review_is_not_fit_or_independent_validation':True,
                'archive':str(REJECTED.relative_to(ROOT))})
    print(json.dumps(new,indent=2))

if __name__=='__main__':main()
