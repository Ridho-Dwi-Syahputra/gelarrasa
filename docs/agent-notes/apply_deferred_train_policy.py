"""Apply participant hold decision without creating categories or deleting sources."""
from datetime import datetime,timezone
import json
import shutil
import subprocess
import sys
import numpy as np
from publish_program_refinement import ROOT,NOTES,digest,inside,move,write_json


def main():
    publication=NOTES/'json/bounded_anchor_publication.json'
    previous=json.loads(publication.read_text());assert previous['status']=='passed'
    input_path=ROOT/'data/preprocessing_inputs.npz';notebook=ROOT/'final-submit/DSC2026_Notebook.ipynb'
    assert digest(input_path)==previous['input_sha256'] and digest(notebook)==previous['notebook_sha256']
    with np.load(input_path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    metadata=json.loads(str(data['metadata'].item()))
    metadata['deferred_train_keys']=['train:1662','train:1885']
    metadata['deferred_train_reason']='participant decision: hold unresolved category, not human anchors or final classifier targets'
    metadata['deferred_policy_applied_after_refinement_fit']=True
    assert not set(metadata['deferred_train_keys'])&set(data['human_seed_keys'])
    data['metadata']=np.array(json.dumps(metadata))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=inside(NOTES/'.runtime'/f'before_deferred_policy_{stamp}');backup.mkdir(parents=True)
    shutil.copy2(input_path,backup/input_path.name);shutil.copy2(notebook,backup/notebook.name);shutil.copy2(publication,backup/publication.name)
    proof_names=['execution.json','validation.json','latest.json','notebook_figures.json','feature_recipe.json','policy.json','skills_validation.json','project_contract.json']
    for name in proof_names:shutil.copy2(NOTES/name,backup/name)
    move(ROOT/'data/processed',backup/'processed');move(ROOT/'final-submit/reports',backup/'reports')
    report={'status':'rebuilding','backup':str(backup.relative_to(ROOT)),'deferred_train_keys':metadata['deferred_train_keys'],
            'human_training_seed_images':len(data['human_seed_keys']),'new_category_assignments_from_deferral':0,
            'refinement_fit_repeated':False,'crop_references_exclude_deferred':True}
    write_json(NOTES/'json/deferred_train_policy_publication.json',report)
    try:
        np.savez_compressed(input_path,**data)
        contract=json.loads((NOTES/'project_contract.json').read_text());contract['deferred_train_keys']=metadata['deferred_train_keys'];contract['final_classifier_must_filter_training_eligible']=True
        write_json(NOTES/'project_contract.json',contract)
        with (backup/'notebook_execution.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'docs/skills/tools/build_notebook.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (backup/'validation.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'docs/skills/tools/validate_collection.py'),'--verify-images'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report.update(status='passed',input_sha256=digest(input_path),notebook_sha256=digest(notebook),manifest_sha256=digest(ROOT/'data/processed/manifest.csv'))
        updated=previous.copy();updated.update(input_sha256=report['input_sha256'],notebook_sha256=report['notebook_sha256'],manifest_sha256=report['manifest_sha256'],
                                             deferred_train_keys=metadata['deferred_train_keys'],post_publication_policy_backup=str(backup.relative_to(ROOT)))
        write_json(publication,updated)
    except BaseException as exc:
        for path,name in [(ROOT/'data/processed','failed_processed'),(ROOT/'final-submit/reports','failed_reports')]:
            if path.exists():move(path,backup/name)
        move(backup/'processed',ROOT/'data/processed');move(backup/'reports',ROOT/'final-submit/reports')
        shutil.copy2(backup/input_path.name,input_path);shutil.copy2(backup/notebook.name,notebook);shutil.copy2(backup/publication.name,publication)
        for name in proof_names:shutil.copy2(backup/name,NOTES/name)
        report.update(status='rolled_back',error=repr(exc));write_json(NOTES/'json/deferred_train_policy_publication.json',report)
        raise
    write_json(NOTES/'json/deferred_train_policy_publication.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
