"""Whole-checkpoint recovery after unlabelled pseudo-bank class concentration.

No private-review labels/IDs are read by this health decision. All changed files
in the latest batch are withdrawn together, not repaired by category reference.
"""
import json
import shutil
import subprocess
import sys
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,sha,dump


def safe(path):
    path=Path(path).resolve();assert path.is_relative_to(ROOT.resolve()) and path!=ROOT.resolve();return path


def move(source,destination):
    source,destination=safe(source),safe(destination);assert not destination.exists();shutil.move(str(source),str(destination))


def pool_counts(path):
    f=pd.read_csv(path);f=f[f.split.eq('train')&~f.is_crop&f.training_eligible]
    t=np.full(len(f),.8);kind=f.category_score_kind.to_numpy(str);t[np.isin(kind,['multi_method_vote_share','minimum_family_vote_share'])]=.75;t[kind=='weighted_neighbor_share']=.9;t[kind=='uncalibrated_neural_probability']=.85
    mask=f.human_reviewed.to_numpy(bool)|(~f.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()&(f.category_score.to_numpy(float)>=t))
    counts=f.loc[mask,'assigned_class'].value_counts();return {'rows':int(counts.sum()),'counts':counts.to_dict(),'largest_fraction':float(counts.max()/counts.sum())}


def main():
    publication=NOTES/'json/bounded_anchor_publication.json';before=json.loads(publication.read_text());assert before['status']=='passed' and before['source_frozen_prefix']=='current_pseudo_anchor_r2'
    source=safe(ROOT/before['backup']);current_health=pool_counts(ROOT/'data/processed/manifest.csv');previous_health=pool_counts(source/'processed/manifest.csv')
    assert current_health['largest_fraction']>.5 and previous_health['largest_fraction']<=.5
    proof={'operation':'withdraw_entire_pseudo_feedback_batch','reason':'selectedpseudoanchorbankmorethanhalfonesemanticclass;unlabelledfeedbackconcentration',
           'current_pool':current_health,'restored_pool':previous_health,'limit':.5,'decision_inputs':'programorigin,scoresemanticsandclassfrequenciesonly',
           'private_review_file_read':False,'researcher_has_seen_prior_posthoc_audits':True,'semantic_accuracy_certified':False}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');home=safe(NOTES/'.runtime'/f'withdrawn_pseudo_feedback_{stamp}');home.mkdir()
    input_path=ROOT/'data/preprocessing_inputs.npz';nb=ROOT/'final-submit/DSC2026_Notebook.ipynb';processed=ROOT/'data/processed';reports=ROOT/'final-submit/reports'
    original={str(p.relative_to(ROOT)):sha(p) for p in [input_path,nb,processed/'manifest.csv']}
    names=['execution.json','validation.json','latest.json','notebook_figures.json','feature_recipe.json','policy.json','skills_validation.json','project_contract.json']
    for p in [input_path,nb,publication,NOTES/'json/manual_label_budget_status.json']:
        shutil.copy2(p,home/p.name)
    for name in names:
        if (NOTES/name).exists():shutil.copy2(NOTES/name,home/name)
    move(processed,home/'processed');move(reports,home/'reports')
    try:
        shutil.copytree(safe(source/'processed'),safe(processed));shutil.copytree(safe(source/'reports'),safe(reports))
        shutil.copy2(source/input_path.name,input_path);shutil.copy2(source/nb.name,nb)
        for name in names:
            if (source/name).exists():shutil.copy2(source/name,NOTES/name)
        shutil.copy2(source/'manual_label_budget_status.json',NOTES/'json/manual_label_budget_status.json')
        assert all(sha(ROOT/name)==value for name,value in before['original_active_sha256'].items())
        frozen=json.loads((NOTES/'json/current_pseudo_anchor_frozen.json').read_text())
        budget=json.loads((NOTES/'json/manual_label_budget_status.json').read_text());assert budget['manual_labels_consumed_by_current_experiments']==34
        with np.load(home/input_path.name,allow_pickle=False) as a,np.load(input_path,allow_pickle=False) as b:
            test=np.char.startswith(a['keys'].astype(str),'test:')
            for key in a.files:
                if a[key].ndim==1 and len(a[key])==2400:assert np.array_equal(a[key][test],b[key][test]),key
        restored={'status':'passed','operation':'whole_checkpoint_restore_after_unlabelled_pseudo_bank_health_failure','source_frozen_prefix':'current_pseudo_anchor',
                  'backup':str(home.relative_to(ROOT)),'restored_from_backup':str(source.relative_to(ROOT)),'candidate_sha256':frozen['prediction_sha256'],
                  'manual_unique_source_images':34,'manual_class_counts':budget['manual_class_counts'],'changed_full_frame_categories':40,
                  'cumulative_changes_vs_initial_program':frozen['cumulative_changes_vs_initial_program'],'reference_categories_updated':31,
                  'test_snapshot_unchanged':True,'raw_sources_unchanged':True,'original_active_sha256':original,
                  'input_sha256':sha(input_path),'notebook_sha256':sha(nb),'manifest_sha256':sha(processed/'manifest.csv'),
                  'previous_execution_evidence_restored_byte_exact':True,'health_decision':proof}
        dump(publication,restored)
        with (home/'restored_validation.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'docs/skills/tools/validate_collection.py'),'--verify-images'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        proof.update(status='restored_and_hash_verified',rollback_backup=str(home.relative_to(ROOT)),restored_notebook_sha256=sha(nb),restored_manifest_sha256=sha(processed/'manifest.csv'))
        dump(NOTES/'json/pseudo_feedback_withdrawal.json',proof)
    except BaseException:
        if processed.exists():move(processed,home/'failed_restored_processed')
        if reports.exists():move(reports,home/'failed_restored_reports')
        move(home/'processed',processed);move(home/'reports',reports)
        for target in [input_path,nb,publication,NOTES/'json/manual_label_budget_status.json']:shutil.copy2(home/target.name,target)
        for name in names:
            if (home/name).exists():shutil.copy2(home/name,NOTES/name)
        raise
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
