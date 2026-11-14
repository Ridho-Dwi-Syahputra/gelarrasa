"""Reversibly publish bounded participant-reviewed seeds and stable program outputs."""
from datetime import datetime, timezone
import json
import hashlib
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from publish_program_refinement import ROOT, NOTES, digest, inside, move, write_json


def main(prefix='bounded_consensus'):
    frozen=json.loads((NOTES/f'json/{prefix}_frozen.json').read_text())
    plan_path=NOTES/f'json/{prefix}_plan.json'
    latest_plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
    expected_plan=frozen.get('primary_plan_sha256',frozen.get('plan_sha256'))
    if expected_plan:assert digest(plan_path)==expected_plan
    acceptance=latest_plan.get('acceptance',latest_plan.get('gate','See frozen method-specific experiment record'))
    family_count=len(frozen.get('qualified_heads') or latest_plan.get('qualified_heads') or latest_plan.get('trusted_banks') or latest_plan.get('banks') or
                     frozen.get('selected_heads') or frozen.get('selected_knn_heads') or frozen.get('families',[]))
    selected=NOTES/f'table-csv/{prefix}_selected.csv'
    assert digest(selected)==frozen['prediction_sha256']
    for name,value in frozen['protected_sha256'].items():assert digest(ROOT/name)==value,name
    result=pd.read_csv(selected,dtype={'image_id':str}).set_index('key')
    seed_path=NOTES/frozen['human_seed_file'];assert digest(seed_path)==frozen['human_seed_sha256']
    seeds=pd.read_csv(seed_path,dtype={'image_id':str});assert seeds.proposed_class.value_counts().le(5).all()
    input_path=ROOT/'data/preprocessing_inputs.npz'
    with np.load(input_path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    index=dict(zip(data['keys'],range(len(data['keys']))));positions=np.array([index[k] for k in result.index])
    assert len(result)==1906 and np.array_equal(data['predicted_class'][positions],result.baseline_class)
    assert np.array_equal(data['source_sha256'][positions],result.source_sha256)
    test_mask=np.char.startswith(data['keys'].astype(str),'test:')
    before_test={k:v[test_mask].copy() for k,v in data.items() if v.ndim==1 and len(v)==2400}
    changed=result.predicted_class.ne(result.baseline_class).to_numpy();human=result.human_reviewed.to_numpy()
    data['predicted_class']=data['predicted_class'].astype('U32');data['predicted_class'][positions]=result.predicted_class.to_numpy(str)
    data['confidence'][positions]=result.category_score.to_numpy('float32')
    data['margin'][positions[changed]]=0
    data['needs_review'][positions[changed|human]]=~human[changed|human]
    for name,column in [('category_source','category_source'),('category_score_kind','category_score_kind')]:
        data[name]=data[name].astype('U64');data[name][positions]=result[column].to_numpy(str)
    data['human_seed_keys']=np.array(['train:'+k for k in seeds.image_id],dtype='U16')
    data['human_seed_labels']=seeds.proposed_class.to_numpy(str)
    data['human_seed_source_sha256']=np.array([data['source_sha256'][index[k]] for k in data['human_seed_keys']])
    for key,label in zip(data['human_seed_keys'],data['human_seed_labels']):assert data['predicted_class'][index[key]]==label
    references_before=data['reference_labels'].copy();data['reference_labels']=data['reference_labels'].astype('U32')
    for i,key in enumerate(data['reference_keys']):
        if key in result.index:data['reference_labels'][i]=result.at[key,'predicted_class']
    geometry=frozen.get('geometry_source_groups')
    geometry_publication=None
    previous_metadata=json.loads(str(data['metadata'].item()))
    existing_geometry=previous_metadata.get('geometry_source_groups')
    if geometry and existing_geometry and existing_geometry.get('eligible_train_group_manifest_sha256')==geometry['group_manifest_sha256']:
        # Existing grouping already represents this exact geometry snapshot.
        # Keep identifiers and partitions stable across label-only publications.
        geometry_publication=existing_geometry
        geometry=None
    if geometry:
        group_path=NOTES/'table-csv/geometry_train_groups.csv'
        assert digest(group_path)==geometry['group_manifest_sha256']
        group_table=pd.read_csv(group_path)
        assert group_table.key.is_unique and group_table.key.str.startswith('train:').all()
        assert len(group_table)==1904
        for row in group_table.itertuples():assert data['source_sha256'][index[row.key]]==row.source_sha256
        train_mask=np.char.startswith(data['keys'].astype(str),'train:')
        old_groups=data['groups'].copy()
        unique=sorted(set(old_groups[train_mask]));parent={g:g for g in unique}
        def find(g):
            while parent[g]!=g:
                parent[g]=parent[parent[g]];g=parent[g]
            return g
        def union(a,b):
            a,b=find(a),find(b)
            parent[max(a,b)]=min(a,b)
        for _,members in group_table.groupby('group',sort=True):
            aliases=[old_groups[index[k]] for k in members.key]
            for alias in aliases[1:]:union(aliases[0],alias)
        components={}
        for group in unique:components.setdefault(find(group),[]).append(group)
        names={root:'train_geom_'+hashlib.sha256('\n'.join(sorted(aliases)).encode()).hexdigest()[:16]
               for root,aliases in components.items()}
        data['groups']=data['groups'].astype('U64')
        data['groups'][train_mask]=np.array([names[find(g)] for g in old_groups[train_mask]])
        data['reference_groups']=data['reference_groups'].astype('U64')
        for i,key in enumerate(data['reference_keys']):data['reference_groups'][i]=data['groups'][index[key]]
        geometry_publication={'method':'train_pixel_phash_and_verified_SIFT_RANSAC_source_groups',
                              'before_raw_train_groups':len(unique),'after_raw_train_groups':len(components),
                              'eligible_train_group_manifest_sha256':geometry['group_manifest_sha256'],
                              'verified_edges_sha256':geometry['accepted_edges_sha256'],
                              'test_rows_used':0,'held_rows_used_for_SIFT':0}
        for _,members in group_table.groupby('group'):
            assert len({data['groups'][index[k]] for k in members.key})==1
    assert all(np.array_equal(data[k][test_mask],v) for k,v in before_test.items())
    metadata=json.loads(str(data['metadata'].item()))
    metadata['version']=3;metadata['prior_assignment_method']=metadata['assignment_method'];metadata['assignment_method']=frozen['method']
    metadata['assignment_sha256']=frozen['prediction_sha256']
    if geometry_publication:
        metadata['group_method']=geometry_publication['method'];metadata['geometry_source_groups']=geometry_publication
    metadata['refinement']={
        'full_frame_train_rows':1906,'manual_source_images':len(seeds),'manual_category_counts':frozen['manual_class_counts'],
        'manual_origin':'participant reviewed train images','manual_limit_per_class':5,
        'changes_vs_previous_program':frozen['changed_rows_vs_baseline'],'lighting_rule_changes':frozen['lighting_rows'],
        'stable_classifier_graph_changes':frozen['stable_automatic_propagation_rows'],
        'extra_cluster_changes':frozen.get('extra_cluster_rows',0),
        'new_algorithmic_changes':frozen.get('new_algorithmic_changes'),
        'new_cluster_changes':frozen.get('new_cluster_changes'),
        'cluster_plan_sha256':frozen.get('cluster_plan_sha256'),
        'new_neural_changes':frozen.get('new_neural_changes',0),
        'new_vit_propagation_changes':frozen.get('new_vit_propagation_changes',0),
        'vit_primary_head':frozen.get('primary_head'),
        'vit_primary_plan_sha256':frozen.get('primary_plan_sha256'),
        'vit_source_feature_sha256':frozen.get('source_feature_sha256'),
        'robust_neural_plan_sha256':frozen.get('robust_plan_sha256'),
        'human_holdout_protocol':frozen.get('human_diagnostic_protocol','leave-source-group-out anchor development'),
        'geometry_source_groups':geometry_publication,
        'weak_lighting_rows_restored':frozen.get('weak_lighting_rows_restored',0),
        'cumulative_changes_vs_initial_program':frozen.get('cumulative_changes_vs_initial_program',frozen['changed_rows_vs_baseline']),
        'unstable_proposals_held_out':frozen['unstable_proposals_not_applied'],
        'feature_family_count':family_count,
        'graph_train_only':True,'source_group_folds':5,
        'human_anchor_weight':frozen.get('primary_head',{}).get('human_weight'),
        'historical_lighting_pseudo_weight':4.,'historical_conflicting_pseudo_weight_regimes':[1.,.05],
        'acceptance':acceptance,'development_guard':latest_plan.get('guard'),
        'method_plan_sha256':digest(plan_path) if plan_path.exists() else None,
        'confidence_calibration':frozen.get('confidence_calibration'),
        'test_rows_in_fit':0,'reference_annotations_used_as_targets':False,
        'human_leave_source_group_out_matches':frozen['human_leave_source_group_out_matches'],
        'human_holdout_rows':frozen['human_holdout_rows'],'semantic_accuracy':None,
        'holdout_limit':'few difficult anchors, not random independent final evaluation',
        'refinement_repeated_by_preprocessing_run':False,
        'refinement_method':frozen['method'],
        'eligible_fit_train_rows':frozen.get('eligible_fit_train_rows'),
        'excluded_hold_keys':frozen.get('excluded_hold_keys',[]),
        'human_seed_sha256':frozen['human_seed_sha256'],
        'validation_limit':frozen.get('validation_limit'),
        'anchor_only_material_extra_changes':frozen.get('anchor_only_material_extra_changes',0),
        'anchor_only_material_plan_sha256':frozen.get('anchor_only_material_plan_sha256'),
    }
    data['metadata']=np.array(json.dumps(metadata))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=inside(NOTES/'.runtime'/f'before_bounded_anchor_{stamp}');backup.mkdir(parents=True)
    notebook=ROOT/'final-submit/DSC2026_Notebook.ipynb'
    shutil.copy2(input_path,backup/input_path.name);shutil.copy2(notebook,backup/notebook.name)
    proof_names=['execution.json','validation.json','latest.json','notebook_figures.json','feature_recipe.json','policy.json','skills_validation.json','project_contract.json']
    for name in proof_names:
        if (NOTES/name).exists():shutil.copy2(NOTES/name,backup/name)
    shutil.copy2(NOTES/'json/manual_label_budget_status.json',backup/'manual_label_budget_status.json')
    move(ROOT/'data/processed',backup/'processed');move(ROOT/'final-submit/reports',backup/'reports')
    report={'status':'rebuilding','backup':str(backup.relative_to(ROOT)),'candidate_sha256':frozen['prediction_sha256'],
            'manual_unique_source_images':len(seeds),'manual_class_counts':frozen['manual_class_counts'],
            'changed_full_frame_categories':int(changed.sum()),'cumulative_changes_vs_initial_program':frozen.get('cumulative_changes_vs_initial_program',int(changed.sum())),
            'source_frozen_prefix':prefix,'reference_categories_updated':int((references_before!=data['reference_labels']).sum()),
            'test_snapshot_unchanged':True,'raw_sources_unchanged':True,'original_active_sha256':frozen['protected_sha256']}
    write_json(NOTES/'json/bounded_anchor_publication.json',report)
    try:
        np.savez_compressed(input_path,**data)
        contract=json.loads((NOTES/'project_contract.json').read_text());contract['active_train_label_manifest']='data/processed/manual_seeds.csv'
        contract['current_scope']='EDA_preprocessing_with_bounded_participant_seeds_and_unverified_program_categories'
        contract['snapshot_assignment_method']=frozen['method'];write_json(NOTES/'project_contract.json',contract)
        budget=json.loads((NOTES/'json/manual_label_budget_status.json').read_text())
        budget['manual_labels_active_in_processed']=len(seeds);budget['status']='notebook_rebuild_in_progress';write_json(NOTES/'json/manual_label_budget_status.json',budget)
        assert not (ROOT/'data/processed').exists()
        with (backup/'notebook_execution.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'docs/skills/tools/build_notebook.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (backup/'validation.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'docs/skills/tools/validate_collection.py'),'--verify-images'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report.update(status='passed',input_sha256=digest(input_path),notebook_sha256=digest(notebook),manifest_sha256=digest(ROOT/'data/processed/manifest.csv'))
        budget['status']='active_preprocessing_published';write_json(NOTES/'json/manual_label_budget_status.json',budget)
    except BaseException as exc:
        for path,name in [(ROOT/'data/processed','failed_processed'),(ROOT/'final-submit/reports','failed_reports')]:
            if path.exists():move(path,backup/name)
        move(backup/'processed',ROOT/'data/processed');move(backup/'reports',ROOT/'final-submit/reports')
        shutil.copy2(backup/input_path.name,input_path);shutil.copy2(backup/notebook.name,notebook)
        for name in proof_names:
            if (backup/name).exists():shutil.copy2(backup/name,NOTES/name)
        shutil.copy2(backup/'manual_label_budget_status.json',NOTES/'json/manual_label_budget_status.json')
        report.update(status='rolled_back',error=repr(exc));write_json(NOTES/'json/bounded_anchor_publication.json',report)
        raise
    write_json(NOTES/'json/bounded_anchor_publication.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'bounded_consensus')
