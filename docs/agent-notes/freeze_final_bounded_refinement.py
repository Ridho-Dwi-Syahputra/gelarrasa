"""Final evidence gates: strong lighting and cluster/graph/classifier agreement."""
import json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    model_path=NOTES/'table-csv/bounded_consensus_selected.csv'
    old_info=json.loads((NOTES/'json/bounded_consensus_frozen.json').read_text());assert sha(model_path)==old_info['prediction_sha256']
    initial=pd.read_csv(model_path,dtype={'image_id':str}).set_index('key')
    publication=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text());assert publication['status']=='passed'
    current=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    current=current[current.split.eq('train')&~current.is_crop].set_index('parent_key').loc[initial.index]
    assert current.assigned_class.equals(initial.predicted_class.rename('assigned_class'))
    result=initial.copy();result['baseline_class']=current.assigned_class.to_numpy()
    result['human_reviewed']=current.human_reviewed.to_numpy()
    for column in ['category_source','category_score','category_score_kind','needs_review']:
        result[column]=current['label_source' if column=='category_source' else column].to_numpy()
    light=json.loads((NOTES/'json/lighting_specificity_frozen.json').read_text())
    weak=[d['image_id'] for d in light['decisions'] if d['accepted'] and min(d['original_specific_hits'],d['hflip_specific_hits'])<2]
    weak_mask=result.image_id.isin(weak)&~result.human_reviewed
    original_manifest=pd.read_csv(ROOT/publication['backup']/'processed/manifest.csv',dtype={'parent_id':str}).set_index('parent_key').loc[result.index]
    result.loc[weak_mask,'predicted_class']=original_manifest.loc[weak_mask,'assigned_class'].to_numpy()
    result.loc[weak_mask,'category_source']='iterative_program_refinement_snapshot'
    result.loc[weak_mask,'category_score']=original_manifest.loc[weak_mask,'category_score'].to_numpy()
    result.loc[weak_mask,'category_score_kind']=original_manifest.loc[weak_mask,'category_score_kind'].to_numpy()
    result.loc[weak_mask,'needs_review']=True
    c=pd.read_csv(NOTES/'table-csv/bounded_overclustering_proposals.csv',dtype={'image_id':str}).set_index('key').loc[result.index]
    a=np.load(NOTES/'.runtime/bounded14_outputs.npz',allow_pickle=False)
    b=np.load(NOTES/'.runtime/bounded14_cleaned_outputs.npz',allow_pickle=False)
    assert np.array_equal(a['keys'],result.index.to_numpy(str)) and np.array_equal(b['keys'],a['keys'])
    accepted=np.zeros(len(result),bool);records=[]
    blocked=set(weak)|set(result.loc[result.human_reviewed,'image_id'])
    for i,row in enumerate(c.itertuples()):
        if not row.changed_candidate or row.image_id in blocked:continue
        label=CLASSES.index(row.clustering_proposal)
        graph_votes=int((a['graph_predictions'][:,i]==label).sum())
        clean_votes=int((b['classifier_predictions'][:,i]==label).sum())
        accept=row.configuration_support>=5 and row.feature_family_support>=3 and graph_votes==3 and clean_votes>=3
        records.append({'image_id':row.image_id,'cluster_class':row.clustering_proposal,'configuration_support':row.configuration_support,
                        'feature_family_support':row.feature_family_support,'graph_support':graph_votes,'clean_classifier_support':clean_votes,'accepted':bool(accept)})
        if accept:
            accepted[i]=True;key=result.index[i];result.at[key,'predicted_class']=row.clustering_proposal
            result.at[key,'category_source']='bounded_anchor_cluster_consensus_snapshot'
            result.at[key,'category_score']=min(graph_votes/3,clean_votes/5)
            result.at[key,'category_score_kind']='multi_method_vote_share';result.at[key,'needs_review']=True
    result['cluster_refinement_changed']=accepted
    result['category_source']=result.category_source.astype(str)
    output=NOTES/'table-csv/bounded_final_selected.csv';result.reset_index().to_csv(output,index=False)
    pd.DataFrame(records).to_csv(NOTES/'table-csv/bounded_cluster_evidence_gate.csv',index=False)
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    cumulative=result.predicted_class.ne(initial.baseline_class)
    report={'status':'frozen_before_final_posthoc_audit','method':'bounded_human_material_refinement_final_v1',
            'human_seed_file':old_info['human_seed_file'],'human_seed_sha256':old_info['human_seed_sha256'],
            'manual_unique_source_images_used':14,'manual_class_counts':old_info['manual_class_counts'],'test_rows_in_fit':0,
            'train_rows':1906,'lighting_rows':int(result.category_source.eq('two_view_lighting_ontology_snapshot').sum()),
            'stable_automatic_propagation_rows':int(result.category_source.eq('bounded_anchor_stable_consensus_snapshot').sum()),
            'extra_cluster_rows':int(accepted.sum()),'weak_lighting_rows_restored':int(weak_mask.sum()),
            'unstable_proposals_not_applied':old_info['unstable_proposals_not_applied'],
            'changed_rows_vs_baseline':int(result.predicted_class.ne(result.baseline_class).sum()),
            'cumulative_changes_vs_initial_program':int(cumulative.sum()),
            'human_leave_source_group_out_matches':old_info['human_leave_source_group_out_matches'],'human_holdout_rows':14,
            'validation_limit':old_info['validation_limit'],'prediction_sha256':sha(output),
            'source_counts':result.category_source.value_counts().to_dict(),'protected_sha256':protected,
            'semantic_accuracy':None,'active_data_modified':False,
            'final_gate_does_not_use_unselected_review_or_archive_targets':True,
            'refinement_heads_used_16_lighting_pseudo_anchors_before_postfilter':True,
            'deferred_keys_not_human_anchors':['train:1662','train:1885']}
    dump(NOTES/'json/bounded_final_frozen.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
