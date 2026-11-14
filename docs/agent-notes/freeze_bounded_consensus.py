"""Freeze agreement across two prior-weight regimes, before private review audit."""
import json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def main():
    a_info=json.loads((NOTES/'json/bounded14_frozen.json').read_text())
    b_info=json.loads((NOTES/'json/bounded14_cleaned_frozen.json').read_text())
    ap=NOTES/'table-csv/bounded14_selected.csv';bp=NOTES/'table-csv/bounded14_cleaned_selected.csv'
    assert sha(ap)==a_info['prediction_sha256'] and sha(bp)==b_info['prediction_sha256']
    a=pd.read_csv(ap,dtype={'image_id':str});b=pd.read_csv(bp,dtype={'image_id':str})
    assert a.key.equals(b.key) and a.baseline_class.equals(b.baseline_class)
    human=a.human_reviewed.to_numpy();lamp=a.lighting_rule_changed.to_numpy()&~human
    stable=a.propagation_changed.to_numpy()&b.propagation_changed.to_numpy()&a.predicted_class.eq(b.predicted_class).to_numpy()&~human&~lamp
    result=a.copy();result['predicted_class']=result.baseline_class
    accepted=human|lamp|stable
    result.loc[accepted,'predicted_class']=a.loc[accepted,'predicted_class']
    result['category_source']='iterative_program_refinement_snapshot'
    result.loc[lamp,'category_source']='two_view_lighting_ontology_snapshot'
    result.loc[stable,'category_source']='bounded_anchor_stable_consensus_snapshot'
    result.loc[human,'category_source']='participant_reviewed_seed'
    result['propagation_changed']=stable;result['stable_across_weight_regimes']=stable
    current=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});current=current[current.split.eq('train')&~current.is_crop].set_index('parent_key').loc[result.key]
    result.loc[~accepted,'category_score']=current.category_score.to_numpy()[~accepted]
    result.loc[~accepted,'category_score_kind']=current.category_score_kind.to_numpy()[~accepted]
    result['needs_review']=True;result.loc[human,'needs_review']=False
    output=NOTES/'table-csv/bounded_consensus_selected.csv';result.to_csv(output,index=False)
    uncertain=(a.propagation_changed|b.propagation_changed)&~stable&~human&~lamp
    queue=a.loc[uncertain,['key','image_id','source_path','baseline_class','predicted_class','classifier_votes','graph_votes_for_classifier_category','local_anchor_banks']].copy()
    queue['alternative_regime_class']=b.loc[uncertain,'predicted_class'].to_numpy()
    queue.sort_values('image_id').to_csv(NOTES/'table-csv/unstable_refinement_candidates.csv',index=False)
    report={'status':'frozen_before_posthoc_audit','method':'bounded_human_anchor_two_regime_stable_consensus_v1',
            'manual_unique_source_images_used':int(human.sum()),'manual_class_counts':a_info['manual_class_counts'],
            'human_seed_file':'table-csv/bounded_human_seeds.csv','human_seed_sha256':a_info['human_seed_sha256'],
            'test_rows_in_fit':0,'train_rows':len(result),'lighting_rows':int(lamp.sum()),
            'stable_automatic_propagation_rows':int(stable.sum()),'unstable_proposals_not_applied':int(uncertain.sum()),
            'changed_rows_vs_baseline':int(result.predicted_class.ne(result.baseline_class).sum()),
            'human_rows_already_matching_baseline':int((human&result.predicted_class.eq(result.baseline_class).to_numpy()).sum()),
            'human_leave_source_group_out_matches':a_info['human_leave_source_group_out_matches'],'human_holdout_rows':a_info['human_holdout_rows'],
            'validation_limit':'anchors are few and deliberately difficult; not official accuracy or a guarantee for other rows',
            'prediction_sha256':sha(output),'protected_sha256':a_info['protected_sha256'],'semantic_accuracy':None,
            'active_data_modified':False,'selection_uses_unselected_review_labels':False,'selection_uses_archive_labels':False}
    dump(NOTES/'json/bounded_consensus_frozen.json',report)
    seeds=pd.read_csv(NOTES/report['human_seed_file'],dtype={'image_id':str})
    budget={'manual_labels_consumed_by_current_experiments':len(seeds),'manual_unique_source_images_ever_used':seeds.image_id.tolist(),
            'manual_class_counts':seeds.proposed_class.value_counts().to_dict(),'remaining_by_class':{c:5-int(seeds.proposed_class.eq(c).sum()) for c in CLASSES},
            'manual_labels_active_in_processed':0,'use_manual_labels_authorized_now':True,
            'source':'explicit user review, bounded expansion following weak seven-anchor holdout',
            'unselected_review_inventory_used_as_training_targets':False,'status':'frozen_pending_notebook_publication'}
    dump(NOTES/'json/manual_label_budget_status.json',budget)
    assert all(sha(ROOT/name)==value for name,value in report['protected_sha256'].items())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
