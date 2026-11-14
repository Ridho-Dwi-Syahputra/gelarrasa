"""Read-only post-freeze triage against participant notes, never model selection."""
import json
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,sha,dump


def main():
    info=json.loads((NOTES/'json/bounded_final_frozen.json').read_text())
    path=NOTES/'table-csv/bounded_final_selected.csv';assert sha(path)==info['prediction_sha256']
    predictions=pd.read_csv(path,dtype={'image_id':str})
    review=pd.read_csv(NOTES/'table-csv/team_review_inventory.csv',dtype={'parent_id':str})
    frame=review.merge(predictions[['image_id','predicted_class','category_source','human_reviewed','lighting_rule_changed','propagation_changed']],left_on='parent_id',right_on='image_id',validate='one_to_one')
    frame['category_changed']=frame.predicted_class.ne(frame.assigned_class)
    frame['matches_stated_preference']=frame.proposed_class.ne('')&frame.predicted_class.eq(frame.proposed_class)
    frame['triage']=frame.team_preference
    firm=frame.team_preference.isin(['archive','program'])
    frame.loc[firm&~frame.matches_stated_preference,'triage']='still_differs_from_stated_preference'
    frame.loc[firm&frame.matches_stated_preference,'triage']='program_agrees_with_stated_preference'
    frame.loc[frame.human_reviewed,'triage']='bounded_participant_anchor'
    frame.loc[frame.parent_id.isin(['1662','1885']),'triage']='participant_hold_no_human_anchor_no_final_training'
    frame.sort_values('parent_id').to_csv(NOTES/'table-csv/bounded_review_outcome.csv',index=False)
    queue=frame[frame.triage.eq('still_differs_from_stated_preference')].sort_values('parent_id')
    queue.to_csv(NOTES/'table-csv/remaining_team_review_differences.csv',index=False)
    heldout=pd.read_csv(NOTES/'table-csv/bounded14_human_holdout.csv',dtype={'image_id':str})
    dump(NOTES/'json/participant_deferred_cases.json',{'source':'explicit participant answer to pending case question',
         'image_ids':['1662','1885'],'human_anchors_allowed':False,'final_classifier_training_allowed':False,
         'raw_and_review_images_preserved':True,'no_new_category_labels_created':True})
    report={'scope':'private posthoc triage of participant notes, not official accuracy and not parameter selection',
            'original_review_images':len(frame),'original_review_images_changed':int(frame.category_changed.sum()),
            'human_anchors_in_original_150':int(frame.human_reviewed.sum()),
            'stated_preferences_still_different':len(queue),'triage_counts':frame.triage.value_counts().to_dict(),
            'unresolved_preference_pairs':queue.groupby(['predicted_class','proposed_class']).size().rename('count').reset_index().to_dict('records'),
            'human_holdout_unrecovered_ids':heldout.loc[~heldout.matches_reviewed_category,'image_id'].tolist(),
            'manual_unique_sources_used_for_training':14,'remaining_manual_capacity_total':26,
            'model_prediction_sha256':info['prediction_sha256'],'posthoc_review_did_not_modify_predictions':True}
    assert sha(path)==info['prediction_sha256'];dump(NOTES/'json/bounded_review_outcome.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
