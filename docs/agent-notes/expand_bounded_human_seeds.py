"""Select eight additional explicitly team-reviewed train anchors, not all review labels."""
import json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, unit, dump, sha


def main():
    old_path=NOTES/'table-csv/minimal_human_seed_proposal.csv'
    old=pd.read_csv(old_path,dtype={'image_id':str})
    review=pd.read_csv(NOTES/'table-csv/team_review_inventory.csv',dtype={'parent_id':str})
    eligible=review[review.team_preference.isin(['archive','program'])&~review.interpretation.str.startswith('interpreted')].copy()
    assert eligible.proposed_class.ne('').all()
    z=np.load(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    lookup=dict(zip(z['keys'],range(len(z['keys']))));x=unit(z['features'])
    groups=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str}).set_index('image_id').group.to_dict()
    used=set(old.image_id);used_groups={groups[k] for k in used};selections=[];unavailable=[]
    lamp=json.loads((NOTES/'json/lighting_specificity_frozen.json').read_text())
    auto_lamps={d['image_id'] for d in lamp['decisions'] if d['accepted']}
    for category in CLASSES:
        pool=eligible[eligible.proposed_class.eq(category)&~eligible.parent_id.isin(used)].copy()
        pool=pool[~pool.parent_id.map(groups).isin(used_groups)]
        if category=='electronic' and (~pool.parent_id.isin(auto_lamps)).any():pool=pool[~pool.parent_id.isin(auto_lamps)]
        if not len(pool):
            unavailable.append(category)
            continue
        positions=np.array([lookup['train:'+k] for k in pool.parent_id]);prior=old[old.proposed_class.eq(category)]
        if len(prior):
            previous=np.array([lookup['train:'+k] for k in prior.image_id]);scores=1-(x[positions]@x[previous].T).max(1)
            chosen=int(scores.argmax());reason='different representation from existing reviewed anchors'
        else:
            scores=(x[positions]@x[positions].T).mean(1);chosen=int(scores.argmax());reason='representative of explicitly reviewed category candidates'
        row=pool.iloc[chosen];used.add(row.parent_id);used_groups.add(groups[row.parent_id])
        selections.append({'image_id':row.parent_id,'source_path':row.source_path,'assigned_class':row.assigned_class,
                           'archive_class':row.archive_class,'team_preference':row.team_preference,'interpretation':row.interpretation,
                           'proposed_class':category,'source':'user_review_of_displayed_train_image','activated_for_training':False,
                           'selection_reason':reason,'selection_score':float(scores[chosen])})
    result=pd.concat([old,pd.DataFrame(selections)],ignore_index=True).sort_values('image_id')
    assert len(result)==len(old)+len(selections) and len(result)<=15 and result.image_id.is_unique and result.proposed_class.value_counts().le(5).all()
    output=NOTES/'table-csv/bounded_human_seeds.csv';result.to_csv(output,index=False)
    dump(NOTES/'json/bounded_human_seed_selection.json',{
        'status':'selected_for_next_refinement','source_review_inventory_sha256':sha(NOTES/'table-csv/team_review_inventory.csv'),
        'previous_seven_seed_sha256':sha(old_path),'selected_seed_sha256':sha(output),'total_unique_sources':len(result),
        'categories_without_explicit_extra_review':unavailable,
        'counts':result.proposed_class.value_counts().to_dict(),'extra_selections':selections,
        'unselected_review_labels_passed_to_refinement_models':False,'test_data_used_for_selection':False,
        'selection_is_not_training_all_reviewed_images':True})
    print(result[['image_id','proposed_class']].to_string(index=False))
    print(result.proposed_class.value_counts().to_dict())


if __name__=='__main__':main()
