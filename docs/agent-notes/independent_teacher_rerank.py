"""Re-rank already frozen OOF candidates with a new independent CNN source.

Targets are automatic teacher evidence, never official semantic ground truth.
No visual-archive labels or competition test rows enter this selection.
"""
import numpy as np
import pandas as pd
import json
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'selection':'new independent eight-class CNN semantic agreement .6 + old independent-source anchor agreement .2 + OOF flip .2',
          'new_teacher_filter':'original confidence>=.95, margin>=.5; baseline agrees OR additional horizontal-view class/probability check passed',
          'manual_labels_consumed':0,'test_rows_in_fit':0,'minimum_teacher_examples_per_class':10,'minimum_predictions_per_class':5,
          'candidate_outputs':'previously frozen OOF matrices; no fitting on new teacher labels during selection',
          'selected_output_is_oof_not_in_sample_refit':True,'reference_archive_read':False,'researcher_has_seen_prior_private_audits':True,
          'semantic_accuracy':None,'protected_sha256':protected}
    dump(NOTES/'json/independent_teacher_selection_plan.json',plan)
    source_frozen=json.loads((NOTES/'json/task8_teacher_frozen.json').read_text())
    assert sha(NOTES/'table-csv/task8_teacher_selected.csv')==source_frozen['prediction_sha256']
    targets=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    base=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});base=base[base.split.eq('train')&~base.is_crop]
    baseline=targets[['key','image_id']].merge(base[['parent_key','assigned_class']],left_on='key',right_on='parent_key',validate='one_to_one').assigned_class.to_numpy()
    teacher=np.load(NOTES/'.runtime/task8_teacher_features.npz',allow_pickle=False)
    assert np.array_equal(teacher['keys'],targets.key.to_numpy(str))
    p=teacher['probabilities'];new_y=p.argmax(1);top=np.sort(p,axis=1)
    decisions=pd.read_csv(NOTES/'table-csv/task8_teacher_decisions.csv',dtype={'image_id':str})
    accepted=set(decisions.loc[decisions.accepted,'key']) if len(decisions) else set()
    verified=(np.array(CLASSES)[new_y]==baseline)|np.array([key in accepted for key in targets.key])
    usable=(p.max(1)>=.95)&((top[:,-1]-top[:,-2])>=.5)&verified
    counts=np.bincount(new_y[usable],minlength=8);assert (counts>=10).all(),counts
    old_y=np.array([CLASSES.index(c) for c in targets.automatic_class]);old_usable=targets.automatic_anchor.to_numpy()
    matrices=[];flips=[];names=[];families=[];hashes={}
    for family,filename in [('old_banks','automatic_full_seed_oof.npz'),('dinov3','dinov3_oof.npz'),('coverage','automatic_prior_weight_oof.npz')]:
        path=NOTES/'.runtime'/filename;z=np.load(path,allow_pickle=False);assert np.array_equal(z['keys'],targets.key.to_numpy(str))
        if not matrices:flip_keys=z['flip_keys']
        else:assert np.array_equal(flip_keys,z['flip_keys'])
        matrices.append(z['predictions']);flips.append(z['flip_predictions']);names.extend(z['candidate'].tolist());families.extend([family]*len(z['candidate']));hashes[filename]=sha(path)
    predictions=np.vstack(matrices);flip_predictions=np.vstack(flips);index=dict(zip(targets.key,range(len(targets))));flip_ids=np.array([index[k] for k in flip_keys]);rows=[]
    for i,(name,pred,fp) in enumerate(zip(names,predictions,flip_predictions)):
        new_macro=float(np.mean([(pred[usable&(new_y==c)]==c).mean() for c in range(8)]))
        old_macro=float(np.mean([(pred[old_usable&(old_y==c)]==c).mean() for c in range(8)]))
        stability=float((pred[flip_ids]==fp).mean());valid=bool((np.bincount(pred,minlength=8)>=5).all())
        rows.append({'index':i,'candidate':name,'family':families[i],'new_teacher_macro_agreement':new_macro,'old_teacher_macro_agreement':old_macro,
                     'oof_flip_agreement':stability,'selection_score':.6*new_macro+.2*old_macro+.2*stability if valid else -1})
    selected=max(rows,key=lambda row:row['selection_score']);chosen=predictions[selected['index']]
    support=(predictions==chosen[None,:]).mean(0)
    class_votes=np.stack([(predictions==c).mean(0) for c in range(8)],axis=1)
    competitor=class_votes.copy();competitor[np.arange(len(chosen)),chosen]=-1
    margin=support-competitor.max(1)
    output=targets[['key','image_id']].copy();output['predicted_class']=np.array(CLASSES)[chosen];output['method']=selected['candidate'];output['oof_variant_support']=support;output['support_margin']=margin;output['ground_truth_verified']=False
    path=NOTES/'table-csv/independent_teacher_selected.csv';output.to_csv(path,index=False)
    pd.DataFrame(rows).to_csv(NOTES/'table-csv/independent_teacher_ranking.csv',index=False)
    baseline_idx=np.array([CLASSES.index(c) for c in baseline])
    baseline_agreement=float(np.mean([(baseline_idx[usable&(new_y==c)]==c).mean() for c in range(8)]))
    report={'status':'frozen_before_reference_audit','manual_labels_consumed':0,'test_rows_processed':0,'candidate_count':len(rows),
            'new_teacher_usable_rows':int(usable.sum()),'new_teacher_counts':dict(zip(CLASSES,counts.tolist())),
            'selected_before_archive_audit':selected,'selected_prediction_sha256':sha(path),'selected_class_counts':dict(zip(CLASSES,np.bincount(chosen,minlength=8).tolist())),
            'baseline_new_teacher_agreement_descriptive_only':baseline_agreement,'baseline_is_not_an_oof_control':True,
            'oof_matrices_sha256':hashes,'source_code_sha256':sha(__file__),'semantic_accuracy':None,'active_data_modified':False,'protected_sha256':protected}
    dump(NOTES/'json/independent_teacher_selection_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k not in ['protected_sha256','oof_matrices_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
