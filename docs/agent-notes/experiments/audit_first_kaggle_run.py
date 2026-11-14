"""Read-only model diagnostic versus the user's fixed reference; never changes predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'final-submit/ekperimen-kaggle/tahap 1/gelarrasa_run'
OUT=Path(__file__).with_name('first_kaggle_run_audit')
OUT.mkdir(exist_ok=True)
TARGET=ROOT/'docs/submission_ (1).csv'
def test_id(x): return f'{int(str(x).split(".")[0]):03d}'
def load_labels(p):
    d=pd.read_csv(p,dtype={'image_id':str}); d['image_id']=d.image_id.map(test_id)
    assert d.image_id.is_unique and len(d)==480
    return d.set_index('image_id').waste_category
reference=load_labels(TARGET)
submission=load_labels(RUN/'DSC2026_Submission.csv').reindex(reference.index)
assert submission.notna().all()
manifest=json.loads((RUN/'logs/run_manifest.json').read_text())
classes=manifest['classes']; names=manifest['models']
table=pd.DataFrame({'image_id':reference.index,'target':reference.to_numpy(),'ensemble':submission.to_numpy()})
table['different']=table.target.ne(table.ensemble)
probabilities={}; predictions={}
summaries=[]
for name in names+['ensemble']:
    d=pd.read_csv(RUN/f'predictions/test_probabilities_{name}.csv',dtype={'image_id':str})
    d['image_id']=d.image_id.map(test_id); d=d.set_index('image_id').reindex(reference.index)
    cols=[f'prob_{c}' for c in classes] if name=='ensemble' else classes
    pp=d[cols].to_numpy(float); assert pp.shape==(480,8) and np.isfinite(pp).all()
    probabilities[name]=pp
    prediction=np.asarray(classes)[pp.argmax(1)]; predictions[name]=prediction
    table[name]=prediction; table[f'{name}_confidence']=pp.max(1)
    mismatch=prediction!=reference.to_numpy()
    summaries.append({'model':name,'same_as_target':int((~mismatch).sum()),'different_from_target':int(mismatch.sum()),'agreement_percent':round(100*(~mismatch).mean(),4),'mean_confidence_on_different':float(pp[mismatch].max(1).mean()),'different_with_confidence_ge_0_8':int((mismatch&(pp.max(1)>=.8)).sum())})
assert np.array_equal(predictions['ensemble'],submission.to_numpy())
assert np.allclose(probabilities['ensemble'],np.mean([probabilities[n] for n in names],axis=0),atol=1e-6)
votes=np.stack([predictions[n]==reference.to_numpy() for n in names],axis=1).sum(1)
table['models_matching_target']=votes
table.to_csv(OUT/'all_test_predictions.csv',index=False)
table[table.different].to_csv(OUT/'submission_differences.csv',index=False)
model_summary=pd.DataFrame(summaries); model_summary.to_csv(OUT/'model_comparison.csv',index=False)
per_class=[]
for c in classes:
    mask=table.target.eq(c)
    per_class.append({'target_class':c,'target_count':int(mask.sum()),'ensemble_count':int(table.ensemble.eq(c).sum()),'same':int((mask&~table.different).sum()),'different':int((mask&table.different).sum())})
class_summary=pd.DataFrame(per_class); class_summary.to_csv(OUT/'class_comparison.csv',index=False)
confusion=pd.crosstab(table.target,table.ensemble).reindex(index=classes,columns=classes,fill_value=0)
confusion.to_csv(OUT/'target_vs_ensemble_confusion.csv')
confusion_pairs=table[table.different].groupby(['target','ensemble']).size().sort_values(ascending=False).rename('count').reset_index()
confusion_pairs.to_csv(OUT/'confusion_pairs.csv',index=False)

active=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'sample_id':str,'parent_id':str})
active_full=active[active.split.eq('train') & active.training_eligible & ~active.is_crop].copy()
active_full['image_id']=active_full.parent_id.str.zfill(4)
old=pd.read_csv(RUN/'labels/train_pseudo_targets.csv',dtype={'image_id':str}); old['image_id']=old.image_id.str.zfill(4)
compare=active_full[['image_id','assigned_class','parent_group']].merge(old,on='image_id',validate='one_to_one')
compare['different_from_active_processed']=compare.assigned_class.ne(compare.waste_category)
compare.to_csv(OUT/'old_train_vs_active_processed.csv',index=False)
included=compare[compare.eligible_for_fit]
train_summary={
    'active_full_frame_rows':len(active_full),'active_crop_rows':int(((active.split=='train')&active.training_eligible&active.is_crop).sum()),
    'old_train_rows_considered':len(old),'old_eligible':int(old.eligible_for_fit.sum()),'old_excluded':int((~old.eligible_for_fit).sum()),
    'shared_active_full_frames':len(compare),'shared_rows_same_label':int((~compare.different_from_active_processed).sum()),'shared_rows_different_label':int(compare.different_from_active_processed.sum()),
    'old_included_and_active_full_frame':len(included),'old_included_same_label':int((~included.different_from_active_processed).sum()),'old_included_different_label':int(included.different_from_active_processed.sum()),
    'active_full_frames_filtered_out_by_old_confidence':int((~compare.eligible_for_fit).sum()),
    'old_included_outside_active_full_frame':int((old.eligible_for_fit & ~old.image_id.isin(active_full.image_id)).sum()),
    'old_eligible_class_counts':old[old.eligible_for_fit].waste_category.value_counts().to_dict(),
    'active_class_counts':active[(active.split=='train')&active.training_eligible].assigned_class.value_counts().to_dict(),
}
history_rows=[]
for name in names:
    d=pd.read_csv(RUN/f'logs/final_{name}_history.csv')
    last=d.iloc[-1]; best=d.loc[d.val_proxy_macro_f1.idxmax()]
    history_rows.append({'model':name,'epochs':len(d),'head_epochs':int(d.stage.eq('head').sum()),'finetune_epochs':int(d.stage.eq('finetune').sum()),'best_train_metric_epoch':int(best.epoch),'final_epoch_train_loss':float(last.train_loss),'best_reported_train_macro_f1':float(best.val_proxy_macro_f1),'validation_columns_equal_train':bool(np.allclose(d.train_proxy_agreement,d.val_proxy_agreement))})
pd.DataFrame(history_rows).to_csv(OUT/'training_history_summary.csv',index=False)
summary={
    'target_file':str(TARGET.relative_to(ROOT)),'target_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),
    'run_submission':str((RUN/'DSC2026_Submission.csv').relative_to(ROOT)),
    'comparison_rows':480,'ensemble_differences':int(table.different.sum()),'ensemble_matches':int((~table.different).sum()),
    'models':summaries,'classes':per_class,'most_common_difference_pairs':confusion_pairs.head(12).to_dict('records'),
    'all_four_models_differ_from_target':int((votes==0).sum()),
    'ensemble_differs_but_at_least_one_model_matches':int((table.different&(votes>0)).sum()),
    'ensemble_matches_despite_at_least_one_model_differing':int((~table.different&(votes<4)).sum()),
    'model_support_histogram':{str(int(k)):int(v) for k,v in pd.Series(votes).value_counts().sort_index().items()},
    'train_population_comparison':train_summary,'final_history':history_rows,
    'oof_proxy':pd.read_csv(RUN/'validation/proxy_summary.csv').to_dict('records'),
    'diagnostic_only':True,'training_or_test_predictions_changed':False,
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('ENSEMBLE DIFFERENCES',summary['ensemble_differences'],'MATCHES',summary['ensemble_matches'])
print(model_summary.to_string(index=False))
print(class_summary.to_string(index=False))
print('TOP PAIRS',confusion_pairs.head(10).to_dict('records'))
print('TRAIN',json.dumps(train_summary))
print('VOTES',summary['model_support_histogram'])
print('HISTORY',history_rows)
print('REPORT',OUT)
