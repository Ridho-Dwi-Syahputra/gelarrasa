"""Apply the remaining eligible predictions of an unchanged frozen prototype model.

No refit, threshold adjustment, review filtering or ID list. The first publication
was capped at20; this continuation uses the same full-bank predictions and gate.
"""
import datetime,json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump

def main():
    prefix='frozen_prototype_continuation';source='augmented_prototype_consensus'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    old=json.loads((NOTES/f'json/{source}_frozen.json').read_text());assert old['accepted_by_development_guard']
    prediction=NOTES/f'.runtime/{source}_outputs.npz'
    plan={'method':'unchanged_frozen_augmented_prototype_batch_continuation_v1','source_frozen_prefix':source,'source_model_sha256':sha(prediction),
          'model_or_threshold_changes':False,'hypothesis':'applyallremainingmodelqualifiedpredictions;firstbatch20capwasoperationalnotlabeldependent',
          'maximum_changes':20,'cap':'sameoriginalminimumclassmarginranking','guard':'samefrozendevelopment27/34,macro.808333333333,7/7covered',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str)
    z=np.load(prediction,allow_pickle=False);assert np.array_equal(z['keys'],keys);primary=z['predictions'];score=z['score']
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=z['covered']&~frame.human_reviewed.to_numpy(bool)&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');selected=rank[gate[rank]][:20];gate[:]=False;gate[selected]=True
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'augmented_source_prototype_snapshot',score[i],'minimum_cosine_class_margin',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');report=old.copy()
    report.update(status='frozen_before_private_prototype_continuation_audit',method=plan['method'],protected_sha256=protected,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),prediction_sha256=sha(dest),source_code_sha256=sha(__file__),changed_rows_vs_baseline=int(gate.sum()),proposed_new_labels=proposals,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict())
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'proposals':proposals,'changes':int(gate.sum()),'refit':False,'changed_threshold':False}),flush=True)

if __name__=='__main__':main()
