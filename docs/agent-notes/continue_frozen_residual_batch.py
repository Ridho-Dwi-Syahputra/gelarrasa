"""Remaining fixed residual-route predictions, no fit or audit filtering."""
import datetime,json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump

def main():
    prefix='frozen_residual_continuation';source='residual_selective_router'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    old=json.loads((NOTES/f'json/{source}_frozen.json').read_text());assert old['accepted_by_development_guard']
    evidence=NOTES/f'table-csv/{source}_evidence.csv'
    plan={'method':'unchanged_frozen_residual_router_batch_continuation_v1','source_frozen_prefix':source,'evidence_sha256':sha(evidence),
          'model_or_threshold_changes':False,'hypothesis':'applyallremainingmodelqualifiedpredictions;firstbatch20capwasoperationalnotlabeldependent',
          'maximum_changes':20,'cap':'sameoriginalminimumcorrectedprobabilityranking','guard':'samefrozendevelopment28/34,macro.85,residual18/19covered',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str)
    e=pd.read_csv(evidence);assert np.array_equal(e.key.to_numpy(str),keys);primary=np.array([CLASSES.index(c) for c in e.candidate_class]);score=e.minimum_corrected_probability.to_numpy(float)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=e.residual_covered.to_numpy(bool)&~e.final_prototype_conflict.to_numpy(bool)&~frame.human_reviewed.to_numpy(bool)&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');selected=rank[gate[rank]][:20];gate[:]=False;gate[selected]=True
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'residual_graph_correction_snapshot',score[i],'uncalibrated_corrected_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');report=old.copy()
    report.update(status='frozen_before_private_residual_continuation_audit',method=plan['method'],protected_sha256=protected,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),prediction_sha256=sha(dest),source_code_sha256=sha(__file__),changed_rows_vs_baseline=int(gate.sum()),proposed_new_labels=proposals,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict())
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'proposals':proposals,'changes':int(gate.sum()),'refit':False,'changed_threshold':False}),flush=True)

if __name__=='__main__':main()
