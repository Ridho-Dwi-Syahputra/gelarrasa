"""Detector-supported plastic prior on organizer TRAIN, with source-held-out check.

The route is evaluated on only 34 lawful human train sources. Neither archive
annotations nor post-freeze review table nor competition test data is read.
"""
import datetime
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump

def main():
    begin=time.time();prefix='detector_gated_prior'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'source_heldout_detector_supported_plastic_prior_v1',
          'source_features':['residual_graph_correction two OOF banks','frozen YOLO object detection stats'],
          'manual_train_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'archive_annotations_read_in_fit':False,'private_review_read_in_fit':False,
          'fixed_prior_boost_log_odds':1.5,'fixed_residual_probability_minimum':.4,
          'detector_condition':'Plastic max probability>=.65 and margin>=.05 versus all other source object classes',
          'base':'unchanged V3+prototype+residual OOF 28/34',
          'guard':'whole34 >=30/34, macro>=.85, at least2 human routes, all human routes correct',
          'gate':'two residual banks argmax Plastic after prior and detector condition; no human source changed',
          'maximum_changes':80,'cap':'descending min(corrected_residual_plastic_probability,detector_Plastic_score),stablekey',
          'score':'minimum of corrected residual probability and source Plastic detector confidence,uncalibrated',
          'limitation':'same34 reused development; detector source ontology differs from competition; prediction not independent accuracy',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lk={k:i for i,k in enumerate(keys)};pos=np.array([lk[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34 and seeds.assigned_class.value_counts().le(5).all()
    res=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz');det=np.load(NOTES/'.runtime/litter_object_features.npz');v3=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz');aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz')
    for z in [res,det,v3,aug]:assert np.array_equal(z['keys'],keys)
    info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall'])
    base=v3['probabilities'][wi].argmax(1);ap,am,ar=aug['oof_predictions'],aug['oof_margins'],aug['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac;acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45)
    base=np.where(acov,ac,base);rp=res['probabilities'];rc=rp[0].argmax(1);rs=np.min([p[np.arange(n),rc] for p in rp],0);base=np.where((rs>=.6)&~(acov&(ac!=rc)),rc,base)
    assert int((base[pos]==gold).sum())==28
    q=rp.astype('float64').copy();q[:,:,4]*=np.exp(plan['fixed_prior_boost_log_odds']);q/=q.sum(2,keepdims=True)
    both=(q.argmax(2)==4).all(0);res_score=q[:,:,4].min(0);dscore=det['detection_stats'][:,:5];dmargin=dscore[:,0]-dscore[:,1:].max(1);dvalid=(dscore[:,0]>=.65)&(dmargin>=.05)
    routed=both&(res_score>=.4)&dvalid&(base!=4)
    predicted=np.where(routed,4,base);matches=int((predicted[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,predicted[pos]));human_route=routed[pos];covered=int(human_route.sum());correct=int((human_route&(predicted[pos]==gold)).sum());guard=matches>=30 and macro>=.85 and covered>=2 and correct==covered
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);gate=routed&~human&(current!=4)
    priority=np.minimum(res_score,dscore[:,0]);order=np.argsort(-priority,kind='stable');pick=order[gate[order]][:80];gate[:]=False;gate[pick]=True
    if not guard:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=['plastic','detector_supported_plastic_prior_snapshot',float(priority[i]),'uncalibrated_source_consensus_score',True]
    selected=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(selected,index=False)
    pd.DataFrame({'key':keys,'candidate_plastic':routed,'residual_min_probability':res_score,'source_plastic_score':dscore[:,0],'source_plastic_margin':dmargin,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    summary={'matches':matches,'macro':macro,'covered':covered,'correct':correct,'guard':guard,'routed_train':int(routed.sum()),'changed_rows':int(gate.sum())}
    report.update(status='frozen_before_private_detector_prior_audit',method=plan['method'],prediction_sha256=sha(selected),protected_sha256=protected,primary_head=summary,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[summary],qualified_heads=[summary] if guard else [],selected_knn_heads=[],human_leave_source_group_out_matches=matches,human_holdout_rows=34,development_balanced_recall=macro,confidence_calibration={'selected':summary,'independent_validation':False},accepted_by_development_guard=bool(guard),proposed_new_labels=int(routed.sum()),changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-begin,validation_limit=plan['limitation'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if guard else 'reject','prediction_sha256':sha(selected),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
