"""Union of frozen selective experts, never selected by private review labels."""
import datetime,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump

def main():
    start=time.time();prefix='selective_feature_union'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'frozen_V3_and_full_source_selective_union_v1','threshold':.8,'experts':['dinov3_base_neighbors','full_distribution_neighbors'],
          'union':'eitherqualifiedexpert;conflictingtrustedexpertsabstain;otherwiseV3fallback','graph':'allqualifiedgraphsmatchroutedclassforoperationalchanges',
          'guard':'matches>=27andmacro>=V3baselineORmatches>=26andmacro>=V3baselineandunioncovered>=16andallcoveredcorrect',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'private_review_or_archive_read':False,
          'researcher_seen_prior_audits':True,'maximum_changes':20,'protected_sha256':protected,'source_code_sha256':sha(__file__),
          'evaluation_limit':'same34reuseddevelopment;featurescorrelated;notindependentAccuracy;graphnotOOFvalidated'}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34
    outputs=[]
    for name in plan['experts']:
        info=json.loads((NOTES/f'json/{name}_frozen.json').read_text());z=np.load(NOTES/f'.runtime/{name}_outputs.npz',allow_pickle=False);assert np.array_equal(z['keys'],keys)
        rows=info['selected_heads'];winner=max(range(len(rows)),key=lambda i:rows[i]['development_balanced_recall'])
        qualified=sorted([i for i,row in enumerate(rows) if row['development_balanced_recall']>=.65 and row['development_balanced_recall']>=rows[winner]['development_balanced_recall']-.1],key=lambda i:(-rows[i]['development_balanced_recall'],-rows[i]['anchor_matches']))[:2]
        pr=z['probabilities'];pred=pr[winner].argmax(1);score=np.min([pr[i,np.arange(n),pred] for i in qualified],axis=0);trusted=score>=.8
        graph=(z['graphs'][qualified]==pred).all(0)
        outputs.append({'pred':pred,'score':score,'trusted':trusted,'graph':graph,'name':name})
    left,right=outputs;conflict=left['trusted']&right['trusted']&(left['pred']!=right['pred'])
    use_right=right['trusted']&~left['trusted'];prediction=np.where(use_right,right['pred'],left['pred'])
    covered=(left['trusted']|right['trusted'])&~conflict
    score=np.where(use_right,right['score'],left['score']);graph=np.where(use_right,right['graph'],left['graph'])
    count=int(covered[pos].sum());correct=int((covered[pos]&(prediction[pos]==gold)).sum());matches=int((prediction[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,prediction[pos]));baseline_macro=float(balanced_accuracy_score(gold,left['pred'][pos]))
    accepted=bool((matches>=27 and macro>=baseline_macro-1e-12) or (matches>=26 and macro>=baseline_macro-1e-12 and count>=16 and correct==count))
    human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=covered&graph&~human&(prediction!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[prediction[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    metrics={'anchor_matches':matches,'development_balanced_recall':macro,'covered':count,'correct':correct,'conflicts':int(conflict[pos].sum()),'accepted':accepted,'proposals':proposals,'changes':int(gate.sum())}
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[prediction],'covered':covered,'conflict':conflict,'right_only':use_right,'minimum_neighbor_share':score,'graph_agreement':graph,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_selective_union_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=metrics,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=plan['experts'],qualified_heads=plan['experts'],selected_knn_heads=[],human_leave_source_group_out_matches=matches,human_holdout_rows=34,development_balanced_recall=macro,confidence_calibration={'selected':metrics,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','accepted_changes':int(gate.sum()),'prediction_sha256':sha(dest),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
