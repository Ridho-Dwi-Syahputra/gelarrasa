"""Route a residual specialist only when its two feature banks agree strongly."""
import datetime,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump

def main():
    start=time.time();prefix='complementary_residual_router'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'prototype_cold_residual_and_strict_warm_complement_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'inputs':['augmented_prototype_consensus','residual_graph_correction','warm_residual_correction','dinov3_base_neighbors'],
          'baseline':'V3OOFwithfrozenaugmentedprototypepartialrouting(.1margin,.45rawcos)',
          'routing':'cold.6unchanged;warmonlywherecoldabstainsandprototypehasnoconflict;warmthresholdgrid.8/.9/.95;neveroverrideconfidentcold',
          'guard':'whole34>=29matchesandmacro>=.85;residualunioncovered>=20,precision>=.9;noclassquotaorIDrules',
          'score':'minimumcorrectedprobabilityoftwocorrelatedbanks;notcorrectnessprobability',
          'maximum_changes':20,'cap':'descendingminimumcorrectedprobability,stablekeyorder','private_review_or_archive_read':False,
          'researcher_seen_prior_audits':True,'protected_sha256':protected,'source_code_sha256':sha(__file__),
          'evaluation_limit':'same34reuseddevelopment;selectivecombinationisnewhypothesis,earlierglobalrejectionpreserved;notindependentAccuracy'}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34
    v3=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz',allow_pickle=False);info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);fallback=v3['probabilities'][wi].argmax(1)
    aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz',allow_pickle=False);res=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz',allow_pickle=False)
    for z in [v3,aug,res]:assert np.array_equal(z['keys'],keys)
    ap=aug['oof_predictions'];am=aug['oof_margins'];ar=aug['oof_raw_similarity'];av=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=av.argmax(1);agree=ap==ac
    acovered=(av.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45)
    baseline=np.where(acovered,ac,fallback)
    ri=json.loads((NOTES/'json/residual_graph_correction_frozen.json').read_text());rwi=max(range(len(ri['selected_heads'])),key=lambda i:(ri['selected_heads'][i]['development_balanced_recall'],ri['selected_heads'][i]['anchor_matches']));rp=res['probabilities'];primary=rp[rwi].argmax(1);score=np.min([p[np.arange(n),primary] for p in rp],0)
    rcov=score>=.6;route=rcov&~(acovered&(ac!=primary));combined=np.where(route,primary,baseline)
    warm=np.load(NOTES/'.runtime/warm_residual_correction_outputs.npz',allow_pickle=False);assert np.array_equal(warm['keys'],keys)
    winfo=json.loads((NOTES/'json/warm_residual_correction_frozen.json').read_text());wwi=max(range(len(winfo['selected_heads'])),key=lambda i:(winfo['selected_heads'][i]['development_balanced_recall'],winfo['selected_heads'][i]['anchor_matches']))
    wc=warm['probabilities'][wwi].argmax(1);ws=np.min([p[np.arange(n),wc] for p in warm['probabilities']],0);development=[]
    for threshold in [.8,.9,.95]:
        wroute=(ws>=threshold)&~rcov&~(acovered&(ac!=wc));candidate=np.where(wroute,wc,combined)
        union=rcov|wroute;uc=np.where(wroute,wc,primary);covered=int(union[pos].sum());correct=int((union[pos]&(uc[pos]==gold)).sum());matches=int((candidate[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,candidate[pos]));eligible=bool(matches>=29 and macro>=.85-1e-12 and covered>=20 and correct/covered>=.9)
        development.append({'threshold':threshold,'anchor_matches':matches,'development_balanced_recall':macro,'residual_covered':covered,'residual_correct':correct,'eligible':eligible})
    choices=[r for r in development if r['eligible']];chosen=max(choices,key=lambda r:(r['anchor_matches'],r['development_balanced_recall'],r['residual_correct']/r['residual_covered'],r['residual_covered'],r['threshold'])) if choices else development[0];accepted=bool(choices)
    covered,correct,matches,macro=[chosen[k] for k in ['residual_covered','residual_correct','anchor_matches','development_balanced_recall']]
    score=ws;primary=wc;current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);final_proto_conflict=aug['covered']&(aug['predictions']!=primary)
    gate=(score>=chosen['threshold'])&~rcov&~final_proto_conflict&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'residual_graph_correction_snapshot',score[i],'uncalibrated_corrected_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    metrics={'anchor_matches':matches,'development_balanced_recall':macro,'residual_covered':covered,'residual_correct':correct,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum())}
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_corrected_probability':score,'cold_residual_covered':rcov,'final_prototype_conflict':final_proto_conflict,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_selective_residual_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=metrics,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'source':p} for p in plan['inputs']],qualified_heads=[{'source':p} for p in plan['inputs']],selected_knn_heads=[],human_leave_source_group_out_matches=matches,human_holdout_rows=34,development_balanced_recall=macro,confidence_calibration={'selected':metrics,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
