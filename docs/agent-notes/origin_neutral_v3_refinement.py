"""Apply the frozen validated V3gate uniformly to machine-origin assignments.

Controlled policy ablation: remove only the machine-origin .95extra barrier;
keep the learned.8minimumshare, both selected graphs,34human protection and all
data exclusions. No refitting, altered probabilities, or private-review lookup.
"""
import datetime
import json
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    prefix='origin_neutral_v3';base_path=NOTES/'json/dinov3_base_neighbors_frozen.json';base=json.loads(base_path.read_text())
    assert base['accepted_by_development_guard'] and base['human_leave_source_group_out_matches']==26
    cal=base['confidence_calibration']['selected'];assert cal['covered']==14 and cal['correct']==14 and cal['threshold']==.8
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    cache_path=NOTES/'.runtime/dinov3_base_neighbors_outputs.npz'
    plan={'method':'origin_neutral_frozen_DINOv3Base_selective_gate_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'scope':'policyablationONLY;allmodelprobabilities/graphs/learnedthresholdunchanged;norefittingorparameterselection',
          'source_frozen_sha256':sha(base_path),'source_output_sha256':sha(cache_path),
          'original_control':'minimumshare>=.8andbothselectedgraphs;earliermachine-originrefinementrequires>=.95',
          'new_gate':'minimumshare>=.8andbothselectedgraphsforallnonhumanmachineorigins;humansalwaysprotected',
          'reason':'machine-originhistoryisnotgroundtruth;new26/34modeland14/14selectivedevelopmentgateapplyuniformly',
          'learned_threshold_evidence':cal,'global_guard':'frozenmodelaccepted26/34andmacro.783333,14/14selective;all34humanunchanged',
          'maximum_changes':20,'cap':'descendingminimumshare,stablekeyorder','private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'same34reuseddevelopment,notindependentAccuracy;changingoriginpolicydoesnotmake14/14aguaranteefornewimages',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);assert len(keys)==1904 and frame.human_reviewed.sum()==34
    z=np.load(cache_path,allow_pickle=False);assert np.array_equal(z['keys'],keys);heads=base['selected_heads']
    wi=max(range(len(heads)),key=lambda i:heads[i]['development_balanced_recall']);qi=[next(i for i,h in enumerate(heads) if h['bank']==q['bank']) for q in base['qualified_heads']];assert len(qi)==2
    p=z['probabilities'];candidate=p[wi].argmax(1);score=np.min([p[i,np.arange(len(keys)),candidate] for i in qi],0);support=(z['graphs'][qi]==candidate).sum(0)
    human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=(score>=cal['threshold'])&(support==2)&~human&(candidate!=current)
    prior_origin=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy();old_gate=gate&(~prior_origin|(score>=.95))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[candidate[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[candidate],'minimum_share':score,'graph_support':support,'human':human,'old_origin_gate':old_gate,'origin_neutral_gate':gate}).to_csv(NOTES/f'table-csv/{prefix}_policy_evidence.csv',index=False)
    report=dict(base);initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_origin_policy_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),
        accepted_by_development_guard=True,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_vit_propagation_changes=int(gate.sum()),new_neural_changes=0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],
        origin_policy_control={'old_gate_changes':int(old_gate.sum()),'origin_neutral_changes':int(gate.sum()),'newly_allowed_machine_origin_changes':int((gate&~old_gate).sum())})
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['global_guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(report['origin_policy_control'],indent=2))


if __name__=='__main__':main()
