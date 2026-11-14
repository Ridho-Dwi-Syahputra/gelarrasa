"""Selective consensus ablation with a predeclared coverage/risk proxy guard.

Wilson lower score is a heuristic ranking on reused development, NOT a valid
post-selection risk certificate. All four teachers share data/features and are
correlated. No full review or visual archive is read by this experiment.
"""
import datetime
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def consensus(probabilities, threshold, support_required):
    pred=probabilities.argmax(2);confident=probabilities.max(2)>=threshold
    votes=np.stack([((pred==k)&confident).sum(0) for k in range(8)],1);cat=votes.argmax(1);support=votes.max(1)
    ordered=np.sort(votes,axis=1);unique=ordered[:,-1]>ordered[:,-2]
    agree=(pred==cat)&confident;score=np.where(agree,probabilities[:,np.arange(len(cat)),cat],1.).min(0)
    covered=unique&(support>=support_required)
    return cat,covered,score,support


def main():
    prefix='selective_consensus'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    baseline=wilson_lower(15,16)
    plan={'method':'correlated_image_teacher_selective_consensus_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'objective':'selectivecoverageandempiricalprecision,notwhole34Accuracy',
          'teachers':['previousDINO_CNN','object_zoom','foldcontained_selftraining','genuine_source_core_repair'],
          'predictions':'all1904usegroupheldoutprobabilities,includingcore_repairOOFnotfullfit;querysourcesexcludedfromeachteacherfit',
          'threshold_grid':[.8,.9,.95,.98],'support_grid':[2,3,4],'consensus':'uniqueclassvoteamongconfidentteachers;tiesabstain',
          'baseline_development':{'correct':15,'covered':16,'wilson_lower_95':baseline},
          'guard':'coverage>=16,precision>=.95,andWilsonlower>=bestprevious15/16lower+.02',
          'selection':'largestWilsonlower,thencoverage,thenmoreteacheragreement,thenstricterthreshold',
          'final_gate':'selectedconsensusand>=2/3finalTRAINgraphs;humanprotected;noIDexceptions',
          'maximum_changes':40,'cap':'descendingminimumconfidentteacherprobability,thenagreement,stablekeyorder',
          'evaluation_limit':'34reusedforconfigurationselection/calibration;teacherscorrelated;WilsonisrankingONLY,notriskboundafterselection;graphsnotusedincalibration',
          'private_review_or_archive_read':False,'researcher_seen_prior_posthoc_audits':True,
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    names=['multifamily_assignment','object_zoom_neighbors','base_self_training','trusted_source_repair'];paths=[NOTES/f'.runtime/{name}_outputs.npz' for name in names]
    z=[np.load(p,allow_pickle=False) for p in paths];assert all(np.array_equal(a['keys'],keys) for a in z)
    zi=json.loads((NOTES/'json/object_zoom_neighbors_frozen.json').read_text());zw=max(range(len(zi['selected_heads'])),key=lambda i:zi['selected_heads'][i]['development_balanced_recall'])
    # Selftraining output stores the chosen policy's OOF probabilities.
    probabilities=np.stack([z[0]['probabilities'],z[1]['probabilities'][zw],z[2]['probabilities'],z[3]['oof_probabilities']])
    assert probabilities.shape==(4,n,8) and np.allclose(probabilities.sum(2),1,atol=1e-5)
    history=[]
    for threshold in plan['threshold_grid']:
        for support in plan['support_grid']:
            candidate,covered,score,votes=consensus(probabilities,threshold,support);count=int(covered[pos].sum());good=int((covered[pos]&(candidate[pos]==gold)).sum())
            precision=good/count if count else None;lower=wilson_lower(good,count) if count else 0.
            eligible=bool(count>=16 and precision>=.95 and lower>=baseline+.02)
            history.append({'threshold':threshold,'support':support,'covered':count,'correct':good,'precision':precision,'wilson_lower_95':lower,'eligible':eligible})
    feasible=[r for r in history if r['eligible']];chosen=max(feasible,key=lambda r:(r['wilson_lower_95'],r['covered'],r['support'],r['threshold'])) if feasible else None
    accepted=chosen is not None
    primary,covered,score,support=consensus(probabilities,chosen['threshold'] if chosen else .98,chosen['support'] if chosen else 4)
    # Smooth fixed teacher probabilities only for the final TRAIN graph. This
    # graph is corroboration, not the preceding selective calibration experiment.
    prior=probabilities.mean(0)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);zoom=np.load(NOTES/'.runtime/dinov2_object_zoom_features.npz',allow_pickle=False);cnn=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(a['keys'],keys) for a in [b,zoom,cnn])
    banks=[unit(np.c_[unit(b['features'][:,:768]),unit(b['features'][:,2304:])]),unit(np.c_[unit(zoom['features'][:,:768]),unit(zoom['features'][:,2304:])]),unit(np.c_[unit(cnn['features']),unit(b['features'][:,:768])])]
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(unit(x-x.mean(0)),prior,pos,gold) for x in banks])
    graph_support=(graphs==primary).sum(0);human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=covered&(graph_support>=2)&~human&(primary!=current);proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-support,-score));ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'covered':covered,'support':support,'minimum_confident_share':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=prior,consensus_class=primary,covered=covered,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_selective_consensus_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'family':name} for name in names],qualified_heads=[{'family':name} for name in names],selected_knn_heads=[],
        human_leave_source_group_out_matches=int((prior[pos].argmax(1)==gold).sum()),human_holdout_rows=34,development_balanced_recall=None,human_diagnostic_protocol=plan['evaluation_limit'],confidence_calibration={'selected':chosen,'history':history,'independent_validation':False,'objective':'selectiveconsensusnotglobal'},
        accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],component_output_sha256={p.name:sha(p) for p in paths})
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline']},indent=2),flush=True)


if __name__=='__main__':main()
