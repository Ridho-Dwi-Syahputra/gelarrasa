"""Risk/coverage specialist selection from existing train-only grouped outputs.

Wilson scores rank development evidence only; reuse of34anchors and multiple
selection mean these are NOT independent statistical risk certificates. This
changes the objective explicitly; prior rejected experiments stay unchanged.
"""
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def wilson_lower(correct,n):
    if n==0:return 0.
    z=1.959963984540054;phat=correct/n
    return float((phat+z*z/(2*n)-z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/(1+z*z/n))


def main():
    prefix='selective_expert'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    baseline=json.loads((NOTES/'json/multifamily_assignment_frozen.json').read_text());base_cal=baseline['confidence_calibration']['selected']
    lower=wilson_lower(base_cal['correct'],base_cal['covered']);minimum_coverage=base_cal['covered'];minimum_balanced=baseline['development_balanced_recall']-.1
    plan={'method':'development_selective_specialist_refinement_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'objective_change':'selectivepseudo-labelquality/coverage,notmaximumwhole34anchoraccuracy;previousrejectionsarekept',
          'reference':'https://proceedings.mlr.press/v97/geifman19a.html','reference_limit':'risk-coverageconceptonly,notSelectivenetarchitectureorclaimedevaluationguarantee',
          'candidates':['current_multifamily','normalized_neighbor','base_neural','cnn_neighbor_mean','fold_contained_self_training'],
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'eligibility':{'minimum_development_matches':20,'minimum_balanced_recall':minimum_balanced,'minimum_coverage':minimum_coverage,'minimum_empirical_precision':.9},
          'selection':'maximum Wilson95lower-score;tie morecoverage;tie higherbalancedrecall;no private-review matching',
          'baseline_selective_evidence':base_cal,'baseline_lower_score':lower,
          'guard':'differentexpertwithWilsonlowerscore>=baseline+.02andcoverage>=baseline;floorwhole34performanceasabove',
          'acceptance':'selectedexpertlearnedthresholdandbothnewfinalTRAINgraphsunanimous;humanprotected;allnonhumaneligible',
          'maximum_changes':40,'cap':'descendingselectedprobability,stablekeyorder',
          'evaluation_limit':'allselection/calibrationuses34developmentanchors;Wilsonrankingisnotanindependentconfidenceguarantee;graphnotusedincalibration',
          'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    def load(name):
        p=NOTES/f'.runtime/{name}_outputs.npz';z=np.load(p,allow_pickle=False);assert np.array_equal(z['keys'],keys)
        return z,sha(p)
    mi,mh=load('multifamily_assignment');no,nh=load('normalized_base');ne,eh=load('base_neural');cn,ch=load('cnn_clean_propagation');st,sh=load('base_self_training')
    ninfo=json.loads((NOTES/'json/normalized_base_frozen.json').read_text());einfo=json.loads((NOTES/'json/base_neural_frozen.json').read_text())
    ni=max(range(len(ninfo['selected_heads'])),key=lambda i:ninfo['selected_heads'][i]['development_balanced_recall'])
    ei=max(range(len(einfo['selected_heads'])),key=lambda i:einfo['selected_heads'][i]['development_balanced_recall'])
    candidates={'current_multifamily':mi['probabilities'],'normalized_neighbor':no['probabilities'][ni],
                'base_neural':ne['probabilities'][ei],'cnn_neighbor_mean':cn['probabilities'].mean(0),'fold_contained_self_training':st['probabilities']}
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    history=[]
    for name,p in candidates.items():
        pred=p[pos].argmax(1);correct=pred==gold;balanced=float(balanced_accuracy_score(gold,pred));matches=int(correct.sum());score=p[pos].max(1)
        for t in plan['threshold_grid']:
            covered=score>=t;count=int(covered.sum());good=int((covered&correct).sum());precision=good/count if count else 0.
            eligible=matches>=20 and balanced>=minimum_balanced-1e-12 and count>=minimum_coverage and precision>=.9
            history.append({'expert':name,'threshold':t,'development_balanced_recall':balanced,'anchor_matches':matches,
                            'covered':count,'correct':good,'empirical_precision':precision,'wilson_ranking_score':wilson_lower(good,count),'eligible':eligible})
    feasible=[h for h in history if h['eligible']];chosen=max(feasible,key=lambda h:(h['wilson_ranking_score'],h['covered'],h['development_balanced_recall'])) if feasible else None
    accepted=chosen is not None and chosen['expert']!='current_multifamily' and chosen['wilson_ranking_score']>=lower+.02
    if chosen is None:chosen={'expert':'current_multifamily','threshold':float('inf'),'anchor_matches':baseline['human_leave_source_group_out_matches'],'development_balanced_recall':baseline['development_balanced_recall']}
    p=candidates[chosen['expert']];primary=p.argmax(1);score=p.max(1)
    bf=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);cf=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert np.array_equal(bf['keys'],keys) and np.array_equal(cf['keys'],keys)
    b=unit(np.c_[unit(bf['features'][:,:768]),unit(bf['features'][:,2304:])]);b=unit(b-b.mean(0));c=unit(np.c_[unit(cf['features']),unit(bf['features'][:,:768])])
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(b,p,pos,gold),graph_prediction(c,p,pos,gold)])
    graph_support=(graphs==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=(score>=chosen['threshold'])&(graph_support==2)&~frame.human_reviewed.to_numpy(bool)&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key');neural_choice=chosen['expert']=='base_neural'
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[
        CLASSES[primary[i]],'robust_neural_assignment_snapshot' if neural_choice else 'vit_knn_label_propagation_snapshot',score[i],
        'uncalibrated_neural_probability' if neural_choice else 'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'expert_probability':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,graphs=graphs,accepted=gate)
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');report=dict(baseline)
    report.update(status='frozen_before_private_selective_expert_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,
                  primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'expert':chosen['expert']}],selected_knn_heads=[],qualified_heads=[{'expert':chosen['expert']}],
                  human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=chosen['development_balanced_recall'],
                  confidence_calibration={'selected':chosen,'independent_validation':False,'wilson_is_ranking_only':True},
                  accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()) if neural_choice else 0,new_vit_propagation_changes=0 if neural_choice else int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),
                  component_output_sha256={'current':mh,'normalized':nh,'neural':eh,'cnn':ch,'selftraining':sh},source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()
