"""Selective grouped kNN development without nearest-human extrapolation veto.

Calibration uses only the existing34budgeted sources. Graph scores are not used
to assess calibration because those full-train graphs clamp all human sources.
"""
import json
import datetime
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    prefix='base_calibrated';source_prefix='base_clean_propagation'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    prior_path=NOTES/f'json/{source_prefix}_frozen.json';prior=json.loads(prior_path.read_text())
    heads=prior['selected_knn_heads'];winner=max(range(len(heads)),key=lambda i:heads[i]['development_balanced_recall'])
    best=heads[winner]['development_balanced_recall']
    trusted=[i for i,h in enumerate(heads) if h['development_balanced_recall']>=.6333333333333333 and h['development_balanced_recall']>=best-.1]
    graph_names=[h['bank'] for h in prior['graph_history']];graph_ids=[i for i,n in enumerate(graph_names) if n in [heads[k]['bank'] for k in trusted]]
    plan={'method':'grouped_development_selective_base_neighbor_refinement_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'source_knn_frozen_sha256':sha(prior_path),'expert_selection':'balancedrecall>=.6333333333333333andwithin.1ofbest;atleast2banks',
          'qualified_heads':[heads[i] for i in trusted],
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'largestcoverageon34groupheldoutanchors,subjectto>=10coveredand>=.9empiricalcorrect;tiespreferhigherthreshold',
          'score':'minimum candidate weighted-neighbor share across qualifiedbanks;neighbor decisions agree whenever score>.5',
          'acceptance':'humanprotected;bothqualifiedknnheadsmeet learnedthreshold andALLtheirfullTRAINgraphsagree;nearesthumanidentitynotrequired',
          'previous_auto_refinements':'also require minimumshare>=max(.95,learnedthreshold)',
          'maximum_changes':40,'cap':'descending minimumneighbor share;stablekeyorder tie',
          'guard':'primary>=20/34andbalancedrecall>=.6333333333333333;>=2qualifiedheadsand2graphs;calibrationthresholdmustexist',
          'evaluation_limit':'calibrationandexpertselectionreusethe34developmentanchors;notindependentfinalaccuracy;fullgraphnotusedforcalibrationevaluation',
          'private_review_or_archive_labels_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    z=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);assert np.array_equal(z['keys'],keys)
    cls=unit(z['features'][:,:768]);banks={'base_cls':cls,'base_local':unit(np.c_[cls,unit(z['features'][:,2304:])])}
    assert all(heads[i]['bank'] in banks for i in trusted)
    auto=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    gf=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=gf.group.to_numpy(),gf.fold.to_numpy()
    seed=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seed.key]);gold=np.array([CLASSES.index(c) for c in seed.assigned_class])
    assert len(pos)==34 and seed.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in auto.automatic_class]);y[pos]=gold
    core=auto.automatic_anchor.to_numpy(bool)|human;base=np.clip(auto.reliable_vote_count.to_numpy(float)/4,.25,1.)
    cached=np.load(NOTES/f'.runtime/{source_prefix}_outputs.npz',allow_pickle=False);assert np.array_equal(cached['keys'],keys)
    primary=cached['knn_predictions'][winner];scores=[];probabilities=[]
    with threadpool_limits(limits=4):
        for h in trusted:
            cfg=heads[h];x=banks[cfg['bank']];sim=x@x.T;p=np.zeros((n,8),np.float64)
            for fold in range(5):
                fit=np.flatnonzero((folds!=fold)&core);query=np.flatnonzero(folds==fold)
                assert not set(groups[fit])&set(groups[query])
                anchors=fit[human[fit]]
                near=np.stack([sim[np.ix_(fit,anchors[y[anchors]==c])].max(1) if (y[anchors]==c).any() else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
                cat=near.argmax(1);value=near[np.arange(len(fit)),cat].copy();near[np.arange(len(fit)),cat]=-np.inf
                remove=(cat!=y[fit])&(value>=.65)&(value-near.max(1)>=.02)&~human[fit];fit=fit[~remove]
                sub=sim[np.ix_(query,fit)];nearest=np.argsort(-sub,axis=1,kind='stable')[:,:cfg['neighbors']]
                val=np.take_along_axis(sub,nearest,axis=1);weight=base[fit].copy();weight[human[fit]]=cfg['human_weight']
                w=np.exp((val-val.max(1,keepdims=True))/.07)*weight[nearest]
                pr=np.stack([(w*(y[fit][nearest]==c)).sum(1) for c in range(8)],axis=1);pr/=pr.sum(1,keepdims=True)
                p[query]=pr
            assert np.array_equal(p.argmax(1),cached['knn_predictions'][h])
            scores.append(p[np.arange(n),primary]);probabilities.append(p)
    score=np.min(scores,axis=0);correct=primary[pos]==gold;history=[]
    for threshold in plan['threshold_grid']:
        selected=score[pos]>=threshold;count=int(selected.sum());matches=int((selected&correct).sum())
        history.append({'threshold':threshold,'covered':count,'correct':matches,'development_selective_precision':matches/count if count else None,
                        'eligible':count>=10 and matches/count>=.9})
    choices=[h for h in history if h['eligible']]
    chosen=max(choices,key=lambda h:(h['covered'],h['threshold'])) if choices else None
    threshold=chosen['threshold'] if chosen else float('inf')
    graph_support=(cached['graph_predictions'][graph_ids]==primary).sum(0)
    preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=(score>=threshold)&(graph_support==len(graph_ids))&~human&(primary!=current)
    gate&=~preserve|(score>=max(.95,threshold))
    proposed=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    matches=int(correct.sum());accepted=chosen is not None and matches>=20 and best>=.6333333333333333 and len(trusted)>=2 and len(graph_ids)>=2
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_neighbor_share':score,'qualified_graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(probabilities),accepted=gate)
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');report=dict(prior)
    report.update(status='frozen_before_private_selective_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head=heads[winner],primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),human_leave_source_group_out_matches=matches,
                  development_balanced_recall=best,confidence_calibration={'selected':chosen,'history':history,'independent_validation':False},
                  accepted_by_development_guard=accepted,proposed_new_labels=proposed,changed_rows_vs_baseline=int(gate.sum()),
                  new_vit_propagation_changes=int(gate.sum()),source_counts=result.category_source.value_counts().to_dict(),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),
                  source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),
         'guard':plan['guard'],'new_category_proposals':proposed,'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,
         'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()
