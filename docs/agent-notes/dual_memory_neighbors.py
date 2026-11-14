"""Separate reliable human and noisy automatic neighbor memories, grouped OOF.

The global-k7control can ignore a human anchor regardless of its weight if no
human enters k7. Stratification and source dedup are separate controlled policies.
No full review/visual archive/test targets are used;34development labels reused.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def neighbors(similarity,pool,k,groups=None):
    limit=min(len(pool),k if groups is None else max(64,k*8))
    local=np.argpartition(-similarity,limit-1,axis=1)[:,:limit]
    order=np.argsort(-np.take_along_axis(similarity,local,1),axis=1,kind='stable');local=np.take_along_axis(local,order,1)
    if groups is None:return pool[local[:,:k]],np.take_along_axis(similarity,local[:,:k],1)
    chosen=np.empty((len(similarity),k),int);values=np.empty((len(similarity),k),float)
    for row in range(len(similarity)):
        seen=set();selected=[]
        for j in local[row]:
            group=groups[pool[j]]
            if group not in seen:seen.add(group);selected.append(int(j))
            if len(selected)==k:break
        if len(selected)<k:
            for j in np.argsort(-similarity[row],kind='stable'):
                group=groups[pool[j]]
                if group not in seen:seen.add(group);selected.append(int(j))
                if len(selected)==k:break
        assert len(selected)==k;chosen[row]=pool[selected];values[row]=similarity[row,selected]
    return chosen,values


def main():
    start=time.time();prefix='dual_memory_neighbors'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    configs=[{'policy':policy,'temperature':temp,'human_weight':hw} for policy in ['global_k7','stratified_memories','group_stratified_memories'] for temp in [.07,.15] for hw in [8.,32.]]
    plan={'method':'reliable_human_and_noisy_automatic_dual_memory_knn_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'hypothesis':'upweightingcannothelpanhumananchorexcludedfromglobaltopk;queryseparatehumanmemory;deduplicatephotographicsourcevotes',
          'configs':configs,'auto_neighbors':7,'human_neighbors':3,'source_dedup':'separatecontrolledpolicy;closestrepresentativeeachgeometrysourcegroup',
          'banks':['DINOregional','DINO_source_dictionary','CNN_source_dictionary'],'statistics':'centeringfitouterTRAINonly',
          'targets':'coldautomaticconsensus+same34human;activeprogramcategoriesunusedastargets',
          'core_filter':'fit-onlynearesthumancategorycontradictionrawcos>=.65andmargin>=.02;humanneverfiltered',
          'folds':'fivegeometrysourcefolds;querynotinanymemoryorstatistics',
          'selection':'best34developmentmacrothenmatchesperbank;globalprimary;best2bankswithin.1andmacro>=.65',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],'threshold_selection':'maxcoverage>=10withprecision>=.9;tiesstricter',
          'guard':'global>=26/34andmacro>.758333333334;ORrisk>=20matches,macro>=.65,coverage>=12,precision>=.95,Wilsonranking>=prior12/12+.02;2qualifiedbanksrequired',
          'acceptance':'minimumselectedbanksharemeetslearnedthresholdandbothselectedTRAINgraphsagree;humanprotected',
          'maximum_changes':20,'cap':'descendingminimumshare,stablekeyorder','evaluation_limit':'34reuseddevelopment,multiplecomparisons;notfinalAccuracy;graphnotcalibrationvalidation',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];human=np.zeros(n,bool);human[pos]=True
    y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold;core=a.automatic_anchor.to_numpy(bool)|human;confidence=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);d=np.load(NOTES/'.runtime/semantic_dictionary_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [b,d,c]);cls=unit(b['features'][:,:768]);regional=unit(np.c_[cls,unit(b['features'][:,2304:])]);t1=unit(d['features'][:,0]);t2=unit(d['features'][:,1])
    banks={'DINOregional':regional,'DINO_source_dictionary':unit(np.c_[cls,.35*t1,.35*t2]),'CNN_source_dictionary':unit(np.c_[unit(c['features']),.5*t1])}
    history=[];diagnostic=[];selected=[];probabilities=[];graph_features=[]
    with threadpool_limits(limits=2):
        for name,raw in banks.items():
            outputs=np.zeros((len(configs),n,8),float)
            for fold in range(5):
                train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);fit=train[core[train]];assert not set(groups[train])&set(groups[query])
                hp=fit[human[fit]];near=np.stack([(raw[fit]@raw[hp[y[hp]==k]].T).max(1) if (y[hp]==k).any() else np.full(len(fit),-np.inf) for k in range(8)],1)
                candidate=near.argmax(1);best=near[np.arange(len(fit)),candidate].copy();near[np.arange(len(fit)),candidate]=-np.inf
                fit=fit[~((candidate!=y[fit])&(best>=.65)&(best-near.max(1)>=.02)&~human[fit])]
                features=unit(raw-raw[train].mean(0));auto=fit[~human[fit]];hp=fit[human[fit]]
                sim=features[query]@features[fit].T;global_idx,global_value=neighbors(sim,fit,7)
                auto_sim=features[query]@features[auto].T;human_sim=features[query]@features[hp].T
                auto_idx,auto_value=neighbors(auto_sim,auto,7);human_idx,human_value=neighbors(human_sim,hp,3)
                da,va=neighbors(auto_sim,auto,7,groups);dh,vh=neighbors(human_sim,hp,3,groups)
                memories={'global_k7':(global_idx,global_value),'stratified_memories':(np.c_[auto_idx,human_idx],np.c_[auto_value,human_value]),'group_stratified_memories':(np.c_[da,dh],np.c_[va,vh])}
                human_query=human[query]
                diagnostic.append({'bank':name,'fold':fold,'human_queries':int(human_query.sum()),'human_queries_with_zero_human_global_neighbors':int((human_query&~human[global_idx].any(1)).sum()),'all_queries_with_zero_human_global_neighbors':int((~human[global_idx].any(1)).sum()),'query_rows':len(query)})
                for i,cfg in enumerate(configs):
                    nn,value=memories[cfg['policy']];weight=confidence[nn].copy();weight[human[nn]]=cfg['human_weight'];w=np.exp((value-value.max(1,keepdims=True))/cfg['temperature'])*weight
                    p=np.stack([(w*(y[nn]==k)).sum(1) for k in range(8)],1);outputs[i,query]=p/p.sum(1,keepdims=True)
            candidates=[]
            for i,cfg in enumerate(configs):
                p=outputs[i];pred=p[pos].argmax(1);record={'bank':name,**cfg,'anchor_matches':int((pred==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,pred))};history.append(record);candidates.append((record,p))
            chosen,p=max(candidates,key=lambda v:(v[0]['development_balanced_recall'],v[0]['anchor_matches']));selected.append(chosen);probabilities.append(p);graph_features.append(unit(raw-raw.mean(0)))
            print('Dual memory bank',chosen,'seconds',round(time.time()-start),flush=True)
        wi=max(range(len(selected)),key=lambda i:(selected[i]['development_balanced_recall'],selected[i]['anchor_matches']));chosen=selected[wi]
        qualified=sorted([i for i,v in enumerate(selected) if v['development_balanced_recall']>=.65 and v['development_balanced_recall']>=chosen['development_balanced_recall']-.1],key=lambda i:(-selected[i]['development_balanced_recall'],-selected[i]['anchor_matches']))[:2]
        primary=probabilities[wi].argmax(1);score=np.min([probabilities[i][np.arange(n),primary] for i in qualified],axis=0) if qualified else np.zeros(n)
        calibration=[]
        for threshold in plan['threshold_grid']:
            covered=score[pos]>=threshold;count=int(covered.sum());good=int((covered&(primary[pos]==gold)).sum());precision=good/count if count else 0.
            calibration.append({'threshold':threshold,'covered':count,'correct':good,'precision':precision,'eligible':count>=10 and precision>=.9})
        feasible=[v for v in calibration if v['eligible']];cal=max(feasible,key=lambda v:(v['covered'],v['threshold'])) if feasible else None
        global_ok=cal is not None and chosen['anchor_matches']>=26 and chosen['development_balanced_recall']>.758333333334
        risk_ok=cal is not None and chosen['anchor_matches']>=20 and chosen['development_balanced_recall']>=.65 and cal['covered']>=12 and cal['precision']>=.95 and wilson_lower(cal['correct'],cal['covered'])>=wilson_lower(12,12)+.02
        accepted=bool((global_ok or risk_ok) and len(qualified)>=2)
        graphs=np.stack([graph_prediction(x,p,pos,gold) for x,p in zip(graph_features,probabilities)])
    graph_support=(graphs[qualified]==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=~human&(primary!=current)&(score>=(cal['threshold'] if cal else np.inf))&(graph_support==len(qualified))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(diagnostic).to_csv(NOTES/f'table-csv/{prefix}_human_memory_diagnostic.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(probabilities),graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_dual_memory_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,qualified_heads=[selected[i] for i in qualified],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
        confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},human_diagnostic_protocol=plan['evaluation_limit'],accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
