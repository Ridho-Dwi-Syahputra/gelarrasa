"""Fold-contained repair of noisy automatic targets from34genuine sources.

Unlike previous teacher self-training, correction targets come directly from
human-source neighborhoods, not from a teacher dominated by the old cold core.
No reviewed mismatch IDs or archive targets are read. Development is reused.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    start=time.time();prefix='trusted_source_repair'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'fold_contained_genuine_source_neighborhood_core_repair_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'original_targets':'coldautomaticconsensusonly,notactiveprogramcategories',
          'repair':'TRAINonly;nearestgenuinehumanclasshascos>=.65andmargin>=.02in>=2/3banksandsameclass;humanprotected',
          'policies':['repair_core','repair_and_add'],'pseudo_weight':.5,'human_weight':8.,
          'banks':['base_regional_centered','cnn_base','semantic_color_centered'],'statistics':'centeringfitouterTRAINonly',
          'neighbors':7,'temperature':.07,'selection':'best34group-heldoutmacrorecall,thenmatches;perbankor3bankmean;2policies',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],'threshold_selection':'maxcoverage>=10andprecision>=.9;tiesstricter',
          'acceptance':'thresholdand2/3teacherbankconsensusand2/3finalTRAINgraphs;humanprotected',
          'guard':'>=25/34andmacro>.75+1e-12andthresholdexists','maximum_changes':40,'cap':'descendingprimaryprobability,stablekeyorder',
          'private_review_or_archive_read_in_fit':False,'researcher_has_seen_prior_aggregate_posthoc_audit':True,
          'evaluation_limit':'34reusedfordevelopment;notuntouchedAccuracy;finalgraphsnotcalibrationvalidation',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];human=np.zeros(n,bool);human[pos]=True
    y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold;core=a.automatic_anchor.to_numpy(bool)|human;base_weight=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert np.array_equal(b['keys'],keys) and np.array_equal(c['keys'],keys)
    h=np.load(ROOT/'data/processed/features.npz',allow_pickle=False);hi=dict(zip(h['keys'],range(len(h['keys']))));color=unit(h['features'][[hi[k] for k in keys],1764:])
    cls=unit(b['features'][:,:768]);regional=unit(np.c_[cls,unit(b['features'][:,2304:])])
    banks=[regional,unit(np.c_[unit(c['features']),cls]),unit(np.c_[regional,.35*color])]
    policies=plan['policies'];oof={p:np.zeros((3,n,8)) for p in policies};history=[]

    def repair(features,train):
        hp=train[human[train]];pred=[];qualified=[]
        for x in features:
            near=np.stack([(x[train]@x[hp[y[hp]==k]].T).max(1) if (y[hp]==k).any() else np.full(len(train),-np.inf) for k in range(8)],1)
            candidate=near.argmax(1);best=near[np.arange(len(train)),candidate].copy();near[np.arange(len(train)),candidate]=-np.inf
            pred.append(candidate);qualified.append((best>=.65)&(best-near.max(1)>=.02))
        votes=np.stack([((np.stack(pred)==k)&np.stack(qualified)).sum(0) for k in range(8)],1)
        candidate=votes.argmax(1);allowed=(votes.max(1)>=2)&~human[train]
        return candidate,allowed

    def vote(x,fit,query,target,weight,exclude_group=False):
        sim=x[query]@x[fit].T
        if exclude_group:sim[groups[query,None]==groups[fit][None,:]]=-np.inf
        nn=np.argsort(-sim,axis=1,kind='stable')[:,:7];values=np.take_along_axis(sim,nn,1);assert np.isfinite(values).all()
        w=np.exp((values-values.max(1,keepdims=True))/.07)*weight[fit[nn]]
        p=np.stack([(w*(target[fit[nn]]==k)).sum(1) for k in range(8)],1);return p/p.sum(1,keepdims=True)

    def learn(train,query,final=False):
        features=[unit(x-x[train].mean(0)) if i!=1 else x for i,x in enumerate(banks)]
        candidate,allowed=repair(features,train);result={}
        for policy in policies:
            target=y.copy();weight=base_weight.copy();weight[human]=8.;eligible=core.copy()
            modify=allowed&((candidate!=y[train])|~core[train]);modify&=core[train] if policy=='repair_core' else True
            changed=train[modify];target[changed]=candidate[modify];weight[changed]=.5;eligible[changed]=True
            fit=train[eligible[train]];assert not final or len(train)==n
            if not final:assert not set(groups[fit])&set(groups[query])
            result[policy]=np.stack([vote(x,fit,query,target,weight,exclude_group=final) for x in features])
            history.append({'fold':'fullTRAIN' if final else int(folds[query[0]]),'policy':policy,'fit':len(fit),'core_repairs':int((modify&core[train]).sum()),'added':int((modify&~core[train]).sum()),'human_fit':int(human[fit].sum())})
        return result,features

    with threadpool_limits(limits=3):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);result,_=learn(train,query)
            for policy in policies:oof[policy][:,query]=result[policy]
        candidates=[]
        for policy,values in oof.items():
            for bank,p in enumerate([*values,values.mean(0)]):
                candidates.append(({'policy':policy,'bank':bank,'anchor_matches':int((p[pos].argmax(1)==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,p[pos].argmax(1)))},p))
        chosen,p_oof=max(candidates,key=lambda v:(v[0]['development_balanced_recall'],v[0]['anchor_matches']));calibration=[]
        for t in plan['threshold_grid']:
            covered=p_oof[pos].max(1)>=t;count=int(covered.sum());good=int((covered&(p_oof[pos].argmax(1)==gold)).sum());calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9})
        feasible=[r for r in calibration if r['eligible']];cal=max(feasible,key=lambda r:(r['covered'],r['threshold'])) if feasible else None
        accepted=bool(chosen['anchor_matches']>=25 and chosen['development_balanced_recall']>.75+1e-12 and cal is not None)
        final,features=learn(np.arange(n),np.arange(n),True);values=final[chosen['policy']];p=values[chosen['bank']] if chosen['bank']<3 else values.mean(0)
        primary=p.argmax(1);score=p.max(1);teacher_support=(values.argmax(2)==primary).sum(0)
        graphs=np.stack([graph_prediction(x,p,pos,gold) for x in features]);graph_support=(graphs==primary).sum(0)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=~human&(primary!=current)&(score>=(cal['threshold'] if cal else np.inf))&(teacher_support>=2)&(graph_support>=2)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame([c[0] for c in candidates]).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,oof_probabilities=p_oof,bank_probabilities=values,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_trusted_source_repair_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'bank':name} for name in plan['banks']],selected_knn_heads=[],qualified_heads=[{'bank':name} for name in plan['banks']],
        human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],human_diagnostic_protocol=plan['evaluation_limit'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
