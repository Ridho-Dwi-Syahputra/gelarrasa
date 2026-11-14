"""Source-group safe augmented exemplars: eight views, one vote per source."""
import json
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def main():
    start=time.time();prefix='augmented_source'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    ap=NOTES/'.runtime/human_source_aug_features.npz';assert sha(ap)==json.loads((NOTES/'json/human_source_aug_extraction.json').read_text())['feature_sha256']
    plan={'method':'source_invariant_augmented_exemplar_neighbors_v1','human_sources':34,'views_per_human_source':8,'new_manual_labels':0,'test_rows':0,
          'targets':'coldautomaticconsensusplus34existinghuman;notcurrentprogramcategories',
          'source_vote':'maxortop2meanviewcosineperhumansource;onevoteperuniquesource,not8independentvotes',
          'folds':'fivegeometrysourcegroupfolds,excludeALLviewsofquerysourcegroup',
          'pooling':['original','max_aug','top2_aug'],'core_policies':['full','canonical8_per_class'],
          'canonical_selection':'fitpartitiononly,highestautomaticvotes,thenclasscentrality;uniquesourcegroups;humanalwaysincluded',
          'configs':{'neighbors':[3,7,15],'human_weight':[8.,32.],'temperature':.07},
          'banks':['dino_global','dino_regional','cnn','dino_cnn'],
          'core_filter':'withinfitonly,nearesthumancategorycontradictioncos>=.65andmargin>=.02;humanprotected',
          'selection':'best34developmentmacrorecallperbank;thenbestindividualorpredefinedprobabilitypool',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoverageat>=.9empiricalprecisionand>=10covered;tiesstricter',
          'guard':'primary>=24/34andmacro>=.75,orselectiveWilsonrank>=previous15/16rank+.02with>=16coveredandmacro>=.65,>=20matches;thresholdrequired',
          'acceptance':'learnedprobabilitythresholdandatleast1of3finalTRAINgraphsagree;humanprotected',
          'maximum_changes':40,'cap':'descendingneighborprobability,stablekeyorder',
          'evaluation_limit':'34developmentanchorsreusedformanyconfigurationchoicesandcalibration;notindependentfinalAccuracy;Wilsonrankingnotriskcertificate;graphnotusedforcalibration',
          'augmentation_feature_sha256':sha(ap),'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);aug=np.load(ap,allow_pickle=False)
    assert np.array_equal(b['keys'],keys) and np.array_equal(c['keys'],keys)
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();assert np.array_equal(aug['keys'],s.key.to_numpy(str))
    lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(v) for v in s.assigned_class])
    assert np.array_equal(aug['source_sha256'],frame.source_sha256.to_numpy(str)[pos])
    bg=unit(b['features'][:,:768]);br=unit(np.c_[bg,unit(b['features'][:,2304:])]);cg=unit(c['features'])
    ag=unit(aug['features'][:,:,:768]);ar=unit(np.concatenate([ag,unit(aug['features'][:,:,2304:])],axis=-1));ac=unit(aug['cnn_features'])
    banks={'dino_global':(bg,ag),'dino_regional':(br,ar),'cnn':(cg,ac),'dino_cnn':(unit(np.c_[bg,cg]),unit(np.concatenate([ag,ac],axis=-1)))}
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups,folds=g.group.to_numpy(),g.fold.to_numpy();human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(v) for v in a.automatic_class]);y[pos]=gold
    core=a.automatic_anchor.to_numpy(bool)|human;reliable=a.reliable_vote_count.to_numpy(float);weight=np.clip(reliable/4,.25,1.)
    configs=[{'neighbors':k,'human_weight':w} for k in [3,7,15] for w in [8.,32.]]
    selected=[];probabilities=[];history=[]
    with threadpool_limits(limits=3):
        for name,(x,views) in banks.items():
            original=x@x.T;view_sim=(x@views.reshape(-1,x.shape[1]).T).reshape(n,34,8)
            augmented={'original':original[:,pos],'max_aug':view_sim.max(2),'top2_aug':np.sort(view_sim,axis=2)[:,:,-2:].mean(2)}
            candidate_outputs=[]
            for pooling,similarity_h in augmented.items():
                sim=original.copy();sim[:,pos]=similarity_h
                for policy in plan['core_policies']:
                    out=np.zeros((len(configs),n,8),np.float64)
                    for fold in range(5):
                        fit=np.flatnonzero((folds!=fold)&core);query=np.flatnonzero(folds==fold);assert not set(groups[fit])&set(groups[query])
                        if policy=='canonical8_per_class':
                            keep=list(fit[human[fit]]);used=set(groups[keep])
                            for klass in range(8):
                                pool=fit[(~human[fit])&(y[fit]==klass)];centrality=original[np.ix_(pool,pool)].mean(1) if len(pool) else np.empty(0)
                                order=np.lexsort((pool,-centrality,-reliable[pool]));chosen=[]
                                for j in pool[order]:
                                    if groups[j] not in used:chosen.append(j);used.add(groups[j])
                                    if len(chosen)==8:break
                                keep.extend(chosen)
                            fit=np.array(sorted(keep))
                        anchors=fit[human[fit]];near=np.stack([sim[np.ix_(fit,anchors[y[anchors]==k])].max(1) if (y[anchors]==k).any() else np.full(len(fit),-np.inf) for k in range(8)],axis=1)
                        category=near.argmax(1);value=near[np.arange(len(fit)),category].copy();near[np.arange(len(fit)),category]=-np.inf
                        remove=(category!=y[fit])&(value>=.65)&(value-near.max(1)>=.02)&~human[fit];fit=fit[~remove]
                        sub=sim[np.ix_(query,fit)];nn=np.argsort(-sub,axis=1,kind='stable')[:,:15]
                        for i,cfg in enumerate(configs):
                            nearest=nn[:,:cfg['neighbors']];value=np.take_along_axis(sub,nearest,axis=1);w=weight[fit].copy();w[human[fit]]=cfg['human_weight']
                            ww=np.exp((value-value.max(1,keepdims=True))/.07)*w[nearest]
                            p=np.stack([(ww*(y[fit][nearest]==k)).sum(1) for k in range(8)],axis=1);out[i,query]=p/p.sum(1,keepdims=True)
                    for i,cfg in enumerate(configs):
                        p=out[i];assert np.isfinite(p).all() and np.allclose(p.sum(1),1);pred=p[pos].argmax(1)
                        info={'bank':name,'view_pooling':pooling,'core_policy':policy,**cfg,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())}
                        history.append(info);candidate_outputs.append((info,p))
            best,p=max(candidate_outputs,key=lambda v:(v[0]['development_balanced_recall'],v[0]['anchor_matches']));selected.append(best);probabilities.append(p)
            print('Augmented source bank:',best,flush=True)
    combinations=[([1.,0,0,0],'dino_global'),([0,1.,0,0],'dino_regional'),([0,0,1.,0],'cnn'),([0,0,0,1.],'dino_cnn'),
                  ([.25,.25,.25,.25],'four_bank_mean'),([0,.5,.5,0],'regional_cnn_mean'),([0,.75,.25,0],'regional_cnn_75_25'),([.5,0,.5,0],'global_cnn_mean')]
    development=[]
    for weights,label in combinations:
        p=sum(w*x for w,x in zip(weights,probabilities));pred=p[pos].argmax(1)
        development.append({'combination':label,'weights':weights,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())})
    chosen=max(development,key=lambda v:(v['development_balanced_recall'],v['anchor_matches']));p=sum(w*x for w,x in zip(chosen['weights'],probabilities));primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());good=int((covered&correct).sum())
        calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9,'wilson_ranking_score':wilson_lower(good,count)})
    feasible=[v for v in calibration if v['eligible']];cal=max(feasible,key=lambda v:(v['covered'],v['threshold'])) if feasible else None
    global_guard=chosen['anchor_matches']>=24 and chosen['development_balanced_recall']>=.75
    selective_guard=cal is not None and cal['covered']>=16 and cal['wilson_ranking_score']>=wilson_lower(15,16)+.02 and chosen['development_balanced_recall']>=.65 and chosen['anchor_matches']>=20
    accepted=cal is not None and (global_guard or selective_guard);threshold=cal['threshold'] if cal else float('inf')
    tf=np.load(NOTES/'.runtime/convnext_texture_features.npz',allow_pickle=False);assert np.array_equal(tf['keys'],keys)
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(unit(br-br.mean(0)),p,pos,gold),graph_prediction(unit(np.c_[cg,bg]),p,pos,gold),graph_prediction(unit(np.c_[unit(tf['covariance']),bg]),p,pos,gold)])
    support=(graphs==primary).sum(0);current=np.array([CLASSES.index(v) for v in frame.assigned_class]);gate=(score>=threshold)&(support>=1)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_bank_development.csv',index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'neighbor_probability':score,'graph_support':support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(probabilities),pooled_probabilities=p,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_augmented_source_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,
                  primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,selected_knn_heads=[],qualified_heads=[selected[i] for i,w in enumerate(chosen['weights']) if w>0],
                  human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
                  human_diagnostic_protocol='fivegeometrygroupfolds;ALLaugmentedviewsofquerysourceexcluded',confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
                  accepted_by_development_guard=bool(accepted),proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
