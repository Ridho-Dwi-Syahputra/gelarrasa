"""Fold-fitted centering/PCA-whitening hypotheses for image-only train features.

All statistics and labels are fit without the query source group. Only the34
existing human sources calibrate development; no private-review targets read.
"""
import json
import datetime
import time
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit
from selective_expert_refinement import wilson_lower


def main(use_texture=False):
    start=time.time();assert use_texture;prefix='full_distribution_neighbors'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'DINOv3Base_full_source_Hellinger_grouped_neighbors_v1' if use_texture else 'fold_fitted_centered_whitened_DINOv2Base_neighbors_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'hypothesis':'fullsourceprobabilitymassratherthantop128;streamedHellingerprojection+V3Base;noontologyorhiddenreviewtargets',
          'statistics':'mean andPCAfitonlynonqueryTRAINsourcegroups;frozenbackbone;queryneverinPCAfit',
          'banks':['full_source','v3_full_T1','v3_object_full_T2','v3_both_full'] if use_texture else ['base_cls','base_local'],'transforms':['center_only','pca128_half_whitening','pca128_whitening'],
          'knn_configs':{'neighbors':[3,7,15],'human_weight':[8.,32.],'temperature':.07},
          'automatic_filter':'originalunmodifiedcosine:exclude contradictory nearesthumancategoryat>=.65andmargin>=.02,fitfoldonly',
          'selection':'perbank maximum34anchor group-heldout balancedrecall;thenbestprimarybank;notprivate review matching',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'largest34developmentcoverageat>=.9empiricalselectiveprecisionand>=10covered;tiesstricter',
          'acceptance':'allqualifiedbanksmeetlearnedminimumneighbor-sharethresholdandalltheirTRAINgraphsagree;humanprotected' if use_texture else 'bothselectedbanks meetlearnedminimumneighbor-sharethreshold andbothfullTRAINgraphsagree;humanprotected',
          'previous_auto_refinements':'minimumshare>=max(.95,learnedthreshold)',
          'graph':'normalizerfitallTRAINforfinalgraphonly,k20,positivecos^4,alpha.98,OOFknnprior,humanclamped',
          'guard':'global:>=27/34andmacro>.783333333334;ORrisk:matches>=20,macro>=.65,covered>=14,precision>=.95,Wilsonranking>=prior14/14+.02;>=2qualifiedbanksandthresholdexists' if use_texture else 'primary>=23/34andbalancedrecall>=.725andcalibrationthresholdexists',
          'maximum_changes':20,'cap':'descending minimum neighbor share;stablekeyorder',
          'evaluation_limit':'34developmentanchorsreusedforselection/calibration;notindependentfinalAccuracy;graphnotindependentlyevaluated',
          'private_review_or_archive_read':False,'qualified_bank_policy':'best2bydevelopmentmacro;correlatednotindependent','source_feature_sha256':sha(NOTES/'.runtime/dinov3_base_train_features.npz'),'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    z=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);assert np.array_equal(z['keys'],keys)
    cls=unit(z['features'][:,:768]);banks={'base_cls':cls,'base_local':unit(np.c_[cls,unit(z['features'][:,2304:])])}
    texture_sha=None
    sp=NOTES/'.runtime/full_source_distribution_features.npz';sd=np.load(sp,allow_pickle=False);assert np.array_equal(sd['keys'],keys)
    object_mean=unit(z['features'][:,1536:2304]);valid_mean=unit(z['features'][:,768:1536]);object_std=unit(z['features'][:,2304:3072])
    source_t1=unit(sd['features'][:,0]);source_t2=unit(sd['features'][:,1]);texture_sha=sha(NOTES/'.runtime/dinov3_base_train_features.npz')
    banks={'full_source':source_t1,'v3_full_T1':unit(np.c_[cls,.35*source_t1]),
           'v3_object_full_T2':unit(np.c_[cls,object_mean,.35*source_t2]),'v3_both_full':unit(np.c_[cls,.35*source_t1,.35*source_t2])}
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    core=a.automatic_anchor.to_numpy(bool)|human;base=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    configs=[{'neighbors':k,'human_weight':w} for k in [3,7,15] for w in [8.,32.]]
    selections=[];probabilities=[];graphs=[];history=[]
    with threadpool_limits(limits=4):
        for name,x in banks.items():
            outputs={t:np.zeros((len(configs),n,8),np.float64) for t in plan['transforms']}
            for fold in range(5):
                train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);fit=train[core[train]]
                assert not set(groups[train])&set(groups[query])
                sim=x@x.T;anchors=fit[human[fit]]
                near=np.stack([sim[np.ix_(fit,anchors[y[anchors]==c])].max(1) if (y[anchors]==c).any() else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
                cat=near.argmax(1);value=near[np.arange(len(fit)),cat].copy();near[np.arange(len(fit)),cat]=-np.inf
                remove=(cat!=y[fit])&(value>=.65)&(value-near.max(1)>=.02)&~human[fit];fit=fit[~remove]
                mean=x[train].mean(0);pca=PCA(n_components=128,svd_solver='randomized',random_state=20261107+fold).fit(x[train])
                projected=pca.transform(x);scale=np.maximum(np.sqrt(pca.explained_variance_),1e-6)
                transforms={'center_only':unit(x-mean),'pca128_half_whitening':unit(projected/np.sqrt(scale)),
                            'pca128_whitening':unit(projected/scale)}
                for transform,features in transforms.items():
                    sub=features[query]@features[fit].T;nn=np.argsort(-sub,axis=1,kind='stable')[:,:15]
                    for i,cfg in enumerate(configs):
                        neighbors=nn[:,:cfg['neighbors']];val=np.take_along_axis(sub,neighbors,axis=1)
                        weight=base[fit].copy();weight[human[fit]]=cfg['human_weight'];w=np.exp((val-val.max(1,keepdims=True))/.07)*weight[neighbors]
                        pr=np.stack([(w*(y[fit][neighbors]==c)).sum(1) for c in range(8)],axis=1);outputs[transform][i,query]=pr/pr.sum(1,keepdims=True)
            best_score=-1;chosen=None;chosen_p=None
            for transform,values in outputs.items():
                for i,cfg in enumerate(configs):
                    pr=values[i];assert np.isfinite(pr).all() and np.allclose(pr.sum(1),1)
                    pred=pr[pos].argmax(1);quality=float(balanced_accuracy_score(gold,pred));matches=int((pred==gold).sum())
                    history.append({'bank':name,'transform':transform,**cfg,'development_balanced_recall':quality,'anchor_matches':matches})
                    if quality>best_score:best_score=quality;chosen=history[-1];chosen_p=pr
            selections.append(chosen);probabilities.append(chosen_p)
            print('Normalized kNN family:',chosen,flush=True)
            if chosen['transform']=='center_only':features=unit(x-x.mean(0))
            else:
                pca=PCA(n_components=128,svd_solver='randomized',random_state=20261107).fit(x);pr=pca.transform(x);scale=np.maximum(np.sqrt(pca.explained_variance_),1e-6)
                features=unit(pr/(np.sqrt(scale) if chosen['transform']=='pca128_half_whitening' else scale))
            sim=features@features.T;np.fill_diagonal(sim,-np.inf);nn=np.argsort(-sim,axis=1,kind='stable')[:,:20]
            edge=np.maximum(np.take_along_axis(sim,nn,axis=1),0)**4
            adj=sparse.csr_matrix((edge.ravel(),(np.repeat(np.arange(n),20),nn.ravel())),shape=(n,n));adj=(adj+adj.T)*.5
            degree=np.asarray(adj.sum(1)).ravel();assert (degree>0).all();transition=sparse.diags(1/degree)@adj
            prior=.8*chosen_p+.2/8;prior[pos]=np.eye(8)[gold];state=prior.copy()
            for _ in range(1500):
                update=.98*(transition@state)+.02*prior;update[pos]=prior[pos];delta=np.abs(update-state).max();state=update
                if delta<1e-6:break
            graphs.append(state.argmax(1))
    winner=max(range(len(selections)),key=lambda i:selections[i]['development_balanced_recall']);chosen=selections[winner]
    qualified=sorted([i for i,s in enumerate(selections) if s['development_balanced_recall']>=.65 and s['development_balanced_recall']>=chosen['development_balanced_recall']-.1],key=lambda i:(-selections[i]['development_balanced_recall'],-selections[i]['anchor_matches']))[:2]
    primary=probabilities[winner].argmax(1);score=np.min([probabilities[i][np.arange(n),primary] for i in qualified],axis=0) if qualified else np.zeros(n);correct=primary[pos]==gold
    calibration=[]
    for t in plan['threshold_grid']:
        selected=score[pos]>=t;count=int(selected.sum());matches=int((correct&selected).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    choices=[r for r in calibration if r['eligible']];cal=max(choices,key=lambda r:(r['covered'],r['threshold'])) if choices else None
    threshold=cal['threshold'] if cal else float('inf')
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    graph_support=(np.stack(graphs)[qualified]==primary).sum(0)
    gate=(score>=threshold)&(graph_support==len(qualified))&~human&(primary!=current)&(~preserve|(score>=max(.95,threshold)))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:20];gate[:]=False;gate[allowed]=True
    global_ok=cal is not None and chosen['anchor_matches']>=27 and chosen['development_balanced_recall']>.783333333334
    risk_ok=cal is not None and chosen['anchor_matches']>=20 and chosen['development_balanced_recall']>=.65 and cal['covered']>=14 and cal['precision']>=.95 and wilson_lower(cal['correct'],cal['covered'])>=wilson_lower(14,14)+.02
    accepted=(global_ok or risk_ok) and len(qualified)>=2
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_neighbor_share':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(probabilities),graphs=np.stack(graphs),accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_normalization_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selections,
                  texture_feature_sha256=texture_sha,qualified_heads=[selections[i] for i in qualified],
                  human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,source_feature_sha256=sha(NOTES/'.runtime/dinov3_base_train_features.npz'),feature_mode='dinov3_base',development_balanced_recall=chosen['development_balanced_recall'],
                  confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
                  accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--texture',action='store_true');main(True)
