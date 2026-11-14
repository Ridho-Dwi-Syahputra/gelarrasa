"""Fold-contained teacher/core repair and student kNN, image-only train data.

Each outer query source group is excluded from teacher targets, graph fitting,
normalizer statistics and all pseudo-label generation. Only34existing human
sources are targets. Full private review and visual annotation folders are unused.
"""
import json
import time
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def graph_prediction(x,prior,human_pos,human_gold):
    n=len(x);sim=x@x.T;np.fill_diagonal(sim,-np.inf);nn=np.argsort(-sim,axis=1,kind='stable')[:,:20]
    edge=np.maximum(np.take_along_axis(sim,nn,axis=1),0)**4
    adj=sparse.csr_matrix((edge.ravel(),(np.repeat(np.arange(n),20),nn.ravel())),shape=(n,n));adj=(adj+adj.T)*.5
    degree=np.asarray(adj.sum(1)).ravel();assert (degree>0).all();transition=sparse.diags(1/degree)@adj
    p=.8*prior+.2/8;p[human_pos]=np.eye(8)[human_gold];state=p.copy()
    for _ in range(1500):
        update=.98*(transition@state)+.02*p;update[human_pos]=p[human_pos];delta=np.abs(update-state).max();state=update
        if delta<1e-6:break
    return state.argmax(1)


def main():
    start=time.time();prefix='base_self_training'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'fold_contained_DINO_CNN_teacher_student_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'targets':'cold automatic consensus and34human;noactiveassignmentcategoryisafittingtarget',
          'teacher':'.75centeredDINOBaseCLSandROIknn7humanweight8 +.25mean4ConvNeXt/DINO/ViTknnbanks;fixedfromprior34development',
          'core_filter':'fitfoldonly,nearesthumancategorycontradictioncos>=.65andmargin>=.02',
          'teacher_fit_predictions':'leaveeachfitquerysourcegroupoutofcore/humaneighbors;humanclampedonlywithingraphfit',
          'pseudo':'fitrows only;teacherprob>=.95andbothfit-onlygraphsagree;addnoncore orrepaircontradictorynonhumancore',
          'policies':['add_only','add_and_repair'],'pseudo_weight':.5,'student':'samefeaturefamiliesandweights,updatedfitcore+acceptedpseudo',
          'normalization':'DINOcenteringfitouterTRAINonly;frozenotherfeatureblocksL2perrow',
          'selection':'maximum34anchor outergroupheldoutbalancedrecall;tiesmorematches;tiesadd_only',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoveragewith>=10coveredand>=.9empiricalselectiveprecision;tiesstricter',
          'acceptance':'learnedprobabilitythreshold,bothfinalTRAINgraphsagree,humanprotected;allnonhumaneligible',
          'guard':'balancedrecall>.75+1e-12and>=25/34matchesandthresholdexists',
          'maximum_changes':40,'cap':'descendingstudentprobability,stablekeyorder',
          'evaluation_limit':'34developmentanchorsreusedforselection/calibration;notindependentfinalAccuracy;finalgraphnotusedincalibrationevaluation',
          'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    bf=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);cf=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);vf=np.load(NOTES/'.runtime/waste_vit_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [bf,cf,vf])
    bc=unit(bf['features'][:,:768]);roi=unit(bf['features'][:,2304:]);cc=unit(cf['features']);vc=unit(vf['features'][:,:768])
    base=unit(np.c_[bc,roi]);cnn_banks=[cc,unit(np.c_[cc,bc]),unit(np.c_[cc,bc,roi]),unit(np.c_[cc,bc,vc])]
    cnn_config=json.loads((NOTES/'json/cnn_clean_propagation_frozen.json').read_text())['selected_knn_heads']
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups,folds=g.group.to_numpy(),g.fold.to_numpy();s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');idx=dict(zip(keys,range(n)));pos=np.array([idx[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    core=a.automatic_anchor.to_numpy(bool)|human;weight=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    outputs={policy:np.zeros((n,8),np.float64) for policy in plan['policies']};history=[]
    def vote(features,fit,query,target,sample_weight,config,exclude_query_group=False,filter_core=False):
        sim=features[query]@features[fit].T
        if filter_core:
            anchors=fit[human[fit]]
            near=np.stack([(features[fit]@features[anchors[y[anchors]==c]].T).max(1) if (y[anchors]==c).any() else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
            cat=near.argmax(1);value=near[np.arange(len(fit)),cat].copy();near[np.arange(len(fit)),cat]=-np.inf
            keep=~((cat!=target[fit])&(value>=.65)&(value-near.max(1)>=.02)&~human[fit]);fit=fit[keep];sim=sim[:,keep]
        if exclude_query_group:sim[groups[query,None]==groups[fit][None,:]]=-np.inf
        nn=np.argsort(-sim,axis=1,kind='stable')[:,:config['neighbors']];val=np.take_along_axis(sim,nn,axis=1)
        assert np.isfinite(val).all()
        w=sample_weight[fit].copy();w[human[fit]]=config['human_weight'];ww=np.exp((val-val.max(1,keepdims=True))/.07)*w[nn]
        p=np.stack([(ww*(target[fit][nn]==c)).sum(1) for c in range(8)],axis=1)
        return p/p.sum(1,keepdims=True)
    with threadpool_limits(limits=4):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);fit=train[core[train]]
            assert not set(groups[train])&set(groups[query])
            dx=unit(base-base[train].mean(0))
            # Fit-row predictions remove the row's entire source group. The
            # outer query cannot enter a teacher graph or pseudo-label pool.
            teacher_d=vote(dx,fit,train,y,weight,{'neighbors':7,'human_weight':8.},True,True)
            teacher_c=np.mean([vote(x,fit,train,y,weight,cfg,True,True) for x,cfg in zip(cnn_banks,cnn_config)],axis=0)
            teacher=.75*teacher_d+.25*teacher_c;candidate=teacher.argmax(1);confidence=teacher.max(1)
            hp=np.flatnonzero(human[train]);hy=y[train[hp]]
            graph_d=graph_prediction(dx[train],teacher_d,hp,hy)
            graph_c=graph_prediction(cnn_banks[1][train],teacher_c,hp,hy)
            stable=(confidence>=.95)&(graph_d==candidate)&(graph_c==candidate)&~human[train]
            for policy in plan['policies']:
                target=y.copy();sample_weight=weight.copy();eligible=core.copy()
                additions=stable&~core[train]
                repairs=stable&core[train]&(candidate!=y[train]) if policy=='add_and_repair' else np.zeros(len(train),bool)
                selected=train[additions|repairs];target[selected]=candidate[additions|repairs];sample_weight[selected]=.5;eligible[selected]=True
                final_fit=train[eligible[train]];assert not set(groups[final_fit])&set(groups[query])
                # The previously selected core-filter has already been applied
                # in teacher fitting. Student uses accepted pool without again
                # discarding pseudo examples because they differ from oldy.
                pdino=vote(dx,final_fit,query,target,sample_weight,{'neighbors':7,'human_weight':8.})
                pcnn=np.mean([vote(x,final_fit,query,target,sample_weight,cfg) for x,cfg in zip(cnn_banks,cnn_config)],axis=0)
                outputs[policy][query]=.75*pdino+.25*pcnn
                history.append({'fold':fold,'policy':policy,'fit_rows':len(final_fit),'pseudo_added':int(additions.sum()),'core_repaired':int(repairs.sum()),'human_train':int(human[train].sum()),'query_rows':len(query)})
            print('Fold-contained self-training fold',fold+1,'/5','seconds',round(time.time()-start),flush=True)
    development=[]
    for policy,p in outputs.items():
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        pred=p[pos].argmax(1);development.append({'policy':policy,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())})
    chosen=max(development,key=lambda v:(v['development_balanced_recall'],v['anchor_matches']));p=outputs[chosen['policy']];primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());matches=int((covered&correct).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    eligible=[r for r in calibration if r['eligible']];cal=max(eligible,key=lambda r:(r['covered'],r['threshold'])) if eligible else None
    threshold=cal['threshold'] if cal else float('inf');dx=unit(base-base.mean(0));graphs=np.stack([graph_prediction(dx,p,pos,gold),graph_prediction(cnn_banks[1],p,pos,gold)])
    graph_support=(graphs==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=(score>=threshold)&(graph_support==2)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    accepted=cal is not None and chosen['development_balanced_recall']>.75+1e-12 and chosen['anchor_matches']>=25
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'student_probability':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,graphs=graphs,accepted=gate)
    report=json.loads((NOTES/'json/multifamily_assignment_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_self_training_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,
                  primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=chosen['development_balanced_recall'],
                  confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,
                  proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
