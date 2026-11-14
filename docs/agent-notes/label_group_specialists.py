"""Low-shot material specialists trained on ONLY the34existing human sources.

All views of each outer query source group are excluded. Candidate/branch choices
reuse the34development anchors, so these are not independent final estimates.
"""
import json
import datetime
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def main():
    prefix='label_group_specialists'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    group_names=[['paper','plastic','cardboard'],['paper','plastic'],['paper','cardboard'],['glass','plastic'],['metal','electronic'],['paper','textile']]
    plan={'method':'genuine_source_material_group_SVM_specialists_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'specialist_targets':'ONLYmanual_seeds34,max5perclass;noautomaticlabelsandnoarchivelabels',
          'groups':group_names,'folds':'samefivegeometrysourcegroups;ALLaugmentedviewsofquerysourceexcluded',
          'banks':['dino_global','dino_regional','cnn','dino_cnn','material_hybrid','physical_scores'],
          'views':['original','augmented8whereavailable'],'source_weight':'1/numberofviews;oneuniquehumansourceisnot8independentlabels',
          'SVM_configs':[{'kernel':k,'gamma':g,'C':c} for k,g in [('linear',None),('rbf',4.)] for c in [1.,10.]],
          'decision_probability':'softmaxovrscores/.5,uncalibrated;binaryscores[-margin,+margin]',
          'branch_thresholds':[.55,.65,.75,.85,.95],
          'branch_qualification':'groupCVmacro>=.8;thresholdcovers>=halfgroupand>=1sourceeachclass,coveredprecision>=.9;whole34nondecrease',
          'selection':'pergroupbestwhole34macro,thenmatches,thenhigherthreshold;greedyatmost3brancheswithstrictwhole34improvement',
          'routing':'onlyifORIGINALbaseargmaxingroup;replacecurrentgroupmasswithspecialistconditionalprobabilities;nevercascadeactivationfromnewclass',
          'base':'previousgroupedmultifamily24/34probabilities,notcurrentprogramcategories',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoverageat>=.9empiricalprecisionand>=10covered;tiesstricter',
          'guard':'>=25/34andmacro>.75+1e-12,thresholdexistsand>=1strictlyimprovingbranch',
          'acceptance':'learnedthresholdandatleast1of3finalTRAINgraphsagree;humanprotected',
          'maximum_changes':40,'cap':'descendingcascadedprobability,stablekeyorder',
          'evaluation_limit':'repeated34developmentselection,notindependentfinalAccuracy;groupedfitdoesnotmakepostselectionunbiased;graphnotusedincalibration',
          'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all()
    lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    gf=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=gf.group.to_numpy(),gf.fold.to_numpy()
    bf=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);cf=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);aug=np.load(NOTES/'.runtime/human_source_aug_features.npz',allow_pickle=False)
    assert np.array_equal(bf['keys'],keys) and np.array_equal(cf['keys'],keys) and np.array_equal(aug['keys'],s.key.to_numpy(str))
    assert np.array_equal(aug['source_sha256'],frame.source_sha256.to_numpy(str)[pos])
    bg=unit(bf['features'][:,:768]);br=unit(np.c_[bg,unit(bf['features'][:,2304:])]);cg=unit(cf['features'])
    ag=unit(aug['features'][:,:,:768]);ar=unit(np.concatenate([ag,unit(aug['features'][:,:,2304:])],axis=-1));ac=unit(aug['cnn_features'])
    material=np.load(NOTES/'.runtime/minc_train_features.npz',allow_pickle=False);assert np.array_equal(material['keys'],keys)
    mf=unit(material['features'][:,1:].mean(1));mp=unit(np.sqrt(np.maximum(material['probabilities'][:,1:].mean(1),1e-8)))
    mh=unit(np.c_[bg,mf]);ph=unit(np.c_[bg,mp])
    banks={'dino_global':(bg,ag),'dino_regional':(br,ar),'cnn':(cg,ac),'dino_cnn':(unit(np.c_[bg,cg]),unit(np.concatenate([ag,ac],axis=-1))),
           'material_hybrid':(mh,mh[pos,None,:]),'physical_scores':(ph,ph[pos,None,:])}
    baseline=np.load(NOTES/'.runtime/multifamily_assignment_outputs.npz',allow_pickle=False);assert np.array_equal(baseline['keys'],keys)
    base=baseline['probabilities'];base_class=base.argmax(1);base_metric=float(balanced_accuracy_score(gold,base[pos].argmax(1)));base_matches=int((base[pos].argmax(1)==gold).sum())
    group_infos=[tuple(sorted(CLASSES.index(c) for c in names)) for names in group_names];candidates={k:[] for k in range(len(group_infos))};history=[]
    def apply(p,info,pair_probability):
        classes=info['classes'];mask=np.isin(base_class,classes)&(pair_probability.max(1)>=info['threshold'])
        q=p.copy();mass=p[:,classes].sum(1);q[np.ix_(mask,classes)]=mass[mask,None]*pair_probability[mask]
        assert np.allclose(q.sum(1),1,atol=1e-5);return q
    with threadpool_limits(limits=3):
        for bank,(x,views) in banks.items():
            for mode in ['original','augmented']:
                if mode=='augmented' and views.shape[1]==1:continue
                v=views[:,:1] if mode=='original' else views;nv=v.shape[1];train_x=v.reshape(-1,x.shape[1]);owner=np.repeat(np.arange(34),nv)
                linear_train=train_x@train_x.T;linear_query=x@train_x.T
                for cfg in plan['SVM_configs']:
                    kt=linear_train if cfg['kernel']=='linear' else np.exp(-2*cfg['gamma']*np.maximum(1-linear_train,0))
                    kq=linear_query if cfg['kernel']=='linear' else np.exp(-2*cfg['gamma']*np.maximum(1-linear_query,0))
                    for number,classes in enumerate(group_infos):
                        p=np.zeros((n,len(classes)),np.float64);valid=True
                        for fold in range(5):
                            src=np.flatnonzero((folds[pos]!=fold)&np.isin(gold,classes));fit=np.flatnonzero(np.isin(owner,src));query=np.flatnonzero(folds==fold)
                            assert not set(groups[pos[owner[fit]]])&set(groups[query])
                            if len(np.unique(gold[owner[fit]]))!=len(classes):valid=False;break
                            model=SVC(kernel='precomputed',C=cfg['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                            model.fit(kt[np.ix_(fit,fit)],gold[owner[fit]],sample_weight=np.full(len(fit),1/nv))
                            assert np.array_equal(model.classes_,classes)
                            scores=model.decision_function(kq[np.ix_(query,fit)])
                            if len(classes)==2:scores=np.stack([-scores,scores],axis=1)
                            p[query]=softmax(scores/.5,axis=1)
                        if not valid:continue
                        subset=np.isin(gold,classes);truth=gold[subset];source_p=p[pos[subset]];pred=np.array(classes)[source_p.argmax(1)]
                        quality=float(balanced_accuracy_score(truth,pred));correct=pred==truth
                        if quality<.8:continue
                        for threshold in plan['branch_thresholds']:
                            covered=source_p.max(1)>=threshold;count=int(covered.sum());good=int((covered&correct).sum())
                            if count<int(np.ceil(subset.sum()/2)) or not all((covered&(truth==c)).any() for c in classes) or good/count<.9:continue
                            info={'group':number,'classes':list(classes),'bank':bank,'views':mode,**cfg,'threshold':threshold,
                                  'group_development_macro':quality,'group_sources':int(subset.sum()),'group_covered':count,'group_correct_covered':good}
                            q=apply(base,info,p);macro=float(balanced_accuracy_score(gold,q[pos].argmax(1)));matches=int((q[pos].argmax(1)==gold).sum())
                            history.append({**info,'whole34_macro':macro,'whole34_matches':matches})
                            if macro>=base_metric-1e-12 and matches>=base_matches:candidates[number].append((info,p,macro,matches))
            print('Material specialist feature bank complete:',bank,flush=True)
    best_groups=[]
    for group,values in candidates.items():
        if values:best_groups.append(max(values,key=lambda v:(v[2],v[3],v[0]['group_development_macro'],v[0]['threshold'])))
    p=base.copy();current_macro=base_metric;current_matches=base_matches;selected=[];remaining=best_groups.copy()
    for _ in range(3):
        options=[]
        for info,prob,_,_ in remaining:
            q=apply(p,info,prob);macro=float(balanced_accuracy_score(gold,q[pos].argmax(1)));matches=int((q[pos].argmax(1)==gold).sum())
            if macro>current_macro+1e-12 or (abs(macro-current_macro)<1e-12 and matches>current_matches):options.append((macro,matches,info,prob,q))
        if not options:break
        macro,matches,info,prob,q=max(options,key=lambda v:(v[0],v[1]));selected.append(info);p=q;current_macro=macro;current_matches=matches
        remaining=[v for v in remaining if v[0]['group']!=info['group']]
    primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold;calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());good=int((covered&correct).sum())
        calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9})
    feasible=[v for v in calibration if v['eligible']];cal=max(feasible,key=lambda v:(v['covered'],v['threshold'])) if feasible else None
    accepted=cal is not None and current_matches>=25 and current_macro>.75+1e-12 and len(selected)>0;threshold=cal['threshold'] if cal else float('inf')
    texture=np.load(NOTES/'.runtime/convnext_texture_features.npz',allow_pickle=False);assert np.array_equal(texture['keys'],keys)
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(unit(br-br.mean(0)),p,pos,gold),graph_prediction(unit(np.c_[cg,bg]),p,pos,gold),graph_prediction(unit(np.c_[unit(texture['covariance']),bg]),p,pos,gold)])
    support=(graphs==primary).sum(0);human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=(score>=threshold)&(support>=1)&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'material_specialist_assignment_snapshot',score[i],'uncalibrated_specialist_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'cascaded_probability':score,'graph_support':support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_material_specialist_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head={'method':'material_group_SVM_cascade','branches':len(selected)},primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,selected_knn_heads=[],qualified_heads=selected,
                  human_leave_source_group_out_matches=current_matches,human_holdout_rows=34,development_balanced_recall=current_macro,
                  human_diagnostic_protocol='samegeometrygroupouterfoldforbaseandspecialists;allquerysourceaugmentationsexcluded',confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
                  accepted_by_development_guard=bool(accepted),proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','selected_heads','human_leave_source_group_out_matches','development_balanced_recall','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()
