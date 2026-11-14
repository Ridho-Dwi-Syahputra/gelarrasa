"""Source-held-out pairwise material experts from the same34 genuine train labels.

The user review table and archive annotations are not read. A pair expert may
route a machine-assigned category only if at least two feature banks support
the opposite category and only one such opposite category is supported.
"""
import datetime,json,time,itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit

def main():
    start=time.time();prefix='paired_material_router'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'all_eligible_pairwise_material_specialists_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'pairs':'allunorderedclasspairswith>=3genuinesourcesperclass;foodwaste1cannotbequalified',
          'banks':['DINOv3CLS_whitened','fullsourceposterior_centered','YOLOstatistics_plus_DINOCLS_centered'],
          'models':['linear_C1','rbf_gamma2_C1','rbf_gamma4_C1','rbf_gamma4_C10'],
          'pair_guard':'source-group-heldout balancedaccuracy>=.8and>=60pctcorrectperclass;atleast2qualifiedbanks',
          'router':'machinebasecategorymustbeoneofpair;atleast2banksfavoroppositewithSVMmargin>=threshold;uniqueoppositecategory',
          'thresholds':[0.,.25,.5,1.], 'selection':'best34developmentglobalmatches,macro,partialprecision,coverage,threshold',
          'global_guard':'>=29/34andmacro>=.85OR>=28/34macro>=.85and>=5routedhumanwith>=.9precision',
          'maximum_changes':20,'cap':'descendingsecondstrongestbankmargin,stablekeyorder',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'34reuseddevelopment,manypaircomparisons,classconditionalroutingnotindependentAccuracy',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    geo=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=geo.group.to_numpy(),geo.fold.to_numpy();human=np.zeros(n,bool);human[pos]=True
    v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);src=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False);yolo=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [v3,src,yolo])
    cls=unit(v3['features'][:,:768]);source=unit(np.c_[unit(src['features'][:,0]),unit(src['features'][:,1])]);det=unit(np.sqrt(np.maximum(yolo['detection_stats'],0)));banks={'DINOv3CLS_whitened':cls,'fullsourceposterior_centered':source,'YOLOstatistics_plus_DINOCLS_centered':unit(np.c_[det,.5*cls])}
    controls=[('linear',1.,0.),('rbf',1.,2.),('rbf',1.,4.),('rbf',10.,4.)]
    pairs=[(a,b) for a,b in itertools.combinations(range(8),2) if (gold==a).sum()>=3 and (gold==b).sum()>=3]
    def transform(raw,fit,name):
        if name.startswith('DINOv3CLS'):
            pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108).fit(raw[fit]);return unit(pca.transform(raw)/np.maximum(np.sqrt(pca.explained_variance_),1e-6))
        return unit(raw-raw[fit].mean(0))
    def scores(x,references,query,positive,kind,C,gamma):
        rr=x[references]@x[references].T;qr=x[query]@x[references].T
        if kind=='rbf':rr=np.exp(-2*gamma*np.maximum(1-rr,0));qr=np.exp(-2*gamma*np.maximum(1-qr,0))
        classifier=SVC(kernel='precomputed',C=C,class_weight='balanced').fit(rr,positive)
        return classifier.decision_function(qr)
    primary=np.zeros(n,int)
    vout=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz',allow_pickle=False);info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);primary[:]=vout['probabilities'][wi].argmax(1)
    aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz',allow_pickle=False);ap,am,ar=aug['oof_predictions'],aug['oof_margins'],aug['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac;acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45);primary=np.where(acov,ac,primary)
    r=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz',allow_pickle=False);ri=json.loads((NOTES/'json/residual_graph_correction_frozen.json').read_text());rwi=max(range(len(ri['selected_heads'])),key=lambda i:(ri['selected_heads'][i]['development_balanced_recall'],ri['selected_heads'][i]['anchor_matches']));rp=r['probabilities'];rc=rp[rwi].argmax(1);rs=np.min([p[np.arange(n),rc] for p in rp],0);primary=np.where((rs>=.6)&~(acov&(ac!=rc)),rc,primary)
    assert int((primary[pos]==gold).sum())==28
    development=[];pair_records=[];pair_margins={};pair_qualified={}
    with threadpool_limits(limits=3):
        for name,raw in banks.items():
            folds_x={fold:transform(raw,np.flatnonzero(folds!=fold),name) for fold in range(5)}
            final_x=transform(raw,np.arange(n),name)
            for a,b in pairs:
                oof=np.zeros((len(controls),n),np.float32)
                for fold,x in folds_x.items():
                    hs=np.flatnonzero((folds[pos]!=fold)&np.isin(gold,[a,b]));ref=pos[hs];q=np.flatnonzero(folds==fold)
                    if len(np.unique(gold[hs]))<2:continue
                    target=(gold[hs]==b).astype(int)
                    for c,(kind,C,gamma) in enumerate(controls):oof[c,q]=scores(x,ref,q,target,kind,C,gamma)
                human_mask=np.isin(gold,[a,b]);pidx=pos[human_mask];y=(gold[human_mask]==b).astype(int)
                options=[]
                for c,(kind,C,gamma) in enumerate(controls):
                    pred=oof[c,pidx]>0;rec_a=float((~pred[y==0]).mean());rec_b=float(pred[y==1].mean());macro=(rec_a+rec_b)/2
                    options.append({'bank':name,'pair':f'{CLASSES[a]}|{CLASSES[b]}','control':c,'kind':kind,'C':C,'gamma':gamma,'recall_a':rec_a,'recall_b':rec_b,'macro':macro})
                best=max(options,key=lambda row:(row['macro'],min(row['recall_a'],row['recall_b']),-row['control']));qualified=best['macro']>=.8 and min(best['recall_a'],best['recall_b'])>=.6
                best['qualified']=bool(qualified);development.extend(options);pair_records.append(best)
                if not qualified:continue
                c=best['control'];kind,C,gamma=controls[c];hs=np.flatnonzero(np.isin(gold,[a,b]));final=scores(final_x,pos[hs],np.arange(n),(gold[hs]==b).astype(int),kind,C,gamma)
                pair_margins.setdefault((a,b),[]).append((name,oof[c],final));pair_qualified[(a,b)]=len(pair_margins[(a,b)])
            print('Pair bank',name,'qualified',sum(v['bank']==name and v['qualified'] for v in pair_records),flush=True)
    pd.DataFrame(pair_records).to_csv(NOTES/f'table-csv/{prefix}_pair_heads.csv',index=False)
    pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    basecurrent=np.array([CLASSES.index(c) for c in frame.assigned_class])
    def route(base,use_final,threshold):
        evidence=np.full((n,8),-np.inf,np.float32);support=np.zeros((n,8),int)
        for (a,b),items in pair_margins.items():
            if len(items)<2:continue
            values=np.stack([item[2] if use_final else item[1] for item in items])
            for old,new,sign in [(a,b,1),(b,a,-1)]:
                target=base==old;v=sign*values[:,target];sorted_margin=np.sort(v,axis=0);second=sorted_margin[-2];valid=second>=threshold
                idx=np.flatnonzero(target)[valid];evidence[idx,new]=np.maximum(evidence[idx,new],second[valid]);support[idx,new]+=1
        qualifies=evidence>=threshold;single=qualifies.sum(1)==1;candidate=evidence.argmax(1);chosen=np.where(single,candidate,base);score=evidence[np.arange(n),candidate];return chosen,single,score,support
    calibration=[]
    for threshold in plan['thresholds']:
        pred,routed,score,_=route(primary,False,threshold);selected=routed[pos];count=int(selected.sum());correct=int((selected&(pred[pos]==gold)).sum());matches=int((pred[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,pred[pos]));precision=correct/count if count else 0.
        eligible=matches>=29 and macro>=.85-1e-12 or matches>=28 and macro>=.85-1e-12 and count>=5 and precision>=.9
        calibration.append({'threshold':threshold,'anchor_matches':matches,'macro':macro,'covered':count,'correct':correct,'precision':precision,'eligible':bool(eligible)})
    choices=[r for r in calibration if r['eligible']];chosen=max(choices,key=lambda r:(r['anchor_matches'],r['macro'],r['precision'],r['covered'],r['threshold'])) if choices else calibration[0];accepted=bool(choices)
    candidate,routed,score,support=route(basecurrent,True,chosen['threshold']);gate=routed&~human&(candidate!=basecurrent);proposals=int(gate.sum());order=np.argsort(-score,kind='stable');take=order[gate[order]][:20];gate[:]=False;gate[take]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[candidate[i]],'genuine_pairwise_material_snapshot',float(1/(1+np.exp(-score[i]))),'uncalibrated_svm_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[candidate],'routed':routed,'second_bank_margin':score,'supporting_pairs':support[np.arange(n),candidate],'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_pairwise_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=pair_records,qualified_heads=[r for r in pair_records if r['qualified']],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['macro'],confidence_calibration={'selected':chosen,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'pairs':len(pairs),'qualified_pair_heads':len(report['qualified_heads']),'chosen':chosen,'calibration':calibration,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()
