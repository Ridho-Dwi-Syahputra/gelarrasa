"""Class-wise trusted-source specialists, with abstention for routing conflicts.

All target labels are the same34human sources. Baseline25/34is source-group OOF.
Per-class thresholds reuse those34sources; zero observed false positives is NOT
a risk guarantee after model/threshold selection. No full review targets read.
"""
import datetime
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    prefix='material_object_specialists'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'material_object_human_one_vs_rest_abstaining_router_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'targets':'ONLYsame34genuinehuman;noautomaticlabelsastargets;noarchiveorfullreview',
          'banks':['materialprob_V3','materialfeature_sourceposterior','litterfeature_V3'],'kernels':['linear','rbf_gamma4'],'C':[1.,10.],
          'thresholds':[0.,.25,.5,1.],'folds':'fivegeometrysourcegroups;allquerysourcegroupsheldout;centeringfitouterTRAINonly',
          'class_qualification':'atleast3humanpositivesexist;OOFpositivecoverage>=max(3,ceil(.6class_count));ZEROobservedfalsepositivesonallremaininghumanclasses',
          'per_bank_class_selection':'maximumpositivecoverage,thenhigherthreshold,thenlistedorder',
          'router':'atleast2/3bankqualifiedpositivesforEXACTLYoneclass;othercasesabstain;nocrossmodelmargincomparison',
          'baseline':'bestDINOv3BaseOOF26/34;strongerguard27andmacro.808333333333',
          'guard':'>=27/34andmacro>=.808333333333;>=3humanroutedandallroutedhumanpredictionscorrect;atleast1qualifiedclass',
          'final_acceptance':'uniquerouterand>=1/3finalTRAINgraphagreement;humanprotected',
          'graphs':'DINOregional,CNNdictionary,source_dictionary;baselineOOFpriorswith.98routerpseudotarget;notindependentvalidation',
          'score':'minimumsigmoidbinarymarginamongpositivebanks,uncalibrated;notprobabilityguarantee',
          'maximum_changes':20,'cap':'descendingminimumsigmoidmargin,thenbanksupport,stablekeyorder',
          'evaluation_limit':'34reuseddevelopmentandmultiplecomparisons;perclass0FPnotgeneralizationguarantee;finalgraphnotcalibrationvalidation',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    sd=np.load(NOTES/'.runtime/semantic_dictionary_features.npz',allow_pickle=False);b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [sd,b,c]);dictionary=unit(np.concatenate([unit(sd['features'][:,0]),unit(sd['features'][:,1])],1))
    regional=unit(np.c_[unit(b['features'][:,:768]),unit(b['features'][:,2304:])]);cnn=unit(np.c_[unit(c['features']),unit(sd['features'][:,0])])
    material=np.load(NOTES/'.runtime/minc_train_features.npz',allow_pickle=False);litter=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);full=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False)
    for source in [material,litter,v3,full]:assert np.array_equal(source['keys'],keys)
    cls=unit(v3['features'][:,:768]);mp=unit(np.sqrt(np.maximum(material['probabilities'][:,1:].mean(1),0)));mf=unit(material['features'][:,1:].mean(1));lf=unit(litter['features']);posterior=unit(np.c_[unit(full['features'][:,0]),unit(full['features'][:,1])])
    banks={'materialprob_V3':unit(np.c_[mp,.5*cls]),'materialfeature_sourceposterior':unit(np.c_[mf,.5*posterior]),'litterfeature_V3':unit(np.c_[lf,.5*cls])}
    basefile=NOTES/'.runtime/dinov3_base_neighbors_outputs.npz';base=np.load(basefile,allow_pickle=False);assert np.array_equal(base['keys'],keys)
    info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall'])
    baseline=base['probabilities'][wi];base_macro=float(balanced_accuracy_score(gold,baseline[pos].argmax(1)))
    selected=[];development=[];all_margins=np.full((3,8,n),-np.inf);all_gates=np.zeros((3,8,n),bool)
    with threadpool_limits(limits=3):
        for bank_no,(name,raw) in enumerate(banks.items()):
            folds_cache=[]
            for fold in range(5):
                source=np.flatnonzero(folds[pos]!=fold);query=np.flatnonzero(folds==fold);assert not set(groups[pos[source]])&set(groups[query])
                features=unit(raw-raw[folds!=fold].mean(0));train=features[pos[source]];query_x=features[query]
                folds_cache.append((source,query,train@train.T,query_x@train.T))
            for category in range(8):
                positives=int((gold==category).sum())
                if positives<3:continue
                candidates=[]
                for kernel in plan['kernels']:
                    for regularizer in plan['C']:
                        margin=np.full(n,-10.,np.float64)
                        for source,query,kt,kq in folds_cache:
                            target=(gold[source]==category).astype(int)
                            if len(np.unique(target))<2:continue
                            if kernel!='linear':train_kernel=np.exp(-8*np.maximum(1-kt,0));query_kernel=np.exp(-8*np.maximum(1-kq,0))
                            else:train_kernel,query_kernel=kt,kq
                            model=SVC(kernel='precomputed',C=regularizer,class_weight='balanced');model.fit(train_kernel,target);margin[query]=model.decision_function(query_kernel)
                        for threshold in plan['thresholds']:
                            covered=margin[pos]>=threshold;tp=int((covered&(gold==category)).sum());fp=int((covered&(gold!=category)).sum())
                            qualified=tp>=max(3,int(np.ceil(.6*positives))) and fp==0
                            record={'bank':name,'class':CLASSES[category],'kernel':kernel,'C':regularizer,'threshold':threshold,'human_class_sources':positives,'covered_positive':tp,'false_positive':fp,'qualified':qualified}
                            development.append(record)
                            if qualified:candidates.append((record,margin.copy()))
                if candidates:
                    chosen,margin=max(candidates,key=lambda v:(v[0]['covered_positive'],v[0]['threshold']));selected.append(chosen)
                    all_margins[bank_no,category]=margin;all_gates[bank_no,category]=margin>=chosen['threshold']
            print('OVR specialist bank complete',name,'qualified',sum(v['bank']==name for v in selected),flush=True)
        votes=all_gates.sum(0).T;classes_supported=votes>=2;routed=classes_supported.sum(1)==1;primary=votes.argmax(1)
        supported_probability=np.where(all_gates,np.where(np.isfinite(all_margins),expit(all_margins),1.),1.)
        scores=supported_probability[:,primary,np.arange(n)].min(0);bank_support=votes[np.arange(n),primary]
        q=baseline.argmax(1).copy();q[routed]=primary[routed];matches=int((q[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,q[pos]));covered=int(routed[pos].sum());correct=int((routed[pos]&(primary[pos]==gold)).sum())
        accepted=bool(matches>=27 and macro>=.808333333333-1e-12 and covered>=3 and covered==correct and len(selected)>0)
        prior=baseline.copy();prior[routed]=.02*baseline[routed]+.98*np.eye(8)[primary[routed]]
        graphs=np.stack([graph_prediction(unit(x-x.mean(0)),prior,pos,gold) for x in banks.values()])
    graph_support=(graphs==primary).sum(0);human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=routed&(graph_support>=1)&~human&(primary!=current);proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-bank_support,-scores));ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'classwise_svm_router_snapshot',scores[i],'uncalibrated_svm_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'uniquely_routed':routed,'bank_support':bank_support,'score':scores,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=prior,routed=routed,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_classwise_ovr_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head={'method':'classwise_OVR_router','qualified_heads':len(selected),'routed_development':covered,'correct_routed':correct},primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,qualified_heads=selected,selected_knn_heads=[],human_leave_source_group_out_matches=matches,human_holdout_rows=34,development_balanced_recall=macro,
        confidence_calibration={'selected':{'routed':covered,'correct':correct},'independent_validation':False,'objective':'classwiseselectiverouting'},human_diagnostic_protocol=plan['evaluation_limit'],accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],baseline_development_macro=base_macro)
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','human_leave_source_group_out_matches','development_balanced_recall','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline']},indent=2),flush=True)


if __name__=='__main__':main()
