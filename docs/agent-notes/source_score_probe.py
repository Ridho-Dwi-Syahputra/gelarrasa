"""Learn local eight-class heads on non-VLM numeric source scores, train only."""
import json
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    start=time.time();prefix='source_score'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'train_only_learned_material_score_heads_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'features':'numeric12classViTprobabilitiesand28YOLOstatistics;optionalDINOv2BaseCLS;no manualsource-to-targetmapping',
          'models':'multinomialL2logistic,C=.1/1/10,automaticweight=.02/.2,humanweight1;balancedclassweights',
          'preprocessing':'StandardScalerfitonlytargetfitfold;L2perfeatureblock;sourcegroupsdisjoint',
          'selection':'perbank34groupheldoutbalancedrecall;primaryhighest;qualifybanks>=.625andwithin.1ofbest',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'largest34developmentcoverageat>=.9empiricalselectiveprecisionand>=10covered;tiesstricter',
          'acceptance':'atleast2qualifiedbanksallmeetlearnedthreshold;bothtrain-onlynormalizedimagegraphsagree;humanprotected',
          'previous_auto_refinements':'minimumprobability>=max(.95,learnedthreshold)',
          'guard':'primary>=23/34andbalancedrecall>=.725andthresholdexistsand>=2qualifiedbanks',
          'maximum_changes':40,'cap':'descending minimum qualified-bank probability,stablekeyorder',
          'evaluation_limit':'shared34developmentanchors,notindependentfinalAccuracy;graphnotusedincalibrationevaluation',
          'unselected_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    vp=NOTES/'.runtime/waste_vit_train_features.npz';dp=NOTES/'.runtime/litter_object_features.npz';bp=NOTES/'.runtime/dinov2_base_train_features.npz'
    v=np.load(vp,allow_pickle=False);d=np.load(dp,allow_pickle=False);b=np.load(bp,allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [v,d,b])
    probability=v['probabilities'];logp=np.log(np.maximum(probability,1e-6));logp-=logp.mean(1,keepdims=True)
    stats=d['detection_stats'];cls=unit(b['features'][:,:768])
    banks={'source_probability':([np.sqrt(probability)],[12]),'source_log_probability':([logp],[12]),
           'two_material_scores':([logp,stats],[12,28]),'image_material_scores':([cls,logp,stats],[768,12,28])}
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups,folds=g.group.to_numpy(),g.fold.to_numpy();s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)))
    pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    core=a.automatic_anchor.to_numpy(bool)|human;quality=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    configs=[{'C':c,'automatic_weight':w} for c in [.1,1.,10.] for w in [.02,.2]]
    histories=[];selections=[];outputs=[]
    with threadpool_limits(limits=4):
        for name,(blocks,dimensions) in banks.items():
            x=np.concatenate(blocks,axis=1);boundaries=np.cumsum([0]+dimensions);prepared=[]
            for fold in range(5):
                fit=np.flatnonzero((folds!=fold)&core);query=np.flatnonzero(folds==fold);assert not set(groups[fit])&set(groups[query])
                scaler=StandardScaler().fit(x[fit]);z=scaler.transform(x)
                z=np.concatenate([unit(z[:,boundaries[i]:boundaries[i+1]]) for i in range(len(dimensions))],axis=1)/np.sqrt(len(dimensions))
                prepared.append((fit,query,z))
            best_score=-1;chosen=None;best_p=None
            for cfg in configs:
                p=np.zeros((n,8),np.float64)
                for fit,query,z in prepared:
                    weight=quality[fit]*cfg['automatic_weight'];weight[human[fit]]=1.
                    model=LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=1000,solver='lbfgs')
                    model.fit(z[fit],y[fit],sample_weight=weight);assert np.array_equal(model.classes_,np.arange(8))
                    p[query]=model.predict_proba(z[query])
                quality_score=float(balanced_accuracy_score(gold,p[pos].argmax(1)));matches=int((p[pos].argmax(1)==gold).sum())
                histories.append({'bank':name,**cfg,'development_balanced_recall':quality_score,'anchor_matches':matches})
                if quality_score>best_score:best_score=quality_score;chosen=histories[-1];best_p=p
            selections.append(chosen);outputs.append(best_p);print('Local material-score head:',chosen,flush=True)
    winner=max(range(len(selections)),key=lambda i:selections[i]['development_balanced_recall']);chosen=selections[winner];best=chosen['development_balanced_recall']
    trusted=[i for i,h in enumerate(selections) if h['development_balanced_recall']>=.625 and h['development_balanced_recall']>=best-.1]
    primary=outputs[winner].argmax(1);score=np.min([outputs[i][np.arange(n),primary] for i in trusted],axis=0) if trusted else np.zeros(n)
    correct=primary[pos]==gold;calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());matches=int((correct&covered).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    eligible=[r for r in calibration if r['eligible']];cal=max(eligible,key=lambda r:(r['covered'],r['threshold'])) if eligible else None
    threshold=cal['threshold'] if cal else float('inf');gz=np.load(NOTES/'.runtime/normalized_base_outputs.npz',allow_pickle=False);assert np.array_equal(gz['keys'],keys)
    graph_support=(gz['graphs']==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    gate=(score>=threshold)&(graph_support==2)&~human&(primary!=current)&(~preserve|(score>=max(.95,threshold)))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    accepted=cal is not None and len(trusted)>=2 and chosen['anchor_matches']>=23 and best>=.725
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',score[i],'uncalibrated_logistic_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(histories).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_probability':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(outputs),accepted=gate)
    report=json.loads((NOTES/'json/normalized_base_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_source_head_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selections,human_leave_source_group_out_matches=chosen['anchor_matches'],
                  development_balanced_recall=best,confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},qualified_heads=[selections[i] for i in trusted],
                  accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),
                  source_feature_sha256={'ViT':sha(vp),'YOLO':sha(dp),'DINOBase':sha(bp)},source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()
