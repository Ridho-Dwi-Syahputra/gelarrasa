"""Inductive residual propagation, inspired by Correct and Smooth (ICLR2021).

Train-only graph in every outer fold; the held-out source is never a graph node,
normalization sample, error reference or pseudo target. Not exact C&S protocol.
"""
import datetime,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit

def main():
    start=time.time();prefix='kernel_residual_interpolation'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    configs=[{'gamma':gamma,'ridge':ridge,'scale':scale} for gamma in [2.,8.,32.] for ridge in [.01,.1,1.] for scale in [1.,2.]]
    plan={'method':'source_excluded_genuine_error_kernel_ridge_v1','reference':'https://arxiv.org/abs/2010.13993','adaptation':'RBFkernelridgeonwarmkNNsource-excludederrors;onlygenuine34residualtargets;noglobalreviewtargets',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'banks':['V3CLS_PCA128white','V3_fullsource_centered'],
          'configs':configs,'baseline':'coldautocoreplusfitfoldgenuinehumansweight32;k7;allfit-rowpredictionsexcludeownsourcegroup;referenceserrornotin-sample',
          'residual':'e=humanonehot-minus-sourceexcludedwarmbaseline;RBFk(x,h)=exp(-gamma(1-cos));correction=Kqh(Khh+ridgeI)^-1e',
          'correction':'fixedscale1/2;cliprenormalize;nohumanclampongraph,nographpseudolabels',
          'normalizer':'outerTRAINonly;querygroupsabsentfromgraph,core,normalizer,residualreferences',
          'selection':'perbank34developmentbalancedrecallthenmatches;primarybestbank;bothbanksagreewithminprobthreshold',
          'threshold_grid':[.6,.7,.8,.9,.95],'guard':'primary>=29andmacro>=.85andcovered>=15with>=.9precision',
          'maximum_changes':20,'cap':'descendingminimumprobability,stablekeyorder','private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'34reuseddevelopmentandmultiplecomparison;graphinductivebutnotindependentfinalAccuracy','protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=g.group.to_numpy(),g.fold.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];y=np.array([CLASSES.index(c) for c in a.automatic_class]);core=a.automatic_anchor.to_numpy(bool)&~np.isin(groups,groups[pos]);weights=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.);y[pos]=gold;weights[pos]=32.
    v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);sd=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False)
    assert np.array_equal(v3['keys'],keys) and np.array_equal(sd['keys'],keys)
    cls=unit(v3['features'][:,:768]);banks={'V3CLS_PCA128white':cls,'V3_fullsource_centered':unit(np.c_[cls,.35*unit(sd['features'][:,0]),.35*unit(sd['features'][:,1])])}
    def normalize(raw,train,name):
        if name.endswith('white'):
            pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108).fit(raw[train]);return unit(pca.transform(raw)/np.maximum(np.sqrt(pca.explained_variance_),1e-6))
        return unit(raw-raw[train].mean(0))
    def base_prediction(x,fit,query):
        sim=x[query]@x[fit].T;sim[groups[query,None]==groups[fit][None,:]]=-np.inf;nn=np.argsort(-sim,axis=1,kind='stable')[:,:7];v=np.take_along_axis(sim,nn,1);assert np.isfinite(v).all()
        w=np.exp((v-v.max(1,keepdims=True))/.07)*weights[fit][nn];p=np.stack([(w*(y[fit][nn]==c)).sum(1) for c in range(8)],1);return p/p.sum(1,keepdims=True)
    def setup(x,train,query,hs):
        references=pos[hs];fit=np.r_[train[core[train]],references];assert len(np.unique(fit))==len(fit)
        reference_prior=base_prediction(x,fit,references);query_prior=base_prediction(x,fit,query)
        residual=np.eye(8)[gold[hs]]-reference_prior
        return query_prior,x[query]@x[references].T,x[references]@x[references].T,residual
    def infer(cache,hs,cfg,query_mode):
        prior,query_similarity,reference_similarity,error=cache
        k=np.exp(-cfg['gamma']*np.maximum(1-reference_similarity,0));kq=np.exp(-cfg['gamma']*np.maximum(1-query_similarity,0))
        coefficient=np.linalg.solve(k+cfg['ridge']*np.eye(len(k)),error)
        corrected=np.maximum(prior+cfg['scale']*(kq@coefficient),1e-8);corrected/=corrected.sum(1,keepdims=True)
        return corrected
    selections=[];outputs=[];history=[]
    with threadpool_limits(limits=3):
        for name,raw in banks.items():
            values=np.zeros((len(configs),n,8))
            for fold in range(5):
                train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);hs=np.flatnonzero(folds[pos]!=fold);assert not set(groups[train])&set(groups[query]);x=normalize(raw,train,name);cache=setup(x,train,query,hs)
                for i,cfg in enumerate(configs):values[i,query]=infer(cache,hs,cfg,True)
            rows=[]
            for i,cfg in enumerate(configs):
                pred=values[i,pos].argmax(1);r={'bank':name,**cfg,'anchor_matches':int((pred==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'index':i};rows.append(r);history.append(r)
            best=max(rows,key=lambda r:(r['development_balanced_recall'],r['anchor_matches']));selections.append(best);outputs.append(values[best['index']]);print('Residual bank',best,flush=True)
    wi=max(range(2),key=lambda i:(selections[i]['development_balanced_recall'],selections[i]['anchor_matches']));chosen=selections[wi];primary=outputs[wi].argmax(1);score=np.min([p[np.arange(n),primary] for p in outputs],axis=0)
    calibration=[]
    for threshold in plan['threshold_grid']:
        mask=score[pos]>=threshold;count=int(mask.sum());correct=int((mask&(primary[pos]==gold)).sum());precision=correct/count if count else 0.;calibration.append({'threshold':threshold,'covered':count,'correct':correct,'precision':precision,'eligible':count>=15 and precision>=.9})
    choices=[r for r in calibration if r['eligible']];cal=max(choices,key=lambda r:(r['covered'],r['threshold'])) if choices else None;threshold=cal['threshold'] if cal else float('inf')
    accepted=cal is not None and chosen['development_balanced_recall']>=.85-1e-12 and chosen['anchor_matches']>=29
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);gate=(score>=threshold)&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'kernel_residual_interpolation_snapshot',score[i],'uncalibrated_corrected_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_corrected_probability':score,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(outputs),accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_residual_correction_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selections,qualified_heads=selections,selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'chosen':chosen,'calibration':calibration,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()
