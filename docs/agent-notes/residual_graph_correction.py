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
    start=time.time();prefix='residual_graph_correction'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    configs=[{'alpha':alpha,'scale':scale,'smooth':smooth} for alpha in [.7,.95] for scale in [1.,2.,4.] for smooth in [False,True]]
    plan={'method':'inductive_train_only_residual_correct_and_smooth_v1','reference':'https://arxiv.org/abs/2010.13993','adaptation':'frozennonVLMfeatures,auto-onlycoldkNNbaseline,34genuineresidualreferences,inductivequery;notpaperreproduction',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'banks':['V3CLS_PCA128white','V3_fullsource_centered'],
          'configs':configs,'baseline':'auto-onlycoldcorek7;excludeALLhuman-sourcegroupsfromautocore;fit-rowpredictionsremoveownsourcegroup',
          'residual':'onehotgenuine-minus-auto-onlyprediction;clampedreferenceerror;train-onlyrowstochasticgraph,k20,cos^4;queryinterpolationafterfit',
          'correction':'fixedscale1/2/4;cliprenormalize;optional.5clampedhumanpredictiondiffusion',
          'normalizer':'outerTRAINonly;querygroupsabsentfromgraph,core,normalizer,residualreferences',
          'selection':'perbank34developmentbalancedrecallthenmatches;primarybestbank;bothbanksagreewithminprobthreshold',
          'threshold_grid':[.6,.7,.8,.9,.95],'guard':'primary>=28andmacro>=.808333333333;ORprimary>=27andmacro>=.808333333333andcovered>=10with>=.9precision',
          'maximum_changes':20,'cap':'descendingminimumprobability,stablekeyorder','private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'34reuseddevelopmentandmultiplecomparison;graphinductivebutnotindependentfinalAccuracy','protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=g.group.to_numpy(),g.fold.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];y=np.array([CLASSES.index(c) for c in a.automatic_class]);core=a.automatic_anchor.to_numpy(bool)&~np.isin(groups,groups[pos]);weights=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
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
        local=dict(zip(train,range(len(train))));hp=np.array([local[i] for i in pos[hs]]);fit=train[core[train]]
        prior=base_prediction(x,fit,train);query_prior=base_prediction(x,fit,query) if len(query) else None
        sim=x[train]@x[train].T;sim[groups[train,None]==groups[train][None,:]]=-np.inf;nn=np.argsort(-sim,axis=1,kind='stable')[:,:20];edge=np.maximum(np.take_along_axis(sim,nn,1),0)**4
        adj=sparse.csr_matrix((edge.ravel(),(np.repeat(np.arange(len(train)),20),nn.ravel())),shape=(len(train),len(train)));adj=(adj+adj.T)*.5;degree=np.asarray(adj.sum(1)).ravel();transition=sparse.diags(1/np.maximum(degree,1e-12))@adj
        if len(query):
            qs=x[query]@x[train].T;qn=np.argsort(-qs,axis=1,kind='stable')[:,:20];qw=np.maximum(np.take_along_axis(qs,qn,1),0)**4;qw/=np.maximum(qw.sum(1,keepdims=True),1e-12)
        else:qn=qw=None
        error=np.zeros_like(prior);error[hp]=np.eye(8)[gold[hs]]-prior[hp]
        return prior,query_prior,hp,transition,qn,qw,error
    def infer(cache,hs,cfg,query_mode):
        prior,qp,hp,t,qn,qw,error=cache;state=error.copy()
        for _ in range(250):
            update=cfg['alpha']*(t@state)+(1-cfg['alpha'])*error;update[hp]=error[hp]
            delta=np.max(np.abs(update-state));state=update
            if delta<1e-6:break
        corrected=np.maximum(prior+cfg['scale']*state,1e-8);corrected/=corrected.sum(1,keepdims=True)
        if cfg['smooth']:
            p=corrected.copy();p[hp]=np.eye(8)[gold[hs]];smooth=p.copy()
            for _ in range(150):
                update=.5*(t@smooth)+.5*p;update[hp]=p[hp];delta=np.max(np.abs(update-smooth));smooth=update
                if delta<1e-6:break
            corrected=smooth
        if not query_mode:return corrected
        query_error=(state[qn]*qw[:,:,None]).sum(1);prediction=np.maximum(qp+cfg['scale']*query_error,1e-8);prediction/=prediction.sum(1,keepdims=True)
        if cfg['smooth']:prediction=.5*prediction+.5*(corrected[qn]*qw[:,:,None]).sum(1)
        return prediction
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
        mask=score[pos]>=threshold;count=int(mask.sum());correct=int((mask&(primary[pos]==gold)).sum());precision=correct/count if count else 0.;calibration.append({'threshold':threshold,'covered':count,'correct':correct,'precision':precision,'eligible':count>=10 and precision>=.9})
    choices=[r for r in calibration if r['eligible']];cal=max(choices,key=lambda r:(r['covered'],r['threshold'])) if choices else None;threshold=cal['threshold'] if cal else float('inf')
    accepted=cal is not None and chosen['development_balanced_recall']>=.808333333333-1e-12 and chosen['anchor_matches']>=27
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);gate=(score>=threshold)&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'residual_graph_correction_snapshot',score[i],'uncalibrated_corrected_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_corrected_probability':score,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(outputs),accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_residual_correction_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selections,qualified_heads=selections,selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'chosen':chosen,'calibration':calibration,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()
