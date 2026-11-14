"""Selective plastic route from cross-fitted train-only visual subclusters.

No archive/review/test input. The same 34 seeds were previously used in method
development, so source-held-out evidence is preliminary, not independent.
"""
import datetime,json,time
import numpy as np,pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit

def main():
    start=time.time();prefix='cluster_plastic_router';protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'source_heldout_pure_plastic_visual_cluster_router_v1','feature':'DINOv3BaseCLS frozen nonVLM',
          'cluster':'PCA64+MiniBatchKMeans128 fit on other geometry train groups only',
          'cluster_targets':'automatic_full_seed_targets pre-human auto anchors + 34lawful manual seeds, weights vote/4 or8',
          'query':'single nearest centroid class posterior with Dirichlet0.1; plastic only if posterior>=.8',
          'guard':'34source-heldout whole>=29/34, macro>=.875, at least1 newly routed anchor, all routed anchors correct',
          'maximum_changes':40,'cap':'descending cluster posterior then stable source key',
          'test_rows':0,'review_or_archive_read':False,'new_manual_labels':0,
          'evaluation_limit':'reused34development, one selected route, proxy auto labels noisy, no independent accuracy',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)};dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);f=complete[complete.training_eligible].reset_index(drop=True);keys=f.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);human=np.zeros(n,bool);human[pos]=True;assert len(pos)==34
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy();targets=np.array([CLASSES.index(c) for c in a.automatic_class]);targets[pos]=gold;fitmask=a.automatic_anchor.to_numpy(bool)|human;weight=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.);weight[pos]=8.
    z=np.load(NOTES/'.runtime/dinov3_base_train_features.npz');assert np.array_equal(z['keys'],keys);x=unit(z['features'][:,:768]);pred=np.full(n,-1,int);conf=np.zeros(n)
    with threadpool_limits(limits=4):
      for fold in range(5):
        tr=np.flatnonzero(folds!=fold);q=np.flatnonzero(folds==fold);fit=tr[fitmask[tr]];assert not set(groups[fit])&set(groups[q])
        pca=PCA(n_components=64,svd_solver='randomized',random_state=20261108).fit(x[tr]);xt=unit(pca.transform(x[tr]));xq=unit(pca.transform(x[q]));xf=xt[np.searchsorted(tr,fit)]
        km=MiniBatchKMeans(n_clusters=128,n_init=3,batch_size=256,max_iter=100,random_state=20261108).fit(xt);clus=km.predict(xf);votes=np.zeros((128,8),float)+.1
        for i,cat,w in zip(clus,targets[fit],weight[fit]):votes[i,cat]+=w
        posterior=votes/votes.sum(1,keepdims=True);pr=posterior[km.predict(xq)];pred[q]=pr.argmax(1);conf[q]=pr.max(1)
    assert (pred>=0).all()
    v=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz');info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);base=v['probabilities'][wi].argmax(1);aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz');ap,am,ar=aug['oof_predictions'],aug['oof_margins'],aug['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac;acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45);base=np.where(acov,ac,base);res=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz')['probabilities'];rc=res[0].argmax(1);rs=np.min([p[np.arange(n),rc] for p in res],0);base=np.where((rs>=.6)&~(acov&(ac!=rc)),rc,base);assert int((base[pos]==gold).sum())==28
    route=(pred==4)&(conf>=.8)&(base!=4);new=np.where(route,4,base);hr=route[pos];matches=int((new[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,new[pos]));covered=int(hr.sum());correct=int((hr&(new[pos]==gold)).sum());guard=matches>=29 and macro>=.875-1e-12 and covered>=1 and correct==covered
    current=np.array([CLASSES.index(c) for c in f.assigned_class]);gate=(pred==4)&(conf>=.8)&(current!=4)&~human;proposals=int(gate.sum());order=np.argsort(-conf,kind='stable');pick=order[gate[order]][:40];gate[:]=False;gate[pick]=True
    if not guard:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=['plastic','visual_cluster_plastic_snapshot',float(conf[i]),'uncalibrated_cluster_posterior',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[pred],'cluster_posterior':conf,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');metrics={'matches':matches,'macro':macro,'covered':covered,'correct':correct,'guard':guard,'proposals':proposals,'changes':int(gate.sum())}
    report.update(status='frozen_before_private_cluster_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=metrics,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[metrics],qualified_heads=[metrics] if guard else [],selected_knn_heads=[],human_leave_source_group_out_matches=matches,human_holdout_rows=34,development_balanced_recall=macro,confidence_calibration={'selected':metrics,'independent_validation':False},accepted_by_development_guard=bool(guard),proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if guard else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
