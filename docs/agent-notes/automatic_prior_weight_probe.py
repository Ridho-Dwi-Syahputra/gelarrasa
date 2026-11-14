"""Automatic class-coverage correction; no human/reference labels or test fit.

Visual clusters are used only to define a harder development partition. They
are never used as predictor input or fitted on competition test data.
"""
import time
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit


def main():
    started=time.time()
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'confidence and class-coverage weighted nearest-neighbor classification','manual_labels_consumed':0,'test_rows_in_fit':0,
          'validation':'five folds of128 unsupervised train-only visual clusters; original source groups kept together',
          'visual_clusters_used_for_partition_only':True,'class_weight_modes':['none','uniform_prior','machine_prior'],
          'prior_exponents':[.5,1.],'neighbors':[7,21],'seed':'all1323 automatic CNN-consensus anchors',
          'machine_prior_source':'class proportions of prior program assignments within the fold training rows only',
          'selection':'.65 macro agreement on heldout automatic anchors +.35 heldout hflip agreement; min5 predictions/class',
          'full_refit_inference_excludes_own_source_group':True,'counts_not_constrained_to_a_quota':True,
          'archive_reference_read':False,'researcher_has_seen_previous_private_audits':True,'protected_sha256':protected}
    dump(NOTES/'json/automatic_prior_weight_plan.json',plan)
    frame=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    manifest=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=frame.merge(manifest[manifest.split.eq('train')&~manifest.is_crop][['parent_key','assigned_class']],left_on='key',right_on='parent_key',validate='one_to_one')
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    def load(folder,filename='original224.npz'):
        return np.load(OLD/f'output/embeddings/{folder}/{filename}',allow_pickle=False)
    def take(z,keys):
        index={key:i for i,key in enumerate(z['keys'])};return z['features'][[index[k] for k in keys]].astype('float32')
    dino=load('stage2_repr_v1','dinov2_features.npz');df=load('stage2_repr_v1','dinov2_hflip224_probe.npz')
    available=set(frame.key);flip_keys=np.array([k for k in df['keys'] if k in available]);index=dict(zip(frame.key,range(len(frame))));flip_global=np.array([index[k] for k in flip_keys])
    xd=unit(take(dino,frame.key));fd=unit(take(df,flip_keys))
    second=load('stage14_second_waste_cnn_v1');sf=load('stage14_second_waste_cnn_v1','hflip224.npz')
    xs=unit(take(second,frame.key));fs=unit(take(sf,flip_keys))
    patch=load('stage5_patch_v1');pf=load('stage5_patch_v1','hflip224.npz');weights=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384)
    xp=unit(take(patch,frame.key)*weights);fp=unit(take(pf,flip_keys)*weights)
    banks={'dino':(xd,fd),'local_patch':(xp,fp),'dino_second':(unit(np.c_[xd,xs]),unit(np.c_[fd,fs]))}
    with threadpool_limits(limits=2):
        visual=KMeans(n_clusters=128,n_init=10,random_state=20261109).fit_predict(xd)
    # Map every source group to one visual partition, even if feature rounding
    # put individual variants on different sides of a clustering boundary.
    frame['visual_cluster']=visual
    group_map=frame.groupby('group').visual_cluster.agg(lambda v:int(v.mode().iloc[0])).to_dict()
    visual=np.array([group_map[g] for g in frame.group])
    y=np.array([CLASSES.index(c) for c in frame.automatic_class]);clean=frame.automatic_anchor.to_numpy();confidence=frame.reliable_vote_count.to_numpy()/4
    baseline=np.array([CLASSES.index(c) for c in frame.assigned_class]);strat=np.where(clean,y,baseline)
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261109).split(frame,strat,groups=visual))
    fold=np.full(len(frame),-1,dtype='int32')
    for i,(tr,va) in enumerate(folds):
        assert not set(visual[tr])&set(visual[va]);assert not set(frame.group.iloc[tr])&set(frame.group.iloc[va]);fold[va]=i
        assert all(np.any(clean[tr]&(y[tr]==c)) for c in range(8))
    frame[['key','image_id','group']].assign(visual_cluster=visual,fold=fold).to_csv(NOTES/'table-csv/automatic_visual_group_split.csv',index=False)
    configs=[{'mode':'none','power':0.,'k':k} for k in [7,21]]+[{'mode':mode,'power':power,'k':k} for mode in ['uniform_prior','machine_prior'] for power in [.5,1.] for k in [7,21]]
    def factors(fit,all_fit,config):
        count=np.bincount(y[fit],minlength=8).astype('float64');source=count/count.sum()
        prior=np.full(8,.125) if config['mode']=='uniform_prior' else np.bincount(baseline[all_fit],minlength=8)/len(all_fit)
        if config['mode']=='none':return np.ones(8)
        return (np.maximum(prior,1e-8)/np.maximum(source,1e-8))**config['power']
    def scores(similarity,fit,all_fit,config):
        nn=np.argsort(-similarity,axis=1,kind='stable')[:,:config['k']];sim=np.take_along_axis(similarity,nn,axis=1)
        assert np.isfinite(sim).all()
        w=np.exp((sim-sim.max(1,keepdims=True))/.07)*confidence[fit][nn]
        result=np.stack([(w*(y[fit][nn]==c)).sum(1) for c in range(8)],axis=1)
        return result*factors(fit,all_fit,config)[None,:]
    history=[];all_preds=[];all_flip=[]
    with threadpool_limits(limits=2):
        for bank,(x,xf) in banks.items():
            similarities=x@x.T;fl=xf@x.T
            for config in configs:
                pred=np.full(len(frame),-1,dtype='int32');flipped=np.full(len(flip_keys),-1,dtype='int32')
                for tr,va in folds:
                    fit=tr[clean[tr]];fr=np.flatnonzero(np.isin(flip_global,va))
                    pred[va]=scores(similarities[np.ix_(va,fit)],fit,tr,config).argmax(1)
                    if len(fr):flipped[fr]=scores(fl[np.ix_(fr,fit)],fit,tr,config).argmax(1)
                macro=float(np.mean([(pred[clean&(y==c)]==c).mean() for c in range(8)]));stability=float((pred[flip_global]==flipped).mean());counts=np.bincount(pred,minlength=8)
                score=.65*macro+.35*stability if (counts>=5).all() else -1
                name=f'{bank}_{config["mode"]}_p{config["power"]}_k{config["k"]}'
                history.append({'candidate':name,'bank':bank,'config':config,'automatic_anchor_macro_agreement':macro,'oof_flip_agreement':stability,'selection_score':score,'class_counts':counts.tolist()});all_preds.append(pred);all_flip.append(flipped)
            print(f'Coverage weighting bank {bank}: hard-group CV done',flush=True)
    chosen=max(history,key=lambda r:r['selection_score']);config=chosen['config'];x,xf=banks[chosen['bank']];fit=np.flatnonzero(clean)
    with threadpool_limits(limits=2):
        similarity=x@x[fit].T
        same_group=frame.group.to_numpy()[:,None]==frame.group.to_numpy()[fit][None,:];similarity[same_group]=-np.inf
        final_scores=scores(similarity,fit,np.arange(len(frame)),config);pred=final_scores.argmax(1)
    table=frame[['key','image_id']].copy();table['predicted_class']=np.array(CLASSES)[pred];table['method']=chosen['candidate'];table['ground_truth_verified']=False
    table.to_csv(NOTES/'table-csv/automatic_prior_weight_selected.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['config','class_counts']} for row in history]).to_csv(NOTES/'table-csv/automatic_prior_weight_cv.csv',index=False)
    np.savez_compressed(NOTES/'.runtime/automatic_prior_weight_oof.npz',keys=frame.key.to_numpy(str),predictions=np.stack(all_preds),candidate=np.array([r['candidate'] for r in history]),flip_keys=flip_keys,flip_predictions=np.stack(all_flip))
    report={'status':'passed','manual_labels_consumed':0,'test_rows_in_fit':0,'train_rows':len(frame),'visual_groups':len(set(visual)),
            'candidate_count':len(history),'selected_before_archive_audit':chosen,'selected_prediction_sha256':sha(NOTES/'table-csv/automatic_prior_weight_selected.csv'),
            'class_counts_refit':dict(zip(CLASSES,np.bincount(pred,minlength=8).tolist())),'semantic_accuracy':None,'active_data_modified':False,
            'protected_sha256':protected,'source_code_sha256':sha(__file__),'seconds':time.time()-started}
    dump(NOTES/'json/automatic_prior_weight_selection_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
