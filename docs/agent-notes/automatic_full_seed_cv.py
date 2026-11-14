"""Train-only grouped development CV using all automatic consensus anchors.

No hand labels, visual-archive annotations, disagreement masks or test rows are
inputs to fitting/selection. Frozen feature files are sliced to train IDs first.
"""
from pathlib import Path
from collections import Counter
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit

SEED = 20261108


def main():
    started=time.time()
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    banks=['dino','local_patch','dino_second','dino_ontology','dino_material','local_material']
    configs=[]
    for bank in banks:
        for kernel,gamma in [('linear',None),('rbf',4.),('rbf',16.)]:
            for c in [1.,10.]:configs.append({'bank':bank,'kind':'svm','kernel':kernel,'gamma':gamma,'C':c})
        for k in [7,21]:configs.append({'bank':bank,'kind':'knn','neighbors':k})
    plan={'seed':SEED,'source':'all reliable automatic independent-CNN consensus anchors; no human labels',
          'manual_labels_consumed':0,'manual_label_budget_reserved':40,'test_rows_in_fit':0,'n_splits':5,
          'feature_normalization':'L2 per block and per sample; no global fitted scaler',
          'fit_targets':'at least two reliable independent source votes with unique winner',
          'sample_weights':'winning vote count / four source families','svm_class_weight':'balanced',
          'selection':'.65 * OOF macro automatic-anchor agreement + .35 * OOF original/hflip agreement; all8 classes and min5 predictions/class',
          'selection_is_development_proxy_not_semantic_accuracy':True,'configs':configs,
          'archive_labels_read':False,'reference_disagreement_ids_read':False,'researcher_has_seen_previous_private_audits':True,
          'protected_sha256':protected,'active_outputs_modified':False}
    dump(NOTES/'json/automatic_full_seed_plan.json',plan)
    manifest=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=manifest[manifest.split.eq('train')&~manifest.is_crop][['parent_key','parent_id','assigned_class','source_sha256','source_path']].rename(columns={'parent_key':'key','parent_id':'image_id'}).sort_values('image_id').reset_index(drop=True)
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    split=pd.read_csv(NOTES/'table-csv/fewshot_train_only_split.csv',dtype={'image_id':str})
    frame=frame.merge(split[['key','group']],on='key',validate='one_to_one')
    old_profile=pd.read_csv(OLD/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    check=frame[['key','source_sha256']].merge(old_profile[['key','source_sha256']],on='key',suffixes=('_current','_cache'),validate='one_to_one')
    assert check.source_sha256_current.eq(check.source_sha256_cache).all()
    assert all(sha(ROOT/r.source_path)==r.source_sha256 for r in frame.itertuples())
    feature_hashes={}
    def archive(relative):
        path=OLD/relative;feature_hashes[relative]=sha(path);return np.load(path,allow_pickle=False)
    def indices(z,keys):
        index={str(key):i for i,key in enumerate(z['keys'])}
        return np.array([index[key] for key in keys])
    dino=archive('output/embeddings/stage2_repr_v1/dinov2_features.npz')
    flip=archive('output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz')
    flip_keys=np.array([key for key in flip['keys'] if key in set(frame.key)])
    assert all(key.startswith('train:') for key in flip_keys)
    frame_index=dict(zip(frame.key,range(len(frame))));flip_global=np.array([frame_index[key] for key in flip_keys])
    xd=unit(dino['features'][indices(dino,frame.key)].astype('float32'))
    fd=unit(flip['features'][indices(flip,flip_keys)].astype('float32'))
    def pair(folder):
        a=archive(f'output/embeddings/{folder}/original224.npz');b=archive(f'output/embeddings/{folder}/hflip224.npz')
        return a,b,indices(a,frame.key),indices(b,flip_keys)
    patch,pf,pi,pfi=pair('stage5_patch_v1')
    patch_weights=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384)
    xp=unit(patch['features'][pi]*patch_weights);fp=unit(pf['features'][pfi]*patch_weights)
    second,sf,si,sfi=pair('stage14_second_waste_cnn_v1')
    xs=unit(second['features'][si].astype('float32'));fs=unit(sf['features'][sfi].astype('float32'))
    ontology,of,oi,ofi=pair('stage11_ontology_cnn_v1')
    xo=unit(ontology['features'][oi].astype('float32'));fo=unit(of['features'][ofi].astype('float32'))
    material=archive('output/embeddings/stage6_semantic_v1/minc_materials_original224.npz')
    mf=archive('output/embeddings/stage6_semantic_v1/minc_materials_hflip224.npz')
    xm=unit(material['probabilities'][indices(material,frame.key)].astype('float32'))
    fm=unit(mf['probabilities'][indices(mf,flip_keys)].astype('float32'))
    vectors={'dino':(xd,fd),'local_patch':(xp,fp),'dino_second':(unit(np.c_[xd,xs]),unit(np.c_[fd,fs])),
             'dino_ontology':(unit(np.c_[xd,xo]),unit(np.c_[fd,fo])),
             'dino_material':(unit(np.c_[xd,xm]),unit(np.c_[fd,fm])),
             'local_material':(unit(np.c_[xp,xm]),unit(np.c_[fp,fm]))}
    generic=archive('output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz')
    sources=[generic['prior_probabilities'][indices(generic,frame.key)]];coverages=[np.ones(len(frame),dtype='float32')]
    translation={'battery':'electronic','biological':'foodwaste','brown-glass':'glass','green-glass':'glass','white-glass':'glass','glass':'glass','clothes':'textile','shoes':'textile',**{c:c for c in ['cardboard','paper','metal','plastic']}}
    for folder,report in [('stage7_waste_cnn_v1','stage7_waste_cnn_20261106'),('stage14_second_waste_cnn_v1','stage14_second_waste_cnn_20261106')]:
        z=archive(f'output/embeddings/{folder}/original224.npz');info=json.loads((OLD/f'reports/errors/{report}/checkpoint_and_ontology.json').read_text())
        probability=z['probabilities'][indices(z,frame.key)]
        evidence=np.zeros((len(frame),8),dtype='float32')
        for column,name in enumerate(info['source_class_order']):
            if name in translation:evidence[:,CLASSES.index(translation[name])]+=probability[:,column]
        cov=evidence.sum(1);sources.append(evidence/np.maximum(cov[:,None],1e-8));coverages.append(cov)
    sources.append(ontology['conditional'][oi]);coverages.append(ontology['coverage'][oi])
    src=np.stack(sources);cov=np.stack(coverages)
    reliable=(src.max(2)>=.8)&(cov>=np.array([0,.6,.6,.2])[:,None])
    votes=np.zeros((len(frame),8),dtype='float32')
    for i in range(4):votes[np.flatnonzero(reliable[i]),src[i,reliable[i]].argmax(1)]+=1
    target=votes.argmax(1);ordered=np.sort(votes,axis=1);clean=(ordered[:,-1]>=2)&(ordered[:,-1]>ordered[:,-2])
    weight=ordered[:,-1]/4
    strat=np.where(clean,target,np.array([CLASSES.index(name) for name in frame.assigned_class]))
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(frame,strat,groups=frame.group))
    fold_id=np.full(len(frame),-1,dtype=int)
    for k,(tr,va) in enumerate(folds):
        assert not set(frame.group.iloc[tr])&set(frame.group.iloc[va]);fold_id[va]=k
        assert all(np.any(clean[tr]&(target[tr]==c)) for c in range(8))
    assert (fold_id>=0).all()
    seed_table=frame[['key','image_id','group']].copy();seed_table['fold']=fold_id;seed_table['automatic_anchor']=clean;seed_table['automatic_class']=np.array(CLASSES)[target];seed_table['reliable_vote_count']=ordered[:,-1]
    seed_table.to_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',index=False)
    oof_bank=[];flip_bank=[];results=[]
    def evaluate(pred,flipped):
        macro=float(np.mean([(pred[clean&(target==c)]==c).mean() for c in range(8)]))
        stability=float((pred[flip_global]==flipped).mean());counts=np.bincount(pred,minlength=8)
        return macro,stability,counts
    def knn_scores(similarity,train,k):
        nearest=np.argsort(-similarity,axis=1,kind='stable')[:,:k]
        similarities=np.take_along_axis(similarity,nearest,axis=1)
        weights=np.exp((similarities-similarities.max(1,keepdims=True))/.07)*weight[train][nearest]
        scores=np.zeros((len(similarity),8),dtype='float32')
        for c in range(8):scores[:,c]=(weights*(target[train][nearest]==c)).sum(1)
        return scores
    with threadpool_limits(limits=4):
        for bank,(x,xf) in vectors.items():
            print(f'Feature bank {bank}: {x.shape[1]} dimensions; anchors {int(clean.sum())}',flush=True)
            linear=x@x.T;flip_linear=xf@x.T
            for config in [c for c in configs if c['bank']==bank]:
                kernel=linear if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-linear,0))
                fk=flip_linear if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-flip_linear,0))
                pred=np.full(len(frame),-1,dtype=np.int32);flipped=np.full(len(flip_keys),-1,dtype=np.int32)
                for tr,va in folds:
                    fit=tr[clean[tr]];flip_rows=np.flatnonzero(np.isin(flip_global,va))
                    if config['kind']=='svm':
                        model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                        model.fit(kernel[np.ix_(fit,fit)],target[fit],sample_weight=weight[fit])
                        pred[va]=model.predict(kernel[np.ix_(va,fit)])
                        if len(flip_rows):flipped[flip_rows]=model.predict(fk[np.ix_(flip_rows,fit)])
                    else:
                        pred[va]=knn_scores(kernel[np.ix_(va,fit)],fit,config['neighbors']).argmax(1)
                        if len(flip_rows):flipped[flip_rows]=knn_scores(fk[np.ix_(flip_rows,fit)],fit,config['neighbors']).argmax(1)
                assert (pred>=0).all() and (flipped>=0).all()
                macro,stability,counts=evaluate(pred,flipped)
                eligible=bool((counts>=5).all());score=.65*macro+.35*stability if eligible else -1
                ident=f'{bank}_{config["kind"]}_'+(f'{config["kernel"]}_g{config["gamma"]}_c{config["C"]}' if config['kind']=='svm' else f'k{config["neighbors"]}')
                results.append({'candidate':ident,'automatic_anchor_macro_agreement':macro,'oof_flip_agreement':stability,'selection_score':score,'eligible':eligible,'changed_oof_vs_baseline':int((np.array(CLASSES)[pred]!=frame.assigned_class).sum()),'config':config,'class_counts':counts.tolist()})
                oof_bank.append(pred);flip_bank.append(flipped)
    selected=max(range(len(results)),key=lambda i:results[i]['selection_score']);chosen=results[selected];config=chosen['config']
    print('Selected from development CV:',chosen['candidate'],flush=True)
    x,xf=vectors[config['bank']];fit=np.flatnonzero(clean)
    with threadpool_limits(limits=4):
        similarity=x@x[fit].T
        if config['kind']=='svm':
            kernel=similarity if config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-similarity,0))
            model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
            model.fit(kernel[fit],target[fit],sample_weight=weight[fit]);scores=model.decision_function(kernel);prediction=model.predict(kernel)
            joblib.dump({'estimator':model,'fit_keys':frame.key.iloc[fit].tolist(),'config':config,'class_order':CLASSES},NOTES/'.runtime/automatic_full_seed_selected.joblib')
        else:
            scores=knn_scores(similarity,fit,config['neighbors']);prediction=scores.argmax(1)
    table=frame[['key','image_id','source_sha256']].copy();table['predicted_class']=np.array(CLASSES)[prediction];table['method']=chosen['candidate'];table['ground_truth_verified']=False
    table.to_csv(NOTES/'table-csv/automatic_full_seed_selected.csv',index=False)
    metrics=pd.DataFrame([{k:v for k,v in r.items() if k not in ['config','class_counts']} for r in results]);metrics.to_csv(NOTES/'table-csv/automatic_full_seed_cv_metrics.csv',index=False)
    np.savez_compressed(NOTES/'.runtime/automatic_full_seed_oof.npz',keys=frame.key.to_numpy(str),predictions=np.stack(oof_bank),flip_keys=flip_keys,flip_predictions=np.stack(flip_bank),candidate=np.array([r['candidate'] for r in results]))
    report={'status':'passed','manual_labels_consumed':0,'train_rows':len(frame),'test_rows_in_fit':0,'auto_anchor_count':int(clean.sum()),'auto_anchor_counts':dict(zip(CLASSES,np.bincount(target[clean],minlength=8).tolist())),
            'n_splits':5,'candidate_count':len(results),'selected_before_archive_audit':chosen,'selected_prediction_sha256':sha(NOTES/'table-csv/automatic_full_seed_selected.csv'),
            'class_counts_refit':dict(zip(CLASSES,np.bincount(prediction,minlength=8).tolist())),'feature_sha256':feature_hashes,
            'source_code_sha256':sha(__file__),'protected_sha256':protected,'semantic_accuracy':None,'active_outputs_modified':False,'seconds':time.time()-started}
    dump(NOTES/'json/automatic_full_seed_selection_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k not in ['feature_sha256','protected_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
