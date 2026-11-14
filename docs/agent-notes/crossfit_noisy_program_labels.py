"""Cross-fitted denoising of historical PROGRAM labels, never archive labels.

Broader coverage retains train subtypes rejected by narrow pretrained consensus.
Any reassignment requires agreement of at least four of five feature families.
"""
import time
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torchvision.transforms.functional import to_tensor,normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit
from task8_teacher_probe import construct


def main():
    started=time.time();torch.set_num_threads(4)
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'cross-fitted noisy-program-label refinement with feature-family consensus',
          'target_origin':'historical routing program assignments from active processed manifest, not visual reference archive',
          'train_rows':1906,'training_label_population':'all working full-frame train, including low external-teacher-coverage subtypes',
          'sample_weight':'.25+.75*historical_program_confidence','manual_labels_consumed':0,'test_rows_in_fit':0,'n_folds':5,
          'selection_within_family':'.5 OOF macro agreement on corroborated high-confidence program core +.5 OOF hflip consistency',
          'reassignment_gate':'at least4of5 feature-family champions agree on a new category; otherwise preserve program category',
          'reference_archive_read':False,'researcher_has_seen_prior_private_audits':True,'semantic_accuracy':None,'protected_sha256':protected}
    dump(NOTES/'json/noisy_program_plan.json',plan)
    target=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});m=m[m.split.eq('train')&~m.is_crop]
    frame=target.merge(m[['parent_key','source_path','source_sha256','assigned_class','category_score']],left_on='key',right_on='parent_key',validate='one_to_one')
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    y=np.array([CLASSES.index(c) for c in frame.assigned_class]);weight=.25+.75*np.clip(frame.category_score.to_numpy(),0,1)
    other=np.array([CLASSES.index(c) for c in frame.automatic_class]);core=frame.automatic_anchor.to_numpy()&(other==y)&(frame.category_score.to_numpy()>=.9)
    assert all(np.any(core&(y==c)) for c in range(8))
    cached=np.load(NOTES/'.runtime/dinov3_train_features.npz',allow_pickle=False);assert np.array_equal(cached['keys'],frame.key.to_numpy(str))
    v3=cached['features'];f3=cached['flip_features'];flip_keys=cached['flip_keys'];idx=dict(zip(frame.key,range(len(frame))));flip_global=np.array([idx[k] for k in flip_keys])
    def load(folder,name='original224.npz'):
        return np.load(OLD/f'output/embeddings/{folder}/{name}',allow_pickle=False)
    def take(z,keys):
        index={key:i for i,key in enumerate(z['keys'])};return z['features'][[index[k] for k in keys]].astype('float32')
    d=load('stage2_repr_v1','dinov2_features.npz');df=load('stage2_repr_v1','dinov2_hflip224_probe.npz')
    xd=unit(take(d,frame.key));fd=unit(take(df,flip_keys))
    p=load('stage5_patch_v1');pf=load('stage5_patch_v1','hflip224.npz');pw=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384)
    xp=unit(take(p,frame.key)*pw);fp=unit(take(pf,flip_keys)*pw)
    task=np.load(NOTES/'.runtime/task8_teacher_features.npz',allow_pickle=False);assert np.array_equal(task['keys'],frame.key.to_numpy(str))
    task_flip_path=NOTES/'.runtime/task8_hflip_features.npz'
    if task_flip_path.exists():
        tf=np.load(task_flip_path,allow_pickle=False);assert np.array_equal(tf['keys'],flip_keys);task_flip=tf['features']
    else:
        model=construct();parts=[]
        with torch.inference_mode():
            for start in range(0,len(flip_keys),8):
                images=[]
                for key in flip_keys[start:start+8]:
                    row=frame.iloc[idx[key]];assert sha(ROOT/row.source_path)==row.source_sha256
                    with Image.open(ROOT/row.source_path) as source:im=ImageOps.exif_transpose(source).convert('RGB')
                    im=ImageOps.mirror(ImageOps.fit(im,(224,224),method=Image.Resampling.BILINEAR,centering=(.5,.5)))
                    images.append(normalize(to_tensor(im),[.485,.456,.406],[.229,.224,.225]))
                raw=model[1][0](model[0](torch.stack(images))).flatten(1);parts.append(nn_unit(raw).numpy().astype('float32'))
        task_flip=np.vstack(parts);np.savez_compressed(task_flip_path,keys=flip_keys,features=task_flip)
        del model
    def fg(a):return unit(np.c_[a[:,:384]*np.sqrt(.5),a[:,768:1152]*.5,a[:,1152:]*.5])
    banks={'dino2':(xd,fd),'dino2_local':(xp,fp),'dino3_global':(unit(v3[:,:768]),unit(f3[:,:768])),
           'dino3_foreground':(fg(v3),fg(f3)),
           'dino3_task8':(unit(np.c_[unit(v3[:,:768]),task['features']]),unit(np.c_[unit(f3[:,:768]),task_flip]))}
    folds=[(np.flatnonzero(frame.fold.to_numpy()!=i),np.flatnonzero(frame.fold.to_numpy()==i)) for i in range(5)]
    configs=[{'kind':'svm','kernel':kernel,'gamma':gamma,'C':c} for kernel,gamma in [('linear',None),('rbf',4.),('rbf',16.)] for c in [.1,1.]]+[{'kind':'knn','neighbors':k} for k in [7,21,51]]
    def knn(sim,fit,k):
        nn=np.argsort(-sim,axis=1,kind='stable')[:,:k];v=np.take_along_axis(sim,nn,axis=1);w=np.exp((v-v.max(1,keepdims=True))/.07)*weight[fit][nn]
        return np.stack([(w*(y[fit][nn]==c)).sum(1) for c in range(8)],axis=1)
    history=[];predictions=[];champions=[]
    with threadpool_limits(limits=4):
        for bank,(x,xf) in banks.items():
            sim=x@x.T;fsim=xf@x.T;local=[]
            for config in configs:
                k=sim if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-sim,0))
                fk=fsim if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-fsim,0))
                pred=np.full(len(frame),-1,dtype='int32');flipped=np.full(len(flip_keys),-1,dtype='int32')
                for tr,va in folds:
                    assert not set(frame.group.iloc[tr])&set(frame.group.iloc[va]);fr=np.flatnonzero(np.isin(flip_global,va))
                    if config['kind']=='svm':
                        model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                        model.fit(k[np.ix_(tr,tr)],y[tr],sample_weight=weight[tr]);pred[va]=model.predict(k[np.ix_(va,tr)])
                        if len(fr):flipped[fr]=model.predict(fk[np.ix_(fr,tr)])
                    else:
                        pred[va]=knn(k[np.ix_(va,tr)],tr,config['neighbors']).argmax(1)
                        if len(fr):flipped[fr]=knn(fk[np.ix_(fr,tr)],tr,config['neighbors']).argmax(1)
                assert (pred>=0).all() and (flipped>=0).all()
                macro=float(np.mean([(pred[core&(y==c)]==c).mean() for c in range(8)]));stability=float((pred[flip_global]==flipped).mean());counts=np.bincount(pred,minlength=8)
                score=.5*macro+.5*stability if (counts>=5).all() else -1
                name=bank+'_'+(f'{config["kernel"]}_g{config["gamma"]}_c{config["C"]}' if config['kind']=='svm' else f'knn{config["neighbors"]}')
                row={'candidate':name,'bank':bank,'config':config,'corroborated_program_core_macro_agreement':macro,'oof_flip_agreement':stability,'selection_score':score,'index':len(predictions),'changes_vs_program':int((pred!=y).sum())}
                predictions.append(pred);history.append(row);local.append(row)
            winner=max(local,key=lambda r:r['selection_score']);champions.append(winner)
            print('Noisy-label crossfit bank complete:',bank,winner['candidate'],flush=True)
    votes=np.stack([predictions[row['index']] for row in champions]);tally=np.stack([(votes==c).sum(0) for c in range(8)],axis=1)
    majority=tally.argmax(1);support=tally.max(1);change=(majority!=y)&(support>=4);final=y.copy();final[change]=majority[change]
    table=frame[['key','image_id','source_sha256']].copy();table['baseline_class']=np.array(CLASSES)[y];table['predicted_class']=np.array(CLASSES)[final];table['family_support']=support;table['program_changed']=change;table['method']='crossfit_full_program_denoising_consensus_v1';table['ground_truth_verified']=False
    output=NOTES/'table-csv/noisy_program_selected.csv';table.to_csv(output,index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='config'} for row in history]).to_csv(NOTES/'table-csv/noisy_program_cv.csv',index=False)
    np.savez_compressed(NOTES/'.runtime/noisy_program_oof.npz',keys=frame.key.to_numpy(str),predictions=np.stack(predictions),candidate=np.array([r['candidate'] for r in history]),family_predictions=votes)
    report={'status':'frozen_before_reference_audit','manual_labels_consumed':0,'test_rows_in_fit':0,'train_rows':1906,'candidate_count':len(history),
            'validation_core_rows':int(core.sum()),'family_champions':champions,'accepted_changes':int(change.sum()),'prediction_sha256':sha(output),
            'source_code_sha256':sha(__file__),'protected_sha256':protected,'semantic_accuracy':None,'active_data_modified':False,'seconds':time.time()-started}
    dump(NOTES/'json/noisy_program_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)


def nn_unit(x):return torch.nn.functional.normalize(x,dim=1)


if __name__=='__main__':main()
