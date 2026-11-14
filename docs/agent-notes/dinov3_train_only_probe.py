"""Image-only DINOv3 dense/global feature probe with no manual labels.

Foreground is a deterministic pixel/background diagnostic, never a class mask.
The official notebook and processed categories remain unchanged by this probe.
"""
from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import timm
from torchvision.transforms.functional import to_tensor, normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def prepare(path,mirror=False):
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source).convert('RGB')
    size=256;scale=size/max(im.size);dims=tuple(max(1,min(size,round(v*scale))) for v in im.size)
    rgb=np.asarray(im.resize(dims,Image.Resampling.BICUBIC)).copy()
    canvas=np.full((size,size,3),[124,116,104],dtype='uint8');valid=np.zeros((size,size),dtype='float32')
    x,y=(size-dims[0])//2,(size-dims[1])//2;canvas[y:y+dims[1],x:x+dims[0]]=rgb;valid[y:y+dims[1],x:x+dims[0]]=1
    border=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]],axis=0).astype('float32')
    center=np.median(border,axis=0);disp=np.linalg.norm(border-center,axis=1)
    foreground=valid.copy();method='full_valid_region'
    if float(np.quantile(disp,.75))<30:
        distance=np.linalg.norm(rgb.astype('float32')-center,axis=2)
        mask=(distance>max(25,float(np.quantile(disp,.95))+10)).astype('uint8')
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),dtype='uint8'))
        fraction=float(mask.mean())
        if .03<=fraction<=.90:
            foreground=np.zeros_like(valid);foreground[y:y+dims[1],x:x+dims[0]]=mask;foreground+=.05*valid
            method='uniform_border_pixel_difference_with_context_floor'
    if mirror:canvas=np.ascontiguousarray(canvas[:,::-1]);valid=valid[:,::-1].copy();foreground=foreground[:,::-1].copy()
    v=cv2.resize(valid,(16,16),interpolation=cv2.INTER_AREA).reshape(-1)
    f=cv2.resize(foreground,(16,16),interpolation=cv2.INTER_AREA).reshape(-1)
    return Image.fromarray(canvas),v,f,method


def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(20261108)
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    model_info=json.loads((NOTES/'.runtime/dinov3_weight.json').read_text())
    model_info['checkpoint_sha256']=sha(model_info['local_weight_path'])
    plan={'model':{k:v for k,v in model_info.items() if k!='local_weight_path'},'size':256,'manual_labels_consumed':0,'test_rows_in_fit':0,
          'descriptor':'L2 CLS, valid-patch mean, foreground-patch mean and foreground-patch std; no text encoder',
          'foreground':'uniform border p75 distance<30, pixel distance>max(25,p95_border+10), accepted area .03-.90; otherwise full valid image',
          'foreground_scope':'input pixel diagnostic, not segmentation ground truth or category annotation',
          'seed_targets':'same 1323 automatic consensus anchors, all source train IDs verified',
          'validation':'same five source-group folds; no heldout or test targets in model fit',
          'selection':'.65 OOF automatic-anchor macro agreement + .35 OOF hflip agreement; min5 predictions/class',
          'protected_sha256':protected,'archive_labels_read':False,'researcher_has_seen_prior_private_audits':True}
    dump(NOTES/'json/dinov3_train_only_plan.json',plan)
    targets=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    manifest=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=targets.merge(manifest[manifest.split.eq('train')&~manifest.is_crop][['parent_key','source_path','source_sha256','assigned_class']],left_on='key',right_on='parent_key',validate='one_to_one')
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    assert all(sha(ROOT/r.source_path)==r.source_sha256 for r in frame.itertuples())
    prior=np.load(NOTES/'.runtime/automatic_full_seed_oof.npz',allow_pickle=False);flip_keys=prior['flip_keys']
    positions=dict(zip(frame.key,range(len(frame))));flip_global=np.array([positions[k] for k in flip_keys])
    cache=NOTES/'.runtime/dinov3_train_features.npz'
    cache_identity=sha(NOTES/'table-csv/automatic_full_seed_targets.csv')+model_info['checkpoint_sha256']
    if cache.exists():
        z=np.load(cache,allow_pickle=False);assert str(z['identity'].item())==cache_identity
        features=z['features'];flip_features=z['flip_features']
        print('Using verified DINOv3 descriptor cache',flush=True)
    else:
        print('Loading pinned image-only DINOv3 weights',flush=True)
        model=timm.create_model(model_info['model'],pretrained=True,num_classes=0,pretrained_cfg_overlay={'file':model_info['local_weight_path']}).eval()
        cfg=timm.data.resolve_model_data_config(model);prefix=model.num_prefix_tokens
        print(json.dumps({'prefix_tokens':prefix,'dimensions':model.num_features,'normalization_mean':cfg['mean'],'normalization_std':cfg['std']}),flush=True)
        items=[(key,False) for key in frame.key]+[(str(key),True) for key in flip_keys]
        blocks=[];methods=[]
        with torch.inference_mode():
            for start in range(0,len(items),8):
                batch_items=items[start:start+8];images=[];valids=[];foregrounds=[]
                for key,mirror in batch_items:
                    path=ROOT/frame.iloc[positions[key]].source_path
                    im,valid,foreground,method=prepare(path,mirror)
                    images.append(normalize(to_tensor(im),cfg['mean'],cfg['std']));valids.append(valid);foregrounds.append(foreground);methods.append(method)
                tokens=model.forward_features(torch.stack(images));patches=tokens[:,prefix:]
                assert patches.shape[1]==256
                valid=torch.from_numpy(np.stack(valids));fg=torch.from_numpy(np.stack(foregrounds))
                valid=valid/valid.sum(1,keepdim=True).clamp_min(1e-8);fg=fg/fg.sum(1,keepdim=True).clamp_min(1e-8)
                mean_valid=(patches*valid[:,:,None]).sum(1);mean_fg=(patches*fg[:,:,None]).sum(1)
                std_fg=((patches-mean_fg[:,None,:]).square()*fg[:,:,None]).sum(1).clamp_min(0).sqrt()
                features_batch=torch.cat([torch.nn.functional.normalize(part,dim=1) for part in [tokens[:,0],mean_valid,mean_fg,std_fg]],dim=1).cpu().numpy().astype('float32')
                blocks.append(features_batch)
                if start==0 or (start//8+1)%16==0:print(f'DINOv3 descriptors {min(start+8,len(items))}/{len(items)} | {time.time()-started:.0f}s',flush=True)
        all_features=np.vstack(blocks);features=all_features[:len(frame)];flip_features=all_features[len(frame):]
        np.savez_compressed(cache,keys=frame.key.to_numpy(str),features=features,flip_keys=flip_keys,flip_features=flip_features,identity=np.array(cache_identity))
        dump(NOTES/'json/dinov3_descriptor_provenance.json',{'model':plan['model'],'rows':len(frame),'flip_rows':len(flip_keys),'dimensions':features.shape[1],
               'normalization_mean':cfg['mean'],'normalization_std':cfg['std'],'foreground_count_original':sum(m!='full_valid_region' for m in methods[:len(frame)]),
               'test_rows':0,'manual_labels':0,'feature_sha256':sha(cache)})
        del model
    assert features.shape==(1906,1536) and np.isfinite(features).all()
    old=np.load(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    old_flip=np.load(OLD/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    ix={key:i for i,key in enumerate(old['keys'])};fx={key:i for i,key in enumerate(old_flip['keys'])}
    xv2=old['features'][[ix[k] for k in frame.key]];fv2=old_flip['features'][[fx[k] for k in flip_keys]]
    def global_part(a):return unit(a[:,:768])
    def foreground_part(a):return unit(np.c_[a[:,:384]*np.sqrt(.5),a[:,768:1152]*.5,a[:,1152:]*.5])
    banks={'v3_cls':(features[:,:384],flip_features[:,:384]),'v3_global':(global_part(features),global_part(flip_features)),
           'v3_foreground':(foreground_part(features),foreground_part(flip_features)),
           'v3_global_v2':(unit(np.c_[global_part(features),xv2]),unit(np.c_[global_part(flip_features),fv2]))}
    clean=frame.automatic_anchor.to_numpy();y=np.array([CLASSES.index(c) for c in frame.automatic_class]);weight=frame.reliable_vote_count.to_numpy()/4
    folds=[(np.flatnonzero(frame.fold.to_numpy()!=i),np.flatnonzero(frame.fold.to_numpy()==i)) for i in range(5)]
    history=[];all_pred=[];all_flip=[]
    def knn(sim,fit,k):
        nn=np.argsort(-sim,axis=1,kind='stable')[:,:k];v=np.take_along_axis(sim,nn,axis=1)
        w=np.exp((v-v.max(1,keepdims=True))/.07)*weight[fit][nn]
        return np.stack([(w*(y[fit][nn]==c)).sum(1) for c in range(8)],axis=1)
    with threadpool_limits(limits=4):
        for name,(x,xf) in banks.items():
            linear=x@x.T;fl=xf@x.T
            configs=[{'kind':'svm','kernel':kernel,'gamma':gamma,'C':c} for kernel,gamma in [('linear',None),('rbf',4.),('rbf',16.)] for c in [1.,10.]]+[{'kind':'knn','neighbors':k} for k in [7,21]]
            for config in configs:
                kernel=linear if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-linear,0))
                fk=fl if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-fl,0))
                pred=np.full(len(frame),-1,dtype='int32');flipped=np.full(len(flip_keys),-1,dtype='int32')
                for tr,va in folds:
                    fit=tr[clean[tr]];flip_rows=np.flatnonzero(np.isin(flip_global,va))
                    assert not set(frame.group.iloc[tr])&set(frame.group.iloc[va])
                    if config['kind']=='svm':
                        model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                        model.fit(kernel[np.ix_(fit,fit)],y[fit],sample_weight=weight[fit]);pred[va]=model.predict(kernel[np.ix_(va,fit)])
                        if len(flip_rows):flipped[flip_rows]=model.predict(fk[np.ix_(flip_rows,fit)])
                    else:
                        pred[va]=knn(kernel[np.ix_(va,fit)],fit,config['neighbors']).argmax(1)
                        if len(flip_rows):flipped[flip_rows]=knn(fk[np.ix_(flip_rows,fit)],fit,config['neighbors']).argmax(1)
                assert (pred>=0).all() and (flipped>=0).all()
                macro=float(np.mean([(pred[clean&(y==c)]==c).mean() for c in range(8)]));stability=float((pred[flip_global]==flipped).mean())
                counts=np.bincount(pred,minlength=8);score=.65*macro+.35*stability if (counts>=5).all() else -1
                ident=name+'_'+(f'{config["kernel"]}_g{config["gamma"]}_c{config["C"]}' if config['kind']=='svm' else f'knn{config["neighbors"]}')
                history.append({'candidate':ident,'bank':name,'config':config,'automatic_anchor_macro_agreement':macro,'oof_flip_agreement':stability,'selection_score':score,'class_counts':counts.tolist()});all_pred.append(pred);all_flip.append(flipped)
            print(f'DINOv3 bank {name}: development CV complete',flush=True)
    chosen=max(history,key=lambda x:x['selection_score']);config=chosen['config'];x,xf=banks[chosen['bank']];fit=np.flatnonzero(clean)
    with threadpool_limits(limits=4):
        sim=x@x[fit].T
        if config['kind']=='svm':
            k=sim if config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-sim,0))
            model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
            model.fit(k[fit],y[fit],sample_weight=weight[fit]);prediction=model.predict(k)
        else:prediction=knn(sim,fit,config['neighbors']).argmax(1)
    table=frame[['key','image_id','source_sha256']].copy();table['predicted_class']=np.array(CLASSES)[prediction];table['method']=chosen['candidate'];table['ground_truth_verified']=False
    table.to_csv(NOTES/'table-csv/dinov3_selected.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['config','class_counts']} for row in history]).to_csv(NOTES/'table-csv/dinov3_cv_metrics.csv',index=False)
    np.savez_compressed(NOTES/'.runtime/dinov3_oof.npz',keys=frame.key.to_numpy(str),candidate=np.array([r['candidate'] for r in history]),predictions=np.stack(all_pred),flip_keys=flip_keys,flip_predictions=np.stack(all_flip))
    report={'status':'passed','selected_before_archive_audit':chosen,'candidate_count':len(history),'train_rows':len(frame),'test_rows_in_fit':0,'manual_labels_consumed':0,
            'selected_prediction_sha256':sha(NOTES/'table-csv/dinov3_selected.csv'),'class_counts_refit':dict(zip(CLASSES,np.bincount(prediction,minlength=8).tolist())),
            'descriptor_sha256':sha(cache),'source_code_sha256':sha(__file__),'protected_sha256':protected,'semantic_accuracy':None,'active_data_modified':False,'seconds':time.time()-started}
    dump(NOTES/'json/dinov3_selection_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
