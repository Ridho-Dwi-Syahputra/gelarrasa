"""Train-only multiscale/second-order CNN texture, no semantic source mapping.

Inspired by bilinear texture descriptors, not an implementation or performance
claim for the end-to-end B-CNN paper. Original organizer pixels remain unchanged.
"""
import json
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
import timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from timm.data import resolve_model_data_config
from fewshot_train_only_probe import ROOT,NOTES,OLD,sha,dump


def main():
    start=time.time();torch.set_num_threads(6)
    info=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(info['repository'],'model.safetensors',revision=info['revision'],local_files_only=True)
    assert sha(weight)==info['checkpoint_sha256']
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);assert len(keys)==1904 and np.char.startswith(keys,'train:').all()
    roi=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);assert np.array_equal(roi['keys'],keys)
    plan={'method':'multiscale_ConvNeXt_and_projected_foreground_covariance_v1','train_rows':1904,'test_rows':0,'held_rows':0,
          'pretrained_source':{k:info[k] for k in ['repository','revision','checkpoint_sha256','architecture']},
          'preprocess':'EXIFRGB224aspectpreservingletterbox,ImageNetmeanstd,bicubic;differentfromoldcentercrop',
          'pooling':'stages192/384/768,validregionandautomaticROImean/std;5376dimensions',
          'second_order':'stage192fixedGaussianprojectionto48;ROIweightedcenteredcovariance;1176uppertrianglesignedsqrtL2',
          'ROI':'existingautomaticobjectboxes;fallbackwholevalidimage;no category inheritance',
          'manual_semantic_decisions':False,'source_class_labels_or_probabilities_read':False,'external_images_added':False,
          'reference':'https://arxiv.org/abs/1504.07889','reference_limit':'inspiredsecondorderrepresentation,notexactB-CNNarchitectureorclaimedpaperaccuracy',
          'source_code_sha256':sha(__file__)}
    dump(NOTES/'json/convnext_texture_plan.json',plan)
    print('Train texture manifest snapshot ready:',len(frame),'rows',flush=True)
    model=timm.create_model('convnext_tiny.fb_in22k',pretrained=False,num_classes=info['classes'])
    state=load_file(weight);model.load_state_dict(state,strict=True);model.eval();cfg=resolve_model_data_config(model)
    mean=np.array(cfg['mean'],np.float32);std=np.array(cfg['std'],np.float32)
    generator=torch.Generator().manual_seed(20261108);projection=torch.randn(192,48,generator=generator)/np.sqrt(192)
    tri=torch.triu_indices(48,48);features=[];covariances=[];offset0=0
    identity=info['checkpoint_sha256']+plan['source_code_sha256'];partial=NOTES/'.runtime/convnext_texture_partial.npz'
    if partial.exists():
        z=np.load(partial,allow_pickle=False)
        if str(z['identity'].item())==identity and np.array_equal(z['keys'],keys[:len(z['keys'])]):offset0=len(z['keys']);features=[z['features']];covariances=[z['covariance']]
    def prepare(row,index):
        assert sha(ROOT/row.source_path)==row.source_sha256
        with Image.open(ROOT/row.source_path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        w,h=im.size;scale=224/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(224-nw)//2,(224-nh)//2
        canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        valid=np.zeros((224,224),np.float32);valid[dy:dy+nh,dx:dx+nw]=1
        object_mask=valid.copy()
        if roi['focus_flags'][index]:
            a,b,c,d=roi['boxes'][index];a,b=int(dx+a*scale),int(dy+b*scale);c,d=int(np.ceil(dx+c*scale)),int(np.ceil(dy+d*scale))
            object_mask[:]=0;object_mask[max(0,b):min(224,d),max(0,a):min(224,c)]=1;object_mask*=valid
            if object_mask.sum()<16:object_mask=valid.copy()
        x=torch.from_numpy(((canvas.astype(np.float32)/255-mean)/std).transpose(2,0,1).copy())
        return x,torch.from_numpy(valid),torch.from_numpy(object_mask)
    def pool(x,mask):
        w=F.interpolate(mask[:,None],size=x.shape[-2:],mode='area');den=w.sum((2,3)).clamp_min(1e-6)
        mu=(x*w).sum((2,3))/den;variance=(x.square()*w).sum((2,3))/den-mu.square()
        return mu,variance.clamp_min(0).sqrt(),w
    with torch.inference_mode():
        for offset in range(offset0,len(frame),16):
            batch=[prepare(row,offset+i) for i,row in enumerate(frame.iloc[offset:offset+16].itertuples())]
            x=torch.stack([v[0] for v in batch]);valid=torch.stack([v[1] for v in batch]);mask=torch.stack([v[2] for v in batch]);x=model.stem(x);parts=[];cov=None
            for stage_id,stage in enumerate(model.stages):
                x=stage(x)
                if stage_id==0:continue
                gm,gs,_=pool(x,valid);rm,rs,rw=pool(x,mask);parts.extend([gm,gs,rm,rs])
                if stage_id==1:
                    spatial=x.flatten(2).transpose(1,2)@projection;ww=rw.flatten(2).transpose(1,2);den=ww.sum(1,keepdim=True).clamp_min(1e-6)
                    centered=spatial-(spatial*ww).sum(1,keepdim=True)/den
                    matrix=(centered*ww).transpose(1,2)@centered/den
                    cov=matrix[:,tri[0],tri[1]];cov=cov.sign()*cov.abs().sqrt();cov=F.normalize(cov,dim=1)
            f=torch.cat(parts,dim=1);assert f.shape[1]==5376 and cov.shape[1]==1176 and torch.isfinite(f).all() and torch.isfinite(cov).all()
            features.append(f.numpy().astype(np.float32));covariances.append(cov.numpy().astype(np.float32));end=offset+len(batch)
            if offset==offset0 or end%128==0 or end==len(frame):
                print(f'CNN texture train {end}/{len(frame)} | {time.time()-start:.0f}s',flush=True)
                temp=partial.with_name('convnext_texture_writing.npz');np.savez_compressed(temp,keys=keys[:end],features=np.concatenate(features),covariance=np.concatenate(covariances),identity=np.array(identity));temp.replace(partial)
    final=NOTES/'.runtime/convnext_texture_features.npz'
    np.savez_compressed(final,keys=keys,features=np.concatenate(features),covariance=np.concatenate(covariances),source_sha256=frame.source_sha256.to_numpy(str),identity=np.array(identity))
    dump(NOTES/'json/convnext_texture_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'statistic_dimensions':5376,'covariance_dimensions':1176,'feature_sha256':sha(final),'source_code_sha256':sha(__file__),'seconds':time.time()-start})


if __name__=='__main__':main()
