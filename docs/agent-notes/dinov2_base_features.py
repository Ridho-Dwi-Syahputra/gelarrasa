"""Official image-only DINOv2 Base numeric weights; train-only representations."""
import json
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
import timm
from fewshot_train_only_probe import ROOT,NOTES,sha,dump

CACHE=NOTES/'.runtime'
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'


def main():
    start=time.time();torch.set_num_threads(6)
    weight=CACHE/'dinov2_vitb14_pretrain.pth'
    if not weight.exists():
        temp=weight.with_suffix('.download')
        with requests.get(URL,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            with temp.open('wb') as out:
                for block in response.iter_content(1024*1024):out.write(block)
        assert temp.stat().st_size>300_000_000
        temp.replace(weight)
    digest=sha(weight)
    source={'repository':'https://github.com/facebookresearch/dinov2','weight_url':URL,'checkpoint_sha256':digest,
            'model':'ViT-B/14 image-only self-supervised,not dino.txt or other VLM','library':'timm local installed architecture',
            'torch_load_weights_only':True,'downloaded_python_executed':False,'external_images_downloaded':False}
    dump(CACHE/'dinov2_base_source.json',source)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    assert len(frame)==1904
    local=np.load(CACHE/'litter_object_features.npz',allow_pickle=False)
    assert np.array_equal(local['keys'],frame.parent_key.to_numpy(str))
    plan={'scope':'train-only image representation,not new targets or final classifier','source':source,
          'train_rows':1904,'test_rows':0,'held_rows':0,'manual_decisions':False,
          'preprocess':'EXIFRGB,224squareletterboxRGB124/116/104,bicubic,ImageNetmeanstd;dynamicpositionresamplingbyinstalledtimm',
          'descriptor':'CLS,patchmean,patchstd,automaticROIpatchmean afterfinalLayerNorm;3072dimensions',
          'ROI':'fixed previouslycomputedobjectbox;no newmanualcrop;fallbackwholevalidimage',
          'localization_feature_sha256':sha(CACHE/'litter_object_features.npz'),'source_code_sha256':sha(__file__),
          'conversion':'strictallinferencekeys;source mask_token excludedbecauseunmaskedimageinference;noindependentofficialforwardparity'}
    dump(NOTES/'json/dinov2_base_feature_plan.json',plan)
    model=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True)
    state=torch.load(weight,map_location='cpu',weights_only=True)
    assert isinstance(state,dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items())
    unused=set(state)-set(model.state_dict());assert unused=={'mask_token'},unused
    state={k:v for k,v in state.items() if k!='mask_token'}
    assert set(state)==set(model.state_dict())
    for k,v in model.state_dict().items():assert state[k].shape==v.shape and torch.isfinite(state[k]).all()
    model.load_state_dict(state,strict=True);model.eval()
    identity=digest+plan['source_code_sha256']
    partial=CACHE/'dinov2_base_partial.npz';features=[];offset0=0
    if partial.exists():
        z=np.load(partial,allow_pickle=False)
        if str(z['identity'].item())==identity and np.array_equal(z['keys'],frame.parent_key.to_numpy(str)[:len(z['keys'])]):
            offset0=len(z['keys']);features=[z['features']]
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    def prepare(row,index):
        assert sha(ROOT/row.source_path)==row.source_sha256
        with Image.open(ROOT/row.source_path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        w,h=im.size;scale=224/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(224-nw)//2,(224-nh)//2
        canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        x1,y1,x2,y2=(local['boxes'][index] if local['focus_flags'][index] else [0,0,w,h])
        x1,x2=dx+x1*scale,dx+x2*scale;y1,y2=dy+y1*scale,dy+y2*scale
        center=(np.arange(16)+.5)*14;roi=(center[None,:]>=x1)&(center[None,:]<x2)&(center[:,None]>=y1)&(center[:,None]<y2)
        if not roi.any():roi[:]=True
        rgb=(canvas.astype(np.float32)/255-mean)/std
        return torch.from_numpy(rgb.transpose(2,0,1).copy()),torch.from_numpy(roi.reshape(-1).astype(np.float32))
    with torch.inference_mode():
        for offset in range(offset0,len(frame),8):
            batch=[prepare(row,offset+i) for i,row in enumerate(frame.iloc[offset:offset+8].itertuples())]
            token=model.forward_features(torch.stack([v[0] for v in batch]));mask=torch.stack([v[1] for v in batch])
            assert token.shape[1:]==(257,768)
            patches=token[:,1:];roi=(patches*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
            f=torch.cat([token[:,0],patches.mean(1),patches.std(1,unbiased=False),roi],dim=1)
            assert torch.isfinite(f).all();features.append(f.numpy().astype(np.float32));end=offset+len(batch)
            if offset==offset0 or end%128==0 or end==len(frame):
                print(f'DINOv2 Base train {end}/{len(frame)} | {time.time()-start:.0f}s',flush=True)
                temp=partial.with_name('dinov2_base_writing.npz')
                np.savez_compressed(temp,keys=frame.parent_key.to_numpy(str)[:end],features=np.concatenate(features),identity=np.array(identity));temp.replace(partial)
    final=CACHE/'dinov2_base_train_features.npz'
    np.savez_compressed(final,keys=frame.parent_key.to_numpy(str),features=np.concatenate(features),source_sha256=frame.source_sha256.to_numpy(str),identity=np.array(identity))
    dump(NOTES/'json/dinov2_base_feature_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'dimensions':3072,
         'feature_sha256':sha(final),'source_code_sha256':sha(__file__),'seconds':time.time()-start})


if __name__=='__main__':main()
