"""DINOv2 Base view with automatic object box enlarged before tokenization.

Train-only frozen feature cache, no category target or image edits. Nonfocused
rows reuse the byte-identical full-frame cache. Boxes come from the frozen local
image-only detector, not per-image human decisions.
"""
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import timm
import torch
from fewshot_train_only_probe import ROOT, NOTES, sha, dump


def main():
    start=time.time(); torch.set_num_threads(4)
    cache=NOTES/'.runtime'; full=np.load(cache/'dinov2_base_train_features.npz',allow_pickle=False)
    local=np.load(cache/'litter_object_features.npz',allow_pickle=False)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str); assert len(keys)==1904 and np.array_equal(keys,full['keys']) and np.array_equal(keys,local['keys'])
    assert np.array_equal(frame.source_sha256.to_numpy(str),full['source_sha256'])
    weight=cache/'dinov2_vitb14_pretrain.pth'; assert sha(weight)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
    plan={'scope':'object enlarged BEFORE tokenization,notROIaveragingafterfull-frame224',
          'train_rows':1904,'test_rows':0,'held_rows':0,'categories_read':False,'new_manual_crops':False,
          'preprocess':'EXIFRGB,frozenYOLOpixelbox,aspectpreserving224letterbox,bicubic,ImageNetmeanstd',
          'padding_color':[124,116,104],'descriptor':'CLS,validpatchmean,validpatchstd,validpatchmean;3072',
          'fallback':'reusefullframeBasecacheforunfocusedrows;noimagescreatedormoved',
          'weight_sha256':sha(weight),'base_feature_sha256':sha(cache/'dinov2_base_train_features.npz'),
          'localization_feature_sha256':sha(cache/'litter_object_features.npz'),'source_code_sha256':sha(__file__)}
    dump(NOTES/'json/dinov2_object_zoom_plan.json',plan)
    identity=plan['weight_sha256']+plan['localization_feature_sha256']+plan['source_code_sha256']
    features=full['features'].copy(); done=~local['focus_flags'].copy(); partial=cache/'dinov2_object_zoom_partial.npz'
    if partial.exists():
        z=np.load(partial,allow_pickle=False)
        if str(z['identity'].item())==identity and np.array_equal(z['keys'],keys): features=z['features'].copy(); done=z['processed'].copy()
    model=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True)
    state=torch.load(weight,map_location='cpu',weights_only=True); assert set(state)-set(model.state_dict())=={'mask_token'}
    model.load_state_dict({k:v for k,v in state.items() if k!='mask_token'},strict=True); model.eval()
    mean=np.array([.485,.456,.406],np.float32); std=np.array([.229,.224,.225],np.float32)
    def prepare(i):
        row=frame.iloc[i]; assert sha(ROOT/row.source_path)==row.source_sha256
        with Image.open(ROOT/row.source_path) as src: im=ImageOps.exif_transpose(src).convert('RGB')
        box=tuple(round(float(v)) for v in local['boxes'][i]); assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
        im=im.crop(box); w,h=im.size; scale=224/max(w,h); nw,nh=max(1,round(w*scale)),max(1,round(h*scale)); dx,dy=(224-nw)//2,(224-nh)//2
        canvas=np.full((224,224,3),[124,116,104],np.uint8); canvas[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        centers=(np.arange(16)+.5)*14; valid=(centers[None,:]>=dx)&(centers[None,:]<dx+nw)&(centers[:,None]>=dy)&(centers[:,None]<dy+nh)
        assert valid.any(); rgb=(canvas.astype(np.float32)/255-mean)/std
        return torch.from_numpy(rgb.transpose(2,0,1).copy()), torch.from_numpy(valid.reshape(-1).astype(np.float32))
    def save(path):
        tmp=path.with_name(path.stem+'_writing.npz'); np.savez_compressed(tmp,keys=keys,features=features,processed=done,focus_flags=local['focus_flags'],boxes=local['boxes'],identity=np.array(identity),source_sha256=frame.source_sha256.to_numpy(str));tmp.replace(path)
    todo=np.flatnonzero(~done); last=0
    with torch.inference_mode():
        for offset in range(0,len(todo),8):
            ids=todo[offset:offset+8]; batch=[prepare(i) for i in ids]; token=model.forward_features(torch.stack([b[0] for b in batch])); mask=torch.stack([b[1] for b in batch])
            assert token.shape[1:]==(257,768); patch=token[:,1:]; count=mask.sum(1,keepdim=True); region=(patch*mask[:,:,None]).sum(1)/count
            variance=((patch-region[:,None,:])**2*mask[:,:,None]).sum(1)/count
            desc=torch.cat([token[:,0],region,variance.clamp_min(0).sqrt(),region],1); assert torch.isfinite(desc).all()
            features[ids]=desc.numpy();done[ids]=True
            if time.time()-last>25 or done.all():
                print('DINO Base object zoom',int(done.sum()),'/1904; seconds',round(time.time()-start),flush=True);save(partial);last=time.time()
    final=cache/'dinov2_object_zoom_features.npz';save(final)
    dump(NOTES/'json/dinov2_object_zoom_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'focused_rows':int(local['focus_flags'].sum()),'feature_sha256':sha(final),'seconds':time.time()-start,'source_code_sha256':sha(__file__)})


if __name__=='__main__':main()
