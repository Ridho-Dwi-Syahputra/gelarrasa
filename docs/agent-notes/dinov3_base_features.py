"""Pinned image-only DINOv3 Base, organizer TRAIN-only numeric descriptors.

No ontology/class labels, test images, downloaded Python or inference service.
Installed timm architecture, strict safetensors keys, bounded CPU batch memory.
"""
import gc
import json
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config
from safetensors.torch import load_file
from huggingface_hub import HfApi,hf_hub_download
from fewshot_train_only_probe import ROOT,NOTES,sha,dump


def main():
    start=time.time();torch.set_num_threads(4);cache=NOTES/'.runtime';source_file=cache/'dinov3_base_source.json'
    repository='timm/vit_base_patch16_dinov3.lvd1689m';name='vit_base_patch16_dinov3.lvd1689m'
    if source_file.exists():source=json.loads(source_file.read_text())
    else:
        info=HfApi().model_info(repository);assert not info.gated
        source={'repository':repository,'revision':info.sha,'gated':info.gated,'model':name,'license':'dinov3-license','model_card':'https://huggingface.co/'+repository,
                'image_only':True,'downloaded_python_executed':False,'external_images_added':False,'source_pretraining_image_overlap_unknown':True}
        dump(source_file,source)
    weight=hf_hub_download(repository,'model.safetensors',revision=source['revision']);source['checkpoint_sha256']=sha(weight);source['local_weight_path']=weight;dump(source_file,source)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);assert len(keys)==1904 and not frame.parent_id.isin(['1662','1885']).any()
    local=np.load(cache/'litter_object_features.npz',allow_pickle=False);assert np.array_equal(local['keys'],keys)
    plan={'method':'DINOv3Base_imageonly_train_descriptors_v1','source':{k:v for k,v in source.items() if k!='local_weight_path'},'train_rows':1904,'test_rows':0,'held_rows':0,'manual_labels_read':False,
          'preprocess':'EXIFRGB,256aspectpreservingletterboxRGB124/116/104,bicubic,modelmeanstd',
          'descriptor':'L2CLS,L2validpatchmean,L2softobjectpatchmean,L2softobjectpatchstd;3072',
          'object_prior':'validfraction*(.1+.9frozenYOLOobjectfraction);fallbackwholevalidregion;notsegmentationtruth',
          'RoPE':'roundinstalledfloat32periodstobfloat16thenfloat32,asmodelcardreferencecompatibilitynote',
          'batch_size':2,'threads':4,'model_load':'localinstalledarchitecture,strictnumericstatekeys,assign=True;noindependentofficialforwardparityclaim',
          'localization_sha256':sha(cache/'litter_object_features.npz'),'source_code_sha256':sha(__file__)}
    dump(NOTES/'json/dinov3_base_feature_plan.json',plan)
    identity=source['checkpoint_sha256']+plan['source_code_sha256']+plan['localization_sha256'];partial=cache/'dinov3_base_partial.npz';blocks=[];done=0
    if partial.exists():
        prev=np.load(partial,allow_pickle=False)
        if str(prev['identity'].item())==identity and np.array_equal(prev['keys'],keys[:len(prev['keys'])]):blocks=[prev['features'].copy()];done=len(prev['keys'])
    model=timm.create_model(name,pretrained=False,num_classes=0);state=load_file(weight,device='cpu')
    assert set(state)==set(model.state_dict()),{'missing':sorted(set(model.state_dict())-set(state))[:10],'extra':sorted(set(state)-set(model.state_dict()))[:10]}
    model.load_state_dict(state,strict=True,assign=True);del state;gc.collect();model.eval()
    if hasattr(model,'rope') and hasattr(model.rope,'periods'):model.rope.periods=model.rope.periods.to(torch.bfloat16).to(torch.float32)
    cfg=resolve_model_data_config(model);prefix=model.num_prefix_tokens;assert model.num_features==768 and cfg['input_size']==(3,256,256)
    mean=np.array(cfg['mean'],np.float32);std=np.array(cfg['std'],np.float32)
    def prepare(i):
        row=frame.iloc[i];assert sha(ROOT/row.source_path)==row.source_sha256
        with Image.open(ROOT/row.source_path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        w,h=im.size;scale=256/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(256-nw)//2,(256-nh)//2
        canvas=np.full((256,256,3),[124,116,104],np.uint8);canvas[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        valid=np.zeros((256,256),np.float32);valid[dy:dy+nh,dx:dx+nw]=1.;obj=valid.copy()
        if local['focus_flags'][i]:
            a,b,c,d=local['boxes'][i];a,c=max(0,int(dx+a*scale)),min(256,int(np.ceil(dx+c*scale)));b,d=max(0,int(dy+b*scale)),min(256,int(np.ceil(dy+d*scale)))
            obj[:]=0;obj[b:d,a:c]=1.;obj*=valid
        mask=np.stack([valid,valid*(.1+.9*obj)])
        return torch.from_numpy(((canvas.astype(np.float32)/255-mean)/std).transpose(2,0,1).copy()),torch.from_numpy(mask)
    last_print=0.
    with torch.inference_mode():
        for offset in range(done,1904,2):
            ids=range(offset,min(1904,offset+2));batch=[prepare(i) for i in ids];tokens=model.forward_features(torch.stack([v[0] for v in batch]));patch=tokens[:,prefix:]
            assert patch.shape[1:]==(256,768)
            masks=F.interpolate(torch.stack([v[1] for v in batch]),size=(16,16),mode='area').flatten(2)
            valid,obj=masks[:,0],masks[:,1];valid_sum=valid.sum(1,keepdim=True).clamp_min(1e-8);obj_sum=obj.sum(1,keepdim=True).clamp_min(1e-8)
            vm=(patch*valid[:,:,None]).sum(1)/valid_sum;om=(patch*obj[:,:,None]).sum(1)/obj_sum;ov=((patch-om[:,None])**2*obj[:,:,None]).sum(1)/obj_sum
            feature=torch.cat([F.normalize(part,dim=1) for part in [tokens[:,0],vm,om,ov.clamp_min(0).sqrt()]],1)
            assert torch.isfinite(feature).all();blocks.append(feature.numpy().astype(np.float32));end=offset+len(batch)
            if time.time()-last_print>=25 or end==1904:
                print('DINOv3 Base train',end,'/1904; seconds',round(time.time()-start),flush=True)
                temp=partial.with_name('dinov3_base_writing.npz');np.savez_compressed(temp,keys=keys[:end],features=np.concatenate(blocks),identity=np.array(identity));temp.replace(partial);last_print=time.time()
    final=cache/'dinov3_base_train_features.npz';np.savez_compressed(final,keys=keys,features=np.concatenate(blocks),source_sha256=frame.source_sha256.to_numpy(str),identity=np.array(identity))
    dump(NOTES/'json/dinov3_base_feature_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'held_rows':0,'dimensions':3072,'prefix_tokens':prefix,'normalization':{'mean':cfg['mean'],'std':cfg['std']},'weight_sha256':source['checkpoint_sha256'],'feature_sha256':sha(final),'source_code_sha256':sha(__file__),'seconds':time.time()-start})
    print('DINOv3 Base features complete',flush=True)


if __name__=='__main__':main()
