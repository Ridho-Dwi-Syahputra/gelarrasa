"""Frozen multiscale image patches for train-only weakly supervised pooling."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
import timm
from timm.data import resolve_model_data_config
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from fewshot_train_only_probe import ROOT,NOTES,OLD,sha,dump


def main():
    start=time.time();torch.set_num_threads(4)
    source=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(source['repository'],'model.safetensors',revision=source['revision'],local_files_only=True);assert sha(weight)==source['checkpoint_sha256']
    allowed=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);keys=allowed['keys'];assert len(keys)==1904 and np.char.startswith(keys.astype(str),'train:').all()
    profile=pd.read_csv(OLD/'output/manifests/stage1_v1/image_profile.csv').set_index('key').loc[keys]
    assert np.array_equal(profile.source_sha256.to_numpy(str),allowed['source_sha256'])
    roi=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);assert np.array_equal(roi['keys'],keys)
    code_sha=sha(__file__);identity=source['checkpoint_sha256']+code_sha;home=NOTES/'.runtime'/('convnext_patch_tokens_'+code_sha[:12]);home.mkdir(exist_ok=True)
    plan={'method':'frozen_ConvNeXt_multiscale_patch_bags_v1','train_rows':1904,'test_rows':0,'held_rows':0,
          'source':{k:source[k] for k in ['repository','revision','checkpoint_sha256']},'preprocess':'EXIFRGB224aspectpreservingletterbox,bicubic,sourceImageNetmeanstd',
          'fine_tokens':'stage192adaptiveavg14x14,196tokensx192','coarse_tokens':'stage7687x7,49tokensx768',
          'soft_region_prior':'validfraction*(.1+.9*automaticobjectfraction);fallbackwholevalidimage;no semanticclassread',
          'storage':'numericfloat16memmap;modeltrainingwillcastbatchfloat32','source_code_sha256':code_sha,'external_images_added':False}
    dump(NOTES/'json/convnext_patch_token_plan.json',plan)
    progress=home/'progress.json';done=0
    if progress.exists():
        prior=json.loads(progress.read_text());assert prior['identity']==identity;done=prior['completed_rows']
    definitions={'fine':((1904,196,192),'fine.npy'),'coarse':((1904,49,768),'coarse.npy'),
                 'fine_weight':((1904,196),'fine_weight.npy'),'coarse_weight':((1904,49),'coarse_weight.npy')}
    arrays={name:np.lib.format.open_memmap(home/file,mode='r+' if (home/file).exists() else 'w+',dtype=np.float16,shape=shape) for name,(shape,file) in definitions.items()}
    model=timm.create_model('convnext_tiny.fb_in22k',pretrained=False,num_classes=source['classes']);model.load_state_dict(load_file(weight),strict=True);model.eval()
    cfg=resolve_model_data_config(model);mean=np.array(cfg['mean'],np.float32);std=np.array(cfg['std'],np.float32)
    def prepare(index):
        row=profile.iloc[index];path=ROOT/row.source_path
        if not path.exists():path=ROOT/'data'/row.source_path
        assert sha(path)==row.source_sha256
        with Image.open(path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        w,h=im.size;scale=224/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(224-nw)//2,(224-nh)//2
        canvas=np.full((224,224,3),[124,116,104],np.uint8);canvas[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        valid=np.zeros((224,224),np.float32);valid[dy:dy+nh,dx:dx+nw]=1;mask=valid.copy()
        if roi['focus_flags'][index]:
            a,b,c,d=roi['boxes'][index];a,c=int(dx+a*scale),int(np.ceil(dx+c*scale));b,d=int(dy+b*scale),int(np.ceil(dy+d*scale))
            mask[:]=0;mask[max(0,b):min(224,d),max(0,a):min(224,c)]=1;mask*=valid
        soft=valid*(.1+.9*mask)
        return torch.from_numpy(((canvas.astype(np.float32)/255-mean)/std).transpose(2,0,1).copy()),torch.from_numpy(soft)
    with torch.inference_mode():
        for offset in range(done,1904,16):
            values=[prepare(i) for i in range(offset,min(1904,offset+16))];x=model.stem(torch.stack([v[0] for v in values]));mask=torch.stack([v[1] for v in values])[:,None]
            for i,stage in enumerate(model.stages):
                x=stage(x)
                if i==1:
                    fine=F.adaptive_avg_pool2d(x,(14,14)).flatten(2).transpose(1,2);fw=F.interpolate(mask,size=(14,14),mode='area').flatten(1)
                if i==3:coarse=x.flatten(2).transpose(1,2);cw=F.interpolate(mask,size=(7,7),mode='area').flatten(1)
            end=offset+len(values)
            for name,value in [('fine',fine),('coarse',coarse),('fine_weight',fw),('coarse_weight',cw)]:
                assert torch.isfinite(value).all();arrays[name][offset:end]=value.numpy().astype(np.float16)
            if offset==done or end%128==0 or end==1904:
                for a in arrays.values():a.flush()
                dump(progress,{'identity':identity,'completed_rows':end})
                print(f'Patch bags train {end}/1904 | {time.time()-start:.0f}s',flush=True)
    files={name:{'path':str((home/file).relative_to(ROOT)),'sha256':sha(home/file),'shape':list(shape)} for name,(shape,file) in definitions.items()}
    index={'status':'passed','train_rows':1904,'test_rows':0,'keys':keys.tolist(),'source_sha256':allowed['source_sha256'].tolist(),'files':files,'identity':identity,'source_code_sha256':code_sha,'seconds':time.time()-start}
    dump(NOTES/'.runtime/convnext_patch_token_index.json',index)
    dump(NOTES/'json/convnext_patch_token_extraction.json',{k:index[k] for k in ['status','train_rows','test_rows','files','source_code_sha256','seconds']})
    print('Patch bags complete',flush=True)


if __name__=='__main__':main()
