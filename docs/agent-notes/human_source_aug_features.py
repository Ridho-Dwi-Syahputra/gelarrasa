"""Eight deterministic views of the same34budgeted sources, not new annotations."""
import json
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageEnhance,ImageDraw
import torch
import timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from timm.data import resolve_model_data_config
from fewshot_train_only_probe import ROOT,NOTES,OLD,sha,dump


def main():
    start=time.time();torch.set_num_threads(2)
    human=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(human)==34 and human.assigned_class.value_counts().le(5).all()
    frame=pd.read_csv(ROOT/'data/processed/manifest.csv').query('split=="train" and not is_crop and training_eligible').set_index('parent_key').loc[human.key]
    keys=human.key.to_numpy(str);base=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);cnn=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    local=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);lookup=dict(zip(base['keys'],range(len(base['keys']))));ids=np.array([lookup[k] for k in keys])
    assert np.array_equal(cnn['keys'],base['keys']) and np.array_equal(local['keys'],base['keys'])
    plan={'scope':'same34existinghumantrainimages,differentsource-preservingviews;no newhumanlabels','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'views':['originalcache','horizontalflip','automaticobjectcrop10pctmargin','objectcrophflip','rotate+15expand','rotate-15expand','brightness.85contrast1.1','brightness1.15contrast.9'],
          'labels':'oneexistingcategorypersource;allviewsofaquerysourcegroupmustbeexcludedfromfit',
          'augmentation_limit':'automaticROIisnotgroundtruth;rawpixelsunchanged;nogeneratedexternalimages',
          'features':'image-onlyDINOBaseCLS/patchmean/std/ROImean3072andConvNeXtprelogits768',
          'source_code_sha256':sha(__file__),'raw_source_sha256':frame.source_sha256.to_numpy(str).tolist()}
    dump(NOTES/'json/human_source_augmentation_plan.json',plan)
    ds=json.loads((NOTES/'.runtime/dinov2_base_source.json').read_text());dp=NOTES/'.runtime/dinov2_vitb14_pretrain.pth';assert sha(dp)==ds['checkpoint_sha256']
    dmodel=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,dynamic_img_size=True)
    state=torch.load(dp,map_location='cpu',weights_only=True);state.pop('mask_token');dmodel.load_state_dict(state,strict=True);del state;dmodel.eval()
    cs=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text());cp=hf_hub_download(cs['repository'],'model.safetensors',revision=cs['revision'],local_files_only=True);assert sha(cp)==cs['checkpoint_sha256']
    cmodel=timm.create_model('convnext_tiny.fb_in22k',pretrained=False,num_classes=cs['classes']);state=load_file(cp);cmodel.load_state_dict(state,strict=True);del state;cmodel.eval()
    cfg=resolve_model_data_config(cmodel);mean=np.array(cfg['mean'],np.float32);std=np.array(cfg['std'],np.float32)
    features=np.zeros((34,8,3072),np.float32);cfeatures=np.zeros((34,8,768),np.float32);features[:,0]=base['features'][ids];cfeatures[:,0]=cnn['features'][ids]
    partial=NOTES/'.runtime/human_source_aug_partial.npz';done=0;identity=plan['source_code_sha256']+ds['checkpoint_sha256']+cs['checkpoint_sha256']
    if partial.exists():
        z=np.load(partial,allow_pickle=False)
        if str(z['identity'].item())==identity and np.array_equal(z['keys'],keys):features=z['features'];cfeatures=z['cnn_features'];done=int(z['completed'].item())
    def prepare(im,mask):
        w,h=im.size;scale=224/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(224-nw)//2,(224-nh)//2
        rgb=np.full((224,224,3),[124,116,104],np.uint8);rgb[dy:dy+nh,dx:dx+nw]=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC))
        region=np.zeros((224,224),np.float32);region[dy:dy+nh,dx:dx+nw]=np.asarray(mask.resize((nw,nh),Image.Resampling.BILINEAR)).astype(np.float32)/255
        roi=torch.nn.functional.interpolate(torch.from_numpy(region)[None,None],size=(16,16),mode='area').flatten()
        if roi.sum()<1e-6:roi[:]=1
        return torch.from_numpy(((rgb.astype(np.float32)/255-mean)/std).transpose(2,0,1).copy()),roi
    with torch.inference_mode():
        for i in range(done,34):
            row=frame.iloc[i];path=ROOT/row.source_path;assert sha(path)==row.source_sha256
            with Image.open(path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
            w,h=im.size;mask=Image.new('L',im.size,255)
            if local['focus_flags'][ids[i]]:
                box=local['boxes'][ids[i]];mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).rectangle(tuple(float(v) for v in box),fill=255)
            a,b,c,d=mask.getbbox() or (0,0,w,h);mx,my=.1*(c-a),.1*(d-b);box=(max(0,int(a-mx)),max(0,int(b-my)),min(w,int(np.ceil(c+mx))),min(h,int(np.ceil(d+my))))
            crop,cm=im.crop(box),mask.crop(box)
            views=[(ImageOps.mirror(im),ImageOps.mirror(mask)),(crop,cm),(ImageOps.mirror(crop),ImageOps.mirror(cm)),
                   (im.rotate(15,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(124,116,104)),mask.rotate(15,expand=True,fillcolor=0)),
                   (im.rotate(-15,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(124,116,104)),mask.rotate(-15,expand=True,fillcolor=0)),
                   (ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(.85)).enhance(1.1),mask),
                   (ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.15)).enhance(.9),mask)]
            batch=[prepare(*v) for v in views];x=torch.stack([v[0] for v in batch]);rm=torch.stack([v[1] for v in batch])
            tokens=dmodel.forward_features(x);patch=tokens[:,1:];regional=(patch*rm[:,:,None]).sum(1)/rm.sum(1,keepdim=True)
            df=torch.cat([tokens[:,0],patch.mean(1),patch.std(1,unbiased=False),regional],dim=1)
            cf=cmodel.forward_head(cmodel.forward_features(x),pre_logits=True);assert df.shape==(7,3072) and cf.shape==(7,768)
            assert torch.isfinite(df).all() and torch.isfinite(cf).all();features[i,1:]=df.numpy();cfeatures[i,1:]=cf.numpy()
            if (i+1)%4==0 or i+1==34:
                np.savez_compressed(partial,keys=keys,features=features,cnn_features=cfeatures,completed=np.array(i+1),identity=np.array(identity))
                print(f'Augmented human source {i+1}/34 | {time.time()-start:.0f}s',flush=True)
    final=NOTES/'.runtime/human_source_aug_features.npz';np.savez_compressed(final,keys=keys,features=features,cnn_features=cfeatures,source_sha256=frame.source_sha256.to_numpy(str),identity=np.array(identity))
    dump(NOTES/'json/human_source_aug_extraction.json',{'status':'passed','human_sources':34,'views_per_source':8,'test_rows':0,'new_manual_labels':0,'feature_sha256':sha(final),'source_code_sha256':sha(__file__),'seconds':time.time()-start})


if __name__=='__main__':main()
