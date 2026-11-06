"""Cache frozen first11 ViT blocks for a real final-block adaptation pilot.

Fit subset is random512 of the existing fit split, selected before any class
evidence. Validation remains group-disjoint. No reference names/labels read.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageEnhance
import torch,timm
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_tensor,normalize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/embeddings/stage6_lastblock_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str}).set_index('key')
    split=pd.read_csv(ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv').set_index('key')
    fit=split.index[split.role.eq('fit_train')].to_numpy();held=split.index[split.role.eq('heldout_train')].to_numpy()
    chosen=np.sort(np.random.default_rng(20261106).choice(fit,512,replace=False))
    probe=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv')
    held_views=[k for k in probe.key if k in set(held)]
    assert not set(split.loc[chosen,'candidate_group'])&set(split.loc[held,'candidate_group'])
    info=json.loads((ROOT/'output/embeddings/stage2_repr_v1/dino_checkpoint.json').read_text())
    weight=hf_hub_download(info['repository'],'model.safetensors',revision=info['revision'],local_files_only=True)
    assert sha(weight)==info['checkpoint_sha256']
    model=timm.create_model(info['model'],pretrained=True,num_classes=0,dynamic_img_size=True,pretrained_cfg_overlay={'file':weight}).eval()
    collections=[('original224',list(chosen)+list(held)),('strong224',list(chosen)),('hflip224',held_views),('original336',held_views)]
    records=[];precision=[]
    old=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    old_map=dict(zip(old['keys'],old['features']))
    def input_image(key,view):
        r=p.loc[key]
        if view=='original336':
            with Image.open(ROOT/r.processed_path) as src:im=src.convert('RGB')
            size=336;scale=size/max(im.size);dims=tuple(max(1,min(size,round(v*scale))) for v in im.size)
            canvas=Image.new('RGB',(size,size),(124,116,104));canvas.paste(im.resize(dims,Image.Resampling.LANCZOS),((size-dims[0])//2,(size-dims[1])//2));im=canvas
        else:
            with Image.open(ROOT/r.preview_path) as src:im=src.convert('RGB')
        if view in ['strong224','hflip224']:im=ImageOps.mirror(im)
        if view=='strong224':
            # Fixed mild photometry; no object crop or material-changing hue shift.
            im=ImageEnhance.Brightness(im).enhance(.95);im=ImageEnhance.Contrast(im).enhance(1.05)
        return normalize(to_tensor(im),[.485,.456,.406],[.229,.224,.225])
    with torch.inference_mode():
        for view,keys in collections:
            n=577 if view=='original336' else 257
            cache=np.lib.format.open_memmap(OUT/f'{view}.npy',mode='w+',dtype='float16',shape=(len(keys),n,384))
            started=time.time()
            for first in range(0,len(keys),8):
                images=torch.stack([input_image(k,view) for k in keys[first:first+8]])
                tokens=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images))))
                for block in model.blocks[:-1]:tokens=block(tokens)
                cache[first:first+len(images)]=tokens.numpy().astype('float16')
                if first==0:
                    full=model.forward_head(model.norm(model.blocks[-1](tokens)))
                    reconstructed=model.forward_head(model.norm(model.blocks[-1](torch.from_numpy(np.array(cache[first:first+len(images)],dtype='float32')))))
                    original=torch.nn.functional.normalize(full,dim=1).numpy();recon=torch.nn.functional.normalize(reconstructed,dim=1).numpy()
                    delta=float(np.max(np.abs(original-recon)));assert delta<.001
                    precision.append(dict(view=view,fp16_roundtrip_max_abs_feature_error=delta))
                    if view=='original224':assert np.max(np.abs(original-np.stack([old_map[k] for k in keys[first:first+len(images)]])))<1e-5
                if first==0 or (first+8)%200==0:print(f'prefix {view} {min(first+8,len(keys))}/{len(keys)} {time.time()-started:.0f}s',flush=True)
            cache.flush();del cache
            pd.DataFrame({'key':keys}).to_csv(OUT/f'{view}_keys.csv',index=False)
            records.append(dict(view=view,rows=len(keys),tokens=n))
    result=dict(status='passed',fit_subset_keys=chosen.tolist(),heldout_keys=held.tolist(),view_records=records,
        source_sha256=sha(__file__),checkpoint_sha256=info['checkpoint_sha256'],cache_precision='float16, converted to float32 for training',
        precision_checks=precision,source_labels_read=False,reference_archive_read=False,
        augmentation='full frame hflip +brightness.95 contrast1.05 for strong fit view; no crop')
    (REPORT/'lastblock_cache.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Last-block cache ready',flush=True)

if __name__=='__main__':main()
