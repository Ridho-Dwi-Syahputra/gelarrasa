"""Frozen image-only global/local descriptors; no semantic or archive inputs.

Patch moments exclude letterbox padding by patch-center geometry. They are a
local-appearance hypothesis, not a foreground mask or a material classifier.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import timm
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/embeddings/stage5_patch_v1'
REPORT = ROOT / 'reports/errors/stage5_material_structure_20261106'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def valid_patch_mask(row, size, flip=False):
    scale = size / max(row.width, row.height)
    width, height = [max(1, min(size, round(v * scale))) for v in (row.width, row.height)]
    left, top = (size-width)//2, (size-height)//2
    centers = np.arange(size//14)*14 + 7
    mask = ((centers[:, None] >= top) & (centers[:, None] < top+height) &
            (centers[None, :] >= left) & (centers[None, :] < left+width))
    if flip:
        mask = np.fliplr(mask)
    assert mask.any()
    return mask.reshape(-1).copy()


def image_input(row, size, flip):
    if size == 224:
        with Image.open(ROOT / row.preview_path) as source:
            im = source.convert('RGB')
    else:
        with Image.open(ROOT / row.processed_path) as source:
            original = source.convert('RGB')
        scale = size / max(original.size)
        dims = tuple(max(1, min(size, round(v*scale))) for v in original.size)
        im = Image.new('RGB', (size, size), (124, 116, 104))
        im.paste(original.resize(dims, Image.Resampling.LANCZOS),
                 ((size-dims[0])//2, (size-dims[1])//2))
    if flip:
        im = ImageOps.mirror(im)
    return normalize(to_tensor(im), [.485, .456, .406], [.229, .224, .225])


def extract(model, table, size=224, flip=False):
    blocks = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(table), 8):
            rows = list(table.iloc[start:start+8].itertuples())
            inputs = torch.stack([image_input(r, size, flip) for r in rows])
            tokens = model.forward_features(inputs)
            cls = model.forward_head(tokens)
            patches = tokens[:, model.num_prefix_tokens:]
            mask = torch.from_numpy(np.stack([valid_patch_mask(r, size, flip) for r in rows])).to(patches.dtype)
            weight = mask / mask.sum(1, keepdim=True)
            mean = (patches*weight[:, :, None]).sum(1)
            std = ((patches-mean[:, None, :]).square()*weight[:, :, None]).sum(1).clamp_min(0).sqrt()
            parts = [torch.nn.functional.normalize(v, dim=1) for v in (cls, mean, std)]
            blocks.append(torch.cat(parts, dim=1).cpu().numpy().astype('float32'))
            if start == 0 or (start+8) % 200 == 0:
                print(f'patches size={size} flip={flip} {min(start+8,len(table))}/{len(table)} {time.time()-started:.0f}s', flush=True)
    features = np.vstack(blocks)
    assert features.shape == (len(table), 1152) and np.isfinite(features).all()
    return features


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    OUT.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / 'output/manifests/stage1_v1/image_profile.csv'
    p = pd.read_csv(profile_path, dtype={'image_id': str})
    probe_path = ROOT / 'output/embeddings/stage2_repr_v1/view_probe_ids.csv'
    probe = pd.read_csv(probe_path)
    probe = p.set_index('key').loc[probe.key].reset_index()
    assert len(p)==2400 and len(probe)==256 and probe.split.eq('train').all()
    checkpoint_path = ROOT / 'output/embeddings/stage2_repr_v1/dino_checkpoint.json'
    info = json.loads(checkpoint_path.read_text())
    plan = dict(started_local=datetime.now().astimezone().isoformat(),
        hypotheses=['Local patch moments may complement global object/scene features.',
                    '32 prototypes may preserve material subtypes that eight groups mix.'],
        model=info['model'], checkpoint_sha256=info['checkpoint_sha256'],
        profile_sha256=sha(profile_path), probe_sha256=sha(probe_path),
        extractor_sha256=sha(__file__),
        feature_blocks=['L2 CLS', 'L2 mean non-padding patch tokens', 'L2 std non-padding patch tokens'],
        views=['original224 all images', 'hflip224 existing random256 train', 'original336 same256 train'],
        fitting_policy='No backbone fitting, labels, reference archive, display names, or test fitting.',
        background_policy='Only padding excluded; remaining background is retained and can confound texture.',
        downstream_design=dict(seeds=[11,29,47], clusters=[8,32],
            feature_branches=['global','local moments','global .5/local mean .25/local std .25'],
            merging='weighted Ward of 32 fitted prototypes to eight, not semantic merging',
            selection='8 occupied heldout groups, min5, maximize .4 flip + .4 scale + .2 cross-representation neighbor agreement; development only',
            archived_reference_used_for_selection=False))
    dump(REPORT/'plan_before_run.json', plan)
    weight = hf_hub_download(info['repository'], 'model.safetensors', revision=info['revision'], local_files_only=True)
    assert sha(weight)==info['checkpoint_sha256']
    model = timm.create_model(info['model'], pretrained=True, num_classes=0,
        dynamic_img_size=True, pretrained_cfg_overlay={'file': weight}).eval()
    for name, table, size, flip in [('original224',p,224,False), ('hflip224',probe,224,True), ('original336',probe,336,False)]:
        features = extract(model, table, size, flip)
        np.savez_compressed(OUT/(name+'.npz'), keys=table.key.to_numpy(str), features=features)
    old = np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz', allow_pickle=False)
    new = np.load(OUT/'original224.npz', allow_pickle=False)
    delta = float(np.max(np.abs(old['features']-new['features'][:, :384])))
    assert np.array_equal(old['keys'],new['keys']) and delta < 1e-5
    dump(REPORT/'feature_integrity.json', dict(status='passed',images=2400,random_train_views=256,
         old_cls_max_abs_difference=delta,reference_archive_read=False,backbone_finetuned=False,
         feature_hashes={v.name:sha(v) for v in OUT.glob('*.npz')},extractor_sha256=sha(__file__)))
    print('Feature extraction passed', flush=True)


if __name__=='__main__':
    main()
