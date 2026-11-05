"""Image-only representation and view diagnostics. No semantic targets are read.

Uses a pinned pretrained DINOv2 checkpoint locally; no inference API, VLM,
competition labels, or test fitting. Frozen features are exploratory artifacts.
"""
from pathlib import Path
import hashlib
import json
import time
import os
import importlib.metadata

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import timm
from huggingface_hub import HfApi, hf_hub_download
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import to_tensor, normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/embeddings/stage2_repr_v1'
REPORT = ROOT / 'reports/errors/stage2_20261105'
MODEL = 'vit_small_patch14_dinov2.lvd142m'
REPO = 'timm/' + MODEL
SEED = 20261105


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def log(msg):
    s = time.strftime('%H:%M:%S') + ' ' + msg
    print(s, flush=True)
    with (REPORT / 'visual_features.log').open('a', encoding='utf-8') as f:
        f.write(s + '\n')


def letterbox(im, size):
    scale = size / max(im.size)
    dims = tuple(max(1, min(size, round(v * scale))) for v in im.size)
    out = Image.new('RGB', (size, size), (124, 116, 104))
    out.paste(im.resize(dims, Image.Resampling.LANCZOS),
              ((size-dims[0])//2, (size-dims[1])//2))
    return out


def extract(model, df, view, batch=12):
    blocks = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(df), batch):
            inputs = []
            for r in df.iloc[start:start+batch].itertuples():
                if view in ('original224', 'hflip224'):
                    with Image.open(ROOT / r.preview_path) as source:
                        im = source.convert('RGB')
                    if view == 'hflip224':
                        im = ImageOps.mirror(im)
                else:
                    with Image.open(ROOT / r.processed_path) as source:
                        im = letterbox(source.convert('RGB'), 336)
                inputs.append(normalize(to_tensor(im), [.485,.456,.406], [.229,.224,.225]))
            x = model(torch.stack(inputs)).cpu().numpy().astype('float32')
            blocks.append(x)
            if (start // batch + 1) % 20 == 0:
                log(f'{view} {min(start+batch,len(df))}/{len(df)} {time.time()-started:.0f}s')
    x = np.vstack(blocks)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    assert np.isfinite(x).all() and len(x) == len(df)
    return x


def save_features(path, df, x):
    assert df.key.is_unique
    np.savez_compressed(path, keys=df.key.to_numpy(dtype=str), features=x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    force_recompute = os.environ.get('GELAR_RECOMPUTE_FEATURES', '0') == '1'
    log(f'Force feature recomputation: {force_recompute}')
    df = pd.read_csv(ROOT / 'output/manifests/stage1_v1/image_profile.csv', dtype={'image_id':str})
    unique = df[df.split.eq('train')].drop_duplicates('pixel_sha256')
    probe = unique.sample(n=256, random_state=SEED).sort_values('key').copy()
    probe[['key','split','image_id','pixel_sha256']].to_csv(OUT / 'view_probe_ids.csv', index=False)
    assert len(df) == 2400 and len(unique) == 1919
    freeze_paths = [ROOT / 'output/groups/stage1_v1/cluster_assignments.csv',
                    ROOT / 'output/manifests/stage1_v1/image_profile.csv',
                    ROOT / 'configs/data_contract.json']
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in freeze_paths}
    plan = dict(date='2026-11-05', purpose='exploratory unlabeled representation ablation',
                full_population=len(df), fit_population='unique train only, in separate clustering script',
                views=['original224_all','hflip224_256_random_train','original336_256_random_train'],
                probe_selection='256 unique train sampled by seed20261105, no labels or disagreements',
                seeds=[11,29,47], pca_components=[None,64,128,256], k=8,
                external_dataset_downloaded=False, archived_labels_read=False,
                semantic_targets_read=False, test_used_for_fitting=False,
                frozen_inputs=frozen, force_feature_recomputation=force_recompute)
    dump(REPORT / 'representation_plan.json', plan)
    provenance_path = OUT / 'dino_checkpoint.json'
    if provenance_path.exists():
        revision = json.loads(provenance_path.read_text())['revision']
    else:
        revision = HfApi().model_info(REPO).sha
    weight = hf_hub_download(repo_id=REPO, filename='model.safetensors', revision=revision)
    provenance = dict(model=MODEL, repository=REPO, revision=revision,
                      checkpoint_sha256=sha(weight), license='apache-2.0', vision_only=True,
                      reference='https://huggingface.co/' + REPO,
                      checkpoint_default_resolution=518, exploratory_resolution=224,
                      finetuned_on_competition=False, inference_device='cpu',
                      torch=torch.__version__, timm=timm.__version__,
                      transform='stage1 full-frame letterbox224 LANCZOS + ImageNet normalization',
                      output='L2-normalized CLS embedding, 384 dimensions')
    dump(provenance_path, provenance)
    log(f'Pinned checkpoint {revision}, SHA256 {provenance["checkpoint_sha256"]}')
    model = timm.create_model(MODEL, pretrained=True, num_classes=0,
                             dynamic_img_size=True,
                             pretrained_cfg_overlay={'file':weight}).eval()
    feature_path = OUT / 'dinov2_features.npz'
    if force_recompute or not feature_path.exists():
        x = extract(model, df, 'original224')
        assert x.shape == (2400,384)
        save_features(feature_path, df, x)
    else:
        z = np.load(feature_path)
        assert np.array_equal(z['keys'],df.key.to_numpy(str)) and z['features'].shape == (2400,384)
        log('Existing DINO features reused after key/shape checks')
    for view in ['hflip224','original336']:
        path = OUT / f'dinov2_{view}_probe.npz'
        if force_recompute or not path.exists():
            save_features(path, probe, extract(model, probe, view, batch=8))
    del model
    log('DINO features complete; starting paired ResNet transform probes')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model.fc = torch.nn.Identity()
    for view in ['hflip224','original336']:
        path = OUT / f'resnet50_{view}_probe.npz'
        if force_recompute or not path.exists():
            save_features(path, probe, extract(model, probe, view, batch=8))
    for rel, digest in frozen.items():
        assert sha(ROOT / rel) == digest, f'Unexpected input change: {rel}'
    dump(OUT / 'feature_validation.json', dict(
        status='passed', total_images=2400, unique_train=1919, probe_images=256,
        input_artifacts_unchanged=True, feature_files={
            p.name:sha(p) for p in OUT.glob('*.npz')}, script_sha256=sha(Path(__file__)),
        libraries={k:importlib.metadata.version(k) for k in ['torch','timm','numpy','pandas','Pillow','huggingface_hub']}))
    log('All feature extraction and immutable-input checks passed')


if __name__ == '__main__':
    main()
