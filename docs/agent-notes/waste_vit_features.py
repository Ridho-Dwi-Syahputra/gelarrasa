"""Pinned image-only waste ViT features; local numeric conversion, train only."""
from functools import partial
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
import timm
from safetensors.torch import load_file
from fewshot_train_only_probe import ROOT, NOTES, sha, dump

CACHE = NOTES / '.runtime'


def construct(source):
    cfg = source['config.json']
    assert cfg['model_type'] == 'vit' and cfg['architectures'] == ['ViTForImageClassification']
    weights = load_file(source['checkpoint_path'])
    model = timm.create_model('vit_base_patch16_224', pretrained=False,
                             num_classes=len(cfg['id2label']), norm_layer=partial(nn.LayerNorm, eps=cfg['layer_norm_eps']))
    converted = {}
    direct = {'cls_token': 'vit.embeddings.cls_token', 'pos_embed': 'vit.embeddings.position_embeddings',
              'patch_embed.proj.weight': 'vit.embeddings.patch_embeddings.projection.weight',
              'patch_embed.proj.bias': 'vit.embeddings.patch_embeddings.projection.bias',
              'norm.weight': 'vit.layernorm.weight', 'norm.bias': 'vit.layernorm.bias',
              'head.weight': 'classifier.weight', 'head.bias': 'classifier.bias'}
    used = set()
    for target, original in direct.items():
        converted[target] = weights[original]
        used.add(original)
    for block in range(12):
        hf = f'vit.encoder.layer.{block}.'
        dest = f'blocks.{block}.'
        for suffix in ['weight', 'bias']:
            parts = [hf + 'attention.attention.' + p + '.' + suffix for p in ['query', 'key', 'value']]
            converted[dest + 'attn.qkv.' + suffix] = torch.cat([weights[k] for k in parts], dim=0)
            used.update(parts)
            for a, b in [('attn.proj', 'attention.output.dense'), ('norm1', 'layernorm_before'),
                         ('norm2', 'layernorm_after'), ('mlp.fc1', 'intermediate.dense'), ('mlp.fc2', 'output.dense')]:
                key = hf + b + '.' + suffix
                converted[dest + a + '.' + suffix] = weights[key]
                used.add(key)
    assert used == set(weights), sorted(set(weights) - used)
    assert set(converted) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert converted[key].shape == value.shape and torch.isfinite(converted[key]).all()
    model.load_state_dict(converted, strict=True)
    return model.eval()


def prepare(path, cfg):
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert('RGB')
        # Match the original ViT processor: direct224 resize,not center crop.
        im = im.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.array(im).astype('float32') / 255.
    values = (values - np.array(cfg['image_mean'], dtype='float32')) / np.array(cfg['image_std'], dtype='float32')
    return torch.from_numpy(values.transpose(2, 0, 1).copy())


def main():
    started = time.time()
    torch.set_num_threads(6)
    frame = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    frame = frame[frame.split.eq('train') & ~frame.is_crop & frame.training_eligible].sort_values('parent_id').reset_index(drop=True)
    assert len(frame) == 1904 and frame.parent_key.str.startswith('train:').all()
    source = json.loads((CACHE / 'waste_vit_source.json').read_text())
    assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
    cfg = source['preprocessor_config.json']
    plan = {'scope': 'image-only pretrained base representations,not direct final predictions',
            'repository': source['repository'], 'revision': source['revision'], 'checkpoint_sha256': source['checkpoint_sha256'],
            'model_type': 'ViTForImageClassification,not CLIP/SigLIP/VLM',
            'source_class_order': [source['config.json']['id2label'][str(i)] for i in range(12)],
            'source_data_overlap_with_competition_unknown': True,
            'train_rows': 1904, 'test_rows': 0, 'held_rows': 0, 'external_images_used': False, 'external_inference': False,
            'descriptor': 'CLS,mean patch,std patch after final LayerNorm,2304dimensions',
            'preprocess': cfg, 'conversion': 'strict complete numeric HFViT-to-timm state mapping; original LN epsilon',
            'conversion_limit': 'state shapes and forward invariants checked;no independent Transformers numerical parity yet',
            'source_code_sha256': sha(__file__),
            'qna_source': 'https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/kxodWG5OBD7gQgP7'}
    dump(NOTES / 'json/waste_vit_feature_plan.json', plan)
    model = construct(source)
    with torch.inference_mode():
        q = model.forward_features(torch.zeros(2, 3, 224, 224))
        assert q.shape == (2, 197, 768) and torch.allclose(q[0], q[1]) and torch.isfinite(q).all()
    identity = source['checkpoint_sha256'] + plan['source_code_sha256']
    partial_cache = CACHE / 'waste_vit_partial.npz'
    features, probabilities, start = [], [], 0
    if partial_cache.exists():
        z = np.load(partial_cache, allow_pickle=False)
        if str(z['identity'].item()) == identity and np.array_equal(z['keys'], frame.parent_key.to_numpy(str)[:len(z['keys'])]):
            start = len(z['keys'])
            features = [z['features']]
            probabilities = [z['probabilities']]
    with torch.inference_mode():
        for offset in range(start, len(frame), 8):
            rows = frame.iloc[offset:offset + 8]
            batch = []
            for row in rows.itertuples():
                assert sha(ROOT / row.source_path) == row.source_sha256
                batch.append(prepare(ROOT / row.source_path, cfg))
            token = model.forward_features(torch.stack(batch))
            feature = torch.cat([token[:, 0], token[:, 1:].mean(1), token[:, 1:].std(1, unbiased=False)], dim=1)
            p = model.head(token[:, 0]).softmax(1)
            features.append(feature.numpy().astype('float32'))
            probabilities.append(p.numpy().astype('float32'))
            end = offset + len(rows)
            if offset == start or end % 128 == 0 or end == len(frame):
                print(f'Image-only waste ViT {end}/{len(frame)} | {time.time() - started:.0f}s', flush=True)
                np.savez_compressed(partial_cache, keys=frame.parent_key.to_numpy(str)[:end], features=np.concatenate(features),
                                    probabilities=np.concatenate(probabilities), identity=np.array(identity))
    final = CACHE / 'waste_vit_train_features.npz'
    np.savez_compressed(final, keys=frame.parent_key.to_numpy(str), features=np.concatenate(features), probabilities=np.concatenate(probabilities),
                        identity=np.array(identity), source_sha256=frame.source_sha256.to_numpy(str))
    dump(NOTES / 'json/waste_vit_feature_extraction.json', {'status': 'passed', 'rows': 1904, 'dimensions': 2304,
          'test_rows': 0, 'held_rows': 0, 'feature_sha256': sha(final), 'source_code_sha256': sha(__file__), 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
