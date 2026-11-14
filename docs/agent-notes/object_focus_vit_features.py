"""Streaming train-only object-focused ViT representations; originals retained."""
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from waste_vit_features import construct
from fewshot_train_only_probe import ROOT, NOTES, sha, dump

CACHE = NOTES / '.runtime'


def main():
    started = time.time()
    torch.set_num_threads(2)
    source = json.loads((CACHE / 'waste_vit_source.json').read_text())
    base_path = CACHE / 'waste_vit_train_features.npz'
    base = np.load(base_path, allow_pickle=False)
    keys = base['keys'].copy()
    assert len(keys) == 1904 and np.char.startswith(keys.astype(str), 'train:').all()
    frame = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    frame = frame[frame.split.eq('train') & ~frame.is_crop & frame.training_eligible].set_index('parent_key').loc[keys]
    assert np.array_equal(frame.source_sha256.to_numpy(str), base['source_sha256'])
    detector_source = json.loads((CACHE / 'litter_yolo_source.json').read_text())
    plan = {'scope': 'automatic object-focused representation;not new supervised crop or inherited class',
            'train_rows': 1904, 'test_rows': 0, 'held_rows': 0, 'external_images_used': False,
            'image_model_revision': source['revision'], 'image_weight_sha256': source['checkpoint_sha256'],
            'localizer_weight_sha256': detector_source['numeric_sha256'],
            'focus_policy': 'frozen pixel-box policy from litter_object_features;fallback whole image',
            'source_features_sha256': sha(base_path), 'source_code_sha256': sha(__file__),
            'only_numeric_features_written': True, 'manual_per_image_decisions': False}
    dump(NOTES / 'json/object_focus_vit_plan.json', plan)
    identity = plan['image_weight_sha256'] + plan['localizer_weight_sha256'] + plan['source_code_sha256']
    features = base['features'].copy()
    probabilities = base['probabilities'].copy()
    processed = np.zeros(len(keys), bool)
    focused = np.zeros(len(keys), bool)
    boxes = np.zeros((len(keys), 4), dtype='float32')
    partial_path = CACHE / 'object_focus_vit_partial.npz'
    if partial_path.exists():
        prev = np.load(partial_path, allow_pickle=False)
        if str(prev['identity'].item()) == identity and np.array_equal(prev['keys'], keys):
            features, probabilities = prev['features'].copy(), prev['probabilities'].copy()
            processed, focused, boxes = prev['processed'].copy(), prev['focus_flags'].copy(), prev['boxes'].copy()
    model = construct(source)
    cfg = source['preprocessor_config.json']
    last_new = time.time()
    last_print = 0.
    def save(path):
        temp = path.with_name(path.stem + '_writing.npz')
        np.savez_compressed(temp, keys=keys, features=features, probabilities=probabilities, processed=processed,
                            focus_flags=focused, boxes=boxes, identity=np.array(identity), source_sha256=base['source_sha256'])
        temp.replace(path)
    with torch.inference_mode():
        while not processed.all():
            path = CACHE / 'litter_object_partial.npz'
            if not path.exists():
                if time.time() - last_new > 180:
                    raise TimeoutError('No localisation progress for180seconds;partial features retained')
                time.sleep(1)
                continue
            try:
                contents = path.read_bytes()
                local = np.load(io.BytesIO(contents), allow_pickle=False)
                local_keys, local_flags, local_boxes = local['keys'].copy(), local['focus_flags'].copy(), local['boxes'].copy()
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                time.sleep(1)
                continue
            ready = len(local_keys)
            assert np.array_equal(keys[:ready], local_keys)
            todo = np.flatnonzero(~processed[:ready])
            if not len(todo):
                if time.time() - last_new > 180:
                    save(partial_path)
                    raise TimeoutError('No new localisation rows;resume later from saved progress')
                time.sleep(1)
                continue
            last_new = time.time()
            boxes[:ready] = local_boxes
            no_crop = todo[~local_flags[todo]]
            processed[no_crop] = True
            crops = todo[local_flags[todo]]
            for start in range(0, len(crops), 4):
                ids = crops[start:start + 4]
                tensors = []
                for i in ids:
                    row = frame.iloc[i]
                    assert sha(ROOT / row.source_path) == row.source_sha256
                    with Image.open(ROOT / row.source_path) as src:
                        im = ImageOps.exif_transpose(src).convert('RGB')
                        box = tuple(round(float(v)) for v in local_boxes[i])
                        assert 0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height
                        im = im.crop(box).resize((224, 224), Image.Resampling.BILINEAR)
                        values = np.array(im).astype('float32') / 255.
                    values = (values - np.array(cfg['image_mean'], dtype='float32')) / np.array(cfg['image_std'], dtype='float32')
                    tensors.append(torch.from_numpy(values.transpose(2, 0, 1).copy()))
                token = model.forward_features(torch.stack(tensors))
                vector = torch.cat([token[:, 0], token[:, 1:].mean(1), token[:, 1:].std(1, unbiased=False)], 1)
                features[ids] = vector.numpy()
                probabilities[ids] = model.head(token[:, 0]).softmax(1).numpy()
                processed[ids], focused[ids] = True, True
                if time.time() - last_print > 25:
                    print(f'Object-focused ViT {processed.sum()}/1904 | cropped{focused.sum()} | {time.time()-started:.0f}s', flush=True)
                    save(partial_path)
                    last_print = time.time()
            save(partial_path)
    final = CACHE / 'object_focus_vit_features.npz'
    save(final)
    assert np.isfinite(features).all() and np.isfinite(probabilities).all()
    dump(NOTES / 'json/object_focus_vit_extraction.json', {'status': 'passed', 'rows': 1904, 'test_rows': 0,
         'focus_rows': int(focused.sum()), 'full_image_fallback_rows': int((~focused).sum()),
         'feature_sha256': sha(final), 'source_code_sha256': sha(__file__), 'seconds': time.time() - started})
    print('Object-focused extraction complete', int(focused.sum()), 'focused images', flush=True)


if __name__ == '__main__':
    main()
