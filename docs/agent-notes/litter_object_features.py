"""Image-only local object features; no external images, review labels or test."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms
from safetensors.torch import load_file
from fewshot_train_only_probe import ROOT, NOTES, sha, dump

CACHE = NOTES / '.runtime'
sys.path.insert(0, str((CACHE / 'ultralytics_pkg').resolve()))
os.environ['YOLO_CONFIG_DIR'] = str((CACHE / 'yolo_settings').resolve())
os.environ['YOLO_AUTOINSTALL'] = 'False'
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics.nn.tasks import SegmentationModel


def prepare(path, size=512):
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert('RGB')
    ratio = size / max(im.size)
    dims = (round(im.width * ratio), round(im.height * ratio))
    pad = ((size - dims[0]) // 2, (size - dims[1]) // 2)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(im.resize(dims, Image.Resampling.BILINEAR), pad)
    return torch.from_numpy(np.array(canvas).transpose(2, 0, 1).copy()).float() / 255., (ratio, pad, im.size)


def main():
    started = time.time()
    torch.set_num_threads(6)
    frame = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    frame = frame[frame.split.eq('train') & ~frame.is_crop & frame.training_eligible].sort_values('parent_id').reset_index(drop=True)
    assert len(frame) == 1904 and frame.parent_key.str.startswith('train:').all()
    source = json.loads((CACHE / 'litter_yolo_source.json').read_text())
    weight = CACHE / 'litter_yolo_numeric.safetensors'
    assert sha(weight) == source['numeric_sha256']
    model = SegmentationModel(cfg=source['architecture'], ch=3, nc=5, verbose=False)
    model.load_state_dict(load_file(str(weight)), strict=True)
    model.eval()
    plan = {'scope': 'additional image-only localisation/feature representation,not source labels as targets',
            'repository': source['repository'], 'revision': source['revision'], 'numeric_sha256': source['numeric_sha256'],
            'source_class_names': source['source_class_names'], 'package': 'ultralytics8.4.204,isolatedtarget,no canonical environment mutation',
            'input': 'RGB512letterbox,pad114,float0..1', 'train_rows': 1904, 'test_rows': 0, 'held_rows': 0,
            'external_images_used': False, 'external_inference': False, 'serialized_callables_executed': False,
            'descriptor': 'three pyramid global mean+max1792;mask-weighted mean896;source detection statistics28',
            'detection': {'threshold': .15, 'nms_iou': .5, 'maximum_proposals': 20},
            'focus': 'highestscore*sqrtarea box,score>=.35,area.08..85,clipnativeframe,5pctmargin',
            'focus_not_new_supervised_crop': True, 'no_class_inheritance_or_per_image_rules': True,
            'source_code_sha256': sha(__file__)}
    dump(NOTES / 'json/litter_object_plan.json', plan)
    identity = source['numeric_sha256'] + plan['source_code_sha256']
    partial = CACHE / 'litter_object_partial.npz'
    vectors, boxes, focus_flags, metadata_rows, start = [], [], [], [], 0
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        if str(z['identity'].item()) == identity and np.array_equal(z['keys'], frame.parent_key.to_numpy(str)[:len(z['keys'])]):
            start = len(z['keys'])
            vectors = list(z['features'])
            boxes = list(z['boxes'])
            focus_flags = list(z['focus_flags'])
            metadata_rows = list(z['detection_stats'])
    def save(path, count):
        np.savez_compressed(path, keys=frame.parent_key.to_numpy(str)[:count], features=np.array(vectors, dtype='float32'),
                            boxes=np.array(boxes, dtype='float32'), focus_flags=np.array(focus_flags, bool),
                            detection_stats=np.array(metadata_rows, dtype='float32'), identity=np.array(identity),
                            source_sha256=frame.source_sha256.to_numpy(str)[:count])
    with torch.inference_mode():
        for offset in range(start, len(frame), 4):
            rows = frame.iloc[offset:offset + 4]
            batch, specs = [], []
            for row in rows.itertuples():
                assert sha(ROOT / row.source_path) == row.source_sha256
                image, spec = prepare(ROOT / row.source_path)
                batch.append(image)
                specs.append(spec)
            raw, auxiliary = model(torch.stack(batch))
            decoded, prototype = raw
            pyramid = auxiliary['feats']
            global_features = torch.cat([torch.cat([t.mean((2, 3)), t.amax((2, 3))], 1) for t in pyramid], 1)
            assert global_features.shape[1] == 1792
            for i, spec in enumerate(specs):
                predictions = decoded[i].transpose(0, 1)
                score, category = predictions[:, 4:9].max(1)
                selected = torch.nonzero(score >= .15, as_tuple=True)[0]
                xywh = predictions[:, :4]
                xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1).clamp(0, 512)
                if len(selected):
                    keep = batched_nms(xyxy[selected], score[selected], category[selected], .5)[:20]
                    selected = selected[keep]
                stats = np.zeros(28, dtype='float32')
                area = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / 512**2).clamp(0, 1)
                for c in range(5):
                    ids = selected[category[selected] == c]
                    if len(ids):
                        stats[c] = float(score[ids].max())
                        stats[5 + c] = float(score[ids].mean())
                        stats[10 + c] = float(area[ids].max())
                        stats[15 + c] = np.log1p(len(ids))
                        stats[20 + c] = float((score[ids] * area[ids]).sum())
                stats[25] = len(selected) / 20.
                best = int(selected[(score[selected] * area[selected].sqrt()).argmax()]) if len(selected) else None
                ratio, pad, native = spec
                native_box = np.array([0., 0., float(native[0]), float(native[1])], dtype='float32')
                focused = False
                if best is not None:
                    stats[26], stats[27] = float(score[best]), float(area[best])
                    xy = xyxy[best].numpy().copy()
                    xy[[0, 2]] = (xy[[0, 2]] - pad[0]) / ratio
                    xy[[1, 3]] = (xy[[1, 3]] - pad[1]) / ratio
                    margin = .05 * max(xy[2] - xy[0], xy[3] - xy[1])
                    xy += np.array([-margin, -margin, margin, margin])
                    xy[[0, 2]] = np.clip(xy[[0, 2]], 0, native[0])
                    xy[[1, 3]] = np.clip(xy[[1, 3]], 0, native[1])
                    fraction = (xy[2] - xy[0]) * (xy[3] - xy[1]) / (native[0] * native[1])
                    focused = score[best] >= .35 and .08 <= fraction <= .85 and min(xy[2] - xy[0], xy[3] - xy[1]) >= 16
                    if focused:
                        native_box = xy
                    mask = (predictions[best, 9:] @ prototype[i].flatten(1)).reshape(prototype.shape[-2:]).sigmoid()
                    # Keep mask support inside the detected rectangle.
                    yy, xx = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]), indexing='ij')
                    box = xyxy[best] * mask.shape[0] / 512
                    mask *= (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
                else:
                    mask = torch.ones(prototype.shape[-2:])
                foreground = []
                for t in pyramid:
                    w = F.interpolate(mask[None, None], size=t.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
                    foreground.append((t[i] * w).sum((1, 2)) / w.sum().clamp_min(1e-6))
                feature = torch.cat([global_features[i], *foreground]).numpy()
                assert feature.shape == (2688,) and np.isfinite(feature).all()
                vectors.append(np.r_[feature, stats])
                boxes.append(native_box)
                focus_flags.append(bool(focused))
                metadata_rows.append(stats)
            end = offset + len(rows)
            if offset == start or end % 128 == 0 or end == len(frame):
                print(f'Image-only localisation {end}/{len(frame)} | focus{sum(focus_flags)} | {time.time()-started:.0f}s', flush=True)
                save(partial, end)
    final = CACHE / 'litter_object_features.npz'
    save(final, len(frame))
    dump(NOTES / 'json/litter_object_extraction.json', {'status': 'passed', 'rows': 1904, 'test_rows': 0,
         'focus_candidate_rows': int(sum(focus_flags)), 'dimensions': 2716, 'feature_sha256': sha(final),
         'source_code_sha256': sha(__file__), 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
