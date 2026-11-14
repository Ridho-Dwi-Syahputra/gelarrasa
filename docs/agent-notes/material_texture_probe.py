"""MINC image-only material descriptors and local geometric matching probes.

No archive annotations, unselected participant labels or test images are inputs.
The Caffe architecture is interpreted from the original authors' deploy file;
only validated numeric weights are loaded. No downloaded Python is executed.
"""
import argparse
import json
import re
import time
from collections import Counter
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump, unit

CACHE = NOTES / '.runtime'


def train_frame():
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    m = m[m.split.eq('train') & ~m.is_crop & m.training_eligible].copy().sort_values('parent_id').reset_index(drop=True)
    assert len(m) == 1904 and m.parent_key.str.startswith('train:').all()
    assert not set(m.parent_id) & {'1662', '1885'}
    return m


class MincNetwork:
    def __init__(self):
        proto = (CACHE / 'minc_deploy-googlenet.prototxt').read_text()
        self.layers = []
        for match in re.finditer(r'\blayers\s*\{', proto):
            depth, pos = 1, match.end()
            while depth:
                depth += (proto[pos] == '{') - (proto[pos] == '}')
                pos += 1
            block = proto[match.end():pos - 1]
            name = re.search(r'\bname:\s*"([^"]+)"', block).group(1)
            kind = re.search(r'\btype:\s*(\w+)', block).group(1)
            bottoms = re.findall(r'\bbottom:\s*"([^"]+)"', block)
            tops = re.findall(r'\btop:\s*"([^"]+)"', block)
            assert len(tops) == 1
            def number(key, default):
                found = re.search(r'\b' + key + r':\s*([.\d]+)', block)
                return float(found.group(1)) if found else default
            self.layers.append({'name': name, 'kind': kind, 'bottoms': bottoms, 'top': tops[0],
                                'kernel': int(number('kernel_size', 1)), 'stride': int(number('stride', 1)),
                                'padding': int(number('pad', 0)), 'size': int(number('local_size', 5)),
                                'alpha': number('alpha', 1e-4), 'beta': number('beta', .75),
                                'average': bool(re.search(r'pool:\s*AVE', block))})
        assert len(self.layers) == 142, len(self.layers)
        z = np.load(CACHE / 'minc_googlenet_numeric.npz', allow_pickle=False)
        self.weights = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        assert len(self.weights) == 116 and all(torch.isfinite(v).all() for v in self.weights.values())
        self.last_use = {name: i for i, l in enumerate(self.layers) for name in l['bottoms']}

    def __call__(self, batch):
        blobs = {'data': batch}
        pooled = None
        for i, l in enumerate(self.layers):
            kind = l['kind']
            xs = [blobs[k] for k in l['bottoms']]
            x = xs[0] if xs else None
            if kind in {'CONVOLUTION', 'INNER_PRODUCT'}:
                key = l['name'].replace('/', '_')
                w, b = self.weights[key + '__weights'], self.weights[key + '__biases']
                out = F.conv2d(x, w, b, l['stride'], l['padding']) if kind == 'CONVOLUTION' else F.linear(x.flatten(1), w.reshape(len(b), -1), b)
            elif kind == 'RELU':
                out = F.relu(x)
            elif kind == 'POOLING':
                if l['average']:
                    out = F.avg_pool2d(x, l['kernel'], l['stride'], l['padding'], ceil_mode=True)
                else:
                    out = F.max_pool2d(x, l['kernel'], l['stride'], l['padding'], ceil_mode=True)
            elif kind == 'LRN':
                out = F.local_response_norm(x, l['size'], l['alpha'], l['beta'], 1.)
            elif kind == 'CONCAT':
                out = torch.cat(xs, dim=1)
            elif kind == 'DROPOUT':
                out = x
            elif kind == 'SOFTMAX':
                out = x.softmax(1)
            else:
                raise ValueError(kind)
            blobs[l['top']] = out
            if l['name'] == 'pool5/7x7_s1':
                pooled = out.flatten(1)
            for name in set(l['bottoms']):
                if self.last_use[name] == i and name != l['top']:
                    del blobs[name]
        assert pooled is not None and pooled.shape[1] == 1024 and out.shape[1] == 23
        return pooled, out


def views(path):
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('RGB')
    thumb = np.array(ImageOps.contain(im, (384, 384)))
    border = np.concatenate([thumb[0], thumb[-1], thumb[:, 0], thumb[:, -1]]).astype('float32')
    center = np.median(border, axis=0)
    dispersion = np.linalg.norm(border - center, axis=1)
    foreground = im
    if np.quantile(dispersion, .75) < 30:
        distance = np.linalg.norm(thumb.astype('float32') - center, axis=2)
        mask = (distance > max(25, float(np.quantile(dispersion, .95)) + 10)).astype('uint8')
        if .03 < mask.mean() < .90:
            ys, xs = np.nonzero(mask)
            # Trim only uniform borders; use quantile bounds to reject one-pixel noise.
            x0, x1 = np.quantile(xs, [.01, .99])
            y0, y1 = np.quantile(ys, [.01, .99])
            pad = .06 * max(x1 - x0, y1 - y0)
            box = (max(0, (x0 - pad) * im.width / thumb.shape[1]),
                   max(0, (y0 - pad) * im.height / thumb.shape[0]),
                   min(im.width, (x1 + pad + 1) * im.width / thumb.shape[1]),
                   min(im.height, (y1 + pad + 1) * im.height / thumb.shape[0]))
            if box[2] > box[0] + 8 and box[3] > box[1] + 8:
                foreground = im.crop(tuple(round(v) for v in box))
    w, h = foreground.size
    closeup = foreground.crop((round(w * .2), round(h * .2), round(w * .8), round(h * .8)))
    output = []
    for image in [im, foreground, closeup]:
        rgb = np.array(ImageOps.fit(image, (224, 224), method=Image.Resampling.BILINEAR)).astype('float32')
        bgr = rgb[:, :, ::-1].copy() - np.array([104., 117., 124.], dtype='float32')
        output.append(torch.from_numpy(bgr.transpose(2, 0, 1).copy()))
    return output


def extract():
    started = time.time()
    torch.set_num_threads(4)
    frame = train_frame()
    source = json.loads((CACHE / 'minc_googlenet_source.json').read_text())
    assert source['numeric_sha256'] == sha(CACHE / 'minc_googlenet_numeric.npz')
    plan = {'scope': 'new frozen image-only material representation; no new human targets',
            'model': 'MINC GoogLeNet, original Caffe deploy graph and converted numeric weights',
            'source': source, 'deploy_sha256': sha(CACHE / 'minc_deploy-googlenet.prototxt'),
            'categories': (CACHE / 'minc_categories.txt').read_text().splitlines(),
            'views': ['full_center_fit', 'uniform_border_foreground_center_fit', 'foreground_center60pct'],
            'preprocess': 'BGR float32 0..255 minus104/117/124, no ImageNet std',
            'train_rows': len(frame), 'test_rows': 0, 'held_rows': 0,
            'source_images_sha256': sha(ROOT / 'data/processed/manifest.csv'),
            'external_images_used': False, 'external_inference': False, 'source_code_sha256': sha(__file__),
            'purpose': 'adapt local assignment heads after extraction; no direct model-to-test use',
            'papers': ['https://labelmaterial.s3.amazonaws.com/release/cvpr2015-minc.pdf'],
            'weights_source_revision': '6fb707ae76df58d7c02424661822cd6dfda60c24'}
    dump(NOTES / 'json/material_texture_plan.json', plan)
    model = MincNetwork()
    with torch.inference_mode():
        feat, prob = model(torch.zeros(2, 3, 224, 224))
        assert feat.shape == (2, 1024) and torch.allclose(prob.sum(1), torch.ones(2))
        assert torch.allclose(prob[0], prob[1])
    identity = source['numeric_sha256'] + plan['deploy_sha256'] + plan['source_code_sha256']
    partial = CACHE / 'minc_train_partial.npz'
    features, probabilities, start = [], [], 0
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        if str(z['identity'].item()) == identity and np.array_equal(z['keys'], frame.parent_key.to_numpy(str)[:len(z['keys'])]):
            features = [z['features']]
            probabilities = [z['probabilities']]
            start = len(z['keys'])
            print('Resume material descriptors:', start, flush=True)
    with torch.inference_mode():
        for offset in range(start, len(frame), 4):
            rows = frame.iloc[offset:offset + 4]
            batch = []
            for row in rows.itertuples():
                assert sha(ROOT / row.source_path) == row.source_sha256
                batch.extend(views(ROOT / row.source_path))
            f, p = model(torch.stack(batch))
            features.append(f.reshape(len(rows), 3, 1024).numpy().astype('float32'))
            probabilities.append(p.reshape(len(rows), 3, 23).numpy().astype('float32'))
            end = offset + len(rows)
            if offset == start or end % 128 == 0 or end == len(frame):
                print(f'MINC material train descriptors {end}/{len(frame)} | {time.time() - started:.0f}s', flush=True)
                np.savez_compressed(partial, keys=frame.parent_key.to_numpy(str)[:end], features=np.concatenate(features),
                                    probabilities=np.concatenate(probabilities), identity=np.array(identity))
    final = CACHE / 'minc_train_features.npz'
    np.savez_compressed(final, keys=frame.parent_key.to_numpy(str), features=np.concatenate(features),
                        probabilities=np.concatenate(probabilities), identity=np.array(identity), source_sha256=frame.source_sha256.to_numpy(str))
    dump(NOTES / 'json/material_texture_extraction.json', {'status': 'passed', 'rows': len(frame), 'views_per_image': 3,
          'dimensions_per_view': 1024, 'test_rows': 0, 'held_rows': 0, 'feature_sha256': sha(final),
          'source_code_sha256': sha(__file__), 'seconds': time.time() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['extract'])
    args = parser.parse_args()
    extract()
