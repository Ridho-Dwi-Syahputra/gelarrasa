"""GELAR RASA stage 1: reproducible technical EDA and unsupervised grouping.

No language-model service, semantic class assignment, or label archive is used.
Frozen CNN features are exploratory; this is not a trained final classifier.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.feature import hog

VERSION = 'stage1_v1'
SEED = 20261105
CONFIG = dict(version=VERSION, seed=SEED, image_size=224,
              alpha_background=[255, 255, 255], pad_fill=[124, 116, 104],
              cnn='resnet50', weights='IMAGENET1K_V2',
              fit_population='unique_train_pixel_hash_only', primary_k=8,
              phash_bits=64, candidate_phash_threshold=6,
              candidate_group_threshold=4, cluster_seeds=[11, 29, 47, 83, 131])
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def frame_write(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')


def log(ctx, message):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message
    print(line, flush=True)
    with (ctx['report'] / 'run.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def setup(project_root=None):
    if project_root is None:
        candidates = [Path.cwd(), *Path.cwd().parents]
        project_root = next((p for p in candidates if (p / 'configs/data_contract.json').exists()), None)
    if project_root is None:
        raise RuntimeError('Set PROJECT_ROOT to the GELAR-RASA project directory.')
    root = Path(project_root).resolve()
    work = Path(os.environ.get('GELAR_WORK_ROOT', str(root))).resolve()
    data = Path(os.environ.get('GELAR_DATA_ROOT', str(root / 'data'))).resolve()
    ctx = dict(root=root, work=work, data=data,
               report=work / 'reports/eda/stage1_20261105',
               processed=work / 'data/processed' / VERSION,
               manifests=work / 'output/manifests' / VERSION,
               groups=work / 'output/groups' / VERSION,
               embeddings=work / 'output/embeddings' / VERSION)
    for k in ['report', 'processed', 'manifests', 'groups', 'embeddings']:
        ctx[k].mkdir(parents=True, exist_ok=True)
    (ctx['report'] / 'figures').mkdir(exist_ok=True)
    (ctx['report'] / 'galleries').mkdir(exist_ok=True)
    (ctx['processed'] / 'cluster_views').mkdir(exist_ok=True)
    config_file = ctx['processed'] / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != CONFIG:
        raise RuntimeError('Existing version has a different config. Use a new version, do not overwrite.')
    dump_json(config_file, CONFIG)
    packages = ['numpy', 'pandas', 'Pillow', 'scipy', 'scikit-learn', 'scikit-image',
                'opencv-python', 'ImageHash', 'matplotlib', 'torch', 'torchvision', 'nbformat']
    versions = {}
    for p in packages:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'not installed under this distribution name'
    dump_json(ctx['report'] / 'environment.json', dict(python=platform.python_version(),
              platform=platform.platform(), packages=versions, config=CONFIG,
              free_disk_gib=shutil.disk_usage(work).free / 2**30))
    (ctx['report'] / 'requirements_observed.txt').write_text('\n'.join(
        f'{k}=={v}' for k, v in versions.items() if 'not installed' not in v) + '\n', encoding='utf-8')
    log(ctx, 'Setup complete. No category targets are loaded.')
    return ctx


def canonical_rgb(path):
    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source)
        has_alpha = 'A' in image.getbands() or 'transparency' in image.info
        if has_alpha:
            rgba = image.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, tuple(CONFIG['alpha_background']) + (255,))
            image = Image.alpha_composite(bg, rgba).convert('RGB')
        else:
            image = image.convert('RGB')
        return image.copy()


def letterbox(image, size=224):
    scale = size / max(image.size)
    dims = tuple(max(1, min(size, round(v * scale))) for v in image.size)
    fitted = image.resize(dims, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), tuple(CONFIG['pad_fill']))
    left, top = (size - dims[0]) // 2, (size - dims[1]) // 2
    canvas.paste(fitted, (left, top))
    return canvas, dict(resized_width=dims[0], resized_height=dims[1],
                        pad_left=left, pad_top=top, scale=scale,
                        padding_fraction=1 - dims[0] * dims[1] / size**2)


def inventory(ctx):
    contract = json.loads((ctx['root'] / 'configs/data_contract.json').read_text(encoding='utf-8'))
    dataset = ctx['data'] / 'Dataset DSC GelarRasa 2026'
    sample = pd.read_csv(ctx['data'] / 'SampleSubmission_DSC.csv', dtype=str)
    if list(sample.columns) != contract['submission_columns'] or sample.image_id.duplicated().any():
        raise AssertionError('Sample schema or duplicate ID failure')
    rows, extra = [], []
    for split in ['train', 'test']:
        folder = dataset / split
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        other = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() not in EXTENSIONS]
        extra.extend(dict(split=split, filename=name) for name in other)
        assert len(files) == contract[f'{split}_count'], (split, len(files))
        assert len({p.stem for p in files}) == len(files), 'Duplicate source ID'
        if split == 'test':
            by_id = {p.stem: p for p in files}
            assert set(by_id) == set(sample.image_id), 'Test IDs differ from sample'
            files = [by_id[i] for i in sample.image_id]
        for p in files:
            assert p.stem.isdigit() and len(p.stem) == contract[f'{split}_id_width']
            rows.append(dict(key=split + ':' + p.stem, split=split, image_id=p.stem,
                             filename=p.name, source_path=str(p.relative_to(ctx['data'])),
                             byte_size=p.stat().st_size, source_sha256=sha(p)))
    df = pd.DataFrame(rows)
    digest = hashlib.sha256(df[['key', 'source_sha256']].to_csv(index=False).encode()).hexdigest()
    prior = ctx['manifests'] / 'input_fingerprint.json'
    if prior.exists():
        assert json.loads(prior.read_text())['digest'] == digest, 'Sources changed; increment version'
    dump_json(prior, dict(digest=digest, images=len(df), ignored_root_files=extra,
                         train_scan='direct children only; subdirectories excluded',
                         sample_target_values_used=False))
    frame_write(df, ctx['manifests'] / 'sources.csv')
    log(ctx, f'Inventory: {len(df)} sources, unique IDs, test order follows sample.')
    return df


def pixel_digest(image):
    return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode() + image.tobytes()).hexdigest()


def profile_one(task):
    ctx, row = task
    row = dict(row)
    path = ctx['data'] / row['source_path']
    try:
        with Image.open(path) as src:
            src.load()
            row.update(source_format=src.format, source_mode=src.mode,
                       original_width=src.width, original_height=src.height,
                       exif_orientation=int(src.getexif().get(274, 1)),
                       has_alpha=('A' in src.getbands() or 'transparency' in src.info),
                       icc_profile_present=bool(src.info.get('icc_profile')),
                       n_frames=getattr(src, 'n_frames', 1))
        image = canonical_rgb(path)
        arr = np.asarray(image)
        thumb = ImageOps.contain(image, (256, 256), Image.Resampling.LANCZOS)
        a = np.asarray(thumb)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        hist = np.bincount(gray.ravel(), minlength=256) / gray.size
        probs = hist[hist > 0]
        row.update(width=image.width, height=image.height, min_side=min(image.size),
                   aspect_ratio=image.width / image.height, megapixels=image.width * image.height / 1e6,
                   pixel_sha256=pixel_digest(image), brightness=float(gray.mean() / 255),
                   contrast=float(gray.std() / 255), saturation=float(hsv[:, :, 1].mean() / 255),
                   grayscale_entropy_bits=float(-(probs * np.log2(probs)).sum()),
                   dark_fraction=float((gray <= 8).mean()), bright_fraction=float((gray >= 247).mean()),
                   laplacian_variance_256=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                   edge_fraction=float((cv2.Canny(gray, 100, 200) > 0).mean()),
                   effective_grayscale=bool(np.max(np.ptp(a.astype(np.int16), axis=2)) <= 2),
                   bytes_per_pixel=row['byte_size'] / (image.width * image.height),
                   decode_status='ok', decode_error='')
        for i, name in enumerate(['r', 'g', 'b']):
            row[f'mean_{name}'] = float(a[:, :, i].mean() / 255)
            row[f'std_{name}'] = float(a[:, :, i].std() / 255)
        row['phash'] = str(imagehash.phash(image, hash_size=8))
        row['phash_hflip'] = str(imagehash.phash(ImageOps.mirror(image), hash_size=8))
        row['phash_vflip'] = str(imagehash.phash(ImageOps.flip(image), hash_size=8))
        row['dhash'] = str(imagehash.dhash(image, hash_size=8))
        row['whash'] = str(imagehash.whash(image, hash_size=8))
        output = ctx['processed'] / 'rgb_native' / row['split'] / (row['image_id'] + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format='PNG', compress_level=3)
        with Image.open(output) as check:
            check.load()
            assert check.mode == 'RGB' and np.array_equal(arr, np.asarray(check))
        boxed, geometry = letterbox(image)
        preview = ctx['processed'] / 'letterbox224' / row['split'] / (row['image_id'] + '.png')
        preview.parent.mkdir(parents=True, exist_ok=True)
        boxed.save(preview, format='PNG', compress_level=3)
        row.update(geometry)
        row.update(processed_path=str(output.relative_to(ctx['work'])),
                   processed_sha256=sha(output), preview_path=str(preview.relative_to(ctx['work'])),
                   preview_sha256=sha(preview), rgb_roundtrip_exact=True)
        small = np.asarray(image.resize((128, 128), Image.Resampling.LANCZOS))
        hsv_small = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        color = np.concatenate([np.histogram(hsv_small[:, :, i], bins=b, range=(0, high))[0]
                                for i, b, high in [(0, 16, 180), (1, 8, 256), (2, 8, 256)]]).astype('float32')
        color /= 128 * 128
        shape = hog(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), orientations=9,
                    pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys').astype('float32')
        return row, color, shape
    except Exception as exc:
        row.update(decode_status='error', decode_error=f'{type(exc).__name__}: {exc}')
        return row, None, None


def profile_and_preprocess(ctx, sources):
    cv2.setNumThreads(1)
    rows, color, shape = [], [], []
    tasks = [(ctx, row) for row in sources.to_dict('records')]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (row, c, h) in enumerate(pool.map(profile_one, tasks)):
            rows.append(row)
            color.append(c)
            shape.append(h)
            if (i + 1) % 200 == 0:
                log(ctx, f'Decode/normalize/hash: {i + 1}/{len(tasks)}')
    df = pd.DataFrame(rows)
    frame_write(df, ctx['manifests'] / 'image_profile.csv')
    errors = df[df.decode_status != 'ok']
    frame_write(errors, ctx['report'] / 'decode_errors.csv')
    if len(errors):
        raise RuntimeError(f'{len(errors)} images failed. See decode_errors.csv; no silent exclusion.')
    np.savez_compressed(ctx['embeddings'] / 'classical_features.npz',
                        keys=df.key.to_numpy(dtype=str), color=np.stack(color), hog=np.stack(shape))
    frame_write(df[['key', 'split', 'image_id', 'source_path', 'source_sha256', 'pixel_sha256',
                    'processed_path', 'processed_sha256', 'preview_path', 'preview_sha256',
                    'width', 'height', 'scale', 'pad_left', 'pad_top', 'padding_fraction',
                    'rgb_roundtrip_exact']], ctx['processed'] / 'preprocessing_manifest.csv')
    log(ctx, 'All images decoded and canonical RGB PNG pixels verified. No files excluded.')
    return df


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def hash_array(values):
    return np.array([list(bytes.fromhex(v)) for v in values], dtype=np.uint8)


def hamming_matrix(left, right):
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    out = np.empty((len(left), len(right)), dtype=np.uint8)
    for start in range(0, len(left), 128):
        out[start:start + 128] = lookup[np.bitwise_xor(left[start:start + 128, None, :], right[None, :, :])].sum(2)
    return out


def pair_table(df, pairs, kind, distances=None):
    records = []
    for i, j in pairs:
        r = dict(key_a=df.iloc[i].key, key_b=df.iloc[j].key,
                 split_pair='-'.join(sorted([df.iloc[i].split, df.iloc[j].split])), kind=kind)
        if distances is not None:
            r['hamming'] = int(distances[i, j])
        records.append(r)
    cols = ['key_a', 'key_b', 'split_pair', 'kind'] + (['hamming'] if distances is not None else [])
    return pd.DataFrame(records, columns=cols)


def audit_duplicates(ctx, df):
    n = len(df)
    exact_uf = UnionFind(n)
    for col in ['source_sha256', 'pixel_sha256']:
        pairs = []
        for indices in df.groupby(col, sort=False).indices.values():
            for i, j in itertools.combinations(indices, 2):
                pairs.append((i, j))
                exact_uf.union(int(i), int(j))
        frame_write(pair_table(df, pairs, col), ctx['groups'] / f'{col}_pairs.csv')
    ph = hash_array(df.phash)
    direct = hamming_matrix(ph, ph)
    flip = np.minimum(hamming_matrix(ph, hash_array(df.phash_hflip)),
                      hamming_matrix(ph, hash_array(df.phash_vflip)))
    minimum = np.minimum(direct, flip)
    # Resampling/threshold quantization can make transformed-hash comparison
    # direction-dependent. An undirected candidate edge uses either orientation.
    minimum = np.minimum(minimum, minimum.T)
    assert np.array_equal(minimum, minimum.T) and not np.diag(minimum).any()
    pairs = list(zip(*np.where(np.triu(minimum <= CONFIG['candidate_phash_threshold'], k=1))))
    candidate = pair_table(df, pairs, 'unconfirmed_perceptual_similarity', minimum)
    if len(candidate):
        candidate['direct_hamming'] = [int(direct[i, j]) for i, j in pairs]
        candidate['flip_improves'] = [bool(minimum[i, j] < direct[i, j]) for i, j in pairs]
        candidate['confirmed_pixel_identical'] = [bool(df.iloc[i].pixel_sha256 == df.iloc[j].pixel_sha256) for i, j in pairs]
    frame_write(candidate, ctx['groups'] / 'phash_candidate_pairs.csv')
    candidate_uf = UnionFind(n)
    for i in range(n):
        candidate_uf.union(i, exact_uf.find(i))
    for i, j in pairs:
        if minimum[i, j] <= CONFIG['candidate_group_threshold']:
            candidate_uf.union(int(i), int(j))
    groups = df[['key', 'split', 'image_id']].copy()
    groups['exact_group'] = [f'exact_{exact_uf.find(i):04d}' for i in range(n)]
    groups['candidate_group'] = [f'candidate_{candidate_uf.find(i):04d}' for i in range(n)]
    frame_write(groups, ctx['groups'] / 'groups.csv')
    group_rows = []
    for group, indices in groups.groupby('candidate_group', sort=False).indices.items():
        if len(indices) < 2:
            continue
        sub = minimum[np.ix_(indices, indices)]
        diameter = int(sub.max())
        group_rows.append(dict(group=group, n=len(indices), phash_diameter=diameter,
                               chaining_risk=diameter > CONFIG['candidate_group_threshold'],
                               keys='|'.join(df.iloc[indices].key)))
    frame_write(pd.DataFrame(group_rows, columns=['group', 'n', 'phash_diameter', 'chaining_risk', 'keys']),
                ctx['groups'] / 'candidate_group_audit.csv')
    train = np.flatnonzero(df.split.eq('train'))
    test = np.flatnonzero(df.split.eq('test'))
    rec = []
    for i in test:
        order = np.argsort(minimum[i, train], kind='stable')[:10]
        for rank, local in enumerate(order, 1):
            j = train[local]
            rec.append(dict(test_id=df.iloc[i].image_id, train_id=df.iloc[j].image_id,
                            rank=rank, phash_distance=int(minimum[i, j]),
                            direct_phash_distance=int(direct[i, j]),
                            byte_identical=df.iloc[i].source_sha256 == df.iloc[j].source_sha256,
                            pixel_identical=df.iloc[i].pixel_sha256 == df.iloc[j].pixel_sha256))
    frame_write(pd.DataFrame(rec), ctx['report'] / 'test_train_phash_top10.csv')
    sensitivity = []
    for t in [0, 2, 4, 6, 8, 10, 12, 16]:
        for a, b, ia, ib in [('train', 'train', train, train), ('test', 'test', test, test), ('test', 'train', test, train)]:
            dist = minimum[np.ix_(ia, ib)]
            count = int(np.triu(dist <= t, k=1).sum()) if a == b else int((dist <= t).sum())
            sensitivity.append(dict(split_pair=f'{a}-{b}', threshold=t, pair_count=count))
    frame_write(pd.DataFrame(sensitivity), ctx['report'] / 'phash_threshold_sensitivity.csv')
    log(ctx, f'Hash audit: {len(candidate)} candidate pairs; similarity is not duplicate proof or category proof.')
    return groups


def distribution_audit(ctx, df):
    metrics = ['width', 'height', 'min_side', 'aspect_ratio', 'byte_size', 'megapixels',
               'brightness', 'contrast', 'saturation', 'grayscale_entropy_bits', 'dark_fraction',
               'bright_fraction', 'laplacian_variance_256', 'edge_fraction', 'padding_fraction']
    records = []
    for col in metrics:
        a = df.loc[df.split.eq('train'), col].to_numpy(float)
        b = df.loc[df.split.eq('test'), col].to_numpy(float)
        bins = np.unique(np.quantile(a, np.linspace(0, 1, 21)))
        bins = np.r_[-np.inf, bins[1:-1], np.inf]
        if len(bins) < 2:
            bins = np.array([-np.inf, np.inf])
        ha, hb = np.histogram(a, bins)[0].astype(float), np.histogram(b, bins)[0].astype(float)
        ha, hb = ha / ha.sum(), hb / hb.sum()
        ks = ks_2samp(a, b)
        records.append(dict(feature=col, train_mean=a.mean(), test_mean=b.mean(),
                            train_median=np.median(a), test_median=np.median(b),
                            train_p01=np.quantile(a, .01), train_p99=np.quantile(a, .99),
                            test_p01=np.quantile(b, .01), test_p99=np.quantile(b, .99),
                            ks_statistic=float(ks.statistic), p_value=float(ks.pvalue),
                            js_distance_base2=float(jensenshannon(ha, hb, base=2)),
                            wasserstein=float(wasserstein_distance(a, b))))
    result = pd.DataFrame(records)
    order = np.argsort(result.p_value.to_numpy())
    raw_q = result.p_value.to_numpy()[order] * len(result) / np.arange(1, len(result) + 1)
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1].clip(0, 1)
    result.loc[order, 'bh_q_value'] = adjusted
    result = result.sort_values('ks_statistic', ascending=False)
    frame_write(result, ctx['report'] / 'train_test_distribution.csv')
    categorical = []
    for feature in ['source_format', 'source_mode', 'exif_orientation', 'has_alpha', 'effective_grayscale']:
        for (split, val), g in df.groupby(['split', feature], dropna=False):
            categorical.append(dict(feature=feature, split=split, value=str(val), n=len(g),
                                     fraction=len(g) / int(df.split.eq(split).sum())))
    frame_write(pd.DataFrame(categorical), ctx['report'] / 'categorical_profile.csv')
    flags = []
    for feature in ['brightness', 'contrast', 'laplacian_variance_256', 'grayscale_entropy_bits', 'aspect_ratio']:
        lo, hi = df.loc[df.split.eq('train'), feature].quantile([.01, .99])
        for r in df.loc[(df[feature] < lo) | (df[feature] > hi)].to_dict('records'):
            flags.append(dict(key=r['key'], feature=feature, value=r[feature], train_p01=lo,
                              train_p99=hi, reason='outside_train_1_to_99_percentiles', action='review_only'))
    for r in df.loc[df.min_side < 224].to_dict('records'):
        flags.append(dict(key=r['key'], feature='min_side', value=r['min_side'],
                          train_p01=np.nan, train_p99=np.nan, reason='native_side_below_224', action='review_only'))
    frame_write(pd.DataFrame(flags), ctx['report'] / 'technical_review_queue.csv')
    log(ctx, 'Distribution audit complete: effect sizes, BH-adjusted p-values, no automatic exclusions.')
    return result


def extract_features(ctx, df):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.transforms.functional import to_tensor, normalize

    torch.manual_seed(SEED)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    weight_file = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-11ad3fa6.pth'
    blocks = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(df), 16):
            tensors = []
            for r in df.iloc[start:start + 16].itertuples():
                with Image.open(ctx['work'] / r.preview_path) as im:
                    tensors.append(normalize(to_tensor(im.convert('RGB')), [0.485, .456, .406], [.229, .224, .225]))
            x = torch.stack(tensors).to(device)
            blocks.append(model(x).cpu().numpy().astype('float32'))
            if (start // 16 + 1) % 10 == 0:
                log(ctx, f'CNN features {min(start+16, len(df))}/{len(df)}, elapsed {time.time()-started:.0f}s')
    features = np.vstack(blocks)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    assert features.shape == (len(df), 2048) and np.isfinite(features).all()
    np.savez_compressed(ctx['embeddings'] / 'resnet50_features.npz', keys=df.key.to_numpy(dtype=str), features=features)
    dump_json(ctx['embeddings'] / 'feature_provenance.json', dict(
        architecture='torchvision.resnet50, fc=Identity', weights='IMAGENET1K_V2',
        weight_sha256=sha(weight_file), weight_url=ResNet50_Weights.IMAGENET1K_V2.url,
        device=str(device), seconds=time.time()-started, dimensions=2048,
        transform='EXIF -> RGB/white alpha -> full-frame letterbox224 -> ImageNet normalization',
        pretrained_default_crop_used=False, competition_finetuning_executed=False,
        purpose='exploratory retrieval and unsupervised clustering, not final category prediction',
        input_fingerprint=json.loads((ctx['manifests'] / 'input_fingerprint.json').read_text())['digest']))
    log(ctx, 'CNN features saved; backbone frozen, no semantic category predictions produced.')
    return features


def fit_clusters(ctx, df, cnn, groups):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
    from sklearn.preprocessing import normalize
    from scipy.optimize import linear_sum_assignment
    from threadpoolctl import threadpool_limits
    import joblib

    train = np.flatnonzero(df.split.eq('train'))
    unique_train = df.loc[df.split.eq('train')].drop_duplicates('pixel_sha256').index.to_numpy()
    fit_idx = unique_train
    classic = np.load(ctx['embeddings'] / 'classical_features.npz')
    c = normalize(classic['color']).astype('float32')
    h = normalize(classic['hog']).astype('float32')
    classic_x = normalize(np.hstack([c, h])).astype('float32')
    results, assignments = [], {}
    with threadpool_limits(limits=4):
        for family, raw in [('cnn', cnn), ('classic', classic_x)]:
            pca = PCA(n_components=64, svd_solver='randomized', random_state=SEED)
            pca.fit(raw[fit_idx])
            x = normalize(pca.transform(raw)).astype('float32')
            dump_json(ctx['embeddings'] / f'{family}_pca_provenance.json',
                      dict(components=64, train_unique_count=len(fit_idx),
                           explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
                           test_used_in_fit=False, unit_normalization_after_pca=True))
            joblib.dump(pca, ctx['embeddings'] / f'{family}_pca.joblib')
            n_seeds = CONFIG['cluster_seeds'] if family == 'cnn' else CONFIG['cluster_seeds'][:2]
            for seed in n_seeds:
                km = KMeans(n_clusters=8, n_init=10, random_state=seed, max_iter=300)
                km.fit(x[fit_idx])
                labels = km.predict(x)
                tag = f'{family}_k8_seed{seed}'
                assignments[tag] = labels
                results.append(dict(method=tag, k=8, family=family, seed=seed,
                    silhouette=float(silhouette_score(x[fit_idx], labels[fit_idx], metric='euclidean')),
                    calinski_harabasz=float(calinski_harabasz_score(x[fit_idx], labels[fit_idx])),
                    davies_bouldin=float(davies_bouldin_score(x[fit_idx], labels[fit_idx])),
                    smallest_cluster=int(np.bincount(labels[fit_idx], minlength=8).min()),
                    largest_cluster=int(np.bincount(labels[fit_idx], minlength=8).max()),
                    pca_explained_variance=float(pca.explained_variance_ratio_.sum())))
                if family == 'cnn' and seed == n_seeds[0]:
                    primary, primary_x, primary_model = labels.copy(), x.copy(), km
                    joblib.dump(km, ctx['embeddings'] / 'primary_kmeans.joblib')
            if family == 'cnn':
                # Count-sensitivity is descriptive; it does not select semantic categories.
                for k in [6, 10, 12, 16]:
                    km = KMeans(n_clusters=k, n_init=10, random_state=11, max_iter=300).fit(x[fit_idx])
                    lab = km.predict(x)
                    results.append(dict(method=f'cnn_k{k}_diagnostic', k=k, family=family, seed=11,
                        silhouette=float(silhouette_score(x[fit_idx], lab[fit_idx])),
                        calinski_harabasz=float(calinski_harabasz_score(x[fit_idx], lab[fit_idx])),
                        davies_bouldin=float(davies_bouldin_score(x[fit_idx], lab[fit_idx])),
                        smallest_cluster=int(np.bincount(lab[fit_idx], minlength=k).min()),
                        largest_cluster=int(np.bincount(lab[fit_idx], minlength=k).max())))
                ag = AgglomerativeClustering(n_clusters=8, linkage='ward').fit(x[fit_idx])
                centroids = np.stack([x[fit_idx][ag.labels_ == k].mean(0) for k in range(8)])
                agg_labels = np.argmin(((x[:, None, :] - centroids[None, :, :])**2).sum(2), axis=1)
                agg_labels[fit_idx] = ag.labels_
                assignments['cnn_ward8_nearest_centroid_for_test'] = agg_labels
                # Restore duplicate train to their source representative, not a different assignment.
                by_hash = dict(zip(df.iloc[fit_idx].pixel_sha256, ag.labels_))
                for i in train:
                    agg_labels[i] = by_hash[df.iloc[i].pixel_sha256]
    assignments_df = df[['key', 'split', 'image_id']].copy()
    for tag, values in assignments.items():
        assignments_df[tag] = values
    frame_write(assignments_df, ctx['groups'] / 'all_cluster_assignments.csv')
    frame_write(pd.DataFrame(results), ctx['report'] / 'cluster_diagnostics.csv')
    stability = []
    for a, b in itertools.combinations(assignments, 2):
        stability.append(dict(method_a=a, method_b=b,
                              ari_train=float(adjusted_rand_score(assignments[a][fit_idx], assignments[b][fit_idx]))))
    frame_write(pd.DataFrame(stability), ctx['report'] / 'cluster_stability.csv')
    aligned = []
    for tag, lab in assignments.items():
        if not tag.startswith('cnn_k8_'):
            continue
        table = np.zeros((8, 8), dtype=int)
        np.add.at(table, (primary[fit_idx], lab[fit_idx]), 1)
        left, right = linear_sum_assignment(-table)
        mapping = dict(zip(right, left))
        aligned.append(np.array([mapping[v] for v in lab]))
    agreement = (np.stack(aligned) == primary).mean(0)
    dists = primary_model.transform(primary_x)
    sorted_dist = np.sort(dists, axis=1)
    margin = (sorted_dist[:, 1] - sorted_dist[:, 0]) / np.maximum(sorted_dist[:, 1], 1e-12)
    assigned_dist = dists[np.arange(len(df)), primary]
    q95 = {k: float(np.quantile(assigned_dist[fit_idx][primary[fit_idx] == k], .95)) for k in range(8)}
    queue = df[['key', 'split', 'image_id', 'processed_path', 'preview_path']].copy()
    queue['cluster_id'] = [f'cluster_{v:02d}' for v in primary]
    queue['cluster_index'] = primary
    queue['centroid_distance'] = assigned_dist
    queue['relative_margin'] = margin
    queue['seed_agreement'] = agreement
    queue['outlying_distance'] = [bool(d > q95[k]) for d, k in zip(assigned_dist, primary)]
    queue['review_reason'] = [('|'.join(r for flag, r in [(a < .8, 'seed_unstable'),
                                      (m < .10, 'small_centroid_margin'), (o, 'distance_above_train_cluster_p95')] if flag))
                              for a, m, o in zip(agreement, margin, queue.outlying_distance)]
    queue['semantic_category'] = ''
    frame_write(queue, ctx['groups'] / 'cluster_assignments.csv')
    frame_write(queue[queue.review_reason.ne('')], ctx['report'] / 'cluster_review_queue.csv')
    mapping_path = ctx['processed'] / 'cluster_category_mapping_TEMPLATE.csv'
    if not mapping_path.exists():
        frame_write(pd.DataFrame(dict(cluster_id=[f'cluster_{k:02d}' for k in range(8)],
                      waste_category=['']*8, mapped_by_team_member=['']*8, evidence=['']*8,
                      status=['pending_team_mapping']*8)), mapping_path)
    # Physical directory views contain the real preprocessed images, using links when possible.
    copied, linked = 0, 0
    # Reruns may change memberships. Remove only verified generated stale links/copies;
    # never touch native images or an unknown/edited file.
    view_root = (ctx['processed'] / 'cluster_views').resolve()
    intended = {(r.split, r.image_id): r for r in queue.itertuples()}
    stale_views = []
    for old in view_root.glob('*/*/*.png'):
        split = old.parent.parent.name
        row = intended.get((split, old.stem))
        if row is None:
            raise AssertionError(f'Unknown file in generated cluster views: {old}')
        target = view_root / split / row.cluster_id / (row.image_id + '.png')
        if old != target:
            assert view_root in old.resolve().parents
            assert sha(old) == sha(ctx['work'] / row.processed_path), str(old)
            stale_views.append(str(old.relative_to(view_root)))
            old.unlink()
    for split in ['train', 'test']:
        for k in range(8):
            (ctx['processed'] / 'cluster_views' / split / f'cluster_{k:02d}').mkdir(parents=True, exist_ok=True)
    for r in queue.itertuples():
        target = ctx['processed'] / 'cluster_views' / r.split / r.cluster_id / (r.image_id + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ctx['work'] / r.processed_path
        if target.exists():
            if sha(target) != sha(source):
                raise AssertionError(f'Unexpected existing cluster view: {target}')
            continue
        try:
            os.link(source, target)
            linked += 1
        except OSError:
            shutil.copy2(source, target)
            copied += 1
    for split in ['train', 'test']:
        actual = {p.stem for p in (ctx['processed'] / 'cluster_views' / split).glob('*/*.png')}
        expected = set(df.loc[df.split.eq(split), 'image_id'])
        assert actual == expected
        assert len(list((ctx['processed'] / 'cluster_views' / split).glob('*/*.png'))) == len(expected)
    dump_json(ctx['groups'] / 'cluster_provenance.json', dict(
        primary='cnn_k8_seed11; fixed before comparing any archive',
        fit_keys=df.iloc[fit_idx].key.tolist(), test_used_in_fit=False,
        duplicate_train_policy='one canonical-pixel representative when fitting PCA and clusters',
        cluster_assignment_sha256=sha(ctx['groups'] / 'cluster_assignments.csv'),
        semantic_names_assigned=False, seed_agreement_is_probability=False,
        grouping_copies=copied, grouping_hardlinks=linked,
        verified_stale_views_removed=stale_views,
        warning='Hardlinked processed views share file content. Treat all processed outputs as immutable.'))
    # Display geometry is for navigation, not evidence that classes separate in two dimensions.
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2, random_state=SEED).fit(cnn[fit_idx])
    xy = p2.transform(cnn)
    projection = df[['key', 'split']].copy()
    projection['pc1'], projection['pc2'], projection['cluster_id'] = xy[:, 0], xy[:, 1], queue.cluster_id
    frame_write(projection, ctx['report'] / 'pca_navigation.csv')
    log(ctx, f'Clusters saved: 8 anonymous groups, {len(fit_idx)} unique train fit samples; no class names inferred.')
    return queue


def embedding_retrieval(ctx, df, cnn, assignments):
    train = np.flatnonzero(df.split.eq('train'))
    test = np.flatnonzero(df.split.eq('test'))
    similarity = cnn @ cnn[train].T
    rows, coverage = [], []
    for i in test:
        nearest = np.argsort(-similarity[i], kind='stable')[:10]
        for rank, jlocal in enumerate(nearest, 1):
            j = train[jlocal]
            rows.append(dict(test_id=df.iloc[i].image_id, train_id=df.iloc[j].image_id,
                             rank=rank, cosine_similarity=float(similarity[i, jlocal]),
                             test_cluster=assignments.iloc[i].cluster_id,
                             train_cluster=assignments.iloc[j].cluster_id,
                             pixel_identical=df.iloc[i].pixel_sha256 == df.iloc[j].pixel_sha256))
        coverage.append(dict(key=df.iloc[i].key, nearest_similarity=float(similarity[i, nearest[0]]),
                             nearest_train_id=df.iloc[train[nearest[0]]].image_id))
    frame_write(pd.DataFrame(rows), ctx['report'] / 'test_train_embedding_top10.csv')
    frame_write(pd.DataFrame(coverage).sort_values('nearest_similarity'), ctx['report'] / 'test_coverage_review.csv')
    # Train retrieval excludes all exact-pixel duplicates, not merely the diagonal.
    within = []
    train_hashes = df.iloc[train].pixel_sha256.to_numpy()
    for i in train:
        sim = similarity[i].copy()
        sim[train_hashes == df.iloc[i].pixel_sha256] = -np.inf
        for rank, jlocal in enumerate(np.argsort(-sim, kind='stable')[:5], 1):
            within.append(dict(image_id=df.iloc[i].image_id, neighbor_id=df.iloc[train[jlocal]].image_id,
                               rank=rank, cosine_similarity=float(sim[jlocal])))
    frame_write(pd.DataFrame(within), ctx['report'] / 'train_train_embedding_top5.csv')
    log(ctx, 'Every test image has 10 pHash and 10 CNN-neighbor candidates. No target propagation.')


def validation_diagnostics(ctx, df, cnn, clusters, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits
    cols = ['width', 'height', 'aspect_ratio', 'byte_size', 'brightness', 'contrast',
            'saturation', 'grayscale_entropy_bits', 'dark_fraction', 'bright_fraction',
            'laplacian_variance_256', 'edge_fraction', 'padding_fraction']
    numeric = df[cols].to_numpy(float)
    numeric[:, [0, 1, 3, 10]] = np.log1p(numeric[:, [0, 1, 3, 10]])
    y = df.split.eq('test').to_numpy(int)
    group_values = groups.candidate_group.to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(numeric, y, group_values))
    results = []
    predictions = df[['key', 'split']].copy()
    predictions['domain_target'] = y
    with threadpool_limits(limits=4):
        for name, x in [('metadata', numeric), ('cnn_fixed_features', cnn)]:
            out = np.full(len(df), np.nan)
            fold_ids = np.full(len(df), -1)
            for fold, (tr, va) in enumerate(folds):
                assert not set(group_values[tr]) & set(group_values[va])
                estimator = make_pipeline(StandardScaler(), LogisticRegression(
                    C=.1, class_weight='balanced', max_iter=1500, random_state=SEED))
                estimator.fit(x[tr], y[tr])
                out[va] = estimator.predict_proba(x[va])[:, 1]
                fold_ids[va] = fold
                results.append(dict(feature_set=name, fold=fold,
                                    roc_auc=float(roc_auc_score(y[va], out[va])), n_validation=len(va)))
            assert np.isfinite(out).all()
            results.append(dict(feature_set=name, fold='pooled_oof',
                                roc_auc=float(roc_auc_score(y, out)), n_validation=len(y)))
            predictions[name + '_oof_domain_probability'] = out
            predictions['fold'] = fold_ids
    frame_write(pd.DataFrame(results), ctx['report'] / 'domain_validation.csv')
    frame_write(predictions, ctx['report'] / 'domain_oof.csv')
    # These folds are a draft only: globally fitted unsupervised targets are not independent truth.
    ti = np.flatnonzero(df.split.eq('train'))
    draft = df.iloc[ti][['key', 'image_id']].copy()
    draft['candidate_group'] = group_values[ti]
    draft['cluster_id'] = clusters.iloc[ti].cluster_id.to_numpy()
    draft['fold'] = -1
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(split.split(ti, draft.cluster_id, draft.candidate_group)):
        assert not set(draft.iloc[tr].candidate_group) & set(draft.iloc[va].candidate_group)
        draft.iloc[va, draft.columns.get_loc('fold')] = fold
    assert draft.fold.ge(0).all()
    frame_write(draft, ctx['work'] / 'output/folds' / VERSION / 'DRAFT_group_folds.csv')
    dump_json(ctx['work'] / 'output/folds' / VERSION / 'README.json', dict(
        status='draft_for_review_not_independent_semantic_validation',
        grouping='conservative candidate pHash components, may overgroup; review chaining',
        stratification='anonymous cluster IDs only; semantic balance unknown',
        limitation='Clusters fit on all unique train. Predicting these targets is agreement, not ground-truth accuracy.',
        next_step='For honest classifier evaluation, fit learned transforms/targets inside training folds or use an authorized independent evaluation source.'))
    log(ctx, '5-fold grouped domain diagnostics completed; draft validation folds have zero candidate-group overlap.')


def finalize(ctx, df, clusters, groups):
    # Verify source immutability again after all preprocessing.
    source_checks = [sha(ctx['data'] / r.source_path) == r.source_sha256 for r in df.itertuples()]
    assert all(source_checks), 'Raw bytes changed during processing'
    assert df.key.is_unique and len(clusters) == len(df) and clusters.key.tolist() == df.key.tolist()
    assert clusters.semantic_category.fillna('').eq('').all()
    grouping = groups.merge(df[['key', 'pixel_sha256']], on='key', validate='one_to_one')
    assert grouping.groupby('pixel_sha256').exact_group.nunique().max() == 1
    summary = dict(total=len(df), train=int(df.split.eq('train').sum()), test=int(df.split.eq('test').sum()),
                   decode_errors=int(df.decode_status.ne('ok').sum()),
                   raw_hashes_rechecked=len(source_checks), rgb_roundtrip_verified=int(df.rgb_roundtrip_exact.sum()),
                   pixel_unique_train=int(df[df.split.eq('train')].pixel_sha256.nunique()),
                   semantic_labels_ready=False, classifier_trained=False, category_accuracy_measured=False,
                   cluster_review_count=int(clusters.review_reason.ne('').sum()),
                   train_review_count=int((clusters.review_reason.ne('') & clusters.split.eq('train')).sum()),
                   test_review_count=int((clusters.review_reason.ne('') & clusters.split.eq('test')).sum()),
                   claim='Executed EDA and unsupervised clustering; no semantic ground truth evaluation')
    dump_json(ctx['report'] / 'summary.json', summary)
    paths = [p for folder in ['manifests', 'groups', 'embeddings'] for p in ctx[folder].glob('*') if p.is_file()]
    dump_json(ctx['report'] / 'artifact_hashes.json', [dict(path=str(p.relative_to(ctx['work'])), sha256=sha(p)) for p in paths])
    log(ctx, 'Stage 1 integrity checks passed; semantic mapping remains for the team.')
    return summary


def run_all(project_root=None):
    ctx = setup(project_root)
    src = inventory(ctx)
    df = profile_and_preprocess(ctx, src)
    groups = audit_duplicates(ctx, df)
    distribution_audit(ctx, df)
    features = extract_features(ctx, df)
    assignments = fit_clusters(ctx, df, features, groups)
    embedding_retrieval(ctx, df, features, assignments)
    validation_diagnostics(ctx, df, features, assignments, groups)
    return ctx, df, assignments, groups, finalize(ctx, df, assignments, groups)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run_all(args.root)
