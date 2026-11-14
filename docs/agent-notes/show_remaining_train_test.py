"""Read-only montage and train/test coverage diagnostics for frozen differences.

No model fit, category updates, or test annotations. Similarity is not accuracy.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'
HOME = NOTES / 'remaining_visual_review'
OLD = ROOT / 'arsip/sebelum_rapi_20261107/output/embeddings/stage2_repr_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(relative):
    with Image.open(ROOT / relative) as source:
        return ImageOps.exif_transpose(source).convert('RGB').copy()


@lru_cache(maxsize=700)
def pixel_canvas(relative):
    image = read_image(relative)
    factor = 224 / max(image.size)
    size = tuple(max(1, min(224, round(v * factor))) for v in image.size)
    canvas = Image.new('RGB', (224, 224), (124, 116, 104))
    canvas.paste(image.resize(size, Image.Resampling.LANCZOS), ((224-size[0])//2, (224-size[1])//2))
    return np.asarray(canvas)


def phash_distance(train, test):
    return min((int(train[a], 16) ^ int(test[b], 16)).bit_count()
               for a in ['phash', 'phash_hflip', 'phash_vflip']
               for b in ['phash', 'phash_hflip', 'phash_vflip'])


def save_sheet(rows, number, total):
    columns = 4
    height = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(height, columns, figsize=(14.4, height * 3.6), squeeze=False, dpi=110)
    for ax in axes.flat: ax.axis('off')
    for ax, row in zip(axes.flat, rows):
        ax.imshow(read_image(row['source_path']))
        ax.set_title(f"train {row['parent_id']}\nprogram: {row['assigned_class']} | arsip: {row['archive_class']}", fontsize=10)
    fig.suptitle(f"150 perbedaan terhadap arsip — lembar {number}/{total}\nBukan 150 kesalahan terverifikasi; urut nomor menaik", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    path = HOME / f'remaining_{number:02d}.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    return path


def pair_sheet(frame):
    fig, axes = plt.subplots(4, 4, figsize=(14.4, 14.8), dpi=110)
    for ax in axes.flat: ax.axis('off')
    for i, row in enumerate(frame.to_dict('records')):
        a, b = axes.flat[2*i], axes.flat[2*i+1]
        a.imshow(read_image(row['train_path']))
        a.set_title(f"train {row['train_id']} | program: {row['assigned_class']}\narsip pembanding: {row['archive_class']}", fontsize=10)
        b.imshow(read_image(row['test_path']))
        b.set_title(f"test {row['test_id']} | cosine {row['cosine']:.3f}\npHash {row['phash_min_64bit']} | SSIM {row['ssim_letterbox224']:.3f}", fontsize=10)
    fig.suptitle('Contoh pasangan train–test: hasil pencarian visual\nTest ditampilkan tanpa kategori; kedekatan bukan bukti label', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    path = HOME / 'train_test_examples.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    return path


def main():
    started = time.time()
    HOME.mkdir(parents=True, exist_ok=True)
    protected = {name: digest(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    differences_path = NOTES / 'table-csv/remaining_program_archive_differences.csv'
    differences = pd.read_csv(differences_path, dtype={'parent_id': str}).sort_values('parent_id')
    assert len(differences) == 150 and differences.parent_id.is_unique
    profile = pd.read_csv(NOTES / 'image_profile.csv', dtype={'image_id': str, 'phash': str, 'phash_hflip': str, 'phash_vflip': str, 'dhash': str})
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    full_keys = manifest.loc[manifest.split.eq('train') & ~manifest.is_crop, 'parent_key'].tolist()
    full = profile.set_index('key').loc[full_keys].reset_index()
    test = profile[profile.split.eq('test')].sort_values('image_id').reset_index(drop=True)
    queries = differences.merge(profile, left_on='parent_id', right_on='image_id', suffixes=('', '_profile'))
    queries = queries[queries.split.eq('train')].sort_values('parent_id').reset_index(drop=True)
    assert len(full) == 1906 and len(test) == 480 and len(queries) == 150
    assert queries.source_path.eq(queries.source_path_profile).all()
    feature_path = OLD / 'dinov2_features.npz'
    validation = json.loads((OLD / 'feature_validation.json').read_text())
    assert digest(feature_path) == validation['feature_files']['dinov2_features.npz']
    z = np.load(feature_path, allow_pickle=False)
    features = z['features'].astype('float32')
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    index = {key: i for i, key in enumerate(z['keys'])}
    train_features = features[[index[k] for k in full.key]]
    test_features = features[[index[k] for k in test.key]]
    query_features = features[[index[k] for k in queries.key]]
    with threadpool_limits(limits=4):
        similarity = query_features @ test_features.T
        coverage = test_features @ train_features.T
    remaining_mask = full.image_id.isin(differences.parent_id).to_numpy()
    nearest_all = coverage.argmax(1)
    best_all = coverage.max(1)
    best_without = coverage[:, ~remaining_mask].max(1)
    gap = best_all - best_without
    top_from_remaining = remaining_mask[nearest_all]
    coverage_table = pd.DataFrame({
        'test_id': test.image_id, 'nearest_train_id': full.image_id.to_numpy()[nearest_all],
        'nearest_is_in_150': top_from_remaining, 'cosine_with_all_1906': best_all,
        'cosine_without_150': best_without, 'cosine_gap': gap,
        'diagnostic_only': True,
    })
    coverage_table.to_csv(NOTES / 'table-csv/remaining_test_feature_coverage.csv', index=False)
    all_training_nearest_test = coverage.max(0)
    rows, phash_rows = [], []
    query_records, test_records = queries.to_dict('records'), test.to_dict('records')
    for qi, query in enumerate(query_records):
        distances = np.array([phash_distance(query, t) for t in test_records])
        order = np.argsort(-similarity[qi], kind='stable')[:3]
        top_for_train = int((top_from_remaining & (full.image_id.to_numpy()[nearest_all] == query['parent_id'])).sum())
        local_gap = gap[top_from_remaining & (full.image_id.to_numpy()[nearest_all] == query['parent_id'])]
        for rank, ti in enumerate(order, 1):
            neighbor = test_records[ti]
            rows.append({'train_id': query['parent_id'], 'assigned_class': query['assigned_class'],
                         'archive_class': query['archive_class'], 'train_path': query['source_path'],
                         'test_id': neighbor['image_id'], 'test_path': neighbor['source_path'], 'rank': rank,
                         'cosine': float(similarity[qi, ti]), 'phash_min_64bit': int(distances[ti]),
                         'same_size': query['width'] == neighbor['width'] and query['height'] == neighbor['height'],
                         'same_decoded_pixels': query['pixel_sha256'] == neighbor['pixel_sha256'],
                         'train_width': query['width'], 'train_height': query['height'],
                         'test_width': neighbor['width'], 'test_height': neighbor['height'],
                         'number_test_top1_this_train': top_for_train,
                         'max_test_coverage_gap_if_removed': float(local_gap.max()) if len(local_gap) else 0.,
                         'nearest_test_similarity_percentile_among_1906_train': float((all_training_nearest_test <= similarity[qi, order[0]]).mean()),
                         'diagnostic_only': True})
        pi = int(distances.argmin()); neighbor = test_records[pi]
        phash_rows.append({'train_id': query['parent_id'], 'train_path': query['source_path'],
                           'test_id': neighbor['image_id'], 'test_path': neighbor['source_path'],
                           'phash_min_64bit': int(distances[pi]), 'cosine': float(similarity[qi, pi]),
                           'same_decoded_pixels': query['pixel_sha256'] == neighbor['pixel_sha256']})
    pairs = pd.DataFrame(rows)
    phash_pairs = pd.DataFrame(phash_rows)
    jobs = {(row.train_path, row.test_path) for row in pairs.itertuples()} | {(row.train_path, row.test_path) for row in phash_pairs.itertuples()}
    def metrics(pair):
        a, b = pixel_canvas(pair[0]), pixel_canvas(pair[1])
        return pair, (float(structural_similarity(a, b, channel_axis=2, data_range=255)),
                      float(np.abs(a.astype('float32') - b.astype('float32')).mean() / 255))
    with ThreadPoolExecutor(max_workers=4) as pool:
        metrics_lookup = dict(pool.map(metrics, sorted(jobs)))
    for table in [pairs, phash_pairs]:
        table['ssim_letterbox224'] = [metrics_lookup[(r.train_path, r.test_path)][0] for r in table.itertuples()]
        table['mae_letterbox224_0to1'] = [metrics_lookup[(r.train_path, r.test_path)][1] for r in table.itertuples()]
    pairs.to_csv(NOTES / 'table-csv/remaining_train_test_top3.csv', index=False)
    phash_pairs.to_csv(NOTES / 'table-csv/remaining_train_test_phash.csv', index=False)
    top = pairs[pairs['rank'].eq(1)].copy()
    selected = []
    for table, count in [(top.sort_values('cosine', ascending=False), 4),
                         (top.sort_values('max_test_coverage_gap_if_removed', ascending=False), 3),
                         (top.sort_values('cosine'), 1)]:
        for key in table.train_id:
            if key not in selected:
                selected.append(key)
                count -= 1
                if count == 0: break
    assert len(selected) == 8
    examples = top[top.train_id.isin(selected)].sort_values('train_id')
    pair_sheet(examples)
    total = math.ceil(len(differences) / 20)
    sheets, sheet_rows = [], []
    for page in range(total):
        subset = differences.iloc[page*20:(page+1)*20]
        path = save_sheet(subset.to_dict('records'), page+1, total)
        sheets.append({'path': str(path.relative_to(ROOT)), 'count': len(subset),
                       'first_id': subset.iloc[0].parent_id, 'last_id': subset.iloc[-1].parent_id})
        sheet_rows.extend(subset.parent_id.tolist())
        print(f'Montage {page+1}/{total}: {len(subset)} images', flush=True)
    assert sheet_rows == differences.parent_id.tolist() and len(set(sheet_rows)) == 150
    def relative_image(path):
        return html.escape(os.path.relpath(ROOT / path, HOME).replace('\\', '/'), quote=True)
    parts = ['<!doctype html><html lang="id"><meta charset="utf-8"><title>150 perbedaan train dan kemiripan test</title>',
             '<style>body{font:15px system-ui;margin:24px;background:#fafafa}section{margin-bottom:24px;padding:18px;background:white;border:1px solid #ddd}figure{display:inline-block;vertical-align:top;width:23%;margin:1%}img{width:100%;height:240px;object-fit:contain}figcaption{white-space:pre-line}code{background:#eee}nav{position:sticky;top:0;background:white;padding:12px}</style>',
             '<nav>150 foto, urut ID. Program dan arsip bukan label resmi. Test tanpa kategori. Tidak ada label atau training yang diubah.</nav>']
    for row in differences.itertuples():
        neighbors = pairs[pairs.train_id.eq(row.parent_id)].sort_values('rank')
        parts.append(f'<section id="train-{row.parent_id}"><h2>Train {row.parent_id}</h2><p>Program: {row.assigned_class} | Arsip pembanding: {row.archive_class}</p>')
        parts.append(f'<figure><a href="{relative_image(row.source_path)}"><img src="{relative_image(row.source_path)}" loading="lazy"></a><figcaption>Train {row.parent_id}\nProgram: {row.assigned_class}\nArsip: {row.archive_class}</figcaption></figure>')
        for r in neighbors.itertuples():
            parts.append(f'<figure><a href="{relative_image(r.test_path)}"><img src="{relative_image(r.test_path)}" loading="lazy"></a><figcaption>Test {r.test_id}; peringkat {r.rank}\nCosine {r.cosine:.4f}; pHash {r.phash_min_64bit}/64\nSSIM {r.ssim_letterbox224:.4f}; MAE {r.mae_letterbox224_0to1:.4f}</figcaption></figure>')
        p = phash_pairs[phash_pairs.train_id.eq(row.parent_id)].iloc[0]
        parts.append(f'<p>Terdekat menurut pHash: test {p.test_id}, jarak {p.phash_min_64bit}/64; cosine {p.cosine:.4f}. Dimensi train {neighbors.iloc[0].train_width}×{neighbors.iloc[0].train_height}; top-1 test {neighbors.iloc[0].test_width}×{neighbors.iloc[0].test_height}.</p></section>')
    parts.append('</html>')
    (HOME / 'index.html').write_bytes(('\n'.join(parts)).encode('utf-8'))
    unique_paths = set(queries.source_path) | set(test.source_path)
    lookup = profile.set_index('source_path').source_sha256.to_dict()
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(lambda p: digest(ROOT / p) == lookup[p], sorted(unique_paths)))
    assert all(verified) and len(verified) == 630
    assert all(digest(ROOT / name) == value for name, value in protected.items())
    exact_pairs = sum(q['pixel_sha256'] == t['pixel_sha256'] for q in query_records for t in test_records)
    report = {
        'status': 'passed', 'train_images_shown': 150, 'contact_sheets': sheets,
        'test_images_compared': 480, 'active_full_frame_train': 1906, 'comparison_pairs': 72000,
        'exact_decoded_pixel_matches': int(exact_pairs),
        'train_with_nearest_test_cosine_at_least': {str(t): int(top.cosine.ge(t).sum()) for t in [.8, .85, .9, .95, .98]},
        'train_with_phash_distance_at_most': {str(t): int(phash_pairs.phash_min_64bit.le(t).sum()) for t in [0, 4, 8, 12]},
        'nearest_cosine_quantiles': {str(t): float(top.cosine.quantile(t)) for t in [0, .25, .5, .75, 1]},
        'test_top1_neighbor_from_150': int(top_from_remaining.sum()),
        'unique_train_of_150_that_are_test_top1': int(coverage_table[coverage_table.nearest_is_in_150].nearest_train_id.nunique()),
        'test_coverage_gap_at_least': {str(t): int((gap >= t).sum()) for t in [.01, .02, .05, .1]},
        'max_coverage_gap': float(gap.max()),
        'train_size_distribution': queries.groupby(['width', 'height']).size().reset_index(name='count').to_dict('records'),
        'top1_pairs_same_original_size': int(top.same_size.sum()),
        'examples': examples[['train_id', 'test_id', 'cosine', 'phash_min_64bit', 'ssim_letterbox224', 'max_test_coverage_gap_if_removed']].to_dict('records'),
        'feature_cache_sha256': digest(feature_path), 'frozen_difference_list_sha256': digest(differences_path),
        'manual_labels_consumed': 0, 'model_fit_performed': False, 'label_changes': 0,
        'test_category_labels_used_or_created': False, 'source_images_hash_verified': 630,
        'active_artifacts_unchanged': True, 'protected_sha256': protected,
        'scope': 'descriptive similarity and representation coverage, not accuracy or evidence to relabel/drop individual images',
        'metric_caveats': 'Cosine thresholds are descriptive, not calibrated semantic probabilities. pHash uses minimum of original/hflip/vflip combinations. SSIM/MAE use 224px RGB letterbox and can be dominated by backgrounds or padding. Missing close matches does not imply irrelevance.',
        'seconds': time.time() - started,
    }
    (NOTES / 'json/remaining_visual_review.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
