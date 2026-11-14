"""Read-only train/test visual coverage for the 17 current review disagreements.

Test enters only this diagnostic. Nothing here updates training eligibility,
categories, hyperparameters, or submitted test predictions.
"""
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from threadpoolctl import threadpool_limits

from show_remaining_train_test import ROOT, NOTES, OLD, digest, phash_distance, pixel_canvas


def main():
    active = json.loads((NOTES / 'json/current_review_status.json').read_text())
    assert active['status'] == 'passed' and active['remaining_review_differences'] == 17
    protected = {p: digest(ROOT / p) for p in ['data/processed/manifest.csv',
                                                'data/preprocessing_inputs.npz',
                                                'final-submit/DSC2026_Notebook.ipynb']}
    differences_path = NOTES / 'table-csv/remaining_current_review_differences.csv'
    diffs = pd.read_csv(differences_path, dtype={'parent_id': str}).sort_values('parent_id')
    assert len(diffs) == 17 and diffs.parent_id.is_unique
    profile = pd.read_csv(NOTES / 'image_profile.csv',
                          dtype={'image_id': str, 'phash': str,
                                 'phash_hflip': str, 'phash_vflip': str})
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    full_keys = manifest.loc[manifest.split.eq('train') & ~manifest.is_crop, 'parent_key'].tolist()
    full = profile.set_index('key').loc[full_keys].reset_index()
    test = profile[profile.split.eq('test')].sort_values('image_id').reset_index(drop=True)
    queries = diffs.merge(profile, left_on='parent_id', right_on='image_id',
                          suffixes=('', '_profile')).sort_values('parent_id').reset_index(drop=True)
    assert len(full) == 1906 and len(test) == 480 and len(queries) == 17
    assert queries.source_path.eq(queries.source_path_profile).all()
    feature_path = OLD / 'dinov2_features.npz'
    validation = json.loads((OLD / 'feature_validation.json').read_text())
    assert digest(feature_path) == validation['feature_files']['dinov2_features.npz']
    z = np.load(feature_path, allow_pickle=False)
    features = z['features'].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    lookup = {str(k): i for i, k in enumerate(z['keys'])}
    train_x = features[[lookup[k] for k in full.key]]
    test_x = features[[lookup[k] for k in test.key]]
    query_x = features[[lookup[k] for k in queries.key]]
    with threadpool_limits(limits=4):
        similarity = query_x @ test_x.T
        coverage = test_x @ train_x.T
    remaining_mask = full.image_id.isin(diffs.parent_id).to_numpy()
    top_all = coverage.argmax(1)
    test_affected = remaining_mask[top_all]
    alternative = coverage[:, ~remaining_mask].max(1)
    gap = coverage[np.arange(len(test)), top_all] - alternative
    rows = []
    test_rows = test.to_dict('records')
    for qi, query in enumerate(queries.to_dict('records')):
        order = np.argsort(-similarity[qi], kind='stable')[:3]
        distances = np.array([phash_distance(query, t) for t in test_rows])
        by_hash = int(distances.argmin())
        q_affected = test_affected & (full.image_id.to_numpy()[top_all] == query['parent_id'])
        for rank, ti in enumerate(order, 1):
            neighbor = test_rows[int(ti)]
            rows.append({'train_id': query['parent_id'], 'current_class': query['current_class'],
                         'review_class_diagnostic_only': query['proposed_class'],
                         'train_path': query['source_path'], 'test_id': neighbor['image_id'],
                         'test_path': neighbor['source_path'], 'rank': rank,
                         'dinov2_cosine': float(similarity[qi, ti]),
                         'phash_hamming_64': int(distances[ti]),
                         'same_decoded_pixels': bool(query['pixel_sha256'] == neighbor['pixel_sha256']),
                         'same_native_size': bool(query['width'] == neighbor['width'] and
                                                  query['height'] == neighbor['height']),
                         'train_width': int(query['width']), 'train_height': int(query['height']),
                         'test_width': int(neighbor['width']), 'test_height': int(neighbor['height']),
                         'test_count_where_this_train_is_nearest': int(q_affected.sum()),
                         'largest_coverage_gap_without_this_train': float(gap[q_affected].max()) if q_affected.any() else 0.,
                         'nearest_phash_test_id': test_rows[by_hash]['image_id'],
                         'nearest_phash_distance': int(distances[by_hash]),
                         'diagnostic_only': True})
    output = pd.DataFrame(rows)
    pairs = {(r.train_path, r.test_path) for r in output.itertuples()}
    def pixel_metric(pair):
        a, b = pixel_canvas(pair[0]), pixel_canvas(pair[1])
        return pair, (float(structural_similarity(a, b, channel_axis=2, data_range=255)),
                      float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean()/255))
    with ThreadPoolExecutor(max_workers=4) as pool:
        metrics = dict(pool.map(pixel_metric, sorted(pairs)))
    output['ssim_224_letterbox'] = [metrics[(r.train_path, r.test_path)][0] for r in output.itertuples()]
    output['mae_224_letterbox'] = [metrics[(r.train_path, r.test_path)][1] for r in output.itertuples()]
    csv_path = NOTES / 'table-csv/remaining17_train_test_similarity.csv'
    output.to_csv(csv_path, index=False)
    top = output[output['rank'].eq(1)].copy()
    affected_test = test.loc[test_affected, ['image_id', 'source_path']].copy()
    affected_test['nearest_train_id'] = full.image_id.to_numpy()[top_all[test_affected]]
    affected_test['cosine_with_all_train'] = coverage[np.arange(len(test))[test_affected], top_all[test_affected]]
    affected_test['cosine_gap_without_17'] = gap[test_affected]
    affected_test.to_csv(NOTES / 'table-csv/remaining17_test_nearest_coverage.csv', index=False)
    report = {
        'status': 'passed', 'active_prefix': active['active_prefix'],
        'review_disagreements': 17, 'test_images': 480,
        'test_with_nearest_train_among_17': int(test_affected.sum()),
        'unique_remaining_train_that_are_test_top1': int(affected_test.nearest_train_id.nunique()),
        'max_test_coverage_cosine_gap_if_17_removed': float(gap.max()),
        'coverage_gap_at_least_0_02': int((gap >= .02).sum()),
        'coverage_gap_at_least_0_05': int((gap >= .05).sum()),
        'top1_cosine_at_least': {str(t): int(top.dinov2_cosine.ge(t).sum()) for t in [.75, .8, .85, .9, .95]},
        'top1_phash_at_most': {str(t): int(top.phash_hamming_64.le(t).sum()) for t in [0, 4, 8, 12]},
        'exact_decoded_pixel_matches_in_top3': int(output.same_decoded_pixels.sum()),
        'source_feature_sha256': digest(feature_path),
        'current_review_sha256': digest(differences_path),
        'train_changes': 0, 'test_training_rows': 0, 'test_labels_created': False,
        'scope': 'descriptive read-only similarity; not a criterion to select train rows or infer test labels',
        'caveat': 'cosine/pHash/SSIM measure visual proximity; none proves category, and low similarity does not prove a train image is dispensable',
    }
    gallery = NOTES / 'remaining_visual_review/remaining17_test_neighbors.html'
    gallery.parent.mkdir(parents=True, exist_ok=True)
    def rel(path):
        return html.escape(os.path.relpath(ROOT / path, gallery.parent).replace('\\', '/'), quote=True)
    pieces = ['<!doctype html><html lang="id"><meta charset="utf-8"><title>17 train dan tetangga test</title>',
              '<style>body{font:15px system-ui;margin:24px;background:#fafafa}section{padding:16px;margin:16px 0;background:#fff;border:1px solid #ddd}figure{display:inline-block;vertical-align:top;width:23%;margin:1%}img{width:100%;height:225px;object-fit:contain}</style>',
              '<h1>17 train yang berbeda dengan review: tetangga visual di test</h1>',
              '<p>Ini pencarian diagnostik. Kemiripan bukan label test dan tidak menentukan data training.</p>']
    for row in top.itertuples():
        subset = output[output.train_id.eq(row.train_id)].sort_values('rank')
        pieces.append(f'<section><h2>Train {row.train_id}</h2><p>Kategori program: {html.escape(row.current_class)}; kategori review terpisah: {html.escape(row.review_class_diagnostic_only)}</p>')
        pieces.append(f'<figure><a href="{rel(row.train_path)}"><img loading="lazy" src="{rel(row.train_path)}"></a><figcaption>Train {row.train_id}</figcaption></figure>')
        for item in subset.itertuples():
            pieces.append(f'<figure><a href="{rel(item.test_path)}"><img loading="lazy" src="{rel(item.test_path)}"></a><figcaption>Test {item.test_id}; cosine {item.dinov2_cosine:.3f}; pHash {item.phash_hamming_64}; SSIM {item.ssim_224_letterbox:.3f}</figcaption></figure>')
        pieces.append(f'<p>Test yang menjadikan train ini tetangga #1: {row.test_count_where_this_train_is_nearest}; jarak pHash terdekat ke test {row.nearest_phash_test_id}: {row.nearest_phash_distance}/64.</p></section>')
    pieces.append('</html>')
    gallery.write_text('\n'.join(pieces), encoding='utf-8')
    report['gallery'] = str(gallery.relative_to(ROOT))
    (NOTES / 'json/remaining17_test_similarity.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    assert all(digest(ROOT / p) == expected for p, expected in protected.items())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
