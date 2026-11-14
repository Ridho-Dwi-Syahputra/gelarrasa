"""Train-only SIFT/RANSAC near-source evidence, independent of full review labels."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main():
    started = time.time()
    cv2.setNumThreads(1)
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    frame = m[m.split.eq('train') & ~m.is_crop & m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    assert len(frame) == 1904 and frame.parent_key.str.startswith('train:').all()
    human = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv').set_index('key')
    assert len(human) == 34
    protected = {name: sha(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    plan = {'scope': 'geometrically verified source-image similarity, not material prediction by appearance',
            'train_rows': len(frame), 'test_rows': 0, 'manual_sources': len(human),
            'candidates': 'union of top4 DINOv2 and top4 DINOv3 global train neighbors per image',
            'sift_keypoints': 600, 'maximum_resolution': 640, 'lowe_ratio': .70,
            'homography': {'method': 'RANSAC', 'threshold': 3., 'iterations': 2000, 'confidence': .995},
            'accepted_edge': '>=16inliers,>=.7inlier ratio, convex-hull area>=.15 of both images, plausible transformed corner bounds',
            'transfer': 'only components with exactly one distinct reviewed human category; no conflicts; no review/test labels outside34 read',
            'protected_sha256': protected, 'source_code_sha256': sha(__file__),
            'limitation': 'matched backgrounds or printed logos may still be misleading; a component transfer remains program evidence, not ground truth'}
    dump(NOTES / 'json/local_geometry_plan.json', plan)
    def take(path):
        z = np.load(path, allow_pickle=False)
        lookup = dict(zip(z['keys'], range(len(z['keys']))))
        return unit(z['features'][[lookup[k] for k in frame.parent_key]].astype('float32'))
    d2 = take(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz')
    d3 = take(NOTES / '.runtime/dinov3_train_features.npz')[:, :768]
    d3 = unit(d3)
    pairs = set()
    for x in [d2, d3]:
        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        nn = np.argsort(-sim, axis=1, kind='stable')[:, :4]
        for i, neighbors in enumerate(nn):
            for j in neighbors:
                pairs.add((min(i, int(j)), max(i, int(j))))
    pairs = sorted(pairs)
    print('Geometric candidate pairs:', len(pairs), flush=True)
    def describe(row):
        with Image.open(ROOT / row.source_path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((640, 640), Image.Resampling.LANCZOS)
            gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=600, contrastThreshold=.04)
        kp, desc = sift.detectAndCompute(gray, None)
        points = np.array([v.pt for v in kp], dtype='float32').reshape(-1, 2)
        return points, desc, gray.shape
    with ThreadPoolExecutor(max_workers=2) as pool:
        features = list(pool.map(describe, frame.itertuples()))
    print('SIFT descriptors ready', len(frame), 'seconds', round(time.time() - started), flush=True)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted, evidence = [], []
    for n, (i, j) in enumerate(pairs):
        a, da, shape_a = features[i]
        b, db, shape_b = features[j]
        if da is None or db is None or min(len(a), len(b)) < 16:
            continue
        matches = matcher.knnMatch(da, db, k=2)
        good = [v[0] for v in matches if len(v) == 2 and v[0].distance < .70 * v[1].distance]
        if len(good) < 16:
            continue
        pa = a[[v.queryIdx for v in good]]
        pb = b[[v.trainIdx for v in good]]
        matrix, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3., maxIters=2000, confidence=.995)
        if matrix is None or mask is None:
            continue
        valid = mask.ravel().astype(bool)
        count = int(valid.sum())
        ca = float(cv2.contourArea(cv2.convexHull(pa[valid]))) / np.prod(shape_a) if count >= 3 else 0.
        cb = float(cv2.contourArea(cv2.convexHull(pb[valid]))) / np.prod(shape_b) if count >= 3 else 0.
        ha, wa = shape_a
        corners = cv2.perspectiveTransform(np.array([[[0, 0], [wa - 1, 0], [wa - 1, ha - 1], [0, ha - 1]]], dtype='float32'), matrix)[0]
        hb, wb = shape_b
        plausible = bool(np.isfinite(corners).all() and corners[:, 0].min() > -wb and corners[:, 0].max() < 2 * wb
                         and corners[:, 1].min() > -hb and corners[:, 1].max() < 2 * hb
                         and cv2.isContourConvex(corners))
        passed = count >= 16 and count / len(good) >= .70 and min(ca, cb) >= .15 and plausible
        evidence.append({'key_a': frame.iloc[i].parent_key, 'key_b': frame.iloc[j].parent_key,
                         'matches': len(good), 'inliers': count, 'inlier_fraction': count / len(good),
                         'hull_fraction_a': ca, 'hull_fraction_b': cb, 'plausible_warp': plausible, 'accepted': passed})
        if passed:
            accepted.append((i, j))
        if n and n % 2000 == 0:
            print('Geometric pairs inspected:', n, '/', len(pairs), flush=True)
    parent = np.arange(len(frame))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in accepted:
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)
    groups = np.array([find(i) for i in range(len(frame))])
    rows = []
    for group in sorted(set(groups)):
        members = np.flatnonzero(groups == group)
        if len(members) < 2:
            continue
        keys = frame.iloc[members].parent_key.tolist()
        anchors = human.loc[human.index.intersection(keys)]
        categories = set(anchors.assigned_class)
        proposed = next(iter(categories)) if len(categories) == 1 else ''
        for pos in members:
            row = frame.iloc[pos]
            rows.append({'key': row.parent_key, 'image_id': row.parent_id, 'current_class': row.assigned_class,
                         'geometry_component': int(group), 'component_size': len(members), 'human_anchors': len(anchors),
                         'conflicting_human_categories': len(categories) > 1, 'proposed_class': proposed,
                         'changed_proposal': bool(proposed and proposed != row.assigned_class),
                         'human_anchor_keys': '|'.join(anchors.index)})
    pd.DataFrame(evidence).to_csv(NOTES / 'table-csv/local_geometry_edges.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(NOTES / 'table-csv/local_geometry_components.csv', index=False)
    report = {'status': 'diagnostic_frozen', 'train_rows': len(frame), 'test_rows': 0, 'candidate_pairs': len(pairs),
              'geometrically_supported_edges': len(accepted), 'component_rows': len(rows),
              'proposed_changes': int(result.changed_proposal.sum()) if len(result) else 0,
              'active_data_modified': False, 'protected_sha256': protected, 'source_code_sha256': sha(__file__),
              'seconds': time.time() - started}
    assert all(sha(ROOT / k) == v for k, v in protected.items())
    dump(NOTES / 'json/local_geometry_frozen.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
