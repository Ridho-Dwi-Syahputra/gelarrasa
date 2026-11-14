"""Fixed anchor-only SVM and harmonic diffusion; no unselected review targets.

Unlike the prior-weight probe, no historical pseudo-label is a fitting target.
Frozen representations remain image-only; test and held sources are excluded.
"""
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main():
    started = time.time()
    previous_info = json.loads((NOTES / 'json/review_revision_frozen.json').read_text())
    previous_path = NOTES / 'table-csv/review_revision_selected.csv'
    assert sha(previous_path) == previous_info['prediction_sha256']
    result = pd.read_csv(previous_path, dtype={'image_id': str}).set_index('key')
    seeds_path = NOTES / previous_info['human_seed_file']
    assert sha(seeds_path) == previous_info['human_seed_sha256']
    seeds = pd.read_csv(seeds_path, dtype={'image_id': str}, usecols=['image_id', 'proposed_class', 'source_sha256'])
    assert seeds.proposed_class.value_counts().le(5).all()
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    frame = m[m.split.eq('train') & ~m.is_crop & m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    assert len(frame) == 1904
    keys = frame.parent_key.to_numpy(str)
    index = dict(zip(keys, range(len(keys))))
    positions = np.array([index['train:' + k] for k in seeds.image_id])
    y = np.array([CLASSES.index(c) for c in seeds.proposed_class])
    human = np.zeros(len(frame), bool)
    human[positions] = True
    groups = frame.parent_group.to_numpy()
    plan = {'method': 'bounded_review_revision_plus_anchor_only_material_diffusion_v1',
            'source_snapshot_sha256': previous_info['prediction_sha256'],
            'human_seed_sha256': sha(seeds_path), 'manual_unique_source_images': len(seeds),
            'test_rows_in_fit': 0, 'held_sources_in_fit': 0, 'fit_train_rows': len(frame),
            'archive_or_unselected_review_labels_read': False, 'historical_pseudo_targets_in_this_probe': 0,
            'svm': {'kernel': 'rbf', 'gamma': 4., 'C': 10., 'class_weight': 'balanced'},
            'graphs': {'neighbors': 20, 'alpha': .99, 'clamped_human_anchors_only': True, 'max_iterations': 2000},
            'gate': '>=4/5 anchor-only SVM heads and>=2/3 harmonic graphs agree; nearest same-category anchor cosine>=.65 and its category cosine margin>=.04 in>=2/5 banks; preserve first-stage changes and lighting anchors',
            'configuration_selection': 'fixed before fitting and posthoc review; no parameter search on unselected labels',
            'source_code_sha256': sha(__file__), 'protected_sha256': previous_info['protected_sha256']}
    dump(NOTES / 'json/bounded_material_diffusion_plan.json', plan)

    def take(path):
        z = np.load(path, allow_pickle=False)
        lookup = dict(zip(z['keys'], range(len(z['keys']))))
        return z['features'][[lookup[k] for k in keys]].astype('float32')

    d2 = unit(take(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz'))
    patch = take(OLD / 'output/embeddings/stage5_patch_v1/original224.npz')
    d3 = take(NOTES / '.runtime/dinov3_train_features.npz')
    task = take(NOTES / '.runtime/task8_teacher_features.npz')
    banks = {'dino2': d2, 'dino2_local': unit(patch * np.repeat([np.sqrt(.5), .5, .5], 384)),
             'dino3_global': unit(d3[:, :768]),
             'dino3_foreground': unit(np.c_[d3[:, :384] * np.sqrt(.5), d3[:, 768:1152] * .5, d3[:, 1152:] * .5]),
             'dino3_task8': unit(np.c_[unit(d3[:, :768]), task])}
    pred, graph_pred, near, diagnostics = [], [], [], []
    with threadpool_limits(limits=4):
        for name, x in banks.items():
            sim = (x @ x.T).astype('float32')
            kernel = np.exp(-8 * np.maximum(1 - sim, 0))
            model = SVC(kernel='precomputed', C=10., class_weight='balanced', decision_function_shape='ovr', break_ties=True)
            model.fit(kernel[np.ix_(positions, positions)], y)
            p = model.predict(kernel[:, positions]).astype('int32')
            p[positions] = y
            pred.append(p)
            nearest = np.stack([sim[:, positions[y == c]].max(1) if np.any(y == c) else np.full(len(frame), -np.inf)
                                for c in range(8)], axis=1)
            near.append(nearest)
            for i, pos in enumerate(positions):
                fit_positions = positions[groups[positions] != groups[pos]]
                fit_y = y[groups[positions] != groups[pos]]
                loo = SVC(kernel='precomputed', C=10., class_weight='balanced', decision_function_shape='ovr', break_ties=True)
                loo.fit(kernel[np.ix_(fit_positions, fit_positions)], fit_y)
                lp = int(loo.predict(kernel[np.ix_([pos], fit_positions)])[0])
                diagnostics.append({'image_id': seeds.iloc[i].image_id, 'reviewed_class': CLASSES[y[i]],
                                    'bank': name, 'prediction': CLASSES[lp],
                                    'heldout_category_has_other_training_anchors': bool(np.any(fit_y == y[i]))})
            if name in {'dino2', 'dino3_foreground', 'dino3_task8'}:
                np.fill_diagonal(sim, -np.inf)
                nn = np.argsort(-sim, axis=1, kind='stable')[:, :20]
                edge = np.maximum(np.take_along_axis(sim, nn, axis=1) - .2, 0) ** 4
                adjacency = sparse.csr_matrix((edge.ravel(), (np.repeat(np.arange(len(frame)), 20), nn.ravel())), shape=(len(frame), len(frame)))
                adjacency = (adjacency + adjacency.T) * .5
                pgraph = sparse.diags(1 / np.asarray(adjacency.sum(1)).ravel()) @ adjacency
                prior = np.zeros((len(frame), 8), dtype='float64')
                prior[positions] = np.eye(8)[y]
                f = prior.copy()
                for step in range(2000):
                    update = .99 * (pgraph @ f) + .01 * prior
                    update[positions] = prior[positions]
                    delta = float(np.abs(update - f).max())
                    f = update
                    if delta < 1e-7:
                        break
                graph_pred.append(f.argmax(1))
            print('Anchor-only material bank:', name, flush=True)
    pred, graph_pred = np.stack(pred), np.stack(graph_pred)
    tally = np.stack([(pred == c).sum(0) for c in range(8)], axis=1)
    major, support = tally.argmax(1), tally.max(1)
    gsupport = (graph_pred == major).sum(0)
    local = np.zeros(len(frame), int)
    for n in near:
        proposed = n[np.arange(len(frame)), major]
        alternatives = n.copy()
        alternatives[np.arange(len(frame)), major] = -np.inf
        margin = proposed - alternatives.max(1)
        local += (proposed >= .65) & (margin >= .04)
    current = result.loc[keys]
    current_labels = np.array([CLASSES.index(c) for c in current.predicted_class])
    changed_before = current.predicted_class.ne(current.baseline_class).to_numpy()
    gate = (support >= 4) & (gsupport >= 2) & (local >= 2) & ~human & ~changed_before
    gate &= ~current.category_source.eq('two_view_lighting_ontology_snapshot').to_numpy() & (major != current_labels)
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[major[i]], 'bounded_anchor_material_diffusion_snapshot', support[i] / 5., 'multi_method_vote_share', True]
    output = NOTES / 'table-csv/review_material_selected.csv'
    result.reset_index().to_csv(output, index=False)
    pd.DataFrame(diagnostics).to_csv(NOTES / 'table-csv/bounded_material_anchor_diagnostics.csv', index=False)
    all_votes = pd.DataFrame({'key': keys, 'classifier_category': np.array(CLASSES)[major], 'classifier_votes': support,
                              'graph_votes': gsupport, 'material_locality_banks': local, 'accepted': gate})
    all_votes.to_csv(NOTES / 'table-csv/bounded_material_evidence.csv', index=False)
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv', dtype={'image_id': str}).set_index('key')
    report = dict(previous_info)
    report.update(method=plan['method'], status='frozen_before_revised_review_audit', prediction_sha256=sha(output),
                  source_code_sha256=sha(__file__), anchor_only_material_plan_sha256=sha(NOTES / 'json/bounded_material_diffusion_plan.json'),
                  anchor_only_material_extra_changes=int(gate.sum()),
                  changed_rows_vs_baseline=int(result.predicted_class.ne(result.baseline_class).sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  material_probe_seconds=time.time() - started,
                  source_counts=result.category_source.value_counts().to_dict())
    assert all(sha(ROOT / k) == v for k, v in report['protected_sha256'].items())
    dump(NOTES / 'json/review_material_frozen.json', report)
    print(json.dumps({'additional_material_changes': int(gate.sum()), 'total_changes': report['changed_rows_vs_baseline'],
                      'seconds': time.time() - started}, indent=2), flush=True)


if __name__ == '__main__':
    main()
