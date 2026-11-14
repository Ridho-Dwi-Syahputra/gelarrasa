"""Locally adapt MINC/texture heads using bounded human labels and cold priors.

Unselected participant review and historical visual annotations are never read.
Hyperparameter selection uses only source-group holdouts of the 34 active
human anchors. It is a small development diagnostic, not final Accuracy.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main():
    started = time.time()
    protected = {name: sha(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    all_frame = manifest[manifest.split.eq('train') & ~manifest.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = all_frame[all_frame.training_eligible].reset_index(drop=True)
    assert len(frame) == 1904
    keys = frame.parent_key.to_numpy(str)
    extraction = json.loads((NOTES / 'json/material_texture_extraction.json').read_text())
    cache = NOTES / '.runtime/minc_train_features.npz'
    assert sha(cache) == extraction['feature_sha256']
    z = np.load(cache, allow_pickle=False)
    assert np.array_equal(z['keys'], keys) and np.array_equal(z['source_sha256'], frame.source_sha256)
    f = z['features']
    assert f.shape == (1904, 3, 1024)
    # Cold automatic priors were frozen before any human seed was activated.
    cold_path = NOTES / 'table-csv/bounded_consensus_selected.csv'
    cold = pd.read_csv(cold_path, dtype={'image_id': str}).set_index('key').loc[keys]
    cold_bundle_path = NOTES / '.runtime/before_bounded_anchor_20261107T084712Z/preprocessing_inputs.npz'
    cold_bundle = np.load(cold_bundle_path, allow_pickle=False)
    assert not any(name.startswith('human_seed') for name in cold_bundle.files)
    cold_index = dict(zip(cold_bundle['keys'], range(len(cold_bundle['keys']))))
    cold_positions = [cold_index[k] for k in keys]
    assert np.array_equal(cold_bundle['predicted_class'][cold_positions], cold.baseline_class.to_numpy(str))
    base = np.array([CLASSES.index(c) for c in cold_bundle['predicted_class'][cold_positions]])
    seeds = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    index = dict(zip(keys, range(len(keys))))
    seed_pos = np.array([index[k] for k in seeds.key])
    seed_y = np.array([CLASSES.index(c) for c in seeds.assigned_class])
    assert len(seeds) == 34 and seeds.assigned_class.value_counts().le(5).all()
    human = np.zeros(len(frame), bool)
    human[seed_pos] = True
    groups_table = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    groups, folds = groups_table.group.to_numpy(), groups_table.fold.to_numpy()
    in_human_group = np.isin(groups, groups[seed_pos])
    score = np.clip(cold_bundle['confidence'][cold_positions].astype(float), 0, 1)
    def take(path, query_keys, column='features'):
        data = np.load(path, allow_pickle=False)
        lookup = dict(zip(data['keys'], range(len(data['keys']))))
        return data[column][[lookup[k] for k in query_keys]].astype('float32')
    dino2 = unit(take(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz', keys))
    global_minc = unit(np.mean(f[:, :2], axis=1))
    detail_minc = unit(np.c_[np.mean(f, axis=1), np.std(f, axis=1)])
    classical = unit(take(ROOT / 'data/processed/features.npz', frame.sample_key.to_numpy(str)))
    banks = {'minc_global': global_minc, 'minc_foreground': unit(f[:, 1]),
             'minc_details': detail_minc, 'minc_dino': unit(np.c_[global_minc, dino2]),
             'classical_texture': classical}
    configs = [{'kernel': 'linear', 'C': 1., 'gamma': None},
               {'kernel': 'rbf', 'C': 1., 'gamma': 4.}, {'kernel': 'rbf', 'C': 10., 'gamma': 4.}]
    regimes = [1., .25]
    plan = {'method': 'locally_adapted_MINC_material_and_texture_crossfit_v1', 'train_rows_in_fit': 1904,
            'test_rows_in_fit': 0, 'held_keys_excluded': ['train:1662', 'train:1885'],
            'manual_sources': 34, 'manual_counts': seeds.assigned_class.value_counts().to_dict(),
            'material_feature_sha256': sha(cache), 'cold_prior_sha256': sha(cold_path),
            'cold_prior_and_weights_bundle_sha256': sha(cold_bundle_path),
            'cold_prior_column': 'baseline_class before first human seed activation',
            'all_review_or_archive_annotations_read': False,
            'banks': {k: v.shape[1] for k, v in banks.items()}, 'candidate_configs_per_bank': configs,
            'pseudo_weight_regimes': regimes, 'human_weight': 32.,
            'selection': 'highest mean balanced accuracy on34leave-source-group-out anchors across both regimes; tie preserves listed order',
            'gate': 'both regimes same category with>=4/5 heads; hybrid head and>=2/3 material-only heads agree in both; same-category anchor cosine>=.65 and category margin>=.02 in>=2banks; preserve human and earlier algorithmic refinements',
            'publication_guard': 'pooled anchor holdout majority recovers>=14/34 and candidate changes<=50; otherwise no changes accepted',
            'evaluation_limit': 'development anchors selected for diversity/difficulty and hyperparameter selection; not independent Accuracy or final validation',
            'protected_sha256': protected, 'source_code_sha256': sha(__file__)}
    dump(NOTES / 'json/material_assignment_plan.json', plan)
    y = base.copy()
    y[seed_pos] = seed_y
    history, chosen_configs, chosen_holdouts, predictions, localities = [], [], [], [], []

    def predict(kernel, tr, va, config, multiplier):
        weights = (.25 + .75 * score[tr]) * multiplier
        weights[human[tr]] = 32.
        model = SVC(kernel='precomputed', C=config['C'], class_weight='balanced', decision_function_shape='ovr', break_ties=True)
        model.fit(kernel[np.ix_(tr, tr)], y[tr], sample_weight=weights)
        return model.predict(kernel[np.ix_(va, tr)]).astype('int32')

    with threadpool_limits(limits=4):
        for name, x in banks.items():
            sim = x @ x.T
            best, best_score, best_loo, best_kernel = None, -np.inf, None, None
            for config in configs:
                kernel = sim if config['kernel'] == 'linear' else np.exp(-2 * config['gamma'] * np.maximum(1 - sim, 0))
                holdouts = []
                for multiplier in regimes:
                    loo = []
                    for pos in seed_pos:
                        tr = np.flatnonzero(groups != groups[pos])
                        loo.append(int(predict(kernel, tr, np.array([pos]), config, multiplier)[0]))
                    holdouts.append(loo)
                holdouts = np.array(holdouts)
                quality = float(np.mean([balanced_accuracy_score(seed_y, hp) for hp in holdouts]))
                history.append({'bank': name, **config, 'mean_anchor_balanced_recall': quality,
                                'anchor_matches_by_prior_regime': [(hp == seed_y).sum().item() for hp in holdouts]})
                if quality > best_score:
                    best, best_score, best_loo, best_kernel = config, quality, holdouts, kernel
            chosen_configs.append({'bank': name, **best, 'anchor_development_balanced_recall': best_score})
            chosen_holdouts.append(best_loo)
            outputs = []
            for multiplier in regimes:
                pred = np.full(len(frame), -1, dtype='int32')
                pred[seed_pos] = seed_y
                for fold in range(5):
                    tr = np.flatnonzero((folds != fold) | in_human_group)
                    va = np.flatnonzero((folds == fold) & ~in_human_group)
                    assert not set(groups[tr]) & set(groups[va])
                    pred[va] = predict(best_kernel, tr, va, best, multiplier)
                for group in sorted(set(groups[in_human_group & ~human])):
                    tr = np.flatnonzero(groups != group)
                    va = np.flatnonzero((groups == group) & ~human)
                    pred[va] = predict(best_kernel, tr, va, best, multiplier)
                assert (pred >= 0).all()
                outputs.append(pred)
            predictions.append(outputs)
            localities.append(np.stack([sim[:, seed_pos[seed_y == c]].max(1) if np.any(seed_y == c) else np.full(len(frame), -np.inf)
                                        for c in range(8)], axis=1))
            print('Material/texture family selected:', chosen_configs[-1], flush=True)
    predictions = np.array(predictions)  # family, prior regime, image
    holdout_votes = np.array(chosen_holdouts).reshape(10, len(seeds))
    anchor_majority = np.stack([(holdout_votes == c).sum(0) for c in range(8)], axis=1).argmax(1)
    anchor_matches = int((anchor_majority == seed_y).sum())
    majors, supports = [], []
    for regime in range(2):
        tally = np.stack([(predictions[:, regime] == c).sum(0) for c in range(8)], axis=1)
        majors.append(tally.argmax(1))
        supports.append(tally.max(1))
    major = majors[0]
    local_support = np.zeros(len(frame), int)
    for nearest in localities:
        positive = nearest[np.arange(len(frame)), major]
        alternative = nearest.copy()
        alternative[np.arange(len(frame)), major] = -np.inf
        local_support += (positive >= .65) & (positive - alternative.max(1) >= .02)
    material_ok = np.ones(len(frame), bool)
    for regime in range(2):
        material_ok &= (predictions[3, regime] == major) & ((predictions[:3, regime] == major).sum(0) >= 2)
    protected_rows = ~frame.label_source.isin(['iterative_program_refinement_snapshot', 'historical_routing_snapshot']).to_numpy()
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate = (majors[0] == majors[1]) & (supports[0] >= 4) & (supports[1] >= 4) & material_ok & (local_support >= 2)
    gate &= ~human & ~protected_rows & (major != current)
    proposed_count = int(gate.sum())
    accepted_run = anchor_matches >= 14 and proposed_count <= 50
    if not accepted_run:
        gate[:] = False
    result = all_frame[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    for column, original in [('baseline_class', 'assigned_class'), ('predicted_class', 'assigned_class'),
                             ('category_source', 'label_source'), ('category_score', 'category_score'),
                             ('category_score_kind', 'category_score_kind'), ('needs_review', 'needs_review'), ('human_reviewed', 'human_reviewed')]:
        result[column] = all_frame[original]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[major[i]], 'material_cnn_crossfit_consensus_snapshot', min(supports[0][i], supports[1][i]) / 5., 'multi_method_vote_share', True]
    output = NOTES / 'table-csv/material_assignment_selected.csv'
    result.reset_index().to_csv(output, index=False)
    pd.DataFrame(history).to_csv(NOTES / 'table-csv/material_anchor_development.csv', index=False)
    holdout = seeds[['key', 'assigned_class']].copy()
    holdout['pooled_development_prediction'] = np.array(CLASSES)[anchor_majority]
    holdout.to_csv(NOTES / 'table-csv/material_anchor_holdout.csv', index=False)
    pd.DataFrame({'key': keys, 'candidate_class': np.array(CLASSES)[major], 'regime1_votes': supports[0],
                  'regime2_votes': supports[1], 'local_anchor_banks': local_support, 'material_hybrid_support': material_ok,
                  'accepted': gate}).to_csv(NOTES / 'table-csv/material_assignment_evidence.csv', index=False)
    prior_report = json.loads((NOTES / 'json/review_final_frozen.json').read_text())
    initial = pd.read_csv(cold_path).set_index('key')
    report = dict(prior_report)
    report.update(status='frozen_before_posthoc_material_audit', method=plan['method'], protected_sha256=protected,
                  prediction_sha256=sha(output), chosen_heads=chosen_configs,
                  material_feature_sha256=sha(cache), material_plan_sha256=sha(NOTES / 'json/material_assignment_plan.json'),
                  human_leave_source_group_out_matches=anchor_matches, human_holdout_rows=34,
                  changed_rows_vs_baseline=int(gate.sum()), cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  new_material_proposals=proposed_count, accepted_by_development_guard=accepted_run,
                  material_changes_accepted=int(gate.sum()), source_counts=result.category_source.value_counts().to_dict(),
                  source_code_sha256=sha(__file__), seconds=time.time() - started,
                  validation_limit=plan['evaluation_limit'], test_rows_in_fit=0,
                  extra_cluster_rows=int(result.category_source.eq('bounded_anchor_cluster_consensus_snapshot').sum()))
    assert all(sha(ROOT / k) == v for k, v in protected.items())
    dump(NOTES / 'json/material_assignment_frozen.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'human_leave_source_group_out_matches', 'new_material_proposals',
                                            'accepted_by_development_guard', 'material_changes_accepted', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
