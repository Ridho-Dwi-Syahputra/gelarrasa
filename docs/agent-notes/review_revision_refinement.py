"""Bounded participant-review revision; train-only model refinement, not bulk relabeling.

Modes are separate so the learning step cannot read the full review inventory.
The original review tables and previous model snapshots remain immutable.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit

PREFIX = 'review_revision'
SEEDS = NOTES / 'table-csv/review_revision_seeds.csv'


def prepare():
    review_path = NOTES / 'table-csv/bounded_review_outcome.csv'
    review = pd.read_csv(review_path, dtype={'parent_id': str}).fillna('')
    # The three leading sheets in the previous answer comprised these 59 rows.
    revised = review.copy()
    leading = revised.triage.eq('still_differs_from_stated_preference')
    assert leading.sum() == 59
    revised['review_revision'] = 'previous_participant_review'
    revised.loc[leading, 'review_revision'] = 'participant_confirms_first_three_disagreement_sheets'
    second = revised.triage.eq('not_reviewed')
    correct_program = {'0260', '0267', '0352', '0396', '0429', '0512', '0531', '0536'}
    assert second.sum() == 20 and correct_program <= set(revised.loc[second, 'parent_id'])
    revised.loc[second, 'proposed_class'] = [
        r.predicted_class if r.parent_id in correct_program else r.archive_class
        for r in revised.loc[second].itertuples()]
    revised.loc[second, 'team_preference'] = [
        'program' if k in correct_program else 'archive' for k in revised.loc[second, 'parent_id']]
    revised.loc[second, 'review_revision'] = 'participant_eight_program_correct_other_twelve_archive_confirmed'
    revised.loc[revised.parent_id.eq('1215'), 'proposed_class'] = 'cardboard'
    revised.loc[revised.parent_id.eq('1215'), 'team_preference'] = 'explicit_cardboard'
    revised.loc[revised.parent_id.eq('1215'), 'review_revision'] = 'participant_explicit_cardboard'
    held = {'1662', '1885'}
    # Preserve the explicit hold decision; the repeated attached old sheet is
    # not an instruction to reintroduce the two held sources.
    revised['confirmed_for_audit'] = revised.proposed_class.isin(CLASSES) & ~revised.parent_id.isin(held)
    revision_path = NOTES / 'table-csv/team_review_revision.csv'
    revised.to_csv(revision_path, index=False)

    previous_path = NOTES / 'table-csv/bounded_human_seeds.csv'
    previous = pd.read_csv(previous_path, dtype={'image_id': str}).fillna('')
    ledger = json.loads((NOTES / 'json/manual_label_budget_status.json').read_text())
    assert set(previous.image_id) == set(ledger['manual_unique_source_images_ever_used'])
    eligible = revised[revised.confirmed_for_audit].copy()
    assert len(eligible) == 126
    z = np.load(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz', allow_pickle=False)
    lookup = dict(zip(z['keys'], range(len(z['keys']))))
    features = unit(z['features'])
    groups = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv', dtype={'image_id': str}).set_index('image_id').group.to_dict()
    selected = previous.to_dict('records')
    used = set(previous.image_id)
    used_groups = {groups[k] for k in used}

    def append(row, reason, score=None):
        selected.append({'image_id': row.parent_id, 'source_path': row.source_path,
                         'proposed_class': row.proposed_class, 'source': 'user_review_of_displayed_train_image',
                         'selection_reason': reason, 'selection_score': score,
                         'review_revision': row.review_revision})
        used.add(row.parent_id)
        used_groups.add(groups[row.parent_id])

    # Explicit per-image instruction, accounted as a human anchor, never hidden
    # inside an algorithmic override.
    append(eligible[eligible.parent_id.eq('1215')].iloc[0], 'explicit participant cardboard decision')
    for category in CLASSES:
        while sum(r['proposed_class'] == category for r in selected) < 5:
            pool = eligible[eligible.proposed_class.eq(category) & ~eligible.parent_id.isin(used)]
            pool = pool[~pool.parent_id.map(groups).isin(used_groups)].sort_values('parent_id')
            if pool.empty:
                break
            positions = np.array([lookup['train:' + k] for k in pool.parent_id])
            prior = [lookup['train:' + r['image_id']] for r in selected if r['proposed_class'] == category]
            if prior:
                scores = 1 - (features[positions] @ features[prior].T).max(1)
            else:
                scores = (features[positions] @ features[positions].T).mean(1)
            choice = int(scores.argmax())
            append(pool.iloc[choice], 'fixed farthest-first train representation coverage; no test use', float(scores[choice]))
    seeds = pd.DataFrame(selected).sort_values('image_id')
    assert seeds.image_id.is_unique and seeds.proposed_class.value_counts().le(5).all()
    assert set(previous.image_id) <= set(seeds.image_id)
    assert not set(seeds.image_id) & held
    source_hash = []
    for row in seeds.itertuples():
        source_hash.append(sha(ROOT / row.source_path))
    seeds['source_sha256'] = source_hash
    seeds.to_csv(SEEDS, index=False)
    dump(NOTES / 'json/review_revision_selection.json', {
        'review_revision_sha256': sha(revision_path), 'previous_review_sha256': sha(review_path),
        'human_seed_sha256': sha(SEEDS), 'previous_unique_used': len(previous),
        'selected_unique_sources': len(seeds), 'selected_class_counts': seeds.proposed_class.value_counts().to_dict(),
        'new_ids': sorted(set(seeds.image_id) - set(previous.image_id)),
        'held_sources_preserved': sorted(held), 'full_review_is_not_training_targets': True,
        'unselected_review_labels_passed_to_fit': False, 'test_used_for_selection': False,
        'interpretation': 'first three sheets refer to the 59-image leading group in the previous answer; repeated old attachments do not revoke explicit holds',
    })
    print(seeds[['image_id', 'proposed_class']].to_string(index=False), flush=True)
    print('Selected:', len(seeds), seeds.proposed_class.value_counts().to_dict(), flush=True)


def fit():
    started = time.time()
    protected = {name: sha(ROOT / name) for name in [
        'data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    # This mode reads ONLY the bounded seed manifest, never the full review or
    # archive labels. All target/category choice below is by the fixed models.
    seeds = pd.read_csv(SEEDS, dtype={'image_id': str}, usecols=[
        'image_id', 'source_path', 'source_sha256', 'proposed_class', 'source'])
    assert seeds.image_id.is_unique and seeds.proposed_class.value_counts().le(5).all()
    assert seeds.source.eq('user_review_of_displayed_train_image').all()
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = manifest[manifest.split.eq('train') & ~manifest.is_crop].copy().sort_values('parent_id').reset_index(drop=True)
    assert len(complete) == 1906
    frame = complete[complete.training_eligible].copy().reset_index(drop=True)
    frame = frame.rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    targets = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv', dtype={'image_id': str})
    frame = frame.merge(targets[['key', 'group', 'fold']], on='key', validate='one_to_one')
    assert len(frame) == 1904 and frame.key.str.startswith('train:').all()
    idx = dict(zip(frame.key, range(len(frame))))
    seed_positions = np.array([idx['train:' + k] for k in seeds.image_id])
    for row, pos in zip(seeds.itertuples(), seed_positions):
        assert frame.iloc[pos].source_path == row.source_path
        assert row.source_sha256 == frame.iloc[pos].source_sha256 == sha(ROOT / row.source_path)
    old_y = np.array([CLASSES.index(c) for c in frame.assigned_class])
    base = old_y.copy()
    human_mask = np.zeros(len(frame), bool)
    human_mask[seed_positions] = True
    seed_labels = np.array([CLASSES.index(c) for c in seeds.proposed_class])
    base[seed_positions] = seed_labels
    lamp_mask = frame.label_source.eq('two_view_lighting_ontology_snapshot').to_numpy() & ~human_mask
    weights = .25 + .75 * np.clip(frame.category_score.to_numpy(), 0, 1)
    weights[lamp_mask] = 4.
    weights[human_mask] = 32.
    champions = json.loads((NOTES / 'json/noisy_program_frozen.json').read_text())['family_champions']
    plan = {'method': 'bounded_review_revision_crossfit_graph_stability_v1',
            'human_seed_file': str(SEEDS.relative_to(NOTES)), 'human_seed_sha256': sha(SEEDS),
            'manual_unique_source_images_used': len(seeds), 'manual_class_counts': seeds.proposed_class.value_counts().to_dict(),
            'eligible_fit_train_rows': len(frame), 'excluded_hold_keys': ['train:1662', 'train:1885'],
            'test_rows_in_fit': 0, 'all_review_inventory_read_in_fit': False, 'archive_annotations_read_in_fit': False,
            'families': champions, 'prior_weight_regimes': [1., .05], 'human_weight': 32., 'lighting_pseudo_weight': 4.,
            'classifier_folds': 5, 'graph_neighbors': 20, 'graph_alpha': .98,
            'gate': 'same new category in both prior regimes; >=4/5 classifiers and >=2/3 graphs, or >=3/5 and3/3; cosine>=.6 to same-category human anchor in>=2 banks',
            'configuration_selection': 'fixed before fitting, no choice by agreement with unselected reviews',
            'validation_limit': 'leave-source-group-out classifier diagnostic on selected anchors; inherited assignment priors are not independently refit; not unbiased semantic Accuracy',
            'protected_sha256': protected, 'source_code_sha256': sha(__file__)}
    dump(NOTES / 'json/review_revision_plan.json', plan)
    # Once the bounded sources enter fitting they remain spent even if an
    # experiment or publication fails.
    ledger = json.loads((NOTES / 'json/manual_label_budget_status.json').read_text())
    assert set(ledger['manual_unique_source_images_ever_used']) <= set(seeds.image_id)
    ledger.update(manual_labels_consumed_by_current_experiments=len(seeds),
                  manual_unique_source_images_ever_used=sorted(seeds.image_id),
                  manual_class_counts=plan['manual_class_counts'],
                  remaining_by_class={c: 5 - int(seeds.proposed_class.eq(c).sum()) for c in CLASSES},
                  source='participant review revision; fixed bounded representative selection',
                  status='review_revision_fitting_active_processed_still_previous')
    dump(NOTES / 'json/manual_label_budget_status.json', ledger)

    def take(path):
        z = np.load(path, allow_pickle=False)
        lookup = dict(zip(z['keys'], range(len(z['keys']))))
        return z['features'][[lookup[k] for k in frame.key]].astype('float32')

    xd = unit(take(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz'))
    patch = take(OLD / 'output/embeddings/stage5_patch_v1/original224.npz')
    xp = unit(patch * np.repeat(np.array([np.sqrt(.5), .5, .5], dtype='float32'), 384))
    v3 = take(NOTES / '.runtime/dinov3_train_features.npz')
    task = take(NOTES / '.runtime/task8_teacher_features.npz')
    banks = {'dino2': xd, 'dino2_local': xp, 'dino3_global': unit(v3[:, :768]),
             'dino3_foreground': unit(np.c_[v3[:, :384] * np.sqrt(.5), v3[:, 768:1152] * .5, v3[:, 1152:] * .5]),
             'dino3_task8': unit(np.c_[unit(v3[:, :768]), task])}
    groups = frame.group.to_numpy()
    folds = frame.fold.to_numpy()
    human_groups = np.isin(groups, groups[human_mask])
    with threadpool_limits(limits=4):
        similarities = {name: x @ x.T for name, x in banks.items()}
    conflict_cache = {}

    def conflicts(available):
        key = tuple(int(v) for v in available)
        if key not in conflict_cache:
            votes, close = [], []
            for sim in similarities.values():
                nearest = np.stack([sim[:, available[base[available] == c]].max(1)
                                    if np.any(base[available] == c) else np.full(len(frame), -np.inf)
                                    for c in range(8)], axis=1)
                votes.append(nearest.argmax(1))
                close.append(nearest)
            tally = np.stack([(np.stack(votes) == c).sum(0) for c in range(8)], axis=1)
            major = tally.argmax(1)
            nearby = sum(v[np.arange(len(frame)), major] >= .6 for v in close)
            conflict_cache[key] = (tally.max(1) >= 3) & (nearby >= 2) & (major != base) & ~human_mask & ~lamp_mask
        return conflict_cache[key]

    def predict(kernel, tr, va, config, multiplier):
        fit_weights = weights[tr].copy()
        if multiplier != 1.:
            available = seed_positions[np.isin(seed_positions, tr)]
            fit_weights[conflicts(available)[tr]] *= multiplier
        if config['kind'] == 'svm':
            model = SVC(kernel='precomputed', C=config['C'], class_weight='balanced', decision_function_shape='ovr', break_ties=True)
            model.fit(kernel[np.ix_(tr, tr)], base[tr], sample_weight=fit_weights)
            return model.predict(kernel[np.ix_(va, tr)]).astype('int32')
        sim = kernel[np.ix_(va, tr)]
        nn = np.argsort(-sim, axis=1, kind='stable')[:, :config['neighbors']]
        value = np.take_along_axis(sim, nn, axis=1)
        w = np.exp((value - value.max(1, keepdims=True)) / .07) * fit_weights[nn]
        return np.stack([(w * (base[tr][nn] == c)).sum(1) for c in range(8)], axis=1).argmax(1)

    graph_cache = {}
    for name in ['dino2', 'dino3_foreground', 'dino3_task8']:
        sim = similarities[name].copy()
        np.fill_diagonal(sim, -np.inf)
        nn = np.argsort(-sim, axis=1, kind='stable')[:, :20]
        val = np.take_along_axis(sim, nn, axis=1)
        edge = np.maximum(val - .2, 0) ** 4
        a = sparse.csr_matrix((edge.ravel(), (np.repeat(np.arange(len(frame)), 20), nn.ravel())), shape=(len(frame), len(frame)))
        a = (a + a.T) * .5
        denom = np.asarray(a.sum(1)).ravel()
        graph_cache[name] = sparse.diags(1 / denom) @ a

    outputs, holdout_rows = [], []
    with threadpool_limits(limits=4):
        for multiplier in [1., .05]:
            classifier_outputs, graph_outputs = [], []
            for winner in champions:
                name, config = winner['bank'], winner['config']
                sim = similarities[name]
                kernel = sim if config['kind'] == 'knn' or config['kernel'] == 'linear' else np.exp(-2 * config['gamma'] * np.maximum(1 - sim, 0))
                pred = np.full(len(frame), -1, dtype='int32')
                pred[human_mask] = base[human_mask]
                for f in range(5):
                    tr = np.flatnonzero((folds != f) | human_groups)
                    va = np.flatnonzero((folds == f) & ~human_groups)
                    assert not set(groups[tr]) & set(groups[va])
                    if len(va):
                        pred[va] = predict(kernel, tr, va, config, multiplier)
                for g in sorted(set(groups[human_groups & ~human_mask])):
                    tr = np.flatnonzero(groups != g)
                    va = np.flatnonzero((groups == g) & ~human_mask)
                    pred[va] = predict(kernel, tr, va, config, multiplier)
                assert (pred >= 0).all()
                classifier_outputs.append(pred)
                # Diagnostic only, never selects the model or gate.
                for row, pos in zip(seeds.itertuples(), seed_positions):
                    tr = np.flatnonzero(groups != groups[pos])
                    label = int(predict(kernel, tr, np.array([pos]), config, multiplier)[0])
                    holdout_rows.append({'image_id': row.image_id, 'reviewed_class': row.proposed_class,
                                         'bank': name, 'prior_multiplier': multiplier, 'prediction': CLASSES[label]})
                print('Crossfit + anchor diagnostic', multiplier, name, flush=True)
            for name, p in graph_cache.items():
                certainty = np.clip(frame.category_score.to_numpy(), .3, .8)
                if multiplier != 1.:
                    certainty[conflicts(seed_positions)] = .05
                prior = np.eye(8, dtype='float32')[base] * certainty[:, None] + (1 - certainty[:, None]) / 8
                prior[lamp_mask] = np.eye(8)[base[lamp_mask]] * .9 + .1 / 8
                prior[human_mask] = np.eye(8)[base[human_mask]]
                f = prior.copy()
                for step in range(1000):
                    updated = .98 * (p @ f) + .02 * prior
                    updated[human_mask | lamp_mask] = prior[human_mask | lamp_mask]
                    delta = float(np.abs(updated - f).max())
                    f = updated
                    if delta < 1e-6:
                        break
                graph_outputs.append(f.argmax(1))
                print('Train-only graph', multiplier, name, step + 1, flush=True)
            classifiers = np.stack(classifier_outputs)
            graphs = np.stack(graph_outputs)
            tally = np.stack([(classifiers == c).sum(0) for c in range(8)], axis=1)
            major = tally.argmax(1)
            cvotes = tally.max(1)
            gvotes = (graphs == major).sum(0)
            local = np.zeros(len(frame), dtype='int32')
            for sim in similarities.values():
                nearest = np.stack([sim[:, seed_positions[seed_labels == c]].max(1)
                                    if np.any(seed_labels == c) else np.full(len(frame), -np.inf) for c in range(8)], axis=1)
                local += nearest[np.arange(len(frame)), major] >= .6
            gate = (((cvotes >= 4) & (gvotes >= 2)) | ((cvotes >= 3) & (gvotes == 3))) & (local >= 2)
            gate &= ~human_mask & ~lamp_mask & (major != base)
            outputs.append({'major': major, 'gate': gate, 'cvotes': cvotes, 'gvotes': gvotes,
                            'classifiers': classifiers, 'graphs': graphs})
    a, b = outputs
    accepted = a['gate'] & b['gate'] & (a['major'] == b['major'])
    final = base.copy()
    final[accepted] = a['major'][accepted]
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    result['baseline_class'] = complete.assigned_class
    result['predicted_class'] = complete.assigned_class
    result['category_source'] = complete.label_source
    result['category_score'] = complete.category_score
    result['category_score_kind'] = complete.category_score_kind
    result['needs_review'] = complete.needs_review
    result['human_reviewed'] = complete.human_reviewed
    result['propagation_changed'] = False
    result = result.set_index('key')
    for i, key in enumerate(frame.key):
        result.at[key, 'predicted_class'] = CLASSES[final[i]]
        if human_mask[i]:
            result.loc[key, ['category_source', 'category_score', 'category_score_kind', 'human_reviewed', 'needs_review']] = [
                'participant_reviewed_seed', 1., 'human_reviewed_category', True, False]
        elif accepted[i]:
            result.loc[key, ['category_source', 'category_score', 'category_score_kind', 'propagation_changed', 'needs_review']] = [
                'bounded_anchor_stable_consensus_snapshot', min(a['cvotes'][i], b['cvotes'][i]) / 5., 'multi_method_vote_share', True, True]
    assert result.loc[['train:1662', 'train:1885'], 'predicted_class'].equals(result.loc[['train:1662', 'train:1885'], 'baseline_class'])
    out = NOTES / 'table-csv/review_revision_selected.csv'
    result.reset_index().to_csv(out, index=False)
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(NOTES / 'table-csv/review_revision_anchor_diagnostics.csv', index=False)
    pooled = holdout.groupby(['image_id', 'reviewed_class']).prediction.agg(lambda s: s.value_counts().index[0]).reset_index()
    holdout_matches = int(pooled.reviewed_class.eq(pooled.prediction).sum())
    np.savez_compressed(NOTES / '.runtime/review_revision_votes.npz', keys=frame.key.to_numpy(str),
                        classifier_original=a['classifiers'], classifier_cleaned=b['classifiers'],
                        graph_original=a['graphs'], graph_cleaned=b['graphs'], accepted=accepted)
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv', dtype={'image_id': str}).set_index('key')
    report = dict(plan)
    report.update(status='frozen_before_revised_review_audit', prediction_sha256=sha(out),
                  train_rows=1906, changed_rows_vs_baseline=int(result.predicted_class.ne(result.baseline_class).sum()),
                  new_algorithmic_changes=int(accepted.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  lighting_rows=int(result.category_source.eq('two_view_lighting_ontology_snapshot').sum()),
                  stable_automatic_propagation_rows=int(result.category_source.eq('bounded_anchor_stable_consensus_snapshot').sum()),
                  extra_cluster_rows=int(result.category_source.eq('bounded_anchor_cluster_consensus_snapshot').sum()),
                  weak_lighting_rows_restored=0,
                  unstable_proposals_not_applied=int((a['gate'] | b['gate']).sum() - accepted.sum()),
                  human_leave_source_group_out_matches=holdout_matches, human_holdout_rows=len(seeds),
                  semantic_accuracy=None, active_data_modified=False, seconds=time.time() - started)
    assert all(sha(ROOT / k) == v for k, v in protected.items())
    dump(NOTES / 'json/review_revision_frozen.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'manual_unique_source_images_used', 'changed_rows_vs_baseline',
                                            'new_algorithmic_changes', 'human_leave_source_group_out_matches', 'seconds']}, indent=2), flush=True)


def cluster():
    """Recheck consistent multi-anchor clusters, retaining fixed evidence gates."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    info = json.loads((NOTES / 'json/review_revision_frozen.json').read_text())
    source = NOTES / 'table-csv/review_revision_selected.csv'
    assert sha(source) == info['prediction_sha256']
    complete = pd.read_csv(source, dtype={'image_id': str}).set_index('key')
    frame = complete[~complete.image_id.isin(['1662', '1885'])].copy()
    seeds = pd.read_csv(SEEDS, dtype={'image_id': str}, usecols=['image_id', 'proposed_class'])
    lookup = dict(zip(frame.image_id, range(len(frame))))
    positions = np.array([lookup[k] for k in seeds.image_id])
    y = np.array([CLASSES.index(c) for c in seeds.proposed_class])
    plan = {'source_prediction_sha256': sha(source), 'human_seed_sha256': sha(SEEDS),
            'train_rows': len(frame), 'manual_unique_source_images': len(seeds), 'test_rows_in_fit': 0,
            'clusters': [32, 64, 128], 'banks': ['dino2', 'dino3_global', 'dino3_task8'],
            'cluster_category_rule': 'at least two human anchors, all agree, otherwise unnamed',
            'acceptance': 'same proposed category in>=5/9 cluster configurations and all3 banks, no conflicting proposals; both historical-prior regimes graphs3/3; cleaned classifiers>=3/5; preserve first-stage changes and lighting anchors',
            'parameters_fixed_before_posthoc_review': True, 'unselected_review_or_archive_labels_read': False,
            'rejected_anchor_only_probe': '2/34 leave-source-group-out diagnostic versus14/34 for prior-assisted probe; no anchor-only predictions published',
            'source_code_sha256': sha(__file__)}
    dump(NOTES / 'json/review_cluster_plan.json', plan)

    def take(path):
        z = np.load(path, allow_pickle=False)
        ix = dict(zip(z['keys'], range(len(z['keys']))))
        return unit(z['features'][[ix[k] for k in frame.index]].astype('float32'))

    dino2 = take(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz')
    z = np.load(NOTES / '.runtime/dinov3_train_features.npz', allow_pickle=False)
    ix = dict(zip(z['keys'], range(len(z['keys']))))
    d3 = unit(z['features'][[ix[k] for k in frame.index], :768])
    task = take(NOTES / '.runtime/task8_teacher_features.npz')
    banks = {'dino2': dino2, 'dino3_global': d3, 'dino3_task8': unit(np.c_[d3, task])}
    proposals, names, history = [], [], []
    with threadpool_limits(limits=4):
        for name, x in banks.items():
            reduced = unit(PCA(n_components=64, svd_solver='randomized', random_state=20261107).fit_transform(x).astype('float32'))
            for k in [32, 64, 128]:
                groups = KMeans(n_clusters=k, n_init=3, max_iter=200, random_state=20261107).fit_predict(reduced)
                proposal = np.full(len(frame), -1, dtype='int32')
                supported = 0
                for g in range(k):
                    categories = y[groups[positions] == g]
                    if len(categories) >= 2 and len(set(categories.tolist())) == 1:
                        proposal[groups == g] = categories[0]
                        supported += 1
                proposals.append(proposal)
                names.append(name)
                history.append({'bank': name, 'k': k, 'clusters_supported_by_two_consistent_anchors': supported})
                print('Bounded multi-anchor clustering:', name, k, supported, flush=True)
    proposals = np.stack(proposals)
    votes = np.load(NOTES / '.runtime/review_revision_votes.npz', allow_pickle=False)
    assert np.array_equal(votes['keys'], frame.index.to_numpy(str))
    result = complete.copy()
    records = []
    accepted = []
    for i, row in enumerate(frame.itertuples()):
        known = np.flatnonzero(proposals[:, i] >= 0)
        choices = set(proposals[known, i].tolist())
        families = len({names[j] for j in known})
        if len(choices) != 1 or len(known) < 5 or families < 3:
            continue
        label = next(iter(choices))
        original_graph = int((votes['graph_original'][:, i] == label).sum())
        clean_graph = int((votes['graph_cleaned'][:, i] == label).sum())
        clean_classifiers = int((votes['classifier_cleaned'][:, i] == label).sum())
        protected = row.human_reviewed or row.predicted_class != row.baseline_class or row.category_source == 'two_view_lighting_ontology_snapshot'
        apply = not protected and CLASSES[label] != row.predicted_class and original_graph == 3 and clean_graph == 3 and clean_classifiers >= 3
        records.append({'key': row.Index, 'class': CLASSES[label], 'cluster_configuration_support': len(known),
                        'cluster_family_support': families, 'original_graph_support': original_graph,
                        'clean_graph_support': clean_graph, 'clean_classifier_support': clean_classifiers,
                        'protected': protected, 'accepted': apply})
        if apply:
            accepted.append(row.Index)
            result.loc[row.Index, ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
                CLASSES[label], 'bounded_anchor_cluster_consensus_snapshot', clean_classifiers / 5., 'multi_method_vote_share', True]
    out = NOTES / 'table-csv/review_final_selected.csv'
    result.reset_index().to_csv(out, index=False)
    pd.DataFrame(records).to_csv(NOTES / 'table-csv/review_cluster_evidence.csv', index=False)
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv', dtype={'image_id': str}).set_index('key')
    report = dict(info)
    report.update(status='frozen_before_final_revised_review_audit', method='bounded_participant_revision_crossfit_graph_cluster_v2',
                  prediction_sha256=sha(out), cluster_plan_sha256=sha(NOTES / 'json/review_cluster_plan.json'),
                  new_cluster_changes=len(accepted),
                  extra_cluster_rows=int(result.category_source.eq('bounded_anchor_cluster_consensus_snapshot').sum()),
                  changed_rows_vs_baseline=int(result.predicted_class.ne(result.baseline_class).sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  rejected_probe='anchor-only material diffusion; insufficient leave-source-group-out performance; no predictions used',
                  source_counts=result.category_source.value_counts().to_dict(), cluster_configurations=history)
    assert all(sha(ROOT / k) == v for k, v in info['protected_sha256'].items())
    dump(NOTES / 'json/review_final_frozen.json', report)
    print(json.dumps({'new_cluster_changes': len(accepted), 'changed_rows_vs_baseline': report['changed_rows_vs_baseline']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'fit', 'cluster'])
    args = parser.parse_args()
    {'prepare': prepare, 'fit': fit, 'cluster': cluster}[args.mode]()
