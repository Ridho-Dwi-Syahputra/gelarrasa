"""Noise-aware neural assignment heads, grouped train-only development.

Uses the already-budgeted34human sources and a cold automatic-consensus core.
Never reads unselected participant review, visual annotation folders or test.
"""
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main(geometry_groups=False, waste_vit=False):
    started = time.time()
    if waste_vit: geometry_groups=True
    prefix = 'vit_neural_heads' if waste_vit else ('robust_geometry_heads' if geometry_groups else 'robust_feature_heads')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protected = {name: sha(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = m[m.split.eq('train') & ~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = complete[complete.training_eligible].reset_index(drop=True)
    assert len(frame) == 1904
    keys = frame.parent_key.to_numpy(str)
    group_info = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    groups, folds = group_info.group.to_numpy(), group_info.fold.to_numpy()
    cold_path = NOTES / '.runtime/before_bounded_anchor_20261107T084712Z/preprocessing_inputs.npz'
    cold = np.load(cold_path, allow_pickle=False)
    assert not any(k.startswith('human_seed') for k in cold.files)
    ci = dict(zip(cold['keys'], range(len(cold['keys']))))
    old_y = np.array([CLASSES.index(cold['predicted_class'][ci[k]]) for k in keys])
    geometry_details = None
    if geometry_groups:
        from sklearn.model_selection import StratifiedGroupKFold
        parent = np.arange(len(frame))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            a, b = find(i), find(j)
            parent[max(a, b)] = min(a, b)
        for old_groups in [groups, frame.parent_group.to_numpy()]:
            first = {}
            for i, group in enumerate(old_groups):
                if group in first:
                    union(i, first[group])
                else:
                    first[group] = i
        edges_path = NOTES / 'table-csv/local_geometry_edges.csv'
        edges = pd.read_csv(edges_path)
        edges = edges[edges.accepted]
        lookup = dict(zip(keys, range(len(keys))))
        for row in edges.itertuples():
            union(lookup[row.key_a], lookup[row.key_b])
        previous_count = len(set(groups))
        groups = np.array([f'geometry_source_{find(i):04d}' for i in range(len(frame))])
        folds = np.full(len(frame), -1, dtype=int)
        for fold, (_, query) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261107).split(np.zeros((len(frame), 1)), old_y, groups)):
            folds[query] = fold
        assert (folds >= 0).all()
        updated = pd.DataFrame({'key': keys, 'group': groups, 'fold': folds, 'source_sha256': frame.source_sha256})
        updated.to_csv(NOTES / 'table-csv/geometry_train_groups.csv', index=False)
        geometry_details = {'old_groups': previous_count, 'new_groups': len(set(groups)),
                            'accepted_edges_sha256': sha(edges_path),
                            'group_manifest_sha256': sha(NOTES / 'table-csv/geometry_train_groups.csv'),
                            'fold_stratification_labels': 'cold automatic labels,not human review'}
        print('Unified source groups:', geometry_details, flush=True)
    seeds = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    index = dict(zip(keys, range(len(keys))))
    pos = np.array([index[k] for k in seeds.key])
    human = np.zeros(len(frame), bool)
    human[pos] = True
    gold = np.array([CLASSES.index(c) for c in seeds.assigned_class])
    assert len(seeds) == 34 and seeds.assigned_class.value_counts().le(5).all()
    y = old_y.copy()
    y[pos] = gold
    core = group_info.automatic_anchor.to_numpy(bool)
    eligible = core | human
    pseudo_weights = np.clip(group_info.reliable_vote_count.to_numpy(float) / 4., .25, 1.)
    def take(path, column='features'):
        z = np.load(path, allow_pickle=False)
        lookup = dict(zip(z['keys'], range(len(z['keys']))))
        return z[column][[lookup[k] for k in keys]].astype('float32')
    task = unit(take(NOTES / '.runtime/task8_teacher_features.npz'))
    dino = unit(take(NOTES / '.runtime/dinov3_train_features.npz')[:, :768])
    material = unit(np.mean(take(NOTES / '.runtime/minc_train_features.npz')[:, :2], axis=1))
    banks = {'task_cnn': task, 'object_material': unit(np.c_[dino, material])}
    vit_source_sha=None
    if waste_vit:
        vit_path=NOTES/'.runtime/waste_vit_train_features.npz'
        vit_info=json.loads((NOTES/'json/waste_vit_feature_extraction.json').read_text())
        assert sha(vit_path)==vit_info['feature_sha256']
        vit=take(vit_path);assert vit.shape==(1904,2304)
        banks={'waste_vit_context':unit(np.c_[unit(vit[:,:768]),unit(vit[:,768:1536]),unit(vit[:,1536:])]),
               'waste_vit_object':unit(np.c_[unit(vit[:,:768]),dino])}
        vit_source_sha=sha(vit_path)
    minimum_matches=16 if waste_vit else 14
    plan = {'method': 'local_waste_ViT_neural_GCE_v1' if waste_vit else ('geometry_grouped_neural_GCE_v1' if geometry_groups else 'grouped_neural_heads_with_generalized_cross_entropy_v1'),
            'train_rows': 1904, 'test_rows_in_fit': 0, 'held_sources_in_fit': 0,
            'manual_sources': 34, 'manual_counts': seeds.assigned_class.value_counts().to_dict(),
            'automatic_core_rows': int(core.sum()), 'fit_target_source_rows': int(eligible.sum()),
            'unselected_review_or_archive_labels_read': False, 'cold_prior_sha256': sha(cold_path),
            'features': {name: x.shape[1] for name, x in banks.items()},
            'geometry_source_groups': geometry_details,
            'waste_vit_feature_sha256':vit_source_sha,
            'architecture': 'LayerNorm(input),Linear256,GELU,Dropout.15,Linear8',
            'epochs': 30, 'warmup_epochs_CE': 5, 'batch_size': 256,
            'optimizer': 'AdamW,lr=.001,weight_decay=.001', 'q': .7,
            'human_loss': 'ordinary cross entropy,weight32',
            'pseudo_loss': 'CEcontrol orGCE=(1-p_y**q)/q after5warmup epochs',
            'selection': 'objective with best pooled balanced recall on34group-heldout human anchors; same folds for both feature banks',
            'folds': 5, 'same_group_fit_query_overlap': False,
            'acceptance': f'selected objective recovers>={minimum_matches}/34development anchors; both banks same new category with probabilities>=.85 and margins>=.4; protect allhuman/earlierrefinement rows; maximum40changes,else none',
            'evaluation_limit': '34selecteddevelopment anchors,used for objective selection,not independent final Accuracy',
            'paper': 'https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf',
            'protected_sha256': protected, 'source_code_sha256': sha(__file__)}
    dump(NOTES / f'json/{prefix}_plan.json', plan)
    target = torch.from_numpy(y).long()
    is_human = torch.from_numpy(human)
    base_weights = torch.from_numpy(pseudo_weights.astype('float32'))
    output, history = [], []
    for bank_number, (name, values) in enumerate(banks.items()):
        x = torch.from_numpy(values.copy())
        objective_outputs = []
        for objective in ['CE', 'GCE']:
            probability = np.full((len(frame), 8), np.nan, dtype='float32')
            for fold in range(5):
                fit = np.flatnonzero((folds != fold) & eligible)
                query = np.flatnonzero(folds == fold)
                assert not set(groups[fit]) & set(groups[query])
                torch.manual_seed(20261107 + bank_number * 100 + fold)
                generator = np.random.default_rng(20261107 + bank_number * 100 + fold)
                model = nn.Sequential(nn.LayerNorm(x.shape[1]), nn.Linear(x.shape[1], 256), nn.GELU(), nn.Dropout(.15), nn.Linear(256, 8))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
                train_weights = base_weights.clone()
                train_weights[is_human] = 32.
                losses = []
                for epoch in range(30):
                    model.train()
                    order = generator.permutation(fit)
                    total, denom = 0., 0.
                    for start in range(0, len(order), 256):
                        batch = order[start:start + 256]
                        logits = model(x[batch])
                        ce = F.cross_entropy(logits, target[batch], reduction='none')
                        if objective == 'GCE' and epoch >= 5:
                            py = logits.softmax(1).gather(1, target[batch, None]).squeeze(1).clamp_min(1e-7)
                            pseudo = (1 - py.pow(.7)) / .7
                            loss_items = torch.where(is_human[batch], ce, pseudo)
                        else:
                            loss_items = ce
                        w = train_weights[batch]
                        loss = (loss_items * w).sum() / w.sum()
                        assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        total += float(loss.detach()) * len(batch)
                        denom += len(batch)
                    losses.append(total / denom)
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(query), 256):
                        q = query[start:start + 256]
                        probability[q] = model(x[q]).softmax(1).numpy()
                history.append({'bank': name, 'objective': objective, 'fold': fold, 'fit_rows': len(fit),
                                'query_rows': len(query), 'human_fit_rows': int(human[fit].sum()),
                                'first_epoch_loss': losses[0], 'last_epoch_loss': losses[-1]})
                print('Neural assignment head:', name, objective, 'fold', fold + 1, '/5', 'seconds', round(time.time() - started), flush=True)
            assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1., atol=1e-5)
            objective_outputs.append(probability)
        output.append(objective_outputs)
    output = np.array(output)  # bank,objective,row,category
    quality = []
    for objective in range(2):
        pooled = output[:, objective, pos].mean(0).argmax(1)
        recalls = [float((pooled[gold == c] == c).mean()) for c in range(8) if np.any(gold == c)]
        quality.append({'objective': ['CE', 'GCE'][objective], 'matches': int((pooled == gold).sum()), 'balanced_recall': float(np.mean(recalls))})
    choice = max(range(2), key=lambda i: quality[i]['balanced_recall'])
    selected = output[:, choice]
    pred = selected.argmax(2)
    confidence = selected.max(2)
    sorted_prob = np.sort(selected, axis=2)
    margin = sorted_prob[:, :, -1] - sorted_prob[:, :, -2]
    candidate = pred[0]
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    preserve = ~frame.label_source.isin(['iterative_program_refinement_snapshot', 'historical_routing_snapshot']).to_numpy()
    gate = (pred[0] == pred[1]) & (confidence >= .85).all(0) & (margin >= .4).all(0) & ~human & ~preserve & (candidate != current)
    proposals = int(gate.sum())
    accepted = quality[choice]['matches'] >= minimum_matches and proposals <= 40
    if not accepted:
        gate[:] = False
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    for new, old in [('baseline_class', 'assigned_class'), ('predicted_class', 'assigned_class'), ('category_source', 'label_source'),
                     ('category_score', 'category_score'), ('category_score_kind', 'category_score_kind'), ('human_reviewed', 'human_reviewed'), ('needs_review', 'needs_review')]:
        result[new] = complete[old]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[candidate[i]], 'robust_neural_assignment_snapshot', float(confidence[:, i].min()), 'uncalibrated_neural_probability', True]
    destination = NOTES / f'table-csv/{prefix}_selected.csv'
    result.reset_index().to_csv(destination, index=False)
    pd.DataFrame(history).to_csv(NOTES / f'table-csv/{prefix}_training.csv', index=False)
    pd.DataFrame(quality).to_csv(NOTES / f'table-csv/{prefix}_development.csv', index=False)
    np.savez_compressed(NOTES / f'.runtime/{prefix}_predictions.npz', keys=keys, probabilities=output, accepted=gate)
    prior_info = json.loads((NOTES / 'json/review_final_frozen.json').read_text())
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv').set_index('key')
    report = dict(prior_info)
    report.update(status='frozen_before_posthoc_robust_head_audit', method=plan['method'], prediction_sha256=sha(destination),
                  protected_sha256=protected, human_leave_source_group_out_matches=quality[choice]['matches'], human_holdout_rows=34,
                  human_diagnostic_protocol='five-source-group-fold development,not leave-one-anchor-out',
                  objective_development=quality, selected_objective=quality[choice]['objective'],
                  robust_plan_sha256=sha(NOTES / f'json/{prefix}_plan.json'),
                  geometry_source_groups=geometry_details,
                  waste_vit_feature_sha256=vit_source_sha,
                  changed_rows_vs_baseline=int(gate.sum()), neural_proposals=proposals, accepted_by_development_guard=accepted,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(), validation_limit=plan['evaluation_limit'],
                  source_code_sha256=sha(__file__), seconds=time.time() - started, test_rows_in_fit=0,
                  eligible_fit_train_rows=1904, new_neural_changes=int(gate.sum()))
    assert all(sha(ROOT / name) == value for name, value in protected.items())
    dump(NOTES / f'json/{prefix}_frozen.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'objective_development', 'selected_objective', 'neural_proposals',
                                            'accepted_by_development_guard', 'new_neural_changes', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-groups', action='store_true')
    parser.add_argument('--waste-vit', action='store_true')
    args=parser.parse_args()
    main(args.geometry_groups,args.waste_vit)
