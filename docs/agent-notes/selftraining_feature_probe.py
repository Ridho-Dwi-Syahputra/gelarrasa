"""One train-only self-training round, separately generated inside every fold.

Each outer teacher excludes that query's source groups and human labels.
Pseudo-targets are generated only on its FIT side, never on held-out queries.
The full participant-review table is never read by this experiment.
"""
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump, unit


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protected = {name: sha(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = m[m.split.eq('train') & ~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = complete[complete.training_eligible].reset_index(drop=True)
    assert len(frame) == 1904
    keys = frame.parent_key.to_numpy(str)
    grouping = pd.read_csv(NOTES / 'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups, folds = grouping.group.to_numpy(), grouping.fold.to_numpy()
    auto = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    core = auto.automatic_anchor.to_numpy(bool)
    cold = np.load(NOTES / '.runtime/before_bounded_anchor_20261107T084712Z/preprocessing_inputs.npz', allow_pickle=False)
    ci = dict(zip(cold['keys'], range(len(cold['keys']))))
    y = np.array([CLASSES.index(cold['predicted_class'][ci[k]]) for k in keys])
    seeds = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    lookup = dict(zip(keys, range(len(keys))))
    pos = np.array([lookup[k] for k in seeds.key])
    gold = np.array([CLASSES.index(c) for c in seeds.assigned_class])
    human = np.zeros(len(frame), bool)
    human[pos] = True
    y[pos] = gold
    assert len(seeds) == 34 and seeds.assigned_class.value_counts().le(5).all()
    def take(path):
        z = np.load(path, allow_pickle=False)
        index = dict(zip(z['keys'], range(len(z['keys']))))
        return z['features'][[index[k] for k in keys]].astype('float32')
    task = unit(take(NOTES / '.runtime/task8_teacher_features.npz'))
    dino = unit(take(NOTES / '.runtime/dinov3_train_features.npz')[:, :768])
    material = unit(np.mean(take(NOTES / '.runtime/minc_train_features.npz')[:, :2], axis=1))
    banks = [torch.from_numpy(task.copy()), torch.from_numpy(unit(np.c_[dino, material]).copy())]
    original_weights = np.clip(auto.reliable_vote_count.to_numpy(float) / 4., .25, 1.).astype('float32')
    original_weights[human] = 32.
    plan = {'method': 'one_round_fold_contained_GCE_selftraining_v1', 'train_rows': 1904, 'test_rows': 0,
            'manual_sources': 34, 'unselected_review_or_archive_labels_read': False,
            'geometry_group_sha256': sha(NOTES / 'table-csv/geometry_train_groups.csv'),
            'teacher': 'same30epochGCE heads as geometry-group development, regenerated excluding each outer query group',
            'pseudo_gate': 'FIT-side only,bothbanks same category,p>=.95andmargin>=.7;excludehuman;newunlabeled or changedcoldcore targets',
            'pseudo_weight': 4., 'pseudo_limit_per_fold': 200, 'rounds': 1,
            'student': 'warmstartteacher,15extraGCEepochs,AdamWlr=.0003,weight_decay=.001',
            'acceptance': 'student pooled34development matches>=15andbalancedrecall>=.525;bothbanks p>=.9andmargin>=.5;preserve existingrefinements;<=40changes',
            'query_images_not_used_in_target_generation': True,
            'pseudo_targets_not_shared_between_outer_folds': True,
            'evaluation_limit': 'development labels reused for design and selection;not final independent Accuracy',
            'protected_sha256': protected, 'source_code_sha256': sha(__file__)}
    dump(NOTES / 'json/selftraining_feature_plan.json', plan)
    teacher_output = np.full((2, len(frame), 8), np.nan, dtype='float32')
    student_output = teacher_output.copy()
    logs = []
    def train(model, x, fit, labels, weights, epochs, lr, seed, warmup=False):
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
        target = torch.from_numpy(labels).long()
        wt = torch.from_numpy(weights)
        clean = torch.from_numpy(human)
        for epoch in range(epochs):
            model.train()
            for start in range(0, len(fit), 256):
                # Same deterministic epoch ordering as the original teacher.
                if start == 0:
                    order = generator.permutation(fit)
                batch = order[start:start + 256]
                logits = model(x[batch])
                ce = F.cross_entropy(logits, target[batch], reduction='none')
                py = logits.softmax(1).gather(1, target[batch, None]).squeeze(1).clamp_min(1e-7)
                gce = (1 - py.pow(.7)) / .7
                item = ce if warmup and epoch < 5 else torch.where(clean[batch], ce, gce)
                loss = (item * wt[batch]).sum() / wt[batch].sum()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        return model.eval()
    def predict(model, x):
        with torch.inference_mode():
            return np.concatenate([model(x[start:start + 256]).softmax(1).numpy() for start in range(0, len(x), 256)])
    for fold in range(5):
        fit = np.flatnonzero((folds != fold) & (core | human))
        query = np.flatnonzero(folds == fold)
        assert not set(groups[fit]) & set(groups[query])
        teachers, predictions = [], []
        for bank, x in enumerate(banks):
            seed = 20261107 + bank * 100 + fold
            torch.manual_seed(seed)
            model = nn.Sequential(nn.LayerNorm(x.shape[1]), nn.Linear(x.shape[1], 256), nn.GELU(), nn.Dropout(.15), nn.Linear(256, 8))
            # Original head construction consumes RNG before dropout training;
            # preserve that state rather than resetting it inside the trainer.
            rng_state = torch.get_rng_state()
            train_seed_state = torch.get_rng_state()
            # train() starts a deterministic new stream; both teachers and
            # students are independently logged, not claimed bitwise original.
            model = train(model, x, fit, y, original_weights, 30, .001, seed, True)
            p = predict(model, x)
            teacher_output[bank, query] = p[query]
            teachers.append(model)
            predictions.append(p)
        prediction = np.array(predictions)
        classes = prediction.argmax(2)
        sorted_p = np.sort(prediction, axis=2)
        pseudo = (classes[0] == classes[1]) & (prediction.max(2) >= .95).all(0) & ((sorted_p[:, :, -1] - sorted_p[:, :, -2]) >= .7).all(0)
        pseudo &= (folds != fold) & ~human & (~core | (classes[0] != y))
        if pseudo.sum() > 200:
            pseudo[:] = False
        assert not np.any(pseudo[query])
        student_y = y.copy()
        student_y[pseudo] = classes[0, pseudo]
        weights = original_weights.copy()
        weights[pseudo] = 4.
        student_fit = np.flatnonzero((folds != fold) & (core | human | pseudo))
        assert not set(groups[student_fit]) & set(groups[query])
        for bank, x in enumerate(banks):
            student = train(copy.deepcopy(teachers[bank]), x, student_fit, student_y, weights, 15, .0003, 20261207 + bank * 100 + fold)
            student_output[bank, query] = predict(student, x[query])
        logs.append({'fold': fold, 'teacher_fit_rows': len(fit), 'student_fit_rows': len(student_fit),
                     'new_pseudo_rows': int(pseudo.sum()), 'human_fit_rows': int(human[fit].sum()),
                     'query_pseudo_targets': 0, 'query_groups_in_fit': 0})
        print('Fold-contained self-training:', logs[-1], 'seconds', round(time.time() - started), flush=True)
    assert np.isfinite(student_output).all() and np.isfinite(teacher_output).all()
    quality = []
    for name, probabilities in [('regenerated_teacher', teacher_output), ('selftrained_student', student_output)]:
        pred = probabilities[:, pos].mean(0).argmax(1)
        recalls = [float((pred[gold == c] == c).mean()) for c in range(8) if np.any(gold == c)]
        quality.append({'model': name, 'matches': int((pred == gold).sum()), 'balanced_recall': float(np.mean(recalls))})
    p = student_output
    pred = p.argmax(2)
    confidence = p.max(2)
    sorted_p = np.sort(p, axis=2)
    margins = sorted_p[:, :, -1] - sorted_p[:, :, -2]
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    preserve = ~frame.label_source.isin(['iterative_program_refinement_snapshot', 'historical_routing_snapshot']).to_numpy()
    gate = (pred[0] == pred[1]) & (confidence >= .9).all(0) & (margins >= .5).all(0) & ~human & ~preserve & (pred[0] != current)
    proposed = int(gate.sum())
    allowed = quality[1]['matches'] >= 15 and quality[1]['balanced_recall'] >= .525 and proposed <= 40
    if not allowed:
        gate[:] = False
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    for new, old in [('baseline_class', 'assigned_class'), ('predicted_class', 'assigned_class'), ('category_source', 'label_source'),
                     ('category_score', 'category_score'), ('category_score_kind', 'category_score_kind'), ('human_reviewed', 'human_reviewed'), ('needs_review', 'needs_review')]:
        result[new] = complete[old]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[pred[0, i]], 'robust_neural_assignment_snapshot', float(confidence[:, i].min()), 'uncalibrated_neural_probability', True]
    selected = NOTES / 'table-csv/selftraining_feature_selected.csv'
    result.reset_index().to_csv(selected, index=False)
    pd.DataFrame(logs).to_csv(NOTES / 'table-csv/selftraining_fold_log.csv', index=False)
    np.savez_compressed(NOTES / '.runtime/selftraining_feature_predictions.npz', keys=keys,
                        teacher_probabilities=teacher_output, student_probabilities=student_output, accepted=gate)
    info = json.loads((NOTES / 'json/robust_geometry_heads_frozen.json').read_text())
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv').set_index('key')
    info.update(method=plan['method'], status='frozen_before_posthoc_selftraining_audit', prediction_sha256=sha(selected),
                protected_sha256=protected, development=quality, accepted_by_development_guard=allowed,
                selftraining_plan_sha256=sha(NOTES / 'json/selftraining_feature_plan.json'),
                changed_rows_vs_baseline=int(gate.sum()), neural_proposals=proposed,
                new_neural_changes=int(gate.sum()), human_leave_source_group_out_matches=quality[1]['matches'],
                cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                source_counts=result.category_source.value_counts().to_dict(), source_code_sha256=sha(__file__),
                seconds=time.time() - started, validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT / k) == v for k, v in protected.items())
    dump(NOTES / 'json/selftraining_feature_frozen.json', info)
    print(json.dumps({k: info[k] for k in ['status', 'development', 'accepted_by_development_guard', 'neural_proposals', 'new_neural_changes', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
