"""Human-source-only assignment probe; no archived reviews/pseudo targets.

The 34 labels are development data reused for selection, not an independent test.
Every outer query source group is excluded from graph/normalizer/SVM fitting.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from scipy.sparse.linalg import splu
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    start = time.time()
    prefix = 'human_manifold'
    protected = {p: sha(ROOT/p) for p in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    plan = {
        'method': 'human_source_only_kernel_and_manifold_v1',
        'targets': 'ONLY existing34participant_source_labels;no automatic categories as targets',
        'new_manual_labels': 0, 'test_rows_in_fit': 0,
        'banks': ['base_global', 'base_regional', 'cnn_base'],
        'svm': [{'kernel': k, 'C': c, 'views': v} for k, c in [('linear', 1.), ('rbf', 10.)] for v in [1, 8]],
        'rbf_gamma': 4., 'svm_scores': 'softmax(OVRdecision/.5),uncalibrated',
        'manifold': [{'neighbors': k, 'alpha': a} for k in [10, 30] for a in [.95, .995]],
        'manifold_edges': 'symmetrizedpositivecosine^8,normalizedLaplacian',
        'manifold_targets': '1/class_human_count;hardclamp;no expected dataset class proportions',
        'manifold_query': '20trainnearestneighbors,softmaxcosine/.07weightedprobabilities',
        'folds': 'fiveexistinggeometrysourcegroups;outerqueryNOTingraph;allsourceviewsheldout',
        'normalizer': 'centeringonouterTRAINonlythenrowL2',
        'selection': 'maximum34developmentbalancedrecall,thenmatches,thenlistedorder',
        'threshold_grid': [.7, .8, .9, .95, .98, .995],
        'threshold_selection': 'maximumcoveragewith>=10coveredand>=.9precision;tiesstricter',
        'guard': '>=25/34andmacro>.75+1e-12andthresholdexists',
        'acceptance': 'learnedthresholdand>=2of3independentfeaturebankbestheadsagree;humanprotected',
        'maximum_changes': 40, 'cap': 'confidence descending,stablekeyorder',
        'private_review_or_archive_read': False, 'protected_sha256': protected,
        'evaluation_limit': 'Repeated34developmentselection;notindependentAccuracy.Foodwastehasonehuman source,sothefoldholdingitoutcannotlearnthatclass.',
        'source_code_sha256': sha(__file__)
    }
    dump(NOTES/f'json/{prefix}_plan.json', plan)
    m = pd.read_csv(ROOT/'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = m[m.split.eq('train') & ~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = complete[complete.training_eligible].reset_index(drop=True)
    keys = frame.parent_key.to_numpy(str); n = len(keys); assert n == 1904
    s = pd.read_csv(ROOT/'data/processed/manual_seeds.csv'); assert len(s) == 34 and s.assigned_class.value_counts().le(5).all()
    lookup = dict(zip(keys, range(n))); pos = np.array([lookup[k] for k in s.key]); gold = np.array([CLASSES.index(c) for c in s.assigned_class])
    human = np.zeros(n, bool); human[pos] = True
    gf = pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups, folds = gf.group.to_numpy(), gf.fold.to_numpy()
    b = np.load(NOTES/'.runtime/dinov2_base_train_features.npz', allow_pickle=False)
    c = np.load(NOTES/'.runtime/convnext_train_features.npz', allow_pickle=False)
    aug = np.load(NOTES/'.runtime/human_source_aug_features.npz', allow_pickle=False)
    assert np.array_equal(b['keys'], keys) and np.array_equal(c['keys'], keys) and np.array_equal(aug['keys'], s.key.to_numpy(str))
    bg = unit(b['features'][:, :768]); ag = unit(aug['features'][:, :, :768]); cg = unit(c['features']); ac = unit(aug['cnn_features'])
    banks = {
        'base_global': (bg, ag),
        'base_regional': (unit(np.c_[bg, unit(b['features'][:, 2304:])]), unit(np.concatenate([ag, unit(aug['features'][:, :, 2304:])], -1))),
        'cnn_base': (unit(np.c_[cg, bg]), unit(np.concatenate([ac, ag], -1)))
    }
    history = []; outputs = {}; configurations = {}; final_cache = {}

    def svm_fit(x, views, source_indices, query, cfg):
        v = views[source_indices, :cfg['views']]; nv = v.shape[1]
        xx = v.reshape(-1, x.shape[1]); y = np.repeat(gold[source_indices], nv)
        kt = xx@xx.T; kq = x[query]@xx.T
        if cfg['kernel'] == 'rbf':
            kt = np.exp(-8*np.maximum(1-kt, 0)); kq = np.exp(-8*np.maximum(1-kq, 0))
        model = SVC(kernel='precomputed', C=cfg['C'], class_weight='balanced', decision_function_shape='ovr', break_ties=True)
        model.fit(kt, y, sample_weight=np.full(len(y), 1/nv))
        scores = model.decision_function(kq)
        if len(model.classes_) == 2: scores = np.stack([-scores, scores], 1)
        q = np.zeros((len(query), 8)); q[:, model.classes_] = softmax(scores/.5, axis=1)
        return q

    def manifold_fit(x, fit, seed_sources, query, cfg):
        xf = x[fit]; sim = xf@xf.T; np.fill_diagonal(sim, -np.inf)
        nn = np.argsort(-sim, axis=1, kind='stable')[:, :cfg['neighbors']]
        w = np.maximum(np.take_along_axis(sim, nn, axis=1), 0)**8
        adj = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(len(fit)), cfg['neighbors']), nn.ravel())), shape=(len(fit), len(fit)))
        adj = (adj+adj.T)*.5
        degree = np.asarray(adj.sum(1)).ravel(); norm = sparse.diags(1/np.sqrt(np.maximum(degree, 1e-12))); trans = norm@adj@norm
        local_lookup = dict(zip(fit, range(len(fit)))); hp = np.array([local_lookup[i] for i in pos[seed_sources]])
        hy = gold[seed_sources]; counts = np.bincount(hy, minlength=8)
        target = np.zeros((len(fit), 8)); target[hp, hy] = 1/np.maximum(counts[hy], 1)
        unl = np.flatnonzero(~np.isin(np.arange(len(fit)), hp)); alpha = cfg['alpha']
        rhs = alpha*trans[unl][:, hp]@target[hp]
        solution = splu((sparse.eye(len(unl))-alpha*trans[unl][:, unl]).tocsc()).solve(rhs)
        p = target.copy(); p[unl] = np.maximum(solution, 0); p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        if np.array_equal(fit, query): return p
        similarity = x[query]@xf.T; neighbors = np.argsort(-similarity, axis=1, kind='stable')[:, :20]
        values = np.take_along_axis(similarity, neighbors, 1); weight = softmax(values/.07, axis=1)
        q = (p[neighbors]*weight[:, :, None]).sum(1); return q/np.maximum(q.sum(1, keepdims=True), 1e-12)

    with threadpool_limits(limits=3):
        for bank, (raw, raw_views) in banks.items():
            specs = [('svm', cfg) for cfg in plan['svm']] + [('manifold', cfg) for cfg in plan['manifold']]
            for number, (kind, cfg) in enumerate(specs):
                name = f'{bank}_{kind}_{number}'; p = np.zeros((n, 8))
                for fold in range(5):
                    fit = np.flatnonzero(folds != fold); query = np.flatnonzero(folds == fold); seeds = np.flatnonzero(folds[pos] != fold)
                    assert not set(groups[fit]) & set(groups[query])
                    mean = raw[fit].mean(0); x = unit(raw-mean); views = unit(raw_views-mean)
                    if kind == 'svm': p[query] = svm_fit(x, views, seeds, query, cfg)
                    else: p[query] = manifold_fit(x, fit, seeds, query, cfg)
                macro = float(balanced_accuracy_score(gold, p[pos].argmax(1))); matches = int((gold == p[pos].argmax(1)).sum())
                outputs[name] = p; configurations[name] = (bank, kind, cfg)
                history.append({'name': name, 'bank': bank, 'kind': kind, **cfg, 'matches': matches, 'balanced_recall': macro})
            print('Human-only bank complete', bank, 'best', max([r for r in history if r['bank']==bank], key=lambda r:(r['balanced_recall'],r['matches'])), 'seconds', round(time.time()-start), flush=True)
        winner = max(history, key=lambda r:(r['balanced_recall'], r['matches']))
        bank_winners = [max([r for r in history if r['bank']==bank], key=lambda r:(r['balanced_recall'],r['matches'])) for bank in banks]

        def final(name):
            if name not in final_cache:
                bank, kind, cfg = configurations[name]; raw, raw_views = banks[bank]
                mean = raw.mean(0); x = unit(raw-mean); views = unit(raw_views-mean); all_rows = np.arange(n); all_seeds = np.arange(34)
                if kind == 'svm': final_cache[name] = svm_fit(x, views, all_seeds, all_rows, cfg)
                else: final_cache[name] = manifold_fit(x, all_rows, all_seeds, all_rows, cfg)
            return final_cache[name]

        oof = outputs[winner['name']]; calibration = []
        for t in plan['threshold_grid']:
            covered = oof[pos].max(1) >= t; count = int(covered.sum()); good = int(((oof[pos].argmax(1)==gold)&covered).sum())
            calibration.append({'threshold': t, 'covered': count, 'correct': good, 'precision': good/count if count else None, 'eligible': count>=10 and good/count>=.9})
        feasible = [r for r in calibration if r['eligible']]; cal = max(feasible, key=lambda r:(r['covered'],r['threshold'])) if feasible else None
        accepted = bool(winner['matches'] >= 25 and winner['balanced_recall'] > .75+1e-12 and cal is not None)
        p = final(winner['name']); pred = p.argmax(1); score = p.max(1)
        bank_pred = np.stack([final(r['name']).argmax(1) for r in bank_winners]); support = (bank_pred == pred).sum(0)
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate = ~human & (pred != current) & (support >= 2) & (score >= (cal['threshold'] if cal else np.inf))
    proposals = int(gate.sum()); order = np.argsort(-score, kind='stable'); chosen = order[gate[order]][:40]; gate[:] = False; gate[chosen] = True
    if not accepted: gate[:] = False
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key':'key', 'parent_id':'image_id'}).copy()
    for dest, src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]: result[dest] = complete[src]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class','category_source','category_score','category_score_kind','needs_review']] = [CLASSES[pred[i]], 'human_source_manifold_assignment_snapshot', score[i], 'uncalibrated_graph_or_kernel_probability', True]
    dest = NOTES/f'table-csv/{prefix}_selected.csv'; result.reset_index().to_csv(dest, index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv', index=False)
    pd.DataFrame({'key':keys, 'candidate_class':np.array(CLASSES)[pred], 'score':score, 'feature_bank_support':support, 'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz', keys=keys, probabilities=p, oof_probabilities=oof, bank_predictions=bank_pred, accepted=gate)
    active = json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix']
    report = json.loads((NOTES/f'json/{active}_frozen.json').read_text())
    initial = pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_human_manifold_audit', method=plan['method'], prediction_sha256=sha(dest), protected_sha256=protected,
        primary_head=winner, primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'), selected_heads=bank_winners, selected_knn_heads=[], qualified_heads=bank_winners,
        human_leave_source_group_out_matches=winner['matches'], human_holdout_rows=34, development_balanced_recall=winner['balanced_recall'],
        human_diagnostic_protocol=plan['folds'], confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
        accepted_by_development_guard=accepted, proposed_new_labels=proposals, changed_rows_vs_baseline=int(gate.sum()), new_neural_changes=0, new_vit_propagation_changes=int(gate.sum()),
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()), source_counts=result.category_source.value_counts().to_dict(),
        source_code_sha256=sha(__file__), validation_limit=plan['evaluation_limit'], seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json', report)
    dump(NOTES/f'json/{prefix}_publication_decision.json', {'decision':'publish' if accepted else 'reject','accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds']}, indent=2),flush=True)


if __name__ == '__main__': main()
