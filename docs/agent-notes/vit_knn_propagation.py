"""Train-only waste-ViT kNN and propagation under7November official clarification.

Only the34budgeted human labels and cold automatic-consensus targets are used.
No remaining-case list, full participant review or visual annotation is read.
"""
import json
import argparse
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main(mode='full'):
    started = time.time()
    is_base=mode in ['base','base_clean']
    is_cnn=mode in ['cnn','cnn_clean']
    protected = {name: sha(ROOT / name) for name in ['data/processed/manifest.csv', 'data/preprocessing_inputs.npz', 'final-submit/DSC2026_Notebook.ipynb']}
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = manifest[manifest.split.eq('train') & ~manifest.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = complete[complete.training_eligible].reset_index(drop=True)
    assert len(frame) == 1904
    keys = frame.parent_key.to_numpy(str)
    prefix={'full':'vit_propagation','detector':'detector_propagation','focus':'focus_propagation','base':'base_propagation','base_clean':'base_clean_propagation','cnn':'cnn_propagation','cnn_clean':'cnn_clean_propagation'}[mode]
    cache = NOTES / ('.runtime/convnext_train_features.npz' if is_cnn else ('.runtime/dinov2_base_train_features.npz' if is_base else ('.runtime/object_focus_vit_features.npz' if mode=='focus' else '.runtime/waste_vit_train_features.npz')))
    extraction = json.loads((NOTES / ('json/convnext_feature_extraction.json' if is_cnn else ('json/dinov2_base_feature_extraction.json' if is_base else ('json/object_focus_vit_extraction.json' if mode=='focus' else 'json/waste_vit_feature_extraction.json')))).read_text())
    assert sha(cache) == extraction['feature_sha256']
    z = np.load(cache, allow_pickle=False)
    assert np.array_equal(z['keys'], keys) and np.array_equal(z['source_sha256'], frame.source_sha256.to_numpy(str))
    f = z['features']
    assert f.shape == (1904, 768 if is_cnn else (3072 if is_base else 2304))
    old = np.load(OLD / 'output/embeddings/stage2_repr_v1/dinov2_features.npz', allow_pickle=False)
    lookup = dict(zip(old['keys'], range(len(old['keys']))))
    dino = unit(old['features'][[lookup[k] for k in keys]])
    cls = unit(f[:, :768])
    texture = unit(np.c_[unit(f[:, 768:1536]), unit(f[:, 1536:])])
    banks = {} if is_base or is_cnn else {'vit_cls': cls, 'vit_texture': texture, 'vit_dino': unit(np.c_[cls, dino]),
             'vit_probability_features': unit(np.sqrt(np.maximum(z['probabilities'], 1e-8)))}
    localisation_sha=None
    if is_cnn:
        bf=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False)
        vf=np.load(NOTES/'.runtime/waste_vit_train_features.npz',allow_pickle=False)
        assert np.array_equal(bf['keys'],keys) and np.array_equal(vf['keys'],keys)
        bc=unit(bf['features'][:,:768]);roi=unit(bf['features'][:,2304:]);vc=unit(vf['features'][:,:768])
        banks={'cnn':cls,'cnn_base':unit(np.c_[cls,bc]),'cnn_base_roi':unit(np.c_[cls,bc,roi]),'cnn_base_vit':unit(np.c_[cls,bc,vc])}
    if mode in ['detector','focus','base','base_clean']:
        local_path=NOTES/'.runtime/litter_object_features.npz'
        local_info=json.loads((NOTES/'json/litter_object_extraction.json').read_text())
        assert sha(local_path)==local_info['feature_sha256']
        local=np.load(local_path,allow_pickle=False)
        assert np.array_equal(local['keys'],keys)
        lf=local['features'];assert lf.shape==(1904,2716)
        localisation_sha=sha(local_path)
        if mode=='detector':
            banks={'detector_global':unit(lf[:,:1792]),'detector_foreground':unit(lf[:,1792:2688]),
                   'detector_vit':unit(np.c_[unit(lf[:,:2688]),cls]),
                   'detector_vit_dino':unit(np.c_[cls,dino,unit(lf[:,:2688])])}
        elif mode=='focus':
            full=np.load(NOTES/'.runtime/waste_vit_train_features.npz',allow_pickle=False)
            assert np.array_equal(full['keys'],keys)
            full_cls=unit(full['features'][:,:768])
            banks={'focus_cls':cls,'full_focus':unit(np.c_[full_cls,cls]),
                   'focus_dino':unit(np.c_[cls,dino]),'focus_detector':unit(np.c_[cls,unit(lf[:,:2688])])}
        else:
            full=np.load(NOTES/'.runtime/waste_vit_train_features.npz',allow_pickle=False)
            assert np.array_equal(full['keys'],keys)
            full_cls=unit(full['features'][:,:768])
            banks={'base_cls':cls,'base_local':unit(np.c_[cls,unit(f[:,2304:])]),
                   'base_vit':unit(np.c_[cls,full_cls]),'base_vit_detector':unit(np.c_[cls,full_cls,unit(lf[:,:2688])])}
    graph_banks=list(banks)[:3]
    auto = pd.read_csv(NOTES / 'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    grouping = pd.read_csv(NOTES / 'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups, folds = grouping.group.to_numpy(), grouping.fold.to_numpy()
    seeds = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    index = dict(zip(keys, range(len(keys))))
    pos = np.array([index[k] for k in seeds.key])
    gold = np.array([CLASSES.index(c) for c in seeds.assigned_class])
    human = np.zeros(len(frame), bool)
    human[pos] = True
    y = np.array([CLASSES.index(c) for c in auto.automatic_class])
    y[pos] = gold
    core = auto.automatic_anchor.to_numpy(bool)
    fit_ok = core | human
    base_weight = np.clip(auto.reliable_vote_count.to_numpy(float) / 4., .25, 1.)
    configs = [{'neighbors': k, 'human_weight': w} for k in [3, 7, 15] for w in [8., 32.]]
    plan = {'method': f'train_only_{mode}_object_ViT_knn_label_propagation_v1', 'train_rows': 1904, 'test_rows': 0,
            'held_sources_excluded': ['train:1662', 'train:1885'], 'human_sources': 34,
            'human_class_counts': seeds.assigned_class.value_counts().to_dict(), 'new_human_labels': 0,
            'source_feature_sha256': sha(cache), 'group_manifest_sha256': sha(NOTES / 'table-csv/geometry_train_groups.csv'),
            'localisation_feature_sha256':localisation_sha,'feature_mode':mode,
            'source_qna': 'https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/kxodWG5OBD7gQgP7',
            'source_qna_reply_utc': '2026-11-07T13:25:19.146Z',
            'targets': 'only cold automatic-consensus core and34budgeted human sources',
            'source_taxonomy_probabilities': 'image-only self-supervised DINOv2Base and descriptor banks;no sourceclassifier-to-targetmapping' if is_base else 'numeric features only,no manual12-to8mapping and no direct pretrained-to-test use',
            'banks': {name: x.shape[1] for name, x in banks.items()}, 'configs': configs,
            'selection': 'per-bank best balanced recall on34five-source-group-fold development anchors; tie keeps config order',
            'knn_temperature': .07, 'graph_banks': graph_banks,
            'graph': {'neighbors': 20, 'alpha': .98, 'iterations': 1500, 'human_seeds_clamped': True, 'automatic_priors_not_clamped': True},
            'acceptance': '>=3/4kNNbanks agree,hybridandonevisualbankagree;>=2/3graphsagree;cosine>=.65andcategorymargin>=.01towardhumananchorsin>=2banks;protect earlierrefinements',
            'guard': 'pooled34anchor matches>=16,balancedrecall>=.525,<=50newcategories;otherwise no changes accepted',
            'evaluation_limit': 'development anchors reused for configuration choice;graph sees allTRAIN and is not independent validation',
            'automatic_core_filter': 'within each fit fold only:exclude automatic anchors whose nearest human category differs,cosine>=.65andcategorymargin>=.02;never excludehuman;do notchangeanytarget' if mode in ['base_clean','cnn_clean'] else None,
            'previous_automatic_refinement_policy': 'may change only under4/4bank,3/3graphand>=3localitysupport;humanalwaysprotected' if is_base or is_cnn else 'protected',
            'unselected_review_or_archive_labels_read': False, 'protected_sha256': protected, 'source_code_sha256': sha(__file__)}
    dump(NOTES / f'json/{prefix}_plan.json', plan)
    predictions, probabilities, holdouts, similarities, history, selections = [], [], [], {}, [], []
    cleaning_history={}

    def vote(sim, fit, query, config):
        if mode in ['base_clean','cnn_clean']:
            anchors=fit[human[fit]]
            nearest=np.stack([sim[np.ix_(fit,anchors[y[anchors]==c])].max(1) if np.any(y[anchors]==c) else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
            category=nearest.argmax(1);value=nearest[np.arange(len(fit)),category].copy()
            nearest[np.arange(len(fit)),category]=-np.inf
            remove=(category!=y[fit])&(value>=.65)&(value-nearest.max(1)>=.02)&~human[fit]
            assert not (remove&human[fit]).any()
            cleaning_history[f'{name}:queryfold{int(folds[query[0]])}']={'before':len(fit),'excluded_automatic':int(remove.sum()),
                'human_retained':int(human[fit].sum()),'class_counts_after':dict(zip(CLASSES,np.bincount(y[fit[~remove]],minlength=8).tolist()))}
            fit=fit[~remove]
        sub = sim[np.ix_(query, fit)]
        nearest = np.argsort(-sub, axis=1, kind='stable')[:, :config['neighbors']]
        values = np.take_along_axis(sub, nearest, axis=1)
        weights = base_weight[fit].copy()
        weights[human[fit]] = config['human_weight']
        w = np.exp((values - values.max(1, keepdims=True)) / .07) * weights[nearest]
        p = np.stack([(w * (y[fit][nearest] == c)).sum(1) for c in range(8)], axis=1)
        return p / p.sum(1, keepdims=True)

    with threadpool_limits(limits=4):
        for name, x in banks.items():
            sim = x @ x.T
            similarities[name] = sim
            best, best_score, best_holdout = None, -np.inf, None
            for config in configs:
                hp = np.full((len(pos), 8), np.nan, dtype='float32')
                for fold in range(5):
                    fit = np.flatnonzero((folds != fold) & fit_ok)
                    query_seed = np.flatnonzero(folds[pos] == fold)
                    assert not set(groups[fit]) & set(groups[pos[query_seed]])
                    if len(query_seed):
                        hp[query_seed] = vote(sim, fit, pos[query_seed], config)
                quality = float(balanced_accuracy_score(gold, hp.argmax(1)))
                history.append({'bank': name, **config, 'development_balanced_recall': quality,
                                'anchor_matches': int((hp.argmax(1) == gold).sum())})
                if quality > best_score:
                    best, best_score, best_holdout = config, quality, hp
            holdouts.append(best_holdout)
            selections.append({'bank': name, **best, 'development_balanced_recall': best_score})
            p = np.full((len(frame), 8), np.nan, dtype='float32')
            # Grouped out-of-fold query predictions for all train rows. Human
            # labels are not overridden here; their held-out outputs diagnose.
            for fold in range(5):
                fit = np.flatnonzero((folds != fold) & fit_ok)
                query = np.flatnonzero(folds == fold)
                assert not set(groups[fit]) & set(groups[query])
                p[query] = vote(sim, fit, query, best)
            assert np.isfinite(p).all()
            probabilities.append(p)
            predictions.append(p.argmax(1))
            print('ViT kNN family selected:', selections[-1], flush=True)
        graph_predictions, graph_history = [], []
        for name in graph_banks:
            sim = similarities[name].copy()
            np.fill_diagonal(sim, -np.inf)
            nn = np.argsort(-sim, axis=1, kind='stable')[:, :20]
            edge = np.maximum(np.take_along_axis(sim, nn, axis=1), 0) ** 4
            a = sparse.csr_matrix((edge.ravel(), (np.repeat(np.arange(len(frame)), 20), nn.ravel())), shape=(len(frame), len(frame)))
            a = (a + a.T) * .5
            pgraph = sparse.diags(1 / np.asarray(a.sum(1)).ravel()) @ a
            prior = probabilities[list(banks).index(name)].copy()
            prior = .8 * prior + .2 / 8
            prior[pos] = np.eye(8)[gold]
            state = prior.copy()
            for step in range(1500):
                update = .98 * (pgraph @ state) + .02 * prior
                update[pos] = prior[pos]
                delta = float(np.abs(update - state).max())
                state = update
                if delta < 1e-6:
                    break
            graph_predictions.append(state.argmax(1))
            graph_history.append({'bank': name, 'iterations': step + 1, 'delta': delta})
    pred = np.array(predictions)
    graphs = np.array(graph_predictions)
    pooled = np.mean(holdouts, axis=0).argmax(1)
    dev_matches = int((pooled == gold).sum())
    dev_balanced = float(balanced_accuracy_score(gold, pooled))
    tally = np.stack([(pred == c).sum(0) for c in range(8)], axis=1)
    major, support = tally.argmax(1), tally.max(1)
    graph_support = (graphs == major).sum(0)
    local = np.zeros(len(frame), dtype=int)
    for sim in similarities.values():
        nearest = np.stack([sim[:, pos[gold == c]].max(1) if np.any(gold == c) else np.full(len(frame), -np.inf) for c in range(8)], axis=1)
        value = nearest[np.arange(len(frame)), major]
        nearest[np.arange(len(frame)), major] = -np.inf
        local += (value >= .65) & (value - nearest.max(1) >= .01)
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    preserve = ~frame.label_source.isin(['iterative_program_refinement_snapshot', 'historical_routing_snapshot']).to_numpy()
    gate = (support >= 3) & (pred[2] == major) & ((pred[:2] == major).any(0)) & (graph_support >= 2) & (local >= 2)
    previous_allowed=(support>=4)&(graph_support>=3)&(local>=3) if is_base or is_cnn else np.zeros(len(frame),bool)
    gate &= ~human & (~preserve|previous_allowed) & (major != current)
    proposed = int(gate.sum())
    accepted = dev_matches >= 16 and dev_balanced >= .525 and proposed <= 50
    if not accepted:
        gate[:] = False
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    for dest, original in [('baseline_class', 'assigned_class'), ('predicted_class', 'assigned_class'), ('category_source', 'label_source'),
                           ('category_score', 'category_score'), ('category_score_kind', 'category_score_kind'), ('human_reviewed', 'human_reviewed'), ('needs_review', 'needs_review')]:
        result[dest] = complete[original]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[major[i]], 'vit_knn_label_propagation_snapshot', support[i] / 4., 'multi_method_vote_share', True]
    destination = NOTES / f'table-csv/{prefix}_selected.csv'
    result.reset_index().to_csv(destination, index=False)
    pd.DataFrame(history).to_csv(NOTES / f'table-csv/{prefix}_development.csv', index=False)
    pd.DataFrame({'key': keys, 'candidate_class': np.array(CLASSES)[major], 'knn_bank_support': support,
                  'graph_support': graph_support, 'local_anchor_banks': local, 'accepted': gate}).to_csv(NOTES / f'table-csv/{prefix}_evidence.csv', index=False)
    np.savez_compressed(NOTES / f'.runtime/{prefix}_outputs.npz', keys=keys, knn_predictions=pred, graph_predictions=graphs,
                        probabilities=np.stack(probabilities),holdout_probabilities=np.stack(holdouts),accepted=gate)
    info = json.loads((NOTES / 'json/robust_geometry_heads_frozen.json').read_text())
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv').set_index('key')
    info.update(status='frozen_before_posthoc_vit_propagation_audit', method=plan['method'], prediction_sha256=sha(destination),
                protected_sha256=protected, human_leave_source_group_out_matches=dev_matches, human_holdout_rows=34,
                human_diagnostic_protocol='five-geometry-source-group-fold kNN development,graph not independently validated',
                development_balanced_recall=dev_balanced, selected_knn_heads=selections, graph_history=graph_history,
                vit_plan_sha256=sha(NOTES / f'json/{prefix}_plan.json'), source_feature_sha256=sha(cache),
                feature_mode=mode,localisation_feature_sha256=localisation_sha,
                automatic_fit_filter_history=cleaning_history,
                accepted_by_development_guard=accepted, proposed_new_labels=proposed, changed_rows_vs_baseline=int(gate.sum()),
                new_vit_propagation_changes=int(gate.sum()), new_neural_changes=0,
                cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                source_counts=result.category_source.value_counts().to_dict(), validation_limit=plan['evaluation_limit'],
                source_code_sha256=sha(__file__), seconds=time.time() - started)
    assert all(sha(ROOT / k) == value for k, value in protected.items())
    dump(NOTES / f'json/{prefix}_frozen.json', info)
    print(json.dumps({k: info[k] for k in ['status', 'human_leave_source_group_out_matches', 'development_balanced_recall',
                                            'proposed_new_labels', 'accepted_by_development_guard', 'new_vit_propagation_changes', 'seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['full','detector','focus','base','base_clean','cnn','cnn_clean'],default='full')
    main(parser.parse_args().mode)
