"""Select primary kNN family by bounded development, not private review matching."""
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def main(mode='full'):
    is_base=mode in ['base','base_clean']
    is_cnn=mode=='cnn'
    source_prefix={'full':'vit_propagation','detector':'detector_propagation','focus':'focus_propagation','base':'base_propagation','base_clean':'base_clean_propagation','cnn':'cnn_propagation'}[mode]
    prefix={'full':'vit_primary','detector':'detector_primary','focus':'focus_primary','base':'base_primary','base_clean':'base_clean_primary','cnn':'cnn_primary'}[mode]
    prior = json.loads((NOTES / f'json/{source_prefix}_frozen.json').read_text())
    assert all(sha(ROOT / name) == value for name, value in prior['protected_sha256'].items())
    heads = prior['selected_knn_heads']
    winner = max(range(len(heads)), key=lambda i: heads[i]['development_balanced_recall'])
    head = heads[winner]
    development = pd.read_csv(NOTES / ('table-csv/vit_knn_development.csv' if mode=='full' else f'table-csv/{source_prefix}_development.csv'))
    row = development[development.bank.eq(head['bank']) & development.neighbors.eq(head['neighbors']) & development.human_weight.eq(head['human_weight'])].iloc[0]
    output_path=NOTES/('.runtime/vit_knn_graph_outputs.npz' if mode=='full' else f'.runtime/{source_prefix}_outputs.npz')
    evidence_path=NOTES/('table-csv/vit_propagation_evidence.csv' if mode=='full' else f'table-csv/{source_prefix}_evidence.csv')
    minimum_matches=23 if is_cnn else (20 if is_base else (16 if mode=='full' else 19))
    minimum_balanced=.725 if is_cnn else (.6333333333333333 if is_base else (.525 if mode=='full' else .6083333333333333))
    plan = {'method': f'development_selected_{mode}_ViT_kNN_with_multibank_propagation_v1',
            'selection': 'highest per-family34anchor balanced recall; not mean of poorly-performing probability banks',
            'primary_head': head, 'primary_anchor_matches': int(row.anchor_matches), 'development_rows': 34,
            'feature_mode':mode,'minimum_anchor_matches':minimum_matches,'minimum_development_balanced_recall':minimum_balanced,
            'acceptance': f'primary category agrees with>=3/4kNNbanks,>=2/3graphs and>=2localitybanks;previousautomaticrefinements require4/4banks,3/3graphs,>=3localitybanks' + ('' if is_base else ';allpreviousrefinementsprotected') + f';humanalwaysprotected;primarydevelopment>={minimum_matches}/34andbalancedrecall>={minimum_balanced};<=50changes',
            'private_review_or_archive_labels_read': False, 'new_manual_targets': 0, 'test_rows_in_fit': 0,
            'evaluation_limit': f'{int(row.anchor_matches)}/34 is selected primary kNN development,not graph Accuracy or independent final evaluation',
            'source_vote_prediction_sha256': sha(output_path),
            'source_frozen_plan_sha256': sha(NOTES / f'json/{source_prefix}_frozen.json'),
            'source_code_sha256': sha(__file__)}
    dump(NOTES / f'json/{prefix}_plan.json', plan)
    votes = np.load(output_path, allow_pickle=False)
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = manifest[manifest.split.eq('train') & ~manifest.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame = complete[complete.training_eligible].reset_index(drop=True)
    keys = frame.parent_key.to_numpy(str)
    assert np.array_equal(votes['keys'], keys)
    evidence = pd.read_csv(evidence_path).set_index('key').loc[keys]
    pred, graphs = votes['knn_predictions'], votes['graph_predictions']
    primary = pred[winner]
    support = (pred == primary).sum(0)
    graph_support = (graphs == primary).sum(0)
    preserve = ~frame.label_source.isin(['iterative_program_refinement_snapshot', 'historical_routing_snapshot']).to_numpy()
    current = np.array([CLASSES.index(c) for c in frame.assigned_class])
    local = evidence.local_anchor_banks.to_numpy()
    local_category_correct = evidence.candidate_class.to_numpy(str) == np.array(CLASSES)[primary]
    gate = (support >= 3) & (graph_support >= 2) & (local >= 2) & local_category_correct
    previous_allowed=(support>=4)&(graph_support>=3)&(local>=3) if is_base or is_cnn else np.zeros(len(frame),bool)
    gate &= ~frame.human_reviewed.to_numpy(bool) & (~preserve|previous_allowed) & (primary != current)
    proposed = int(gate.sum())
    accepted = int(row.anchor_matches) >= minimum_matches and head['development_balanced_recall'] >= minimum_balanced and proposed <= 50
    if not accepted:
        gate[:] = False
    result = complete[['parent_key', 'parent_id', 'source_path', 'source_sha256']].rename(columns={'parent_key': 'key', 'parent_id': 'image_id'})
    for dest, source in [('baseline_class', 'assigned_class'), ('predicted_class', 'assigned_class'), ('category_source', 'label_source'),
                         ('category_score', 'category_score'), ('category_score_kind', 'category_score_kind'), ('human_reviewed', 'human_reviewed'), ('needs_review', 'needs_review')]:
        result[dest] = complete[source]
    result = result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i], ['predicted_class', 'category_source', 'category_score', 'category_score_kind', 'needs_review']] = [
            CLASSES[primary[i]], 'vit_knn_label_propagation_snapshot', support[i] / 4., 'multi_method_vote_share', True]
    destination = NOTES / f'table-csv/{prefix}_selected.csv'
    result.reset_index().to_csv(destination, index=False)
    initial = pd.read_csv(NOTES / 'table-csv/bounded_consensus_selected.csv').set_index('key')
    report = dict(prior)
    report.update(status='frozen_before_any_private_primary_head_audit', method=plan['method'], prediction_sha256=sha(destination),
                  primary_head=head, primary_plan_sha256=sha(NOTES / f'json/{prefix}_plan.json'),
                  human_leave_source_group_out_matches=int(row.anchor_matches), development_balanced_recall=head['development_balanced_recall'],
                  human_diagnostic_protocol='selected primary kNNfive-source-group-fold development;graphnot independently validated',
                  validation_limit=plan['evaluation_limit'], proposed_new_labels=proposed, changed_rows_vs_baseline=int(gate.sum()),
                  accepted_by_development_guard=accepted, new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index, 'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(), source_code_sha256=sha(__file__))
    assert all(sha(ROOT / name) == value for name, value in report['protected_sha256'].items())
    dump(NOTES / f'json/{prefix}_frozen.json', report)
    dump(NOTES / f'json/{prefix}_publication_decision.json', {'decision': 'publish' if accepted else 'reject',
         'prediction_sha256': sha(destination), 'guard': plan['acceptance'], 'primary_head_development_matches': int(row.anchor_matches),
         'new_category_proposals': proposed, 'private_review_used_for_selection': False,
         'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k: report[k] for k in ['status', 'primary_head', 'human_leave_source_group_out_matches',
                                            'proposed_new_labels', 'accepted_by_development_guard', 'new_vit_propagation_changes']}, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['full','detector','focus','base','base_clean','cnn'],default='full')
    main(parser.parse_args().mode)
