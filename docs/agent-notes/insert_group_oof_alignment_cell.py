from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "final-submit" / "DSC2026_Notebook.ipynb"
AUDIT = ROOT / "docs" / "agent-notes" / "table-csv" / "archive_aligned_residual_oof.csv"
frame = pd.read_csv(AUDIT)
reference = dict(zip(frame.parent_key.astype(str), frame.reference_class_audit.astype(str)))
assert len(reference) == 88

markdown = '''## Rekonsiliasi kelompok kandidat\n\nFoto tersisa diprediksi dengan tiga model yang dilatih silang menurut source group. Kategori acuan dari tim baru dibandingkan setelah prediksi fold dibuat. Foto dipertahankan jika sedikitnya dua model sepakat pada kategori acuan; yang lain ditahan dari training. Ini ukuran kesesuaian terhadap referensi, bukan validasi semantik independen.'''

code = r'''from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

REFERENCE_BY_KEY = __REFERENCE_JSON__
assert len(REFERENCE_BY_KEY) == 88
full_before = train_rows.loc[~train_rows.is_crop.to_numpy(bool)].copy()
residual = full_before[full_before.parent_key.isin(REFERENCE_BY_KEY)].copy()
fit_pool = full_before[~full_before.parent_key.isin(REFERENCE_BY_KEY)].copy()
assert len(full_before) == 1887 and len(residual) == 88 and len(fit_pool) == 1799

feature_by_key = {str(k): embeddings[i] for i, k in enumerate(model_rows.sample_key.astype(str))}
residual = residual.sort_values('parent_key').reset_index(drop=True)
fit_pool = fit_pool.sort_values('parent_key').reset_index(drop=True)
X_res = np.stack([feature_by_key[k] for k in residual.parent_key.astype(str)])
X_fit_all = np.stack([feature_by_key[k] for k in fit_pool.parent_key.astype(str)])
y_fit_all = fit_pool.assigned_class.astype(str).to_numpy()
res_groups = residual.parent_group.astype(str).to_numpy()
estimators = {
    'logistic': lambda: LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
    'svm_rbf': lambda: SVC(C=4.0, gamma='scale', kernel='rbf', decision_function_shape='ovr'),
    'knn7': lambda: KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine', n_jobs=2),
}
preds = {name: np.full(len(residual), '', dtype=object) for name in estimators}
folds = list(GroupKFold(n_splits=5).split(residual, groups=res_groups))
with threadpool_limits(limits=4):
    for fold, (_, val_idx) in enumerate(folds):
        held_groups = set(res_groups[val_idx])
        fit_mask = ~fit_pool.parent_group.astype(str).isin(held_groups).to_numpy()
        X_fit, y_fit = X_fit_all[fit_mask], y_fit_all[fit_mask]
        assert set(y_fit) == set(CLASSES)
        for name, create in estimators.items():
            model = create().fit(X_fit, y_fit)
            preds[name][val_idx] = model.predict(X_res[val_idx]).astype(str)

matrix = np.stack([preds[n] for n in estimators], axis=1)
consensus, votes, agrees = [], [], []
for i, row in enumerate(matrix):
    names, counts = np.unique(row, return_counts=True)
    j = int(np.argmax(counts))
    consensus.append(str(names[j])); votes.append(int(counts[j]))
    agrees.append(str(names[j]) == REFERENCE_BY_KEY[str(residual.iloc[i].parent_key)])
consensus = np.asarray(consensus, dtype=str)
votes = np.asarray(votes, dtype=int)
agrees = np.asarray(agrees, dtype=bool)
accept = (votes >= 2) & agrees
accepted = set(residual.loc[accept, 'parent_key'].astype(str))
rejected = set(residual.loc[~accept, 'parent_key'].astype(str))
assert len(accepted) == 70 and len(rejected) == 18

evidence = residual[['parent_key','parent_id','parent_group','assigned_class']].copy()
evidence['reference_class'] = evidence.parent_key.map(REFERENCE_BY_KEY)
evidence['crossfit_consensus_class'] = consensus
evidence['model_votes'] = votes
evidence['reference_match'] = agrees
evidence['accepted_for_training'] = accept
evidence['decision'] = np.where(accept, 'relabel_to_crossfit_consensus', 'hold_out_of_training')
for i, row in residual.iterrows():
    key = str(row.parent_key)
    selected = processed.sample_key.eq(key)
    assert selected.sum() == 1
    old = resolve(processed.loc[selected, 'output_path'].iloc[0])
    if accept[i]:
        category = str(consensus[i])
        new = PROCESSED / 'train' / category / old.name
        source = 'group_oof_consensus_reference_match_v1'
        eligible, reason = True, ''
    else:
        category = str(row.assigned_class)
        new = PROCESSED / 'review' / 'excluded_train' / category / old.name
        source = str(processed.loc[selected, 'label_source'].iloc[0])
        eligible, reason = False, 'group_oof_reference_alignment_disagreement'
        deferred_keys.add(key)
    new.parent.mkdir(parents=True, exist_ok=True)
    if old.resolve() != new.resolve(): os.replace(old, new)
    rel = relative(new)
    for table in (processed, model_rows):
        mask = table.sample_key.eq(key)
        table.loc[mask, 'assigned_class'] = category
        table.loc[mask, 'output_path'] = rel
        table.loc[mask, 'training_eligible'] = eligible
        table.loc[mask, 'deferral_reason'] = reason
        if accept[i]:
            table.loc[mask, 'label_source'] = source
            table.loc[mask, 'category_score'] = float(votes[i] / 3)
            table.loc[mask, 'category_score_kind'] = 'three_model_group_oof_vote_share'
            table.loc[mask, 'needs_review'] = True
        else:
            table.loc[mask, 'label_status'] = 'held_after_group_oof_reference_check'

processed.to_csv(PROCESSED / 'manifest.csv', index=False)
model_rows = model_rows[model_rows.split.eq('test') | (model_rows.split.eq('train') & model_rows.training_eligible)].reset_index(drop=True)
train_positions = np.flatnonzero(model_rows.split.eq('train').to_numpy())
test_positions = np.flatnonzero(model_rows.split.eq('test').to_numpy())
train_rows = model_rows.iloc[train_positions].reset_index(drop=True)
test_rows = model_rows.iloc[test_positions].reset_index(drop=True)
X_train = np.stack([feature_by_key[k] for k in train_rows.sample_key.astype(str)])
X_test = np.stack([feature_by_key[k] for k in test_rows.sample_key.astype(str)])
y_train = np.asarray([CLASSES.index(c) for c in train_rows.assigned_class], dtype=int)
groups = train_rows.parent_group.to_numpy(str)
assert len(train_rows) == 1876 and len(test_rows) == 480 and set(y_train) == set(range(8))
summary['training_eligible_train_samples'] = len(train_rows)
summary['train_class_counts'] = train_rows.assigned_class.value_counts().reindex(CLASSES, fill_value=0).to_dict()
summary['group_oof_reference_alignment'] = {'fit_pool_full_frames':1799,'candidates':88,'accepted':70,'held_out':18,'minimum_model_votes':2,'test_rows_in_fit':0}
summary['deferred_train_keys'] = sorted(deferred_keys)
summary['deferred_train_samples'] = int((processed.split.eq('train') & ~processed.training_eligible).sum())
evidence.to_csv(PROCESSED / 'group_oof_alignment_evidence.csv', index=False)
display(evidence.decision.value_counts().to_frame('jumlah'))
print('Diterima:', len(accepted), '| ditahan:', len(rejected))'''.replace('__REFERENCE_JSON__', json.dumps(reference, separators=(',', ':')))

nb = nbformat.read(NOTEBOOK, as_version=4)
anchor = next((i for i, cell in enumerate(nb.cells) if cell.cell_type == 'markdown' and cell.source.startswith('## 13. Validasi kelompok pada target otomatis')), None)
if anchor is None:
    raise RuntimeError('Target notebook insertion point not found')
if any(cell.cell_type == 'markdown' and cell.source.startswith('## Rekonsiliasi kelompok kandidat') for cell in nb.cells):
    raise RuntimeError('Alignment cell already exists; refusing duplicate insertion')
nb.cells[anchor:anchor] = [nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code)]
nbformat.write(nb, NOTEBOOK)
print(f'Inserted before original validation cell; notebook now has {len(nb.cells)} cells.')
