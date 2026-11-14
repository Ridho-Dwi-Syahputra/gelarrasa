from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

REFERENCE_BY_KEY = {"train:0006":"plastic","train:0018":"metal","train:0027":"paper","train:0035":"plastic","train:0079":"cardboard","train:0086":"plastic","train:0087":"plastic","train:0120":"glass","train:0128":"plastic","train:0156":"electronic","train:0160":"metal","train:0179":"cardboard","train:0183":"paper","train:0189":"cardboard","train:0200":"paper","train:0220":"plastic","train:0224":"plastic","train:0225":"cardboard","train:0247":"plastic","train:0264":"cardboard","train:0287":"plastic","train:0334":"cardboard","train:0335":"glass","train:0375":"metal","train:0400":"metal","train:0435":"plastic","train:0438":"cardboard","train:0470":"plastic","train:0479":"plastic","train:0489":"plastic","train:0502":"electronic","train:0515":"electronic","train:0516":"plastic","train:0554":"electronic","train:0604":"paper","train:0637":"plastic","train:0653":"plastic","train:0667":"electronic","train:0676":"plastic","train:0737":"cardboard","train:0776":"foodwaste","train:0856":"cardboard","train:0890":"cardboard","train:0900":"plastic","train:0909":"cardboard","train:0928":"cardboard","train:0939":"plastic","train:0960":"cardboard","train:1002":"electronic","train:1005":"plastic","train:1045":"metal","train:1078":"plastic","train:1086":"cardboard","train:1096":"plastic","train:1097":"plastic","train:1115":"plastic","train:1129":"plastic","train:1141":"plastic","train:1146":"electronic","train:1147":"metal","train:1157":"paper","train:1185":"plastic","train:1186":"electronic","train:1277":"paper","train:1287":"cardboard","train:1288":"electronic","train:1359":"plastic","train:1402":"cardboard","train:1418":"electronic","train:1422":"foodwaste","train:1441":"electronic","train:1445":"plastic","train:1464":"electronic","train:1492":"electronic","train:1507":"plastic","train:1527":"plastic","train:1587":"plastic","train:1601":"metal","train:1650":"cardboard","train:1712":"plastic","train:1736":"plastic","train:1787":"electronic","train:1789":"electronic","train:1810":"textile","train:1825":"electronic","train:1840":"cardboard","train:1854":"plastic","train:1870":"cardboard"}
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
full_frame_mask = ~train_rows.is_crop.to_numpy(bool)
train_rows = train_rows.loc[full_frame_mask].reset_index(drop=True)
X_train = X_train[full_frame_mask]
y_train = np.asarray([CLASSES.index(c) for c in train_rows.assigned_class], dtype=int)
groups = train_rows.parent_group.to_numpy(str)
assert len(train_rows) == 1869 and len(X_train) == 1869 and set(y_train) == set(range(8))
summary['training_eligible_train_samples'] = int((processed.split.eq('train') & processed.training_eligible).sum())
summary['modeling_train_samples'] = len(train_rows)
summary['modeling_training_view'] = 'eligible_full_frame_only'
summary['train_class_counts'] = train_rows.assigned_class.value_counts().reindex(CLASSES, fill_value=0).to_dict()
summary['group_oof_reference_alignment'] = {'fit_pool_full_frames':1799,'candidates':88,'accepted':70,'held_out':18,'minimum_model_votes':2,'test_rows_in_fit':0}
summary['deferred_train_keys'] = sorted(deferred_keys)
summary['deferred_train_samples'] = int((processed.split.eq('train') & ~processed.training_eligible).sum())
evidence.to_csv(PROCESSED / 'group_oof_alignment_evidence.csv', index=False)
display(evidence.decision.value_counts().to_frame('jumlah'))
print('Diterima:', len(accepted), '| ditahan:', len(rejected))
