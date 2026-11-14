"""Prepare a fixed normalized-neighbor ablation on enlarged object pixels."""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'normalized_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("start=time.time();prefix='texture_neighbors' if use_texture else 'normalized_base'", "start=time.time();assert not use_texture;prefix='object_zoom_neighbors'"),
        ("'fold_fitted_centered_whitened_DINOv2Base_neighbors_v1'", "'object_enlarged_DINOv2Base_grouped_neighbors_v1'"),
        ("'hypothesis':'remove common image directions or reduce correlated background features before distance-based label transfer'", "'hypothesis':'enlargefrozenautomaticobjectboxBEFOREbackbone224;fullframefallback;no newtrainingcrops;fitnormalizerperfold'"),
        ("'base_cls'", "'zoom_cls'"), ("'base_local'", "'zoom_local'"),
        ("z=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False)", "z=np.load(NOTES/'.runtime/dinov2_object_zoom_features.npz',allow_pickle=False)"),
        ("'primary>=23/34andbalancedrecall>=.725andcalibrationthresholdexists'", "'primary>=24/34andbalancedrecall>=.75andcalibrationthresholdexistsand2qualifiedbanks'"),
        ("(24 if use_texture else 23)", "24"), ("(.75 if use_texture else .725)", ".75"),
        ("report=json.loads((NOTES/'json/base_clean_primary_frozen.json').read_text());initial=", "active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial="),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,source_feature_sha256=sha(NOTES/'.runtime/dinov2_object_zoom_features.npz'),development_balanced_recall="),
        ("'new_manual_labels':0,'test_rows':0,", "'new_manual_labels':0,'test_rows':0,'source_feature_sha256':sha(NOTES/'.runtime/dinov2_object_zoom_features.npz'),"),
    ]
    for old,new in replacements:
        assert old in source,old
        source=source.replace(old,new)
    compile(source,'object_zoom_neighbors.py','exec')
    (NOTES/'object_zoom_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
