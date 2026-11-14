"""Fixed webinar-style HOG/color ablation; no new image labels or test fitting."""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'normalized_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("start=time.time();prefix='texture_neighbors' if use_texture else 'normalized_base'", "start=time.time();assert use_texture;prefix='webinar_hybrid_neighbors'"),
        ("'multiscale_texture_DINOBase_grouped_neighbors_v1'", "'webinar_HOG_color_DINOBase_grouped_ablation_v1'"),
        ("'hypothesis':'remove common image directions or reduce correlated background features before distance-based label transfer'", "'hypothesis':'fixedHOG1764andHSV48complementfrozenDINO;L2eachblock;fixed.35handcraftedblockweight;onlyTRAINrowsbeforefitting'"),
        ("['regional_semantic','texture_semantic','multiscale_semantic','texture_region']", "['semantic','semantic_hog','semantic_color','semantic_hog_color']"),
        ("report=json.loads((NOTES/'json/base_clean_primary_frozen.json').read_text());initial=", "active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial="),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall="),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    first=source.index('    if use_texture:\n        tp=')
    last=source.index("    a=pd.read_csv",first)
    source=source[:first]+'''    hp=ROOT/'data/processed/features.npz';hz=np.load(hp,allow_pickle=False)
    lookup_features=dict(zip(hz['keys'],range(len(hz['keys']))))
    classic=hz['features'][[lookup_features[k] for k in keys]].copy()
    assert classic.shape==(1904,1812) and all(k.startswith('train:') for k in keys)
    hog=unit(classic[:,:1764]);color=unit(classic[:,1764:]);semantic=unit(np.c_[cls,unit(z['features'][:,2304:])])
    texture_sha=sha(hp)
    banks={'semantic':semantic,'semantic_hog':unit(np.c_[semantic,.35*hog]),
           'semantic_color':unit(np.c_[semantic,.35*color]),'semantic_hog_color':unit(np.c_[semantic,.35*hog,.35*color])}
'''+source[last:]
    source=source.replace('main(parser.parse_args().texture)','main(True)')
    compile(source,'webinar_hybrid_neighbors.py','exec')
    (NOTES/'webinar_hybrid_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
