"""Fixed material-vs-semantic ablation using existing train-only MINC cache.

No material ontology category is manually mapped to competition labels. Learned
neighbors use cold program targets and34human sources, with grouped development.
"""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'normalized_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("start=time.time();prefix='texture_neighbors' if use_texture else 'normalized_base'", "start=time.time();assert use_texture;prefix='material_semantic_neighbors'"),
        ("'multiscale_texture_DINOBase_grouped_neighbors_v1'", "'MINC_surface_DINO_semantic_grouped_neighbors_v1'"),
        ("'hypothesis':'remove common image directions or reduce correlated background features before distance-based label transfer'", "'hypothesis':'frozenmaterialsurfacefeaturescomplementsemanticshape;noMINC-to-targetmanualmapping;fixedblockweights'"),
        ("['regional_semantic','texture_semantic','multiscale_semantic','texture_region']", "['material_only','global_material','regional_material','regional_material_distribution']"),
        ("report=json.loads((NOTES/'json/base_clean_primary_frozen.json').read_text());initial=", "active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial="),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall="),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    first=source.index('    if use_texture:\n        tp=');last=source.index('    a=pd.read_csv',first)
    source=source[:first]+'''    mp=NOTES/'.runtime/minc_train_features.npz';mz=np.load(mp,allow_pickle=False)
    assert np.array_equal(mz['keys'],keys) and np.array_equal(mz['source_sha256'],frame.source_sha256.to_numpy(str))
    material=unit(mz['features'][:,1:].mean(1));distribution=unit(np.sqrt(np.maximum(mz['probabilities'][:,1:].mean(1),1e-8)))
    regional=unit(np.c_[cls,unit(z['features'][:,2304:])]);texture_sha=sha(mp)
    banks={'material_only':material,'global_material':unit(np.c_[cls,.5*material]),
           'regional_material':unit(np.c_[regional,.5*material]),'regional_material_distribution':unit(np.c_[regional,.35*distribution])}
'''+source[last:]
    source=source.replace('main(parser.parse_args().texture)','main(True)')
    compile(source,'material_semantic_neighbors.py','exec')
    (NOTES/'material_semantic_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
