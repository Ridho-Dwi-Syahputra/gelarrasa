"""Fixed source-vocabulary ablation; no ontology-to-material manual mapping."""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'normalized_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("start=time.time();prefix='texture_neighbors' if use_texture else 'normalized_base'", "start=time.time();assert use_texture;prefix='semantic_dictionary_neighbors'"),
        ("'multiscale_texture_DINOBase_grouped_neighbors_v1'", "'full_source_dictionary_DINO_grouped_neighbors_v1'"),
        ("'hypothesis':'remove common image directions or reduce correlated background features before distance-based label transfer'", "'hypothesis':'retain21k-sourceinformationbefore8classontology;fixedsqrt-top128sourceprobabilities,independent512randomprojection;learnneighborslocally'"),
        ("['regional_semantic','texture_semantic','multiscale_semantic','texture_region']", "['source_dictionary','regional_dictionary_T1','regional_dictionary_T2','global_both_dictionaries']"),
        ("'primary>=24/34andbalancedrecall>=.75andcalibrationthresholdexistsand>=2qualifiedbanks'", "'primary>=25/34andbalancedrecall>.75andcalibrationthresholdexistsand>=2qualifiedbanks'"),
        ("chosen['anchor_matches']>=(24 if use_texture else 23)", "chosen['anchor_matches']>=25"),
        ("chosen['development_balanced_recall']>=(.75 if use_texture else .725)", "chosen['development_balanced_recall']>.75+1e-12"),
        ("report=json.loads((NOTES/'json/base_clean_primary_frozen.json').read_text());initial=", "active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial="),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall="),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    first=source.index('    if use_texture:\n        tp=');last=source.index('    a=pd.read_csv',first)
    source=source[:first]+'''    sp=NOTES/'.runtime/semantic_dictionary_features.npz';sz=np.load(sp,allow_pickle=False)
    assert np.array_equal(sz['keys'],keys) and np.array_equal(sz['source_sha256'],frame.source_sha256.to_numpy(str))
    t1=unit(sz['features'][:,0]);t2=unit(sz['features'][:,1]);regional=unit(np.c_[cls,unit(z['features'][:,2304:])]);texture_sha=sha(sp)
    banks={'source_dictionary':t1,'regional_dictionary_T1':unit(np.c_[regional,.5*t1]),
           'regional_dictionary_T2':unit(np.c_[regional,.5*t2]),'global_both_dictionaries':unit(np.c_[cls,.35*t1,.35*t2])}
'''+source[last:]
    source=source.replace('main(parser.parse_args().texture)','main(True)')
    compile(source,'semantic_dictionary_neighbors.py','exec')
    (NOTES/'semantic_dictionary_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
