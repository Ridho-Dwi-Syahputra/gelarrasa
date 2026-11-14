"""Predeclare DINOv3 Base grouped learner before feature extraction completes."""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'normalized_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit", "from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit\nfrom selective_expert_refinement import wilson_lower"),
        ("start=time.time();prefix='texture_neighbors' if use_texture else 'normalized_base'", "start=time.time();assert use_texture;prefix='dinov3_base_neighbors'"),
        ("'multiscale_texture_DINOBase_grouped_neighbors_v1'", "'DINOv3Base_spatial_dictionary_grouped_neighbors_v1'"),
        ("'hypothesis':'remove common image directions or reduce correlated background features before distance-based label transfer'", "'hypothesis':'new85.6Mimage-onlyencoder,denseobjectfeaturesandontology-freesourceclassifiercomplement;notVLM;nohiddenreviewtargets'"),
        ("['regional_semantic','texture_semantic','multiscale_semantic','texture_region']", "['v3_CLS','v3_object_mean','v3_dense','v3_source_dictionary']"),
        ("z=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False)", "z=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False)"),
        ("'maximum_changes':40", "'maximum_changes':20"),
        ("allowed=rank[gate[rank]][:40]", "allowed=rank[gate[rank]][:20]"),
        ("'primary>=24/34andbalancedrecall>=.75andcalibrationthresholdexistsand>=2qualifiedbanks'", "'global:>=26/34andmacro>.758333333334;ORrisk:matches>=20,macro>=.65,covered>=12,precision>=.95,Wilsonranking>=prior12/12+.02;>=2qualifiedbanksandthresholdexists'"),
        ("report=json.loads((NOTES/'json/base_clean_primary_frozen.json').read_text());initial=", "active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial="),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,source_feature_sha256=sha(NOTES/'.runtime/dinov3_base_train_features.npz'),feature_mode='dinov3_base',development_balanced_recall="),
        ("qualified=[i for i,s in enumerate(selections) if s['development_balanced_recall']>=.625 and s['development_balanced_recall']>=chosen['development_balanced_recall']-.1] if use_texture else list(range(len(selections)))", "qualified=sorted([i for i,s in enumerate(selections) if s['development_balanced_recall']>=.65 and s['development_balanced_recall']>=chosen['development_balanced_recall']-.1],key=lambda i:(-selections[i]['development_balanced_recall'],-selections[i]['anchor_matches']))[:2]"),
        ("'private_review_or_archive_read':False", "'private_review_or_archive_read':False,'qualified_bank_policy':'best2bydevelopmentmacro;correlatednotindependent','source_feature_sha256':sha(NOTES/'.runtime/dinov3_base_train_features.npz')"),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    first=source.index('    if use_texture:\n        tp=');last=source.index('    a=pd.read_csv',first)
    source=source[:first]+'''    sp=NOTES/'.runtime/semantic_dictionary_features.npz';sd=np.load(sp,allow_pickle=False);assert np.array_equal(sd['keys'],keys)
    object_mean=unit(z['features'][:,1536:2304]);valid_mean=unit(z['features'][:,768:1536]);object_std=unit(z['features'][:,2304:3072])
    source_t1=unit(sd['features'][:,0]);source_t2=unit(sd['features'][:,1]);texture_sha=sha(NOTES/'.runtime/dinov3_base_train_features.npz')
    banks={'v3_CLS':cls,'v3_object_mean':unit(np.c_[cls,object_mean]),'v3_dense':unit(np.c_[cls,valid_mean,object_mean,object_std]),
           'v3_source_dictionary':unit(np.c_[cls,.35*source_t1,.35*source_t2])}
'''+source[last:]
    old="accepted=cal is not None and chosen['anchor_matches']>=(24 if use_texture else 23) and chosen['development_balanced_recall']>=(.75 if use_texture else .725) and len(qualified)>=2"
    new="""global_ok=cal is not None and chosen['anchor_matches']>=26 and chosen['development_balanced_recall']>.758333333334
    risk_ok=cal is not None and chosen['anchor_matches']>=20 and chosen['development_balanced_recall']>=.65 and cal['covered']>=12 and cal['precision']>=.95 and wilson_lower(cal['correct'],cal['covered'])>=wilson_lower(12,12)+.02
    accepted=(global_ok or risk_ok) and len(qualified)>=2"""
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace('main(parser.parse_args().texture)','main(True)')
    compile(source,'dinov3_base_neighbors.py','exec');(NOTES/'dinov3_base_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
