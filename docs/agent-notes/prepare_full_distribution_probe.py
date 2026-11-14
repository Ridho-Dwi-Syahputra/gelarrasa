"""Predeclare full-source-distribution + DINOv3 Base complementarity ablation."""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'dinov3_base_neighbors.py').read_text(encoding='utf-8')
    replacements=[
        ("prefix='dinov3_base_neighbors'", "prefix='full_distribution_neighbors'"),
        ("'DINOv3Base_spatial_dictionary_grouped_neighbors_v1'", "'DINOv3Base_full_source_Hellinger_grouped_neighbors_v1'"),
        ("'new85.6Mimage-onlyencoder,denseobjectfeaturesandontology-freesourceclassifiercomplement;notVLM;nohiddenreviewtargets'", "'fullsourceprobabilitymassratherthantop128;streamedHellingerprojection+V3Base;noontologyorhiddenreviewtargets'"),
        ("semantic_dictionary_features.npz", "full_source_distribution_features.npz"),
        ("['v3_CLS','v3_object_mean','v3_dense','v3_source_dictionary']", "['full_source','v3_full_T1','v3_object_full_T2','v3_both_full']"),
        ("chosen['anchor_matches']>=26", "chosen['anchor_matches']>=27"),
        ("chosen['development_balanced_recall']>.758333333334", "chosen['development_balanced_recall']>.783333333334"),
        ("wilson_lower(12,12)+.02", "wilson_lower(14,14)+.02"),
        ("global:>=26/34andmacro>.758333333334;ORrisk:matches>=20,macro>=.65,covered>=12,precision>=.95,Wilsonranking>=prior12/12+.02", "global:>=27/34andmacro>.783333333334;ORrisk:matches>=20,macro>=.65,covered>=14,precision>=.95,Wilsonranking>=prior14/14+.02"),
        ("cal['covered']>=12", "cal['covered']>=14"),
    ]
    for old,new in replacements:
        assert old in source,old
        source=source.replace(old,new)
    first=source.index("    banks={'v3_CLS':");last=source.index('    a=pd.read_csv',first)
    source=source[:first]+'''    banks={'full_source':source_t1,'v3_full_T1':unit(np.c_[cls,.35*source_t1]),
           'v3_object_full_T2':unit(np.c_[cls,object_mean,.35*source_t2]),'v3_both_full':unit(np.c_[cls,.35*source_t1,.35*source_t2])}
'''+source[last:]
    compile(source,'full_distribution_neighbors.py','exec');(NOTES/'full_distribution_neighbors.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
