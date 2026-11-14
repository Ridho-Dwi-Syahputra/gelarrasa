"""Register a precision-first ablation, explicitly trading coverage for risk.

This is separate from the failed whole34 guard and failed coverage16 consensus.
The34development labels are reused: no independent confidence certificate.
"""
from fewshot_train_only_probe import NOTES


def main():
    source=(NOTES/'selective_expert_refinement.py').read_text(encoding='utf-8')
    replacements=[
        ("prefix='selective_expert'", "prefix='risk_first_expert'"),
        ("NOTES/'json/multifamily_assignment_frozen.json'", "NOTES/'json/selective_expert_frozen.json'"),
        ("minimum_coverage=base_cal['covered'];minimum_balanced=baseline['development_balanced_recall']-.1", "minimum_coverage=10;minimum_balanced=.65"),
        ("'method':'development_selective_specialist_refinement_v1'", "'method':'precision_first_source_group_expert_ablation_v1'"),
        ("'objective_change':'selectivepseudo-labelquality/coverage,notmaximumwhole34anchoraccuracy;previousrejectionsarekept'", "'objective_change':'precisionFIRSTsubset:tradecoveragefor>=95percentempiricalprecisionandhigherWilsonrankingthanprior15/16;notmaximumwhole34accuracy;allpriorrejectionsremain;noindependentguarantee'"),
        ("['current_multifamily','normalized_neighbor','base_neural','cnn_neighbor_mean','fold_contained_self_training']", "['object_zoom','fold_contained_self_training','current_multifamily','trusted_repair','source_contrastive','semantic_color']"),
        ("'minimum_empirical_precision':.9", "'minimum_empirical_precision':.95"),
        ("'guard':'differentexpertwithWilsonlowerscore>=baseline+.02andcoverage>=baseline;floorwhole34performanceasabove'", "'guard':'Wilsonranking>=prior15/16lower+.02;coverage>=10;precision>=.95;globalmacro>=.65andmatches>=20;NOTcoverage>=16;priorguardsunchanged'"),
        ("'maximum_changes':40", "'maximum_changes':20"),
        ("'private_review_or_archive_read':False", "'private_review_or_archive_read':False,'researcher_seen_prior_aggregate_audits':True"),
        ("and precision>=.9\n", "and precision>=.95\n"),
        ("allowed=rank[gate[rank]][:40]", "allowed=rank[gate[rank]][:20]"),
        ("neural_choice=chosen['expert']=='base_neural'", "neural_choice=chosen['expert']=='source_contrastive'"),
        ("component_output_sha256={'current':mh,'normalized':nh,'neural':eh,'cnn':ch,'selftraining':sh}", "component_output_sha256=output_hashes"),
        ("human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=", "human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall="),
    ]
    for old,new in replacements:
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    first=source.index("    mi,mh=load('multifamily_assignment')")
    last=source.index("    s=pd.read_csv",first)
    source=source[:first]+'''    names=['object_zoom_neighbors','base_self_training','multifamily_assignment','trusted_source_repair','source_contrastive','webinar_hybrid_neighbors']
    data={};output_hashes={}
    for name in names:
        data[name],output_hashes[name]=load(name)
    def primary(name):
        z=data[name];p=z['probabilities']
        if name=='trusted_source_repair':return z['oof_probabilities']
        if p.ndim==3:
            info=json.loads((NOTES/f'json/{name}_frozen.json').read_text())
            i=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);return p[i]
        return p
    candidates={label:primary(name) for label,name in zip(plan['candidates'],names)}
'''+source[last:]
    compile(source,'risk_first_expert.py','exec')
    (NOTES/'risk_first_expert.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
