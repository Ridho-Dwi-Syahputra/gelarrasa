"""Broader posterior support ablation, frozen separately before evaluation."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'weighted_source_concept.py').read_text(encoding='utf-8')
text=text.replace("prefix='weighted_source_concept'","prefix='broad_source_concept'")
text=text.replace('genuine_seed_weighted_source_concept_visual_propagation_v2','genuine_seed_broad_source_concept_visual_propagation_v3')
text=text.replace("'object_topk':5,'object_ref_min_probability':.05,'object_query_min_probability':.05,'object_min_mapped_mass':.2", "'object_topk':10,'object_ref_min_probability':.01,'object_query_min_probability':.01,'object_min_mapped_mass':.1")
text=text.replace("'pure_visual_floor_override':.65", "'pure_visual_floor_override':.45")
text=text.replace('rawcosfloor.65','rawcosfloor.45').replace('>=max(.65,floor):chosen=int(labels[0])','>=floor:chosen=int(labels[0])')
text=text.replace("'object':.55,'material':.65", "'object':.45,'material':.65")
text=text.replace("'object':(object_idx[:,:5],object_prob[:,:5],.05,.2,visual,.55)", "'object':(object_idx[:,:10],object_prob[:,:10],.01,.1,visual,.45)")
text=text.replace('>=.65:chosen=candidate','>=.45:chosen=candidate')
text=text.replace('withrawcos>=.65andwhitenedclassmargin','withrawcos>=.45andweightedwhitenedclassmargin')
text=text.replace("'guard':'whole34fallbackV3baseline26nondecreasingmatchesandmacro;partialcovered>=3andprecision>=.8;atleast1learnedsourceconcept'", "'guard':'whole34matches>=27andmacro>=.808333333333;partialcovered>=5andprecision>=.9;atleast1learnedsourceconcept'")
text=text.replace('ok=count>=3 and precision>=.8 and matches>=baseline_matches and macro>=baseline_macro-1e-12', 'ok=count>=5 and precision>=.9 and matches>=27 and macro>=.808333333333-1e-12')
(folder/'broad_source_concept.py').write_text(text,encoding='utf-8')
print('Prepared broad source posterior support probe; top10, .01prob, .1mass, .45cos; stronger development guard27/34.')
