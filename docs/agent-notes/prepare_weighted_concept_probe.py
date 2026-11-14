"""Create a separately frozen concept-conditioned, confidence-weighted ablation."""
from pathlib import Path

folder=Path(__file__).resolve().parent
source=(folder/'semantic_concept_seed_propagation.py').read_text(encoding='utf-8')
source=source.replace("prefix='semantic_concept_seed'", "prefix='weighted_source_concept'")
source=source.replace('genuine_seed_source_concept_visual_propagation_v1','genuine_seed_weighted_source_concept_visual_propagation_v2')
source=source.replace("'configs':['object','material','object_then_material']", "'configs':['object','material','object_then_material']")
source=source.replace("'visual_check':'propagatedclassmustmatchnearesthumanprototypeclassinV3PCA128whitenedspace;rawsourceprototypecosflooralsoapplies'", "'visual_check':'conceptconditionedprototypecomparison;nooverallnearestclassveto;rawcosfloor.65;referenceconfidenceweightedclassscore'")
source=source.replace("'class_fraction_min':.9", "'class_fraction_min':.9,'reference_score':'whitecos+.1log(refconceptprob/refdominantconceptprob)', 'pure_visual_floor_override':.65")
source=source.replace("for j,p in zip(idx[pos[h]],prob[pos[h]]):\n                if p>=minprob:refs.setdefault(int(j),[]).append(int(h))", "for j,p in zip(idx[pos[h]],prob[pos[h]]):\n                if p>=minprob:refs.setdefault(int(j),[]).append(int(h))")
old="class_score=np.array([(whitened[q]@whitened[pos[sources[gold[sources]==c]]].T).max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])"
new="""reference_probability=np.array([prob[pos[h],np.flatnonzero(idx[pos[h]]==j)[0]]/max(prob[pos[h],0],1e-12) for h in sources])
                    reference_score=whitened[q]@whitened[pos[sources]].T+.1*np.log(np.maximum(reference_probability,1e-12))
                    class_score=np.array([reference_score[gold[sources]==c].max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])"""
assert old in source
source=source.replace(old,new).replace('>=floor:chosen=int(labels[0])','>=max(.65,floor):chosen=int(labels[0])')
source=source.replace('covered=(total>=minmass)&(fraction>=.9)&(candidate==nearest_class)','covered=(total>=minmass)&(fraction>=.9)')
assert 'candidate==nearest_class' not in source
(folder/'weighted_source_concept.py').write_text(source,encoding='utf-8')
print('Created weighted_source_concept.py; original frozen experiment preserved.')
