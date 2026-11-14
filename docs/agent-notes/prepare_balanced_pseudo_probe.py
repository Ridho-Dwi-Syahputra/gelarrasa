"""Derive a separate, single-round class-budgeted probe from the frozen baseline.

Not a continuation of the withdrawn unbounded feedback protocol. New hypothesis:
class/source-diverse pseudo bank, bounded output budget per predicted category.
No review inventory or image annotation archive is read.
"""
from fewshot_train_only_probe import NOTES


def main():
    source = (NOTES/'current_pseudo_anchor_refinement.py').read_text(encoding='utf-8')
    changes = [
        ("prefix='current_pseudo_anchor' if round_number==1 else f'current_pseudo_anchor_r{round_number}'", "assert round_number==1\n    prefix='balanced_pseudo_anchor'"),
        ("'method':'current_program_pseudo_anchor_neighborhood_transfer_v1'", "'method':'class_budgeted_source_diverse_pseudo_anchor_transfer_v1','maximum_rounds':1,'maximum_pseudo_sources_per_class':20,'maximum_changes_per_predicted_class':8,'selection_hypothesis':'reducepseudo-classconcentration;notassumeequaltrueclasssizes;notDARPimplementation'"),
        ("'bank_fixed_before_fit':True", "'bank_fixed_before_fit':True,'pseudo_source_selection':'onegeometrysourcepercategory,farthestfirstDINOregional,highestprogramscoreinitial;humanalwaysretained;atmost20nonhuman/class'"),
        ("'gate':'atleast3/5banks agree", "'gate':'atleast4/5banks agree"),
        ("anchor=human|(~old_source&(score>=threshold));pos=np.flatnonzero(anchor)", """eligible_pseudo=~human&~old_source&(score>=threshold)
    anchor=human.copy()
    for category in range(8):
        occupied=set(groups[human&(current==category)])
        candidates=np.array([i for i in np.flatnonzero(eligible_pseudo&(current==category)) if groups[i] not in occupied],dtype=int)
        if not len(candidates):continue
        first=int(candidates[np.argsort(-score[candidates],kind='stable')[0]])
        selected=[first];used_groups={groups[first]}|set(groups[human&(current==category)])
        while len(selected)<20:
            allowed=np.array([i for i in candidates if groups[i] not in used_groups],dtype=int)
            if not len(allowed):break
            reference=np.r_[np.flatnonzero(human&(current==category)),selected]
            distance=1-(banks['base_local'][allowed]@banks['base_local'][reference].T).max(1)
            chosen=int(allowed[np.argmax(distance)]);selected.append(chosen);used_groups.add(groups[chosen])
        assert len(set(groups[selected]))==len(selected) and not(set(groups[selected])&occupied)
        anchor[selected]=True
    pos=np.flatnonzero(anchor)
    assert all(((anchor&~human)&(current==k)).sum()<=20 for k in range(8))"""),
        ("gate=(support>=3)", "gate=(support>=4)"),
        ("proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-mean_probability,-support));allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True", """proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-mean_probability,-support))
    allowed=[];class_budget=np.zeros(8,int)
    for i in rank[gate[rank]]:
        if class_budget[candidate[i]]>=8:continue
        allowed.append(i);class_budget[candidate[i]]+=1
        if len(allowed)>=40:break
    gate[:]=False;gate[np.array(allowed,dtype=int)]=True
    assert all((gate&(candidate==k)).sum()<=8 for k in range(8))"""),
        ("'cap':'descending agreeing-bank support,then mean classprobability,stablekeyorder'", "'cap':'atmost8perpredictedclassand40total;descending agreeing-bank support,then mean classprobability,stablekeyorder'"),
    ]
    for old, new in changes:
        assert source.count(old)==1, (old, source.count(old))
        source=source.replace(old,new)
    source='''"""Single-round class-budgeted pseudo-source ablation.
All labels are current program pseudo labels except the same34human sources.
Only program consistency is measured; no independent semantic Accuracy.
"""\n'''+source[source.index('import json'):]
    compile(source,'balanced_pseudo_anchor_refinement.py','exec')
    (NOTES/'balanced_pseudo_anchor_refinement.py').write_text(source,encoding='utf-8')


if __name__=='__main__':main()
