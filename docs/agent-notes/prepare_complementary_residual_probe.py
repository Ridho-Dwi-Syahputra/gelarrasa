"""Prepare strict warm-residual routing where the cold specialist abstains."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'residual_selective_router.py').read_text(encoding='utf-8')
text=text.replace("prefix='residual_selective_router'","prefix='complementary_residual_router'")
text=text.replace('frozen_prototype_plus_residual_selective_router_v1','prototype_cold_residual_and_strict_warm_complement_v1')
text=text.replace("'inputs':['augmented_prototype_consensus','residual_graph_correction','dinov3_base_neighbors']", "'inputs':['augmented_prototype_consensus','residual_graph_correction','warm_residual_correction','dinov3_base_neighbors']")
text=text.replace("'routing':'residualtwo-bankminimumwinningprobability>=.6;doNOTreplaceconflictingqualifiedprototype;otherwiseresidual;baselinefallback'", "'routing':'cold.6unchanged;warmonlywherecoldabstainsandprototypehasnoconflict;warmthresholdgrid.8/.9/.95;neveroverrideconfidentcold'")
text=text.replace("'guard':'whole34>=27matchesandmacro>=.808333333333;residualcovered>=18,precision>=.9;noclassquotaorIDrules'", "'guard':'whole34>=29matchesandmacro>=.85;residualunioncovered>=20,precision>=.9;noclassquotaorIDrules'")
old="""    covered=int(rcov[pos].sum());correct=int((rcov[pos]&(primary[pos]==gold)).sum());matches=int((combined[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,combined[pos]));accepted=bool(matches>=27 and macro>=.808333333333-1e-12 and covered>=18 and correct/covered>=.9)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);final_proto_conflict=aug['covered']&(aug['predictions']!=primary)
    gate=rcov&~final_proto_conflict&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True"""
new="""    warm=np.load(NOTES/'.runtime/warm_residual_correction_outputs.npz',allow_pickle=False);assert np.array_equal(warm['keys'],keys)
    winfo=json.loads((NOTES/'json/warm_residual_correction_frozen.json').read_text());wwi=max(range(len(winfo['selected_heads'])),key=lambda i:(winfo['selected_heads'][i]['development_balanced_recall'],winfo['selected_heads'][i]['anchor_matches']))
    wc=warm['probabilities'][wwi].argmax(1);ws=np.min([p[np.arange(n),wc] for p in warm['probabilities']],0);development=[]
    for threshold in [.8,.9,.95]:
        wroute=(ws>=threshold)&~rcov&~(acovered&(ac!=wc));candidate=np.where(wroute,wc,combined)
        union=rcov|wroute;uc=np.where(wroute,wc,primary);covered=int(union[pos].sum());correct=int((union[pos]&(uc[pos]==gold)).sum());matches=int((candidate[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,candidate[pos]));eligible=bool(matches>=29 and macro>=.85-1e-12 and covered>=20 and correct/covered>=.9)
        development.append({'threshold':threshold,'anchor_matches':matches,'development_balanced_recall':macro,'residual_covered':covered,'residual_correct':correct,'eligible':eligible})
    choices=[r for r in development if r['eligible']];chosen=max(choices,key=lambda r:(r['anchor_matches'],r['development_balanced_recall'],r['residual_correct']/r['residual_covered'],r['residual_covered'],r['threshold'])) if choices else development[0];accepted=bool(choices)
    covered,correct,matches,macro=[chosen[k] for k in ['residual_covered','residual_correct','anchor_matches','development_balanced_recall']]
    score=ws;primary=wc;current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool);final_proto_conflict=aug['covered']&(aug['predictions']!=primary)
    gate=(score>=chosen['threshold'])&~rcov&~final_proto_conflict&~human&(primary!=current);proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)"""
assert old in text;text=text.replace(old,new)
text=text.replace("'minimum_corrected_probability':score,'residual_covered':rcov", "'minimum_corrected_probability':score,'cold_residual_covered':rcov")
(folder/'complementary_residual_router.py').write_text(text,encoding='utf-8')
print('Prepared strict complementary specialist; guard29/34 fixed before evaluation.')
