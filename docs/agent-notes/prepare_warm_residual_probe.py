"""Warm-base residual ablation with source-excluded human prediction errors."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'residual_graph_correction.py').read_text(encoding='utf-8')
text=text.replace("prefix='residual_graph_correction'","prefix='warm_residual_correction'")
text=text.replace('inductive_train_only_residual_correct_and_smooth_v1','inductive_source_excluded_warm_residual_correct_and_smooth_v2')
text=text.replace("'baseline':'auto-onlycoldcorek7;excludeALLhuman-sourcegroupsfromautocore;fit-rowpredictionsremoveownsourcegroup'", "'baseline':'coldautocoreplusfitfoldgenuinehumansweight32;k7;allfit-rowpredictionsexcludeownsourcegroup;referenceserrornotin-sample'")
text=text.replace("'guard':'primary>=28andmacro>=.808333333333;ORprimary>=27andmacro>=.808333333333andcovered>=10with>=.9precision'", "'guard':'primary>=28andmacro>=.85andcovered>=15with>=.9precision'")
old="weights=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)"
assert old in text;text=text.replace(old,old+";y[pos]=gold;weights[pos]=32.")
old="fit=train[core[train]]\n        prior=base_prediction(x,fit,train)"
new="fit=np.r_[train[core[train]],pos[hs]];assert len(np.unique(fit))==len(fit)\n        prior=base_prediction(x,fit,train)"
assert old in text;text=text.replace(old,new)
text=text.replace('count>=10 and precision>=.9','count>=15 and precision>=.9')
text=text.replace("chosen['development_balanced_recall']>=.808333333333-1e-12 and chosen['anchor_matches']>=27", "chosen['development_balanced_recall']>=.85-1e-12 and chosen['anchor_matches']>=28")
(folder/'warm_residual_correction.py').write_text(text,encoding='utf-8')
print('Prepared warm residual ablation: all source-reference predictions remove own source group.')
