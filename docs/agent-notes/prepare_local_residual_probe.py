"""Predeclare a local, source-posterior residual graph ablation."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'residual_graph_correction.py').read_text(encoding='utf-8')
text=text.replace("prefix='residual_graph_correction'","prefix='local_source_residual'")
text=text.replace('inductive_train_only_residual_correct_and_smooth_v1','inductive_local_source_posterior_residual_v1')
text=text.replace("'banks':['V3CLS_PCA128white','V3_fullsource_centered']", "'banks':['full_source_posterior_centered','V3object_centered']")
text=text.replace('train-onlyrowstochasticgraph,k20,cos^4','train-onlyrowstochasticgraph,k7,cos^8')
text=text.replace("'guard':'primary>=28andmacro>=.808333333333;ORprimary>=27andmacro>=.808333333333andcovered>=10with>=.9precision'", "'guard':'primary>=28andmacro>=.85andcovered>=18with>=.9precision'")
old="banks={'V3CLS_PCA128white':cls,'V3_fullsource_centered':unit(np.c_[cls,.35*unit(sd['features'][:,0]),.35*unit(sd['features'][:,1])])}"
new="banks={'full_source_posterior_centered':unit(np.c_[unit(sd['features'][:,0]),unit(sd['features'][:,1])]),'V3object_centered':unit(np.c_[cls,unit(v3['features'][:,1536:2304])])}"
assert old in text;text=text.replace(old,new)
text=text.replace('[:,:20]','[:,:7]').replace('np.arange(len(train)),20','np.arange(len(train)),7').replace('),0)**4','),0)**8')
text=text.replace('count>=10 and precision>=.9','count>=18 and precision>=.9')
text=text.replace("chosen['development_balanced_recall']>=.808333333333-1e-12 and chosen['anchor_matches']>=27", "chosen['development_balanced_recall']>=.85-1e-12 and chosen['anchor_matches']>=28")
(folder/'local_source_residual.py').write_text(text,encoding='utf-8')
print('Prepared k7/cos8 graph, source posterior and V3 object banks, same34references.')
