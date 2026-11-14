"""Separate kernel residual interpolation, not another graph smoothing grid."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'warm_residual_correction.py').read_text(encoding='utf-8')
text=text.replace("prefix='warm_residual_correction'","prefix='kernel_residual_interpolation'")
text=text.replace('inductive_source_excluded_warm_residual_correct_and_smooth_v2','source_excluded_genuine_error_kernel_ridge_v1')
text=text.replace("configs=[{'alpha':alpha,'scale':scale,'smooth':smooth} for alpha in [.7,.95] for scale in [1.,2.,4.] for smooth in [False,True]]", "configs=[{'gamma':gamma,'ridge':ridge,'scale':scale} for gamma in [2.,8.,32.] for ridge in [.01,.1,1.] for scale in [1.,2.]]")
text=text.replace("'reference':'https://arxiv.org/2010.13993'", "'reference':'kernelridgeonhumanresiduals'")
text=text.replace("'adaptation':'frozennonVLMfeatures,auto-onlycoldkNNbaseline,34genuineresidualreferences,inductivequery;notpaperreproduction'", "'adaptation':'RBFkernelridgeonwarmkNNsource-excludederrors;onlygenuine34residualtargets;noglobalreviewtargets'")
text=text.replace("'residual':'onehotgenuine-minus-auto-onlyprediction;clampedreferenceerror;train-onlyrowstochasticgraph,k20,cos^4;queryinterpolationafterfit'", "'residual':'e=humanonehot-minus-sourceexcludedwarmbaseline;RBFk(x,h)=exp(-gamma(1-cos));correction=Kqh(Khh+ridgeI)^-1e'")
text=text.replace("'correction':'fixedscale1/2/4;cliprenormalize;optional.5clampedhumanpredictiondiffusion'", "'correction':'fixedscale1/2;cliprenormalize;nohumanclampongraph,nographpseudolabels'")
text=text.replace("'guard':'primary>=28andmacro>=.85andcovered>=15with>=.9precision'", "'guard':'primary>=29andmacro>=.85andcovered>=15with>=.9precision'")
start=text.index('    def setup(');end=text.index('    selections=[];',start)
new='''    def setup(x,train,query,hs):
        references=pos[hs];fit=np.r_[train[core[train]],references];assert len(np.unique(fit))==len(fit)
        reference_prior=base_prediction(x,fit,references);query_prior=base_prediction(x,fit,query)
        residual=np.eye(8)[gold[hs]]-reference_prior
        return query_prior,x[query]@x[references].T,x[references]@x[references].T,residual
    def infer(cache,hs,cfg,query_mode):
        prior,query_similarity,reference_similarity,error=cache
        k=np.exp(-cfg['gamma']*np.maximum(1-reference_similarity,0));kq=np.exp(-cfg['gamma']*np.maximum(1-query_similarity,0))
        coefficient=np.linalg.solve(k+cfg['ridge']*np.eye(len(k)),error)
        corrected=np.maximum(prior+cfg['scale']*(kq@coefficient),1e-8);corrected/=corrected.sum(1,keepdims=True)
        return corrected
'''
text=text[:start]+new+text[end:]
text=text.replace("chosen['anchor_matches']>=28", "chosen['anchor_matches']>=29")
text=text.replace('residual_graph_correction_snapshot','kernel_residual_interpolation_snapshot')
(folder/'kernel_residual_interpolation.py').write_text(text,encoding='utf-8')
print('Prepared genuine residual kernel interpolation; no graph or new labels.')
