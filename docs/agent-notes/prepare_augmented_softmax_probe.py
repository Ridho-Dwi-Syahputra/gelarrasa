"""Prepare a separate learned source-view head; preserve frozen NN experiment."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'augmented_prototype_consensus.py').read_text(encoding='utf-8')
text=text.replace('from sklearn.decomposition import PCA','from sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression')
text=text.replace("prefix='augmented_prototype_consensus'","prefix='augmented_source_softmax'")
text=text.replace('three_family_augmented_genuine_source_prototypes_v1','three_family_augmented_genuine_source_softmax_v1')
text=text.replace("'source_vote':'maxviewcosinesimilarityperSOURCE,thenmaxperclass;noautomaticlabeltargets'", "'source_vote':'L2multinomialsoftmaxonallfit-sourceviews,C10,classbalanced,eachsourceunitweight;noautomaticlabeltargets'")
text=text.replace("'margin_grid':[.02,.05,.1],'raw_similarity_grid':[.45,.55,.65]", "'margin_grid':[.05,.2,.5],'raw_similarity_grid':[.45,.55,.65]")
text=text.replace("'guard':'matches>=26andmacro>=.783333333333;covered>=5,precision>=.9'", "'guard':'matches>=27andmacro>=.808333333333;covered>=5,precision>=.9'")
text=text.replace('score\':\'minimumcosineclassmargin;notprobabilityofcorrectlabel', 'score\':\'minimumsoftmaxclassmargin;uncalibratedandnotcorrectnessprobability')
text=text.replace('maxviewcanmatchbackground','learnedheadcanoverfit34sources;sourcegroupsheldout')
old="""            sim=(q@r.T).reshape(len(query),len(hs),views.shape[1]).max(-1)
            rawsim=(x[query]@views[hs].reshape(-1,x.shape[1]).T).reshape(len(query),len(hs),views.shape[1]).max(-1)
            scores=np.stack([sim[:,gold[hs]==c].max(1) if (gold[hs]==c).any() else np.full(len(query),-1.) for c in range(8)],1)
            rawscore=np.stack([rawsim[:,gold[hs]==c].max(1) if (gold[hs]==c).any() else np.full(len(query),-1.) for c in range(8)],1)"""
new="""            labels=np.repeat(gold[hs],views.shape[1]);weights=np.full(len(labels),1./views.shape[1])
            head=LogisticRegression(C=10.,class_weight='balanced',max_iter=1000,solver='lbfgs').fit(r,labels,sample_weight=weights)
            scores=np.zeros((len(query),8));scores[:,head.classes_]=head.predict_proba(q)
            rawsim=(x[query]@views[hs].reshape(-1,x.shape[1]).T).reshape(len(query),len(hs),views.shape[1]).max(-1)
            rawscore=np.stack([rawsim[:,gold[hs]==c].max(1) if (gold[hs]==c).any() else np.full(len(query),-1.) for c in range(8)],1)"""
assert old in text;text=text.replace(old,new)
text=text.replace('ok=matches>=26 and macro>=.783333333333-1e-12','ok=matches>=27 and macro>=.808333333333-1e-12')
text=text.replace('augmented_source_prototype_snapshot','genuine_augmented_softmax_snapshot').replace('minimum_cosine_class_margin','minimum_softmax_class_margin')
(folder/'augmented_source_softmax.py').write_text(text,encoding='utf-8')
print('Prepared source-held-out softmax head; C fixed10 before fit.')
