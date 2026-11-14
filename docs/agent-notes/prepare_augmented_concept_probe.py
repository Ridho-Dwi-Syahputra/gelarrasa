"""Source-level, multi-view concept associations; no new semantic annotations."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'semantic_concept_seed_propagation.py').read_text(encoding='utf-8')
text=text.replace('from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit','from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit\nfrom augmented_source_concepts import load_augmented_source_concepts')
text=text.replace("prefix='semantic_concept_seed'","prefix='augmented_concept_seed'")
text=text.replace('genuine_seed_source_concept_visual_propagation_v1','genuine_source_multiview_concept_association_v1')
text=text.replace("'object_topk':5,'object_ref_min_probability':.05,'object_query_min_probability':.05,'object_min_mapped_mass':.2", "'object_topk':10,'object_reference_views':8,'object_ref_min_probability':.1,'object_query_min_probability':.03,'object_min_mapped_mass':.1")
text=text.replace("'visual_check':'propagatedclassmustmatchnearesthumanprototypeclassinV3PCA128whitenedspace;rawsourceprototypecosflooralsoapplies'", "'visual_check':'queryCNNunitcosagainstmaximumover8viewsofmatchedgenuinesource;purefloor.45;conflictclassmargin.03andcosfloor.55'")
text=text.replace("'guard':'whole34fallbackV3baseline26nondecreasingmatchesandmacro;partialcovered>=3andprecision>=.8;atleast1learnedsourceconcept'", "'guard':'whole34fallbackprototype-plus-residual28nondecreasingmatchesandmacro.85;partialcovered>=5andprecision>=.9'")
text=text.replace("'pure_visual_similarity_min':{'object':.55,'material':.65}", "'pure_visual_similarity_min':{'object':.45,'material':.65}")
text=text.replace("'conflicted_concept':'onlyresolvebyV3nearestprototypewithrawcos>=.65andwhitenedclassmargin>=.05;otherwiseabstain'", "'conflicted_concept':'per-sourcemaxviewcos+.1logmaximumreferenceconceptmass;classmargin.03;querycos.55;sourcevotesnotviewcounts'")
text=text.replace("'object':(object_idx[:,:5],object_prob[:,:5],.05,.2,visual,.55)", "'object':(object_idx[:,:10],object_prob[:,:10],.03,.1,visual,.45)")
start=text.index("    baseline_matches=int(");end=text.index('    oof={',start)
text=text[:start]+'''    baseclass=baseline.argmax(1)
    augmodel=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz',allow_pickle=False);res=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz',allow_pickle=False)
    for bank in [augmodel,res]:assert np.array_equal(bank['keys'],keys)
    ap,am,ar=augmodel['oof_predictions'],augmodel['oof_margins'],augmodel['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac
    acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45)
    baseclass=np.where(acov,ac,baseclass);ri=json.loads((NOTES/'json/residual_graph_correction_frozen.json').read_text());rwi=max(range(len(ri['selected_heads'])),key=lambda i:(ri['selected_heads'][i]['development_balanced_recall'],ri['selected_heads'][i]['anchor_matches']));rp=res['probabilities'];rc=rp[rwi].argmax(1);rs=np.min([p[np.arange(n),rc] for p in rp],0);route=(rs>=.6)&~(acov&(ac!=rc));baseclass=np.where(route,rc,baseclass);baseline=np.eye(8)[baseclass]
    baseline_matches=int((baseclass[pos]==gold).sum());baseline_macro=float(balanced_accuracy_score(gold,baseclass[pos]));assert baseline_matches==28
    augidx,augprob,augcnn,augsha=load_augmented_source_concepts(s.key.to_numpy(str));cnn=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);assert np.array_equal(cnn['keys'],keys);cnnquery=unit(cnn['features'])
''' +text[end:]
old='''        for h in hs:
            for j,p in zip(idx[pos[h]],prob[pos[h]]):
                if p>=minprob:refs.setdefault(int(j),[]).append(int(h))'''
new='''        concept_strength={}
        for h in hs:
            indices=augidx[h].ravel() if name=='object' else idx[pos[h]]
            masses=augprob[h].ravel() if name=='object' else prob[pos[h]]
            for j,p in zip(indices,masses):
                threshold=.1 if name=='object' else minprob
                if p>=threshold:
                    refs.setdefault(int(j),set()).add(int(h));concept_strength[(int(j),int(h))]=max(float(p),concept_strength.get((int(j),int(h)),0.))
        refs={j:sorted(hs) for j,hs in refs.items()}'''
assert old in text;text=text.replace(old,new)
old='''                if len(labels)==1:
                    if float((pure_visual[q]@pure_visual[pos[sources]].T).max())>=floor:chosen=int(labels[0])
                else:
                    class_score=np.array([(whitened[q]@whitened[pos[sources[gold[sources]==c]]].T).max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])
                    refs_class=sources[gold[sources]==candidate]
                    if class_score[order[0]]-class_score[order[1]]>=.05 and float((visual[q]@visual[pos[refs_class]].T).max())>=.65:chosen=candidate'''
new='''                if name=='object':
                    local_similarity=(augcnn[sources]@cnnquery[q]).max(1)
                    if len(labels)==1:
                        if local_similarity.max()>=.45:chosen=int(labels[0])
                    else:
                        strengths=np.array([concept_strength[(int(j),int(h))] for h in sources]);scores_by_source=local_similarity+.1*np.log(np.maximum(strengths,1e-12))
                        class_score=np.array([scores_by_source[gold[sources]==c].max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])
                        if class_score[order[0]]-class_score[order[1]]>=.03 and local_similarity[gold[sources]==candidate].max()>=.55:chosen=candidate
                elif len(labels)==1:
                    if float((pure_visual[q]@pure_visual[pos[sources]].T).max())>=floor:chosen=int(labels[0])
                else:
                    class_score=np.array([(whitened[q]@whitened[pos[sources[gold[sources]==c]]].T).max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])
                    refs_class=sources[gold[sources]==candidate]
                    if class_score[order[0]]-class_score[order[1]]>=.05 and float((visual[q]@visual[pos[refs_class]].T).max())>=.65:chosen=candidate'''
assert old in text;text=text.replace(old,new)
text=text.replace('covered=(total>=minmass)&(fraction>=.9)&(candidate==nearest_class)','covered=(total>=minmass)&(fraction>=.9)')
text=text.replace('ok=count>=3 and precision>=.8 and matches>=baseline_matches and macro>=baseline_macro-1e-12','ok=count>=5 and precision>=.9 and matches>=baseline_matches and macro>=baseline_macro-1e-12')
text=text.replace('source_concept_cache_sha256=source_sha)', 'source_concept_cache_sha256=source_sha,augmented_concept_cache_sha256=augsha)')
(folder/'augmented_concept_seed.py').write_text(text,encoding='utf-8')
print('Prepared max-per-source multi-view concept associations; same34annotations and28/34fallback.')
