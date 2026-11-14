"""Separate genuine-anchor binary specialists for material/object features."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'source_dictionary_ovr_router.py').read_text(encoding='utf-8')
text=text.replace("prefix='source_dictionary_ovr'","prefix='material_object_specialists'")
text=text.replace('source_dictionary_human_one_vs_rest_abstaining_router_v1','material_object_human_one_vs_rest_abstaining_router_v1')
text=text.replace("'banks':['source_dictionary','DINOregional','CNNdictionary']", "'banks':['materialprob_V3','materialfeature_sourceposterior','litterfeature_V3']")
text=text.replace("'baseline':'bestsemantic_dictionary_neighborsOOF25/34,notcurrentclampedhumanclasses'", "'baseline':'bestDINOv3BaseOOF26/34;strongerguard27andmacro.808333333333'")
text=text.replace("'guard':'>=26/34andmacro>baseline+.000000000001;>=3humanroutedandallroutedhumanpredictionscorrect;atleast1qualifiedclass'", "'guard':'>=27/34andmacro>=.808333333333;>=3humanroutedandallroutedhumanpredictionscorrect;atleast1qualifiedclass'")
old="    banks={'source_dictionary':dictionary,'DINOregional':regional,'CNNdictionary':cnn}"
new="""    material=np.load(NOTES/'.runtime/minc_train_features.npz',allow_pickle=False);litter=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False);v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);full=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False)
    for source in [material,litter,v3,full]:assert np.array_equal(source['keys'],keys)
    cls=unit(v3['features'][:,:768]);mp=unit(np.sqrt(np.maximum(material['probabilities'][:,1:].mean(1),0)));mf=unit(material['features'][:,1:].mean(1));lf=unit(litter['features']);posterior=unit(np.c_[unit(full['features'][:,0]),unit(full['features'][:,1])])
    banks={'materialprob_V3':unit(np.c_[mp,.5*cls]),'materialfeature_sourceposterior':unit(np.c_[mf,.5*posterior]),'litterfeature_V3':unit(np.c_[lf,.5*cls])}"""
assert old in text;text=text.replace(old,new)
text=text.replace("'.runtime/semantic_dictionary_neighbors_outputs.npz'","'.runtime/dinov3_base_neighbors_outputs.npz'").replace("'json/semantic_dictionary_neighbors_frozen.json'","'json/dinov3_base_neighbors_frozen.json'")
text=text.replace('matches>=26 and macro>base_macro+1e-12','matches>=27 and macro>=.808333333333-1e-12')
(folder/'material_object_specialists.py').write_text(text,encoding='utf-8')
print('Prepared material/object classwise specialist ablation;34genuine targets only.')
