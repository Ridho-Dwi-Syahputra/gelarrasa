"""Prepare a DN4-inspired descriptor measure, without reproducing episodic DN4."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'augmented_prototype_consensus.py').read_text(encoding='utf-8')
text=text.replace("prefix='augmented_prototype_consensus'","prefix='local_descriptor_consensus'")
text=text.replace('three_family_augmented_genuine_source_prototypes_v1','two_scale_genuine_seed_local_descriptor_consensus_v1')
text=text.replace("'views_per_source':8", "'views_per_source':1,'reference':'https://arxiv.org/abs/1903.12290','adaptation':'frozenConvNeXtpatches,fixedprojection128,trainmeanremoval;notepisodicallytrainedDN4'")
text=text.replace("'banks':['V2CLS_ROImaxview','CNNfullsourceT1T2maxview','V3CLS_PCA128whiteoriginal']", "'banks':['fine7x7projected128','coarse7x7projected128']")
text=text.replace("'source_vote':'maxviewcosinesimilarityperSOURCE,thenmaxperclass;noautomaticlabeltargets'", "'source_vote':'mean3nearestreferencepatchcosineperclass,softROIweightedquerymean;34genuineonly;refsweight>=.25max'")
text=text.replace("'candidate':'atleast2of3banksagree;minmarginandminrawsimilarityonagreeingbanks;3bankconflictsabstain'", "'candidate':'bothpatchscalesagree;minimumclassmarginandminimumabsolutewinninglocalscore'")
text=text.replace("'margin_grid':[.02,.05,.1],'raw_similarity_grid':[.45,.55,.65]", "'margin_grid':[.0,.01,.03,.05],'raw_similarity_grid':[.3,.5,.7]")
text=text.replace("'normalizer':'centeringfitouterTRAINonly;V3PCAwhiteningfitouterTRAINonly;all8querysourceviewsheldouttogether'", "'normalizer':'fixedrandomprojectionnotdatafitted;patchmeanfitouterTRAINonly;allquerysourcegroupsheldoutfromreferences'")
text=text.replace("'guard':'matches>=26andmacro>=.783333333333;covered>=5,precision>=.9'", "'guard':'matches>=27andmacro>=.808333333333;covered>=5,precision>=.9'")
start=text.index('    v2=np.load(');end=text.index('    def route(',start)
new='''    index_path=NOTES/'.runtime/convnext_patch_token_index.json';index=json.loads(index_path.read_text());assert np.array_equal(index['keys'],keys)
    bags=[];weights=[]
    rng=np.random.default_rng(20261108)
    for name,dimension in [('fine',192),('coarse',768)]:
        record=index['files'][name];path=ROOT/record['path'];assert sha(path)==record['sha256'];raw=np.load(path,mmap_mode='r',allow_pickle=False)
        weightrecord=index['files'][name+'_weight'];wp=ROOT/weightrecord['path'];assert sha(wp)==weightrecord['sha256'];w=np.load(wp,allow_pickle=False).astype(np.float32)
        if name=='fine':
            raw=np.asarray(raw,dtype=np.float32).reshape(n,7,2,7,2,192).mean((2,4)).reshape(n,49,192)
            w=w.reshape(n,7,2,7,2).mean((2,4)).reshape(n,49)
        projection=rng.normal(size=(dimension,128)).astype(np.float32)/np.sqrt(128)
        with threadpool_limits(limits=2):bag=np.asarray(raw,dtype=np.float32)@projection
        bags.append(bag);weights.append(w)
    def infer(train,query,hs):
        predictions=[];margins=[];rawscores=[]
        for raw,w in zip(bags,weights):
            mean=np.average(raw[train].reshape(-1,128),axis=0,weights=w[train].ravel())
            x=norm(raw-mean);scores=np.full((len(query),8),-1.,np.float32)
            for category in range(8):
                source=pos[hs[gold[hs]==category]]
                if not len(source):continue
                keep=w[source]>=.25*np.maximum(w[source].max(1,keepdims=True),1e-6);references=x[source][keep];k=min(3,len(references))
                for first in range(0,len(query),32):
                    ids=query[first:first+32];sim=x[ids].reshape(-1,128)@references.T
                    local=np.partition(sim,-k,axis=1)[:,-k:].mean(1).reshape(len(ids),49)
                    scores[first:first+len(ids),category]=(local*w[ids]).sum(1)/np.maximum(w[ids].sum(1),1e-12)
            order=np.argsort(-scores,axis=1,kind='stable');pred=order[:,0];winning=scores[np.arange(len(query)),pred];margin=winning-scores[np.arange(len(query)),order[:,1]]
            predictions.append(pred);margins.append(margin);rawscores.append(winning)
        return np.stack(predictions),np.stack(margins),np.stack(rawscores)
'''
text=text[:start]+new+text[end:]
text=text.replace('np.empty((3,n)', 'np.empty((2,n)').replace('np.zeros((3,n)', 'np.zeros((2,n)')
text=text.replace('ok=matches>=26 and macro>=.783333333333-1e-12','ok=matches>=27 and macro>=.808333333333-1e-12')
text=text.replace('augmented_source_prototype_snapshot','local_descriptor_seed_snapshot')
(folder/'local_descriptor_consensus.py').write_text(text,encoding='utf-8')
print('Prepared fixed two-scale, source-held-out local descriptor test; no new image inference.')
