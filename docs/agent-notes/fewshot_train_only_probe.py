"""Fixed few-shot/graph probe; train only, no visual-archive labels or test fit.

Automatic CNN-consensus seeds are explicitly NOT human annotations. The review
gallery is a separate pending input for a later supervised run.
"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'
OLD = ROOT / 'arsip/sebelum_rapi_20261107'
CLASSES = ['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
SEED = 20261107


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def source_groups(frame):
    """Exact pixels and pHash<=4, computed on these train rows only."""
    parent = np.arange(len(frame))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a,b):
        a,b = find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    hashes = [int(x,16) for x in frame.phash]
    for i in range(len(frame)):
        for j in range(i):
            if (hashes[i]^hashes[j]).bit_count()<=4 or (frame.iloc[i].pixel_sha256==frame.iloc[j].pixel_sha256):
                union(i,j)
    return np.array([f'train_group_{find(i):04d}' for i in range(len(frame))])


def main():
    started=time.time()
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'seed':SEED,'manual_labels_consumed':0,'seed_type':'automatic independent CNN consensus, not human ground truth',
          'seeds_per_class':5,'train_only_graph':True,'test_rows_in_fit':0,'heldout_rows_in_graph_fit':0,
          'grouping':'train-only exact pixels or pHash Hamming<=4','validation_fraction_groups':.2,
          'candidate_list':['prototype_dino','ridge_dino_waste','graph_dino','graph_dino_waste'],
          'graph_neighbors':20,'graph_alpha':.95,'graph_iterations':200,'edge_weight':'positive cosine cubed',
          'ridge_lambda':.01,'prototype_temperature':.07,'seed_selection':'CNN consensus quality pool then DINO farthest representatives; source-group unique',
          'selection':'highest macro agreement on heldout automatic CNN-consensus anchors; tie uses listed candidate order',
          'archive_labels_read_by_experiment':False,'archive_agreement_used_for_selection':False,
          'researcher_has_seen_prior_archive_audits':True,'active_outputs_modified':False,
          'semantic_accuracy':None,'protected_sha256':protected,
          'papers':['https://www.microsoft.com/en-us/research/publication/learning-local-global-consistency/',
                    'https://proceedings.mlr.press/v119/ziko20a.html',
                    'https://research.google/pubs/fixmatch-simplifying-semi-supervised-learning-with-consistency-and-confidence/']}
    dump(NOTES/'json/fewshot_train_only_plan.json',plan)
    manifest=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'sample_id':str,'parent_id':str})
    working=manifest[manifest.split.eq('train')&~manifest.is_crop].sort_values('parent_id').copy()
    profile=pd.read_csv(NOTES/'image_profile.csv',dtype={'image_id':str,'phash':str})
    frame=working[['parent_key','parent_id','source_path','source_sha256','assigned_class']].rename(columns={'parent_key':'key','parent_id':'image_id'})
    frame=frame.merge(profile[['key','pixel_sha256','phash']],on='key',validate='one_to_one').reset_index(drop=True)
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    assert all(sha(ROOT/r.source_path)==r.source_sha256 for r in frame.itertuples())
    # Vectorized exact-pixel IDs avoid repeated dataframe lookups in grouping.
    raw_hashes=[int(v,16) for v in frame.phash]; pixels=frame.pixel_sha256.to_numpy()
    parent=np.arange(len(frame))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for i in range(len(frame)):
        for j in range(i):
            if (raw_hashes[i]^raw_hashes[j]).bit_count()<=4 or pixels[i]==pixels[j]:
                a,b=find(i),find(j)
                if a!=b:parent[max(a,b)]=min(a,b)
    groups=np.array([f'train_group_{find(i):04d}' for i in range(len(frame))])
    fit,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(frame,groups=groups))
    assert not set(groups[fit]) & set(groups[val])
    positions={key:i for i,key in enumerate(frame.key)}
    source_hashes={}
    def block(relative):
        path=OLD/relative; z=np.load(path,allow_pickle=False)
        index={key:i for i,key in enumerate(z['keys'])}; ids=np.array([index[k] for k in frame.key])
        source_hashes[relative]=sha(path)
        return z,ids
    d,di=block('output/embeddings/stage2_repr_v1/dinov2_features.npz')
    xd=unit(d['features'][di].astype('float32'))
    generic,gi=block('output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz')
    source_probabilities=[generic['prior_probabilities'][gi]];coverages=[np.ones(len(frame),dtype='float32')]
    translation={'battery':'electronic','biological':'foodwaste','brown-glass':'glass','green-glass':'glass','white-glass':'glass','glass':'glass','clothes':'textile','shoes':'textile',**{c:c for c in ['cardboard','paper','metal','plastic']}}
    second_features=None
    for stage,folder in [(7,'stage7_waste_cnn_v1'),(14,'stage14_second_waste_cnn_v1')]:
        z,zi=block(f'output/embeddings/{folder}/original224.npz')
        info=json.loads((OLD/f'reports/errors/stage{stage}_waste_cnn_20261106/checkpoint_and_ontology.json').read_text()) if stage==7 else json.loads((OLD/'reports/errors/stage14_second_waste_cnn_20261106/checkpoint_and_ontology.json').read_text())
        probabilities=z['probabilities'][zi]
        evidence=np.zeros((len(frame),8),dtype='float32')
        for column,name in enumerate(info['source_class_order']):
            if name in translation:evidence[:,CLASSES.index(translation[name])]+=probabilities[:,column]
        coverage=evidence.sum(1);source_probabilities.append(evidence/np.maximum(coverage[:,None],1e-8));coverages.append(coverage)
        if stage==14:second_features=unit(z['features'][zi].astype('float32'))
    expert,ei=block('output/embeddings/stage11_ontology_cnn_v1/original224.npz')
    source_probabilities.append(expert['conditional'][ei]);coverages.append(expert['coverage'][ei])
    src=np.stack(source_probabilities);coverage=np.stack(coverages)
    reliable=(src.max(2)>=.8)&(coverage>=np.array([0,.6,.6,.2])[:,None])
    votes=np.zeros((len(frame),8),dtype='float32')
    for i in range(len(src)):votes[np.flatnonzero(reliable[i]),src[i,reliable[i]].argmax(1)]+=1
    y=votes.argmax(1);rank=np.sort(votes,axis=1);clean=(rank[:,-1]>=2)&(rank[:,-1]>rank[:,-2])
    quality=rank[:,-1]+.01*np.max(src,axis=(0,2))
    xm=unit(np.concatenate([xd,second_features],axis=1))
    fitset=set(fit);seeds=[]
    for category in range(8):
        pool=np.array([i for i in fit if clean[i] and y[i]==category])
        assert len(pool)>=5,(CLASSES[category],len(pool))
        pool=pool[np.argsort(-quality[pool],kind='stable')[:80]]
        center=unit(xd[pool].mean(0,keepdims=True))[0]
        selected=[int(pool[np.argmax(xd[pool]@center)])]
        while len(selected)<5:
            allowed=np.array([i for i in pool if groups[i] not in set(groups[selected])])
            assert len(allowed)>0
            distance=1-(xd[allowed]@xd[selected].T).max(1)
            selected.append(int(allowed[np.argmax(distance)]))
        seeds.extend(selected)
    seeds=np.array(seeds);assert len(seeds)==40 and set(seeds)<=fitset
    seed_table=frame.iloc[seeds][['key','image_id','source_path','source_sha256']].copy()
    seed_table['suggested_class']=np.array(CLASSES)[y[seeds]]
    seed_table['source']='automatic CNN consensus; pending team annotation'
    seed_table.sort_values('image_id').to_csv(NOTES/'table-csv/seed_review_candidates.csv',index=False)
    gallery(seed_table.sort_values('image_id'))
    frame[['key','image_id']].assign(group=groups,partition=np.where(np.isin(np.arange(len(frame)),fit),'fit','heldout')).to_csv(NOTES/'table-csv/fewshot_train_only_split.csv',index=False)
    truth=clean[val]; assert all(np.any(truth & (y[val]==c)) for c in range(8))
    def metric(pred):return float(np.mean([(pred[val][truth&(y[val]==c)]==c).mean() for c in range(8)]))
    def graph(x):
        train=x[fit];sim=train@train.T;np.fill_diagonal(sim,-np.inf)
        neighbors=np.argsort(-sim,axis=1,kind='stable')[:,:20]
        weights=np.maximum(np.take_along_axis(sim,neighbors,axis=1),0)**3
        edges=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(fit)),20),neighbors.ravel())),shape=(len(fit),len(fit)))
        edges=(edges+edges.T)*.5;degree=np.asarray(edges.sum(1)).ravel();norm=sparse.diags(1/np.sqrt(np.maximum(degree,1e-10)));adj=norm@edges@norm
        seed_rows={global_i:local_i for local_i,global_i in enumerate(fit)}
        target=np.zeros((len(fit),8),dtype='float32')
        for i in seeds:target[seed_rows[int(i)],y[i]]=1
        f=target.copy()
        for _ in range(200):f=.95*(adj@f)+.05*target
        f/=np.maximum(f.sum(1,keepdims=True),1e-10)
        query=x[val]@train.T;nn=np.argsort(-query,axis=1,kind='stable')[:,:20]
        w=np.maximum(np.take_along_axis(query,nn,axis=1),0)**3
        q=(f[nn]*w[:,:,None]).sum(1)/np.maximum(w.sum(1,keepdims=True),1e-10)
        result=np.zeros((len(x),8),dtype='float32');result[fit]=f;result[val]=q
        return result
    candidates={}
    with threadpool_limits(limits=4):
        centers=unit(np.stack([xd[seeds[y[seeds]==c]].mean(0) for c in range(8)]))
        candidates['prototype_dino']=softmax((xd@centers.T)/.07,axis=1)
        target=np.eye(8,dtype='float32')[y[seeds]];train=xm[seeds]
        coefficients=np.linalg.solve(train@train.T+.01*np.eye(40),target)
        candidates['ridge_dino_waste']=softmax((xm@train.T@coefficients)/.07,axis=1)
        candidates['graph_dino']=graph(xd)
        candidates['graph_dino_waste']=graph(xm)
    history=[];fingerprints={}
    baseline=np.array([CLASSES.index(name) for name in frame.assigned_class])
    for name,probability in candidates.items():
        pred=probability.argmax(1)
        table=frame[['key','image_id']].copy();table['predicted_class']=np.array(CLASSES)[pred];table['score']=probability.max(1)
        table['method']=name;table['ground_truth_verified']=False
        path=NOTES/f'table-csv/fewshot_{name}.csv';table.to_csv(path,index=False);fingerprints[name]=sha(path)
        history.append({'candidate':name,'heldout_automatic_anchor_macro_agreement':metric(pred),'occupied_classes':len(set(pred)),
                        'changed_vs_baseline':int((pred!=baseline).sum()),'class_counts':dict(zip(CLASSES,np.bincount(pred,minlength=8).tolist()))})
    selected=max(history,key=lambda r:r['heldout_automatic_anchor_macro_agreement'])
    report={'status':'passed','train_rows':1906,'fit_rows':len(fit),'heldout_rows':len(val),'test_rows_in_fit':0,'manual_annotations_consumed':0,
            'seeds_per_class':dict(Counter(seed_table.suggested_class)),'heldout_consensus_anchor_rows':int(truth.sum()),
            'baseline_heldout_automatic_anchor_macro_agreement':metric(baseline),'candidates':history,'selected_before_archive_audit':selected['candidate'],
            'selected_prediction_sha256':fingerprints[selected['candidate']],'all_prediction_sha256':fingerprints,
            'source_feature_sha256':source_hashes,'code_sha256':sha(__file__),'archive_labels_read':False,'semantic_accuracy':None,
            'active_outputs_modified':False,'seconds':time.time()-started,'human_seed_review_ready':True}
    dump(NOTES/'json/fewshot_selection_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k not in ['all_prediction_sha256','source_feature_sha256','code_sha256']},indent=2),flush=True)


def gallery(frame):
    rows=[]
    for r in frame.itertuples():
        image=html.escape(os.path.relpath(ROOT/r.source_path,NOTES).replace('\\','/'),quote=True)
        options=''.join(f'<option>{name}</option>' for name in CLASSES)
        rows.append(f'<article data-id="{r.image_id}" data-sha="{r.source_sha256}"><a href="{image}" target="_blank"><img src="{image}" loading="lazy"></a><b>{r.image_id}</b><p>Usulan program: {r.suggested_class}</p><select><option value="">Belum diperiksa</option>{options}</select></article>')
    text='''<!doctype html><html lang="id"><meta charset="utf-8"><title>Acuan train: maksimal 5 per kelas</title>
<style>body{font:16px system-ui;margin:24px;color:#162533}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}article{border:1px solid #ccd5df;padding:12px}img{width:100%;height:230px;object-fit:contain}b{display:block}select,input,button{padding:10px;font:inherit}header{position:sticky;top:0;background:white;padding:12px 0}p{line-height:1.5}</style>
<header><h1>Acuan train untuk diperiksa tim</h1><p>40 kandidat dipilih dari konsensus program dan keragaman visual, tanpa membaca arsip label. Pilih kategori hanya setelah memeriksa gambar. Usulan program bukan label manusia. Gambar ambigu boleh dibiarkan kosong. Maksimal 5 gambar unik per kelas; tidak ada data test di galeri ini.</p><input id="reviewer" placeholder="Nama pemeriksa dari tim"><button id="export">Unduh label hasil pemeriksaan</button><p id="counts"></p></header><main>'''+''.join(rows)+'''</main><script>
const classes='''+json.dumps(CLASSES)+''';
function data(){return Array.from(document.querySelectorAll('article')).filter(a=>a.querySelector('select').value).map(a=>({image_id:a.dataset.id,label:a.querySelector('select').value,source_sha256:a.dataset.sha}));}
function counts(){const c=Object.fromEntries(classes.map(x=>[x,0]));data().forEach(r=>c[r.label]++);return c;}
function refresh(){document.getElementById('counts').textContent=JSON.stringify(counts());}
document.querySelectorAll('select').forEach(s=>s.addEventListener('change',refresh));refresh();
document.getElementById('export').onclick=()=>{const c=counts(),name=document.getElementById('reviewer').value.trim();if(!name)return alert('Isi nama pemeriksa dari tim.');if(classes.some(k=>c[k]>5))return alert('Maksimal 5 gambar per kelas.');if(classes.some(k=>c[k]<1))return alert('Perlu minimal satu acuan yang diperiksa untuk setiap kelas.');const payload={source:'team_manual_review',reviewer:name,reviewed_at:new Date().toISOString(),maximum_per_class:5,labels:data()};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='human_seeds_max5_per_class.json';a.click();URL.revokeObjectURL(a.href);};
</script></html>'''
    (NOTES/'seed_review.html').write_bytes(text.encode('utf-8'))


if __name__=='__main__':main()
