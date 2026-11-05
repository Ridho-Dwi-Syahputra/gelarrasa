"""Small SCAN-inspired head experiment, not a reproduction of full SCAN.

Original implementation of neighbor consistency minus marginal entropy,
following Van Gansbeke et al., ECCV 2020, Eq. 2:
https://arxiv.org/html/2005.12320v2
Adaptations: frozen DINOv2, full-batch graph, KMeans initialization, no image
augmentation in training or self-labeling stage. No semantic targets are read.
"""
from pathlib import Path
import hashlib
import json
import time
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/groups/stage2_neighbor_v1'
REP = ROOT / 'reports/errors/stage2_20261105/neighbor_refinement'
SEEDS = [11,29,47]
WEIGHTS = [.5,2.0]
EPOCHS = 120


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')


def graph(x, pool, hashes, pool_hashes):
    s = x @ pool.T
    s[hashes[:,None] == pool_hashes[None,:]] = -np.inf
    return torch.from_numpy(np.argsort(-s,axis=1,kind='stable')[:,:10].copy())


def initialize(x, seed):
    km = KMeans(n_clusters=8,n_init=10,max_iter=300,random_state=seed).fit(x)
    head = torch.nn.Linear(x.shape[1],8)
    c = torch.from_numpy(km.cluster_centers_)
    with torch.no_grad():
        head.weight.copy_(20*c)
        head.bias.copy_(-10*c.square().sum(1))
    return head


def components(p, q, nn):
    dot = (p[:,None,:]*q[nn]).sum(-1).clamp_min(1e-8)
    consistency = -dot.log().mean()
    avg = p.mean(0).clamp_min(1e-8)
    entropy = -(avg*avg.log()).sum()
    return consistency,entropy


def evaluate(head, x, train_x, nn):
    with torch.no_grad():
        p = head(x).softmax(-1)
        q = head(train_x).softmax(-1)
        c,h = components(p,q,nn)
        labels = p.argmax(-1)
        sizes = torch.bincount(labels,minlength=8)
        return dict(consistency=float(c),marginal_entropy=float(h),
                    selection_score=float(c-2*h),min_cluster=int(sizes.min()),
                    max_cluster=int(sizes.max()),occupied_clusters=int((sizes>0).sum()),
                    hard_neighbor_agreement=float((labels[:,None]==q.argmax(-1)[nn]).float().mean()))


def fit(x_np, hash_values, seed, weight, epochs, val_np=None, val_hash=None):
    torch.manual_seed(seed)
    x = torch.from_numpy(x_np)
    nn = graph(x_np,x_np,hash_values,hash_values)
    head = initialize(x_np,seed)
    optim = torch.optim.Adam(head.parameters(),lr=.03)
    if val_np is not None:
        vx=torch.from_numpy(val_np); vn=graph(val_np,x_np,val_hash,hash_values)
    best = None
    curve=[]
    for epoch in range(epochs+1):
        if epoch:
            optim.zero_grad()
            p=head(x).softmax(-1)
            c,h=components(p,p,nn)
            loss=c-weight*h
            loss.backward()
            optim.step()
        if epoch % 10 == 0 or epoch==epochs:
            tr=evaluate(head,x,x,nn)
            row=dict(epoch=epoch,seed=seed,entropy_weight=weight,
                     **{'train_'+k:v for k,v in tr.items()})
            if val_np is not None:
                va=evaluate(head,vx,x,vn)
                row.update({'validation_'+k:v for k,v in va.items()})
                eligible=tr['occupied_clusters']==8 and va['occupied_clusters']==8
                if eligible and (best is None or va['selection_score']<best['score']):
                    best=dict(score=va['selection_score'],epoch=epoch,
                              state=copy.deepcopy(head.state_dict()),metrics=va)
            curve.append(row)
    return head,best,curve


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True);REP.mkdir(parents=True,exist_ok=True)
    started=time.time()
    df=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    feature=ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz'
    z=np.load(feature,allow_pickle=False)
    assert np.array_equal(z['keys'],df.key.to_numpy(str))
    raw=z['features'].astype('float32')
    assert np.isfinite(raw).all() and np.allclose(np.linalg.norm(raw,axis=1),1,atol=1e-5)
    idx=df[df.split.eq('train')].drop_duplicates('pixel_sha256').index.to_numpy()
    hashes=df.pixel_sha256.to_numpy()
    group_path=ROOT/'output/groups/stage1_v1/groups.csv'
    groups=pd.read_csv(group_path).set_index('key').loc[df.key,'candidate_group'].to_numpy()
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261105).split(raw[idx],groups=groups[idx]))
    train,val=idx[a],idx[b]
    assert not set(groups[train])&set(groups[val])
    plan=dict(method='SCAN-inspired neighbor consistency head refinement',backbone='frozen DINOv2 ViT-S/14',
              features_sha256=sha(feature),groups_sha256=sha(group_path),
              seeds=SEEDS,entropy_weights=WEIGHTS,epochs=EPOCHS,optimizer='Adam',lr=.03,
              init='KMeans8 seed-matched centers: W=20*center,b=-10*norm(center)^2',
              objective='mean(-log(dot(p_anchor,p_neighbor))) - lambda*H(mean(p_anchor))',
              nearest_neighbors=10,duplicate_policy='unique train; exclude every matching pixel hash',
              validation='20% train candidate groups; validation graph points ONLY to fit train',
              selection='minimum validation consistency minus 2*entropy at 10-epoch checkpoints; 8 occupied clusters in both subsets',
              caveat='entropy encourages balance; true category sizes unknown. No semantic accuracy is measured.',
              no_archive_labels=True,no_semantic_targets=True,test_used_in_fit=False,
              training_augmentations=False,self_labeling_stage=False,full_scan_reproduction=False,
              reference='https://arxiv.org/html/2005.12320v2',script_sha256=sha(Path(__file__)))
    dump(REP/'plan.json',plan)
    split=df.iloc[idx][['key','split','pixel_sha256']].copy()
    split['candidate_group']=groups[idx]
    split['role']=np.where(np.isin(idx,train),'fit_train','heldout_train')
    split.to_csv(REP/'group_split.csv',index=False)
    curves=[];results=[];eligible=[]
    with threadpool_limits(limits=2):
        for weight,seed in itertools.product(WEIGHTS,SEEDS):
            _,best,curve=fit(raw[train],hashes[train],seed,weight,EPOCHS,raw[val],hashes[val])
            curves.extend(curve)
            result=dict(seed=seed,entropy_weight=weight,eligible=best is not None)
            if best is not None:
                result.update(epoch=best['epoch'],**best['metrics'])
                eligible.append((best['score'],seed,weight,best['epoch']))
            results.append(result)
            print(json.dumps(result),flush=True)
            pd.DataFrame(curves).to_csv(REP/'learning_curves.csv',index=False)
            pd.DataFrame(results).to_csv(REP/'candidate_validation.csv',index=False)
        if not eligible:
            dump(REP/'result.json',dict(status='no_noncollapsed_candidate',semantic_accuracy=None))
            return
        score,seed,weight,epoch=min(eligible)
        head,_,curve=fit(raw[idx],hashes[idx],seed,weight,epoch)
        with torch.no_grad():
            probabilities=head(torch.from_numpy(raw)).softmax(-1).numpy()
        assert probabilities.shape==(2400,8) and np.allclose(probabilities.sum(1),1,atol=1e-5)
        labels=probabilities.argmax(1)
        counts=np.bincount(labels[idx],minlength=8)
        out=df[['key','split','image_id','pixel_sha256']].copy()
        out['cluster_index']=labels;out['cluster_id']=[f'cluster_{v:02d}' for v in labels]
        out['uncalibrated_head_probability']=probabilities.max(1)
        out.to_csv(OUT/'assignments.csv',index=False)
        np.savez_compressed(OUT/'predictions.npz',keys=df.key.to_numpy(str),probabilities=probabilities)
        torch.save(dict(state_dict=head.state_dict(),input_dimension=384,config=plan),OUT/'head.pt')
        pd.DataFrame(curve).to_csv(REP/'refit_learning_curve.csv',index=False)
        km=KMeans(n_clusters=8,n_init=10,random_state=seed).fit(raw[idx])
        result=dict(status='complete',selected_seed=seed,entropy_weight=weight,epochs=epoch,
                    validation_selection_score=score,train_cluster_sizes=counts.tolist(),
                    test_cluster_sizes=np.bincount(labels[df.split.eq('test')],minlength=8).tolist(),
                    ari_vs_same_seed_kmeans=float(adjusted_rand_score(km.predict(raw[idx]),labels[idx])),
                    refit_no_collapsed_cluster=bool(counts.min()>0),semantic_accuracy=None,
                    assignment_sha256=sha(OUT/'assignments.csv'),seconds=time.time()-started)
        dump(REP/'result.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
