"""Reliable class-specific CNN seeds -> train-only graph/view student.

Seeds are named by public non-VLM CNN evidence, not assistant classifications.
Uncertain rows do not become hard targets. Per-class seed caps balance training
evidence and never impose a class quota on the resulting predictions.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from stage6_material_resolution import network
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/groups/stage6_material_resolution_calibrated_v2'
OUT=ROOT/'output/groups/stage6_semantic_seed_v3'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'
REPORT=COMMON/'clean_seeds_v3'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):p.write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8')

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    features=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    x=features['features'];index=dict(zip(features['keys'],range(len(x))))
    source=np.load(SOURCE/'named_probabilities.npz',allow_pickle=False)
    assert np.array_equal(features['keys'],source['keys']) and np.array_equal(source['keys'],p.key.to_numpy(str))
    prior=source['prior_probabilities'];hard=prior.argmax(1);ordered=np.sort(prior,axis=1)
    source_checkpoint=torch.load(SOURCE/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
    fit_keys=source_checkpoint['fit_keys'];val_keys=source_checkpoint['heldout_keys']
    fi=np.array([index[k] for k in fit_keys]);vi=np.array([index[k] for k in val_keys])
    split=pd.read_csv(ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv').set_index('key')
    assert not set(split.loc[fit_keys,'candidate_group'])&set(split.loc[val_keys,'candidate_group'])
    with threadpool_limits(limits=4):
        similarity=x[fi]@x[fi].T;np.fill_diagonal(similarity,-np.inf)
        nn=np.argsort(-similarity,axis=1,kind='stable')[:,:10]
        nv=np.argsort(-(x[vi]@x[fi].T),axis=1,kind='stable')[:,:10]
    flip=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    lookup=dict(zip(flip['keys'],range(len(flip['keys']))))
    tf=np.array([i for i,k in enumerate(fit_keys) if k in lookup]);tv=np.array([i for i,k in enumerate(val_keys) if k in lookup])
    xf=torch.from_numpy(flip['features'][[lookup[fit_keys[i]] for i in tf]])
    vf=torch.from_numpy(flip['features'][[lookup[val_keys[i]] for i in tv]])
    xt=torch.from_numpy(x[fi]);xv=torch.from_numpy(x[vi])
    reliable=(prior.max(1)>=.5)&((ordered[:,-1]-ordered[:,-2])>=.1)
    # Development cores are filtered by the same fixed, model-only confidence
    # rule, never by whether our student agrees or whether an archive agrees.
    core=reliable[vi]
    counts=np.bincount(hard[vi][core],minlength=8)
    assert (counts>0).all(),dict(zip(CLASSES,counts.tolist()))
    plan=dict(task='CNN-reliable named seed training with uncertain data retained for consistency',class_order=CLASSES,
        fit_keys=fit_keys,heldout_keys=val_keys,seed_caps=[24,48,96],kinds=['linear','mlp'],epochs=180,lr=.003,
        seed_rules='CNN prior max>=.5, top2 margin>=.1; highest class confidence up to cap, fit-only; no output class quotas.',
        unlabelled_loss='0.05 train-neighbor log consistency +0.2 paired-view JS',
        selection='0.50 macro CNN-core agreement +0.25 flip +0.25 neighbor; all8 heldout classes min5; not semantic Accuracy',
        core_counts=dict(zip(CLASSES,counts.tolist())),source_sha256=sha(__file__),
        ontology_sha256=sha(ROOT/'eksperimen/skrip/stage6_semantic_ontology.py'),reference_archive_read=False,
        prior_source_sha256=sha(SOURCE/'named_probabilities.npz'),test_in_fit_or_selection=False)
    dump(REPORT/'plan_before_fit.json',plan)
    history=[];best=None
    def evaluate(head):
        head.eval()
        with torch.inference_mode():
            train=head(xt).argmax(1).numpy();v=head(xv).argmax(1).numpy();flip_prediction=head(vf).argmax(1).numpy()
        macro=float(np.mean([(v[core & (hard[vi]==c)]==c).mean() for c in range(8)]))
        flip_score=float((v[tv]==flip_prediction).mean());neighbor=float((v[:,None]==train[nv]).mean())
        n=np.bincount(v,minlength=8)
        return dict(score=.5*macro+.25*flip_score+.25*neighbor,macro_CNN_core_agreement=macro,
            flip=flip_score,neighbor=neighbor,occupied=int((n>0).sum()),min_class=int(n.min()))
    # Both previous versions are controls under the SAME new criterion, rather
    # than comparing scores calculated with different pseudo-teacher definitions.
    for name,path,report in [('raw_v1',ROOT/'output/groups/stage6_material_resolution_v1',COMMON),('calibrated_v2',SOURCE,COMMON/'calibrated_v2')]:
        ck=torch.load(path/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
        head=network(ck['kind']);head.load_state_dict(ck['state_dict']);m=evaluate(head)
        history.append(dict(candidate=name,epoch=ck['selected']['epoch'],**m))
        if m['occupied']==8 and m['min_class']>=5 and (best is None or m['score']>best['score']):
            best=dict(candidate=name,kind=ck['kind'],epoch=ck['selected']['epoch'],**m,state=copy.deepcopy(head.state_dict()))
    for cap in [24,48,96]:
        seed_positions=[]
        for c in range(8):
            available=np.flatnonzero(reliable[fi]&(hard[fi]==c))
            ranked=available[np.argsort(-prior[fi[available],c],kind='stable')]
            selected=ranked[:cap]
            assert len(selected)>=3,(CLASSES[c],len(selected))
            seed_positions.extend(selected.tolist())
        seeds=np.array(sorted(seed_positions));y=torch.from_numpy(hard[fi[seeds]])
        seed_table=p.iloc[fi[seeds]][['key','split','image_id']].copy();seed_table['model_seed_class']=np.array(CLASSES)[y.numpy()]
        seed_table.to_csv(OUT/f'seed_cap{cap}.csv',index=False)
        # Inverse seed count changes only the loss contribution; predictions are
        # never forced to have matching class counts.
        class_weight=torch.from_numpy((len(seeds)/(8*np.bincount(y.numpy(),minlength=8))).astype('float32'))
        for kind in ['linear','mlp']:
            torch.manual_seed(20261106);head=network(kind)
            optimizer=torch.optim.Adam(head.parameters(),lr=.003,weight_decay=.0001)
            for epoch in range(1,181):
                head.train();logits=head(xt);q=logits.softmax(-1)
                ce=F.cross_entropy(logits[seeds],y,weight=class_weight)
                consistency=-(q[:,None,:]*q[nn]).sum(-1).clamp_min(1e-8).log().mean()
                alt=head(xf).softmax(-1);orig=q[tf];mean=(orig+alt)/2
                js=.5*((orig*(orig.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+
                       (alt*(alt.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                loss=ce+.05*consistency+.2*js;assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step()
                if epoch%10==0:
                    metric=evaluate(head);name=f'core_cap{cap}_{kind}';history.append(dict(candidate=name,epoch=epoch,**metric))
                    if metric['occupied']==8 and metric['min_class']>=5 and (best is None or metric['score']>best['score']+1e-7):
                        best=dict(candidate=name,kind=kind,epoch=epoch,**metric,state=copy.deepcopy(head.state_dict()))
            print(f'clean semantic seeds cap{cap} {kind}, {len(seeds)} source seeds',flush=True)
    assert best is not None
    head=network(best['kind']);state=best.pop('state');head.load_state_dict(state);head.eval()
    torch.save(dict(state_dict=state,kind=best['kind'],selected=best,fit_keys=fit_keys,heldout_keys=val_keys,class_order=CLASSES,plan=plan),OUT/'selected_semantic_head.pt')
    with torch.inference_mode():probability=head(torch.from_numpy(x)).softmax(-1).numpy()
    pred=probability.argmax(1);order=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy()
    result['predicted_class']=np.array(CLASSES)[pred];result['uncalibrated_probability']=probability.max(1)
    result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=prior.argmax(1)
    result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='Reliable_nonVLM_CNN_semantic_seeds_graph_view_student_v3'
    result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=probability,prior_probabilities=prior)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    summary=dict(status='passed',heads_trained=6,selected=best,source_ontology='public non-VLM model classes',
        mixed_output_names=0,train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),
        train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),
        fit_images=len(fi),heldout_images=len(vi),reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started)
    dump(REPORT/'summary.json',summary)
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt'),summary_sha256=sha(REPORT/'summary.json'),source_sha256=sha(__file__),
        ontology_sha256=sha(ROOT/'eksperimen/skrip/stage6_semantic_ontology.py'),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),
        selection='same CNN-core/view/neighbor criterion across controls and six new seed heads, before any reference archive read',
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt')))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
