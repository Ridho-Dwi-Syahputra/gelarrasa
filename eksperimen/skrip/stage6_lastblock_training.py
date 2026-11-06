"""Actual final-ViT-block + norm + named-head training, no archived labels.

Uses random512 fit images and independent CNN-derived pseudo-labels. This is
a small adaptation pilot, not full SPICE or full-backbone fine-tuning.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch,timm
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from stage6_material_resolution import network
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'output/embeddings/stage6_lastblock_v1'
BASE=ROOT/'output/groups/stage6_material_resolution_v1'
OUT=ROOT/'output/groups/stage6_lastblock_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):p.write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8')

class Tail(torch.nn.Module):
    def __init__(self,block,norm,head):super().__init__();self.block=block;self.norm=norm;self.head=head
    def features(self,tokens):return F.normalize(self.norm(self.block(tokens))[:,0],dim=1)
    def forward(self,tokens):return self.head(self.features(tokens))

def main():
    global BASE
    pointer=REPORT/'selected_material_run.json'
    if pointer.exists():BASE=ROOT/json.loads(pointer.read_text())['output']
    started=time.time();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    cache_plan=json.loads((REPORT/'lastblock_cache.json').read_text());assert cache_plan['status']=='passed'
    base=torch.load(BASE/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv')
    original=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    index=dict(zip(original['keys'],range(len(original['keys']))))
    pools={view:np.load(CACHE/f'{view}.npy',mmap_mode='r') for view in ['original224','strong224','hflip224','original336']}
    keys={view:pd.read_csv(CACHE/f'{view}_keys.csv').key.tolist() for view in pools}
    original_index=dict(zip(keys['original224'],range(len(keys['original224']))))
    train_keys=cache_plan['fit_subset_keys'];val_keys=cache_plan['heldout_keys']
    tr=np.array([original_index[k] for k in train_keys]);va=np.array([original_index[k] for k in val_keys])
    assert keys['strong224']==train_keys
    checkpoints=json.loads((ROOT/'output/embeddings/stage2_repr_v1/dino_checkpoint.json').read_text())
    weight=hf_hub_download(checkpoints['repository'],'model.safetensors',revision=checkpoints['revision'],local_files_only=True)
    backbone=timm.create_model(checkpoints['model'],pretrained=True,num_classes=0,dynamic_img_size=True,pretrained_cfg_overlay={'file':weight}).eval()
    head=network(base['kind']);head.load_state_dict(base['state_dict']);head.eval()
    initial=Tail(copy.deepcopy(backbone.blocks[-1]),copy.deepcopy(backbone.norm),head)
    del backbone
    with torch.inference_mode():
        pseudo=head(torch.from_numpy(original['features'][[index[k] for k in train_keys]])).softmax(-1)
        fit_reference=head(torch.from_numpy(original['features'][[index[k] for k in base['fit_keys']]])).argmax(1).numpy()
    probabilities=np.load(BASE/'named_probabilities.npz',allow_pickle=False)
    prior=probabilities['prior_probabilities']
    reliable=(pseudo.max(1).values.numpy()>=.7)&(pseudo.argmax(1).numpy()==prior[[index[k] for k in train_keys]].argmax(1))
    plan=dict(source_sha256=sha(__file__),task='Train final ViT block, norm and named head, cached frozen first11 blocks',
        random_fit_subset=train_keys,heldout_keys=val_keys,fit_images=512,heldout_images=381,
        learning_rates=[.00001,.00003],head_lr=.0001,epochs=4,batch=16,
        objective='soft pseudo-target CE with reliability weights +.25 original/strong view JS',
        reliability='probability>=.7 and agrees independent CNN material/object evidence',reliable_subset=int(reliable.sum()),
        confidence_not_calibrated=True,base_checkpoint_sha256=sha(BASE/'selected_semantic_head.pt'),base_output=str(BASE.relative_to(ROOT)),
        reference_archive_read=False,test_used_for_fit_or_selection=False,
        selection='.30 flip +.20 scale +.25 CNN evidence agreement +.25 fixed train-neighbor agreement, all8 occupied min5')
    dump(REPORT/'lastblock_plan_before_fit.json',plan)
    xs=original['features'];fit_idx=np.array([index[k] for k in base['fit_keys']]);val_idx=np.array([index[k] for k in val_keys])
    with threadpool_limits(limits=2):near=np.argsort(-(xs[val_idx]@xs[fit_idx].T),axis=1,kind='stable')[:,:10]
    view_positions=np.array([val_keys.index(k) for k in keys['hflip224']])
    assert keys['hflip224']==keys['original336'] and len(view_positions)==87
    def infer(model,view,ids=None):
        model.eval();ids=np.arange(len(pools[view])) if ids is None else ids;blocks=[]
        with torch.inference_mode():
            for first in range(0,len(ids),16):
                tensor=torch.from_numpy(np.array(pools[view][ids[first:first+16]],dtype='float32'))
                blocks.append(model(tensor).softmax(-1).numpy())
        return np.vstack(blocks)
    def evaluate(model):
        probability=infer(model,'original224',va);v=probability.argmax(1)
        fl=infer(model,'hflip224').argmax(1);sc=infer(model,'original336').argmax(1)
        flip=float((v[view_positions]==fl).mean());scale=float((v[view_positions]==sc).mean())
        teacher=float((v==prior[val_idx].argmax(1)).mean());neighbor=float((v[:,None]==fit_reference[near]).mean())
        counts=np.bincount(v,minlength=8)
        return dict(score=.3*flip+.2*scale+.25*teacher+.25*neighbor,flip=flip,scale=scale,
            cnn_evidence_agreement=teacher,neighbor_agreement=neighbor,occupied=int((counts>0).sum()),min_class=int(counts.min()))
    baseline=evaluate(initial);history=[dict(lr=0.,epoch=0,**baseline)]
    best=dict(lr=0.,epoch=0,**baseline,state=copy.deepcopy(initial.state_dict()))
    weight_sample=torch.from_numpy(np.where(reliable,1.,.15).astype('float32'))
    # Detach ordinary tensors; indexing in the backward pass must not receive
    # inference-mode tensors saved by autograd.
    target=torch.tensor(pseudo.numpy())
    changed=[]
    for lr in [.00001,.00003]:
        model=copy.deepcopy(initial)
        optimizer=torch.optim.AdamW([{'params':list(model.block.parameters())+list(model.norm.parameters()),'lr':lr},
                                    {'params':model.head.parameters(),'lr':.0001}],weight_decay=.01)
        generator=np.random.default_rng(20261106)
        for epoch in range(1,5):
            order=generator.permutation(512);loss_sum=0.
            for first in range(0,512,16):
                ids=order[first:first+16];model.train()
                weak=torch.from_numpy(np.array(pools['original224'][tr[ids]],dtype='float32'))
                strong=torch.from_numpy(np.array(pools['strong224'][ids],dtype='float32'))
                logits=model(weak);q=logits.softmax(-1);s=model(strong).softmax(-1);mean=(q+s)/2
                ce=-(target[ids]*logits.log_softmax(-1)).sum(1)
                js=.5*((q*(q.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1)+
                       (s*(s.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1)).mean()
                loss=(ce*weight_sample[ids]).mean()+.25*js
                assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward()
                if first==0 and epoch==1:
                    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max()>0 for parameter in model.block.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*len(ids)
            metric=evaluate(model);history.append(dict(lr=lr,epoch=epoch,loss=loss_sum/512,**metric))
            print(f'lastblock lr{lr} epoch{epoch} score{metric["score"]:.6f}',flush=True)
            if metric['occupied']==8 and metric['min_class']>=5 and metric['score']>best['score']+1e-7:
                best=dict(lr=lr,epoch=epoch,**metric,state=copy.deepcopy(model.state_dict()))
        delta=max(float((value-initial.block.state_dict()[name]).abs().max()) for name,value in model.block.state_dict().items())
        assert delta>0
        changed.append(dict(lr=lr,backbone_parameter_max_abs_change=delta))
    pd.DataFrame(history).to_csv(REPORT/'lastblock_development_metrics.csv',index=False)
    initial.load_state_dict(best.pop('state'));initial.eval()
    torch.save(dict(state_dict=initial.state_dict(),kind=base['kind'],selected=best,plan=plan,class_order=CLASSES),OUT/'selected_tail.pt')
    result=dict(status='passed',actual_backbone_blocks_trained=1,conditions_trained=2,total_training_epochs=8,
        selected=best,control=baseline,selected_adaptation=best['epoch']>0,
        fit_images=512,heldout_images=381,reliable_pseudo_targets=int(reliable.sum()),
        reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started,backbone_update_proofs=changed,
        checkpoint_sha256=sha(OUT/'selected_tail.pt'),base_assignments_unchanged=True,
        note='Development pseudo-evidence and small augmentation probes; not official Accuracy or full-backbone SPICE.')
    dump(REPORT/'lastblock_training.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
