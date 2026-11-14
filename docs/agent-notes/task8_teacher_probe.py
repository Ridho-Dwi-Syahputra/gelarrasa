"""Pinned image-only eight-class teacher, train-only automatic refinement.

No external images, hosted inference, hand labels or visual archive are inputs.
This is a label-evidence probe, not a final fine-tuned competition classifier.
"""
import time
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms.functional import to_tensor,normalize
from safetensors.torch import load_file
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


class ConcatPool(nn.Module):
    def forward(self,x):
        return torch.cat([nn.functional.adaptive_max_pool2d(x,1),nn.functional.adaptive_avg_pool2d(x,1)],dim=1)


def construct():
    body=nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    head=nn.Sequential(ConcatPool(),nn.Flatten(),nn.BatchNorm1d(4096),nn.Dropout(.25),
                       nn.Linear(4096,512,bias=False),nn.ReLU(inplace=True),nn.BatchNorm1d(512),nn.Dropout(.5),nn.Linear(512,8,bias=False))
    model=nn.Sequential(body,head)
    state=load_file(str(NOTES/'.runtime/task8_weights.safetensors'))
    assert all(torch.isfinite(t).all() for t in state.values())
    model.load_state_dict(state,strict=True)
    return model.eval()


def main():
    started=time.time();torch.set_num_threads(4)
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    origin=json.loads((NOTES/'json/task8_tensor_extraction.json').read_text())
    mapping={'cardboard':'cardboard','e-waste':'electronic','glass':'glass','metal':'metal','organic':'foodwaste','paper':'paper','plastic':'plastic','textile':'textile'}
    order=[CLASSES.index(mapping[name]) for name in origin['source_class_order']];assert set(order)==set(range(8))
    plan={'method':'additional image-only eight-class teacher with conservative horizontal-view gate',
          'teacher':{key:origin[key] for key in ['repository','revision','original_checkpoint_sha256','tensor_file_sha256','source_class_order']},
          'source_category_mapping':mapping,'transform':'RGB center-crop resize224 bilinear, ImageNet mean/std from exported transform metadata',
          'original_probability_minimum':.95,'original_margin_minimum':.5,'hflip_probability_minimum':.9,'hflip_class_agreement_required':True,
          'manual_labels_consumed':0,'test_rows_processed':0,'external_image_data_downloaded':False,'inference_api_used':False,
          'source_pretraining_image_overlap_unknown':True,'fine_tuned_on_competition_here':False,'final_classifier_ready':False,
          'reference_archive_read':False,'researcher_has_seen_previous_private_audits':True,'protected_sha256':protected,
          'qualification':'automatic train label evidence only; participant training/adaptation still required before a final classifier submission'}
    dump(NOTES/'json/task8_teacher_plan.json',plan)
    frame=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});frame=frame[frame.split.eq('train')&~frame.is_crop][['parent_key','parent_id','source_path','source_sha256','assigned_class']].rename(columns={'parent_key':'key','parent_id':'image_id'}).sort_values('image_id').reset_index(drop=True)
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    model=construct();assert model(torch.zeros(1,3,224,224)).shape==(1,8)
    def transform(path,flip=False):
        with Image.open(path) as source:im=ImageOps.exif_transpose(source).convert('RGB')
        im=ImageOps.fit(im,(224,224),method=Image.Resampling.BILINEAR,centering=(.5,.5))
        if flip:im=ImageOps.mirror(im)
        return normalize(to_tensor(im),[.485,.456,.406],[.229,.224,.225])
    cache=NOTES/'.runtime/task8_teacher_features.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False);assert np.array_equal(z['keys'],frame.key.to_numpy(str));probability=z['probabilities'];features=z['features']
    else:
        probabilities=[];features=[]
        with torch.inference_mode():
            for start in range(0,len(frame),8):
                rows=frame.iloc[start:start+8];images=[]
                for row in rows.itertuples():assert sha(ROOT/row.source_path)==row.source_sha256;images.append(transform(ROOT/row.source_path))
                raw=model[1][0](model[0](torch.stack(images))).flatten(1)
                logits=model[1][2:](raw);source_prob=logits.softmax(1).numpy();task_prob=np.zeros_like(source_prob);task_prob[:,order]=source_prob
                probabilities.append(task_prob);features.append(nn.functional.normalize(raw,dim=1).numpy().astype('float32'))
                if start==0 or (start//8+1)%24==0:print(f'Eight-class teacher {min(start+8,len(frame))}/{len(frame)} | {time.time()-started:.0f}s',flush=True)
        probability=np.vstack(probabilities);features=np.vstack(features)
        np.savez_compressed(cache,keys=frame.key.to_numpy(str),probabilities=probability,features=features,weight_sha256=np.array(origin['tensor_file_sha256']))
    pred=probability.argmax(1);sorted_probability=np.sort(probability,axis=1)
    baseline=np.array([CLASSES.index(c) for c in frame.assigned_class])
    candidates=np.flatnonzero((pred!=baseline)&(probability.max(1)>=.95)&((sorted_probability[:,-1]-sorted_probability[:,-2])>=.5))
    print(f'Eight-class teacher: {len(candidates)} high-confidence disagreements to verify with hflip',flush=True)
    final=baseline.copy();decisions=[]
    with torch.inference_mode():
        for start in range(0,len(candidates),8):
            ids=candidates[start:start+8];batch=torch.stack([transform(ROOT/frame.iloc[i].source_path,True) for i in ids]);source_prob=model(batch).softmax(1).numpy();task_prob=np.zeros_like(source_prob);task_prob[:,order]=source_prob
            for i,q in zip(ids,task_prob):
                agrees=int(q.argmax())==int(pred[i]) and float(q.max())>=.9
                if agrees:final[i]=pred[i]
                decisions.append({'key':frame.iloc[i].key,'image_id':frame.iloc[i].image_id,'baseline_class':CLASSES[baseline[i]],'teacher_class':CLASSES[pred[i]],'original_probability':float(probability[i].max()),'hflip_probability':float(q.max()),'accepted':agrees})
    result=frame[['key','image_id','source_sha256']].copy();result['predicted_class']=np.array(CLASSES)[final];result['method']='task8_high_confidence_hflip_gate_v1';result['ground_truth_verified']=False
    output=NOTES/'table-csv/task8_teacher_selected.csv';result.to_csv(output,index=False)
    pd.DataFrame(decisions).to_csv(NOTES/'table-csv/task8_teacher_decisions.csv',index=False)
    teacher_only=frame[['key','image_id']].copy();teacher_only['predicted_class']=np.array(CLASSES)[pred];teacher_only['confidence']=probability.max(1);teacher_only.to_csv(NOTES/'table-csv/task8_teacher_raw_predictions.csv',index=False)
    report={'status':'frozen_before_reference_audit','method':'task8_high_confidence_hflip_gate_v1','manual_labels_consumed':0,'test_rows_processed':0,'train_rows':1906,
            'candidate_disagreements':len(candidates),'accepted_changes':int((final!=baseline).sum()),'prediction_sha256':sha(output),
            'teacher_weight_sha256':origin['tensor_file_sha256'],'class_counts':dict(zip(CLASSES,np.bincount(final,minlength=8).tolist())),
            'semantic_accuracy':None,'active_data_modified':False,'protected_sha256':protected,'source_code_sha256':sha(__file__),'seconds':time.time()-started}
    dump(NOTES/'json/task8_teacher_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
