"""Image previews, pixel retrieval and conservative, unlabeled crop proposals.

Display interpretations are explicitly isolated from every numerical decision.
No archive label, model API or LLM service is read/called by this module.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import html
import json
import os
import textwrap

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.metrics import structural_similarity
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/eda/stage3_visual_20261106'
PROCESSED = ROOT / 'data/processed/stage3_crop_candidates_v1'
CLASSES = ['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
cv2.setNumThreads(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')


def partition_digest(assignments):
    text = '\n'.join(f'{r.key},{r.cluster_index}' for r in assignments.sort_values('key').itertuples())
    return hashlib.sha256(text.encode()).hexdigest()


def load_context():
    import torch
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'figures').mkdir(exist_ok=True)
    (REPORT/'galleries').mkdir(exist_ok=True)
    p = pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv', dtype={'image_id':str})
    a = pd.read_csv(ROOT/'output/groups/stage2_neighbor_v1/assignments.csv', dtype={'image_id':str})
    g = pd.read_csv(ROOT/'output/groups/stage1_v1/groups.csv')
    assert len(p)==len(a)==2400 and p.key.is_unique and set(p.key)==set(a.key)
    df = p.merge(a[['key','cluster_index','uncalibrated_head_probability']], on='key', validate='one_to_one')
    df = df.merge(g[['key','exact_group','candidate_group']], on='key', validate='one_to_one')
    mapping = json.loads((ROOT/'configs/display_label_mapping.json').read_text(encoding='utf-8'))
    assert mapping['purpose']=='private_eda_display_only' and not mapping['training_target']
    assert mapping['partition_sha256']==partition_digest(a), 'Display interpretations require review after a partition change.'
    names = {int(k):v['display_name'] for k,v in mapping['groups'].items()}
    df['display_label'] = df.cluster_index.map(names)
    df['mixed_display_group'] = df.cluster_index.map({int(k):v['mixed'] for k,v in mapping['groups'].items()})
    path = ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz'
    z = np.load(path, allow_pickle=False)
    assert np.array_equal(z['keys'], p.key.to_numpy(str))
    features = z['features']
    checkpoint = torch.load(ROOT/'output/groups/stage2_neighbor_v1/head.pt', map_location='cpu', weights_only=True)
    assert checkpoint['config']['features_sha256']==sha(path)
    head = torch.nn.Linear(384,8); head.load_state_dict(checkpoint['state_dict']);head.eval()
    with torch.inference_mode():probabilities=head(torch.from_numpy(features)).softmax(-1).numpy()
    assert np.array_equal(probabilities.argmax(1), df.cluster_index)
    order=np.sort(probabilities,axis=1)
    df['model_margin']=order[:,-1]-order[:,-2]
    df['model_entropy']=-(probabilities*np.log(np.maximum(probabilities,1e-12))).sum(1)
    tr=np.where(df.split.eq('train'))[0]
    with threadpool_limits(limits=4):similarity=features@features[tr].T
    similarity[df.pixel_sha256.to_numpy()[:,None]==df.iloc[tr].pixel_sha256.to_numpy()[None,:]]=-np.inf
    df['nearest_other_train_cosine']=similarity.max(1)
    training=df[df.split.eq('train')]
    thresholds={c:dict(p01=float(training[c].quantile(.01)),p05=float(training[c].quantile(.05)),
                        p95=float(training[c].quantile(.95)),p99=float(training[c].quantile(.99)))
                for c in ['laplacian_variance_256','brightness','contrast','grayscale_entropy_bits','edge_fraction',
                          'nearest_other_train_cosine','model_margin']}
    df['flag_low_sharpness']=df.laplacian_variance_256<thresholds['laplacian_variance_256']['p05']
    df['flag_exposure']=(df.brightness<thresholds['brightness']['p01'])|(df.brightness>thresholds['brightness']['p99'])
    df['flag_low_contrast']=df.contrast<thresholds['contrast']['p05']
    df['flag_busy_texture']=df.edge_fraction>thresholds['edge_fraction']['p95']
    df['flag_novel_appearance']=df.nearest_other_train_cosine<thresholds['nearest_other_train_cosine']['p05']
    df['flag_model_ambiguous']=(df.model_margin<thresholds['model_margin']['p05'])|df.mixed_display_group
    train_sizes=set(zip(training.width,training.height))
    df['size_seen_in_train']=[(int(w),int(h)) in train_sizes for w,h in zip(df.width,df.height)]
    frozen=[ROOT/'configs/data_contract.json',ROOT/'output/groups/stage2_neighbor_v1/assignments.csv',
            ROOT/'output/groups/stage2_neighbor_v1/head.pt',path]
    context=dict(root=ROOT,report=REPORT,df=df,features=features,head=head,thresholds=thresholds,
                 names=names,mapping=mapping,frozen={str(p.relative_to(ROOT)):sha(p) for p in frozen})
    dump(REPORT/'quality_thresholds.json',dict(fit_population='train only',thresholds=thresholds,
          note='Technical screening flags; not proof of blur, wrong labels, or multiple semantic objects.'))
    return context


def distribution_tables(ctx):
    df=ctx['df']
    table=pd.crosstab(df.display_label,df.split).reindex(columns=['train','test'],fill_value=0)
    table.index.name='nama_tampilan_sementara'
    table.to_csv(REPORT/'display_group_distribution.csv')
    fig,ax=plt.subplots(figsize=(11,5.5));table.plot.barh(ax=ax,color=['#3182bd','#f59e0b'])
    ax.set_title('Jumlah gambar per nama kelompok sementara');ax.set_xlabel('Jumlah gambar');ax.set_ylabel('')
    ax.text(0,-.2,'Interpretasi visual untuk EDA pribadi; kelompok campuran tidak dipaksakan menjadi satu kelas.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(REPORT/'figures/distribution.jpg',dpi=130);plt.close(fig)
    taxonomy=pd.DataFrame({'kelas_resmi':CLASSES,'jumlah_train_terverifikasi':[pd.NA]*8,
                           'jumlah_test_terverifikasi':[pd.NA]*8})
    taxonomy.to_csv(REPORT/'official_class_counts_unavailable.csv',index=False)
    sizes=df.groupby(['split','width','height']).size().rename('jumlah').reset_index()
    sizes.to_csv(REPORT/'dimensions_by_split.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for split,color in [('train','#3182bd'),('test','#f59e0b')]:
        t=df[df.split.eq(split)]
        axes[0].scatter(t.width,t.height,s=9,alpha=.35,label=split,color=color)
        axes[1].hist(t.aspect_ratio,bins=np.linspace(.1,3,35),density=True,histtype='step',label=split,color=color)
        axes[2].hist(np.log10(t.laplacian_variance_256+1),bins=30,density=True,histtype='step',label=split,color=color)
    for ax,title in zip(axes,['Lebar × tinggi native','Aspect ratio','log10(Laplacian variance + 1)']):ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(REPORT/'figures/dimensions_quality.jpg',dpi=130);plt.close(fig)
    return table,taxonomy,sizes


def propose_regions(rgb):
    """Classical, per-image proposals; component/marker counts are not object truth."""
    height,width=rgb.shape[:2];scale=min(1.,384/max(height,width))
    work=cv2.resize(rgb,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA)
    lab=cv2.cvtColor(cv2.GaussianBlur(work,(5,5),0),cv2.COLOR_RGB2LAB).astype('float32')
    h,w=lab.shape[:2];b=max(2,round(min(h,w)*.04))
    border=np.vstack([lab[:b].reshape(-1,3),lab[-b:].reshape(-1,3),lab[:, :b].reshape(-1,3),lab[:,-b:].reshape(-1,3)])
    center=np.median(border,axis=0)
    border_distance=np.linalg.norm(border-center,axis=1)
    border_spread=float(np.quantile(border_distance,.8))
    distance=np.linalg.norm(lab-center,axis=2)
    threshold=float(np.clip(np.quantile(border_distance,.8)+12,20,70))
    mask=(distance>threshold).astype('uint8')
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=ndi.binary_fill_holes(mask).astype('uint8')
    coverage=float(mask.mean())
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    minimum=max(60,int(h*w*.035))
    components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=minimum]
    candidates=[(labels==i) for i in components]
    method='connected_components'
    if len(candidates)<2 and .08<coverage<.90:
        edt=ndi.distance_transform_edt(mask)
        peaks=peak_local_max(edt,min_distance=max(12,round(min(h,w)*.13)),threshold_rel=.38,exclude_border=False,num_peaks=4)
        if len(peaks)>=2:
            markers=np.zeros_like(mask,dtype='int32')
            for n,(y,x) in enumerate(peaks,1):markers[y,x]=n
            regions=watershed(-edt,markers,mask=mask.astype(bool))
            candidates=[regions==i for i in range(1,len(peaks)+1) if np.count_nonzero(regions==i)>=minimum]
            method='watershed_markers'
    boxes=[]
    if len(candidates)>=2 and .06<coverage<.93:
        for region in sorted(candidates,key=np.count_nonzero,reverse=True)[:4]:
            yy,xx=np.where(region);pad=max(3,round(max(np.ptp(yy),np.ptp(xx))*.05))
            x0=max(0,int(np.floor((xx.min()-pad)/scale)));x1=min(width,int(np.ceil((xx.max()+pad+1)/scale)))
            y0=max(0,int(np.floor((yy.min()-pad)/scale)));y1=min(height,int(np.ceil((yy.max()+pad+1)/scale)))
            if min(x1-x0,y1-y0)>=48 and (x1-x0)*(y1-y0)<.88*height*width:
                boxes.append((x0,y0,x1,y1))
    if len(boxes)<2:boxes=[]
    reliable_background=border_spread<25
    train_candidate=bool(len(boxes)>=2 and method=='connected_components' and reliable_background)
    return dict(boxes=boxes,proposal_method=method,foreground_fraction=coverage,
                border_spread=border_spread,possible_multiple_regions=bool(boxes),
                conservative_component_candidate=train_candidate)


def crop_audit(ctx):
    df=ctx['df'];PROCESSED.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(ROOT/'output/folds/stage1_v1/DRAFT_group_folds.csv')
    assert folds.groupby('candidate_group').fold.nunique().max()==1
    fold_map=folds.groupby('candidate_group').fold.first().to_dict()
    records=[];crops=[]
    def work(row):
        with Image.open(ROOT/row.processed_path) as image:rgb=np.asarray(image.convert('RGB'))
        result=propose_regions(rgb);saved=[]
        for n,box in enumerate(result['boxes']):
            # Test crops are previews, never additional training rows.
            folder=PROCESSED/('train_candidates' if row.split=='train' else 'test_previews')/row.image_id
            folder.mkdir(parents=True,exist_ok=True);path=folder/f'region_{chr(65+n)}.png'
            crop=Image.fromarray(rgb).crop(box);crop.save(path)
            with Image.open(path) as restored:assert np.array_equal(np.asarray(restored),rgb[box[1]:box[3],box[0]:box[2]])
            saved.append(dict(crop_key=f'{row.key}:region_{chr(65+n)}',parent_key=row.key,split=row.split,
                parent_group=row.candidate_group,parent_pixel_sha256=row.pixel_sha256,
                parent_fold=fold_map.get(row.candidate_group) if row.split=='train' else None,
                x0=box[0],y0=box[1],x1=box[2],y1=box[3],crop_width=crop.width,crop_height=crop.height,
                crop_path=path.relative_to(ROOT).as_posix(),crop_sha256=sha(path),
                proposal_method=result['proposal_method'],
                augmentation_candidate=bool(row.split=='train' and result['conservative_component_candidate']),
                promoted_to_training=False,label_status='unlabeled_algorithmic_candidate'))
        summary=dict(key=row.key,split=row.split,proposal_count=len(saved),**{k:v for k,v in result.items() if k!='boxes'})
        return summary,saved
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(summary,saved) in enumerate(pool.map(work,list(df.itertuples())),1):
            records.append(summary);crops.extend(saved)
            if n%300==0:print(f'Analisis region/crop: {n}/{len(df)}',flush=True)
    region=pd.DataFrame(records)
    columns=['crop_key','parent_key','split','parent_group','parent_pixel_sha256','parent_fold',
             'x0','y0','x1','y1','crop_width','crop_height','crop_path','crop_sha256','proposal_method',
             'augmentation_candidate','promoted_to_training','label_status']
    manifest=pd.DataFrame(crops,columns=columns)
    region.to_csv(REPORT/'region_audit.csv',index=False)
    manifest.to_csv(PROCESSED/'crop_manifest.csv',index=False)
    assert not manifest.promoted_to_training.any()
    assert manifest[manifest.split.eq('test')].augmentation_candidate.eq(False).all()
    assert manifest.empty or manifest.groupby('parent_group').parent_fold.nunique().max()<=1
    ctx['df']=df.merge(region.drop(columns='split'),on='key',validate='one_to_one')
    ctx['crops']=manifest
    dump(PROCESSED/'provenance.json',dict(method='Per-image Lab border contrast + connected components; marker watershed for touching-region hypotheses',
         source_images=len(df),crop_count=len(manifest),labels_assigned=False,test_used_for_training=False,
         parent_fold_inherited=True,raw_images_changed=False,watershed_regions_not_verified_objects=True,
         augmentation_promoted=False,reason='Region proposals require validation; inheriting a parent material label can be wrong.'))
    return region,manifest


def crop_model_probe(ctx):
    """Diagnostic only: cropped views can change the anonymous model prediction."""
    import torch
    import timm
    from huggingface_hub import hf_hub_download
    from torchvision.transforms.functional import to_tensor,normalize
    torch.set_num_threads(4)
    metadata=json.loads((ROOT/'output/embeddings/stage2_repr_v1/dino_checkpoint.json').read_text())
    weight=hf_hub_download(repo_id=metadata['repository'],filename='model.safetensors',revision=metadata['revision'])
    assert sha(weight)==metadata['checkpoint_sha256']
    model=timm.create_model(metadata['model'],pretrained=True,num_classes=0,dynamic_img_size=True,
                            pretrained_cfg_overlay={'file':weight}).eval()
    crops=ctx['crops'];selection=[]
    for split,n in [('train',96),('test',48)]:
        subset=crops[crops.split.eq(split)].sort_values('crop_key')
        if len(subset):selection.append(subset.sample(min(n,len(subset)),random_state=20261106))
    if not selection:
        result=pd.DataFrame();ctx['crop_probe']=result;return result
    selected=pd.concat(selection).sort_values('crop_key').reset_index(drop=True)
    features={}
    for view in ['direct','hflip']:
        blocks=[]
        with torch.inference_mode():
            for start in range(0,len(selected),12):
                inputs=[]
                for r in selected.iloc[start:start+12].itertuples():
                    with Image.open(ROOT/r.crop_path) as source:im=source.convert('RGB')
                    im=ImageOps.contain(im,(224,224),Image.Resampling.LANCZOS)
                    canvas=Image.new('RGB',(224,224),(124,116,104));canvas.paste(im,((224-im.width)//2,(224-im.height)//2))
                    if view=='hflip':canvas=ImageOps.mirror(canvas)
                    inputs.append(normalize(to_tensor(canvas),[.485,.456,.406],[.229,.224,.225]))
                block=model(torch.stack(inputs)).numpy();block/=np.maximum(np.linalg.norm(block,axis=1,keepdims=True),1e-12)
                blocks.append(block)
        features[view]=np.vstack(blocks).astype('float32')
        print(f'Probe crop {view}: {len(selected)} tampilan',flush=True)
    with torch.inference_mode():
        direct=ctx['head'](torch.from_numpy(features['direct'])).softmax(-1).numpy()
        mirrored=ctx['head'](torch.from_numpy(features['hflip'])).softmax(-1).numpy()
    parent=ctx['df'].set_index('key')
    selected['parent_display_label']=selected.parent_key.map(parent.display_label)
    selected['crop_display_label']=[ctx['names'][int(k)] for k in direct.argmax(1)]
    selected['crop_probability_uncalibrated']=direct.max(1)
    selected['crop_changed_model_group']=direct.argmax(1)!=selected.parent_key.map(parent.cluster_index).to_numpy()
    selected['crop_flip_stable']=direct.argmax(1)==mirrored.argmax(1)
    selected['interpretation']='diagnostic views only; no training labels generated'
    selected.to_csv(REPORT/'crop_model_view_probe.csv',index=False)
    np.savez_compressed(PROCESSED/'crop_probe_features.npz',crop_keys=selected.crop_key.to_numpy(str),**features)
    ctx['crop_probe']=selected
    return selected


def pixel_neighbors(ctx):
    df=ctx['df'];train=df[df.split.eq('train')].reset_index(drop=True);test=df[df.split.eq('test')].reset_index(drop=True)
    def rgb64(path):
        with Image.open(ROOT/path) as im:return np.asarray(im.convert('RGB').resize((64,64),Image.Resampling.BILINEAR),dtype='float32')/255
    with ThreadPoolExecutor(max_workers=4) as pool:small=np.stack(list(pool.map(rgb64,df.processed_path)))
    ti=np.where(df.split.eq('train'))[0];vi=np.where(df.split.eq('test'))[0]
    a=small[vi].reshape(len(vi),-1);b=small[ti].reshape(len(ti),-1)
    flipped=small[vi,:,::-1,:].reshape(len(vi),-1)
    with threadpool_limits(limits=4):
        norm_b=np.mean(b*b,axis=1)
        mse=np.maximum(0,np.mean(a*a,axis=1)[:,None]+norm_b[None,:]-2*(a@b.T)/a.shape[1])
        mse_flip=np.maximum(0,np.mean(flipped*flipped,axis=1)[:,None]+norm_b[None,:]-2*(flipped@b.T)/a.shape[1])
        embeddings=ctx['features'][vi]@ctx['features'][ti].T
    hash_candidates=pd.read_csv(ROOT/'reports/eda/stage1_20261105/test_train_phash_top10.csv',dtype={'test_id':str,'train_id':str})
    hash_candidates['test_id']=hash_candidates.test_id.str.zfill(3);hash_candidates['train_id']=hash_candidates.train_id.str.zfill(4)
    train_index=dict(zip(train.image_id,range(len(train))))
    @lru_cache(maxsize=2400)
    def gray128(path):
        with Image.open(ROOT/path) as im:return np.asarray(im.convert('L').resize((128,128),Image.Resampling.BILINEAR))
    rows=[]
    for i,t in enumerate(test.itertuples()):
        candidates=set(np.argsort(mse[i])[:4])|set(np.argsort(mse_flip[i])[:3])|set(np.argsort(-embeddings[i])[:2])
        candidates|={train_index[s] for s in hash_candidates[(hash_candidates.test_id==t.image_id)&(hash_candidates['rank']<=2)].train_id if s in train_index}
        gt=gray128(t.processed_path)
        for j in candidates:
            r=train.iloc[j];g=gray128(r.processed_path)
            direct=float(structural_similarity(gt,g,data_range=255))
            mirror=float(structural_similarity(gt,g[:,::-1],data_range=255))
            equal_size=(t.width==r.width and t.height==r.height)
            rows.append(dict(test_key=t.key,train_key=r.key,test_id=t.image_id,train_id=r.image_id,
                test_display_label=t.display_label,train_display_label=r.display_label,
                rgb_mse64=float(mse[i,j]),rgb_mse64_hflip=float(mse_flip[i,j]),
                ssim128_direct=direct,ssim128_hflip=mirror,ssim128_best=max(direct,mirror),
                best_alignment='direct' if direct>=mirror else 'hflip',
                same_native_dimensions=bool(equal_size),test_width=int(t.width),test_height=int(t.height),
                train_width=int(r.width),train_height=int(r.height),
                aspect_log_distance=float(abs(np.log(t.aspect_ratio/r.aspect_ratio))),
                pixel_identical=bool(t.pixel_sha256==r.pixel_sha256),source_byte_identical=bool(t.source_sha256==r.source_sha256),
                dino_cosine=float(embeddings[i,j])))
        if (i+1)%120==0:print(f'Kemiripan piksel+SSIM: {i+1}/480 test',flush=True)
    pairs=pd.DataFrame(rows).sort_values(['test_key','ssim128_best'],ascending=[True,False])
    pairs['rank_ssim']=pairs.groupby('test_key').cumcount()+1
    # Native RGB MAE is computed only for the best aligned candidate with equal dimensions.
    best=pairs[pairs.rank_ssim==1].copy();lookup=df.set_index('key');native=[];equal_fraction=[];maximum=[]
    for r in best.itertuples():
        if not r.same_native_dimensions:
            native.append(np.nan);equal_fraction.append(np.nan);maximum.append(np.nan);continue
        with Image.open(ROOT/lookup.loc[r.test_key,'processed_path']) as im:x=np.asarray(im.convert('RGB'),dtype='float32')
        with Image.open(ROOT/lookup.loc[r.train_key,'processed_path']) as im:y=np.asarray(im.convert('RGB'),dtype='float32')
        if r.best_alignment=='hflip':y=y[:,::-1]
        native.append(float(np.abs(x-y).mean()/255))
        equal_fraction.append(float(np.all(x==y,axis=2).mean()))
        maximum.append(float(np.abs(x-y).max()/255))
    best['native_rgb_mae_after_selected_alignment']=native
    best['native_exact_rgb_pixel_fraction_after_alignment']=equal_fraction
    best['native_max_channel_difference_after_alignment']=maximum
    best['rgb_mse64_selected_alignment']=np.where(best.best_alignment.eq('hflip'),best.rgb_mse64_hflip,best.rgb_mse64)
    best['conservative_similarity_candidate']=(best.ssim128_best.ge(.99)&best.rgb_mse64_selected_alignment.le(.0002)&best.aspect_log_distance.le(.03))
    sizes=set(zip(train.width,train.height));size_counts=train.groupby(['width','height']).size().to_dict()
    best['train_images_with_test_dimensions']=[int(size_counts.get((w,h),0)) for w,h in zip(best.test_width,best.test_height)]
    pairs.to_csv(REPORT/'pixel_ssim_candidate_pairs.csv',index=False)
    best.to_csv(REPORT/'all_test_best_pixel_neighbors.csv',index=False)
    assert best.test_key.nunique()==480
    dump(REPORT/'pixel_method.json',dict(full_scan='480×1920 RGB MSE at 64×64, direct and horizontal flip',
        ssim='128×128 grayscale SSIM over union of RGB top4, flipped RGB top3, DINO top2 and pHash top2 candidates; not exhaustive SSIM',
        size_metric='Native canonical RGB width and height; exact size match is descriptive only',
        resizing='BILINEAR full-frame stretch for comparable arrays; may distort aspect ratio. Native hashes checked separately.',
        exact_pixel_test_count=int(best.pixel_identical.sum()),test_same_size_available=int(sum((w,h) in sizes for w,h in zip(test.width,test.height))),
        ssim_ge_099_candidates=int(best.ssim128_best.ge(.99).sum()),
        conservative_similarity_candidates=int(best.conservative_similarity_candidate.sum()),
        screening_rule='SSIM>=0.99, selected-view RGB MSE<=0.0002, abs(log aspect-ratio quotient)<=0.03. Screening only, not calibrated duplicate truth.',
        test_count=480,semantic_labels_used_for_selection=False))
    ctx['pixel_pairs']=pairs;ctx['pixel_best']=best
    return pairs,best


def contact_sheet(ctx, frame, name, title, columns=4, annotation=None):
    frame=frame.reset_index(drop=True);width=columns*260;height=70+int(np.ceil(len(frame)/columns))*275
    sheet=Image.new('RGB',(width,height),'#f5f7fa');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);large=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    draw.text((12,12),title,font=large,fill='#172554')
    for i,r in enumerate(frame.itertuples()):
        with Image.open(ROOT/r.processed_path) as source:im=ImageOps.contain(source.convert('RGB'),(248,195))
        x=(i%columns)*260;y=60+(i//columns)*275
        sheet.paste(im,(x+(260-im.width)//2,y))
        text=f'{r.key} | {r.width}×{r.height}\n{r.display_label}'
        if annotation:text+='\n'+annotation(r)
        draw.multiline_text((x+6,y+200),'\n'.join(textwrap.fill(t,35) for t in text.splitlines()),font=font,fill='#172554',spacing=2)
    path=REPORT/'figures'/f'{name}.jpg';sheet.save(path,quality=87);return path


def make_previews(ctx):
    df=ctx['df'];paths={}
    for split,n in [('train',12),('test',24)]:
        rows=df[df.split.eq(split)].sample(n,random_state=20261106).sort_values('key')
        paths[f'{split}_sample']=contact_sheet(ctx,rows,f'{split}_sample',f'Preview {split}: sampel acak, seed tetap')
    for split in ['train','test']:
        # Balanced representatives with names from display metadata, never labels used by models.
        rows=pd.concat([g.sort_values('model_margin',ascending=False).head(2) for _,g in df[df.split.eq(split)].groupby('cluster_index')])
        paths[f'{split}_representatives']=contact_sheet(ctx,rows,f'{split}_representatives',f'{split}: contoh tiap nama kelompok sementara')
    cases=[('low_sharpness','laplacian_variance_256',True,'Indikasi ketajaman rendah; detail sederhana juga bisa bernilai rendah'),
           ('busy_scene','edge_fraction',False,'Tekstur padat / latar ramai; bukan bukti label salah'),
           ('novel_appearance','nearest_other_train_cosine',True,'Penampilan test paling jauh dari tetangga train'),
           ('ambiguous','model_margin',True,'Model paling ambigu; probabilitas belum dikalibrasi')]
    for name,metric,ascending,title in cases:
        rows=df[df.split.eq('test')].sort_values(metric,ascending=ascending).head(8)
        paths['test_'+name]=contact_sheet(ctx,rows,'test_'+name,title,annotation=lambda r,m=metric:f'{m}={getattr(r,m):.3f}')
    flags=[c for c in df.columns if c.startswith('flag_')]+['possible_multiple_regions']
    summary=df.groupby('split')[flags].sum().T.rename_axis('indikator')
    summary.to_csv(REPORT/'technical_flag_counts.csv')
    ctx['preview_paths']=paths
    return paths,summary


def crop_previews(ctx, split, method=None):
    df=ctx['df'];manifest=ctx['crops']
    available=manifest[manifest.split.eq(split)]
    if method:
        available=available[available.proposal_method.eq(method)]
        if method=='connected_components':
            conservative=set(df.loc[df.conservative_component_candidate,'key'])
            available=available[available.parent_key.isin(conservative)]
    parents=available.groupby('parent_key').size().sort_values(ascending=False).head(8).index
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    sheet=Image.new('RGB',(1080,max(1,len(parents))*300+45),'white');draw=ImageDraw.Draw(sheet)
    description='konservatif: region terpisah' if method=='connected_components' else 'hipotesis watershed: risiko membelah satu benda' if method else 'seluruh metode'
    draw.text((10,8),f'{split}: asli + crop — {description}',font=font,fill='black')
    for i,key in enumerate(parents):
        row=df.set_index('key').loc[key];sub=manifest[manifest.parent_key.eq(key)].head(3)
        with Image.open(ROOT/row.processed_path) as im:original=im.convert('RGB')
        scaled=ImageOps.contain(original,(250,210));overlay=scaled.copy();ink=ImageDraw.Draw(overlay)
        sx=scaled.width/original.width;sy=scaled.height/original.height
        for j,r in enumerate(sub.itertuples()):
            ink.rectangle([r.x0*sx,r.y0*sy,r.x1*sx,r.y1*sy],outline=['#dc2626','#2563eb','#16a34a'][j],width=3)
        y=40+i*300;sheet.paste(overlay,(8,y))
        caption=f'{key}\n'+'\n'.join(textwrap.wrap(row.display_label,34))
        draw.multiline_text((8,y+213),caption,font=font,fill='black',spacing=2)
        for j,r in enumerate(sub.itertuples()):
            with Image.open(ROOT/r.crop_path) as im:thumb=ImageOps.contain(im.convert('RGB'),(250,210))
            x=280+j*265;sheet.paste(thumb,(x,y))
            text=f'Region {chr(65+j)} | {r.crop_width}×{r.crop_height}\n{r.proposal_method}'
            probe=ctx.get('crop_probe',pd.DataFrame())
            if not probe.empty and r.crop_key in set(probe.crop_key):
                pred=probe.set_index('crop_key').loc[r.crop_key]
                text+='\n'+pred.crop_display_label
            wrapped='\n'.join(textwrap.fill(line,34) for line in text.splitlines())
            draw.multiline_text((x,y+213),wrapped,font=font,fill='black',spacing=2)
    suffix=('_'+method) if method else ''
    path=REPORT/'figures'/f'{split}_crop_proposals{suffix}.jpg';sheet.save(path,quality=87);return path


def similarity_previews(ctx):
    df=ctx['df'].set_index('key');best=ctx['pixel_best']
    # Exact matches plus high SSIM non-exact and low-similarity cases, selected numerically.
    selected=pd.concat([best[best.pixel_identical].head(3),best[~best.pixel_identical].nlargest(5,'ssim128_best'),best.nsmallest(2,'ssim128_best')]).drop_duplicates('test_key')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    sheet=Image.new('RGB',(1050,len(selected)*245+40),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'Kemiripan test–train: gambar asli, pasangan, dan selisih piksel setelah alignment',font=font,fill='black')
    for i,r in enumerate(selected.itertuples()):
        y=40+i*245
        for j,key in enumerate([r.test_key,r.train_key]):
            row=df.loc[key]
            with Image.open(ROOT/row.processed_path) as im:thumb=ImageOps.contain(im.convert('RGB'),(240,185))
            sheet.paste(thumb,(j*270+5,y))
            draw.text((j*270+5,y+188),f'{key} | {row.width}×{row.height}\n{row.display_label}',font=font,fill='black')
        with Image.open(ROOT/df.loc[r.test_key,'processed_path']) as im:a=np.asarray(im.convert('RGB').resize((192,192)),dtype='float32')
        with Image.open(ROOT/df.loc[r.train_key,'processed_path']) as im:b=np.asarray(im.convert('RGB').resize((192,192)),dtype='float32')
        if r.best_alignment=='hflip':b=b[:,::-1]
        diff=np.abs(a-b).mean(2)/255;heat=cv2.applyColorMap(np.uint8(diff*255),cv2.COLORMAP_INFERNO)[:,:,::-1]
        sheet.paste(Image.fromarray(heat),(555,y))
        draw.multiline_text((760,y+15),f'SSIM128={r.ssim128_best:.6f}\nMSE64 (align)={r.rgb_mse64_selected_alignment:.7f}\nMAE native (align)={r.native_rgb_mae_after_selected_alignment:.6f}\nUkuran native sama: {r.same_native_dimensions}\nHash pixel identik: {r.pixel_identical}\nAlignment: {r.best_alignment}',font=font,fill='black',spacing=7)
    path=REPORT/'figures/pixel_similarity_cases.jpg';sheet.save(path,quality=87);return path


def all_test_gallery(ctx):
    df=ctx['df'];best=ctx['pixel_best'].set_index('test_key');lookup=df.set_index('key')
    pages=[];tests=df[df.split.eq('test')].sort_values('key')
    def src(path,page):return html.escape(Path(os.path.relpath(ROOT/path,page.parent)).as_posix(),quote=True)
    for page_number,start in enumerate(range(0,len(tests),32),1):
        page=REPORT/'galleries'/f'test_{page_number:02d}.html';cards=[]
        for r in tests.iloc[start:start+32].itertuples():
            n=best.loc[r.key];neighbor=lookup.loc[n.train_key]
            flags=[c[5:].replace('_',' ') for c in df.columns if c.startswith('flag_') and getattr(r,c)]
            if r.possible_multiple_regions:flags.append('kandidat beberapa region')
            cards.append(f'<article><h3>{r.key}: {html.escape(r.display_label)}</h3><img loading="lazy" src="{src(r.processed_path,page)}"><img loading="lazy" src="{src(neighbor.processed_path,page)}"><p>Test {r.width}×{r.height}; train {neighbor.width}×{neighbor.height}. SSIM128 {n.ssim128_best:.3f}; exact pixel {n.pixel_identical}.</p><p>Indikator: {html.escape(", ".join(flags) or "tidak ada flag teknis")}</p></article>')
        page.write_text('<!doctype html><meta charset="utf-8"><title>Audit seluruh test</title><style>body{font:16px Arial;margin:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{border:1px solid #bbb;padding:12px}img{width:46%;height:210px;object-fit:contain}p{font-size:14px}</style><h1>Preview test dan tetangga train</h1><p>Nama adalah interpretasi tampilan sementara untuk EDA pribadi. Bukan ground truth atau hasil verifikasi kategori.</p><a href="index.html">Indeks</a><main>'+''.join(cards)+'</main>',encoding='utf-8')
        pages.append(page)
    index=REPORT/'galleries/index.html'
    index.write_text('<!doctype html><meta charset="utf-8"><h1>Seluruh 480 test</h1><ul>'+''.join(f'<li><a href="{p.name}">Halaman {i+1}: test {i*32+1}–{min((i+1)*32,480)}</a></li>' for i,p in enumerate(pages))+'</ul>',encoding='utf-8')
    dump(REPORT/'gallery_coverage.json',dict(test_images=480,pages=len(pages),image_keys=tests.key.tolist(),pixel_neighbor_coverage=best.index.nunique()))
    return index


def finalize(ctx):
    for relative,digest in ctx['frozen'].items():assert sha(ROOT/relative)==digest,relative
    crops=ctx['crops'];df=ctx['df']
    summary=dict(status='passed',source_images=2400,train_images=1920,test_images=480,
        tested_test_neighbors=480,all_test_preview_coverage=480,
        crop_count=len(crops),train_crop_count=int(crops.split.eq('train').sum()),test_preview_crop_count=int(crops.split.eq('test').sum()),
        parent_fold_inherited=True,test_crops_used_for_training=False,training_labels_changed=False,
        baseline_model_refit=False,baseline_sources_unchanged=True,display_interpretation_source=ctx['mapping']['source'],
        display_names_approved_for_competition=False,semantic_accuracy=None,
        multi_region_train=int(df[df.split.eq('train')].possible_multiple_regions.sum()),
        multi_region_test=int(df[df.split.eq('test')].possible_multiple_regions.sum()),
        frozen_input_hashes=ctx['frozen'],script_sha256=sha(Path(__file__)))
    df.drop(columns='cluster_index').to_csv(REPORT/'image_visual_audit.csv',index=False)
    dump(REPORT/'summary.json',summary)
    return summary
