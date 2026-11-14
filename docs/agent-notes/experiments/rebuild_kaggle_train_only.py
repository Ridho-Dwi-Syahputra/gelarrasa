"""Replace the frozen label payload with a train-only preprocessing pipeline."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / 'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb'
notebook = json.loads(TARGET.read_text(encoding='utf-8'))


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(True)}


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.splitlines(True)}


notebook['cells'][0] = markdown('''# GELAR RASA 2026 — EDA, preprocessing, GPU training, submission

Notebook ini membaca gambar mentah dari dataset Kaggle, menghitung fitur **train saja** untuk membentuk kelompok dan label otomatis, lalu membuat crop dan delapan folder processed. Training model hanya dimulai bila pemeriksaan seed yang ditahan mencapai ambang kualitas; hasil yang terlalu lemah menghentikan Run All sebelum membuang waktu GPU. Tidak ada tabel assignment tersembunyi atau payload label terkompresi. Ke-37 seed yang sudah ditinjau peserta ditulis terbuka di sel preprocessing; tiap kelas paling banyak lima seed. Daftar pengecualian dan koordinat crop yang pernah ditetapkan peserta juga terlihat langsung di sel itu.

Label baru berasal dari EfficientNetV2-S image-only, fitur warna, KMeans, SVM seed, kNN seed, propagasi graf train, serta alokasi delapan kelas yang seimbang. Test hanya melalui preprocessing dan inferensi; kategori sample submission diabaikan. Jumlah eligible tetap **1.879 foto utuh + 7 crop = 1.886 sampel** karena kebijakan pengecualian dan crop yang sama, tetapi **assignment kategori dapat berbeda** dari snapshot historis. CSV `labels/train_targets.csv` dan laporan `processed/processing_report.json` mencatat hasil Run All. Skor validasi adalah agreement terhadap label otomatis, bukan ground truth independen.

Tahap model berikutnya memakai DINOv3/SVM/group-kNN, Extra Trees, XGBoost, empat CNN, dan kebijakan ensemble berdasarkan validasi berkelompok train. Semua output ditulis di `/kaggle/working/gelarrasa_run/`.
''')
notebook['cells'][1]['source']=''.join(notebook['cells'][1]['source']).replace('active_processed_hybrid_v3','train_only_cluster_gpu_v1').splitlines(True)

notebook['cells'][6] = markdown('''## 3. Bentuk sampel dan fitur train dari gambar mentah

Sel ini menentukan kebijakan crop dan pengecualian secara terbuka. Citra RGB yang sudah diorientasi dipadatkan ke kanvas 224×224, lalu encoder EfficientNetV2-S non-VLM menghitung fitur visual. Hanya 37 seed train yang ditinjau peserta menjadi target manusia. File test diproses untuk inferensi, tanpa label atau fitting.
''')

notebook['cells'][7] = code('''from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from scipy.fft import dctn
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from collections import Counter

HUMAN_SEEDS = {
    'plastic': ['0056','0085','0237','1051','1115'],
    'metal': ['0072','1180','1605'],
    'electronic': ['0086','0956','1132','1434','1863'],
    'paper': ['0141','0256','0396','1114','1637'],
    'foodwaste': ['0200','0265','0636','1039'],
    'glass': ['0251','0889','1399','1685','1855'],
    'cardboard': ['0346','0955','1027','1215','1522'],
    'textile': ['0403','0504','1154','1332','1451'],
}
assert sum(map(len,HUMAN_SEEDS.values()))==37 and max(map(len,HUMAN_SEEDS.values()))<=5
SEED_LABELS={sample_id:category for category,ids in HUMAN_SEEDS.items() for sample_id in ids}
assert len(SEED_LABELS)==37
POLICY={
    'drop': ['0134','0298','0545','0963','1028','1043','1465','1890'],
    'reserve': ['0376'],
    'hold': ['0058','0079','0147','0160','0183','0189','0225','0310','0321','0400','0498','0559','0964','0978','1018','1142','1157','1281','1436','1515','1531','1657','1662','1848','1885','1904','1913'],
    'crops': [
        ('0415','calculator',(0,115,235,522)),
        ('0437','laptop',(0,44,373,429)),
        ('0437','apple',(370,292,466,427)),
        ('0951','printed_page',(143,145,445,398)),
        ('0951','calculator',(456,142,640,503)),
        ('1007','red_bag',(267,63,472,232)),
        ('1351','lowerleft_box',(0,349,118,484)),
    ],
    'output_size':224,'max_content_side':202,'padding_rgb':[127,127,127],
}
assert len(POLICY['hold'])==27 and len(POLICY['crops'])==7
assert not (set(POLICY['drop'])|set(POLICY['reserve'])|set(POLICY['hold'])) & set(SEED_LABELS)
def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def rgb_source(path):
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source)
        if 'A' in im.getbands() or 'transparency' in im.info:
            rgba=im.convert('RGBA')
            im=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba)
        return im.convert('RGB').copy()
def processed_canvas(im):
    side=POLICY['max_content_side']; size=POLICY['output_size']
    content=ImageOps.contain(im,(side,side),method=Image.Resampling.LANCZOS)
    out=Image.new('RGB',(size,size),tuple(POLICY['padding_rgb']))
    out.paste(content,((size-content.width)//2,(size-content.height)//2))
    return out
raw_sources=[]
for split,paths in [('train',raw_train_paths),('test',raw_test_paths)]:
    for path in paths: raw_sources.append({'split':split,'file_name':path.name,'sha256':file_sha256(path)})
assert len(raw_sources)==2400 and len({(r['split'],r['file_name']) for r in raw_sources})==2400
PROCESSED_ROOT=WORK_ROOT/'processed'
STAGING=PROCESSED_ROOT/'staging'; STAGING.mkdir(parents=True,exist_ok=True)
crop_by_parent={}
for parent,region,bbox in POLICY['crops']:crop_by_parent.setdefault(parent,[]).append((region,bbox))
sample_rows=[]
for split,paths in [('train',raw_train_paths),('test',raw_test_paths)]:
    for path in paths:
        parent=path.stem
        if split=='train' and parent in set(POLICY['drop']+POLICY['reserve']):continue
        im=rgb_source(path)
        regions=crop_by_parent[parent] if split=='train' and parent in crop_by_parent else [(None,(0,0,im.width,im.height))]
        for region,bbox in regions:
            x0,y0,x1,y1=bbox
            assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height,(parent,region,bbox,im.size)
            sample_id=parent+'__'+region if region else parent
            stage=STAGING/split/f'{sample_id}.png';stage.parent.mkdir(parents=True,exist_ok=True)
            processed_canvas(im.crop(bbox)).save(stage,format='PNG')
            sample_rows.append({'split':split,'sample_id':sample_id,'parent_id':parent,'source_file':path.name,
                'source_path':str(path),'bbox':list(bbox),'is_crop':region is not None,'region_name':region or '',
                'training_eligible':split=='train' and parent not in POLICY['hold'],'staging_path':str(stage)})
candidates=pd.DataFrame(sample_rows)
assert candidates.sample_id.is_unique or candidates.groupby('split').sample_id.apply(lambda s:s.is_unique).all()
assert len(candidates.query("split == 'train'"))==1913 and len(candidates.query("split == 'test'"))==480

class StageDataset(Dataset):
    def __init__(self,paths):
        self.paths=list(paths)
        self.tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.paths[index]) as im:return self.tf(im.convert('RGB'))
try: feature_encoder=models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).eval().to(DEVICE)
except Exception as exc:raise RuntimeError('Bobot EfficientNetV2-S tidak tersedia. Aktifkan Internet Kaggle.') from exc
feature_encoder.classifier=torch.nn.Identity()
def extract_stage_features(paths):
    loader=DataLoader(StageDataset(paths),batch_size=24,shuffle=False,num_workers=2 if os.name!='nt' else 0,pin_memory=DEVICE.type=='cuda')
    chunks=[]
    with torch.inference_mode(),torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda',dtype=torch.float16):
        for batch in loader:chunks.append(feature_encoder(batch.to(DEVICE,non_blocking=True)).float().cpu().numpy())
    return normalize(np.concatenate(chunks).astype(np.float32))
stage_features=extract_stage_features(candidates.staging_path)
assert stage_features.shape[0]==len(candidates) and np.isfinite(stage_features).all()
def color_descriptor(path):
    import cv2
    rgb=np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hist=[]
    for channel,bins,high in [(0,18,180),(1,12,256),(2,12,256)]:
        count=np.histogram(hsv[:,:,channel],bins=bins,range=(0,high))[0].astype(np.float32)
        hist.extend(count/max(1,count.sum()))
    pix=rgb.astype(np.float32)/255
    return np.asarray(hist+list(pix.mean((0,1)))+list(pix.std((0,1))),dtype=np.float32)
stage_color=np.stack([color_descriptor(path) for path in candidates.staging_path])
del feature_encoder;gc.collect();torch.cuda.empty_cache()
print('Train-only input rows:',int(candidates.split.eq('train').sum()),'| test inference rows:',int(candidates.split.eq('test').sum()))
''')

notebook['cells'][8] = markdown('''## 4. Cluster train, propagasi label, dan materialisasi processed

KMeans dan graf kNN di-fit **hanya pada train**. Probabilitas SVM, kNN seed, dan propagasi graf digabung, lalu dirata-ratakan ringan per subcluster. Seed peserta tetap sebagai jangkar. Foto yang ditahan tetap tercatat, tanpa masuk training. Semua crop berbagi source group induknya. Manifest dan laporan metode dibuat saat Run All; tidak dibaca dari file label lain.
''')

notebook['cells'][9] = code('''train_mask=candidates.split.eq('train').to_numpy()
train_candidates=candidates.loc[train_mask].reset_index(drop=True)
train_feature=stage_features[train_mask]
train_color=stage_color[train_mask]
fit_mask=train_candidates.training_eligible.to_numpy(bool)
color_scaler=StandardScaler().fit(train_color[fit_mask])
color_view=normalize(color_scaler.transform(train_color).astype(np.float32))
cluster_feature=normalize(np.concatenate([train_feature,.25*color_view],axis=1))
seed_indices=np.flatnonzero(train_candidates.parent_id.isin(SEED_LABELS))
assert len(seed_indices)==37 and train_candidates.iloc[seed_indices].is_crop.sum()==0
assert fit_mask[seed_indices].all()
seed_target=np.array([CLASSES.index(SEED_LABELS[x]) for x in train_candidates.iloc[seed_indices].parent_id],dtype=np.int64)
assert len(np.unique(seed_target))==len(CLASSES)

# The only independent labels available to this raw-only pipeline are the 37
# participant-reviewed seed images. Estimate whether they can support the
# automatic assignment before spending GPU hours on student models.
seed_cv_prediction=np.full(len(seed_indices),-1,dtype=int)
for fit_seed,check_seed in StratifiedKFold(3,shuffle=True,random_state=SEED).split(cluster_feature[seed_indices],seed_target):
    fold_model=SVC(C=3.0,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=SEED)
    fold_model.fit(cluster_feature[seed_indices[fit_seed]],seed_target[fit_seed])
    fold_prob=fold_model.predict_proba(cluster_feature[seed_indices[check_seed]])
    seed_cv_prediction[check_seed]=fold_model.classes_[fold_prob.argmax(axis=1)]
LABEL_CV_AGREEMENT=float(np.mean(seed_cv_prediction==seed_target))
print(f'Participant-seed held-out diagnostic: {LABEL_CV_AGREEMENT:.3f} on {len(seed_indices)} seed predictions')

CLUSTER_COUNT=96
clusterer=MiniBatchKMeans(n_clusters=CLUSTER_COUNT,batch_size=512,n_init=5,random_state=SEED)
clusterer.fit(cluster_feature[fit_mask])
cluster_ids=clusterer.predict(cluster_feature)
svm=SVC(C=3.0,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=SEED)
svm.fit(cluster_feature[seed_indices],seed_target)
svm_prob=svm.predict_proba(cluster_feature).astype(np.float32)
assert list(svm.classes_)==list(range(len(CLASSES)))
seed_similarity=cluster_feature@cluster_feature[seed_indices].T
seed_dist=1.-seed_similarity
nearest_seed=np.argsort(seed_dist,axis=1)[:,:min(7,len(seed_indices))]
neighbor_dist=np.take_along_axis(seed_dist,nearest_seed,axis=1)
neighbor_weight=1./np.maximum(neighbor_dist,.025)**2
knn_prob=np.zeros_like(svm_prob)
for i in range(len(train_candidates)):
    np.add.at(knn_prob[i],seed_target[nearest_seed[i]],neighbor_weight[i])
knn_prob/=np.maximum(knn_prob.sum(axis=1,keepdims=True),1e-12)
base_prob=.60*svm_prob+.40*knn_prob
base_prob[seed_indices]=np.eye(len(CLASSES),dtype=np.float32)[seed_target]

eligible_positions=np.flatnonzero(fit_mask)
eligible_feature=cluster_feature[fit_mask]
neighbors=NearestNeighbors(n_neighbors=21,metric='cosine',algorithm='brute').fit(eligible_feature)
dist,adj=neighbors.kneighbors(eligible_feature)
rows=np.repeat(np.arange(len(eligible_positions)),20); cols=adj[:,1:].reshape(-1)
weights=np.exp(-dist[:,1:].reshape(-1)/.06).astype(np.float32)
graph=sparse.csr_matrix((weights,(rows,cols)),shape=(len(eligible_positions),)*2)
graph=(graph+graph.T)*.5
graph=sparse.diags(1./np.maximum(np.asarray(graph.sum(axis=1)).ravel(),1e-12))@graph
seed_positions=np.searchsorted(eligible_positions,seed_indices)
prop_fit=base_prob[fit_mask].copy()
for _ in range(35):
    prop_fit=.65*(graph@prop_fit)+.35*base_prob[fit_mask]
    prop_fit[seed_positions]=base_prob[seed_indices]
prop=base_prob.copy();prop[fit_mask]=prop_fit
if (~fit_mask).any():
    held_dist,held_adj=neighbors.kneighbors(cluster_feature[~fit_mask],n_neighbors=20)
    held_weight=np.exp(-held_dist/.06);held_weight/=np.maximum(held_weight.sum(axis=1,keepdims=True),1e-12)
    prop[~fit_mask]=.7*base_prob[~fit_mask]+.3*(held_weight[:,:,None]*prop_fit[held_adj]).sum(axis=1)
cluster_mean=np.stack([prop[fit_mask & (cluster_ids==k)].mean(axis=0) if np.any(fit_mask & (cluster_ids==k)) else np.full(len(CLASSES),1/len(CLASSES)) for k in range(CLUSTER_COUNT)])
final_prob=.85*prop+.15*cluster_mean[cluster_ids]
final_prob[seed_indices]=base_prob[seed_indices]
assigned_idx=final_prob.argmax(axis=1)
assigned_idx[seed_indices]=seed_target

# A balanced assignment prevents rare classes from vanishing with only 37
# human-reviewed seeds. The prior is uniform across the eight known classes;
# it is not estimated from old labels or from test images.
eligible_positions=np.flatnonzero(fit_mask)
free_positions=eligible_positions[~np.isin(eligible_positions,seed_indices)]
quotas=np.full(len(CLASSES),len(eligible_positions)//len(CLASSES),dtype=int)
quotas[:len(eligible_positions)%len(CLASSES)]+=1
seed_counts=np.bincount(seed_target,minlength=len(CLASSES))
slots=np.repeat(np.arange(len(CLASSES)),quotas-seed_counts)
assert len(slots)==len(free_positions)
score=np.log(np.maximum(final_prob[free_positions][:,slots],1e-9))
selected_rows,selected_slots=linear_sum_assignment(-score)
assigned_idx[free_positions[selected_rows]]=slots[selected_slots]
assigned_idx[seed_indices]=seed_target

# Group only source-like near duplicates; crop and its parent inherit the same group.
def phash(path):
    with Image.open(path) as im:gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),np.float32)
    low=dctn(gray,type=2,norm='ortho')[:8,:8]
    return (low>np.median(low[1:])).reshape(64)
unique_parents=train_candidates.drop_duplicates('parent_id').reset_index(drop=True)
parent_index={name:i for i,name in enumerate(unique_parents.parent_id)}
representative=np.array([train_candidates.index[train_candidates.parent_id.eq(p)].tolist()[0] for p in unique_parents.parent_id])
parent_hash=np.stack([phash(train_candidates.iloc[i].staging_path) for i in representative])
parent_feature=train_feature[representative]
uf=list(range(len(unique_parents)))
def root(i):
    while uf[i]!=i:
        uf[i]=uf[uf[i]];i=uf[i]
    return i
for i in range(len(unique_parents)):
    hamming=np.count_nonzero(parent_hash[i+1:]!=parent_hash[i],axis=1)
    cosine=parent_feature[i+1:]@parent_feature[i]
    for off in np.flatnonzero((hamming<=4)&(cosine>=.96)):
        j=i+1+int(off);a,b=root(i),root(j)
        if a!=b:uf[max(a,b)]=min(a,b)
parent_groups={p:'source_'+unique_parents.iloc[root(i)].parent_id for p,i in parent_index.items()}
group_ids=np.array([parent_groups[p] for p in train_candidates.parent_id])
for group in set(group_ids):
    idx=np.flatnonzero(group_ids==group)
    # Crops of one source may depict different materials (e.g. laptop/apple).
    # Share the validation group, but never force their category to match.
    full=idx[(~train_candidates.iloc[idx].is_crop.to_numpy()) & train_candidates.iloc[idx].training_eligible.to_numpy(bool)]
    fixed=full[np.isin(full,seed_indices)]
    if len(fixed) and len(set(assigned_idx[fixed]))==1:assigned_idx[full]=assigned_idx[fixed[0]]
    elif len(full)>1:assigned_idx[full]=np.bincount(assigned_idx[full],minlength=len(CLASSES)).argmax()
assigned_idx[seed_indices]=seed_target

train_candidates['cluster_id']=cluster_ids
train_candidates['parent_group']=group_ids
train_candidates['assigned_class']=[CLASSES[i] for i in assigned_idx]
train_candidates['confidence']=final_prob.max(axis=1)
train_candidates['human_reviewed']=train_candidates.parent_id.isin(SEED_LABELS)
train_candidates['label_source']=np.where(train_candidates.human_reviewed,'participant_reviewed_seed','train_only_balanced_cluster_graph_v2')
train_candidates['deferral_reason']=np.where(train_candidates.training_eligible,'','participant_noise_cleaning')
test_candidates=candidates.loc[~train_mask].copy().reset_index(drop=True)
test_candidates['cluster_id']=-1;test_candidates['parent_group']=['test_'+p for p in test_candidates.parent_id]
test_candidates['assigned_class']='';test_candidates['confidence']=np.nan
test_candidates['human_reviewed']=False;test_candidates['label_source']='unlabeled_test';test_candidates['deferral_reason']=''
processed=pd.concat([train_candidates,test_candidates],ignore_index=True)
for row in processed.itertuples():
    source=Path(row.staging_path)
    if row.split=='test':dest=PROCESSED_ROOT/'test'/f'{row.sample_id}.png'
    elif row.training_eligible:dest=PROCESSED_ROOT/'train'/row.assigned_class/f'{row.sample_id}.png'
    else:dest=PROCESSED_ROOT/'review/excluded_train'/row.assigned_class/f'{row.sample_id}.png'
    dest.parent.mkdir(parents=True,exist_ok=True)
    source.replace(dest)
    processed.at[row.Index,'output_path']=str(dest)
    if row.is_crop:
        native=PROCESSED_ROOT/'crops'/f'{row.sample_id}.png';native.parent.mkdir(parents=True,exist_ok=True)
        rgb_source(row.source_path).crop(tuple(row.bbox)).save(native,format='PNG')
        processed.at[row.Index,'native_crop_path']=str(native)
    else:processed.at[row.Index,'native_crop_path']=''
    processed.at[row.Index,'actual_pixel_sha256']=hashlib.sha256(rgb_source(dest).tobytes()).hexdigest()
for reserve in POLICY['reserve']:
    source=next(p for p in raw_train_paths if p.stem==reserve)
    dest=PROCESSED_ROOT/'review/reserved'/f'{reserve}.png';dest.parent.mkdir(parents=True,exist_ok=True)
    rgb_source(source).save(dest,format='PNG')
processed.drop(columns='staging_path').to_csv(PROCESSED_ROOT/'manifest.csv',index=False)
source_checks=pd.DataFrame(raw_sources);source_checks.to_csv(WORK_ROOT/'eda/source_integrity.csv',index=False)
train_frame=processed[processed.split.eq('train')&processed.training_eligible].reset_index(drop=True)
test_frame=processed[processed.split.eq('test')].reset_index(drop=True)
assert len(train_frame)==1886 and int(train_frame.is_crop.sum())==7 and len(test_frame)==480
assert train_frame.assigned_class.value_counts().reindex(CLASSES,fill_value=0).gt(0).all()
assert test_frame.assigned_class.eq('').all()
assert train_frame[train_frame.human_reviewed].assigned_class.value_counts().le(5).all()
assert train_frame.groupby('parent_id').parent_group.nunique().le(1).all()
train_paths=[Path(p) for p in train_frame.output_path];test_paths=[Path(p) for p in test_frame.output_path]
train_ids=train_frame.sample_id.tolist();test_ids=test_frame.parent_id.map(norm_id).tolist()
pseudo=np.array([CLASSES.index(c) for c in train_frame.assigned_class],np.int64)
eligible_idx=np.arange(len(train_frame));groups=train_frame.parent_group.astype(str).to_numpy()
assert set(test_ids)==set(sample.image_id_norm)
train_frame.to_csv(WORK_ROOT/'labels/train_targets.csv',index=False)
label_payload={'policy':POLICY,'seed_labels':SEED_LABELS,'sources':raw_sources,
    'train_assignment':train_frame[['sample_id','assigned_class','parent_group','cluster_id']].to_dict('records'),
    'feature_model':'torchvision efficientnet_v2_s IMAGENET1K_V1','algorithm':'svm_seed_knn_seed_graph_kmeans96_balanced_v2','seed':SEED}
STATE_SHA256=hashlib.sha256(json.dumps(label_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(WORK_ROOT/'labels/processing_state.json').write_text(json.dumps(label_payload,indent=2),encoding='utf-8')
proof={'state_sha256':STATE_SHA256,'raw_source_count':len(raw_sources),'train_candidates':len(train_candidates),
    'eligible_train':len(train_frame),'eligible_full_frame':int((~train_frame.is_crop).sum()),
    'eligible_crops':int(train_frame.is_crop.sum()),'held_train_samples':int((~train_candidates.training_eligible).sum()),
    'test_unlabeled':len(test_frame),'train_source_groups':int(train_frame.parent_group.nunique()),
    'class_counts':train_frame.assigned_class.value_counts().to_dict(),'human_seed_count':len(seed_indices),
    'seed_cv_agreement':LABEL_CV_AGREEMENT,'minimum_seed_cv_for_training':0.70,
    'labels_recomputed':True,'label_method':'train_only_balanced_cluster_graph_v2',
    'test_used_in_fitting':False,'previous_snapshot_used':False}
(PROCESSED_ROOT/'processing_report.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))
if LABEL_CV_AGREEMENT<0.70:
    raise RuntimeError(
        f'Automatic train labels are not reliable enough for GPU training: '
        f'held-out participant-seed agreement={LABEL_CV_AGREEMENT:.3f}<0.70. '
        'Processed images and the diagnostic manifest were saved. '
        'The historical 1,886-label assignment cannot be regenerated by this '
        '37-seed method from raw images alone; do not train on these labels.')
''')

for index in [11,14,16,18,20,22,24]:
    if index==11:
        content=''.join(notebook['cells'][index]['source']).replace('Distribusi train processed aktif','Distribusi train hasil algoritma').replace("print('Low-confidence exclusion from previous Kaggle version is disabled; frozen eligible rows retained:',len(train_frame))","print('Train eligible setelah cluster, crop, dan kebijakan pengecualian:',len(train_frame))")
        notebook['cells'][index]['source']=content.splitlines(True)
    elif notebook['cells'][index]['cell_type']=='markdown':
        content=''.join(notebook['cells'][index]['source']).replace('assignment processed yang sudah dibekukan','assignment train yang dihitung di sel 4').replace('snapshot', 'hasil algoritma train-only').replace('processed aktif','processed hasil Run All')
        notebook['cells'][index]['source']=content.splitlines(True)

notebook['metadata'].pop('gelarrasa_pipeline',None)
notebook['metadata']['gelarrasa_pipeline']={'version':'train_only_cluster_gpu_v1','label_origin':'computed_in_notebook_from_raw_train'}
notebook['cells'][25]['source']=''.join(notebook['cells'][25]['source']).replace('processing_equivalence','processing_report').replace('processed/equivalence_report.json','processed/processing_report.json').splitlines(True)
TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Rebuilt notebook:',TARGET,'cells',len(notebook['cells']))
