"""Focused validation of stage2 outputs and immutable stage1/raw inputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import html
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reports/errors/stage2_20261105'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    summary=json.loads((REP/'diagnostics/audit_summary.json').read_text())
    checks={}
    for rel,digest in summary['source_hashes_before_and_after'].items():
        assert sha(ROOT/rel)==digest,rel
    checks['stage1_and_archive_files_unchanged']=len(summary['source_hashes_before_and_after'])
    plan=json.loads((REP/'representation_plan.json').read_text())
    for rel,digest in plan['frozen_inputs'].items():
        assert sha(ROOT/rel)==digest,rel
    checks['active_contract_unchanged']=True
    df=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    def raw_check(r):
        path=ROOT/'data'/r['source_path']
        assert sha(path)==r['source_sha256'],str(path)
        return True
    with ThreadPoolExecutor(max_workers=4) as executor:
        checks['raw_hashes_verified']=sum(executor.map(raw_check,df.to_dict('records')))
    new=ROOT/'data/processed/stage2_neighbor_v1'
    manifest=pd.read_csv(new/'cluster_manifest.csv',dtype={'image_id':str})
    assert manifest.key.is_unique and set(manifest.key)==set(df.key)
    def copy_check(r):
        assert sha(ROOT/r['folder_path'])==r['png_sha256'],r['folder_path']
        return True
    with ThreadPoolExecutor(max_workers=4) as executor:
        checks['folder_copy_hashes_verified']=sum(executor.map(copy_check,manifest.to_dict('records')))
    for split,n in [('train',1920),('test',480)]:
        folders=[p for p in (new/split).iterdir() if p.is_dir()]
        assert len(folders)==8 and len(list((new/split).glob('*/*.png')))==n
    checks['eight_folders_each_split']=True
    features={
        'resnet50':ROOT/'output/embeddings/stage1_v1/resnet50_features.npz',
        'dinov2':ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz'}
    with threadpool_limits(limits=4):
        count=0
        for name,path in features.items():
            z=np.load(path,allow_pickle=False)
            assert np.array_equal(z['keys'],df.key.to_numpy(str))
            x=normalize(z['features']).astype('float32')
            assert np.isfinite(x).all()
            for branch in ['nopca','pca64','pca128','pca256']:
                folder=ROOT/f'output/groups/stage2_repr_v1/{name}'
                model=joblib.load(folder/f'{branch}_seed11_model.joblib')
                assert model['feature_sha256']==sha(path)
                xx=x if model['reducer'] is None else normalize(model['reducer'].transform(x)).astype('float32')
                prediction=model['kmeans'].predict(xx)
                saved=pd.read_csv(folder/f'{branch}_assignments.csv')
                assert np.array_equal(saved.key,df.key) and np.array_equal(prediction,saved.cluster_index)
                count+=len(saved)
        checks['kmeans_predictions_reproduced']=count
    checkpoint=torch.load(ROOT/'output/groups/stage2_neighbor_v1/head.pt',map_location='cpu',weights_only=True)
    model=torch.nn.Linear(384,8);model.load_state_dict(checkpoint['state_dict']);model.eval()
    x=np.load(features['dinov2'])['features']
    with torch.inference_mode():
        p=model(torch.from_numpy(x)).softmax(-1).numpy()
    saved=np.load(ROOT/'output/groups/stage2_neighbor_v1/predictions.npz')
    assignments=pd.read_csv(ROOT/'output/groups/stage2_neighbor_v1/assignments.csv')
    assert np.array_equal(saved['keys'],df.key)
    assert np.allclose(p,saved['probabilities'],atol=1e-6) and np.isfinite(p).all()
    assert np.array_equal(p.argmax(1),assignments.cluster_index)
    assert np.array_equal(manifest.cluster_index,assignments.cluster_index)
    checks['trained_head_predictions_reproduced']=len(p)
    split=pd.read_csv(REP/'neighbor_refinement/group_split.csv')
    assert len(split)==1919 and split.split.eq('train').all()
    a=split[split.role.eq('fit_train')];b=split[split.role.eq('heldout_train')]
    assert not set(a.candidate_group)&set(b.candidate_group)
    assert not set(a.pixel_sha256)&set(b.pixel_sha256)
    checks['train_holdout_group_overlap']=0
    # The three real exact train/test duplicate pairs must receive the same assignments.
    merged=df[['key','pixel_sha256']].merge(assignments[['key','cluster_index']],on='key')
    assert merged.groupby('pixel_sha256').cluster_index.nunique().max()==1
    checks['pixel_identical_assignment_consistent']=True
    galleries=list((REP/'galleries').glob('*.html'))+[REP/'diagnostics/all_disagreements.html']
    links=0
    for page in galleries:
        for rel in re.findall(r'(?:src|href)="([^"]+)"',page.read_text(encoding='utf-8')):
            if rel.startswith(('http:','https:','#','data:')):
                continue
            assert (page.parent/html.unescape(rel)).resolve().exists(),(page,rel)
            links+=1
    checks['gallery_file_targets_checked']=links
    checks['all_480_test_neighbors']=pd.read_csv(REP/'dinov2_test_train_top5.csv').test_key.nunique()==480
    assert checks['all_480_test_neighbors']
    # Preserve provenance for all new code and main outputs.
    files=list((ROOT/'eksperimen/skrip').glob('stage2_*.py'))
    files += [Path(__file__),ROOT/'eksperimen/skrip/materialize_stage2_clusters.py',
              ROOT/'eksperimen/skrip/audit_stage1_disagreements.py']
    result=dict(status='passed',checks=checks,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
                scope='Integrity, inference reproducibility, train-only split, coverage. NOT semantic Accuracy.',
                category_accuracy=None,submission_created=False)
    (REP/'final_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
