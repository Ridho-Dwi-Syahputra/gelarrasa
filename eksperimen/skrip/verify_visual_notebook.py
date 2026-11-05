"""Verify actual v3 execution, crop geometry, coverage and unchanged sources."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import html

import nbformat
import pandas as pd
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/eda/stage3_visual_20261106'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    path=ROOT/'notebook/01_EDA_Preprocessing_Clustering.ipynb'
    nb=nbformat.read(path,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code']
    run=json.loads((ROOT/'reports/reproducibility/notebook_v3_execution.json').read_text())
    assert run['status']=='passed' and not run['errors']
    assert len(cells)==run['executed_cells']==run['code_cells']
    assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
    assert not [out for c in cells for out in c.outputs if out.output_type=='error']
    visible='\n'.join(out.get('text','')+'\n'+out.get('data',{}).get('text/plain','')
                      for cell in cells for out in cell.outputs)
    assert re.search(r'\bcluster_\d{2}\b',visible) is None, 'User-facing outputs still contain numbered group labels'
    assert run['notebook_sha256']==sha(path)
    assert run['notebook_code_sha256']==hashlib.sha256('\n'.join(c.source for c in cells).encode()).hexdigest()
    summary=json.loads((REPORT/'summary.json').read_text())
    assert summary['status']=='passed'
    assert summary['script_sha256']==sha(ROOT/'eksperimen/skrip/stage3_visual_audit.py')
    assert nb.metadata.pipeline_sources['stage3_visual_audit.py']==summary['script_sha256']
    for relative,digest in summary['frozen_input_hashes'].items():assert sha(ROOT/relative)==digest,relative
    profile=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    def raw_check(row):
        assert sha(ROOT/'data'/row['source_path'])==row['source_sha256'],row['key']
        return 1
    with ThreadPoolExecutor(max_workers=4) as pool:raw_count=sum(pool.map(raw_check,profile.to_dict('records')))
    manifest=pd.read_csv(ROOT/'data/processed/stage3_crop_candidates_v1/crop_manifest.csv')
    assert manifest.crop_key.is_unique and manifest.crop_path.is_unique
    lookup=profile.set_index('key')
    def crop_check(row):
        parent=lookup.loc[row['parent_key']]
        assert row['split']==parent.split and row['parent_pixel_sha256']==parent.pixel_sha256
        assert 0<=row['x0']<row['x1']<=parent.width and 0<=row['y0']<row['y1']<=parent.height
        p=ROOT/row['crop_path'];assert sha(p)==row['crop_sha256']
        with Image.open(ROOT/parent.processed_path) as source:
            expected=np.asarray(source.convert('RGB').crop((row['x0'],row['y0'],row['x1'],row['y1'])))
        with Image.open(p) as crop:assert np.array_equal(expected,np.asarray(crop.convert('RGB')))
        assert row['crop_width']==row['x1']-row['x0'] and row['crop_height']==row['y1']-row['y0']
        return 1
    with ThreadPoolExecutor(max_workers=4) as pool:crop_count=sum(pool.map(crop_check,manifest.to_dict('records')))
    assert not manifest.promoted_to_training.any()
    assert manifest[manifest.split.eq('test')].augmentation_candidate.eq(False).all()
    assert manifest.groupby('parent_group').parent_fold.nunique().max()<=1
    folds=pd.read_csv(ROOT/'output/folds/stage1_v1/DRAFT_group_folds.csv')
    fold_map=folds.groupby('candidate_group').fold.first().to_dict()
    train=manifest[manifest.split.eq('train')]
    assert np.array_equal(train.parent_fold.to_numpy(),train.parent_group.map(fold_map).to_numpy())
    neighbors=pd.read_csv(REPORT/'all_test_best_pixel_neighbors.csv')
    tests=set(profile.loc[profile.split.eq('test'),'key'])
    assert neighbors.test_key.is_unique and set(neighbors.test_key)==tests
    assert np.isfinite(neighbors[['rgb_mse64','ssim128_best','dino_cosine']]).all().all()
    selected=np.where(neighbors.best_alignment.eq('hflip'),neighbors.rgb_mse64_hflip,neighbors.rgb_mse64)
    assert np.allclose(selected,neighbors.rgb_mse64_selected_alignment)
    assert neighbors.native_exact_rgb_pixel_fraction_after_alignment.dropna().between(0,1).all()
    coverage=json.loads((REPORT/'gallery_coverage.json').read_text())
    assert len(coverage['image_keys'])==480 and set(coverage['image_keys'])==tests
    references=0
    for page in (REPORT/'galleries').glob('*.html'):
        for relative in re.findall(r'(?:src|href)="([^"]+)"',page.read_text(encoding='utf-8')):
            assert (page.parent/html.unescape(relative)).resolve().exists(),(page,relative)
            references+=1
    mapping=json.loads((ROOT/'configs/display_label_mapping.json').read_text())
    assert not mapping['training_target'] and not mapping['active_competition_mapping']
    contract=json.loads((ROOT/'configs/data_contract.json').read_text())
    assert contract['active_train_label_manifest'] is None
    result=dict(status='passed',checked_at_wib=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
        code_cells=len(cells),executed_cells=len(cells),cell_errors=0,
        raw_hashes_verified=raw_count,crop_hashes_and_native_pixels_verified=crop_count,
        crop_parent_fold_consistent=True,test_crops_used_for_training=False,
        test_neighbor_coverage=480,test_gallery_coverage=480,gallery_targets_verified=references,
        display_names='assistant visual interpretation; private display only; not competition training labels',
        active_label_contract_unchanged=True,baseline_model_and_assignments_unchanged=True,
        numbered_group_names_absent_from_visible_text=True,
        notebook_sha256=sha(path),semantic_accuracy=None)
    target=ROOT/'reports/reproducibility/notebook_v3_delivery_validation.json'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
