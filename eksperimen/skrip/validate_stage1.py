"""Validate concrete stage-1 artifacts after notebook Run All."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/eda/stage1_20261105'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return pd.read_csv(p,dtype={'image_id':str,'test_id':str,'train_id':str})


def run():
    profile=read(ROOT/'output/manifests/stage1_v1/image_profile.csv')
    clusters=read(ROOT/'output/groups/stage1_v1/cluster_assignments.csv')
    groups=read(ROOT/'output/groups/stage1_v1/groups.csv')
    assert len(profile)==2400 and profile.key.is_unique
    assert profile.groupby('split').size().to_dict()=={'test':480,'train':1920}
    assert profile.decode_status.eq('ok').all()
    assert profile.key.tolist()==clusters.key.tolist()==groups.key.tolist()
    assert clusters.semantic_category.fillna('').eq('').all()
    checked=0
    for r in profile.itertuples():
        assert sha(ROOT/'data'/r.source_path)==r.source_sha256, r.key
        for path_col,hash_col in [('processed_path','processed_sha256'),('preview_path','preview_sha256')]:
            p=ROOT/getattr(r,path_col)
            assert sha(p)==getattr(r,hash_col), str(p)
            checked+=1
    for split,n in [('train',1920),('test',480)]:
        p=ROOT/'data/processed/stage1_v1/cluster_views'/split
        files=list(p.glob('cluster_*/*.png'))
        assert len(files)==n and len({f.stem for f in files})==n
        assert len(list(p.glob('cluster_*')))==8
        assignments=clusters[clusters.split.eq(split)].set_index('image_id')
        for f in files:
            assert f.parent.name==assignments.loc[f.stem,'cluster_id']
            src=ROOT/assignments.loc[f.stem,'processed_path']
            assert sha(f)==sha(src)
    ga=read(ROOT/'output/groups/stage1_v1/candidate_group_audit.csv')
    assert not ga.loc[ga.n.eq(2),'chaining_risk'].any(), 'Two-member group cannot have a longer chain'
    fit=json.loads((ROOT/'output/groups/stage1_v1/cluster_provenance.json').read_text())
    assert all(key.startswith('train:') for key in fit['fit_keys'])
    assert len(fit['fit_keys'])==profile[profile.split.eq('train')].pixel_sha256.nunique()
    assert sha(ROOT/'output/groups/stage1_v1/cluster_assignments.csv')==fit['cluster_assignment_sha256']
    for name in ['resnet50_features.npz','classical_features.npz']:
        arrays=np.load(ROOT/'output/embeddings/stage1_v1'/name)
        assert list(arrays['keys'])==profile.key.tolist()
        for k in arrays.files:
            if k!='keys':assert np.isfinite(arrays[k]).all()
    for filename in ['test_train_phash_top10.csv','test_train_embedding_top10.csv']:
        table=read(REPORT/filename)
        assert len(table)==4800 and table.groupby('test_id').size().eq(10).all()
        assert set(table.test_id)==set(profile.loc[profile.split.eq('test'),'image_id'])
        assert set(table.train_id)<=set(profile.loc[profile.split.eq('train'),'image_id'])
        assert not table.duplicated(['test_id','rank']).any()
    folds=read(ROOT/'output/folds/stage1_v1/DRAFT_group_folds.csv')
    assert len(folds)==1920 and folds.groupby('candidate_group').fold.nunique().max()==1
    domain=read(REPORT/'domain_oof.csv').merge(groups[['key','candidate_group']],on='key',validate='one_to_one')
    assert len(domain)==2400 and domain.groupby('candidate_group').fold.nunique().max()==1
    for col in ['metadata_oof_domain_probability','cnn_fixed_features_oof_domain_probability']:
        assert domain[col].between(0,1).all()
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href'] and v:self.links.append(v)
    links=set()
    for page in (REPORT/'galleries').glob('*.html'):
        parser=Links();parser.feed(page.read_text(encoding='utf-8'))
        for u in parser.links:
            parsed=urlsplit(u)
            if not parsed.scheme and parsed.path:
                target=(page.parent/unquote(parsed.path)).resolve()
                assert target.exists(),(page.name,u)
                links.add(target)
    assert (REPORT/'galleries/all_test_neighbors.html').read_text(encoding='utf-8').count('<section id="test-')==480
    figures=list((REPORT/'figures').glob('*.png'))
    assert len(figures)==10
    for p in figures:
        with Image.open(p) as im:im.verify()
    nb=nbformat.read(ROOT/'notebook/01_EDA_Preprocessing_Clustering.ipynb',as_version=4)
    nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)==25 and all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    for filename,digest in nb.metadata.pipeline_sources.items():
        assert sha(ROOT/'eksperimen/skrip'/filename)==digest,'Notebook source does not match script'
    assert json.loads((REPORT/'notebook_execution.json').read_text())['status']=='passed'
    result=dict(status='passed',source_hashes_checked=2400,processed_hashes_checked=checked,
                cluster_view_hashes_checked=2400,train_images=1920,test_images=480,
                code_cells_executed=len(code),figure_files=len(figures),gallery_local_targets=len(links),
                test_neighbor_sections=480,group_overlap_in_draft_folds=0,
                semantic_labels_ready=False,category_accuracy_measured=False)
    (REPORT/'final_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
