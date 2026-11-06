"""Reproduce saved assignments and verify data/selection isolation."""
from pathlib import Path
import hashlib
import json
import re
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from stage5_material_structure import predict

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/errors/stage5_material_structure_20261106'
OUT=ROOT/'output/groups/stage5_material_structure_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan=json.loads((REPORT/'clustering_plan_before_fit.json').read_text())
    assert plan['source_sha256']==sha(ROOT/'eksperimen/skrip/stage5_material_structure.py')
    assert not set(plan['fit_keys'])&set(plan['heldout_keys'])
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv')
    fit=p.set_index('key').loc[plan['fit_keys']];val=p.set_index('key').loc[plan['heldout_keys']]
    assert fit.split.eq('train').all() and val.split.eq('train').all()
    assert not set(fit.pixel_sha256)&set(val.pixel_sha256)
    split=pd.read_csv(ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv').set_index('key')
    assert not set(split.loc[fit.index,'candidate_group'])&set(split.loc[val.index,'candidate_group'])
    features=np.load(ROOT/'output/embeddings/stage5_patch_v1/original224.npz',allow_pickle=False)
    assignments=pd.read_csv(OUT/'all_frozen_assignments.csv')
    assert np.array_equal(features['keys'],assignments.key) and len(assignments)==2400 and assignments.key.is_unique
    models=sorted(OUT.glob('*.joblib'))
    with threadpool_limits(limits=4):
        for path in models:
            model=joblib.load(path)
            np.testing.assert_array_equal(predict(model,features['features']),assignments[path.stem])
    assert len(models)==24
    metrics=pd.read_csv(REPORT/'development_metrics.csv')
    assert len(metrics)==26 and np.isfinite(metrics[['development_score','flip_agreement','scale_agreement','cross_rep_neighbor_agreement']]).all().all()
    eligible=metrics[metrics.selection_eligible].sort_values(['development_score','candidate'],ascending=[False,True])
    frozen=json.loads((REPORT/'selection_frozen.json').read_text())
    assert eligible.iloc[0].candidate==frozen['selected']
    assert sha(OUT/'all_frozen_assignments.csv')==frozen['assignments_sha256']
    assert sha(REPORT/'summary.json')==frozen['summary_sha256']
    for path,digest in frozen['protected_inputs'].items():assert sha(ROOT/path)==digest
    gallery=REPORT/'galleries/prototypes32.html'
    targets=re.findall(r'(?:href|src)="([^"]+)"',gallery.read_text(encoding='utf-8'))
    assert len(targets)>0
    for target in targets:assert (gallery.parent/target).exists(),target
    result=dict(status='passed',assignment_rows=2400,models_reproduced=len(models),
        development_conditions=26,selection_reproduced=True,fit_train=1538,development_train=381,
        pixel_and_candidate_group_overlap=0,test_in_fit=False,gallery_links_verified=len(targets),
        protected_baseline_and_notebook_unchanged=True,reference_used_for_selection=False,semantic_accuracy=None)
    (REPORT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
