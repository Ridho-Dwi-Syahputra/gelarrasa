"""Private structural comparison ONLY after stage5 assignment/selection freeze.

Never computes a category mapping or feeds information back into clustering.
K=32 is not directly comparable to K=8 Accuracy; report completeness and
pairwise recall as well as homogeneity/precision to reveal fragmentation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/errors/stage5_material_structure_20261106'
OUT=ROOT/'output/groups/stage5_material_structure_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(reference,prediction):
    table=pd.crosstab(reference,prediction).to_numpy(dtype='int64')
    choose2=lambda x:(x*(x-1)//2).sum()
    both=choose2(table)
    predicted_pairs=choose2(table.sum(0));reference_pairs=choose2(table.sum(1))
    h,c,v=homogeneity_completeness_v_measure(reference,prediction)
    return dict(groups=int(len(set(prediction))),ari=float(adjusted_rand_score(reference,prediction)),
        nmi=float(normalized_mutual_info_score(reference,prediction)),homogeneity=float(h),
        completeness=float(c),v_measure=float(v),
        pairwise_precision=float(both/predicted_pairs),pairwise_recall=float(both/reference_pairs))


def main():
    frozen=json.loads((REPORT/'selection_frozen.json').read_text())
    assert frozen['status']=='frozen_before_reference_audit'
    assignments=OUT/'all_frozen_assignments.csv'
    assert sha(assignments)==frozen['assignments_sha256']
    assert sha(REPORT/'summary.json')==frozen['summary_sha256']
    assert sha(ROOT/'eksperimen/skrip/stage5_material_structure.py')==frozen['source_sha256']
    for path,digest in frozen['protected_inputs'].items():assert sha(ROOT/path)==digest
    df=pd.read_csv(assignments,dtype={'image_id':str})
    p=df[df.split.eq('train')].copy();assert len(p)==1920 and p.image_id.is_unique
    # All training/selection and file integrity checks precede this reference read.
    classes={'glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste'}
    references=[]
    for folder in (ROOT/'data/Dataset DSC GelarRasa 2026/train/label').iterdir():
        if folder.is_dir() and folder.name in classes:
            for image in folder.iterdir():
                if image.is_file() and image.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                    references.append(dict(image_id=image.stem,reference=folder.name))
    ref=pd.DataFrame(references);assert len(ref)==1920 and ref.image_id.is_unique
    p=p.merge(ref,on='image_id',validate='one_to_one')
    # Compare a fixed subset decided before any archive read. No "best purity" search.
    names=list(dict.fromkeys(['current_student',frozen['selected'],'global_k8_s11','global_k32_s11','hybrid_k8_s11','hybrid_k32_s11','hybrid_k32_ward8_s11']))
    rows=[dict(candidate=name,**metrics(p.reference,p[name])) for name in names]
    results=pd.DataFrame(rows)
    results.to_csv(REPORT/'private_reference_partition_metrics.csv',index=False)
    local=ROOT/'data/processed/stage5_private_reference_audit_only';local.mkdir(parents=True,exist_ok=True)
    p[['key','image_id','reference',*names]].to_csv(local/'posthoc_partitions.csv',index=False)
    for path,digest in frozen['protected_inputs'].items():assert sha(ROOT/path)==digest
    assert sha(assignments)==frozen['assignments_sha256'] and sha(REPORT/'summary.json')==frozen['summary_sha256']
    result=dict(status='passed',train_rows=len(p),selected_before_reference=frozen['selected'],
        reference_is_ground_truth=False,semantic_mapping_created=False,reference_used_for_fitting_or_selection=False,
        protected_artifacts_unchanged=True,semantic_accuracy=None,
        note='Post-hoc structural agreement only. Fine partitions can improve homogeneity by splitting classes; completeness/recall quantify that cost.',
        metrics=rows)
    (REPORT/'private_reference_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(results.to_string(index=False));print('Frozen audit passed',flush=True)


if __name__=='__main__':main()
