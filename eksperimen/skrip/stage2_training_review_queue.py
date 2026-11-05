"""Create structural review queue first; then isolate a retrospective archive join.

No fitting, semantic mapping, label corrections, or artifact promotion here.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reports/errors/stage2_20261105'


def align(base, other, fit):
    counts=np.zeros((8,8),int)
    np.add.at(counts,(base[fit],other[fit]),1)
    a,b=linear_sum_assignment(-counts)
    mapping=np.empty(8,int);mapping[b]=a
    return mapping[other]


def main():
    df=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    head_path=ROOT/'output/groups/stage2_neighbor_v1/assignments.csv'
    digest=hashlib.sha256(head_path.read_bytes()).hexdigest()
    head=pd.read_csv(head_path,dtype={'image_id':str})
    pca=pd.read_csv(ROOT/'output/groups/stage2_repr_v1/dinov2/pca128_assignments.csv')
    old=pd.read_csv(ROOT/'output/groups/stage1_v1/cluster_assignments.csv')
    for frame in [head,pca,old]:
        assert np.array_equal(frame.key,df.key)
    z=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz')
    assert np.array_equal(z['keys'],df.key)
    x=z['features'];fit=df[df.split.eq('train')].drop_duplicates('pixel_sha256').index.to_numpy()
    hashes=df.pixel_sha256.to_numpy()
    s=x@x[fit].T
    s[hashes[:,None]==hashes[fit][None,:]]=-np.inf
    neighbors=fit[np.argsort(-s,axis=1,kind='stable')[:,:10]]
    labels=head.cluster_index.to_numpy()
    pca_aligned=align(labels,pca.cluster_index.to_numpy(),fit)
    old_aligned=align(labels,old.cluster_index.to_numpy(),fit)
    queue=df[['key','split','image_id','preview_path','min_side','padding_fraction','pixel_sha256']].copy()
    queue['new_cluster']=labels;queue['old_cluster']=old.cluster_index
    queue['changed_from_old_aligned']=labels!=old_aligned
    queue['alternative_algorithm_disagrees']=labels!=pca_aligned
    queue['head_probability_uncalibrated']=head.uncalibrated_head_probability
    queue['neighbor_cohesion_dino']=(labels[:,None]==labels[neighbors]).mean(1)
    queue['neighbor_cohesion_old_partition_in_dino']=(old.cluster_index.to_numpy()[:,None]==old.cluster_index.to_numpy()[neighbors]).mean(1)
    views=pd.read_csv(REP/'view_robustness_per_image.csv')
    chosen=views[views.method.eq('dinov2_neighbor_refinement')].groupby('key').unchanged.agg(['all','count'])
    queue['view_probe_tested']=queue.key.isin(chosen.index)
    queue['view_changed']=queue.key.map(~chosen['all']).eq(True)
    queue['review_score']=(2*queue.alternative_algorithm_disagrees.astype(int)
                           +2*queue.neighbor_cohesion_dino.lt(.7).astype(int)
                           +queue.head_probability_uncalibrated.lt(.8).astype(int)
                           +queue.view_changed.astype(int))
    queue['review_needed']=queue.review_score.gt(0)
    queue.to_csv(REP/'algorithmic_review_queue_all.csv',index=False)
    # Below this line archives are a post-hoc slice only. No archive semantic column
    # is copied into the active queue or folder manifest, and nothing is fitted.
    archive=pd.read_csv(REP/'diagnostics/posthoc_image_diagnostics.csv',usecols=['key','disagrees_archive'])
    posthoc=queue.merge(archive,on='key',validate='one_to_one')
    posthoc=posthoc[posthoc.disagrees_archive]
    posthoc.to_csv(REP/'diagnostics/old_disagreements_structural_followup.csv',index=False)
    summary=[]
    for split,g in posthoc.groupby('split'):
        summary.append(dict(split=split,old_disagreements=len(g),
                            reassigned_after_cluster_alignment=int(g.changed_from_old_aligned.sum()),
                            still_needs_algorithmic_review=int(g.review_needed.sum()),
                            algorithm_disagrees=int(g.alternative_algorithm_disagrees.sum()),
                            low_cohesion=int(g.neighbor_cohesion_dino.lt(.7).sum()),
                            view_probe_tested=int(g.view_probe_tested.sum()),view_changed=int(g.view_changed.sum()),
                            note='Reassignment is NOT evidence of correction. No archive label used for fitting or selection.'))
    (REP/'diagnostics/structural_followup_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Train-only technical sample, ranked by structural uncertainty. No class names.
    sample=posthoc[posthoc.split.eq('train')].sort_values(['review_score','neighbor_cohesion_dino','key'],ascending=[False,True,True]).head(32)
    sample[['key','new_cluster','old_cluster','review_score']].to_csv(REP/'diagnostics/train_followup_sample.csv',index=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for page in range(2):
        records=sample.iloc[page*16:(page+1)*16]
        sheet=Image.new('RGB',(1200,1460),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'Train structural review {page+1}; technical inspection only',fill='black',font=font)
        for pos,r in enumerate(records.itertuples()):
            col=pos%4;row=pos//4;x0=col*300+5;y0=row*350+45
            with Image.open(ROOT/r.preview_path) as im:
                tile=ImageOps.contain(im.convert('RGB'),(285,285))
            sheet.paste(tile,(x0+(285-tile.width)//2,y0))
            draw.text((x0,y0+287),f'{r.key} old:{r.old_cluster:02d} new:{r.new_cluster:02d}',fill='black',font=font)
            draw.text((x0,y0+309),f'review:{r.review_score} neighbors:{r.neighbor_cohesion_dino:.1f}',fill='black',font=font)
        sheet.save(REP/f'diagnostics/train_followup_{page+1}.png')
    assert hashlib.sha256(head_path.read_bytes()).hexdigest()==digest
    print(json.dumps(summary,indent=2))
    print(queue.groupby('split').agg(images=('key','size'),review=('review_needed','sum')).to_string())


if __name__=='__main__':
    main()
