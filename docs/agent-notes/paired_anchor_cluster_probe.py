"""Two-feature, two-human-anchor train-only cluster propagation probe."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "docs/agent-notes"


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv", dtype={"parent_id": str})
    frame = manifest.loc[manifest.split.eq("train") & ~manifest.is_crop & manifest.training_eligible].sort_values("parent_id").reset_index(drop=True)
    assert len(frame) == 1887
    sources = [ROOT / "data/processed/dinov2_small_embeddings.npz",
               NOTES / ".runtime/dinov2_object_zoom_features.npz"]
    runs = []
    with threadpool_limits(limits=4):
        for source in sources:
            z = np.load(source, allow_pickle=False)
            positions = {key:i for i,key in enumerate(z["keys"])}
            x = normalize(z["features"][[positions[key] for key in frame.parent_key]].astype("float64"))
            runs.append([KMeans(n_clusters=32,n_init=5,random_state=seed).fit_predict(x)
                         for seed in range(20261109,20261114)])
    humans = pd.read_csv(ROOT / "data/processed/manual_seeds.csv")
    assert len(humans) == 35
    human = dict(zip(humans.key,humans.assigned_class))
    positions = {key:i for i,key in enumerate(frame.parent_key)}
    proposed = {}
    eligible_pairs = []
    for category,group in humans.groupby("assigned_class"):
        for anchor_a,anchor_b in itertools.combinations(group.key,2):
            a,b=positions[anchor_a],positions[anchor_b]
            masks=[]
            for bank in runs:
                together=sum(labels[a]==labels[b] for labels in bank)
                if together<4:break
                masks.append(sum((labels==labels[a]) & (labels==labels[b]) for labels in bank)>=4)
            if len(masks)!=2:continue
            stable=masks[0]&masks[1]
            if any(human[key]!=category for key in frame.parent_key[stable] if key in human):continue
            eligible_pairs.append((category,anchor_a,anchor_b,int(stable.sum())))
            for index in np.flatnonzero(stable):proposed.setdefault(index,set()).add(category)
    choice={index:next(iter(categories)) for index,categories in proposed.items() if len(categories)==1}
    result=frame[["parent_key","parent_id","assigned_class"]].copy()
    result["proposed_class"]=[choice.get(i,cls) for i,cls in enumerate(result.assigned_class)]
    result["changed"]=result.proposed_class.ne(result.assigned_class)
    output=NOTES/"table-csv/paired_anchor_cluster_candidate.csv"
    result.to_csv(output,index=False)
    report={"status":"frozen_candidate_not_published","train_rows":len(frame),"test_rows_in_fit":0,
            "archive_labels_read_by_experiment":False,"human_anchors":len(human),"feature_banks":len(runs),
            "eligible_pairs":eligible_pairs,"represented_rows":len(choice),"proposed_changes":int(result.changed.sum()),
            "changes_by_pair":{" -> ".join(p):int(n) for p,n in result.loc[result.changed].groupby(["assigned_class","proposed_class"]).size().items()},
            "candidate_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"active_processed_modified":False}
    (NOTES/"json/paired_anchor_cluster_candidate.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="eligible_pairs"}|{"eligible_pair_count":len(eligible_pairs)},indent=2))


if __name__=="__main__":main()
