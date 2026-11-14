"""Audit current eligible folders against the participant comparison reference.

The 9 Nov experiment also used the participant-supplied comparison classes
after source-group cross-fitted predictions to accept or hold residual rows.
They were not model-fit targets. Organizer permission is participant-reported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'docs/agent-notes'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    manifest=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=manifest[manifest.split.eq('train')&~manifest.is_crop&manifest.training_eligible].sort_values('parent_id').copy()
    annotation={}
    for folder in (ROOT/'data/Dataset DSC GelarRasa 2026/train/label').iterdir():
        if not folder.is_dir():continue
        for image in folder.iterdir():
            if image.is_file() and image.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                assert image.stem not in annotation
                annotation[image.stem]=folder.name
    assert len(annotation)==1920 and len(frame)==1869
    revision=pd.read_csv(NOTES/'table-csv/team_review_revision.csv',dtype={'parent_id':str}).fillna('')
    confirmed=revision[revision.confirmed_for_audit.eq(True)]
    explicit=dict(zip(confirmed.parent_id,confirmed.proposed_class))
    comparison=frame[['parent_id','parent_key','source_path','output_path','assigned_class','label_source']].copy()
    comparison['archive_visual_class']=comparison.parent_id.map(annotation)
    comparison['explicit_participant_review']=comparison.parent_id.map(explicit).fillna('')
    comparison['requested_audit_reference']=comparison.parent_id.map({**annotation,**explicit})
    comparison['participant_exception']=comparison.requested_audit_reference.ne(comparison.archive_visual_class)
    comparison['differs_from_requested_reference']=comparison.assigned_class.ne(comparison.requested_audit_reference)
    comparison['use']='posthoc_confirmation_after_group_oof_reference_filter'
    output=NOTES/'table-csv/current_reference_with_exceptions.csv'
    comparison.to_csv(output,index=False)
    remaining=comparison[comparison.differs_from_requested_reference]
    remaining.to_csv(NOTES/'table-csv/remaining_reference_with_exceptions.csv',index=False)
    exceptions=comparison[comparison.participant_exception]
    report={'status':'audit_completed_after_alignment_filter','eligible_full_frames':len(comparison),
            'raw_archive_disagreements':int(comparison.assigned_class.ne(comparison.archive_visual_class).sum()),
            'explicit_participant_exceptions':len(exceptions),
            'exceptions_already_preserved':bool(exceptions.assigned_class.eq(exceptions.requested_audit_reference).all()),
            'remaining_after_exceptions':len(remaining),
            'active_confirmed_review_sources':int(comparison.explicit_participant_review.ne('').sum()),
            'all_active_confirmed_reviews_preserved':bool(comparison.loc[comparison.explicit_participant_review.ne(''),'assigned_class'].eq(comparison.loc[comparison.explicit_participant_review.ne(''),'explicit_participant_review']).all()),
            'exception_ids':exceptions.parent_id.tolist(),'active_outputs_modified':False,
            'comparison_sha256':sha(output),'reference_is_official_ground_truth':False}
    (NOTES/'json/current_reference_with_exceptions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert all(sha(ROOT/name)==value for name,value in protected.items())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
