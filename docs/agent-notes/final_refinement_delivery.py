"""Verify the published bounded refinement and build a compact follow-up gallery."""
import html
import json
import os
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    published=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text());assert published['status']=='passed'
    frozen=json.loads((NOTES/'json/bounded_final_frozen.json').read_text())
    expected=pd.read_csv(NOTES/'table-csv/bounded_final_selected.csv',dtype={'image_id':str}).set_index('key')
    assert sha(NOTES/'table-csv/bounded_final_selected.csv')==frozen['prediction_sha256']
    manifest_path=ROOT/'data/processed/manifest.csv';assert sha(manifest_path)==published['manifest_sha256']
    m=pd.read_csv(manifest_path,dtype={'parent_id':str});full=m[m.split.eq('train')&~m.is_crop].set_index('parent_key').loc[expected.index]
    assert len(full)==1906 and full.assigned_class.equals(expected.predicted_class.rename('assigned_class'))
    assert full.source_sha256.equals(expected.source_sha256)
    manual=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(manual)==14 and manual.assigned_class.value_counts().le(5).all()
    held=m[m.split.eq('train')&~m.training_eligible]
    assert set(held.parent_id)=={'1662','1885'} and not held.human_reviewed.any()
    provenance=json.loads((ROOT/'data/processed/provenance.json').read_text())
    forbidden={'train:1662','train:1885'}
    assert not any(forbidden.intersection(row['neighbor_keys']) for row in provenance['summary']['crop_assignment_rows'])
    first=json.loads((NOTES/'json/program_refinement_publication.json').read_text())
    with np.load(ROOT/first['backup']/'preprocessing_inputs.npz',allow_pickle=False) as original, np.load(ROOT/'data/preprocessing_inputs.npz',allow_pickle=False) as current:
        mask=np.char.startswith(original['keys'].astype(str),'test:')
        for key in original.files:
            value=original[key]
            if value.ndim==1 and len(value)==2400:assert np.array_equal(value[mask],current[key][mask]),key
    review=pd.read_csv(NOTES/'table-csv/bounded_review_outcome.csv',dtype={'parent_id':str})
    queue=pd.read_csv(NOTES/'table-csv/remaining_team_review_differences.csv',dtype={'parent_id':str})
    assert len(queue)==59 and queue.parent_id.is_unique
    program_correct=review[review.team_preference.eq('program')]
    assert len(program_correct)==9 and program_correct.predicted_class.eq(program_correct.assigned_class).all()
    home=NOTES/'remaining_visual_review'
    def image_path(value):return html.escape(os.path.relpath(ROOT/value,home).replace('\\','/'),quote=True)
    nearest=pd.read_csv(NOTES/'table-csv/remaining_train_test_top3.csv',dtype={'train_id':str,'test_id':str})
    parts=['<!doctype html><html lang="id"><meta charset="utf-8"><title>59 kasus tindak lanjut</title>',
           '<style>body{font:16px system-ui;margin:24px}section{padding:16px;border:1px solid #ddd;margin:18px 0}figure{display:inline-block;vertical-align:top;width:30%;margin:1%}img{width:100%;height:260px;object-fit:contain}figcaption{white-space:pre-line}</style>',
           '<h1>59 kasus yang masih berbeda dari pilihan review tim</h1>',
           '<p>Snapshot setelah refinement terakhir. Ini audit privat, bukan Accuracy resmi. Sebagian pilihan tim berasal dari interpretasi daftar; tidak semuanya dipakai sebagai target pelatihan. Test tidak diberi kategori. Acuan manusia aktif hanya 14 sumber.</p>']
    for row in queue.sort_values('parent_id').itertuples():
        parts.append(f'<section><h2>Train {row.parent_id}</h2><p>Kategori aktif: <b>{row.predicted_class}</b> | Pilihan review: <b>{row.proposed_class}</b><br>Asal kategori aktif: {row.category_source}<br>Catatan penafsiran review: {html.escape(row.interpretation)}</p>')
        parts.append(f'<figure><a href="{image_path(row.source_path)}"><img src="{image_path(row.source_path)}" loading="lazy"></a><figcaption>Train {row.parent_id}</figcaption></figure>')
        for near in nearest[nearest.train_id.eq(row.parent_id)].sort_values('rank').head(2).itertuples():
            parts.append(f'<figure><a href="{image_path(near.test_path)}"><img src="{image_path(near.test_path)}" loading="lazy"></a><figcaption>Test {near.test_id}; cosine {near.cosine:.3f}\nKemiripan bukan bukti kategori.</figcaption></figure>')
        parts.append('</section>')
    (home/'latest_review.html').write_bytes(('\n'.join(parts)+'</html>').encode('utf-8'))
    index=home/'index.html';text=index.read_text(encoding='utf-8')
    marker='<p><strong>SNAPSHOT HISTORIS sebelum perbaikan terbaru.</strong> <a href="latest_review.html">Buka 59 kasus tindak lanjut dengan kategori aktif terbaru.</a></p>'
    if marker not in text:index.write_bytes((marker+text).encode('utf-8'))
    report={'status':'passed','full_frame_train_assignments_verified':1906,'changed_categories_vs_start_of_current_task':42,
            'participant_reviewed_source_images':14,'participant_category_counts':manual.assigned_class.value_counts().to_dict(),
            'remaining_human_capacity':{c:5-int(manual.assigned_class.eq(c).sum()) for c in CLASSES},
            'training_eligible_train_samples':int((m.split.eq('train')&m.training_eligible).sum()),'stored_train_samples':int(m.split.eq('train').sum()),
            'deferred_sources':['1662','1885'],'deferred_not_crop_references':True,'test_snapshot_unchanged':True,
            'original_program_preferences_preserved':9,'remaining_differences_from_stated_team_preferences':59,
            'not_reviewed_original_cases':20,'tentative_original_cases':20,'human_holdout_matches':5,'human_holdout_rows':14,
            'human_holdout_not_official_accuracy':True,'notebook_sha256':published['notebook_sha256'],
            'notebook_execution':json.loads((NOTES/'execution.json').read_text()),
            'scope':'published preprocessing and evidence integrity, not a guarantee of correct category for every image'}
    dump(NOTES/'json/final_refinement_delivery.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='notebook_execution'},indent=2))


if __name__=='__main__':main()
