"""Post-publication audit only: never fit or select predictions from full reviews."""
import html
import json
import os
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def main():
    publication = json.loads((NOTES / 'json/bounded_anchor_publication.json').read_text())
    assert publication['status'] == 'passed' and publication['source_frozen_prefix'] == 'review_final'
    frozen = json.loads((NOTES / 'json/review_final_frozen.json').read_text())
    selected_path = NOTES / 'table-csv/review_final_selected.csv'
    assert sha(selected_path) == frozen['prediction_sha256'] == publication['candidate_sha256']
    expected = pd.read_csv(selected_path, dtype={'image_id': str}).set_index('key')
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    complete = manifest[manifest.split.eq('train') & ~manifest.is_crop].set_index('parent_key').loc[expected.index]
    assert len(complete) == 1906 and complete.assigned_class.equals(expected.predicted_class.rename('assigned_class'))
    assert complete.label_source.equals(expected.category_source.rename('label_source'))
    assert complete.source_sha256.equals(expected.source_sha256)
    assert sha(ROOT / 'data/processed/manifest.csv') == publication['manifest_sha256']
    human = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    assert len(human) == 34 and human.assigned_class.value_counts().le(5).all()
    assert human.key.is_unique and 'train:1215' in set(human.key)
    assert complete.at['train:1215', 'assigned_class'] == 'cardboard'
    held = manifest[manifest.split.eq('train') & ~manifest.training_eligible]
    assert set(held.parent_id) == {'1662', '1885'} and not held.human_reviewed.any()
    provenance = json.loads((ROOT / 'data/processed/provenance.json').read_text())
    assert not any({'train:1662', 'train:1885'}.intersection(r['neighbor_keys']) for r in provenance['summary']['crop_assignment_rows'])
    backup = ROOT / publication['backup']
    previous = pd.read_csv(backup / 'processed/manifest.csv', dtype={'parent_id': str})
    old_test = previous[previous.split.eq('test')].set_index('sample_key').sort_index()
    new_test = manifest[manifest.split.eq('test')].set_index('sample_key').sort_index()
    assert old_test.source_sha256.equals(new_test.source_sha256)
    assert old_test.output_sha256.equals(new_test.output_sha256)
    with np.load(backup / 'preprocessing_inputs.npz', allow_pickle=False) as before, np.load(ROOT / 'data/preprocessing_inputs.npz', allow_pickle=False) as after:
        test = np.char.startswith(before['keys'].astype(str), 'test:')
        for name in before.files:
            if before[name].ndim == 1 and len(before[name]) == 2400:
                assert np.array_equal(before[name][test], after[name][test]), name
    review = pd.read_csv(NOTES / 'table-csv/team_review_revision.csv', dtype={'parent_id': str}).fillna('')
    review['current_class'] = review.parent_id.map(complete.set_index('parent_id').assigned_class)
    review['current_source'] = review.parent_id.map(complete.set_index('parent_id').label_source)
    review['current_matches_review'] = review.confirmed_for_audit & review.current_class.eq(review.proposed_class)
    review['current_human_anchor'] = review.parent_id.map(lambda k: 'train:' + k in set(human.key))
    confirmed = review[review.confirmed_for_audit]
    queue = confirmed[~confirmed.current_matches_review].sort_values('parent_id')
    assert len(confirmed) == 126 and len(queue) == 48
    assert not queue.current_human_anchor.any()
    correct = {'0260', '0267', '0352', '0396', '0429', '0512', '0531', '0536'}
    assert confirmed[confirmed.parent_id.isin(correct)].current_matches_review.all()
    previously_matched = confirmed.predicted_class.eq(confirmed.proposed_class)
    assert confirmed.loc[previously_matched, 'current_matches_review'].all()
    review.to_csv(NOTES / 'table-csv/current_review_outcome.csv', index=False)
    queue.to_csv(NOTES / 'table-csv/remaining_current_review_differences.csv', index=False)
    execution = json.loads((NOTES / 'execution.json').read_text())
    validation = json.loads((NOTES / 'skills_validation.json').read_text())
    assert execution['status'] == validation['status'] == 'passed'
    assert execution['sha256'] == sha(ROOT / 'final-submit/DSC2026_Notebook.ipynb')
    assert validation['image_hashes_verified'] == 4793 and validation['manual_train_seed_images'] == 34
    full_changes = expected.predicted_class.ne(expected.baseline_class)
    manual_changes = full_changes & expected.human_reviewed
    proof = {'status': 'passed', 'scope': 'EDA/preprocessing and private participant-review diagnostic, not official semantic Accuracy or competition certification',
             'changes_this_revision': int(full_changes.sum()), 'manual_anchor_category_changes_this_revision': int(manual_changes.sum()),
             'algorithmic_category_changes_this_revision': int((full_changes & ~expected.human_reviewed).sum()),
             'cumulative_changes_vs_original_150_snapshot': frozen['cumulative_changes_vs_initial_program'],
             'human_unique_sources_used': 34, 'human_counts': human.assigned_class.value_counts().to_dict(),
             'remaining_manual_capacity_by_class': {c: 5 - int(human.assigned_class.eq(c).sum()) for c in CLASSES},
             'confirmed_review_cases': len(confirmed), 'matched_confirmed_review': int(confirmed.current_matches_review.sum()),
             'remaining_review_differences': len(queue),
             'before_revision_differences_on_same_confirmed_set': int(confirmed.predicted_class.ne(confirmed.proposed_class).sum()),
             'program_correct_eight_preserved': True, 'previous_review_matches_regressed': 0,
             'tentative_review_cases': int((~review.confirmed_for_audit & review.triage.eq('program_tentative')).sum()),
             'conditional_review_cases': int(review.triage.eq('conditional_review_or_exclusion').sum()),
             'held_train_ids': ['1662', '1885'], 'train_samples_stored': 1913, 'train_samples_eligible': 1911,
             'test_snapshot_and_processed_pixels_unchanged': True,
             'publication_backup': publication['backup'], 'source_prediction_sha256': sha(selected_path),
             'notebook_sha256': execution['sha256'], 'manifest_sha256': publication['manifest_sha256'],
             'code_cells_executed': execution['executed'], 'embedded_figures': execution['embedded_figures'],
             'image_hashes_verified': validation['image_hashes_verified'],
             'remaining_review_used_for_fit_or_model_selection': False,
             'rejected_probe': frozen['rejected_probe'], 'final_classifier_or_submission_created': False,
             'preprocessing_run_refits_assignment_heads': False}
    dump(NOTES / 'json/review_revision_delivery.json', proof)
    home = NOTES / 'remaining_visual_review'
    nearest = pd.read_csv(NOTES / 'table-csv/remaining_train_test_top3.csv', dtype={'train_id': str, 'test_id': str})
    def url(path):
        return html.escape(os.path.relpath(ROOT / path, home).replace('\\', '/'), quote=True)
    parts = ['<!doctype html><html lang="id"><meta charset="utf-8"><title>48 kasus review tersisa</title>',
             '<style>body{font:16px system-ui;margin:24px}section{border:1px solid #ddd;padding:16px;margin:18px 0}figure{display:inline-block;vertical-align:top;width:30%;margin:1%}img{width:100%;height:260px;object-fit:contain}</style>',
             '<h1>48 gambar masih berbeda dari review tim</h1><p>Audit setelah prediksi dibekukan. Review manusia terkonfirmasi:126; cocok78, berbeda48. Tidak semua review dijadikan target: hanya34acuan dalam batas5perkelas. Gambar test ditampilkan tanpa kategori dan tidak masuk fit.20kasus tentatif,2bersyarat dan2ditahan terpisah.</p>']
    for row in queue.itertuples():
        parts.append(f'<section><h2>Train {row.parent_id}</h2><p>Kategori aktif: <b>{row.current_class}</b> | Keputusan review tim: <b>{row.proposed_class}</b><br>Asal aktif: {row.current_source}</p>')
        parts.append(f'<figure><img src="{url(row.source_path)}"><figcaption>Train {row.parent_id}</figcaption></figure>')
        for neighbor in nearest[nearest.train_id.eq(row.parent_id)].sort_values('rank').head(2).itertuples():
            parts.append(f'<figure><img src="{url(neighbor.test_path)}"><figcaption>Test {neighbor.test_id}; cosine {neighbor.cosine:.3f}; bukan bukti label</figcaption></figure>')
        parts.append('</section>')
    (home / 'latest_review.html').write_text('\n'.join(parts) + '</html>', encoding='utf-8')
    report = f'''# Hasil revisi review tim — 7 November 2026

Status aktif terbaru: **26 kategori berubah dari versi sebelum revisi ini**:16perubahan pada tambahan acuan manusia dan10perubahan algoritmis (9konsensus classifier/graph dan1cluster). Total sejak snapshot150lama:68kategori berubah. Semua perubahan dapat dilacak pada [tabel prediksi beku](table-csv/review_final_selected.csv).

## Keputusan pengguna dan batas penerapan

Tiga lembar awal dalam jawaban sebelumnya dipahami sebagai59kasus perbedaan: pengguna mengonfirmasi pilihan reviewnya. Untuk20gambar yang sebelumnya belum ditinjau,0260/0267/0352/0396/0429/0512/0531/0536mengikuti kategori program;12lainnya mengikuti kategori arsip yang dikonfirmasi pengguna.1215ditetapkan **cardboard**, dicatat sebagai acuan manusia.1662/1885tetap ditahan, bukan acuan;20kasus tentatif dan0676/1650bersyarat belum diubah menjadi label manusia.

Total keputusan tegas untuk audit kini126gambar. Hanya34sumber menjadi target manusia aktif, bukan seluruh126. Enam kelas sudah memakai5acuan;metal3danfoodwaste1. Sisa kapasitas6hanya pada metal2danfoodwaste4. Jatah tidak direset oleh pergantian eksperimen, fold, augmentasi atau rollback. Asal review manusia tidak disamarkan sebagai keluaran algoritma.

## Eksperimen

Lima keluarga fitur visual non-VLM, source-group crossfit, tiga graph fit-train dan dua pembobotan prior mempertahankan9perubahan otomatis yang stabil. Clustering32/64/128pada tiga keluarga, minimal dua acuan sekelas tanpa konflik per cluster, lalu konfirmasi tiga graph dan classifier, menambahkan1perubahan:0562menjadi paper.1904foto train eligible masuk fit;1662/1885dan semua test dikecualikan.

Probe tambahan yang hanya memakai34acuan tanpa prior menghasilkan51usulan. Probe **ditolak secara keseluruhan**: diagnostik leave-source-group-out hanya2/34kategori pulih, dibanding14/34pada model dengan prior.51usulan tidak dipublikasikan dan tidak diseleksi satu-satu dari kecocokan review. Diagnostik acuan sengaja sulit dan kecil ini bukan Accuracy dataset, dan target prior memiliki riwayat fit sendiri.

## Audit setelah pembekuan

Pada kumpulan konfirmasi yang sama: sebelum revisi54/126cocok (72berbeda); sekarang78/126cocok (**48masih berbeda**). Tidak ada keputusan review yang sebelumnya cocok menjadi berbeda. Delapan nomor yang dinyatakan programnya benar tetap cocok. Ini kesesuaian terhadap review tim, bukan ground truth panitia. [48kasus tersisa](table-csv/remaining_current_review_differences.csv) dan [galeri aktif](remaining_visual_review/latest_review.html) tidak menjadi target/filter fit atau seleksi prediksi.

Saya belum menemukan metode yang membenarkan seluruh48kasus secara andal tanpa melampaui batas target manusia yang dipakai. Banyak perbedaan masih berhubungan dengan pembungkus plastik–kertas, glass–material lain, serta cardboard–paper. Menambahkan cluster saja tidak memberi informasi material baru. Metode representasi yang diadaptasi lebih lanjut masih mungkin diuji; hasil saat ini tidak membuktikan semua algoritma mustahil.

## Notebook dan data

Notebook tunggal [DSC2026_Notebook.ipynb](../../final-submit/DSC2026_Notebook.ipynb) lulus Run All dari processed kosong: {execution['executed']}sel kode,0error,{execution['embedded_figures']}visual.1913sampel train tersimpan langsung pada8folder;1911eligible,480test,7crop asli.1215ada pada cardboard. Semua4793hash gambar sumber/turunan diperiksa. Data asli dan piksel test tidak berubah. Semua isi processed dibuat notebook.

Notebook mematerialisasikan snapshot penyempurnaan yang dibekukan dan menilai ulang crop; tidak mengulangi fit seluruh assignment heads. Classifier final, validasi independen dan CSV belum dibuat. Data operasional tetap diperlukan agar panitia dapat mereproduksi, dan paket final belum dinyatakan siap dikumpulkan. Notebook tidak membaca/menyebut folder administrasi atau layanan model bahasa. Catatan sumber, eksperimen dan perbandingan tetap di direktori ini.

Bukti utama: [validasi revisi](json/review_revision_delivery.json), [publikasi](json/bounded_anchor_publication.json), [Run All](execution.json), [validasi koleksi](skills_validation.json), [ledger manusia](json/manual_label_budget_status.json).
'''
    (NOTES / 'REVIEW_REVISION_RESULT.md').write_text(report, encoding='utf-8')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
