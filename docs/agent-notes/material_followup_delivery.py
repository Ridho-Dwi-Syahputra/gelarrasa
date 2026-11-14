"""Verify active material follow-up, then update the separate private review audit."""
import html
import json
import os
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def main():
    pub = json.loads((NOTES / 'json/bounded_anchor_publication.json').read_text())
    assert pub['status'] == 'passed' and pub['source_frozen_prefix'] == 'robust_geometry_heads'
    frozen = json.loads((NOTES / 'json/robust_geometry_heads_frozen.json').read_text())
    candidate_path = NOTES / 'table-csv/robust_geometry_heads_selected.csv'
    assert sha(candidate_path) == pub['candidate_sha256'] == frozen['prediction_sha256']
    decision = json.loads((NOTES / 'json/material_publication_decision.json').read_text())
    assert decision['prediction_sha256'] == frozen['prediction_sha256']
    assert not decision['full_review_or_remaining48_read_for_this_decision']
    expected = pd.read_csv(candidate_path, dtype={'image_id': str}).set_index('key')
    manifest = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    full = manifest[manifest.split.eq('train') & ~manifest.is_crop].set_index('parent_key').loc[expected.index]
    assert len(full) == 1906 and full.assigned_class.equals(expected.predicted_class.rename('assigned_class'))
    assert full.label_source.equals(expected.category_source.rename('label_source'))
    assert full.source_sha256.equals(expected.source_sha256)
    assert sha(ROOT / 'data/processed/manifest.csv') == pub['manifest_sha256']
    human = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    assert len(human) == 34 and human.assigned_class.value_counts().le(5).all()
    budget = json.loads((NOTES / 'json/manual_label_budget_status.json').read_text())
    assert budget['manual_labels_consumed_by_current_experiments'] == 34
    held = manifest[manifest.split.eq('train') & ~manifest.training_eligible]
    assert set(held.parent_id) == {'1662', '1885'} and not held.human_reviewed.any()
    provenance = json.loads((ROOT / 'data/processed/provenance.json').read_text())
    assert not any({'train:1662', 'train:1885'}.intersection(r['neighbor_keys']) for r in provenance['summary']['crop_assignment_rows'])
    start = json.loads((NOTES / 'json/material_phase_start.json').read_text())
    backup = ROOT / start['initial_backup']
    assert sha(backup / 'processed/manifest.csv') == start['initial_manifest_sha256']
    previous = pd.read_csv(backup / 'processed/manifest.csv', dtype={'parent_id': str})
    before = previous[previous.split.eq('train') & ~previous.is_crop].set_index('parent_key').loc[expected.index]
    old_test = previous[previous.split.eq('test')].set_index('sample_key').sort_index()
    new_test = manifest[manifest.split.eq('test')].set_index('sample_key').sort_index()
    assert old_test.source_sha256.equals(new_test.source_sha256) and old_test.output_sha256.equals(new_test.output_sha256)
    with np.load(backup / 'preprocessing_inputs.npz', allow_pickle=False) as old, np.load(ROOT / 'data/preprocessing_inputs.npz', allow_pickle=False) as new:
        mask = np.char.startswith(old['keys'].astype(str), 'test:')
        for key in old.files:
            if old[key].ndim == 1 and len(old[key]) == 2400:
                assert np.array_equal(old[key][mask], new[key][mask]), key
        geo_metadata = json.loads(str(new['metadata'].item()))['geometry_source_groups']
    edges = pd.read_csv(NOTES / 'table-csv/local_geometry_edges.csv')
    edges = edges[edges.accepted]
    assert len(edges) == 116
    group_lookup = full.parent_group.to_dict()
    for edge in edges.itertuples():
        assert group_lookup[edge.key_a] == group_lookup[edge.key_b]
    assert manifest.groupby(['split', 'parent_group']).draft_partition.nunique().max() == 1
    review = pd.read_csv(NOTES / 'table-csv/team_review_revision.csv', dtype={'parent_id': str}).fillna('')
    current_classes = full.set_index('parent_id').assigned_class
    prior_classes = before.set_index('parent_id').assigned_class
    review['current_class'] = review.parent_id.map(current_classes)
    review['current_source'] = review.parent_id.map(full.set_index('parent_id').label_source)
    review['phase_start_class'] = review.parent_id.map(prior_classes)
    review['current_matches_review'] = review.confirmed_for_audit & review.current_class.eq(review.proposed_class)
    review['phase_start_matches_review'] = review.confirmed_for_audit & review.phase_start_class.eq(review.proposed_class)
    review['current_human_anchor'] = review.parent_id.map(lambda k: 'train:' + k in set(human.key))
    confirmed = review[review.confirmed_for_audit]
    remaining = confirmed[~confirmed.current_matches_review].sort_values('parent_id')
    recovered = confirmed[~confirmed.phase_start_matches_review & confirmed.current_matches_review]
    regressed = confirmed[confirmed.phase_start_matches_review & ~confirmed.current_matches_review]
    assert len(confirmed) == 126 and len(remaining) == 45 and len(recovered) == 3 and len(regressed) == 0
    assert set(recovered.parent_id) == {'0432', '1510', '1817'}
    assert not remaining.current_human_anchor.any()
    review.to_csv(NOTES / 'table-csv/current_review_outcome.csv', index=False)
    remaining.to_csv(NOTES / 'table-csv/remaining_current_review_differences.csv', index=False)
    code = json.loads((NOTES / 'execution.json').read_text())
    validation = json.loads((NOTES / 'skills_validation.json').read_text())
    assert code['status'] == validation['status'] == 'passed'
    assert code['sha256'] == sha(ROOT / 'final-submit/DSC2026_Notebook.ipynb') == pub['notebook_sha256']
    assert validation['image_hashes_verified'] == 4793
    proof = {'status': 'passed', 'scope': 'EDA/preprocessing plus assignment experiments,not official Accuracy or competition certification',
             'phase_start_review_differences': 48, 'remaining_review_differences': 45, 'confirmed_review_cases': 126,
             'matches_confirmed_review': 81, 'recovered_review_ids': recovered.parent_id.tolist(), 'regressed_review_cases': 0,
             'full_frame_category_changes': int(full.assigned_class.ne(before.assigned_class).sum()),
             'category_changes': expected[expected.predicted_class.ne(expected.baseline_class)][['image_id', 'baseline_class', 'predicted_class']].to_dict('records'),
             'human_sources_used': 34, 'new_human_sources_used': 0, 'human_class_counts': human.assigned_class.value_counts().to_dict(),
             'source_groups_before_eligible': frozen['geometry_source_groups']['old_groups'],
             'source_groups_after_eligible': frozen['geometry_source_groups']['new_groups'],
             'geometry_edges_kept_same_group_and_partition': len(edges), 'geometry_input_metadata': geo_metadata,
             'train_samples_stored': 1913, 'train_samples_eligible': 1911, 'test_samples': 480,
             'test_input_and_pixels_unchanged': True, 'held_train_ids': ['1662', '1885'],
             'code_cells_executed': code['executed'], 'embedded_figures': code['embedded_figures'], 'image_hashes_verified': 4793,
             'notebook_sha256': code['sha256'], 'manifest_sha256': pub['manifest_sha256'], 'input_sha256': pub['input_sha256'],
             'guard_decision_preceded_private_review_audit': True, 'remaining_review_used_as_targets_or_filter': False,
             'accepted_development_diagnostic': {'CE': 14, 'GCE': 15, 'rows': 34, 'protocol': 'five geometry-source-group folds,development only'},
             'MINC_head_rejected_diagnostic': {'matches': 11, 'rows': 34},
             'selftraining_rejected_diagnostic': {'teacher': 14, 'student': 14, 'rows': 34},
             'final_classifier_or_submission_created': False, 'notebook_refits_assignment_heads': False}
    dump(NOTES / 'json/material_followup_delivery.json', proof)
    home = NOTES / 'remaining_visual_review'
    nearest = pd.read_csv(NOTES / 'table-csv/remaining_train_test_top3.csv', dtype={'train_id': str, 'test_id': str})
    def path(value):
        return html.escape(os.path.relpath(ROOT / value, home).replace('\\', '/'), quote=True)
    parts = ['<!doctype html><html lang="id"><meta charset="utf-8"><title>45 kasus tersisa</title>',
             '<style>body{font:16px system-ui;margin:24px}section{border:1px solid #ddd;margin:18px 0;padding:16px}figure{display:inline-block;width:30%;margin:1%;vertical-align:top}img{width:100%;height:260px;object-fit:contain}</style>',
             '<h1>45kasus masih berbeda dari review tim</h1><p>126keputusan tegas:81cocok,45berbeda. Awal tahap ini48berbeda;0432/1510/1817menjadi cocok.34acuan manusia tetap,tidak ada tambahan. Audit ini dilakukan setelah kandidat dan keputusan publikasi dibekukan,bukan input fit.20tentatif,2bersyarat,2ditahan terpisah. Test ditampilkan tanpa kategori,tidak masuk training.</p>']
    for row in remaining.itertuples():
        parts.append(f'<section><h2>Train {row.parent_id}</h2><p>Program:<b>{row.current_class}</b> | Review tim:<b>{row.proposed_class}</b><br>Asal:{row.current_source}</p>')
        parts.append(f'<figure><img src="{path(row.source_path)}"><figcaption>Train {row.parent_id}</figcaption></figure>')
        for n in nearest[nearest.train_id.eq(row.parent_id)].sort_values('rank').head(2).itertuples():
            parts.append(f'<figure><img src="{path(n.test_path)}"><figcaption>Test {n.test_id};cosine{n.cosine:.3f},bukan bukti kategori</figcaption></figure>')
        parts.append('</section>')
    (home / 'latest_review.html').write_text('\n'.join(parts) + '</html>', encoding='utf-8')
    text = f'''# Penyempurnaan material dan source group — 7 November 2026

Status aktif: **48menjadi45perbedaan terhadap review tim**. Dari126keputusan tegas,81cocok. Tiga yang pulih:0432cardboard→glass;1510foodwaste→plastic;1817textile→paper. Tidak ada keputusan yang sebelumnya cocok menjadi berbeda.34acuan manusia tetap,max5/kelas; tidak ada tambahan jatah.20tentatif,0676/1650bersyarat dan1662/1885ditahan terpisah.

## Perubahan yang diterapkan

Lima foto utuh berubah oleh head neural:0158paper→plastic;0432cardboard→glass;0673cardboard→paper;1510foodwaste→plastic;1817textile→paper. Hanya tiga masuk kumpulan review tegas; dua lainnya tetap prediksi belum terverifikasi, bukan tambahan klaim benar. Total perubahan sejak snapshot150lama adalah73kategori.

SIFT/RANSAC memeriksa8047pasangan, mendapat116koneksi kuat yang mencakup183foto. Komponen tersebut sudah konsisten kategori sehingga tidak memberi relabel baru langsung. Namun90koneksi menyatukan kelompok sumber yang sebelumnya berbeda dan70koneksi melintasi fold lama. Seluruh koneksi kuat kini satu group/partisi. Untuk1904foto eligible,1870source group menjadi1777. Ini group sumber foto, bukan jumlah kategori; kategori tetap8. Data test tidak masuk penemuan group ini.

## Eksperimen dan evaluasi

1. MINC GoogLeNet non-VLM:23material,142layergraphCaffeasli,58layerparameter;1904foto×3tampilan. SVMmaterial/tekstur dan hybrid lokal memakai prior dingin sebelum acuan manusia, serta34acuan. Diagnostik11/34tidak melewati guard14; usulan tidak diterapkan.
2. Head neural LayerNorm/Linear256/GELU/Dropout/Linear8 pada dua bankfitur:taskCNNdanobject+MINC. Hanya inti konsensus otomatis dan acuan manusia menjadi target. CEuntukmanusia;CEatauGCEq=.7untukpseudo. Pada5foldsourcegroupbaru,CE14/34danGCE15/34;GCEdipilih berdasarkan acuan saja. Lima perubahan memenuhi kesepakatan kedua bank,p>=.85,margin>=.4dan perlindungan terhadap refinement lama. Probabilitas tidak terkalibrasi.
3. Self-training satu putaran, pseudo-label hanya dibuat pada sisi fit masing-masing outer fold, tanpa querygroup/labelquery. Teacher14/34danstudent14/34;guard15gagal sehingga tidak ada usulan self-training diterapkan. Streamdropoutteacherregenerasi berbeda dari percobaan sebelumnya dan tidak diklaim bitwise sama.

Angka diagnostik berasal dari34acuan yang dipilih untuk pengembangan, bukan sampel acak atau Accuracy independen. Pemilihan objective juga melihat diagnostik tersebut. **81/126**adalah kesesuaian terhadap review pengguna setelah pembekuan, bukan Accuracy panitia. Seluruh126review maupun45kasus tersisa tidak menjadi target atau filter training. Keputusan publikasi direkam sebelum audit privat.

Sumber primer dan protokol ada pada [kartu riset](MATERIAL_TEXTURE_RESEARCH.md). GCE: Zhang & Sabuncu,NeurIPS2018,[paper](https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf). MINC:[halaman penulis](https://opensurfaces.cs.cornell.edu/publications/minc/). Tidak menambah dataset gambar eksternal,tidak memakaiVLMatauAPIinference. Bobot hanya dijalankan lokal; sourceasal dan checksum dicatat.

## Notebook dan processed

Notebook tunggal [DSC2026_Notebook.ipynb](../../final-submit/DSC2026_Notebook.ipynb) lulus Run All dari processed kosong:{code['executed']}selkode,0error,{code['embedded_figures']}visual.4793hashgambar diperiksa.1913sampel train tetap disimpan pada8folder,1911eligible,480test,7nativecrop. Semua116edgegeometri diperiksa berada pada groupdanpartisi yang sama. Metadata sumber dan kategori dipisahkan;1662/1885bukan acuan atau referensi crop. Piksel test dan sumber asli tidak berubah.

Run All preprocessing mematerialisasikan snapshot assignment dan menghitung kategori crop; belum mengulang seluruh fit assignment heads. Classifier final,validasi independen danCSVsubmissionbelum dibuat. Metode yang dicoba tidak membuktikan semua alternatif algoritma mustahil;45kasus belum terselesaikan secara andal pada tahap ini. Jangan memaksa45label dariarsip atau menurunkan batas penerimaan agar tampak cocok.

Bukti: [validasi tahap](json/material_followup_delivery.json),[publikasi](json/bounded_anchor_publication.json),[keputusan sebelum audit](json/material_publication_decision.json),[Run All](execution.json),[validasi koleksi](skills_validation.json). Galeri aktif:[45kasus](remaining_visual_review/latest_review.html); tabel:[kasus tersisa](table-csv/remaining_current_review_differences.csv).
'''
    (NOTES / 'MATERIAL_REFINEMENT_RESULT.md').write_text(text, encoding='utf-8')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
