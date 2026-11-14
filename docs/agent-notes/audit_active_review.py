"""Audit the current frozen publication; no fitting or model selection here."""
import html
import json
import os
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT, NOTES, sha, dump


def main():
    pub = json.loads((NOTES / 'json/bounded_anchor_publication.json').read_text())
    assert pub['status'] == 'passed'
    prefix = pub['source_frozen_prefix']
    frozen = json.loads((NOTES / f'json/{prefix}_frozen.json').read_text())
    selected = NOTES / f'table-csv/{prefix}_selected.csv'
    assert sha(selected) == frozen['prediction_sha256'] == pub['candidate_sha256']
    expected = pd.read_csv(selected, dtype={'image_id': str}).set_index('key')
    m = pd.read_csv(ROOT / 'data/processed/manifest.csv', dtype={'parent_id': str})
    full = m[m.split.eq('train') & ~m.is_crop].set_index('parent_key').loc[expected.index]
    assert len(full) == 1906 and full.assigned_class.equals(expected.predicted_class.rename('assigned_class'))
    assert full.source_sha256.equals(expected.source_sha256) and full.label_source.equals(expected.category_source.rename('label_source'))
    assert sha(ROOT / 'data/processed/manifest.csv') == pub['manifest_sha256']
    manual = pd.read_csv(ROOT / 'data/processed/manual_seeds.csv')
    assert len(manual) == 34 and manual.assigned_class.value_counts().le(5).all()
    budget = json.loads((NOTES / 'json/manual_label_budget_status.json').read_text())
    assert budget['manual_labels_consumed_by_current_experiments'] == 34
    held = m[m.split.eq('train') & ~m.training_eligible]
    assert set(held.parent_id) == {'1662', '1885'} and not held.human_reviewed.any()
    previous = pd.read_csv(ROOT / pub['backup'] / 'processed/manifest.csv', dtype={'parent_id': str})
    old_full = previous[previous.split.eq('train') & ~previous.is_crop].set_index('parent_key').loc[expected.index]
    a = previous[previous.split.eq('test')].set_index('sample_key').sort_index()
    b = m[m.split.eq('test')].set_index('sample_key').sort_index()
    assert a.source_sha256.equals(b.source_sha256) and a.output_sha256.equals(b.output_sha256)
    with np.load(ROOT / pub['backup'] / 'preprocessing_inputs.npz', allow_pickle=False) as before, np.load(ROOT / 'data/preprocessing_inputs.npz', allow_pickle=False) as after:
        test = np.char.startswith(before['keys'].astype(str), 'test:')
        for key in before.files:
            if before[key].ndim == 1 and len(before[key]) == 2400:
                assert np.array_equal(before[key][test], after[key][test]), key
    review = pd.read_csv(NOTES / 'table-csv/team_review_revision.csv', dtype={'parent_id': str}).fillna('')
    lookup = full.set_index('parent_id')
    review['current_class'] = review.parent_id.map(lookup.assigned_class)
    review['current_source'] = review.parent_id.map(lookup.label_source)
    review['before_publication_class'] = review.parent_id.map(old_full.set_index('parent_id').assigned_class)
    review['current_matches_review'] = review.confirmed_for_audit & review.current_class.eq(review.proposed_class)
    review['before_matches_review'] = review.confirmed_for_audit & review.before_publication_class.eq(review.proposed_class)
    review['current_human_anchor'] = review.parent_id.map(lambda v: 'train:' + v in set(manual.key))
    firm = review[review.confirmed_for_audit]
    queue = firm[~firm.current_matches_review].sort_values('parent_id')
    recovered = firm[~firm.before_matches_review & firm.current_matches_review]
    regressions = firm[firm.before_matches_review & ~firm.current_matches_review]
    assert len(firm) == 126 and not queue.current_human_anchor.any()
    review.to_csv(NOTES / 'table-csv/current_review_outcome.csv', index=False)
    queue.to_csv(NOTES / 'table-csv/remaining_current_review_differences.csv', index=False)
    execution = json.loads((NOTES / 'execution.json').read_text())
    validation = json.loads((NOTES / 'skills_validation.json').read_text())
    assert execution['status'] == validation['status'] == 'passed' and execution['sha256'] == pub['notebook_sha256']
    assert sha(ROOT / 'final-submit/DSC2026_Notebook.ipynb') == execution['sha256']
    assert validation['image_hashes_verified'] == 4793
    proof = {'status': 'passed', 'active_prefix': prefix, 'confirmed_review_cases': 126,
             'matches_confirmed_review': int(firm.current_matches_review.sum()), 'remaining_review_differences': len(queue),
             'remaining_before_publication': int((~firm.before_matches_review).sum()),
             'recovered_ids_this_publication': recovered.parent_id.tolist(), 'regressed_ids_this_publication': regressions.parent_id.tolist(),
             'full_frame_changes_this_publication': int(full.assigned_class.ne(old_full.assigned_class).sum()),
             'cumulative_changes_vs_initial_program': frozen.get('cumulative_changes_vs_initial_program'),
             'human_sources_used': 34, 'human_class_counts': manual.assigned_class.value_counts().to_dict(),
             'new_manual_labels': 0, 'held_ids': ['1662', '1885'], 'test_pixels_and_operational_fields_unchanged': True,
             'code_cells_executed': execution['executed'], 'embedded_figures': execution['embedded_figures'],
             'image_hashes_verified': 4793, 'train_samples': 1913, 'eligible_train_samples': 1911,
             'notebook_sha256': execution['sha256'], 'manifest_sha256': pub['manifest_sha256'],
             'source_prediction_sha256': frozen['prediction_sha256'],
             'review_reference_sha256': sha(NOTES / 'table-csv/team_review_revision.csv'),
             'review_used_for_fit_or_selection': False,
             'scope': 'post-freeze review agreement,not official semantic Accuracy or competition certification'}
    dump(NOTES / 'json/current_review_status.json', proof)
    home = NOTES / 'remaining_visual_review'
    neighbors = pd.read_csv(NOTES / 'table-csv/remaining_train_test_top3.csv', dtype={'train_id': str, 'test_id': str})
    def path(value):
        return html.escape(os.path.relpath(ROOT / value, home).replace('\\', '/'), quote=True)
    parts = ['<!doctype html><html lang="id"><meta charset="utf-8">', f'<title>{len(queue)} kasus tersisa</title>',
             '<style>body{font:16px system-ui;margin:24px}section{border:1px solid #ddd;margin:18px 0;padding:16px}figure{display:inline-block;width:30%;margin:1%;vertical-align:top}img{width:100%;height:260px;object-fit:contain}</style>',
             f'<h1>{len(queue)} kasus berbeda dari review tim</h1><p>Dari126keputusan tegas,{proof["matches_confirmed_review"]}sesuai. Acuan manusia tetap34,max5perkelas. Audit dilakukan setelah prediksi dibekukan; bukan target atau seleksi model.20tentatif,2bersyarat,2ditahan terpisah. Test ditampilkan tanpa kategori,tidak masuk fit.</p>']
    for row in queue.itertuples():
        parts.append(f'<section><h2>Train {row.parent_id}</h2><p>Program:<b>{row.current_class}</b> | Review:<b>{row.proposed_class}</b><br>Asal:{row.current_source}</p><figure><img src="{path(row.source_path)}"><figcaption>Train {row.parent_id}</figcaption></figure>')
        for n in neighbors[neighbors.train_id.eq(row.parent_id)].sort_values('rank').head(2).itertuples():
            parts.append(f'<figure><img src="{path(n.test_path)}"><figcaption>Test {n.test_id};cosine{n.cosine:.3f};bukan bukti kategori</figcaption></figure>')
        parts.append('</section>')
    (home / 'latest_review.html').write_text('\n'.join(parts) + '</html>', encoding='utf-8')
    text = f'''# Status kategori aktif

**{len(queue)} perbedaan tersisa dari 126 keputusan review tegas; {proof['matches_confirmed_review']} kini sesuai.** Sebelum publikasi ini: {proof['remaining_before_publication']} berbeda. Pulih pada tahap ini: {', '.join(proof['recovered_ids_this_publication']) or 'tidak ada'}. Regresi review: {', '.join(proof['regressed_ids_this_publication']) or 'tidak ada'}.

Metode aktif: `{frozen['method']}`. Publikasi memperbarui {proof['full_frame_changes_this_publication']} kategori foto utuh; total sejak snapshot 150 lama: {proof['cumulative_changes_vs_initial_program']}. Perubahan di luar kumpulan review tetap prediksi belum terverifikasi. Acuan manusia tetap 34, maksimal lima per kelas; sisa hanya metal dua dan foodwaste empat. Foto 1662/1885 tetap ditahan. Sebanyak 20 keputusan tentatif dan dua bersyarat tidak dijadikan keputusan tegas.

Notebook tunggal lulus Run All preprocessing: {execution['executed']} sel kode, nol error dan {execution['embedded_figures']} visual. Sebanyak 4.793 hash gambar diperiksa. Train tetap 1.913 sampel dalam delapan folder, dengan 1.911 eligible; test 480 dan native crop tujuh. Raw dan piksel test tetap utuh. Semua isi processed dibuat notebook dari raw dan input snapshot. Run All belum mengulang seluruh fitting assignment heads; classifier final/CSV belum dibuat.

Snapshot Q&A resmi terbaru dan penegasan kronologi pengguna menjadi patokan. Izin otomatis train-only dan non-VLM tidak mengubah batas manual atau melonggarkan larangan test/VLM/dataset eksternal. Seluruh review bukan target atau pemilih prediksi. Keselarasan review bukan Accuracy panitia atau jaminan semua gambar benar.

Bukti: [status JSON](json/current_review_status.json), [publikasi](json/bounded_anchor_publication.json), [Run All](execution.json), [validasi](skills_validation.json). [Galeri kasus tersisa](remaining_visual_review/latest_review.html), [tabel audit](table-csv/remaining_current_review_differences.csv), dan [klarifikasi terbaru](QNA_UPDATE_20261107.md). Laporan tahap sebelumnya tetap historis.
'''
    (NOTES / 'CURRENT_LABEL_STATUS.md').write_text(text, encoding='utf-8')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
