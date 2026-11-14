"""Synchronize source-refinement evidence after publication and private audit."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];NOTES=ROOT/'docs/agent-notes'

def main():
    state=json.loads((NOTES/'json/current_review_status.json').read_text());pub=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text());assert pub['status']=='passed' and state['active_prefix']==pub['source_frozen_prefix']
    prefixes=['semantic_dictionary_neighbors','dinov3_base_neighbors','origin_neutral_v3','full_distribution_neighbors','semantic_concept_seed','weighted_source_concept','selective_feature_union','augmented_prototype_consensus','frozen_prototype_continuation','augmented_source_softmax','broad_source_concept','material_object_specialists','local_descriptor_consensus','residual_graph_correction','residual_selective_router','warm_residual_correction','frozen_residual_continuation','complementary_residual_router','local_source_residual','kernel_residual_interpolation','augmented_concept_seed','residual_teacher_student','paired_material_router','detector_gated_prior','cluster_plastic_router','pairwise_detector_consensus']
    rows=[]
    for prefix in prefixes:
        path=NOTES/f'json/{prefix}_frozen.json'
        if not path.exists():continue
        r=json.loads(path.read_text());rows.append({'prefix':prefix,'development_matches':r.get('human_leave_source_group_out_matches'),'development_macro_recall':r.get('development_balanced_recall'),'accepted_guard':r.get('accepted_by_development_guard'),'algorithm_changes':r.get('changed_rows_vs_baseline'), 'independent_accuracy':False})
    pd.DataFrame(rows).to_csv(NOTES/'table-csv/source_refinement_experiment_summary.csv',index=False)
    remaining=pd.read_csv(NOTES/'table-csv/remaining_current_review_differences.csv',dtype=str)
    # Column names are discovered without converting this audit into a fit input.
    label_column=next((c for c in ['proposed_class','review_class','confirmed_class','expected_class','review_choice','review_label'] if c in remaining),None)
    counts=remaining[label_column].value_counts().to_dict() if label_column else {}
    follow={'target_remaining_differences':0,'target_achieved':state['remaining_review_differences']==0,'active_prefix':state['active_prefix'],'remaining_review_differences':state['remaining_review_differences'],'matches_review':state['matches_confirmed_review'],'review_cases':126,'previous_remaining':26,'new_manual_labels':0,'human_sources_used':34,'remaining_review_class_counts':counts,'source_experiments':rows,'notebook_sha256':state['notebook_sha256'],'manifest_sha256':state['manifest_sha256'],'scope':state['scope'],'runall_limit':'materializesfrozenassignmentandreevaluatescrops;notfullrefitofassignmentheads;classifier/submissionnotyetmade'}
    follow['algorithm_recovered_review_ids']=['0785','1176','1230','1417','0969','1079','0714','1684']
    follow['user_review_amendments']=[{'image_id':'0536','old_class':'paper','new_class':'cardboard','model_changed':False,'audit_count_before':18,'audit_count_after':17}]
    follow['review_reference_sha256']=state.get('review_reference_sha256')
    (NOTES/'json/zero_target_followup_status.json').write_text(json.dumps(follow,indent=2)+'\n',encoding='utf-8')
    table='\n'.join(f"| {r['prefix']} | {r['development_matches']}/34 | {r['development_macro_recall']} | {'lolos' if r['accepted_guard'] else 'ditolak'} | {r['algorithm_changes']} |" for r in rows)
    report=f'''# Pengujian fitur sumber dan koreksi residual, 8 November 2026

Status aktif: **{state['remaining_review_differences']} berbeda / {state['matches_confirmed_review']} sesuai dari126review tegas**. Prefix `{state['active_prefix']}`. Target nol {'tercapai pada review ini saja' if follow['target_achieved'] else 'BELUM tercapai'}. Acuan manusia tetap34sumber unik, maksimal5/kelas; tidak ada kuota baru atau koreksi ID khusus.

Review126dibaca hanya oleh audit setelah keluaran model dibekukan. Model tidak membaca folder arsip anotasi atau kategori test. Peneliti sudah melihat audit sebelumnya; ini pengembangan adaptif, bukan blind holdout. Keselarasan review tidak sama dengan Accuracy resmi.

Ralat eksplisit pengguna8November:0536dipastikan **cardboard**, menggantikan acuan audit sebelumnya paper. Program memang sudahcardboard; manifest, input, notebook dan34seed TIDAKdiubah. Sebelum ralat, eksperimen menurunkan26ke18(8pemulihan algoritmis). Sesudah ralat jumlahaudit17; satu pengurangan ini adalah revisi acuan, bukan koreksi model. Bukti `json/user_review_amendment_0536.json` dan hash referensi review dicatat terpisah.

| Metode/prefix | Acuan pengembangan cocok | Macro recall | Guard | Perubahan program |
|---|---:|---:|---|---:|
{table}

Jumlah perubahan program bukan jumlah koreksi terverifikasi. Metode ditolak menyimpan0perubahan aktif. Batas20pada prototipe adalah batch operasional; sisa15diterapkan terpisah memakai prediksi/threshold yang persis sama, tanpa refit atau pemilihan berdasarkan audit. Semua perubahan di luar126review masih belum diverifikasi.

## Temuan

- Pemetaan CNN sumber ke8kategori sebelumnya hanya mencakup median0.3108massa probabilitas;357/1904foto mencakup kurang0.1. Ini diagnosis hilangnya informasi sumber, bukan prevalensi target. Distribusi penuh21841logit kini diuji sebelum proyeksi tetap512dimensi. T1/T2 sama-sama tidak membaca label atau test untuk membentuk proyeksi.
- DINOv3Base image-only85.6M mengekstrak1904train eligible, dimensi3072: CLS, validpatchmean, objectmean, objectstd; objectmean berada1536:2304. Jangan tertukar dengan urutan DINOv2Base. Small/CLIP/DINO.txt bukan model yang dipakai.
- Source-dictionary menurunkan26ke25(0785); DINOv3Base25ke23(1176,1230); penyamaan threshold untuk seluruh asal prediksi mesin23ke22(1417). Tidak ada tambahan acuan manusia.
- Banyaknya augmentasi tidak menambah ukuran sampel manusia:34sumberx8tampilan tetap34acuan, bukan272anotasi atau272validation samples. Seluruh tampilan satu sumber ditahan bersama.
- Descriptor lokal mengambil patch cached49lokasi pada dua skala, memproyeksikan128tetap, dan mencocokkan3tetangga patch perkelas. Ini adaptasi ideDN4 tanpa episodic training; hasil gagal guard, tidak dipublikasikan.
- Koreksi residual mengikuti gagasan Correct and Smooth, tetapi graph hanya sisi train setiapfold. Query tidak masuk graph/PCA/core/residual. Spesialis residual global26/34 gagal guard. Routing selektif terpisah memakai residual pada19acuan(18sesuai), prototipe sebagai perlindungan konflik, dan fallbackV3; mendapat28/34,macro.85. Semua angka memakai34acuan pengembangan berulang, bukan sertifikat risiko.
- Dua batch residual27perubahan menurunkan22ke20(0969,1079),lalu20ke18(0714,1684); tidak ada regresi review pada kedua batch. Varian warm, localgraph, kernelresidual dan multi-viewconcept berikutnya gagal guard, tidak diterapkan. Seluruh8pemulihan otomatis pada putaran sumber:0785,1176,1230,1417,0969,1079,0714,1684.
- Guru residual dengan pseudo-label korektif yang ditambang hanya dari train memperoleh25/34,macro0.7583: proposal plastic dalam tiapfold hanya1–5, sedangkan foodwaste sekitar20dan electronic16–18. Guard ditolak; tidak ada perubahan aktif.
- Spesialis bahan berpasangan memakai tiga bank gambar dan21pasang kelas pada34acuan. Pasangan paper/plastic serta plastic/cardboard tidak memenuhi guard dua bank; routing terbaik28/34,macro0.85,namun satu acuan yang berubah justru salah. Guard ditolak; tidak ada perubahan aktif.
- Model guru limbah delapan kelas pralatih juga salah pada kelima acuan plastic yang sah, sering menyebut paper/cardboard/metal/glass/textile. Akurasi tinggi pada dataset sumber tidak membuktikan pembeda kemasan lentur di foto kompetisi. Batas lima acuan plastic sudah habis, sehingga penyamaan paksa terhadap review tidak dapat dibenarkan oleh validasi train yang tersedia.
- Model sumber 13kelas yang memiliki label plastic wrappers diuji secara lokal pada34acuan sah; skor plastic-vs-lain AUC0.6759. Embedding akhir model itu dengan SVM source-heldout mendapat AUC0.2069–0.4069, tidak diekstrak ke seluruh train. Source model diperoleh hanya sebagai bobot, tanpa mengunduh citra data eksternal.
- Bias prior plastic pada dua bank residual menaikkan OOF keseluruhan dari28ke29/34 tetapi hanya2dari4acuan yang dialihkan benar; percobaan ini tidak dipublikasikan. Penambahan bukti detektor Plastic membuat OOF30/34,macro0.90,dengan2/2perubahan acuan benar. Publikasi train-only23perubahan lulus Run All dan4793hash, tetapi audit pascabeku menemukan satu pemulihan (0964) dan tiga regresi (0597,0969,1176): sisa17menjadi19. Snapshot sebelumnya dipulihkan utuh; model detektor ditandai eksperimen gagal, bukan metode aktif. Audit ini mengungkap batas generalisasi acuan34dan bukan validasi independen.
- Guru plastik dari114acuan otomatis positif dan835negatif memberi AUC hingga0.9034untuk kelas plastik tetapi tidak ada konfigurasi yang mencapai guard keseluruhan34acuan. Gradient boosting delapan kelas pada1.323acuan otomatis hanya13–19/34; ditolak. Seluruh17kasus tersisa adalah grup gambar train tunggal tanpa pasangan duplikat train terverifikasi yang mengandung acuan manusia.
- Kluster visual64/128/256 pada DINOv3 dan ConvNeXt hanya18–25/34 bila dipakai mengganti semua kelas. Routing selektif satu konfigurasi29/34,macro0.875,mengubah27prediksi train; Run All dan hash lulus. Audit pascabeku pulihkan0964tetapi regresikan0714, sehingga tetap17. Snapshot stabil sebelumnya dipulihkan, dan kluster ini ditolak sebagai jalur aktif. Kluster dekat secara visual belum memberi bukti bahan kemasan yang cukup.

## Artefak operasional dan batas reproduksi

Uji tambahan bag-of-visual-words ConvNeXt dengan codebook train-only dan validasi kelompok sumber memperoleh 15/34 (patch kasar) serta 16/34 (patch halus); recall plastic 0 dan 0,2. Ini di bawah metode aktif 28/34. Tidak ada perubahan label aktif. Bukti: `json/patch_vocabulary_probe_plan.json` dan `json/patch_vocabulary_probe_result.json`.

Spesialis plastik ExtraTrees dengan fitur DINOv3, detektor objek, dan acuan otomatis pra-review mencapai AUC 0,855 pada 34 acuan kelompok sumber. Ambang terbaik memberi 29/34, tetapi satu dari tiga acuan yang dialihkan ke plastik salah; tidak satu pun konfigurasi lolos guard. Tidak ada perubahan label aktif. Bukti: `json/tree_plastic_material_probe_plan.json`, `json/tree_plastic_material_probe_result.json`.

Usulan delapan ahli perkelas dilatih pada 1.323 acuan otomatis pra-review (200 kandidat plastik) dan34acuan manusia. Jika skor ahli tertinggi langsung menjadi kelas, hanya16/34acuan sah cocok; routing selektif tidak menambah hasil aktif. Ahli berpasangan plastik-vs-tujuhbahan dengan banyak acuan otomatis juga gagal bila sendiri: terbaik27/34. Sinyal gabungan ahli berpasangan dan detektor memenuhi guard pengembangan30/34,macro0,9,dua rute benar, sehingga diterapkan reversibel pada39gambar train. RunAll13/13dan4793hash lulus. Audit pascabeku: empat gambar pulih (0310,0978,1436,1515), tetapi sepuluh gambar yang semula sesuai bergeser; selisih17menjadi23. Snapshot17dipulihkan utuh. Guard pada34acuan tidak cukup untuk generalisasi ke kemasan lainnya. Bukti `json/class_experts_probe_result.json`, `json/pairwise_auto_experts_probe_result.json`, `json/pairwise_detector_consensus_frozen.json`, `json/bounded_anchor_publication.json`.

Pemeriksaan asal target menunjukkan auto-label awal cocok9/34acuan manusia, termasuk1/5acuan plastik. Pada16acuan yang oleh sistem auto dinyatakan anchor, hanya5cocok. Sampel manusia ini dipilih karena sulit, sehingga rasio tersebut bukan estimasi akurasi semua1.323anchor; tetapi ia membantah asumsi bahwa200kandidat plastik otomatis sudah menjadi200contoh plastik tepercaya. Perbaikan ahli perkelas harus lebih dulu meningkatkan kemurnian pseudo-label atau representasi pada kelompok sulit, lalu diuji pada sumber independen. Bukti silang dihitung dari `data/processed/manual_seeds.csv` dan `table-csv/automatic_full_seed_targets.csv` tanpa membaca review pascabeku.

Klarifikasi pengguna8November: yang dimaksud adalah17sisa, dan hasil yang sudah sesuai ingin dibekukan. Uji pembatasan otomatis tanpa memilihIDdari review menggunakan top5 dan top10 skor gabungan tiga sumber. Top5: lima perubahan train, tidak ada pemulihan/regresi pada126review, sisa17. Top10: satu pulih dan satu regresi, sisa17. Kedua batch dipulihkan ke snapshot aktif17karena tidak memberi perbaikan bersih. Ini menunjukkan pembatasan volume saja tidak memisahkan prediksi yang benar dari yang salah; mengunci126gambar berdasarkan hasil review akan menggunakan koreksi manusia perIDmelampaui kuota, sehingga tidak diterapkan dalam jalur lomba. Bukti `json/capped_pairwise_consensus_frozen.json`, `json/capped_pairwise_consensus_10_frozen.json`, dan arsip rollback lokal.

Notebook `final-submit/DSC2026_Notebook.ipynb` menjalankan13sel kode,0error,16visual. Validator memeriksa4793hash. Train1913sampel/1911eligible, test480, tujuh crop native. Foto1662/1885ditahan. Raw/test utuh dan gambar sulit lain tetap dipertahankan.

Run All mematerialisasikan `data/preprocessing_inputs.npz` dan mengevaluasi crop; **belum mengulang seluruh fit assignment heads dari awal**. Belum membuat classifier atau CSV final. Notebook tidak memanggil/membaca folder docs. Script eksperimen dan provenance berada di agent-notes. Menyediakan seluruh refit dalam notebook final dan akses dataset untuk panitia tetap pekerjaan berikutnya sebelum pengumpulan.

Bukti mesin terbaru: `json/current_review_status.json`, `json/bounded_anchor_publication.json`, `execution.json`, `skills_validation.json`; daftar dan galeri aktif diperbarui audit. Detail guard/fit ada pada pasangan plan/frozen tiapprefix.
'''
    (NOTES/'SOURCE_METHOD_REFINEMENT_RESULT.md').write_text(report,encoding='utf-8')
    (NOTES/'CURRENT_WORK.md').write_text(f'''# Status kerja aktif

{state['remaining_review_differences']}berbeda/{state['matches_confirmed_review']}sesuai dari126review, prefix `{state['active_prefix']}`. Manual34tetap; tidak memakai test/anotasi arsip sebagai target. Status nol: {follow['target_achieved']}.

Tidak ada ekstraksi DINO atau publisher yang perlu diulang setelah status publication passed. Cache1904DINOv3Base lengkap. Lihat SOURCE_METHOD_REFINEMENT_RESULT.md dan json/current_review_status.json untuk bukti aktif. Notebook13/13,0error,16visual,4793hash; RunAll assignment snapshot+crop, belumrefitsemuaheads/classifier/CSV. Jangan mengklaim seluruh train/test benar.

Riset lanjutan harus memakai34acuan yang sama dan train-only, freezeplan sebelumfit, guard sebelumaudit, lalu publisher/RunAll/validator, baru audit. Jangan memilih perubahan perID dari sisa review atau menghapus foto sulit. Kuota metal2/food4tidak dapat dipindah ke kelas lain. Tidak ada spawnagent/GUI/push/uploadtanpaotorisasi.
''',encoding='utf-8')
    print(json.dumps({'remaining':state['remaining_review_differences'],'prefix':state['active_prefix'],'experiments':len(rows),'audit_columns':list(remaining.columns)}))

if __name__=='__main__':main()
