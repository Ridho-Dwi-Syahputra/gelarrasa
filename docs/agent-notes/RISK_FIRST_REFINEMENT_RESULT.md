# Perbaikan selektif berprioritas presisi

Status aktif: **26 berbeda / 100 sesuai dari126review tegas**; sebelumnya26berbeda. Target nol belum tercapai.

Eksperimen ini mengubah tujuan secara eksplisit sebelum audit: menerima cakupan lebih kecil bila presisi pengembangan dan ranking Wilson membaik. Guard lama25/34dan guard konsensus cakupan16tetap ditolak pada rekornya; hasil-hasil tersebut tidak diubah menjadi lolos. Plan baru mencakup enam kandidat, ambang tetap, minimal10acuan terjangkau, presisi empiris minimal0.95, macrorecall keseluruhan minimal0.65 serta rankingWilson lebih tinggi0.02dari15/16sebelumnya. Wilson di sini hanya alat ranking pada34acuan yang dipakai berulang, bukan batas risiko yang sah setelah pencarian.

Pemenang object_zoom, ambang0.98, mencapai12/12pada bagian pengembangan yang diterima. Seluruh34tetap22sesuai/macro0.70; jangan mengubah12/12ini menjadi klaim Accuracy100persen untuk dataset. Prediksi query dikeluarkan menurut kelompok sumber; dua graph final train-only harus setuju. Graph bukan validasi independen. Tidak ada test, arsip anotasi, atau daftar26review sebagai target/filter/seleksi per-ID.

Publikasi menerapkan20dari38usulan, sesuai cap20yang sudah dicatat. Review setelah pembekuan: pulih tidak ada; regresi tidak ada. Perubahan di luar review belum terverifikasi. Semua34sumber manusia tetap, tanpa tambahan kuota;1662/1885tetap ditahan.

Notebook EDA/preprocessing dieksekusi ulang dari processed kosong:13sel kode,0error,16visual. Validasi penuh memeriksa4793hash gambar. Raw dan test utuh. Ini materialisasi snapshot assignment serta evaluasi crop; belum fitting ulang seluruh assignment heads atau classifier/submission final.

Bukti: json/risk_first_expert_plan.json, json/risk_first_expert_frozen.json, json/bounded_anchor_publication.json, json/current_review_status.json, execution.json dan skills_validation.json. Daftar tersisa: table-csv/remaining_current_review_differences.csv; galeri remaining_visual_review/latest_review.html.

## Pengujian lanjutan setelah publikasi

- MINC permukaan + DINO bentuk, empat bank fitur dengan centering/PCA fit per source-groupfold: terbaik24/34danmacro0.75. Tidak ada ambang minimum share lintasbank yang memenuhi cakupan10dengan presisi0.90. Ditolak; tidak diterapkan. Tidak ada pemetaan manual dari23kategori material pretrained ke8kelas lomba.
- Trusted-gradient reweighting, kontrol uniform versus gradient alignment pada dua bank, lima fold: kontrol terbaik23/34danmacro0.725; gradient-alignment terbaik23/34tetapimacro0.6083. Kedua tujuan, global dan precision-first, gagal guard. Tidak diterapkan. Formula epsilon-zero satu langkah SGD linear diperiksa terhadap autodiff; selisih maksimum1.3878e-17. Kebenaran implementasi matematika tidak membuktikan manfaat model.
- Diagnostik pada34acuan yang sama: enam guru sama-sama salah pada5acuan ketika kelompoknya ditahan; setidaknya satu benar pada29. Model utama salah tetapi guru lain benar pada5. Ini mendukung pengujian pemilihan lokal, tetapi tidak memberikan aturan pemilih yang sudah terbukti berhasil pada26review lain. Kelima sumber tersebut adalah bagian dari34acuan yang sudah dipakai, bukan tambahan manual.

Sisa aktif tetap26sesudah seluruh pengujian ini. Dua eksperimen terakhir tidak membaca126review atau arsip anotasi sebagai target/filter/pemilih prediksi. Peneliti telah melihat audit agregat sebelumnya, sehingga hasil pengembangan tidak independen dari seluruh proses desain. Tidak ada klaim bahwa26semuanya kesalahan terhadap ground truth panitia atau bahwa target nol mustahil; target itu belum tercapai pada bukti yang tersedia.

Rekor tambahan: json/material_semantic_neighbors_frozen.json, json/gradient_reweight_frozen.json, table-csv/gradient_reweight_training.csv, json/human_source_shared_error_diagnostic.json. Rujukan konsep gradient reweighting: [Ren dkk., ICML2018](https://proceedings.mlr.press/v80/ren18a.html). Implementasi adalah adaptasi head linear dengan clean CE tambahan dan acuan terkurasi, bukan reproduksi penuh atau clean validation set yang unbiased.
