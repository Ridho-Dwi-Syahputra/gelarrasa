# Eksperimen hybrid untuk kandidat label train

## Metode

Menerapkan materi webinar CNN embedding + HOG + histogram warna → StandardScaler → RBF-SVM pada foto train mentah. Notebook membandingkan CNN saja, HOG+warna saja, dan fusi CNN+HOG+warna. CNN yang dipakai adalah frozen EfficientNet-B0 pretrained non-VLM. SVM memakai `class_weight=balanced`; pilihan C/gamma dilakukan di inner grouped folds. Fitur dan label kandidat dibuat hanya dari train; test tidak diekstrak dan tidak digunakan untuk fitting.

Seed peserta berjumlah 37 (maksimal lima per kelas). Evaluasi memakai nested grouped OOF pada seed, dengan grup yang menyatukan hash file/piksel identik dan seed yang pHash-nya berjarak paling banyak 4. Kandidat untuk seluruh train ditulis terpisah. Kandidat tidak menimpa assignment aktif atau folder processed.

## Hasil lokal yang benar-benar dijalankan

| Fitur | Seed OOF accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN embedding | 0.0270 | 0.0417 | 0.0625 |
| HOG + warna | 0.0000 | 0.0000 | 0.0000 |
| CNN + HOG + warna | 0.0000 | 0.0000 | 0.0000 |

Metrik ini berasal dari 37 seed saja dan sangat tidak memadai. Hybrid tidak terbukti mampu merekonstruksi assignment aktif atau menghasilkan label semantik yang dapat dipercaya. Karena itu kandidat hanya bahan eksperimen, bukan pengganti label train aktif. Kecocokan OOF terhadap seed juga bukan validasi ground truth independen.

## Artefak

- `features/hybrid_labeling/train_cnn_hog_color.npz`: 1.920 baris fitur train dan identitas sumber.
- `labels/hybrid_train_label_candidates.csv`: 1.920 kandidat; memuat prediksi tiap ablation, margin SVM mentah, sumber label, dan hash foto sumber.
- `validation/hybrid_label_seed_oof.csv`: 37 prediksi seed OOF per metode.
- `validation/hybrid_label_seed_oof_summary.csv`: ringkasan metrik di atas.
- `validation/hybrid_label_inner_selection.csv`: parameter terpilih per outer fold.
- `logs/hybrid_label_feature_provenance.json`: konfigurasi, dimensi fitur, sumber bobot, cakupan data, dan status penggunaan.

Notebook: `final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb`, sel “Hybrid CNN + HOG + warna untuk kandidat label train”.

## Agreement with the currently active manifest (posthoc only)

After freezing the candidate output, a separate ID join found **393/1,906** full-frame assignments matching the active manifest (**1,513 differ**). Among 1,879 eligible active rows, **383 match and 1,496 differ**. This audit did not feed back into the model, parameters, or candidate labels. It confirms that this webinar hybrid experiment is not a replay of the active assignment and must not be used as a replacement.

Per-ID detail: `validation/hybrid_candidate_vs_active_manifest.csv`; summary: `validation/hybrid_candidate_vs_active_manifest_summary.json`. These are parity diagnostics against the active project manifest, not semantic accuracy.


## Pembaruan pelaporan webinar (13 November 2026)

Sel notebook hybrid kini juga menghitung dan menyimpan precision, recall, F1 dan support per kelas dari grouped OOF, confusion matrix OOF lengkap, serta durasi persiapan fitur, nested OOF dan refit kandidat. Ini mengikuti daftar evaluasi pada materi webinar. File tersebut baru dihasilkan setelah sel dijalankan ulang; belum ada hasil dari revisi pelaporan ini. Semua target tetap 37 seed tim dan citra train mentah. Kandidat tidak mengubah manifest aktif.

## Revisi penerapan materi hybrid (13 November 2026)

Kode sel hybrid pada notebook diperbarui setelah pemeriksaan metode: kini membandingkan CNN-only, HOG-only, RGB-histogram-only, HOG+RGB, dan CNN+HOG+RGB. Histogram memakai 256 bin per kanal (768 fitur). Setiap blok distandardisasi dari bagian fit fold lalu dinormalisasi L2 per gambar; proporsi blok dan C/gamma dipilih pada inner grouped folds. Semua fitur tetap berasal dari foto train mentah, sementara 37 seed peserta menjadi satu-satunya target manusia. Cache lama otomatis tidak cocok dengan identitas resep baru dan akan dihitung ulang.

Angka OOF dan tabel kandidat di atas adalah keluaran **resep sebelumnya** (histogram 32 bin/kanal dan scaler gabungan); jangan menganggapnya sebagai hasil revisi. Notebook revisi belum dijalankan, sehingga belum ada skor atau label kandidat baru yang dapat dilaporkan. Tahap ini tetap menghasilkan kandidat otomatis terpisah dan tidak menimpa assignment aktif atau folder processed. Penerimaan sebagai assignment aktif masih memerlukan guard evaluasi seed yang memadai dan rekonstruksi/parity assignment penuh; metode ini sendiri tidak menjamin label sama dengan acuan lama.
