# Kontrak artefak eksperimen citra

Gunakan bersama `docs/agent-notes/project_contract.json`. Spesifikasi ini untuk implementasi berikutnya; belum ada fold, checkpoint, atau OOF model yang dihasilkan. Lokasi relatif dihitung dari root proyek. Hash merekam isi, bukan sekadar nama file.

Revisi aturan 5 November: label aktif harus berasal dari metode peserta dan mapping cluster tim yang terdokumentasi; anotasi visual asisten lama tidak dipakai. Sertakan `cluster_run_id`, `assignment_sha256`, `cluster_mapping_sha256` dan populasi yang dipakai fit pembentuk label. Jika target dihasilkan dari seluruh train sebelum split, catat ketergantungan/transductive tersebut; jangan menyebut agreement sebagai Accuracy semantik terhadap ground truth independen. Baca [aturan aktif](RULES_AND_RUBRIC.md).

## Dataset, label, dan kelompok

- Manifest: `image_id, filename, relative_path, sha256, width, height, source_type`. ID selalu string; train empat digit, test tiga digit. Satu ID satu sumber. Salinan `train/label` tidak dihitung sebagai sumber tambahan.
- Versi label: `label_version`, SHA-256 CSV label dan audit, waktu, sumber anotasi, serta tautan ledger koreksi. Membuat versi baru tidak menghapus versi lama.
- Tabel kelompok: `image_id, group_id`. Pasangan pendukung disimpan terpisah: `id_a, id_b, relation, evidence, review_status`. Hubungan semantic-similar saja tidak cukup untuk menyatukan sumber.
- Fold: `image_id, group_id, repeat, fold`. Metadata: `fold_version, label_version, manifest_sha256, groups_sha256, splitter, seed, n_splits`. Tiap repeat harus memuat setiap ID sekali sebagai validation; tidak ada kelompok yang terpotong.

Perubahan label bisa mengubah keseimbangan fold. Periksa ulang, tetapi jangan otomatis mengganti fold hanya demi meningkatkan skor. Rekam alasan bila fold version berubah.

## Prediksi dan embedding

Prediksi CSV: `image_id,p_glass,p_paper,p_electronic,p_metal,p_plastic,p_cardboard,p_textile,p_foodwaste`. Metadata sidecar JSON mencatat `run_id, prediction_kind, label_version, fold_version, class_order, row_ids_sha256, checkpoint_ids, config_sha256, transform_id`. OOF juga menyimpan fold/group tiap ID; test tidak diberi label target dari sample.

Periksa nilai finite dalam [0,1], delapan kolom unik, jumlah probabilitas per baris mendekati 1 (toleransi numerik dicatat), dan tidak ada ID hilang/berlebih/berulang. Berkas logits harus diberi jenis berbeda dan dikonversi dengan softmax sebelum memakai kontrak probabilitas; keluaran SVM yang belum terkalibrasi bukan probabilitas.

Jangan menebak urutan kelas dari posisi kategori dalam sample. Class mapping harus berasal dari model atau metadata training. Contoh mapping alfabetis `cardboard,electronic,foodwaste,glass,metal,paper,plastic,textile` menjadi urutan kanonis memakai indeks `[3,5,1,4,6,0,7,2]`. Terapkan reorder berdasarkan nama, bukan menghafal angka untuk semua model.

Embedding cache merekam: encoder/checkpoint hash, transform, dimensi, dtype, urutan ID, train_ids/group_ids yang digunakan melatih encoder, dan `frozen_external` atau `supervised_fold`. Kedua sisi suatu fold (train dan validation) harus memakai **encoder fold yang sama**. Embedding dari beberapa encoder fine-tuned berbeda tidak otomatis berbagi sistem koordinat; jangan menumpuknya menjadi satu matriks global untuk melatih classifier tanpa desain yang membuktikan kesetaraannya.

## Evaluasi bertingkat

Repeat CV pada gambar yang sama mengukur sensitivitas split/seed; itu bukan holdout baru setelah gambar tersebut berkali-kali dipakai memilih metode. Nested CV menilai prosedur seleksi di dalam tiap outer-train. Jika pilihan keluarga/aturan label sudah dioptimalkan dari seluruh data, ungkap bahwa bias desain sebelumnya tidak otomatis hilang dengan mengganti splitter.

Untuk stacking atau memilih bobot yang membutuhkan skor independen:

1. Pisahkan outer-train dan outer-validation per kelompok.
2. Hanya di outer-train, buat inner-OOF model dasar; fit seluruh preprocessing dan encoder yang supervised di inner-train masing-masing.
3. Pelajari meta-model/bobot/kalibrasi dari inner-OOF tersebut.
4. Dengan jadwal yang ditentukan tanpa outer-validation, latih ulang model dasar di outer-train (atau gunakan kebijakan ensemble inner yang sudah dipilih), prediksi outer-validation, lalu terapkan meta-model.
5. Gabungkan hasil seluruh outer fold. Seluruh tahap memiliki provenance.

Cross-validating meta-model di atas satu matriks OOF global belum cukup: baris meta-training dapat dihasilkan oleh base model yang sudah melihat label outer-validation. CV internal `StackingClassifier` membangun data meta-training; bukan skor final independen dengan sendirinya. [Dokumentasi stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), [nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Dua OOF dari fold berbeda masih dapat dirata-ratakan dengan bobot tetap yang ditentukan sebelumnya bila setiap prediksi benar-benar tidak melihat label/kelompok barisnya dan seluruh prosedur model tetap. Jangan mengubah nama fold agar tampak sama. Catat provenance berbeda dan keterbatasan perbandingan; tuning bobot/meta-model memerlukan protokol bertingkat di atas.

## Strategi final yang dikunci

Pilih dan catat salah satu sebelum inferensi test: **ensemble checkpoint seluruh fold** dengan bobot yang ditetapkan, atau **refit seluruh train** memakai jadwal epoch/optimizer-step dan preprocessing yang ditentukan dari CV. Penambahan ukuran train mengubah jumlah step per epoch; nyatakan satuan schedule. Jangan memilih satu fold karena kebetulan skornya paling tinggi. Test tidak dipakai untuk early stopping atau menentukan kapan refit dihentikan.

Kandidat final mencatat strategi, checkpoint IDs/hash, class mapping, manifest dan label version, prediksi sumber, serta CSV hash. Simpan label-vector fingerprint sesudah penyelarasan ID agar file yang namanya berbeda tetapi prediksinya sama terdeteksi.
