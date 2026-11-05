# Pilot self-training dua putaran ? 6 November 2026

## Apa yang benar-benar dilatih

Dua teacher dan empat head student dilatih di atas fitur DINOv2 frozen. Ini bukan fine-tuning backbone atau classifier kategori terverifikasi. Input training hanya fitur gambar, kelompok sumber, dan pseudo-group dari teacher; kode tidak membaca arsip referensi maupun nama tampilan. Label LLM bukan target, filter atau kriteria berhenti.

Teacher dibuat ulang hanya pada 1.538 train-fit. Sebanyak 381 heldout kelompok dan pixel-unik dipisahkan; tidak ada overlap. Ini development split yang pernah dipakai, sehingga bukan evaluasi final independen. Probe flip tersedia pada 169 fit dan 87 heldout dari 256 train acak yang sudah dipilih sebelumnya.

## Seleksi, cadangan, dan masuk kembali

Confidence>=0,95, margin>=0,5, konsensus sepuluh tetangga>=0,9 dan kesepakatan dua teacher menghasilkan 1.110 pseudo-seed fit. Sebanyak 428 contoh fit lain disimpan sebagai cadangan. Seluruh raw tetap ada.

Putaran pertama membandingkan kontrol seed cross-entropy + graph consistency dengan dua bobot konsistensi view. Putaran kedua menerima empat contoh cadangan yang kini memenuhi syarat confidence/margin, tetangga, kesepakatan dengan teacher dan flip. Penerimaan ini bersumber dari model; bukan karena cocok dengan arsip. Batas eksperimen dua putaran.

Student yang terpilih adalah kontrol seed-CE/graph pada epoch20. Putaran kedua tidak melampauinya. Baseline proyek tidak diganti; tidak ada label kategori yang dipromosikan.

## Hasil proxy development

| Ukuran | Teacher fit | Student terpilih |
|---|---:|---:|
| Kesepakatan tetangga heldout | 0,900262 | 0,901837 |
| Kestabilan flip heldout | 86/87 | 87/87 |
| Kelompok heldout terisi | 8 | 8 |
| Ukuran kelompok heldout terkecil | 32 | 32 |

Kenaikannya kecil. Ini konsistensi partisi, bukan Accuracy kategori. Probabilitas belum dikalibrasi; kumpulan yang konsisten masih dapat mencampur material. Backbone frozen dan subset view kecil membatasi makna hasil. Tidak ada klaim memperbaiki public/private leaderboard.

## Pembanding setelah model dibekukan

Checkpoint dipilih dan disimpan sebelum audit membaca arsip. Alignment nama hanya untuk tampilan dan berasal dari overlap prediksi pada bagian fit, bukan dari arsip. Pada cohort sama 1.346 gambar yang memiliki nama tunggal pada kedua model, kesesuaian terhadap arsip berubah 1.180?1.182: dua bergeser menuju referensi dan nol menjauh. Sebanyak 574 kasus campuran tidak dinilai tunggal. Angka ini tidak dapat langsung dibandingkan dengan hitungan audit baseline global karena cohort berbeda.

Kasus0046: teacher, student dan referensi sama-sama bertema paper; tidak ada override gambar. Referensi adalah arsip visual asisten, bukan ground truth resmi. Tidak ada pelatihan ulang atau pemilihan checkpoint berdasarkan hasil pembanding ini.

## Bukti dan artefak

- `plan.json`: parameter dan aturan pemilihan proxy.
- `development_metrics.csv`: teacher dan seluruh kondisi/epoch; `development_curves.jpg` disimpan lokal dan tertanam pada notebook.
- `summary.json`: status, ukuran data, hasil dan batas klaim.
- `frozen_student_validation.json`: reproduksi probabilitas 2.400 gambar, group overlap nol, serta audit setelah freeze.
- `source_snapshot_after_run.json`: hash kode yang terakhir berubah sebelum plan dan checkpoint hasil run.
- `output/groups/stage4_self_training_pilot_v1/`: checkpoint teacher, student putaran1, hasil terpilih, pseudo-seed/fold, prediksi dan pemeriksaan ulang cadangan.
- `data/processed/stage4_frozen_student_reference_audit_only/posthoc_reference_queue.csv`: pembanding privat setelah pembelajaran, tidak menjadi input training.

Pilot dijalankan pada sel terakhir `notebook/01_EDA_Preprocessing_Clustering.ipynb`. Seluruh notebook lulus10/10 dengan output tersimpan. Dataset, gambar terpisah, checkpoint dan embedding tetap lokal.

## Dasar metode

[Pseudo-labeling dan confirmation bias, Arazo et al.](https://arxiv.org/abs/1908.02983) menjelaskan risiko memperkuat pseudo-label salah. [FixMatch](https://arxiv.org/abs/2001.07685) menjadi rujukan confidence dan konsistensi view, tetapi metode aslinya semi-supervised dan bukan implementasi pilot ini. [SCAN](https://arxiv.org/abs/2005.12320) menjadi dasar konsep representasi dan clustering terpisah; kode ini hanya adaptasi head/graph, bukan reproduksi penuh paper.
