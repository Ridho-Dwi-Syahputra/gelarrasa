---
name: vision-training-diagnostics
description: Menguji kebenaran loop training citra, memilih percobaan berdasarkan kurva dan error, serta menyiapkan run hemat GPU yang dapat dilanjutkan sebelum memperbesar sweep.
---

# Diagnostik training sebelum sweep

Baca [CNN transfer learning](../cnn-transfer-learning/SKILL.md) dan [kontrak artefak](../../references/ARTIFACT_CONTRACTS.md). Terapkan setelah tahap training diminta; skill preparation-only tidak memulai latihan.

Periksa satu batch: ID, citra setelah inverse-normalization, target integer dalam 0–7, mapping kelas, bentuk logits `(N,8)`, loss finite, gradien head, serta mode train/eval. Saat backbone frozen, cek gradien dan running statistics, bukan hanya flag parameter. Jangan menyamakan training Accuracy pada target MixUp dengan Accuracy label keras biasa.

Untuk diagnosis implementasi, uji apakah model dapat mempelajari subset kecil train tanpa augmentasi berat. Jika tidak, cek mapping, skala piksel, loss/logits, LR dan optimizer sebelum mengganti model. Ini sanity check in-sample, **bukan** ukuran generalisasi atau dasar memilih submission.

Pisahkan gagal training, underfit, overfit, dan masalah label berdasarkan kurva, gradien, serta galeri. Tuning bertahap: head/freeze → LR backbone → resolusi/augmentasi → loss/regularisasi → seed. Kandidat setting bukan resep terbaik yang telah terbukti. Learning-rate finder, EMA, class weights, focal loss, serta sampler berbobot adalah ablation terpisah. Hindari class weight dan oversampling bersamaan tanpa alasan; Accuracy utama tidak identik dengan mengoptimalkan recall seimbang.

Ukur images/sec, peak VRAM, waktu epoch dan validation, serta biaya I/O sebelum memperbanyak run. Bila batch diperkecil atau gradient accumulation berubah, catat effective batch, jumlah optimizer steps, dan scheduler steps. Checkpoint harus memuat semua state yang diperlukan; bandingkan resume dengan run pendek kontinu dengan toleransi numerik terdokumentasi.

Keluaran: sanity-check report, kurva, config, checkpoint, dan satu hipotesis untuk perbaikan. Jangan menamai smoke test sebagai training tuntas atau menyatakan hiperparameter “terbaik” tanpa pembanding setara.
