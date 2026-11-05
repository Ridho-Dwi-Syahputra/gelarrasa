---
name: gelar-rasa-skill-router
description: Memilih skill dan memeriksa aturan aktif pada setiap tugas GELAR-RASA, lalu mengarahkan pekerjaan ke panduan data, metode, evaluasi, laporan, atau pengumpulan yang relevan.
---

# Pemilih skill per tugas

Pada setiap prompt yang menyangkut GELAR-RASA, cek [indeks skill](../../README.md), [aturan aktif](../../references/RULES_AND_RUBRIC.md), dan status proyek. Jika sudah dibaca dalam konteks aktif dan tidak berubah, gunakan catatan yang tersedia; jangan membaca ulang seluruh vendor atau menjalankan pencarian web setiap prompt.

Tentukan hasil yang diminta dan batas tindakan dari percakapan. Pilih 1–3 skill paling relevan, tambah bila pekerjaan memang melintasi beberapa tahap. Permintaan dokumen tidak otomatis memulai training. Jangan membuat permintaan izin baru untuk pekerjaan yang sudah diotorisasi. Bila sumber baru mengubah aturan, perbarui konteks sebelum pekerjaan yang bergantung padanya.

| Tugas | Rute |
|---|---|
| Label/cluster | unsupervised-waste-labeling + waste-label-audit |
| EDA/preprocessing | image-eda-duplicates + image-preprocessing-audit; tambah image-domain-shift-audit jika membandingkan train/test |
| Pemodelan | skill metode terkait + grouped-image-validation; periksa batas LLM dalam aturan sebelum membuat model kompetisi |
| Evaluasi/perbaikan | image-error-analysis + grouped-image-validation |
| Laporan | competition-analysis-report + analytical-figures |
| Pemeriksaan PDF/notebook/CSV | report-package-audit + reproducible-image-submission |
| GPU/porting | kaggle-gpu-workflow |
| Riset metode | vision-research-review |

Cek prasyarat aktual, bukan sekadar file tersedia. Label visual asisten lama berstatus arsip, bukan label aktif. Jangan mengaktifkan CLIP/SigLIP/VLM, layanan inference eksternal, atau koreksi per gambar karena ada contoh di skill vendor. Aturan panitia adalah bukti batas kompetisi; bantuan konseptual, riset, dan pengaturan berkas tidak boleh dilaporkan sebagai model yang sudah dibuat peserta.

Untuk tugas substantif, sebutkan singkat skill yang digunakan pada pembaruan awal. Saat selesai laporkan artefak, verifikasi yang benar-benar dijalankan, dan keterbatasan yang relevan. Instruksi ini berlaku pada workspace/proyek yang memuatnya; jangan menjanjikan aktivasi di aplikasi atau workspace lain.
