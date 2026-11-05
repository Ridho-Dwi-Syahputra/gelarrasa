# Struktur proyek GELAR RASA

Mengambil pola `eksperimen/skrip`, `output/folds`, `output/oof`, `output/submission`, dan `final` dari LOGIKA-UI. Mengambil `notebook`, `data/processed`, audit citra, dan `reports` dari INFEST-USK. Ini struktur persiapan; folder hasil kosong tidak berarti training sudah dijalankan.

```text
GELAR-RASA/
├── AGENTS.md                         petunjuk singkat agent proyek
├── README.md                         pintu masuk dan status
├── configs/
│   └── data_contract.json            kelas, metrik, input dan schema
├── data/
│   ├── Dataset DSC GelarRasa 2026/    data asli, lokasi dipertahankan
│   │   ├── train/                    1.920 sumber asli
│   │   │   └── label/<8 kelas>/      arsip anotasi lama; dikecualikan dari jalur kompetisi
│   │   └── test/                     480 sumber test
│   ├── SampleSubmission_DSC.csv      format/ID, bukan label test
│   └── processed/                    turunan preprocessing versi eksplisit
├── docs/
│   ├── materi/                       PDF webinar asli
│   ├── rules/                        handbook, aturan dan sources/ Q&A/template
│   ├── secret/                       arsip anotasi visual test lama
│   ├── train_label_review/           label, bukti, manifest, riwayat
│   ├── skills/
│   │   ├── README.md                 indeks kemampuan
│   │   ├── custom/<nama>/SKILL.md    22 skill khusus proyek
│   │   ├── vendor/<repo>/<skill>/   17 paket snapshot, 240 file
│   │   ├── references/              materi, riset, rubrik, rencana audit, kontrak
│   │   ├── sources/                 metadata sumber, lisensi, checksum
│   │   └── tools/                   pengunduh snapshot dan validator lokal
│   ├── eksperimen.md                 ledger percobaan dan status
│   └── STRUKTUR_PROYEK.md
├── eksperimen/
│   ├── skrip/                        implementasi eksperimen bernomor
│   ├── konfigurasi/                  config tiap run, bukan hyperparameter tersembunyi
│   └── cache/                        cache dapat dibangun ulang
├── notebook/                         notebook utama bertahap, belum dibuat
├── output/
│   ├── manifests/                    manifest training/inferensi
│   ├── groups/                       pasangan kemiripan dan group_id
│   ├── folds/                        fold yang dikunci dan versinya
│   ├── embeddings/                   fitur dengan metadata provenance
│   ├── checkpoints/                  best/resume checkpoint per run
│   ├── oof/                          probabilitas validation per ID
│   ├── test_pred/                    probabilitas test per ID
│   └── submission/                   CSV kandidat dan manifestnya
├── reports/
│   ├── eda/                          profil dan gambar audit
│   ├── errors/                       galeri error/ambigu
│   └── figures/                      visualisasi laporan
└── final/                            paket final yang telah diuji
```

## Konvensi penyimpanan

Mulai dari [aturan aktif](skills/references/RULES_AND_RUBRIC.md) dan [pemilih skill](skills/custom/gelar-rasa-skill-router/SKILL.md). Tabel integritas/profil/galeri teknis disimpan di `reports/eda/` ketika dijalankan. Turunan transform di `data/processed/<version>/` dengan hubungan ke raw. Label aktif hasil metode peserta terpisah dari arsip manual. [Kontrak artefak](skills/references/ARTIFACT_CONTRACTS.md) menentukan metadata sumber, cluster, fold, fitur, dan probabilitas.

Run: `expNNN_deskripsi`, misalnya `exp001_hog_color_svm`; script: `001_hog_color_svm.py`. Metadata run menyebut label/fold version, hash input, parameter, seed, perangkat, dan waktu. Refit seluruh train dibedakan dari model fold.

Gunakan satu lokasi kanonis tiap artefak. Tidak dibuat folder `submit/`, `submission/`, dan `docs/submit/` tambahan; kandidat baru berada di `output/submission/`. Arsip `docs/secret` yang sudah ada tetap riwayat lama. Root data resmi tidak dipindah ke `data/raw/` agar path pengguna tidak rusak.

Notebook utama belum dibuat: saat tahap implementasi, gunakan satu notebook yang memanggil script/modul teruji. Jangan menyalin OCR aksara, sampler berbasis test, atau aturan kategori INFEST ke limbah tanpa hipotesis dan pengujian. Tidak menyalin bank eksperimen tabular LOGIKA, KS, atau batas aturan kompetisi lain.

[Panduan Kaggle](skills/custom/kaggle-gpu-workflow/SKILL.md) memetakan `DATA_ROOT` dan `WORK_ROOT` ke runtime tujuan, dengan manifest/label/fold yang sama. GPU hanya opsi eksekusi berikutnya; persiapan ini belum menghasilkan notebook atau sesi GPU. Snapshot vendor memuat subtree terpilih, bukan seluruh dependensi transitif; subskill di dalamnya tidak menambah jumlah paket snapshot.

Validator paket mendukung `--verify-images` untuk hash train asli/salinan dan `--no-write` untuk mencetak hasil tanpa mengganti `docs/skills/sources/validation.json`. Hash tidak membuktikan gambar dapat didecode atau labelnya benar; cakupan lengkap ada di [indeks skill](skills/README.md#pemeriksaan-paket).

Folder kosong diberi `.gitkeep` hanya agar kerangka ini dapat dipertahankan bila proyek kelak memakai Git. Tidak ada repositori/commit baru yang dibuat pada tahap ini.
