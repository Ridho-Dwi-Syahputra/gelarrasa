---
name: reproducible-image-submission
description: Mengemas pipeline klasifikasi gambar yang dapat dijalankan ulang dan memvalidasi CSV GELAR RASA terhadap ID sample, mapping kelas, checkpoint, dan riwayat submission.
---

# Reproduksi dan submission

Baca [kontrak](../../references/PROJECT_CONTEXT.md) dan `docs/agent-notes/project_contract.json`. Notebook utama `final-submit/DSC2026_Notebook.ipynb` memuat kode alur aktif; cell berurutan dari input → EDA → preprocessing → training/evaluasi → inferensi. Hindari ketergantungan diam-diam pada variabel notebook atau cache dari run lama.

Input path dapat dikonfigurasi; semua path proyek relatif terhadap root. Data mentah tidak dipindah. Sertakan manifest, label hasil metode peserta beserta assignment/mapping cluster, fold, config, dependency versions, seed, sumber/checksum bobot, dan runtime. Anotasi asisten lama tidak digunakan. Bedakan “script tersedia”, “smoke test lulus”, dan “Run All selesai”; jangan klaim rerunnable tanpa bukti eksekusi relevan.

Tentukan strategi final sebelum test: ensemble semua checkpoint fold atau full-train refit dengan epoch/step schedule ditentukan dari CV. Jangan memilih hanya fold tertinggi atau menggunakan test untuk early stopping. Simpan kebijakan, unit schedule, dan checkpoint IDs sesuai [kontrak artefak](../../references/ARTIFACT_CONTRACTS.md).

Inferensi membaca sample dengan `image_id` string. Join prediksi lewat ID, kemudian susun ulang persis sample; jangan mengandalkan urutan directory listing. Assert 480 unik, dua kolom tepat, tidak ada missing/extra ID, tidak ada NaN, dan label dalam delapan kelas. Periksa probabilitas delapan kolom sebelum argmax. `drop_last=False` saat inferensi.

Nama kandidat internal boleh `final-submit/output/submission_NNN.csv`; paket resmi memakai `DSC2026_Submission_NamaTim_NamaKetua.csv`, notebook/laporan sesuai [aturan](../../references/RULES_AND_RUBRIC.md). Simpan sha256 dan label-vector fingerprint; catat checkpoint asal dan evaluasi nyata. Public score tetap null sampai pengguna/API resmi memberi hasil.

Submission_99 adalah anotasi visual, tidak boleh disebut prediksi model terlatih. Paket final harus mengungkap asal anotasi dan model. Keluaran: CSV, manifest submission, laporan pemeriksaan, notebook/config untuk reproduksi. Upload kompetisi merupakan tindakan terpisah dari membuat file.
