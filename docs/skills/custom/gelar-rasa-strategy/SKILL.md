---
name: gelar-rasa-strategy
description: Menentukan eksperimen klasifikasi limbah GELAR RASA berikutnya dari bukti data, label, validasi, dan hasil model; gunakan untuk perencanaan atau diagnosis stagnasi skor.
---

# Strategi berbasis bukti

Baca [konteks](../../references/PROJECT_CONTEXT.md), [aturan dan rubrik](../../references/RULES_AND_RUBRIC.md), serta [playbook](../../references/EXPERIMENT_PLAYBOOK.md). Bedakan fakta yang sudah diukur, dugaan penyebab, metode kandidat, serta hasil yang belum tersedia. Accuracy adalah metrik prediksi; penilaian keseluruhan mencakup Kaggle 70, laporan 20, notebook 10. Kualitas setiap kelas dan reproduksibilitas tetap dicatat. Label visual asisten lama tidak menjadi dasar eksperimen; panduan metode ditujukan kepada peserta dan tidak mengizinkan LLM membuat model atau mengklasifikasikan data kompetisi.

Mulai dari status aktual dalam `docs/eksperimen.md`, bukan nama file “best”. Periksa kesetaraan label_version, fold_version, ID prediksi, dan urutan kelas. Jika belum ada OOF yang dapat dipercaya, kerjakan kontrak/label/grouping sebelum menafsirkan selisih model.

Untuk setiap keputusan, tulis ringkas: masalah teramati → penjelasan alternatif → pemeriksaan pembeda → satu perubahan → ukuran keberhasilan. Prioritaskan koreksi pipeline dan informasi citra yang hilang sebelum memperbanyak seed/model. Pilih baseline, transfer learning, atau hybrid mengikuti bukti residual. Estimasikan biaya satu run dari pilot sebelum sweep.

Kenaikan public tidak dijanjikan dari kenaikan CV. Jangan menjelaskan metode peserta lain hanya berdasarkan skor. Peringkat saat Accuracy sama bergantung aturan tie-break/final yang perlu diverifikasi.

Keluaran: satu kartu eksperimen terukur, lokasi artefak, keputusan lanjut/tolak, dan batas interpretasi. Selesaikan lingkup yang diminta; untuk permintaan pengumpulan skill, jangan mulai training otomatis.
