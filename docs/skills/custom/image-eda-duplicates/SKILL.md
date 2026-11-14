---
name: image-eda-duplicates
description: Membangun manifest citra, audit kualitas dan distribusi, serta kelompok duplikat atau satu sumber sebelum validasi klasifikasi gambar GELAR RASA.
---

# EDA dan duplikat

Gunakan [kontrak data](../../references/PROJECT_CONTEXT.md). Enumerasi train asli dari manifest dan test dari ID sample. **Jangan rglob train yang sekaligus memuat `train/label/`**. Verifikasi setiap ID mempunyai satu file, semua ekstensi tercakup, gambar dapat didekode, dan format output tetap tiga digit untuk test.

Catat dimensi, aspect ratio, mode, EXIF orientation, byte hash, pixel hash setelah decode/orientasi yang terdefinisi, blur/exposure sebagai indikator, serta source_type hasil pemeriksaan visual. Jangan menyamakan indikator blur dengan kualitas label. Previews bukan pengganti sumber training.

Untuk grouping: SHA-256 exact → kandidat perceptual hash (termasuk flip jika relevan) → pencarian tetangga embedding opsional → konfirmasi visual. Bedakan identical, same-photo transform, same-object session, semantic-similar, dan unrelated. Tetangga “sama-sama botol” bukan duplikat. Hindari penggabungan rantai besar hanya karena pasangan jarak lemah; tinjau komponen sebelum menetapkan group_id.

Train 0848/1717 sudah exact duplicate; ini titik awal, bukan audit lengkap. Simpan pasangan dan bukti tanpa menghapus sumber. Untuk train–test overlap, laporkan coverage; jangan menganggap kedekatan visual sebagai label test atau memakai label sample.

Keluaran tabel/hash/kemiripan: `docs/agent-notes/`; gambar dan galeri notebook di `final-submit/reports/`. Uji distribusi sumber dan kelas per group sebelum membuat fold. Gunakan [praktik pemisahan sumber](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/) sebagai dasar evaluasi.
