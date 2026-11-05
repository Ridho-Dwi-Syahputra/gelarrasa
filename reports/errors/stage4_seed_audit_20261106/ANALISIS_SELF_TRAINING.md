# Audit persiapan self-training ? 6 November 2026

**Pembaruan:** audit persiapan ini kemudian diikuti pilot dua putaran dengan dua teacher dan empat head student. Hasil actual ada di `reports/errors/stage4_self_training_20261106/PROTOCOL_AND_RESULTS.md`; arsip tetap tidak menjadi input training.

## Ide pengguna dan batas penggunaan referensi

Pengguna mengusulkan memakai arsip label visual sebagai acuan, memilih data yang tampak bersih, melatih ulang model, lalu mengulangi sampai hasilnya mendekati arsip. Secara teknis ini pembelajaran dari weak labels/distillation, bukan pemulihan label tanpa supervisi. Jika arsip menentukan filter, target, atau kriteria berhenti, label arsip tetap menjadi sumber supervisi. Aturan kompetisi yang diberikan pengguna melarang klasifikasi/model memakai LLM/VLM; proses tersebut tidak boleh disebut jalur kompetisi tanpa LLM hanya karena notebook tidak memanggil API.

Perbandingan arsip yang diminta pengguna dilakukan post-hoc/read-only, terpisah dari notebook dan kode seleksi. `read_only_reference_comparison.json` pada folder audit referensi mencatat SHA assignment sebelum/sesudah sama. Nilai perbandingan bukan Accuracy ground truth dan tidak dipakai memilih model atau ambang.

## Kasus 0046

- Folder tinjauan tahap 1 menempatkannya pada alias electronic.
- Partisi lama memiliki margin centroid 0,037006546 dan flag `small_centroid_margin`, meskipun seed agreement 1,0.
- Head DINO tahap 2 yang sudah ada sebelum permintaan terbaru menempatkannya pada kelompok bertema paper, dengan probabilitas head belum dikalibrasi 0,89370793.
- Arsip dan rujukan pengguna menyebut paper. Tidak dilakukan pemindahan atau koreksi manual pada gambar/assignment.

Ini menunjukkan folder lama bukan gambaran hasil terbaru, dan kesepakatan seed saja belum menandakan target kategori benar. Dugaan bahwa fitur pretrained lama lebih dipengaruhi isi foto di cetakan daripada material kertas masih hipotesis; belum dibuktikan dengan ablation/attribution.

## Audit kandidat inti tanpa membaca arsip

Kode: `eksperimen/skrip/stage4_seed_audit.py`.

Filter yang dinyatakan sebelum audit: probabilitas head >=0,95; margin >=0,5; minimal 90% dari sepuluh tetangga pixel-unik sekelompok; dua partisi model selaras setelah alignment pada train; satu baris per pixel-unik. Tidak ada nama kelas atau arsip label yang dibaca oleh kode seleksi.

Hasil:

- 1.920 baris train, 1.919 pixel-unik.
- 1.233 kandidat inti model.
- 686 contoh unik tetap sebagai cadangan ambigu.
- Satu baris duplikat dipertahankan, tidak dijadikan seed tambahan.
- Tidak menghapus gambar asli, menetapkan label kategori, melatih student, atau mempromosikan kandidat ke training.

Manifest lokal: `data/processed/stage4_seed_audit_v1/seed_and_reserve_manifest.csv`, `seed_candidates.csv`, `reserve_preserved.csv`. Semuanya membawa kelompok/fold induk. Confidence adalah skor model yang belum dikalibrasi, bukan peluang label benar. Kelompok yang konsisten masih dapat mencampur material.

Teacher saat ini pernah direfit pada semua train unik. Karena itu audit ini bukan hasil holdout independen. Eksperimen student berikutnya harus membangun teacher, filter dan student hanya di bagian fit, dengan kelompok holdout dibekukan sebelum seleksi. Data ambigu tetap menjadi reservoir untuk evaluasi konsistensi, bukan dihapus agar skor tampak naik. Keberhasilan tidak dinilai dari kemiripan dengan arsip LLM pada jalur lomba.

## Referensi primer dan implikasi

- [Arazo et al., Pseudo-Labeling and Confirmation Bias](https://arxiv.org/abs/1908.02983): pseudo-label naif dapat memperkuat label awal yang salah. Ini alasan iterasi tidak boleh berhenti hanya karena model makin sepakat dengan dirinya sendiri.
- [FixMatch](https://arxiv.org/abs/2001.07685): confidence dan konsistensi weak/strong views adalah mekanisme seleksi/pelatihan. Metode aslinya semi-supervised, sehingga tidak boleh menganggap label awal yang sah otomatis tersedia pada dataset tanpa label ini.
- [SCAN](https://arxiv.org/abs/2005.12320): representasi self-supervised dan clustering dipisahkan untuk pengelompokan tanpa ground truth. Kode tahap 2 proyek adalah adaptasi head pada backbone frozen, bukan reproduksi penuh metode tersebut.

Pengguna kemudian memperjelas arsip dipakai sebagai pembanding dan kasus berbeda disimpan untuk ditinjau lagi. Ini diimplementasikan sebagai antrean audit terpisah pada `data/processed/stage4_reference_audit_only/`; antrean tidak mengubah mask seed/reserve. `review_queue_validation.json` membuktikan hash assignment, mask seed dan kontrak aktif tetap sama. Tampilan hasil terbaru tersedia di `data/processed/stage3_named_display_review/`, dengan 0046 pada folder paper. Audit persiapan ini sendiri tidak melatih student atau mempromosikan label kategori. Pilot berikutnya benar-benar melatih empat head student dalam dua putaran; hasil dan batas evaluasinya dicatat terpisah pada `reports/errors/stage4_self_training_20261106/PROTOCOL_AND_RESULTS.md`.
