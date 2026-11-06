# Hasil aktual dan keputusan - 6 November 2026

Sebanyak 18 fitting KMeans menghasilkan 24 kondisi assignment, ditambah dua kontrol:26 kondisi diperiksa. Fitur global serta patch mean/std diekstrak dari 2400 gambar menggunakan checkpoint yang sama. CLS hasil ekstraksi ulang cocok dengan cache lama, maksimum selisih 4,47e-8. Tiga uji geometri/penggabungan/bobot blok lulus. Sebanyak 24 model tersimpan direproduksi untuk seluruh 2400 assignment; 756 target galeri valid.

## Hasil pengujian yang tidak memakai arsip

- Split fit 1.538 dan development 381, tanpa overlap kelompok/pixel. Probe view 87 heldout. Split ini pernah digunakan untuk development; bukan final holdout independen.
- Kandidat terpilih tetap **current_student**, skor proxy 0.968946.
- Kandidat baru terdekat **hybrid_k32_ward8_s47**, skor 0.966736. Selisih kecil dan berasal dari proxy tanpa ground truth.
- Backbone tetap frozen pada ablation ini. Belum melatih ulang representasi gambar.

## Audit struktur setelah semua hasil dibekukan

Perbandingan privat terhadap arsip visual asisten pada 1.920 train, tanpa pemetaan nama kategori baru:

| Partisi tetap yang dibandingkan | ARI | Homogeneity | Completeness | Pairwise precision | Pairwise recall |
|---|---:|---:|---:|---:|---:|
| current_student | 0.6041 | 0.6919 | 0.6961 | 0.6460 | 0.6642 |
| global_k8_s11 | 0.4835 | 0.5999 | 0.6094 | 0.5335 | 0.5698 |
| global_k32_s11 | 0.2962 | 0.7934 | 0.4850 | 0.7610 | 0.2152 |
| hybrid_k8_s11 | 0.5241 | 0.6346 | 0.6419 | 0.5723 | 0.5999 |
| hybrid_k32_s11 | 0.3135 | 0.7946 | 0.4909 | 0.7734 | 0.2289 |
| hybrid_k32_ward8_s11 | 0.5275 | 0.6461 | 0.6563 | 0.5708 | 0.6092 |

Homogeneity lebih tinggi pada 32 prototipe disertai completeness dan pairwise recall yang jauh lebih rendah. Ini menunjukkan pemecahan kelas menjadi banyak kelompok; tidak boleh disimpulkan Accuracy delapan kelas naik. Hybrid delapan memperbaiki kontrol global-KMeans seed11 pada pembanding, tetapi tetap di bawah student lama. Audit tidak dipakai memilih ulang model atau seed.

## Apa yang sudah diketahui

1. Mengulang head dan menambah prototipe pada encoder yang sama belum menghasilkan pemisahan material yang lebih baik secara meyakinkan. Hipotesis fitur lokal berguna sebagian, tetapi belum menjadi solusi akhir.
2. Ward menyatukan prototipe berdasarkan geometri, tanpa pengetahuan taksonomi material. Delapan hasil merge tidak otomatis merupakan delapan kategori soal. Tidak ada nama kategori baru atau koreksi per gambar yang dibuat.
3. Arah berikut yang belum dicoba adalah adaptasi **representasi dan head bersama**, memakai pseudo-label algoritmis yang konsisten dan augmentasi yang mempertahankan objek/material. Ini tahap berbeda dari 18 fitting ini, selaras dengan tahap joint training SPICE. Fine-tuning juga dapat memperkuat kesalahan pseudo-label, sehingga perlu pembanding kontrol serta saringan model dan batas iterasi.
4. Penamaan/penyatuan kelompok menjadi delapan material memerlukan pemetaan semantik oleh tim. Memaksa pencocokan menggunakan arsip LLM sebagai loss, target, filter atau pemilih checkpoint akan menjadi supervisi dari arsip; perbandingan yang dilakukan di sini tetap post-hoc.

**Keputusan:** belum mengganti student, folder bernama aktif, kontrak label atau notebook utama. Hitungan student sebelumnya tetap 1.186 sama, 164 berbeda, 570 campuran terhadap arsip train. Tidak ada klaim Accuracy 100% atau submission baru. Semua gambar asli/cadangan tetap ada.

Galeri 32 prototipe untuk pemeriksaan ada pada `galleries/prototypes32.html`. Catatan penelitian dan protokol lengkap ada pada `RESEARCH_AND_PROTOCOL.md`; source akses/batas replikasi pada `research_sources.json`. Ini bukti eksperimen yang tidak berhasil menggantikan kontrol, bukan bukti bahwa clustering material mustahil.
