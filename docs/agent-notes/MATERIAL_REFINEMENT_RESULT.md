# Hasil eksperimen material dan source group ? 7 November 2026

**Perbedaan terhadap review tim turun dari 48 menjadi 45.** Dari 126 keputusan tegas, 81 kini sesuai. Tiga kasus yang pulih adalah 0432 ? glass, 1510 ? plastic, dan 1817 ? paper. Tidak ada keputusan yang sebelumnya sesuai menjadi berbeda. Jumlah acuan manusia tetap 34, maksimal lima per kelas; tahap ini tidak memakai tambahan jatah.

## Perubahan aktif

| Train | Sebelumnya | Sekarang | Status audit |
|---|---|---|---|
| 0158 | paper | plastic | Belum tercakup review tegas |
| 0432 | cardboard | glass | Sesuai review tim |
| 0673 | cardboard | paper | Belum tercakup review tegas |
| 1510 | foodwaste | plastic | Sesuai review tim |
| 1817 | textile | paper | Sesuai review tim |

Lima perubahan berasal dari head neural lokal. Dua yang belum diperiksa tim tetap berstatus prediksi, bukan klaim benar. Total perubahan sejak snapshot 150 perbedaan lama adalah 73 kategori. Selain 45 perbedaan tegas, masih ada 20 keputusan tentatif, 0676/1650 yang bersyarat, serta 1662/1885 yang ditahan. Angka-angka ini tidak boleh dijumlahkan sebagai kesalahan terverifikasi.

## Eksperimen yang dijalankan

1. **Material khusus:** MINC GoogLeNet non-VLM mengekstrak fitur dari 1.904 foto train dalam tiga tampilan. Head material/tekstur dan hybrid lokal hanya memulihkan 11 dari 34 acuan saat source group-nya ditahan. Guard penerimaan 14 tidak terpenuhi; usulannya tidak diterapkan.
2. **Loss tahan noise:** dua bank fitur dipakai oleh head neural LayerNorm?Linear256?GELU?Dropout?Linear8. Target otomatis berasal dari inti konsensus dan snapshot sebelum acuan manusia pertama. Acuan manusia memakai cross-entropy; target otomatis dibandingkan memakai CE dan GCE dengan q=0,7. Pada lima fold source group geometri yang sama, CE memulihkan 14/34 dan GCE 15/34. Lima perubahan melewati kesepakatan kedua bank, probabilitas ?0,85, margin ?0,4, dan perlindungan terhadap refinement sebelumnya. Probabilitas belum terkalibrasi.
3. **Self-training:** satu putaran dilakukan terpisah dalam setiap outer fold. Pseudo-label hanya dibuat pada sisi fit, tanpa query group atau label query. Teacher dan student masing-masing memulihkan 14/34; guard 15 gagal, sehingga tidak ada perubahan dari jalur ini. Stream dropout teacher regenerasi berbeda dari percobaan sebelumnya dan tidak diklaim bitwise sama.

Diagnostik 34 acuan digunakan untuk pengembangan dan pemilihan objective, bukan Accuracy independen. Kesesuaian 81/126 berasal dari audit terhadap review pengguna setelah prediksi dibekukan, bukan ground truth panitia. Seluruh 126 review maupun 45 kasus tersisa tidak menjadi target atau filter training. Keputusan publikasi direkam sebelum audit privat.

Sumber primer: [MINC, Bell dkk.](https://opensurfaces.cs.cornell.edu/publications/minc/) dan [GCE, Zhang & Sabuncu, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf). Detail sumber, bobot dan implementasi ada pada [kartu riset](MATERIAL_TEXTURE_RESEARCH.md). Tidak menambah dataset gambar eksternal atau memakai VLM/API inference. Bobot dijalankan lokal, dengan asal dan checksum dicatat. Parity numerik terhadap engine Caffe dan skor benchmark MINC belum direplikasi.

## Grouping sumber foto

SIFT/RANSAC memeriksa 8.047 pasangan dan menghasilkan 116 hubungan kuat yang mencakup 183 foto. Kategori dalam komponennya sudah konsisten, sehingga jalur ini tidak memberi relabel langsung. Namun, 90 hubungan menghubungkan kelompok klasifier lama yang berbeda dan 70 melintasi fold lama. Semua hubungan kuat kini berada pada group dan partisi yang sama.

Untuk 1.904 foto eligible, union kelompok klasifier berubah dari 1.870 menjadi 1.777. Grouping input raw memiliki denominator 1.920 dan berubah dari 1.869 menjadi 1.792. Source group adalah pembagian sumber foto; kategori tetap delapan. Test dan dua foto yang ditahan tidak dipakai untuk fitting geometri.

## Notebook dan data

[Notebook tunggal](../../final-submit/DSC2026_Notebook.ipynb) lulus Run All dari processed kosong: 13 sel kode, nol error dan 16 visual. Sebanyak 4.793 hash gambar diperiksa. Train tetap berisi 1.913 sampel dalam delapan folder, dengan 1.911 eligible; test 480 dan native crop tujuh. Semua 116 hubungan geometri diperiksa berada pada group/partisi yang sama. Piksel test dan sumber asli tidak berubah. Foto 1662/1885 tetap bukan acuan atau referensi crop.

Run All preprocessing mematerialisasikan snapshot assignment dan menghitung kategori crop; belum mengulangi seluruh fit assignment heads. Classifier final, validasi independen dan CSV submission belum dibuat. **Sebanyak 45 kasus belum terselesaikan secara andal.** Eksperimen ini tidak membuktikan semua alternatif algoritma mustahil, dan bukan alasan untuk menyalin label arsip atau menurunkan batas penerimaan agar tampak cocok.

Bukti: [validasi tahap](json/material_followup_delivery.json), [publikasi](json/bounded_anchor_publication.json), [keputusan sebelum audit](json/material_publication_decision.json), [Run All](execution.json), dan [validasi koleksi](skills_validation.json). [Galeri aktif 45 kasus](remaining_visual_review/latest_review.html) serta [tabel kasus tersisa](table-csv/remaining_current_review_differences.csv) hanya untuk audit.
