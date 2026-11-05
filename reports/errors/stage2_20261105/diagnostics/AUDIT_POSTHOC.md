# Audit post-hoc ketidaksepakatan tahap 1

Label arsip adalah pembanding retrospektif, bukan ground truth. Mapping alias tetap berasal dari train; test tidak dipetakan ulang. Tidak ada model/parameter dipilih melalui hasil ini, tidak ada koreksi kategori, dan artefak aktif tidak berubah.

## Cakupan

Seluruh 1.920 train dan 480 test dianalisis: 590 train dan 171 test berbeda dari alias cluster. HTML memuat semua 761 kasus. Tiga contact sheet memuat sampel low-margin, high-margin, dan kontrol yang sesuai; tidak menampilkan label semantik pada gambar.

## Kestabilan bukan kebenaran

- test: 171 berbeda; 69 margin<0,10; 1 tidak stabil antar-seed; 168 sepakat pada seluruh seed; 102 bahkan margin≥0,10 dan semua seed sepakat. Hanya 8 min-side<224 dan 1 padding>50%.
- train: 590 berbeda; 257 margin<0,10; 11 tidak stabil antar-seed; 570 sepakat pada seluruh seed; 333 bahkan margin≥0,10 dan semua seed sepakat. Hanya 79 min-side<224 dan 6 padding>50%.

Kasus dengan margin kuat dan seed konsisten menunjukkan bahwa hanya mengulang seed atau menghapus kasus rendah-confidence tidak mencakup seluruh ketidaksepakatan. Ini tidak membuktikan pihak mana salah. Ambang margin 0,10 merupakan irisan deskriptif tetap, bukan probabilitas benar yang terkalibrasi.

## Konsentrasi kasus

- test: cluster_03 dan cluster_06 mencakup 129/171 (75.4%) dari seluruh ketidaksepakatan.
- train: cluster_03 dan cluster_06 mencakup 335/590 (56.8%) dari seluruh ketidaksepakatan.

Cluster_03 memuat delapan kategori arsip pada train; alias tetapnya hanya mencakup 78/269 train dan 7/90 test. Ini menunjukkan satu nama untuk satu cluster belum cukup mewakili isi arsip; mengganti satu alias tidak menyelesaikan semua pencampuran. Tidak dilakukan remapping.

## Metadata dan pengelompokan pembanding

`metadata_global_associations.csv` membandingkan seluruh kasus berbeda versus sesuai per split. `metadata_within_cluster_associations.csv` mengulangi pembandingan dalam cluster dan split yang sama untuk mengurangi confounding komposisi cluster. Rank-biserial positif berarti fitur cenderung lebih besar pada kelompok berbeda. Median/IQR dan ukuran sampel tersedia; tidak menganggap asosiasi sebagai penyebab, tidak melakukan pemilihan model berdasarkan label arsip.

Median resolusi test sama-sama min-side 416 piksel. Arah perbedaan ketajaman tidak konsisten antar-split: median Laplacian variance pada train berbeda/sesuai adalah 835,80/741,58, sedangkan test 561,89/880,28. Jadi tidak ada bukti bahwa blur atau resolusi rendah menjadi penjelasan tunggal.

Perbandingan dalam cluster juga menunjukkan confounding: pada test cluster_06 (46 berbeda, 41 sesuai), median padding berbeda/sesuai 0/0,25 dan brightness 0,826/0,682. Pada train cluster_05 (35 berbeda, 181 sesuai), entropy 6,952/5,419. Temuan ini mendukung pemeriksaan latar/format pengambilan per cluster; tidak membuktikan penyebab atau membenarkan crop/normalisasi otomatis. Kelompok dengan sedikit kasus perlu dibaca hati-hati.

## Tetangga fitur

Setiap gambar dicari 10 tetangga train pada embedding ResNet50 L2 asli. Seluruh canonical-pixel-identik dikeluarkan, termasuk self-match. `nn10_cluster_cohesion` mengukur proporsi tetangga dengan ID cluster yang sama. `nn10_archive_agreement` hanya diagnostik kesesuaian arsip, bukan ketepatan model. Semua baris disimpan dalam audit terpisah.

Median cohesion pada kasus berbeda adalah 0,8 train dan 0,9 test; kontrol sesuai masing-masing 1,0. Median kesesuaian arsip dengan tetangga juga masih 0,8 pada kedua split berbeda. Artinya banyak lingkungan lokal masih kompak meskipun satu alias cluster tidak cocok dengan arsip. Ini bukan pengukuran akurasi dan tidak dipakai memilih embedding/parameter.

## Batas pemeriksaan citra

Contact sheet dipilih mekanis menurut cluster/split dan margin. Ukuran objek, latar, occlusion, atau bentuk dapat ditinjau sebagai kondisi teknis; gambar tidak diberi kategori baru oleh agent. Galeri tidak membuktikan ketepatan material dan tidak mengenali satu kesalahan public dari skor 0,9930.

## Berkas

- `all_disagreements.html`: seluruh kasus dengan filter split/cluster/ID.
- `posthoc_image_diagnostics.csv`: metadata, alias arsip, margin, seed, dan tetangga seluruh 2.400 gambar.
- `per_cluster_disagreement.csv` dan crosstab: distribusi ketidaksepakatan, bukan confusion matrix ground truth.
- `disagreement_low_margin.png`, `disagreement_high_margin.png`, `agreement_controls.png`: inspeksi teknis terpilih.
- `audit_summary.json`: provenance dan bukti bahwa artefak aktif tidak berubah.
