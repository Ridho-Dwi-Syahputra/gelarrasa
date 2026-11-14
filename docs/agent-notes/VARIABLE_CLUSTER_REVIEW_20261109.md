# Uji cluster dengan jumlah fleksibel — 9 November 2026

## Tujuan

Menguji apakah kelompok train yang jumlahnya tidak dipatok delapan dapat dibentuk di sekitar contoh yang telah dikonfirmasi peserta, lalu menurunkan label kelompok dari seed peserta. Arsip anotasi visual lama dan tabel review 126 kasus tidak dimasukkan sebagai target, fitur, atau dasar pemilihan parameter. Hanya 35 sumber unik dalam `data/processed/manual_seeds.csv` dipakai sebagai seed berlabel; batas lima per kelas dipatuhi. Test tidak ikut fit.

## Hasil yang diuji

**MiniBatchKMeans dengan K=32, 64, 128, 256.** Embedding DINOv2-small train full-frame dinormalisasi L2. Dalam setiap fold GroupKFold, seluruh grup sumber milik seed validasi dikeluarkan dari fit clustering. Cluster hanya dipetakan bila setidaknya satu seed train memberi dukungan dan kemurnian suara seed minimal 80%.

| K | Cakupan seed validasi | Tepat pada seed yang tercakup |
|---:|---:|---:|
| 32 | 47,1% | 20,3% |
| 64 | 42,9% | 35,9% |
| 128 | 47,1% | 42,3% |
| 256 | 37,1% | 61,5% |

K=256 memberi angka tertinggi pada cakupan kecil, tetapi hanya sekitar 13 dari 35 seed yang memperoleh prediksi cluster; hasilnya tidak cukup kuat untuk menyebarkan kelas ke ratusan foto. Hasil antarseed/run juga bervariasi.

**Graf kNN cosine dan propagasi label harmonik**, dengan k=5/10/20/40 dan alpha 0,3/0,6/0,9. Setiap fold membangun graf hanya dari train fold; grup seed validasi tidak berada di graf. Hasil terbaik akurasi seed 34,3% (12/35), macro recall 33,3%. Ambang skor 0,60 hanya meloloskan satu seed pada konfigurasi terbaik; skor tersebut tidak terkalibrasi sebagai confidence yang dapat dipakai untuk pseudo-label massal.

## Keputusan

Kedua keluarga eksperimen **ditolak dan tidak mengubah** `data/processed`, snapshot input, manifest, notebook, atau submission. Hitungan audit tetap 88 perbedaan terhadap referensi setelah 13 pengecualian peserta; 101 terhadap arsip mentah. Angka ini adalah kesesuaian pembanding historis, bukan jumlah kesalahan label resmi.

Hasil menegaskan bahwa cluster embedding DINOv2 belum memisahkan material berdasarkan kategori secara cukup andal. Menambah K membuat kelompok lebih kecil tetapi tidak memperbaiki akurasi pada kelompok yang belum dilihat. Propagasi graf menyebarkan bias tetangga, bukan mengatasi ambiguitas material. Metode yang secara jujur dapat menerapkan koreksi peserta saat ini adalah cluster khusus yang sudah lolos bukti sebelumnya (1039-anchor foodwaste dan pasangan geometri 0064/1379); hasilnya tidak dapat digeneralisasi otomatis ke semua koreksi.

## Reproduksi

- `variable_cluster_seed_probe.py` menghasilkan `json/variable_cluster_seed_probe.json` dan `table-csv/variable_cluster_seed_probe_summary.csv`.
- `variable_cluster_graph_probe.py` menghasilkan `json/variable_cluster_graph_probe.json` dan `table-csv/variable_cluster_graph_probe.csv`.
- Semua fit memakai embedding dan seed train saja; catatan JSON menyimpan checksum manifest, cache embedding, dan daftar seed.
- Pembanding arsip hanya dipakai oleh audit posthoc yang telah ada, bukan dua eksperimen ini.

## Langkah yang diperlukan untuk koreksi lebih luas

Jangan memilih K atau label cluster berdasarkan kecocokan terhadap arsip anotasi lama. Untuk memperluas label berbasis koreksi manusia, harus ada seed yang benar-benar diperiksa tim dan masih berada dalam kuota maksimal lima per kelas; jika lebih banyak gambar telah diperiksa, jangan memasukkan lebih dari batas tersebut sebagai label train. Alternatif yang layak ialah meminta klarifikasi panitia tentang penggunaan koreksi/penamaan cluster berskala besar, atau menambah fitur domain-material dan membuktikan manfaatnya dengan validasi grup pada seed yang sah sebelum menyentuh label aktif.
