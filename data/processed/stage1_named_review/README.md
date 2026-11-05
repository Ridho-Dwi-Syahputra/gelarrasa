# Folder peninjauan bernama kategori

**Status: tampilan perbandingan arsip; bukan label training atau prediksi kategori yang terverifikasi.**

Nama direktori mengikuti kategori yang paling banyak muncul pada arsip lama di setiap cluster. Keanggotaan cluster tetap persis hasil algoritma; tidak ada gambar dipindah sendiri-sendiri. Arsip bersumber dari anotasi visual asisten sebelumnya, bukan ground truth resmi. Nama folder tidak mengubah asal pengelompokan menjadi label algoritmis material yang sah.

Struktur: `train/<nama kategori>/<cluster_id>/<image_id>.png` dan struktur yang sama untuk `test/`.

| Cluster | Nama folder peninjauan | Train | Test | Porsi kategori arsip terbanyak |
|---|---|---:|---:|---:|
| cluster_00 | textile | 188 | 36 | 95.7% |
| cluster_01 | metal | 272 | 62 | 68.4% |
| cluster_02 | cardboard | 213 | 50 | 96.2% |
| cluster_03 | plastic | 269 | 90 | 29.0% |
| cluster_04 | foodwaste | 170 | 46 | 98.8% |
| cluster_05 | electronic | 216 | 61 | 83.8% |
| cluster_06 | glass | 302 | 87 | 52.3% |
| cluster_07 | paper | 290 | 48 | 60.0% |

Porsi di atas adalah komposisi arsip, **bukan Accuracy**. Cluster 03 sangat campuran; nama plastic hanya penanda blok arsip terbesar, bukan pernyataan bahwa semua anggotanya plastik.

File merupakan salinan independen dari hasil RGB native, bukan hardlink. Sebanyak 2.400 salinan diverifikasi SHA-256-nya. Notebook utama, assignment beku, template mapping tim, serta kontrak data aktif tidak berubah.

Jangan arahkan loader training atau exporter submission ke folder ini. Jalur aktif tetap `stage1_v1/cluster_views`; nama kategori aktif menunggu pemetaan tim yang berasal dari penelaahan cluster sesuai aturan.
