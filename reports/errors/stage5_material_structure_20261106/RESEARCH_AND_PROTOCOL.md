# Mencari kelompok material yang lebih sesuai - 6 November 2026

## Pertanyaan dan diagnosis yang dapat diuji

Pengguna ingin hasil clustering mendekati kategori material pada arsip pembanding. Peningkatan agreement terhadap arsip bukan kriteria pemilihan eksperimen. Nama kategori aktif belum dipromosikan dan tidak tersedia ground truth resmi train. Kelompok delapan yang ada menyisakan campuran; itu menunjukkan bahwa delapan kelompok representasi tidak otomatis sama dengan delapan material.

Pilot sebelumnya melatih empat head di atas encoder frozen. Keanggotaan dapat berubah, tetapi representasi gambar belum dilatih ulang untuk memisahkan material. Dugaan bahwa konteks/bentuk mengalahkan tekstur adalah hipotesis, bukan hasil attribution. Tiga perubahan berikut diuji sebelum audit arsip: pooling patch non-padding, bobot blok global/lokal yang eksplisit, dan 32 prototipe sebelum penggabungan geometris ke delapan. Seluruhnya adaptasi kecil; bukan implementasi penuh IIC atau SPICE.

## Dasar primer

- [IIC, Ji et al., ICCV 2019](https://arxiv.org/abs/1807.06653), [kode penulis](https://github.com/xu-ji/IIC): auxiliary overclustering merupakan komponen metode. Eksperimen lokal hanya membandingkan prototipe 32 dan penggabungan Ward; tidak melatih auxiliary neural head atau objective IIC.
- [SPICE, Niu et al., TIP 2022](https://arxiv.org/html/2103.09382v3): tahap akhir melatih representasi dan head bersama dengan pseudo-label yang dipilih melalui konsistensi tetangga. Head frozen-feature saja belum menjalankan tahap itu. Paper juga memperlihatkan pentingnya kualitas pseudo-label; konsistensi tinggi tidak otomatis menjamin material benar.
- [Bell et al., CVPR 2015](https://arxiv.org/abs/1412.0623): material recognition mempertimbangkan patch dan konteks. Ini mendukung hipotesis fitur lokal; hasil dataset/label MINC bukan bukti performa kita dan datanya tidak dipakai.
- [DINOv2 resmi](https://github.com/facebookresearch/dinov2): checkpoint citra non-VLM yang sudah dipin menyediakan representasi global serta token patch. Pooling mean/std non-padding merupakan adaptasi lokal.

## Protokol sebelum melihat pembanding

- Encoder sama, pinned SHA, inference lokal CPU. Semua 2.400 gambar ditransform satu per satu tanpa fitting pada test. Uji ulang CLS harus cocok dengan cache lama dalam toleransi 1e-5.
- Blok: CLS 384, mean 384 dan std 384 token patch non-padding. Padding dikeluarkan berdasarkan pusat patch dan geometri letterbox. Latar asli masih ada; tidak diklaim sebagai mask objek atau segmentasi.
- Global-only, local-only, dan hybrid dengan massa kuadrat .5/.25/.25. Ini bobot ditentukan sebelum run, bukan hasil optimasi arsip.
- KMeans 8/32, tiga seed 11/29/47, n_init 10, max_iter 300. Tambahan weighted-Ward 32-ke8 pada global/hybrid. Penggabungan Ward geometris, bukan bukti material sama.
- Fit1538 dan development 381 kelompok terpisah memakai split yang telah digunakan sebelumnya. Scaling/fitting cluster hanya pada fit. Probe flip/resolusi memakai 87 heldout dari probe train 256 yang sudah dipilih acak sebelumnya.
- Pemilihan kandidat delapan: semua kelompok terisi, minimal 5 gambar heldout per kelompok, skor 0,4 flip + 0,4 scale + .2kesepakatan tetangga lintas representasi. Baseline full-train hanya kontrol deskriptif, tidak eligible karena melihat development.
- Semua kandidat dipertahankan. Diagnostic 32 ditentukan sebelumnya: hybrid seed 11. Tidak memilih jumlah kelompok atau seed berdasarkan purity arsip. Silhouette dan konsistensi bukan Accuracy.
- Assignment, summary, kode dan artefak aktif dibekukan dengan hash sebelum skrip pembanding terpisah berjalan. Audit arsip menghitung ARI/NMI, homogeneity/completeness serta pairwise precision/recall; tidak membuat mapping nama kategori.

## Batas interpretasi dan kelanjutan

Menambah kelompok dapat memecah satu material menjadi banyak subtipe. Homogeneity yang lebih tinggi harus dibaca bersama completeness dan recall, bukan disebut Accuracy yang naik. Kelompok 32 belum menjadi label delapan kelas; penamaan/penyatuan semantik oleh tim dan ketentuan jumlah cluster perlu dibedakan dari eksperimen geometri ini.

Bila fitur lokal tidak memperbaiki proxy dan struktur pembanding, tidak dipromosikan. Langkah riset berikut yang berbeda dari pengulangan head adalah fine-tuning representasi pada train kompetisi dengan target algoritmis yang konsisten dan view yang mempertahankan material. Itu belum dijalankan oleh uji frozen-feature ini. Tidak ada jaminan partisi tanpa supervisi akan identik dengan taksonomi LLM; arsip pembanding pun bukan ground truth resmi.

Hasil aktual disimpan terpisah pada summary.json, development_metrics.csv dan private_reference_audit.json setelah run selesai. Notebook utama dan baseline lama dipertahankan; tidak ada submission baru dari eksperimen ini.
