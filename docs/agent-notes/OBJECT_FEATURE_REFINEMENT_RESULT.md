# Penyempurnaan fitur objek — 7 November 2026

**Status aktif: 26 berbeda / 100 sesuai dari 126 review tegas; 34 sumber manusia tetap, maksimal lima per kelas.** Prefix `balanced_pseudo_anchor`. Lihat [status dan bukti terbaru](CURRENT_LABEL_STATUS.md). Bagian awal berikut memuat riwayat sebelum pengujian 8 November; hasil tidak semuanya tanpa regresi.

Publikasi `detector_primary` mengubah sepuluh kategori otomatis. Audit sesudah keputusan publikasi menemukan dua pemulihan: 1246 dan 1557. Sebanyak 91 dari 126 review tegas sesuai, 35 berbeda; tidak ada regresi review pada tahap ini. Delapan perubahan lain berada di luar review tegas dan belum terverifikasi. Status setelah eksperimen lanjutan selalu mengacu ke [status aktif](CURRENT_LABEL_STATUS.md), bukan angka historis laporan ini.

## Perbandingan pengembangan

| Metode | Sesuai pada 34 acuan | Macro recall | Keputusan tahap ini |
|---|---:|---:|---|
| ViT + DINO, gambar utuh, sebelumnya | 19 | 0,6083 | Baseline aktif sebelumnya |
| Fitur objek + ViT + DINO | 20 | 0,6333 | Dipilih sebelum audit privat; sepuluh perubahan |
| Crop objek otomatis + DINO | 19 | 0,6083 | Tidak menggantikan pemenang |
| SVM fitur ViT | 16 | 0,5333 | Ditolak oleh guard pengembangan |
| Graph dengan acuan otomatis dibatasi | 18 | 0,5833 | Ditolak oleh guard pengembangan |

Ini diagnostik pengembangan pada lima fold kelompok sumber. Acuan yang sama dipakai memilih konfigurasi; bukan evaluasi final independen atau Accuracy panitia. Graph pada eksperimen kNN menggunakan seluruh train dan tidak diklaim sebagai validasi graph independen. Eksperimen sparse graph membangun graph hanya pada fit fold, kemudian menginterpolasi query yang dikeluarkan.

## Pengolahan objek

Sebanyak 1.904 gambar train eligible diproses. YOLO image-only menghasilkan kandidat fokus yang memenuhi kriteria pada 1.122 gambar; 782 tetap memakai gambar utuh. Fitur piramida dan foreground menjadi masukan head lokal. Taksonomi lima kelas model sumber tidak disalin sebagai kategori delapan kelas kompetisi. Crop otomatis ini hanya tampilan fitur, tidak menambah crop training atau mengubah piksel sumber.

Bobot pretrained diambil sebagai tensor numerik. Checkpoint YOLO dibaca secara simbolik untuk mengekstrak tensor, tanpa mengeksekusi fungsi terserialisasi. ViT dikonversi secara numerik dan diperiksa dengan perhitungan forward pembanding. Tidak ada gambar eksternal yang ditambahkan atau inference pihak ketiga.

Sumber model: [waste ViT](https://huggingface.co/watersplash/waste-classification), [YOLO litter materials](https://huggingface.co/CatSat/yolov11-litter-materials). Pengujian representasi berikut memakai [DINOv2 resmi](https://github.com/facebookresearch/dinov2); ketersediaan model atau benchmark sumber tidak menjamin hasil pada kompetisi ini.

## Batas dan bukti

34 sumber manusia tetap, maksimal lima per kelas; tidak menambah koreksi manual. Test dan foto 1662/1885 tidak masuk fit. Notebook preprocessing lulus 13 sel kode, nol error, 16 visual; 4.793 hash gambar diverifikasi. Snapshot assignment tetap input: Run All mematerialisasikan kategori dan menilai crop, belum mengulang seluruh fitting assignment heads atau melatih classifier/submission final.

Bukti keputusan sebelum audit: `json/object_feature_selection_decision.json`, `json/detector_primary_publication_decision.json`, `json/detector_primary_frozen.json`. Bukti aktif: `json/bounded_anchor_publication.json`, `json/current_review_status.json`, `execution.json`, `skills_validation.json`.

## Tahap DINOv2 Base dan normalisasi

Bobot resmi image-only Base diekstrak pada seluruh 1.904 train eligible, tanpa test atau dua sumber yang ditahan. Acuan otomatis yang bertentangan kuat dengan acuan manusia pada sisi fit dapat dikeluarkan dari fit; tidak diganti menjadi label review yang tidak termasuk jatah. Setiap fold mempertahankan seluruh acuan manusia di sisi fit dan mengeluarkan kelompok sumber query.

- Base mentah: primary22/34,macrorecall0,7000; tiga usulan.
- Base dengan penyaringan core: primary23/34,macrorecall0,7250; tiga perubahan diterapkan. Audit memulihkan0787/1566,35menjadi33,tanpa regresi.
- Pemilihan bank yang lolos pengembangan memberi dua usulan saja dan tidak menggantikan protokol perbandingan yang sudah ditetapkan. Pada delapan acuan yang benar menurut primary, voting bank lemah sebelumnya tidak mencapai dukungan tiga bank; ini diagnosis pengembangan, bukan alasan koreksi individual.
- Keyakinan kNN tanpa normalisasi tidak mencapai kriteria selektif90%pada minimal10acuan; tidak diterapkan. Skor tinggi bukan bukti kategori benar.
- SVM pada Base yang disaring:22/34,macrorecall0,7000; tidak melampaui guard23/34dan0,7250,ditolak.
- Centering/PCA128 half-whitening: normalizer fit hanya sisi train masing-masing fold. Primary memilih center_only padaCLS+ROI,23/34dan0,7250; bankCLS memilih PCA128 half-whitening,22/34dan0,7000. Ambang minimumshare0,95 dipilih pada34acuan:11tercakup,10benar. Kalibrasi memakai acuan pengembangan yang sama dan bukan evaluasi final independen.
- Dua bank dan dua graph train harus menyetujui kategori. Sebanyak53usulan memenuhi syarat; cap40yang ditetapkan sebelum fit memilih skor tertinggi. Audit setelah publikasi menemukan empat pemulihan:0168,0539,0826,1530. Sisa33menjadi29,tanpa regresi review. Sebanyak36perubahan di luar review tegas belum terverifikasi; jangan menyebut seluruh40benar.
- Head lokal dari probabilitas12kelasViT/statistik28YOLO,termasuk gabungan fiturBase,hanya mencapai16/34dan0,5333; tidak diterapkan. Tidak ada pemetaan manual kelas pretrained ke delapan target.

Seluruh tahap memakai34sumber manusia yang sama dan target inti otomatis, tanpa membaca daftar review lain untuk fit/filter/pemilihan. Remaining review adalah audit setelah pembekuan. Lima fold, preprocessing, pemilihan head dan kalibrasi berbagi acuan pengembangan; belum ada evaluasi semantik final yang independen. Graph final memanfaatkan semuaTRAINdanclamp34acuan, sehingga persetujuan graph bukan validasi independen pada34acuan itu.

Bukti terbaru: `json/base_comparison_protocol.json`, `json/base_comparison_decision.json`, `json/normalized_base_plan.json`, `json/normalized_base_publication_decision.json`, `json/normalized_base_frozen.json`. Sumber numerik dan SHA ada pada catatan ekstraksi. Notebookv14lulus13sel/0error/16visualdan4.793hash; processed tetap1913sampeltrain/1911eligible/480test. Semua kebijakan asli dan dua penahanan tetap berlaku.

## Pengujian lanjutan 8 November 2026

**17 berbeda / 109 sesuai dari 126 review tegas; 34 sumber manusia tetap, maksimal lima per kelas.** Angka ini perbandingan setelah pembekuan prediksi, bukan ground truth panitia. Seluruh prediksi diterapkan sebelum membuka audit lengkap; audit bukan target, filter acuan, atau pemilih perubahan per-ID. Peneliti telah melihat audit tahap sebelumnya sehingga risiko adaptasi tingkat metode tetap perlu diakui.

| Eksperimen | Hasil pengembangan/keputusan | Dampak audit publikasi |
|---|---|---|
| Neural Base CE/GCE | Perbaikan43target cold-core yang dulu tertukar routing; terbaik22/34, ditolak | Tidak diterapkan |
| Campuran DINO/CNN | 24/34, macro0.75; 13 perubahan | 29 menjadi28; 0597 pulih |
| Teacher/student fold-contained | Global22/34; cakupan selektif15/16 | Guard global ditolak; eksperimen selektif terpisah14perubahan tidak mengurangi28 |
| Tekstur ConvNeXt orde dua | Terbaik21/34; ditolak | Tidak diterapkan |
| Pseudo-acuan program, putaran1 | 184 acuan awal; konsistensi program0.8533/macro0.7148; 40perubahan | 0343 pulih,28 menjadi27 |
| Pseudo-acuan putaran2 | 40perubahan; bank berikut264dengan138textile | 0536regresi; batch ditarik utuh karena bias konsentrasi, kembali27 |
| Putaran3 protokol lama | Prediksi dihitung, tidak dipublikasikan | Ditahan karena kesehatan bank; bukan perbaikan aktif |
| Photometric/SIFT | 19edge ketat, SSIM/coverage/MAE piksel | Tidak ada label yang langsung disalin |
| Attention bag patch | Terbaik19/34; ditolak | Tidak diterapkan |
| Delapan augmentasi per acuan | Tetap34sumber, terbaik justru tampilan asli23/34 | Ditolak; augmentasi tidak menambah kuota/ukuran evaluasi |
| SVM spesialis kelompok material | Tidak ada cabang lolos guard | Ditolak |
| Graph/kernel hanya34manusia | Terbaik20/34, macro0.55 | Ditolak |
| Bank pseudo source-diverse | 143acuan, maksimum20otomatis/kelas,4/5bankdan batas8perkelas/40total; 32perubahan | 0868/1503pulih,0536regresi;27 menjadi26 |
| Zoom objek sebelum DINO Base | 1122foto difokuskan,782fallback;22/34,macro0.70; selektif15/16 | Guard keseluruhan ditolak,tidak diterapkan |
| Supervised metric adaptation | 22/34,macro0.70;group-heldout,augmentasi sumber yang sama | Ditolak |
| Hybrid HOG/HSV dari webinar | Semantik+warna24/34,macro0.75;minimumshare lintasbank tidak mencapai kalibrasi | Ditolak |
| Campuran objek/warna/metric | Terbaik24/34,macro0.75;tidak melampaui pembanding | Ditolak |
| Perbaikan cold-core dari tetangga manusia | 24/34,macro0.75;ambang0.95mencakup13,dengan12sesuai | Ditolak oleh guard25/34danmacrolebih0.75 |
| Konsensus selektif empat guru | Konfigurasi10/10hanya mencakup10; tidak ada aturan lolos coverage16/precision0.95/Wilsonproxy | Ditolak; tidak menurunkan ambang sesudah melihat hasil |

Perubahan32pada bank source-diverse bukan 32koreksi terverifikasi. Konsistensi pseudo0.7890/macro0.6981 mengukur keselarasan dengan target program, bukan Accuracy manusia. Tidak ada human holdout independen untuk bank ini; semua acuan pseudo sudah dipengaruhi34sumber upstream. Pembatasan kelas hanya pada pemilihan acuan/budget update; jumlah akhir gambar tiap kelas tidak dipaksakan sama.

Rujukan konsep: [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) untuk risiko/cakupan; [B-CNN](https://arxiv.org/abs/1504.07889) untuk fitur orde dua; [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html) untuk pooling patch; [DARP](https://papers.nips.cc/paper_files/paper/2020/hash/a7968b4339a1b85b7dbdb362dc44f9c4-Abstract.html) untuk hipotesis bias pseudo menuju kelas mayoritas. Ini adaptasi/hipotesis lokal, bukan reproduksi persis keempat metode.

Ekstraksi tambahan objek diperbesar sebelum tokenisasi memproses1904train eligible,memfokuskan1122objek dan memakai fallback pada782lainnya; tidak membuat crop train baru. Hash fitur fa928bc82eb354e5fa5be783bd51064529e0e1d3a0ad4f8a5d0eabaf947e8b19. Semua eksperimen memiliki plan/frozen/development tersendiri. Keputusan diterapkan hanya bila guard yang dicatat sebelum audit lolos. Metode yang gagal tidak mengubah data aktif. Bukti terkini pada json/current_review_status.json, json/bounded_anchor_publication.json, execution.json dan skills_validation.json.

Audit agregat akhir sesudah seluruh fit dibekukan:18plastic,3paper,3glass,2cardboard menurut review tim. Kuota manusia masing-masing empat kelas ini sudah5/5; sisa metal2/foodwaste4 tidak bisa dialihkan. Angka ini tidak digunakan sebagai target/filterfit atau perintah koreksi per-ID. Target nol belum tercapai. Menyamakan dengan review tidak membuktikan Accuracy resmi100persen; anotasi arsip tetap tidak dipakai sebagai target.

Keselarasan review bukan Accuracy panitia. Perubahan di luar review belum terverifikasi. Run All mematerialisasikan snapshot assignment dan menghitung crop; belum mengulang seluruh fit assignment heads atau membuat classifier/CSV final. Dua sumber held tetap dikecualikan; raw/test utuh; manual34tetap dan ledger tidak direset oleh rollback.
