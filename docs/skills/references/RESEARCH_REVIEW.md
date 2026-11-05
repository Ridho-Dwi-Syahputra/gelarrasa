# Riset menjadi hipotesis yang bisa diuji

Daftar lengkap tautan dan cakupan pembacaan ada di [SOURCE_REGISTER](SOURCE_REGISTER.md). Catatan ini merupakan sintesis untuk proyek, bukan reproduksi paper.

**Pembaruan aturan 5 November:** gagasan paper di bawah bukan izin kompetisi. [Q&A aktif](RULES_AND_RUBRIC.md) melarang LLM/VLM termasuk CLIP/SigLIP, serta label/koreksi gambar manual satu per satu. Anotasi lama tidak digunakan. Metode audit label/relabeling pada paper disimpan sebagai konteks ilmiah, bukan tindakan yang boleh disalin ke proyek. Rute aktif berawal dari clustering peserta, penamaan cluster oleh tim, kemudian training/fine-tuning non-VLM dengan data panitia.

## 1. Kualitas anotasi dan hybrid adalah dua kandidat paling relevan

Nguyen dkk. membandingkan fitur klasik, deep learning, dan classifier klasik di atas fitur deep; mereka juga meninjau label salah. Versi preprint melaporkan hasil hingga 100% pada benchmark tertentu, sedangkan abstrak jurnal terkait melaporkan 99,72% pada TrashNet. **Itu bukan hasil GELAR RASA** dan tidak dapat dipindahkan sebagai ekspektasi skor. Urutan split, training backbone, feature selection, dan koreksi target perlu diaudit sebelum mereplikasi. [Preprint](https://arxiv.org/html/2510.21833v1), [jurnal](https://www.sciencedirect.com/science/article/pii/S2090447926000894).

Eksperimen proyek yang diusulkan: pada fold dan label yang sama, bandingkan HOG+warna+SVM, embedding frozen+linear, embedding frozen+SVM, CNN fine-tuning, serta embedding+HOG+warna. Seluruh scaler/PCA/seleksi fitur dilatih hanya pada bagian train setiap fold. CNN yang pernah dilatih memakai label validation tidak boleh menghasilkan fitur untuk menilai validation itu.

Catatan pemeriksaan: narasi bagian 4.4 dataset A menyebut SVM/LoR sempurna, tetapi Tabel 11 mencatat top-100 masing-masing 99,20%/99,18%. Jangan mengutip narasi itu sebagai bukti hasil sempurna pada konfigurasi tersebut. Ini ketidakselarasan pelaporan yang perlu diperiksa sebelum reproduksi. [Bagian 4.4 dan Tabel 11](https://arxiv.org/html/2510.21833v1).

## 2. Label perlu diperiksa, bukan langsung disalahkan atau dianggap benar

Confident Learning menempatkan ketidakpastian label sebagai objek analisis. Gunakan probabilitas OOF untuk mengurutkan kandidat masalah; metode ini tidak membuktikan label benar/salah, terutama saat anotasi punya kesalahan sistematis. [Publikasi JAIR](https://dspace.mit.edu/entities/publication/39b0f8e0-b673-42c6-99f6-0af829bef7f5).

Arsip lama memuat 51 label meragukan, tetapi arsip itu tidak dipakai untuk membangun target baru. Dalam batas aturan sekarang, diagnosis kualitas/cluster dapat dilakukan lewat metode peserta tanpa koreksi manual per gambar. Keterbatasan label turunan perlu dinyatakan, bukan ditutupi skor agreement tinggi.

## 3. Foto bersih dan gambar dalam konteks tidak ekuivalen

RealWaste membahas klasifikasi limbah lokasi nyata versus material yang lebih bersih. Relevansinya adalah kebutuhan mengevaluasi domain/sumber, bukan menyalin jumlah kelas atau datasetnya. Akses full text tidak berhasil saat pencarian ini; kesimpulan rinci di luar abstrak belum ditelaah. [RealWaste](https://www.mdpi.com/2078-2489/14/12/633).

TACO adalah tugas deteksi/segmentasi limbah dalam konteks. Ide pemisahan objek dapat berguna sebagai diagnostik, tetapi anotasi bounding box atau mask tidak otomatis tersedia untuk tugas kita. [TACO](https://arxiv.org/abs/2003.06975).

Usulan lokal: evaluasi terpisah untuk satu objek, objek campuran, tekstur saja, ilustrasi, foto stock, dan latar ramai. Uji full-frame terhadap crop yang terverifikasi, jangan menerapkan background removal pada semua citra tanpa pembanding.

## 4. Transfer learning layak menjadi baseline kuat

EfficientNet menyeimbangkan ukuran model dan resolusi; EfficientNetV2 mengeksplorasi efisiensi training. ConvNeXt menawarkan keluarga CNN lain. Pilihan ini mendukung pembandingan beberapa keluarga, bukan argumen bahwa model terbesar pasti menang. [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html), [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html), [ConvNeXt](https://arxiv.org/abs/2201.03545).

Usulan: EfficientNet-B0 head-only dahulu; tambah fine-tuning LR backbone lebih kecil, lalu pembanding ConvNeXt-Tiny atau EfficientNetV2-S. BatchNorm dan preprocessing mengikuti checkpoint. Ukur resolusi 224 sebelum 320/384. Tidak perlu langsung 640 karena jumlah data kecil dan detail yang dibutuhkan perlu dibuktikan.

## 5. Embedding frozen memberi pembanding yang berbeda

DINOv2 mempelajari representasi visual umum. Embedding frozen dapat digunakan untuk probe linear atau mencari tetangga mirip; kedekatan embedding tidak membuktikan foto sama atau label sama. [DINOv2](https://arxiv.org/abs/2304.07193).

CLIP memberi konteks literatur representasi gambar–teks, tetapi **bukan kandidat kompetisi aktif**: Q&A melarang VLM pada semua tahap termasuk feature extraction. Tidak ada prompt/zero-shot/voting VLM yang dijalankan. [CLIP](https://arxiv.org/abs/2103.00020).

## 6. Augmentasi dan pseudo-label adalah opsi uji

MixUp menginterpolasi input/target; CutMix menukar patch dengan penyesuaian label. Pada limbah, patch dapat menghilangkan objek utama atau mengubah proporsi material. Uji satu per satu dan tampilkan hasil transform. [MixUp](https://arxiv.org/abs/1710.09412), [CutMix](https://arxiv.org/abs/1905.04899).

FixMatch menggabungkan pseudo-label percaya diri dengan konsistensi augmentasi. Untuk proyek ini, opsi tersebut menunggu aturan yang jelas dan baseline supervised yang layak. Evaluasi awal dapat menyembunyikan sebagian label train dalam inner split, tanpa memasukkan validation sebagai data unlabeled training. [FixMatch](https://arxiv.org/abs/2001.07685).

## 7. Validasi dan interpretasi

Kajian leakage mendukung pemisahan unit sumber dan dokumentasi proses evaluasi. Terapkan grouping pada duplikat/varian foto sebelum augmentasi; jangan menggunakan hasil public berulang sebagai data latih terselubung. [Kapoor dan Narayanan](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).

Grad-CAM adalah alat menelusuri area yang memengaruhi skor model, bukan pembuktian bahwa model memahami material. [Grad-CAM](https://arxiv.org/abs/1610.02391).

Temperature scaling mengatur confidence. Untuk satu model dengan suhu positif tunggal, argmax tidak berubah sehingga Accuracy top-1 juga tidak berubah; manfaatnya ada pada confidence atau kombinasi model, bukan “bonus akurasi” otomatis. [Guo dkk.](https://proceedings.mlr.press/v70/guo17a.html).

## 8. EDA dan preprocessing yang diprioritaskan

Dokumentasi Datalab mendukung pemeriksaan label, kemiripan, outlier, serta kualitas citra melalui prediksi di luar sampel dan fitur. Terapkan sebagai antrean tinjauan visual; peringatan bukan izin otomatis menghapus gambar atau mengganti label. Untuk jarak global gunakan representasi yang berbagi encoder/koordinat; lihat [kontrak artefak](ARTIFACT_CONTRACTS.md). [cleanlab image audit](https://docs.cleanlab.ai/stable/tutorials/datalab/image.html).

Pillow membedakan koreksi orientasi dan beberapa perilaku resize; scikit-image menjelaskan bahwa konversi dtype bukan selalu konversi rentang intensitas. Implikasi proyek: audit orientasi, mode, rentang piksel, padding dan objek yang terpotong sebelum membandingkan model. Ini koreksi pipeline yang dapat diuji, bukan klaim bahwa sharpening/CLAHE selalu meningkatkan Accuracy. [Pillow](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html), [scikit-image](https://scikit-image.org/docs/stable/user_guide/data_types.html).

Urutan tindakan dan artefak yang belum dibuat dirinci dalam [rencana EDA/preprocessing](EDA_PREPROCESS_PLAN.md). Panduan itu menghubungkan riset dengan audit 1.920 train dan 480 test tanpa menyamakan label lokal dengan jawaban panitia.
