# Percobaan material dan geometri ? 7 November 2026

## Hipotesis

Representasi objek dapat mengelompokkan gambar dari bentuk atau produk, padahal kompetisi membedakan bahan. Representasi material khusus mungkin menambah sinyal untuk glass/plastic, paper/cardboard, metal dan textile. Ini hipotesis yang diuji pada seluruh train eligible, bukan aturan per ID untuk mencocokkan48review tersisa.

## Sumber primer

- Bell, Upchurch, Snavely & Bala (CVPR2015), *Material Recognition in the Wild with the Materials in Context Database*: https://labelmaterial.s3.amazonaws.com/release/cvpr2015-minc.pdf . Full text tersedia; bagian representasi patch dan graph model diperiksa. Paper membedakan klasifikasi patch dan segmentasi penuh.85.2%mean class Accuracy yang dilaporkan adalah pada MINC, bukan Gelar Rasa.
- Halaman proyek penulis: https://opensurfaces.cs.cornell.edu/publications/minc/ . Menyediakan model23material, termasuk fabric, food, glass, metal, paper dan plastic; tidak menyediakan kategori cardboard/electronic sesuai taksonomi lomba. Karena itu keluaran sumber tidak menjadi label akhir langsung.
- Konversi bobot Caffe ke susunan tensor PyTorch: https://github.com/keunhong/kitnn/tree/6fb707ae76df58d7c02424661822cd6dfda60c24 . Tidak mengeksekusi kode Python repository. Membaca implementasi preprocessing BGR dan susunan kelas. Hanya model numerik yang digunakan.
- Pembanding metode tekstur: Deep TEN, https://hangzhang.org/PyTorch-Encoding/tutorials/texture.html . Dokumentasi resmi diperiksa; DeepTEN tidak dijalankan dalam percobaan ini.

## Implementasi dan bukti

Deploy GoogLeNet asli diambil dari arsip model resmi penulis; categories dan prototxt disimpan dalam cache lokal. Tidak mengekstrak atau memakai contoh gambar yang dibundel maupun dataset MINC. Bobot36.4MBdari konversi dibaca dengan unpickler terbatas pada konstruksi array/dtype dan encodingLatin1; semua arrayfloat32finite divalidasi dan disimpan ulang tanpa pickle.

SHA bobot asli: `354e380615edd03d48e18958d384e1dd4ea5b9dd645035f7c2ea261c551b5fcf`. SHA bobot numerik: `0932d459da0eb58d899f7216fc8cb5c68f1489cab7ef954f9f26cf9d10cfd0dc`. Graph142layer ditafsirkan dengan operatorPyTorch dari prototxt, bukan menjalankan skrip unduhan. Tensor58layerparameter diperiksa. Belum ada pengujian parity numerik terhadap engineCaffe atau replikasi skor benchmark paper; keterbatasan ini harus dipertahankan dalam pelaporan.

Model dijalankan pada1904foto train eligible, tanpa1662/1885dan tanpa test. Tiga tampilan: center-fit foto utuh, center-fit ROI dari border seragam bila memenuhi syarat, dan pusat60%dari ROI. Ini hanya tampilan fitur; tidak memindahkan label parent ke crop baru. Fitur pool1024perview dan probabilitas23material disimpan lokal.

Head lokal memakai34acuan manusia yang sudah dipakai, maksimum5perkelas, serta prior otomatis dan bobot dari snapshot sebelum aktivasi acuan manusia pertama. Leave-source-group-out pada34acuan dipakai untuk pengembangan/pemilihan head, bukan evaluasi independen final. Pengelompokan asli, rawhash, SHAfitur dan parameter dicatat. Tidak membaca48review tersisa atau seluruh126keputusan sebagai target.

## Geometri lokal

8047pasangan dari top4tetanggaDINOv2 danDINOv3diperiksa menggunakanSIFT, Lowe ratio.70, homographyRANSAC3piksel. Koneksi kuat mensyaratkan>=16inlier, proporsi>=.70, convex-hull>=.15pada kedua gambar dan warp masuk akal.116koneksi mencakup183foto; seluruh komponen sudah konsisten kategori programnya. Tidak ada transfer kategori tambahan. Hal ini membuktikan jalur near-source yang diuji belum memberi acuan baru, bukan bahwa semua metode geometri mustahil.

## Aturan seleksi yang dibekukan

Fitur materialglobal/foreground/detail, gabungan material-DINO, dan HOG/warna fit ke prior dingin +34acuan. Tiga kandidatSVMperkeluarga dan dua bobot prior dipilih berdasarkan diagnostik acuan saja. Usulan perubahan memerlukan>=4/5headpada kedua bobot, headhybrid dan>=2/3headmaterial sepakat, kedekatan dan margin ke acuan pada>=2bank. Acuan manusia dan refinement terdahulu dilindungi. Guardpublikasi: diagnostik pooled>=14/34dan perubahan<=50; jika gagal, tidak ada prediksi percobaan yang diterapkan. Reviewprivat baru dibaca setelah prediksi dan keputusan guarddibekukan.
