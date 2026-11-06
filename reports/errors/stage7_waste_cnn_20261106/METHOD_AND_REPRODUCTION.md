# Model citra bernama untuk menyelesaikan kelompok campuran

## Perubahan yang diuji

Kategori soal merupakan campuran material (glass, paper, metal, plastic), jenis objek (electronic), kemasan (cardboard), dan sumber limbah (textile, foodwaste). Mengubah jumlah cluster pada satu embedding tidak memberi korespondensi kategori tersebut secara otomatis.

Uji stage6 menggunakan CNN material MINC GoogLeNet dan objek ImageNet ResNet50. Skor dari dua ruang label berbeda semula menghasilkan dominasi plastic. Versi berikut menormalisasi evidence dan memprioritaskan kategori objek electronic/cardboard secara generik. Tiga percobaan head dan satu pilot adaptasi blok akhir dicatat terpisah; tidak memakai arsip LLM.

Uji stage7 memakai base **ResNet50 yang sudah dipretrain khusus limbah**, kemudian melatih head delapan kategori pada train kompetisi. Ini bukan model dari nol atau model yang hanya dipretrain ImageNet. Sumber bobot dan asal pretraining diungkap; dataset gambar eksternal tidak diunduh. Pengetahuan kategori berasal dari CNN non-VLM serta korespondensi taksonomi tertulis, bukan peninjauan gambar oleh asisten.

## Sumber primer dan korespondensi

- [Model card pembuat, MIT](https://huggingface.co/kendrickfff/my_resnet50_garbage_classificationv1.2): ResNet50 dengan12 kelas limbah. Klaim skor pembuat bukan hasil kita.
- [Kode app pembuat pada commit tetap](https://huggingface.co/spaces/kendrickfff/resnet50_garbage_classification_v1.2/blob/ab3b912a730f2b6f1986aba8986fbc0d8e5e36b5/app.py): urutan kelas dibaca sebagai literal AST, tidak dieksekusi. Transform Resize256, CenterCrop224, mean/std ImageNet. Ini mencegah kesalahan memakai urutan daftar naratif pada README sebagai indeks model.
- [Cornell MINC](https://opensurfaces.cs.cornell.edu/publications/minc/): material CNN non-VLM; model lama memakai mean RGB124/117/104. Model/config/category diekstrak; demo image pada arsip bobot tidak dibaca atau dipakai.

Battery menjadi electronic; biological menjadi foodwaste; tiga warna glass digabung glass; clothes/shoes menjadi textile; kategori lainnya yang namanya sesuai dipertahankan. Upstream trash tidak dianggap material baru: sisa probabilitasnya mendapat evidence lunak dari CNN material/objek. Identifikasi perangkat elektronik yang kuat dari ImageNet dapat mengatasi prediksi material casing.

## Training dan pemilihan

- Fit1538 dan development381 kelompok terpisah, dari1919 train pixel-unik. Satu duplikat asli tetap ada. Tidak menggunakan test untuk fitting, target atau pemilihan.
- Bandingkan fitur waste-CNN, DINO, serta hybrid L2 per blok dengan bobot setara. Linear/MLP baru dilatih180 epoch pada masing-masing cabang, target lunak dari evidence CNN dan bobot confidence; view consistency dan graph merupakan regularisasi.
- Pemilihan: macro agreement dengan subset CNN-teacher berconfidence>=0,7, flip stability, dan tetangga train pada graph hybrid. Semua8 kategori harus muncul di development, minimal5. Ini proxy, bukan Accuracy resmi.
- Model lama dibandingkan pada kriteria yang sama sebelum model baru dipilih. Tidak memilih berdasarkan kesamaan dengan arsip LLM.
- Checkpoint/assignment dan source hash dibekukan sebelum audit pembanding terpisah. Label LLM hanya dibaca oleh audit setelah seluruh model dipilih.

## Makna hasil bernama

Semua gambar dapat memiliki satu prediksi kategori tanpa folder campuran. **Itu bukan bukti bahwa ambiguitas atau kesalahan semantik sudah nol.** Confidence belum dikalibrasi, `needs_review` tetap direkam, dan label tidak disebut ground truth. Audit harus melaporkan berapa dari570 kasus lama yang kini sesuai atau berbeda terhadap arsip; tidak hanya menyebut campuran menjadi nol.

Source model memiliki pretraining domain limbah; overlap gambar pretraining dengan sumber kompetisi belum diketahui. Split lokal tidak menghapus kemungkinan overlap tersebut dan tidak boleh disebut generalisasi independen terhadap seluruh pretraining. Kelayakan bobot task-specific tidak diklaim sebagai persetujuan eksplisit panitia. Ini kandidat dengan provenance lengkap dan training lokal, bukan jaminan akurasi sempurna atau lolos verifikasi.

## Reproduksi lokal

Jalankan dengan `.venv/Scripts/python.exe`, dari root proyek. Data resmi tetap lokal. Tahap prasyarat menghasilkan manifest stage1, DINO stage2, dan probe view yang sudah ada.

1. `stage6_object_evidence.py`: evidence objek ResNet50.
2. `stage6_download_material_weights.py`: pin bobot Cornell; tidak mengambil dataset MINC.
3. Caffe reader terisolasi: `python -m pip install --no-deps --target output/runtime/opencv4 opencv-python-headless==4.12.0.88`. OpenCV utama proyek tidak diganti.
4. `stage6_material_evidence.py`: evidence material.
5. `stage6_material_resolution.py`, kemudian `stage6_material_resolution.py --variant calibrated`.
6. `stage6_semantic_seed_training.py`: perbandingan seed bersih, kontrol dan pemilihan proxy.
7. `stage7_waste_cnn_features.py`: base domain limbah dengan indeks kelas/transform pembuat yang dipin.
8. `stage7_waste_semantic_training.py`: train/pilih head delapan kategori.
9. `stage6_finalize_predictions.py`: reproduksi probabilitas, salinan per kelas, dan audit arsip setelah freeze.

Pilot blok ViT adalah cabang terpisah: `stage6_lastblock_cache.py` lalu `stage6_lastblock_training.py`. Jika model family berubah, hasil tail lama tidak otomatis digunakan untuk mengganti kandidat yang berbeda.

Dataset, gambar salinan, bobot, cache prefix dan file besar tetap lokal. Hasil aktual/denominator/coverage ada pada summary dan final audit setelah eksekusi selesai. Tidak ada submission yang dikirim otomatis.
