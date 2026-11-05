# Skill eksternal: seleksi dan adaptasi

Pencarian: skills.sh leaderboard; `npx skills find "computer vision"`, `"pytorch"`, `"image classification"`, `"data quality"`, `"image preprocessing"`, `"exploratory data analysis"`, dan `"kaggle"`; pencarian web/GitHub; metadata dan tree melalui GitHub API. Tanggal pencatatan: 4 November 2026. Jumlah instalasi/stars adalah snapshot, bukan jaminan mutu. Sumber dipilih berdasar kecocokan isi, asal, dan deklarasi lisensi setelah penyaringan. Paket tambahan mendahulukan kurasi/EDA dan preprocessing karena label train dibuat sendiri.

## Tujuh belas paket snapshot yang disimpan

| Paket | Sumber | Snapshot popularitas | Yang diambil / adaptasi |
|---|---|---|---|
| huggingface-vision-trainer | [Hugging Face](https://github.com/huggingface/skills) | ≈2K instalasi skill; 11.264 stars repo | Resep image classification/timm dan reliability. Cloud Jobs/Hub bukan default proyek lokal |
| pytorch-patterns | [ECC](https://github.com/affaan-m/ecc) | ≈10,9K instalasi; 296.816 stars | Loop, checkpoint, seed. Perbaiki asumsi loader jpg-only, AMP CUDA, dan mean loss per-batch |
| computer-vision-opencv | [Mindrally](https://github.com/Mindrally/skills) | ≈3,9K instalasi; 293 stars | RGB/BGR, dtype, resize, fitur lokal. Banyak bagian video/tracking tidak diperlukan |
| senior-computer-vision | [alirezarezvani](https://github.com/alirezarezvani/claude-skills) | ≈1,2K instalasi; 29.265 stars | Audit dataset, pemilihan arsitektur. Alur utama detection/deployment perlu diganti klasifikasi |
| pytorch-lightning | [K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills) | 56.045 stars repo; instalasi skill tidak dicatat | Module/DataModule/checkpoint; tidak wajib memakai multi-GPU/logger cloud |
| scikit-learn | K-Dense, tautan di atas | Sama; instalasi skill tidak dicatat | Pipeline/scaler/SVM/CV; ganti random split contoh dengan group split lokal |
| exploratory-data-analysis | K-Dense, tautan di atas | Sama; sekitar 2,2K instalasi skill saat penelusuran tambahan | Manifest dan profil data; helper generik bukan pelabel gambar |
| scientific-critical-thinking | K-Dense, tautan di atas | Sama; instalasi skill tidak dicatat | Bukti, bias, batas klaim; kerangka klinis tidak diadopsi untuk skor CV |
| [scientific-writing](../vendor/K-Dense-AI__scientific-agent-skills/scientific-writing/SKILL.md) | K-Dense, tautan di atas | Repo/commit sama; dipilih dari isi, bukan jumlah instalasi | Keterlacakan klaim, konsistensi hasil, struktur tulisan; gunakan BAB I–IV resmi, bukan aturan jurnal klinis |
| [scientific-visualization](../vendor/K-Dense-AI__scientific-agent-skills/scientific-visualization/SKILL.md) | K-Dense, tautan di atas | Repo/commit sama | Grafik jujur, denominator, uncertainty, export; preset jurnal tidak menggantikan template kompetisi |
| [fiftyone-dataset-curation](../vendor/voxel51__fiftyone-skills/fiftyone-dataset-curation/SKILL.md) | [Voxel51, pengembang FiftyOne](https://github.com/voxel51/fiftyone-skills) | 48 stars repo; instalasi tidak dicatat | Tinjauan mutu, tag, dan galeri; jangan otomatis menghapus data |
| [fiftyone-find-duplicates](../vendor/voxel51__fiftyone-skills/fiftyone-find-duplicates/SKILL.md) | Voxel51, tautan di atas | Sama | Kandidat exact/near-duplicate; konfirmasi pasangan dan simpan group |
| [fiftyone-embeddings-visualization](../vendor/voxel51__fiftyone-skills/fiftyone-embeddings-visualization/SKILL.md) | Voxel51, tautan di atas | Sama | Menavigasi gambar/cluster; proyeksi 2D bukan bukti label |
| [fiftyone-model-evaluation](../vendor/voxel51__fiftyone-skills/fiftyone-model-evaluation/SKILL.md) | Voxel51, tautan di atas | Sama | Cabang classification, confusion matrix, slice errors |
| [albumentations](../vendor/VectorSpaceLab__AREX-Skill/albumentations/SKILL.md) | [AREX-Skill](https://github.com/VectorSpaceLab/AREX-Skill) | 347 stars repo; instalasi tidak dicatat | Pipeline augmentasi, normalisasi, reproducibility; 5 subskill tersimpan |
| [cleanlab](../vendor/VectorSpaceLab__AREX-Skill/cleanlab/SKILL.md) | AREX-Skill, tautan di atas | Sama | Audit label/duplikat/outlier dan multiannotator; 7 subskill tersimpan |
| [scikit-image](../vendor/VectorSpaceLab__AREX-Skill/scikit-image/SKILL.md) | AREX-Skill, tautan di atas | Sama | Dtype, warna, transform, fitur klasik; 5 subskill tersimpan |

Subtree setiap paket terpilih dan lisensi root disalin ke `vendor/`: **240 file, 2.027.190 byte**, setelah dua tambahan reporting pada 5 November. Subskill tidak dihitung lagi sebagai paket tambahan. Commit/URL/ukuran/SHA-256 ada di [vendor_manifest.json](../sources/vendor_manifest.json); adaptasi berada di `custom/`. Ini bukan seluruh repository/dependensi transitif. Tree seluruh AREX terpotong oleh API; tiga subtree terpilih diambil lengkap bersama lisensi root.

Lisensi tingkat repo: Hugging Face, Mindrally, Voxel51, dan AREX Apache-2.0; ECC, alirezarezvani, dan K-Dense MIT. Header skill K-Dense `pytorch-lightning` menyebut Apache-2.0, dan `scikit-learn` BSD-3-Clause. Header AREX albumentations menyebut MIT, cleanlab Apache-2.0, sedangkan scikit-image `NOASSERTION`; catat ketidakjelasan itu, jangan menyatakannya sebagai lisensi spesifik yang telah dipastikan. Simpan deklarasi per-file; lisensi root tidak menghapus pernyataan lebih spesifik. Kode library dan bobot memiliki lisensi tersendiri. File LICENSE upstream disertakan tanpa perubahan; deklarasi entrypoint dicatat pada [license_declarations.json](../sources/license_declarations.json).

## Temuan konkret dari penyaringan isi

1. **ECC Dataset memakai `glob("*.jpg")`.** Untuk proyek ini gunakan manifest sehingga PNG/format lain tidak hilang. Inferensi juga tidak boleh mewarisi `drop_last=True` dari contoh train loader.
2. **ECC inisialisasi bobot.** Contoh `model.apply(...)` cocok untuk model baru, tetapi akan merusak bobot pretrained bila disalin setelah load. Mean loss batch tidak mewakili loss per sampel bila batch terakhir lebih kecil.
3. **Hugging Face berorientasi Jobs/Hub.** Contoh memiliki upload, credential, monitoring, dan biaya cloud. Bagian itu disimpan sebagai referensi; tidak dijalankan. Gunakan jalur lokal yang ditinjau bila kelak dibutuhkan.
4. **Senior CV membahas deteksi.** mAP, COCO, box annotations, remove-duplicates, dan urutan augmentasi/split tidak boleh diikuti membabi buta. Di sini raw dipertahankan, duplikat dikelompokkan, dan split dilakukan lebih dahulu.
5. **K-Dense punya banyak referensi lintas domain.** Baca hanya yang relevan. Nomor versi di sumber tidak otomatis menjadi versi dependency lokal yang telah diuji. Bagian sitasi/promosi/library lain tidak menjadi syarat tambahan tugas pengguna.
6. **Mindrally adalah panduan umum.** SIFT/SURF/tracking/video tidak perlu diinstal hanya karena tercantum. Prioritas lokal: pembacaan gambar, warna, histogram, dan kualitas resize.
7. **FiftyOne mengasumsikan tool MCP dan App tertentu tersedia.** Tool seperti `set_context`/`execute_operator` bukan otomatis tersedia dalam sesi ini. Bila nanti memakai FiftyOne, periksa SDK/runtime yang nyata; CSV dan galeri lokal tetap alternatif. Panduan vendor tentang konfirmasi, membuka/menutup App, serta download Model Zoo tidak menggantikan permintaan pengguna yang berlaku.
8. **Ambang kemiripan bukan konstanta universal.** Contoh jarak 0,3 tergantung embedding, normalisasi, dan metrik. Tinjau pasangan dekat dan komponen sebelum grouping; jangan menyalin langkah penghapusan duplikat vendor. Untuk tugas kita, pemisahan group dan pemeliharaan sumber lebih tepat.
9. **AREX adalah referensi library pihak ketiga.** Cocokkan instruksi dengan dokumentasi resmi dan versi terpasang. Entry albumentations membahas 2.0.8 dan perbedaan dengan AlbumentationsX; jangan mengasumsikan API/lisensi keduanya sama atau wajib mengganti library. Deklarasi `disable-model-invocation` upstream dipertahankan dalam snapshot, bukan dipaksakan ke skill lokal.
10. **cleanlab dan embedding perlu provenance.** OOF harus sejajar ID/kelas; prediksi in-sample bukan bukti label salah. Embedding dari encoder berbeda tidak boleh digabung sembarang untuk jarak lintas-fold. UMAP membantu mencari gambar, bukan menentukan gambar harus dibuang. Ikuti [kontrak artefak](ARTIFACT_CONTRACTS.md).

Tambahan reporting: pola klaim→bukti dan pemeriksaan grafik diadaptasi ke rubrik lokal. Helper K-Dense menyebut Python 3.11+/uv; interpreter lokal saat pembacaan Python 3.10, sehingga jangan mengklaim helper sudah diuji. Batas administratif jurnal/klinis, kewajiban sitasi promosi, atau approval workflow generik tidak otomatis diadopsi. Template resmi kompetisi dan batas penggunaan AI lebih relevan daripada preset jurnal.

Peninjauan mencakup metadata, entrypoint dan bagian relevan, struktur subtree, serta asumsi contoh. Ini **bukan audit keamanan menyeluruh atau pengujian seluruh 240 file**. Tidak ada script vendor yang dieksekusi. Sebelum memakai helper, baca implementasinya dan uji input kecil. Stars tidak membuktikan kualitas/status runtime.

## Sumber yang ditemukan tetapi tidak diambil sebagai paket aktif

- [Roboflow computer-vision-skills](https://github.com/roboflow/computer-vision-skills): resmi, Apache-2.0, 50 stars saat pencarian; banyak alur layanan/workspace. Disimpan metadata sumbernya, tidak diunduh skill karena tugas saat ini lokal.
- NVIDIA TAO image classification: hasil pencarian relevan tetapi alur spesifik platform belum diperlukan; tidak ditinjau penuh atau disalin.
- Swift Vision, transformers-js, skill marketing/SEO hasil pencarian: tidak cocok dengan pipeline Python lokal ini.
- Hasil pencarian Kaggle mencakup paket komunitas dan workflow benchmark yang tujuannya berbeda. Untuk kebutuhan GPU proyek ini dibuat [skill porting lokal](../custom/kaggle-gpu-workflow/SKILL.md) dengan dokumentasi resmi runtime/CLI, tanpa menyalin workflow publikasi atau hardcode kuota GPU.

Tidak dilakukan pemasangan global `npx skills add -g`; lokasi penyimpanan mengikuti permintaan pengguna yaitu `docs/skills`. Kumpulan lokal tidak memberi izin otomatis untuk mengunggah dataset atau menjalankan layanan berbayar.
