# Konteks dan kontrak data

**Riset/ablation terbaru6 November:** stage5 menguji fitur lokal dan32 prototipe (18 fitting/26 kondisi). Student stage4 tetap terpilih; belum ada pengganti atau fine-tuning encoder. Homogeneity32 membaik tetapi completeness turun; bukan peningkatan Accuracy. Lihat `reports/errors/stage5_material_structure_20261106/RESULTS_AND_DECISION.md`. Notebook utama tetap v3 yang sudah lulus10/10; eksperimen stage5 dijalankan sebagai skrip terpisah. Jumlah pembanding student fulltrain:1186 sama,164 berbeda,570 campuran.


**Klarifikasi pembanding:** pengguna meminta arsip dipakai untuk catatan perbedaan/ditinjau lagi. Antrean audit dipisahkan dari filter model. Seed-model-only: 1.233 kandidat, 686 cadangan, satu duplikat dipertahankan; pilot terpisah kini melatih empat head student dalam dua putaran tanpa label arsip sebagai input. Tampilan head terbaru bernama ada pada `data/processed/stage3_named_display_review/`, termasuk 0046 pada paper.

**Terbaru, 6 November ? EDA visual v3:** notebook utama kini sepuluh sel yang memuat preview train/test, distribusi nama sementara, kandidat crop, serta seluruh 480 tetangga test pada tingkat piksel. Head v2 dipakai ulang setelah verifikasi hash, bukan fitting ulang. Nama dari interpretasi asisten diminta pengguna untuk EDA pribadi, dicatat pada `configs/display_label_mapping.json`, dan tidak menjadi target training/validasi/submission. Kelompok campuran dipertahankan. Laporan: `reports/eda/stage3_visual_20261106/ANALISIS_VISUAL.md`; bukti: `reports/reproducibility/notebook_v3_delivery_validation.json`. Paragraf v2 di bawah adalah riwayat.

**Kebijakan repo terbaru:** kode, notebook beroutput, dan dokumen kecil diunggah; dataset, gambar terpisah, embedding/checkpoint, backup, serta dokumen besar tetap lokal. Versi v3 terbaru lulus 10/10 sel dengan nol error. Riwayat versi v2 sebelumnya: 35/35 sel, 177 output, nol error. Pratinjau HTML hasil terbaru tersedia lokal pada `reports/reproducibility/notebook_preview/01_EDA_Preprocessing_Clustering.html`.

**Tahap 2, 5 November 2026:** audit 590 perbedaan train/171 test selesai secara programatis; sampel teknis diperiksa visual. DINOv2 image-only diekstrak untuk2.400 gambar, delapan konfigurasi representasi dan enam head konsistensi tetangga diuji tanpa label. Head terpilih menurut loss validation tanpa label direfit pada train unik. Seluruh1.920 train dan480test memiliki delapan folder anonim baru di `data/processed/stage2_neighbor_v1/`. Manifest/hash lulus. Konsistensi perubahan resolusi253/256 versus baseline241/256 adalah kestabilan, bukan Accuracy. Mapping kategori aktif tetap belum tersedia. Laporan rinci: `reports/errors/stage2_20261105/ANALISIS_DAN_PENANGANAN.md`; arsip tidak dipakai fitting, selection, atau semantic mapping.

**Pembaruan 5 November 2026:** [aturan dan Q&A yang kini tersedia](RULES_AND_RUBRIC.md) menggantikan status lama. Label visual asisten serta `submission_99` menjadi arsip yang dikecualikan dari jalur kompetisi. Label aktif yang berasal dari metode peserta belum tersedia. Penamaan cluster oleh tim boleh; pelabelan/koreksi gambar manual satu per satu tidak boleh.

Tahap 1 kini dijalankan: 2.400 citra didekode/preprocess, audit hash, retrieval seluruh test, clustering non-VLM anonim, dan diagnostik domain. [Notebook](../../../notebook/01_EDA_Preprocessing_Clustering.ipynb) kini menggabungkan tahap 1+2 dan lulus Run All 35/35 sel kode, 0 error, 1.264,14 detik dari `.venv`; [bukti terbaru](../../../reports/reproducibility/notebook_v2_delivery_validation.json) memverifikasi raw dan enam eksekusi modul; [analisis](../../../reports/eda/stage1_20261105/ANALISIS_TAHAP_1.md) dan [hasil validasi](../../../reports/eda/stage1_20261105/final_validation.json) tersedia. Cluster bukan label material; mapping tim masih kosong. Klarifikasi pengguna mengizinkan bantuan penulisan kode tanpa panggilan LLM di notebook. Perbandingan arsip yang diminta dilakukan terpisah setelah assignment dibekukan, tidak memengaruhi jalur kompetisi.

## Fakta terverifikasi lokal

- Train: 1.920 gambar. Panitia tidak menyediakan label train, menurut klarifikasi yang dikirim pengguna. Delapan label dibuat melalui pengamatan visual asisten; **bukan ground truth resmi**.
- Test: 480 gambar. Kolom kategori pada sample merupakan contoh format, bukan jawaban test.
- Metrik kompetisi: **Accuracy**, berdasarkan deskripsi kompetisi yang ditempel pengguna. Macro-F1 dan recall per kelas adalah diagnostik pendamping dari webinar, bukan pengganti metrik kompetisi.
- Kontrak CSV: `image_id,waste_category`, ID test tiga digit, urutan persis sample, satu label per gambar.
- Train asli berada langsung dalam `data/Dataset DSC GelarRasa 2026/train/`; `train/label/` berisi salinan. Jangan menggabungkan kedua lokasi melalui recursive glob karena akan menghitung gambar dua kali.
- Arsip label lama: `docs/train_label_review/train_labels.csv`; bukti visual: `train_labels_audit.csv`; sumber/hash: `manifest.json`; pemeriksaan integritas: `final_audit.json`. Ini bukan label aktif untuk training, validasi, seed atau mapping nama cluster.
- Snapshot anotasi 4 November: cardboard 273, textile 219, metal 238, plastic 304, electronic 211, paper 215, foodwaste 248, glass 212. Ada 49 medium dan 2 low, yakni 0376 dan 1890. Ini keyakinan subjektif, bukan probabilitas.
- Satu kelompok duplikat byte-identik train: 0848 dan 1717. Near-duplicate belum diaudit menyeluruh.
- Arsip anotasi visual test lama sekarang berada di `docs/secret/submission_.csv`; hash identik dengan arsip awal `submission_99.csv`. Screenshot pengguna pada 5 November menunjukkan public Accuracy **0,9930**, dengan bagian public sekitar 30% menurut pengguna. Catatan sumber: `reports/public_scores/submission_99_user_report_20261105.json`. Ini bukan hasil yang diambil langsung via Kaggle/API. Jangan menjadikannya ground truth seluruh test/train, target validasi, atau bukti model terlatih. Private score dan identitas kesalahan belum diketahui.

Nama kelas dan urutan probabilitas kanonis disimpan di `configs/data_contract.json`. Urutan ini pilihan internal proyek; keluaran CSV tetap memakai nama kelas. `ImageFolder` biasanya mengurutkan folder secara alfabetis: cocokkan `class_to_idx` secara eksplisit sebelum membandingkan/ensemble.

## Informasi kompetisi dan tingkat kepastian

| Informasi | Sumber | Status |
|---|---|---|
| Delapan kelas, 1.920/480, Accuracy, CSV | Deskripsi pengguna; salinan di `sources/competition_description_user.txt` | Tersedia |
| Peserta menentukan label berdasar karakteristik citra | Screenshot jawaban panitia dalam percakapan | Tersedia; jangan mengubahnya menjadi klaim label resmi |
| Pretrained non-VLM | Handbook h.7 dan jawaban resmi Padlet | Base diperbolehkan; pelatihan/fine-tuning peserta pada train kompetisi diperlukan |
| Data eksternal, inference API, LLM/VLM | Handbook h.7; Q&A SigLIP | Dilarang pada lingkup yang dirinci dalam aturan aktif; VLM termasuk feature extraction |
| Label tiap gambar manual | Q&A penamaan/koreksi cluster | Tidak diperbolehkan; hasil program clustering dengan nama cluster oleh peserta menjadi jalur yang dijelaskan panitia |
| Q&A resmi | Snapshot 5 November: 19 posting, 31 komentar | Diarsipkan lengkap; 17 balasan panitia dibedakan dari komentar peserta |
| Subclustering/koreksi otomatis train | Pertanyaan terbaru belum mendapat jawaban | Jangan menganggap pertanyaan sebagai izin |
| Test pseudo-label | Izin eksplisit belum ditemukan dalam sumber yang dibaca | Jangan memperluas izin clustering train menjadi izin pseudo-label test |
| Banyak skor 1,0000 | Screenshot pengguna | Tidak membuktikan seluruh test benar atau juara akumulasi; Kaggle 70, laporan 20, notebook 10 |

Jangan mengimpor aturan DSCO-BINUS atau metrik KS LOGIKA-UI. Gunakan sumber GELAR-RASA terbaru. Penelitian, pengelolaan sumber, dan penulisan panduan tidak sama dengan menjalankan model/klasifikasi memakai LLM. Jika pengguna membawa klarifikasi resmi baru, simpan sumber dan perbarui status; jangan meminta konfirmasi yang sama lagi. Tabel saat ini mencatat aturan yang sudah eksplisit dan pertanyaan yang memang belum dijawab, bukan larangan tambahan buatan skill.

Paket final perlu menunjukkan model peserta dan pelatihan pada data kompetisi. Jangan mengganti nama anotasi lama menjadi hasil clustering/training. Anotasi test lama hanya arsip asal-usul, tidak digunakan memilih atau memperbaiki prediksi. Aturan dasar Kaggle yang ditempel juga melarang submission menggabungkan hasil hand-labeling/human prediction validation/test.

## Kesalahan yang harus dihindari

Angka kelas dalam slide bukan label resmi dan tidak boleh dijadikan kuota. Kesamaan nama kelas dengan dataset online bukan alasan mengambil data luar. Skor sempurna tidak membuktikan cara peserta lain bekerja. Evaluasi terhadap target turunan dari clustering mengukur agreement dengan metode tersebut, bukan kebenaran semantik yang terjamin. Jangan melaporkan validasi terhadap anotasi asisten lama sebagai hasil kompetisi.
