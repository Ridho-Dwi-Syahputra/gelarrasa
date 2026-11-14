# Aturan dan rubrik aktif — diperiksa 5 November 2026 WIB

## Pembaruan Q&A resmi — 7 November 2026, 20.25 WIB

[Jawaban pada pertanyaan enam poin](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/kxodWG5OBD7gQgP7) diperiksa melalui endpoint publik Padlet. [Snapshot terbaru](../../rules/sources/padlet_qna_20261107.json) mencakup seluruh 24 posting dan 38 komentar, dengan 23 jawaban akun pembuat papan/panitia. Pagination posting selesai; seluruh komentar per posting dibaca. Snapshot sebelumnya tetap historis.

- **Manual:** maksimal lima gambar per kelas, total maksimal 40. Label tersebut boleh ditulis langsung di notebook. Jatah tidak menjadi 40 bebas lintas kelas. Pencatatan konservatif proyek tetap berdasarkan sumber unik yang pernah dipakai.
- **Otomatis:** pseudo-labeling, kNN, label propagation, dan self-training secara eksplisit diperbolehkan pada train panitia, tanpa test, dataset eksternal, atau model/layanan AI yang dilarang. Ini sekarang memiliki sumber panitia langsung, bukan hanya interpretasi proyek.
- **Pretrained non-VLM:** ResNet, EfficientNet, DINOv2 dan model computer vision non-VLM boleh dijalankan lokal untuk feature extraction atau fine-tuning. Jangan membaca jawaban Hugging Face lama sebagai larangan seluruh frozen feature extraction. Model siap pakai untuk langsung menebak test tanpa pengembangan/pelatihan lokal tetap bukan jalur yang disetujui jawaban lama.
- **Tetap dilarang:** CLIP/VLM, termasuk zero-shot prompt teks; dataset eksternal seperti TrashNet/RealWaste; penggunaan test dalam training. Pertanyaan baru tentang image encoder VLM saja belum dijawab dan bukan izin untuk memakainya.
- **Nama cluster:** jawaban terbaru mengizinkan nama folder sebagai identitas cluster. Tidak menjelaskan tegas penghitungan kuota untuk penamaan semantik seluruh anggota; jangan menganggapnya celah untuk menyamarkan koreksi individual atau cluster satu gambar.
- **Anotasi dengan bantuan AI:** jawaban 5 November pada [zV61Q6ENleERaO98](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/zV61Q6ENleERaO98) dan [YDgnZeRkJxVLWwrA](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/YDgnZeRkJxVLWwrA) melarang bantuan AI untuk anotasi/pelabelan. Simpan ini bersama izin eksplisit metode otomatis dan non-VLM terbaru; jangan mengubah arsip prediksi visual asisten menjadi ground truth. Otorisasi bantuan kode dari pengguna tidak otomatis menjadi izin panitia untuk seluruh penggunaan AI.

**Patokan aktif adalah klarifikasi resmi terbaru.** Perhatikan waktu pertanyaan dan waktu jawabannya, bukan urutan tampil atau tanggal pengambilan snapshot. Pertanyaan [LNV1Q7K2x7boQmq3](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/LNV1Q7K2x7boQmq3) diajukan 5 November pukul 15.35 WIB dan dijawab 23.16 WIB dengan larangan confidence individual. Pertanyaan [do3MQJdz2Gx6W15w](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/do3MQJdz2Gx6W15w) diajukan 4 November pukul 23.45 WIB dan dijawab 5 November pukul 23.17 WIB dengan izin perbaikan otomatis. Pertanyaan enam poin terbaru diajukan **7 November pukul 09.22 WIB** dan dijawab **20.25 WIB**, secara eksplisit mengizinkan metode pelabelan otomatis dan pretrained non-VLM. Gunakan ketentuan 7 November untuk metode yang telah dijelaskannya; jawaban confidence lama hanya riwayat dan tidak menjadi larangan aktif yang menghalangi izin terbaru. Instruksi pengguna terbaru juga menegaskan prioritas kronologis ini. Jangan memperluas izin menjadi penggunaan VLM, test atau dataset eksternal yang tetap dilarang.

Pertanyaan baru tentang test tanpa label untuk fitting clustering/normalisasi/TTA belum dijawab. Tidak ada izin memakai test dalam fit; pisahkan transformasi inferensi tetap dari estimasi parameter pada test. Subclustering dan jumlah cluster sesuai strategi, serta cleaning noise, mempunyai jawaban izin pada posting do3MQJdz2Gx6W15w/PVKBQO0wN9pkZj5x. Tetap tanpa koreksi massal dari arsip.

Ringkasan perubahan dan dampak terhadap pekerjaan aktif ada pada [catatan Q&A](../../agent-notes/QNA_UPDATE_20261107.md). Ketentuan historis di bawah tidak boleh menimpa pembaruan terbaru ini.

## Pembaruan resmi diverifikasi 7 November 2026

Handbook terbaru berhasil diunduh dari tautan bit.ly resmi menjadi [snapshot 9 halaman](../../rules/sources/handbook_latest_20261107.pdf). Hash SHA-256: 7380e15d7f511d118dea7a891ff730b77fcd6852104bcd65f1693acfb028960c. Pengumuman yang ditempel pengguna cocok dengan halaman 7-8 PDF terbaru.

- Halaman 7: hand labeling train tidak disarankan, tetapi diizinkan maksimal **5 gambar per kelas**. Larangan total koreksi manual dalam snapshot Q&A lama di bawah digantikan oleh batas baru ini untuk label manusia pada train.
- Maksimal 40 acuan manusia untuk delapan kelas; batas dicatat atas ID sumber unik secara keseluruhan, bukan direset per fold/augmentasi. Ini cara pencatatan konservatif proyek, bukan rincian tambahan yang dikutip dari panitia.
- Hand labeling test dilarang. Data test dilarang dalam training; graph, PCA/scaler, prototype, pemilihan hyperparameter dan pseudo-label training harus hanya memakai train sesuai partisi. Validation internal tidak masuk graph fit pada evaluasi yang dilaporkan.
- Halaman 8: satu notebook meliputi seluruh alur; rerun menghasilkan skor submission yang sama dan dataset harus dapat diakses panitia. Paket aktif saat ini masih EDA/preprocessing, belum notebook final training-submission, dan akses dataset panitia belum dipublikasikan oleh pekerjaan ini.
- Larangan AutoML/LLM/VLM/generative AI dan inference pihak ketiga tetap ada pada halaman 8. Izin 5 label manusia tidak mengubah label visual asisten lama menjadi label manusia. Label yang akan dipakai harus benar-benar diperiksa anggota tim dan asalnya dicatat.
- Arsip label lama hanya audit pasca-pembekuan. Tidak menyalin 174 perbedaan ke target atau menyetel metode dari kecocokan arsip. Riset/probe boleh memakai seed konsensus program, tetapi wajib disebut seed otomatis, bukan hasil pemeriksaan tim.

Penjelasan historis berikut tetap disimpan untuk provenance; pada konflik jumlah hand label gunakan pembaruan ini. Klarifikasi pengguna sebelumnya tentang bantuan kode tanpa panggilan model bahasa tetap dicatat sebagai sumber pengguna.

Dokumen ini menggantikan status aturan yang belum diketahui dalam catatan 4 November. Sumber: [handbook 8 halaman](<../../rules/HANDBOOK PRELIMINARY ROUND DSC GELAR RASA 2026.pdf>), [aturan Kaggle yang diberikan pengguna](../../rules/sources/competition_rules_user_20261105.txt), [Q&A resmi Padlet](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk), dan [template resmi](https://docs.google.com/document/d/1mtwAMvd3a8Zn7zr4fo9gZtsT3zWDdr_7bgvwA77xrEE/edit).

Snapshot [Q&A](../../rules/sources/padlet_qna_20261105.json) memuat seluruh 19 posting dan 31 komentar yang tersedia saat pengambilan, termasuk 17 jawaban akun panitia; pagination keduanya selesai. Pertanyaan peserta dan komentarnya bukan keputusan panitia. Q&A dapat bertambah sesudah snapshot; periksa pembaruan untuk keputusan yang bergantung aturan. Checksum sumber ada di [manifest sumber](../../rules/sources/source_manifest.json).

## Penilaian preliminary

| Komponen | Bobot maksimum | Bukti sumber |
|---|---:|---|
| Kaggle | 70 | Handbook halaman 4; di dalamnya private 70%, public 30% |
| Laporan analisis | 20 | Halaman 4–5 |
| Notebook analysis | 10 | Halaman 5–6 |

Laporan: struktur/kerapian 5; proses analisis dan insight 3; alur sistem 3; metode/alasan model 3; hasil/performa 3; pengembangan/implementasi lanjutan 3. Notebook: kejelasan interpretasi 5, kualitas/kerapian kode 5. Lima tim lolos berdasarkan akumulasi komponen, bukan public leaderboard saja (halaman 4).

Secara aritmetika, bobot relatif private/public terhadap keseluruhan adalah 49/21 bila komponen Kaggle dibobotkan langsung. **Rumus konversi skor/ranking Kaggle menjadi poin tidak dirinci**, jadi jangan menyatakan `49*accuracy_private + 21*accuracy_public` sebagai formula resmi. Public 1,0000 tidak membuktikan semua 480 test benar atau menjamin peringkat satu. Aturan dasar Kaggle yang ditempel menyebut private ranking dan waktu submission lebih awal untuk tie; jangan mengasumsikan itu menjelaskan tie pada akumulasi penilaian panitia tanpa klarifikasi.

## Batas metode yang memengaruhi proyek

| Topik | Ketentuan yang ditemukan | Implikasi |
|---|---|---|
| Data | Dataset panitia saja; data eksternal dilarang (handbook h.7, aturan 4) | Referensi paper/library bukan izin menambah gambar eksternal |
| Pretrained | Boleh dijalankan sendiri; Q&A mengizinkan non-VLM ResNet/EfficientNet | Jangan menyamakan pretrained dengan larangan semua bobot eksternal |
| Model siap pakai | Q&A `post_lkROZP94G6A2WjMg`: Hugging Face sebagai base; wajib fine-tuning dengan train kompetisi; model jadi tanpa pelatihan untuk langsung test tidak boleh | Frozen/zero-shot bukan jalur final yang otomatis diterima |
| LLM/AutoML/generative AI | Dilarang untuk membuat model, klasifikasi, atau prediksi terhadap data kompetisi (h.7, aturan 9) | Tidak memakai anotasi/prediksi LLM lama sebagai target training, seleksi, atau submission |
| VLM | Q&A `post_NvylWExOjy4wZ0OX`: SigLIP/VLM dilarang, termasuk feature extraction dan tahapan lain | CLIP/SigLIP bukan kandidat aktif, walau dijalankan lokal |
| Inference API | Layanan/API pihak ketiga untuk inference/klasifikasi/deteksi/prediksi dilarang (h.7, aturan 8) | Komputasi sendiri berbeda dari mengirim gambar ke API model |
| Penamaan cluster | Q&A `post_4b3zaM6e0xkrQ2j7`, `post_BJkrQA1NxgEgQEge`: nama cluster boleh manual setelah clustering model peserta | Nama kategori diberikan oleh anggota tim; agent tidak mengklasifikasikan gambar/cluster kompetisi |
| Label tiap gambar | Q&A `post_BJkrQA1NxgEgQEge`, `post_lkDVaKDL3deGQPp9`: pelabelan, koreksi, atau pemindahan keanggotaan satu per satu secara manual tidak boleh | Jangan menjadikan folder label lama sebagai dasar lalu menyebutnya hasil clustering |
| Sudah telanjur manual | Q&A `post_O7A9QmRkJMe3Z6x3`: panitia menyarankan pemodelan ulang | Bangun jalur baru dengan provenance yang sebenarnya; jangan menghapus riwayat |
| Kelanjutan supervised | Q&A `post_lkDVaKDL3deGQPp9`: hasil clustering berlabel melalui metode/program dapat menjadi dasar model klasifikasi | Evaluasi terhadap target turunan tetap bukan semantic ground truth resmi |
| Cleaning/drop train | Q&A `post_R7dXadKX1MDmW6bl` menyerahkan penggunaan data/penanganan noise pada strategi peserta | Simpan alasan dan exclusion mask; bukan izin koreksi manual per gambar |

Belum terjawab dalam snapshot: rincian subclustering/jumlah cluster, penggantian label otomatis oleh model, dan pertanyaan anotasi dengan bantuan AI. Pertanyaan tersebut **bukan izin**. Izin umum preprocessing tidak membatalkan larangan eksplisit. Ruang bantuan LLM untuk menulis kode model atau naskah laporan tidak diperjelas Q&A; aturan 9 mencakup “membuat model”. Tahap saat ini membaca aturan dan menyiapkan panduan, bukan menjalankan pembuatan model atau klasifikasi kompetisi dengan LLM.

## Dampak langsung terhadap artefak lama

Klarifikasi tambahan pengguna, 5 November 2026: “kalau kode gapapa, asalkan ga manggil llm di notebooknya”. Ini otorisasi bantuan penulisan kode dan sumbernya adalah percakapan pengguna; belum merupakan kutipan baru dalam snapshot Q&A. Implementasi EDA/clustering dapat memakai algoritma lokal tanpa panggilan LLM/VLM. Asal anotasi lama tetap tidak boleh disamarkan atau menjadi target/pemetaan aktif. Perbandingan arsip yang secara eksplisit diminta pengguna hanya diagnostik terpisah setelah assignment dibekukan; tidak memengaruhi fitur, pemilihan k, cluster, nama kategori, atau submission.

`arsip/sebelum_rapi_20261107/docs/train_label_review/` dan `train/label/` berisi anotasi visual asisten, bukan label yang berasal dari clustering peserta. `arsip/sebelum_rapi_20261107/docs/secret/submission_99.csv` juga anotasi visual test. Simpan sebagai **arsip yang tidak dipakai dalam jalur kompetisi**. Jangan digunakan untuk seed label, pemetaan nama cluster, skor validasi, training, atau pemilihan prediksi; jangan menyamarkan asalnya sebagai human labels atau cluster outputs. Keberhasilan hash/integritas tidak menyatakan artefak itu memenuhi aturan.

Raw train/test tetap dipertahankan. Manifest baru untuk label hasil metode peserta harus terpisah dan merekam feature extractor, transform, clustering run, mapping cluster-ke-kelas, serta perubahan algoritmis. Status label aktif saat aturan ini dicatat: **belum tersedia**. Tidak ada label baru yang dibuat oleh penulisan skill.

## Pengumpulan dan jadwal

- Proyek: 31 Oktober–14 November 2026; tenggat **14 November 2026 pukul 23.59 WIB** (halaman 2, 8). Penilaian 15–20 November; pengumuman 22 November; final 28 November (timeline halaman 2).
- Maksimal 3 submission per hari; reset **07.00 WIB** menurut jawaban resmi `post_e9YpQNXwMxLlWxjM`.
- Tim 2–3 orang, nama tim/akun sesuai registrasi, merge sebelum submit. Tidak bertukar solusi antartim.
- Tiga berkas: `DSC2026_Submission_NamaTim_NamaKetua.csv`, `DSC2026_Notebook_NamaTim_NamaKetua.ipynb`, `DSC2026_Laporan_NamaTim_NamaKetua.pdf`.
- Pengumpulan oleh ketua/perwakilan melalui [form panitia](https://bit.ly/PreliminarySubmissionDSCGelarRasa2026). Akses berkas tidak boleh terkunci. Tidak ada form yang diisi atau dikirim dalam tugas ini.
- Template wajib: PDF A4, Times New Roman 12 pt, spasi 1,15, justify, APA 7. Maksimum **10 halaman isi**, di luar cover, daftar isi/gambar/tabel, daftar pustaka, dan lampiran. Buat salinan Google Docs untuk pengerjaan; jangan meminta akses edit dokumen induk. Salinan unduhan lokal bukan klaim telah membuat salinan di akun Google.

Struktur/template dan pemetaan bukti ke rubrik ada di [REPORT_BLUEPRINT](REPORT_BLUEPRINT.md). Gunakan handbook/aturan/Q&A sebagai sumber keputusan; materi webinar dan skill vendor tidak dapat mengizinkan metode yang dilarang panitia.
