# Pembaruan Q&A panitia — 7 November 2026

Kedua tautan pengguna sudah dibaca melalui endpoint publik Padlet. Snapshot lengkap kini memuat **24 posting, 38 komentar, dan 23 jawaban resmi**, dibandingkan snapshot sebelumnya yang berisi 19/31/17. Terdapat lima posting baru serta tujuh komentar baru, enam di antaranya dari akun pembuat papan/panitia. Seluruh halaman posting dan komentar selesai diambil. Profil peserta, sesi, cookie, token, dan URL media bertanda tangan tidak disimpan.

## Jawaban yang paling relevan

Pada [pertanyaan enam poin](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/kxodWG5OBD7gQgP7), panitia menjawab pada **7 November 2026 pukul 20.25 WIB**:

| Topik | Ketentuan yang dijelaskan | Dampak pada pekerjaan |
|---|---|---|
| Manual train | Lima gambar per kelas, maksimal 40; boleh ditulis di notebook | Ledger tetap 34 terpakai, sisa metal dua dan foodwaste empat |
| Nama cluster | Nama folder boleh sebagai identitas cluster | Bukan izin tersurat untuk koreksi individual tanpa batas |
| Pelabelan otomatis | Pseudo-labeling, kNN, label propagation, self-training boleh memakai train panitia | Eksperimen berikut dapat menempuh jalur ini dengan provenance yang jelas |
| Pretrained | Model CV non-VLM seperti ResNet, EfficientNet, DINOv2 boleh untuk feature extraction/fine-tuning secara lokal | Frozen representation dan head lokal mempunyai dasar izin eksplisit |
| VLM | CLIP dan prompt teks tidak diperbolehkan | Tidak memakai CLIP/SigLIP atau image encoder VLM sebagai jalan pintas |
| Data eksternal | Dataset selain panitia, termasuk TrashNet/RealWaste, tidak boleh | Tidak mengunduh dataset tambahan untuk training; bedakan dataset dengan bobot pretrained yang diizinkan |

Jawaban [do3MQJdz2Gx6W15w](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/do3MQJdz2Gx6W15w) mengizinkan subclustering dengan jumlah sesuai strategi, cleaning otomatis, dan perbaikan train melalui model. Jawaban [PVKBQO0wN9pkZj5x](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/PVKBQO0wN9pkZj5x) juga mengizinkan penanganan noise/ambiguitas sebagai cleaning.

## Kronologi dan batas aktif

- **Gunakan jawaban resmi terbaru sebagai patokan.** Pertanyaan confidence lama diajukan 5 November pukul 15.35 WIB dan dijawab 23.16 WIB. Pertanyaan do3 diajukan 4 November pukul 23.45 WIB dan dijawab 5 November pukul 23.17 WIB. Pertanyaan enam poin terbaru diajukan **7 November pukul 09.22 WIB**, lalu dijawab **20.25 WIB**. Baik waktu pertanyaan maupun jawabannya lebih baru. Untuk workflow ini, izin otomatis terbaru yang dipakai; larangan confidence lama hanya disimpan sebagai riwayat dan tidak menghalangi metode yang sekarang jelas diizinkan. Pengguna juga menegaskan cara membaca kronologi tersebut.
- Bantuan AI untuk anotasi/pelabelan dilarang dalam jawaban [zV61Q6ENleERaO98](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/zV61Q6ENleERaO98) dan [YDgnZeRkJxVLWwrA](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/YDgnZeRkJxVLWwrA). Simpan batas ini bersama izin model CV otomatis terbaru. Jangan mengambil label visual asisten lama sebagai target. Otorisasi bantuan kode dari pengguna tidak dilaporkan sebagai izin baru panitia.
- Pertanyaan baru [PR3NWxylnN1yab0O](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/PR3NWxylnN1yab0O) tentang image encoder VLM saja belum dijawab. Jawaban VLM lama tetap melarangnya, termasuk feature extraction.
- Pertanyaan [KxJvagRAe63VWAg0](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/KxJvagRAe63VWAg0) mengenai test tanpa label untuk clustering/normalisasi/TTA belum dijawab. Test tetap tidak masuk fit; transformasi inferensi tetap berbeda dari fitting parameter pada test, dan jangan mengarang jawaban panitia untuk pertanyaan tersebut.

## Dampak pada target 45 kasus

Kelanjutan tetap membangun metode umum pada train: seed terbatas, representasi non-VLM, subclustering, kNN/propagation atau self-training. Daftar 45 kasus hanya audit setelah kandidat dibekukan. Tidak ada izin menyalin seluruh review menjadi target atau membuat aturan per ID agar semua tampak benar. Tidak ada tambahan jatah manusia atau perubahan data/assignment hanya karena pembaruan sumber ini.

Snapshot: [padlet_qna_20261107.json](../rules/sources/padlet_qna_20261107.json). Pedoman aktif: [RULES_AND_RUBRIC.md](../skills/references/RULES_AND_RUBRIC.md). Sumber lama tetap dipertahankan untuk membandingkan perubahan.
