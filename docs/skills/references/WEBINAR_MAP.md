# Materi webinar menjadi keputusan kerja

Nomor halaman berikut adalah halaman PDF, bukan angka footer slide. PDF asli tetap di `docs/materi/`. Ekstraksi teks lokal disimpan untuk pencarian; gambar/diagram tetap dirujuk ke PDF.

**Batas penerapan per 5 November 2026:** [aturan dan Q&A](RULES_AND_RUBRIC.md) mengatur penggunaan materi ini. Human review dalam webinar bukan izin pelabelan/koreksi manual tiap gambar, dan bukan izin LLM/VLM melakukan klasifikasi. Label aktif harus memiliki asal-usul dari metode peserta; arsip anotasi visual asisten tidak dipakai. Penamaan cluster oleh tim diperbolehkan sesuai Q&A.

## Materi 1 — Intelligence in Action, 37 halaman

| Halaman | Pokok materi | Penerapan untuk GELAR RASA |
|---|---|---|
| 4–8 | Klasifikasi delapan kelas; label dan quality assurance sebelum model | Catat asal-usul hasil clustering, penamaan cluster oleh tim, serta label_version; tanpa koreksi manual tiap gambar |
| 9–10 | Contoh distribusi dan variasi objek | Jadikan konteks, jangan paksakan jumlah kelas slide ke anotasi |
| 11–13 | RGB, dtype, normalisasi, split, augmentasi | Manifest lengkap; split per sumber; transform train terpisah dari validation |
| 14 | HOG + histogram warna + SVM, transfer learning, hybrid | Tiga baseline terpisah dengan fold dan versi label sama |
| 16, 21 | Freeze backbone, latih head, kemudian fine-tune LR kecil | Uji bertahap; atur BatchNorm dan simpan checkpoint terbaik |
| 17–18 | Embedding CNN + HOG/warna | Ablasi embedding-only melawan tambahan fitur; dimensi/normalisasi tiap blok dicatat |
| 19 | Satu ID harus cocok satu file | Jangan recursive-glob train asli plus salinan `label/` |
| 20 | Augmentasi ringan; hindari crop/flip destruktif | Tinjau contoh transform sebelum training; tekstur material harus tetap terlihat |
| 22–26 | Learning curves, leakage, tuning, callbacks | Early stopping hanya pada validation; preprocessing fit di fold train |
| 25 | Resolusi kandidat 160/192/224 | Angka 640 pada contoh lain bukan kewajiban; pilih melalui uji detail vs biaya |
| 27–29 | Accuracy, Macro-F1, confusion matrix; satu faktor per eksperimen | Accuracy utama sesuai deskripsi kompetisi; laporkan pasangan error dan jumlah kesalahan |
| 30 | Grad-CAM, audit latar/watermark | Bukti diagnostik, bukan bukti kausal atau jaminan generalisasi |
| 31–35 | PILAH, human review, recovery/contamination/biaya/energi | Presentasikan dampak sebagai terukur atau skenario, jangan mengarang penghematan |

Rujukan: [PDF Materi 1](<../../materi/Materi 1 (Intelligence in Action_ Machine Learning for Sustainable Solution).pdf>).

## Materi 2 — Seeing is Believing, 38 halaman

Halaman 6–9 menghubungkan data dengan dampak dan bentuk visual; 12–19 membahas pemilihan grafik, baseline, kontras, dan clutter; 20–26 membahas audiens serta kebutuhan stakeholder; 33–34 memberi kriteria kelengkapan, konsistensi, akurasi, dan integritas.

Penerapan: tampilkan distribusi label dengan jumlah absolut, confusion matrix dengan nama kelas dan denominator, galeri kesalahan beserta alasan, serta kurva performa terhadap runtime. Bedakan jumlah gambar dari tonase limbah. Satu foto tidak memberi informasi massa atau pengurangan emisi tanpa asumsi tambahan. Gunakan sumbu nol untuk perbandingan panjang bar; jelaskan skala lain bila digunakan.

Rujukan: [PDF Materi 2](<../../materi/Materi 2 (Seeing is Believing_ The Art of Data Visualization for Sustainability).pdf>).

## Notulensi dan PPT acara

[Notulensi, 4 halaman](<../../materi/Notulensi Webinar Gelar Rasa 2026.pdf>) merangkum baseline → transfer → hybrid, evaluasi dan human review. Jawaban early stopping tidak berarti validation boleh dimasukkan ke training sambil tetap disebut validation independen. Refit semua train merupakan tahap terpisah setelah konfigurasi dipilih.

[PPT acara, 81 halaman](<../../materi/PPT Webinar Gelar Rasa 2026.pdf>) memuat konteks, jadwal acara, tata tertib, dan pengantar materi. Bukan bukti hasil model pada dataset kita. Diagram contoh kelas yang tampak tidak konsisten pada materi bukan label ground truth.

Larangan augmentasi validation/test dalam slide diterapkan pada evaluasi standar. TTA adalah cabang inferensi terpisah yang opsional: hanya diuji setelah baseline, deterministik, label-preserving, dan dilaporkan terpisah; jangan menyamarkannya sebagai evaluasi satu-view standar.
