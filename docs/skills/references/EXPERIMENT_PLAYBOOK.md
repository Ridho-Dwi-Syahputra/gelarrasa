# Cara berpikir dan urutan kerja

**Revisi 5 November:** aturan aktif ada di [RULES_AND_RUBRIC](RULES_AND_RUBRIC.md). Urutan ini panduan konseptual bagi peserta, bukan pelaksanaan pembuatan model oleh LLM. Anotasi asisten lama dikecualikan, termasuk untuk memilih/nama cluster atau menghitung validasi.

## Sasaran dan pertanyaan awal

Kejar Accuracy kompetisi tertinggi dengan pipeline yang dapat dijelaskan dan diulang. Jangan menyamakan jumlah model dengan mutu, atau kenaikan satu fold dengan kenaikan leaderboard. Tanyakan: kesalahan mana yang bisa diperbaiki, bukti apa yang mendukung, dan eksperimen termurah apa yang dapat membedakan penjelasannya?

Fakta awal: raw train tanpa label resmi; anotasi visual lama tidak menjadi input aktif; duplikat exact ada; grouping belum lengkap. Hipotesis yang perlu diuji peserta: struktur sumber, detail saat resize, latar dominan, dan representasi clustering. Kesimpulan semantik tidak boleh diasumsikan dari skor agreement terhadap target turunan sendiri.

## Tahap prioritas

1. **P0 — Kontrak dan aturan.** Periksa raw ID/RGB/EXIF dan provenance. Tandai label visual lama sebagai arsip yang dikecualikan. Output: manifest raw dan status label aktif.
2. **P1 — Grouping dan validasi.** Kandidat SHA-256/pHash/embedding → konfirmasi visual → group_id. Kunci fold, cukup kelompok setiap kelas. Bandingkan evaluasi random sebagai diagnostik saja bila perlu, bukan memilih split yang skornya paling tinggi.
3. **P2 — Pelabelan dengan metode peserta.** Bandingkan rancangan fitur non-VLM dan clustering. Tim menamai cluster; simpan assignment dan mapping. Jangan memasukkan label manual lama. Setelah itu baru classifier baseline pada target turunan; laporkan keterbatasan evaluasinya.
4. **P3 — Transfer learning.** EfficientNet-B0/MobileNetV2 head-only → partial/full fine-tuning. ConvNeXt-Tiny/EfficientNetV2-S sebagai pembanding setelah baseline. Ukur kontribusi resize, augmentasi, dan model secara terpisah.
5. **P4 — Hybrid dan diagnosis.** Uji embedding-only, HOG/warna-only, gabungan bila relevan. Semua backbone non-VLM, training peserta, tanpa inference API. Jangan mengubah target per gambar agar cocok dengan prediksi.
6. **P5 — Gabungan terkendali.** Rata-rata probabilitas 2–3 model dengan kesalahan berbeda; kalibrasi/TTA diuji pada OOF/inner-validation. Pseudo-label hanya cabang tersendiri bila aturan memperbolehkan.
7. **P6 — Paket akhir.** Kunci konfigurasi, refit sesuai kebijakan terpilih, inferensi dari checkpoint, validasi CSV, simpan provenance. Pilih final dengan bukti lintas fold dan perubahan per gambar.

Semua tahap di atas **belum dijalankan sebagai training pada pekerjaan pengumpulan skill ini**. Perencanaan laporan/rubrik berlangsung paralel; hasilnya diisi dari artefak nyata. Anotasi lama bukan label aktif yang siap training.

## Kartu eksperimen

Setiap run menjawab satu hipotesis. Simpan: `run_id`, `parent_run`, `hypothesis`, `one_change`, `label_version`, `manifest_sha256`, `fold_version`, `seed`, `model/pretrained_tag`, transform, loss/optimizer, runtime, hardware, versi library, Accuracy, macro-F1, error count, OOF path, dan keputusan lanjut/tolak.

Nama: `exp001_hog_color_svm`, `exp002_effb0_frozen`, dst. Angka mengurutkan percobaan, bukan menyatakan kualitas. Training gagal harus ditulis gagal, bukan dihapus dari ledger. Perubahan label atau fold membutuhkan pengulangan pembanding relevan; jangan membandingkan angka berlainan target seolah ekuivalen.

## Membaca hasil

Bandingkan paired correctness pada ID validation sama, bukan rata-rata skor saja. Laporkan model A benar/B salah dan kebalikannya. Dengan seluruh 480 gambar, satu kesalahan setara 1/480 ≈ 0,002083 Accuracy; ukuran public subset belum diketahui, sehingga langkah skor public tidak dapat dihitung dari angka ini.

Perubahan 1–2 prediksi dapat rapuh; periksa stabilitas seed dan unit sumber dengan prosedur yang sesuai aturan. Bootstrap/repeat CV tidak menghapus bias pemilihan banyak kandidat. Outer evaluation dijelaskan di [ARTIFACT_CONTRACTS](ARTIFACT_CONTRACTS.md), tetapi target cluster tetap proxy. Jangan memilih fold termudah atau mengoreksi target per gambar untuk membenarkan model.

## Kondisi berhenti per eksperimen

Hentikan perluasan ketika hipotesis sudah terjawab, konfigurasi gagal pemeriksaan data, atau tidak ada manfaat yang bertahan pada pembanding yang setara. Tidak ada target angka internal yang menjamin top 1. Jika banyak skor sama sempurna, aturan tie-break/final perlu dibaca; jangan menyimpulkan urutannya dari screenshot.
