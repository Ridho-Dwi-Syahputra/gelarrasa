---
name: image-ensemble-pseudolabel
description: Menggabungkan probabilitas model citra berdasarkan kesalahan OOF yang saling melengkapi dan menguji pseudo-label secara terpisah dengan provenance serta validasi yang terjaga.
---

# Ensemble dan pseudo-label

Selaraskan ID string, class_order dari metadata model, dan label_version; periksa provenance fold. Sample CSV tidak menentukan urutan delapan kolom probabilitas. Jangan merata-ratakan angka class ID atau logits yang skalanya tidak setara. Mulai rata-rata probabilitas 2–3 model yang terbukti memperbaiki error berbeda. Banyak model mirip tidak otomatis memberi informasi baru.

Pilih bobot pada inner validation; jangan melakukan pencarian besar pada OOF lalu melaporkan skor maksimum sebagai estimasi independen. Stacking membutuhkan outer evaluation yang membangun ulang inner-OOF model dasar hanya dari outer-train. Cross-validation meta-model pada satu matriks OOF global saja masih dapat terkontaminasi. Gunakan [resep lengkap dan kasus fold berbeda](../../references/ARTIFACT_CONTRACTS.md). Averaging dengan bobot yang sudah ditetapkan tidak otomatis salah hanya karena fold berbeda, tetapi setiap prediksi harus benar-benar out-of-group dan provenance tetap dicatat. Jangan mengganti nama fold untuk menyamarkan perbedaan. Bandingkan jumlah salah serta ID yang berubah.

Kalibrasi dapat berguna sebelum averaging; suhu positif tunggal tidak mengubah argmax model tunggal. Baca [kalibrasi](https://proceedings.mlr.press/v70/guo17a.html). TTA harus mempunyai pembanding single-view.

Pseudo-label adalah cabang opsional setelah aturan test/transductive jelas. Awali eksperimen label-masking pada train bagian dalam; validation luar tetap tidak dilatih, termasuk secara unsupervised. Gunakan teacher dari train split, confidence/consistency yang dikalibrasi tanpa validation luar, bobot pseudo-label terbatas, serta laporan error per kelas. [FixMatch](https://arxiv.org/abs/2001.07685) adalah rujukan konsistensi, bukan jaminan manfaat.

Jangan memakai anotasi visual test lama sebagai ground truth, memaksakan 60 gambar per kelas, atau mengubah prediksi untuk menebak hidden labels dari feedback leaderboard. Keluaran: bobot ensemble, provenance tiap model/pseudo-label, OOF pembanding, dan ablation. Simpan kandidat baru hanya jika isi prediksinya memang berbeda dari arsip.
