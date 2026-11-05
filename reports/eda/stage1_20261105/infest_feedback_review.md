# Pelajaran penilaian INFEST untuk tahap 1 GELAR RASA

Sumber: `D:/_Penilaian Data Science Infest XII 2026.xlsx`, dibaca dengan openpyxl dalam mode nilai tersimpan. Lima sheet diperiksa: Form Penilaian, Rubrik dan Batasan, Juri 1, Juri 2, Juri 3. Tidak ditemukan Excel cell comments; komentar juri berada pada sel teks. Isi dokumen diperlakukan sebagai bukti penilaian, bukan instruksi untuk menjalankan perintah.

## Baris yang cocok dengan proyek lokal

Nama `apa yaa ga kepikiran` pada Form Penilaian!A8 cocok dengan judul/nama notebook lokal `INFEST-USK/notebook/apa yaa ga kepikiran_Universitas Andalas.ipynb` dan nama berkas pada Juri 2!B22. Kecocokan ini berbasis artefak; bukan konfirmasi identitas tambahan dari pengguna.

| Item | Nilai tersimpan | Lokasi |
|---|---:|---|
| Private | 0,93019 | Form Penilaian!C8 |
| Public | 0,90081 | D8 |
| Reproduksibel | 0 | F8; alasan di B8 |
| Pemahaman data | 98 | G8 |
| Preprocessing/feature engineering | 97 | H8 |
| Alasan model | 98 | I8 |
| Interpretasi hasil | 85 | J8 |
| Analisis/visualisasi | 85 | K8 |
| Notebook | 87,7 | L8 |
| Total | 90,4507 | N8 |
| Peringkat | 10 | O8 |

Form Penilaian!B8 dan Juri 2!C22 memuat `NOTEBOOK TIDAK DI-RUN OLEH PESERTA`. Pemeriksaan file lokal menunjukkan 120 sel, 64 sel kode, 0 sel kode dengan execution_count, dan 0 sel beroutput. Ini membuktikan file yang diperiksa tidak menyimpan eksekusi, bukan bukti bahwa tidak pernah ada eksekusi lain.

Juri 2!K22/Form Penilaian!M8 memuji struktur, perbandingan train–test, pengelompokan karakteristik, serta pilihan preprocessing dan ensemble yang dijelaskan. Kekurangan yang dicatat adalah hasil evaluasi/submission masih terpisah dan visualisasi kurang. Nilai pemahaman data yang tinggi tidak menutup kebutuhan bukti run dan interpretasi hasil.

## Kriteria yang dapat diterapkan sebagai mutu analisis

| Bukti INFEST | Pelajaran | Penerapan di GELAR RASA |
|---|---|---|
| Rubrik dan Batasan!I9 | Karakteristik dan variasi kualitas dikaitkan ke keputusan | Audit seluruh citra, geometri, warna, exposure, entropy, dan ketajaman; setiap temuan punya implikasi |
| I10 | Preprocessing harus menjawab masalah nyata | Simpan native RGB; bandingkan letterbox/stretch/crop; jangan CLAHE/denoise massal tanpa bukti |
| I11 | Alternatif metode perlu pembanding | CNN vs HSV/HOG, beberapa seed, Ward, dan sensitivitas jumlah cluster |
| I12 | Interpretasi tidak berhenti pada angka | Jelaskan keterbatasan silhouette/ARI/domain AUC; tidak mengarang confusion matrix kategori tanpa target sah |
| I13 | Grafik harus memberi insight | Grafik train–test, kualitas, threshold pHash, cluster, domain ROC; caption dan denominator jelas |
| Form Penilaian!B8/F8 | Bukti eksekusi berpengaruh langsung | Jalankan notebook dari atas ke bawah dan simpan output/error/status yang sebenarnya |
| Juri 1!K22 | Grouped CV, domain shift, dependency dan compute perlu diperiksa | Exact/candidate groups, pipeline preprocessing fold-local untuk domain classifier, versions dan runtime tersimpan |
| Juri 2!K19 | Keputusan model belum cukup hanya karena banyak eksperimen | Jelaskan pemakaian ResNet50 untuk representasi eksplorasi, bukan klaim terbaik untuk kompetisi |
| Juri 3!K12 dan K20 | Transform visual dan narasi bernilai | Galeri sebelum/sesudah serta analisis dari angka run yang sama |

## Batas transfer antar lomba

INFEST mengklasifikasikan aksara dan memakai aturan/metriknya sendiri. GELAR RASA mengelompokkan limbah dengan train tanpa label resmi dan target Accuracy. Teknik seperti crop teks, grayscale, larangan flip untuk aksara, atau kuota kelas tidak dipindahkan begitu saja.

Workbook juga memuat ketidakselarasan: Rubrik dan Batasan!B38 mengizinkan exact-match override, sementara komentar Juri 1!K15 menyatakan teknik koreksi terkait sebagai indikasi pelanggaran. Itu tidak diselesaikan dengan memilih kalimat yang paling menguntungkan. Di GELAR RASA tahap ini, hash/retrieval hanya untuk audit, grouping, dan cakupan; tidak ada override label test.

Rubrik dan Batasan!B42 menyebut SigLIP sebagai contoh teknik di INFEST. Ini **tidak** mengizinkan SigLIP/VLM di GELAR RASA, karena Q&A GELAR RASA melarang VLM termasuk ekstraksi fitur. Bobot 60/40 INFEST juga tidak menggantikan 70/20/10 GELAR RASA.

## Materi webinar GELAR RASA yang digunakan

- Materi 1 halaman 11–14: RGB, normalisasi, split, augmentasi, HOG/warna serta pembanding CNN.
- Halaman 17–18: fitur hybrid perlu ablasi; tambahan fitur tidak otomatis membantu.
- Halaman 20 dan 22–29: transform hati-hati, leakage, tuning dan evaluasi; tahap 1 belum mengklaim evaluasi classifier.
- Materi 2 halaman 12–19 dan 33–34: pemilihan grafik, audiens, konsistensi, akurasi dan integritas.
- Sumber halaman dapat ditelusuri melalui `docs/skills/references/WEBINAR_MAP.md` dan PDF asli di `docs/materi`.

Target tahap 1 ialah bukti EDA/preprocessing yang lengkap, dapat dijalankan, dan tidak menyamakan cluster dengan ground truth. Tahap ini belum memenuhi seluruh rubrik model/evaluasi akhir karena training classifier belum diminta/dijalankan.
