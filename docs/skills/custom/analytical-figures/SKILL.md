---
name: analytical-figures
description: Merancang grafik, tabel, dan diagram analisis citra untuk laporan GELAR-RASA dari data terverifikasi, dengan denominator, sumber, caption, dan keterbacaan PDF yang jelas.
---

# Visual analisis untuk laporan

Mulai dari pertanyaan analitis dan [rubrik laporan](../../references/REPORT_BLUEPRINT.md). Pilih grafik yang mengubah keputusan: distribusi/kualitas, struktur cluster, perbandingan model, kurva belajar, atau error. Bedakan hitungan citra, kelompok sumber, dan kelas hasil metode; tidak ada ground truth yang muncul karena nama grafik.

Gunakan Matplotlib/Seaborn/Plotly sesuai kebutuhan. Grafik publikasi dibuat dari tabel nyata, bukan imagegen. Diagram alur memakai bentuk/vektor yang mencerminkan implementasi. Simpan data grafik, script, transform/aggregasi, run_id, versi label, dan keluaran PDF/SVG/PNG. Jangan mengarang hasil untuk memenuhi desain.

Bar biasanya mulai nol; bila sumbu metrik dipotong pada line/dot chart, tandai dan pertahankan konteks. Tampilkan n serta denominator persentase; interval harus menyebut metode/unit sampling. Kelompok duplikat bukan pengamatan independen. Gunakan urutan kategori dan skala warna konsisten pada panel pembanding. Nilai hilang berbeda dari nol.

Gunakan warna ditambah label/marker, font terbaca saat dicetak, caption yang menjelaskan sumber dan makna tanpa perlu membaca paragraf lain. Tabel confusion matrix harus menjelaskan target evaluasinya dan normalisasi baris/kolom. Hindari 3D dekoratif atau smoothing yang menyembunyikan variasi.

Pelajari [scientific-visualization](../../vendor/K-Dense-AI__scientific-agent-skills/scientific-visualization/SKILL.md) seperlunya; preset Nature/Science bukan aturan kompetisi. Uji versi library lokal sebelum menyalin contoh. Periksa file export pada ukuran akhir, clipping, legenda, transparansi dan font. Hasil: visual + tabel asal + caption + catatan pemeriksaan.
