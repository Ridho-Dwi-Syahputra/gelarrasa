> **Snapshot historis sebelum eksperimen material.** Status aktif:[MATERIAL_REFINEMENT_RESULT.md](MATERIAL_REFINEMENT_RESULT.md),45perbedaan dari126reviewtegas,34acuan tetap.

# Hasil revisi review tim — 7 November 2026

Status aktif terbaru: **26 kategori berubah dari versi sebelum revisi ini**:16perubahan pada tambahan acuan manusia dan10perubahan algoritmis (9konsensus classifier/graph dan1cluster). Total sejak snapshot150lama:68kategori berubah. Semua perubahan dapat dilacak pada [tabel prediksi beku](table-csv/review_final_selected.csv).

## Keputusan pengguna dan batas penerapan

Tiga lembar awal dalam jawaban sebelumnya dipahami sebagai59kasus perbedaan: pengguna mengonfirmasi pilihan reviewnya. Untuk20gambar yang sebelumnya belum ditinjau,0260/0267/0352/0396/0429/0512/0531/0536mengikuti kategori program;12lainnya mengikuti kategori arsip yang dikonfirmasi pengguna.1215ditetapkan **cardboard**, dicatat sebagai acuan manusia.1662/1885tetap ditahan, bukan acuan;20kasus tentatif dan0676/1650bersyarat belum diubah menjadi label manusia.

Total keputusan tegas untuk audit kini126gambar. Hanya34sumber menjadi target manusia aktif, bukan seluruh126. Enam kelas sudah memakai5acuan;metal3danfoodwaste1. Sisa kapasitas6hanya pada metal2danfoodwaste4. Jatah tidak direset oleh pergantian eksperimen, fold, augmentasi atau rollback. Asal review manusia tidak disamarkan sebagai keluaran algoritma.

## Eksperimen

Lima keluarga fitur visual non-VLM, source-group crossfit, tiga graph fit-train dan dua pembobotan prior mempertahankan9perubahan otomatis yang stabil. Clustering32/64/128pada tiga keluarga, minimal dua acuan sekelas tanpa konflik per cluster, lalu konfirmasi tiga graph dan classifier, menambahkan1perubahan:0562menjadi paper.1904foto train eligible masuk fit;1662/1885dan semua test dikecualikan.

Probe tambahan yang hanya memakai34acuan tanpa prior menghasilkan51usulan. Probe **ditolak secara keseluruhan**: diagnostik leave-source-group-out hanya2/34kategori pulih, dibanding14/34pada model dengan prior.51usulan tidak dipublikasikan dan tidak diseleksi satu-satu dari kecocokan review. Diagnostik acuan sengaja sulit dan kecil ini bukan Accuracy dataset, dan target prior memiliki riwayat fit sendiri.

## Audit setelah pembekuan

Pada kumpulan konfirmasi yang sama: sebelum revisi54/126cocok (72berbeda); sekarang78/126cocok (**48masih berbeda**). Tidak ada keputusan review yang sebelumnya cocok menjadi berbeda. Delapan nomor yang dinyatakan programnya benar tetap cocok. Ini kesesuaian terhadap review tim, bukan ground truth panitia. [48kasus tersisa](table-csv/remaining_current_review_differences.csv) dan [galeri aktif](remaining_visual_review/latest_review.html) tidak menjadi target/filter fit atau seleksi prediksi.

Saya belum menemukan metode yang membenarkan seluruh48kasus secara andal tanpa melampaui batas target manusia yang dipakai. Banyak perbedaan masih berhubungan dengan pembungkus plastik–kertas, glass–material lain, serta cardboard–paper. Menambahkan cluster saja tidak memberi informasi material baru. Metode representasi yang diadaptasi lebih lanjut masih mungkin diuji; hasil saat ini tidak membuktikan semua algoritma mustahil.

## Notebook dan data

Notebook tunggal [DSC2026_Notebook.ipynb](../../final-submit/DSC2026_Notebook.ipynb) lulus Run All dari processed kosong: 13sel kode,0error,16visual.1913sampel train tersimpan langsung pada8folder;1911eligible,480test,7crop asli.1215ada pada cardboard. Semua4793hash gambar sumber/turunan diperiksa. Data asli dan piksel test tidak berubah. Semua isi processed dibuat notebook.

Notebook mematerialisasikan snapshot penyempurnaan yang dibekukan dan menilai ulang crop; tidak mengulangi fit seluruh assignment heads. Classifier final, validasi independen dan CSV belum dibuat. Data operasional tetap diperlukan agar panitia dapat mereproduksi, dan paket final belum dinyatakan siap dikumpulkan. Notebook tidak membaca/menyebut folder administrasi atau layanan model bahasa. Catatan sumber, eksperimen dan perbandingan tetap di direktori ini.

Bukti utama: [validasi revisi](json/review_revision_delivery.json), [publikasi](json/bounded_anchor_publication.json), [Run All](execution.json), [validasi koleksi](skills_validation.json), [ledger manusia](json/manual_label_budget_status.json).
