# Audit visual pribadi arsip cardboard ? 6 November 2026

Sebanyak **272 gambar** pada folder cardboard dilihat satu per satu pada preview proporsional maksimal 320 px. **62 kandidat** dibuka ulang dari file asli; setelah tinjauan ulang, **49 tetap memerlukan keputusan manusia** dan **223 tidak menunjukkan masalah jelas pada pengamatan ini**. Tidak ada label benar resmi dan tidak ada klaim 223 telah dipastikan benar.

Sumber ulasan: pengamatan visual asisten/LLM atas arsip lama, atas permintaan pengguna. Ini bukan label cluster algoritmis, anotasi panitia, target pelatihan, atau pengesahan metode kompetisi. Keputusan manusia atas catatan ini tidak mengubah asal anotasi menjadi hasil clustering. Semua gambar, kategori folder, model dan prediksi aktif dipertahankan.

Kemampuan menahan cairan tidak menentukan bahan tunggal. Karton minuman dapat tersusun dari paperboard dan lapisan pelindung polimer; lihat [penjelasan material Tetra Pak](https://www.tetrapak.com/en-gb/solutions/packaging/packaging-material). Sumber ini menjelaskan bahan, bukan menetapkan label resmi lomba.

## Enam prioritas utama

| ID | Alasan perlu dicek |
|---|---|
| 0075 | Tekstur berpetak menyerupai anyaman/pelapis; tepi robek belum memperlihatkan struktur corrugated yang pasti. |
| 0850 | Bentuk dan lipatan alas menyerupai kantong kraft; perlu memastikan apakah badan fleksibel atau kotak board. |
| 1210 | Objek menyerupai mouse komputer dengan bagian seperti scroll wheel; cek apakah perangkat elektronik fungsional atau model kerajinan karton. Jangan pilih material casing saja. |
| 1322 | Objek menyerupai mouse komputer dengan komponen seperti scroll wheel, sama seperti 1210; cek fungsi elektronik versus model kerajinan. |
| 1499 | Tekstur seperti anyaman/pelapis dan bagian robek, sejenis 0075; perlu membedakan kain/pelapis dan karton. |
| 1557 | Foto asli menunjukkan kemasan rokok, bukan makanan. Lapisan foil/kertas/film dominan dan badan flip-top kaku tidak jelas; periksa soft-pack versus kotak board. |

## Daftar lengkap 49 kandidat

| Jenis tinjauan | Nomor gambar |
|---|---|
| Fungsi perangkat versus material casing | 1210, 1322 |
| Bahan objek belum jelas | 0075, 1499, 0239, 0250, 0875, 1131, 1806, 1878 |
| Batas kertas, paperboard, dan pembungkus | 0850, 1557, 0396, 1368, 1548, 1892 |
| Beberapa objek/material dalam satu foto | 0008, 0323, 0420, 0734, 0737, 0955, 0992, 1004, 1069, 1077, 1158, 1225, 1283, 1302, 1319 |
| Detail bahan belum cukup | 0134, 0214, 0298, 0545, 0574, 1043, 1402, 1650, 1742 |
| Kemasan karton berlapis/komposit | 0774, 1027, 1244, 1246, 1417, 1522, 1751 |
| Kemasan serat cetak atau molded pulp | 0189, 1637 |

## Dugaan awal yang tidak diteruskan

0010, 0061, 0260, 0431, 0553, 0597, 0920, 1050, 1176, 1212, 1837, 1910, 1920.

Contoh: 0010, 0597, dan 1176 memperlihatkan badan kemasan rokok berbentuk board setelah file aslinya dibuka; film luar tidak otomatis membuat seluruh objek menjadi plastic. 1212 didominasi kotak pizza: noda/residu tidak otomatis membuatnya foodwaste. 0553/1910 memperlihatkan struktur panel karton meskipun berbentuk bangku. 1920 menyerupai pelindung sudut dari karton laminasi, bukan harus corrugated. Ini tetap penilaian visual sementara.

## Berkas untuk ditinjau

- [Galeri seluruh 49 kandidat](review_gallery.html).
- [CSV 49 kandidat dan alasan](needs_human_review_49.csv).
- [Ulasan seluruh 272 gambar](all_272_private_reviews.csv).
- [Template keputusan pribadi](human_decision_template.csv): kolom keputusan kosong, tidak diterapkan otomatis.
- source_manifest.json menyimpan hash sumber; dua JSONL merekam putaran awal dan tinjauan file asli.

Tidak melakukan training, mengubah isi folder label, memperbaiki prediksi model, atau mengubah notebook kompetisi dalam audit ini.
