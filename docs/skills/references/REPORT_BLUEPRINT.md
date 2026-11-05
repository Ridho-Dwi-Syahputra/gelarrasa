# Rancangan laporan berbasis rubrik

Ini panduan penulisan, bukan laporan hasil eksperimen. Ikuti [aturan aktif](RULES_AND_RUBRIC.md), [template panitia](../../rules/sources/official_report_template_20261105.docx), dan [ketentuan teks](../../rules/sources/official_report_template_20261105.txt). Tidak ada angka akurasi/hasil baru yang diciptakan untuk mengisi bagian kosong.

## Struktur dan alokasi isi

| Bagian wajib | Isi yang menjawab rubrik | Alokasi awal, bukan aturan panitia |
|---|---|---:|
| BAB I Pendahuluan | Latar belakang limbah; masalah klasifikasi; tujuan terukur; manfaat dengan batas klaim | 1 halaman |
| BAB II Desain Sistem | Deskripsi 1.920/480 data; asal label yang sah; alur sistem; preprocessing; EDA; metode/arsitektur dan alasannya | 4 halaman |
| BAB III Hasil dan Analisis | Hasil aktual; validasi dan batas target turunan; error/ablasi; insight; rencana implementasi lanjutan | 4 halaman |
| BAB IV Penutup | Jawaban rumusan masalah; keterbatasan; saran yang mengikuti bukti | 1 halaman |

Di luar batas isi: cover, daftar isi, daftar gambar, daftar tabel, pustaka, lampiran. Jangan memindahkan seluruh analisis utama ke lampiran agar narasi inti kosong. Judul/flowchart harus menunjukkan alur yang benar-benar dilakukan. Template memiliki penomoran BAB III 3.1, 3.2, 3.4, 3.5; pertahankan judul wajib dan catat koreksi penomoran di salinan kerja bila dirapikan, tanpa menghilangkan bagian.

## Bukti untuk setiap komponen nilai

| Rubrik | Poin | Bukti minimum yang perlu disiapkan |
|---|---:|---|
| Struktur/kerapian | 5 | Urutan template; konsistensi istilah/nomor; PDF terbaca; sitasi APA 7 |
| Analisis dan insight | 3 | Grafik/tabel EDA dengan denominator; temuan spesifik yang mengubah keputusan |
| Alur sistem | 3 | Diagram raw → fitur/clustering → mapping kategori oleh tim → training → validasi/inferensi, sesuai implementasi nyata |
| Metode dan alasan | 3 | Kesesuaian metode dengan data, pembanding dan keterbatasan; provenance pretrained non-VLM |
| Hasil/performa | 3 | Nilai dari log; split/group/label version; jumlah salah dan kelas; pisahkan proxy score, public dan private |
| Pengembangan/implementasi | 3 | Pengguna sasaran, input-output, alur penanganan ketidakpastian, biaya/latensi terukur atau skenario berasumsi |
| Interpretasi notebook | 5 | Setiap hasil penting dijelaskan: apa yang tampak, maknanya, keputusan berikutnya |
| Kerapian kode notebook | 5 | Urutan eksekusi, konfigurasi jelas, fungsi reusable, path portable, input/checkpoint/output terlacak |

## Daftar klaim dan bukti

Gunakan tabel sederhana `claim_id, section, claim, source_path, run_id, locator, status`. Status: observed, literature, assumption, planned, unavailable. Contoh tanpa angka buatan: klaim performa merujuk file metrik dan run nyata; klaim aturan merujuk halaman handbook/URL jawaban; ide sistem lanjutan diberi status planned. Tidak perlu menyalin sistem administrasi jurnal klinis dari vendor.

Untuk hasil utama tampilkan ID artefak yang sama di notebook, tabel laporan, dan manifest submission. Label cluster otomatis tidak berubah menjadi ground truth hanya karena classifier cocok dengannya. Bedakan evaluasi kestabilan clustering, kesepakatan terhadap target turunan, dan Accuracy Kaggle terhadap label tersembunyi.

## Grafik yang bernilai

Pilih 4–6 tampilan yang menyampaikan bukti utama: distribusi/quality data, alur sistem, stabilitas atau struktur cluster, pembanding metode pada evaluasi sama, galeri/error yang boleh dianalisis tanpa LLM mengklasifikasikan citra, serta trade-off performa/biaya. Angka cluster jangan dilabel sebagai distribusi ground truth. Confusion matrix memerlukan target evaluasi yang asalnya dinyatakan; tanpa target yang sah jangan membuatnya dari anotasi lama.

Plot dihasilkan dari tabel nyata dengan Matplotlib/Seaborn/alat setara; diagram sistem dari struktur implementasi. Jangan memakai gambar generatif untuk mengarang chart/hasil atau memalsukan contoh prediksi. Periksa ukuran cetak, legenda, satuan, n, warna konsisten per kategori, dan caption mandiri.

## Paket akhir

Pemeriksaan PDF: A4, Times New Roman 12, spasi 1,15, justify, maksimum 10 halaman inti yang batasnya ditandai; font/tabel/gambar tidak terpotong; pustaka cocok dengan sitasi; nomor halaman dan daftar isi diperbarui. Periksa seluruh halaman hasil render, bukan teks ekstraksi saja. Margin dan detail yang tidak ditulis jangan dikarang sebagai aturan; pertahankan template yang diunduh.

Notebook: sertakan interpretasi singkat dan asal setiap artefak. Nyatakan apakah benar-benar Run All atau hanya inspeksi struktur. CSV mengikuti sample dan nama resmi, serta benar-benar berasal dari checkpoint/prosedur yang dicatat. Keberadaan skill tidak membuktikan notebook atau laporan siap dikumpulkan.
