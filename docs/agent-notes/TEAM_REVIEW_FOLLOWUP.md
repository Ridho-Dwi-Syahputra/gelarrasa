# Tindak lanjut review tim, 7 November 2026

Ini catatan sebelum implementasi. Status terbaru:14acuan sudah dipakai dan42kategori berubah;1662/1885ditahan. Lihat [hasil implementasi](BOUNDED_REFINEMENT_RESULT.md). Proposal7di bawah bersifat historis, bukan ledger aktif.

Review pengguna dicatat pada table-csv/team_review_inventory.csv. Statusnya96preferensi arsip,9preferensi program,20preferensi program tentatif,20belum dinilai(lembar2),4kasus bersyarat dan1ambigu(1215). Preferensi daftar lembar3/4 ditandai sebagai interpretasi, bukan konfirmasi kelas baru yang ditebak asisten.

Tidak memindahkan label aktif atau menghapus raw dari review ini. Jika seluruh105usulan kategori tegas dipakai sebagai human targets, distribusinya plastic32,paper23,electronic17,glass13,cardboard12,textile4,metal3,foodwaste1: banyak kelas melewati batas5. Tidak menyamarkan seluruh review sebagai keluaran algoritma.

## Lampu dan aturan yang diuji

Pencarian embedding DINOv2 saja tidak cukup: tetangga0147/0594 berupa bentuk lain dengan cosine rendah. Kosakata pretrained CNN21k mengangkat kandidat test254/364/440, tetapi pencarian ini memiliki false positives dan bukan label kelas test. Belum ditemukan padanan bohlam bulat/spiral yang meyakinkan; test364 patut dilihat sebagai kandidat perangkat pencahayaan dengan bentuk berbeda.

Top1 sumber untuk lampu sering berupa `bulb.n.04`: definisinya hanya bentuk bulat/tetesan. Ini berbeda dari `light_bulb.n.01`, sebuah lampu listrik. Jangan memetakan semua bentuk bulb menjadi electronic.

Probe otomatis menambahkan akar spesifik electrical lighting dan memakai gate top1 spesifik atau bentuk umum yang didukung minimal2nama spesifik pada top5, disertai konsistensi flip horizontal. Seluruh1906foto utuh train dipindai; tidak memakai ID user sebagai mask, target manual, atau label test. Hasil beku:16usulan perubahan, pada0335,0594,0815,0865,0932,0939,0956,1070,1132,1406,1431,1497,1628,1641,1908,1918.0147belum lolos gate. Ini kandidat dan belum menjadi kebenaran semantik terverifikasi.

Bukti: json/lighting_specificity_plan.json, json/lighting_specificity_frozen.json, table-csv/lighting_specificity_selected.csv. Data aktif dan notebook tetap tidak berubah; tidak ada human target dipakai pada probe.

## Kasus bersyarat

| Train | Bukti pencarian tanpa label test | Penanganan sementara |
|---|---|---|
| 0676 | DINO mengarah ke328/265/303; CNN ke118/377. Adegan dan konteks belum meyakinkan sebagai padanan objek utama. | Tahan di review; ketiadaan tetangga kuat belum membuktikan data tidak berguna. |
| 1650 | Dua representasi menempatkan231 sebagai kandidat terdekat; DINO juga453. | Ada pola tumpukan serupa untuk ditinjau; tidak otomatis dibuang. |
| 1662 | CNN mengangkat085/128, selain473; terdapat media cetak/kemasan bergambar makanan untuk dibandingkan. | Pertahankan konteks cetakan saat review; jangan crop gambar makanan lalu menganggapnya sisa pangan fisik. |
| 1885 | Tetangga numerik teratas didominasi adegan makanan;085/128 tetap relevan sebagai contoh konteks cetak/kemasan. | Bedakan objek media dengan isi gambar. Tidak mengubah crop menjadi foodwaste secara otomatis. |

Kemiripan tidak memberi label benar test. Ketidakhadiran padanan dekat juga tidak menjadi aturan cleaning tunggal. Train yang labelnya sudah masuk akal masih dapat memberi variasi bahan/bentuk meskipun test tidak mempunyai foto sejenis yang dekat di embedding.

## Proposal hemat label manusia

Tujuh ID prioritas dari review eksplisit pengguna:0056plastic,0072metal,0085plastic,0141paper,0147electronic,0251glass,0256paper. Distribusi:paper2,plastic2,glass1,metal1,electronic1. Ini **proposal**, belum dipakai sebagai target atau diterapkan ke dataset. File: table-csv/minimal_human_seed_proposal.csv. Tujuannya memberi acuan terbatas bagi metode train-only berikutnya, bukan menyalin seluruh arsip atau melebihi batas per kelas.
