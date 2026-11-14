# Konteks aktif GELAR RASA -7 November2026

## Buka dan jalankan

- Notebook tunggal: final-submit/DSC2026_Notebook.ipynb.
- Data asli: data/Dataset DSC GelarRasa 2026/{train,test}. Train1920 tanpa label resmi, test480; delapan kategori dan Accuracy sebagai metrik utama.
- Data aktif: data/processed/{train,test,crops,review}. Tidak ada folder stage/hash pada lokasi ini.
- Gambar/galeri notebook: final-submit/reports. Output model/submission: final-submit/output (kosong sebelum training). Catatan audit, eksperimen dan cache: docs/agent-notes.
- Pedoman agen: docs/skills/AGENTS.md. Semua pedoman/proyek berada di docs/skills, kecuali README root.
- Riwayat sebelum cleanup: arsip/sebelum_rapi_20261107. Peta pemindahan dan bukti file utuh: migration_manifest.json di folder tersebut. Path pada laporan lama adalah path historis.

## Lingkup dan kebijakan input

Fokus sekarang EDA dan preprocessing; training/evaluasi classifier dan CSV submission akan dilanjutkan pada notebook yang sama. Aturan preprocessing: keep0313/0343/1817;8sumber tidak masuk himpunan kerja (0134,0298,0545,0963,1028,1043,1465,1890);0376 cadangan tinjauan. Crop0415 kalkulator;0437 laptop dan apel;0951 halaman dan kalkulator;1007 kantong merah;1351 kotak kiri bawah. Hasil1913sampel train dari1911induk dan480test,7nativecrop. Sumber asli tidak dihapus. Kategori crop tidak diwariskan dari parent.0343 disimpan untuk sampler train kelak; sampler berbobot belum aktif.

Penerapan webinar untuk EDA/preprocessing meliputi profil RGB/dtype/kualitas, duplikat/pHash, group split draft, preview augmentasi train-only, resolusi160/192/224, HOG/warna dan proyeksi fit train-only. Training SVM/CNN/hybrid, callbacks, OOF, Grad-CAM dan dampak fisik tetap tahap berikut; status rinci ada di docs/agent-notes/webinar_coverage.csv setelah Run All.

## Pembanding cluster historis

Baseline stage17 historis:1.740cocok/180berbeda dari1920 terhadap folder arsip; setelah8drop:1735cocok/177berbeda dari1912. Setelah kebijakan reserve dan penggantian parent oleh crop, denominator pembandingan foto utuh menjadi1906: baseline1732cocok/174berbeda. Jangan mencampurkan denominator atau membandingkan kategori crop dengan kategori foto induk.

Hasil historis sebelum acuan manusia: penyempurnaan otomatis lima keluarga fitur, lima source-groupfold, minimal4/5persetujuan dan iterasi berjangkar tetap. Berhenti pada putaran2 karena17perubahan<1%. Dari1906foto utuh,1756cocok/150berbeda terhadap arsip;42perbedaan lama menjadi cocok dan18perbedaan baru muncul. Saat itu jatah manual0/40. Ini audit setelah pembekuan, bukan Accuracy resmi. Daftar150 dan kartunya kini snapshot historis, bukan jumlah kesalahan aktif.

Hasil aktif terbaru (8 November 2026): **17 berbeda / 109 sesuai dari 126 review tegas; 34 sumber manusia tetap, maksimal lima per kelas.** Prefix `frozen_residual_continuation`. Urutan audit publikasi: 29 menjadi 28 melalui campuran DINO/CNN; tetap 28 melalui eksperimen selektif; menjadi 27 lewat acuan pseudo. Putaran feedback kedua sempat 28, kemudian seluruh batch ditarik karena konsentrasi bank textile melampaui 50%; kembali 27. Hipotesis baru bank source-diverse maksimal 20 pseudo/kelas menghasilkan 32 perubahan otomatis, dua pemulihan (0868/1503) dan satu regresi (0536), sehingga 26. Bukti status aktif di CURRENT_LABEL_STATUS.md dan json/current_review_status.json selalu mengungguli angka historis. Bukan seluruh 32 perubahan sudah terbukti benar.

Tidak ada tambahan acuan manusia. Dua sumber 1662/1885 tetap ditahan, 20 review tentatif dan dua bersyarat tetap terpisah. Keselarasan review bukan Accuracy panitia. Perubahan di luar review belum terverifikasi. Run All mematerialisasikan snapshot assignment dan menghitung crop; belum mengulang seluruh fit assignment heads atau membuat classifier/CSV final.

## Aturan dan sumber

Baca RULES_AND_RUBRIC.md. Arsip visual asisten dan train/label tidak dipakai sebagai target, filter, pemetaan aktif atau seleksi model kompetisi. Perbandingan yang diminta pengguna adalah audit terpisah atas prediksi yang dibekukan. Model/pipeline tidak boleh disebut patuh atau akurat100% hanya karena folder kategori terisi. Asal keputusan pengguna/asisten untuk crop tetap dinyatakan.

Data/gambar/model/output/arsip tidak di-push; kode, notebook dengan output dan dokumen/tabel kecil dapat di-push bila diminta. Semua pekerjaan di latar belakang tanpa mengambil fokus layar.

## Pembagian train terbaru

Notebook telah lulus Run All dari processed kosong setelah publikasi terakhir:13/13sel kode,0error dan16visual;4793hashgambar diverifikasi. Train dibagi langsung ke delapan folder kategori. Input operasional data/preprocessing_inputs.npz versi3 memuat snapshot kategori,34acuan manusia,group,hash sumber dan referensi fitur fit-train. Manifest acuan aktif data/processed/manual_seeds.csv dibuat notebook. Kategori nonmanual belum terverifikasi; acuan yang telah digunakan fit bukan evaluasi final independen.

1662dan1885ditahan sesuai jawaban pengguna:training_eligible=False, bukan acuan manusia atau referensi crop.1913train tetap tersimpan,1911eligible. Loader training wajib menyaring manifest. Raw tetap utuh. Assignment heads dilatih di tahap eksperimen; Run All ini tidak mengulang fit tersebut. Crop dinilai terpisah dengan DINOv2 dan voting tujuh tetangga. Classifier/submission final tetap tahap berikutnya.

Notebook tidak bergantung pada docs/agent-notes dan tidak menyebut folder docs atau layanan model bahasa. Catatan agent dikumpulkan oleh builder di luar notebook. Metadata preprocessing dan features.npz dibuat notebook di data/processed. Bukti aktual terbaru ada di agent-notes/execution.json setelah run selesai.

## Pembaruan group geometri

SIFT/RANSACpada8047pasangan train eligible menghasilkan116edgekuat,mencakup183foto. Komponen sudah konsisten kategori,sehingga tidak memberi relabel langsung. Untuk1904foto eligible,uniongroupklasifier1870menjadi1777; groupinginputraw1920memiliki denominator berbeda(1869menjadi1792). Semua116edge berada pada parentgroupdanpartisi yang sama. Tidak memakai test/1662/1885dalam fit geometri. Sourcegroupadalah pembagian sumber,sedangkan kategori tetap8.
