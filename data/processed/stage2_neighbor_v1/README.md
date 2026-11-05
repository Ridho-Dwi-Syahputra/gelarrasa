# Kelompok citra tahap 2

Seluruh 1.920 train dan 480 test dibagi ke delapan folder cluster_00 sampai cluster_07. Sumber assignment: head clustering berbasis tetangga yang dilatih pada 1.919 pixel-unik train. Test hanya diprediksi dengan model tetap. Berkas PNG mempertahankan ukuran native setelah preprocessing tahap 1.

Ini kandidat pengelompokan algoritmis; nama cluster belum menjadi label material terverifikasi. Nomor cluster tidak sama artinya dengan nomor stage1. Semua anggota ditetapkan algoritma, tanpa pemindahan per gambar. Template penamaan seluruh cluster tersedia untuk tim; mapping kategori aktif tidak diganti otomatis.

`cluster_manifest.csv` memuat setiap ID, folder, hash sumber dan PNG, serta skor head yang belum terkalibrasi. `folder_counts.csv` memuat jumlah per cluster. `provenance.json` mencatat sumber model dan verifikasi 2.400 salinan. Folder adalah salinan independen; sumber raw dan stage1 tetap dipertahankan.
