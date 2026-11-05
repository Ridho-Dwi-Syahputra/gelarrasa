# Menjalankan EDA dan clustering GELAR-RASA

Notebook utama: `notebook/01_EDA_Preprocessing_Clustering.ipynb`.
Kernel: **Python (GELAR-RASA .venv)**. Lingkungan lokal berada di `.venv/` pada root proyek.
Notebook mengerjakan tahap 1 dan tahap 2 dalam urutan sel yang sama, termasuk ekstraksi fitur, training head clustering, evaluasi tanpa label, dan pembentukan folder train/test.

## Membuka hasil dan menjalankan ulang

Di VS Code, buka folder `GELAR-RASA`, buka notebook, lalu pilih kernel **Python (GELAR-RASA .venv)** atau interpreter `.venv/Scripts/python.exe`. Output dari eksekusi terakhir tersimpan di notebook; tidak perlu Run All untuk sekadar membaca grafik dan tabel.

Untuk JupyterLab, jalankan dari PowerShell pada root proyek:

```powershell
.\start_notebook.ps1
```

Server lokal memakai alamat loopback dan autentikasi bawaan Jupyter. Jangan menjalankan Run All dari dua kernel sekaligus karena keduanya menulis artefak dengan versi yang sama. Tutup server dengan Ctrl+C pada terminal yang meluncurkannya.

Untuk menjalankan semua sel dan menyimpan output otomatis dari terminal:

```powershell
.\.venv\Scripts\python.exe eksperimen/skrip/execute_project_notebook.py
```

Status per sel: `reports/reproducibility/notebook_v2_progress.json`.
Bukti eksekusi lengkap: `reports/reproducibility/notebook_v2_execution.json`.
Pemeriksaan dari dalam kernel: `notebook_v2_in_kernel_checks.json` pada folder yang sama.

`RUN_STAGE2=True` dan `RECOMPUTE_DINO=True` adalah default. Backbone checkpoint dapat memakai cache unduhan, tetapi fitur dan fitting dihitung ulang. Proses CPU dapat memerlukan puluhan menit. Set `RUN_STAGE2=False` hanya jika sengaja ingin menjalankan baseline tahap 1; jangan menyebut run tersebut sebagai seluruh pipeline.

## Menyiapkan ulang setelah clone

Data, gambar dan artefak disimpan dengan Git biasa. Kuota Git LFS akun tidak tersedia ketika push, sehingga PDF webinar 897 MiB disimpan sebagai 20 potongan yang dapat dipulihkan tanpa perubahan byte. Skrip pemulihan memverifikasi SHA-256 setiap potongan dan dokumen lengkap. File PDF asli di komputer kerja tetap utuh.

```powershell
git clone https://github.com/Ridho-Dwi-Syahputra/gelarrasa.git
Set-Location gelarrasa
py -3.10 eksperimen/skrip/restore_large_files.py
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name gelar-rasa-venv --display-name "Python (GELAR-RASA .venv)"
.\start_notebook.ps1
```

`requirements.txt` menetapkan dependensi utama. `requirements-lock.txt` mencatat versi yang benar-benar terpasang saat Run All. Paket CPU PyTorch diambil dari indeks wheel CPU resmi. Data kompetisi disimpan dalam repo privat tim sesuai permintaan pengguna; tidak ada unduhan dataset eksternal dalam notebook.

## Keluaran

- EDA baseline: `reports/eda/stage1_20261105/`.
- Perbandingan representasi, kurva head dan kestabilan view: `reports/errors/stage2_20261105/`.
- Kelompok gambar: `data/processed/stage2_neighbor_v1/train/cluster_00..07` dan `test/cluster_00..07`.
- Manifest berisi ID, asal dan hash: `data/processed/stage2_neighbor_v1/cluster_manifest.csv`.
- Checkpoint head dan assignment: `output/groups/stage2_neighbor_v1/`.

Nomor cluster belum merupakan kategori material terverifikasi. Accuracy kategori tetap belum diukur. Arsip perbandingan terpisah tidak menjadi input notebook. Snapshot notebook dan artefak sebelum perluasan disimpan di `reports/reproducibility/backups/20261105_before_notebook_v2/`.

Notebook mengeksekusi modul tahap 2 dari `eksperimen/skrip/`, sehingga folder proyek harus ikut tersedia. Jangan membagikan hanya notebook lalu menganggap modul dan inputnya otomatis tersedia.
