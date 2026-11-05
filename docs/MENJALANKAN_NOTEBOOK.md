# Menjalankan notebook GELAR-RASA

Notebook utama: `notebook/01_EDA_Preprocessing_Clustering.ipynb`.
Versi saat ini: **EDA visual v3**, sembilan sel kode. Kernel: **Python (GELAR-RASA .venv)**.

Notebook memuat preview, distribusi nama kelompok sementara, indikator kualitas, proposal crop seluruh gambar, probe view crop, dan pencarian piksel seluruh test?train. Head terlatih baseline dipakai ulang dengan verifikasi hash; `REBUILD_BASELINE=False` adalah default. Jika artefak belum tersedia, resep baseline akan dijalankan dari data resmi lokal. `REBUILD_BASELINE=True` juga membangun ulang baseline dan dapat memerlukan puluhan menit pada CPU. Pemetaan tampilan perlu ditinjau kembali bila partisi model berubah.

## Membaca hasil

Output grafik dan tabel tersimpan di notebook. Pratinjau HTML lokal tersedia di `reports/reproducibility/notebook_preview/01_EDA_Preprocessing_Clustering.html`. Ekspor ini membaca output aktual dan tidak menjalankan training.

Galeri seluruh 480 test: `reports/eda/stage3_visual_20261106/galleries/index.html`. Galeri membutuhkan gambar lokal; gambar terpisah tidak diunggah ke Git. Jika tab editor masih menampilkan versi lama, buka ulang file dari disk pada waktu yang nyaman bagi pengguna.

Nama material adalah interpretasi tampilan sementara oleh asisten sesuai permintaan pengguna, dengan sumber eksplisit pada `configs/display_label_mapping.json`. Kelompok campuran tidak dipaksakan menjadi satu kelas. Nama tersebut bukan label resmi atau target training/validasi/submission kompetisi. Crop tetap kandidat tanpa label; tidak ada perubahan assignment atau checkpoint baseline.

## Menjalankan di latar belakang

Dari root proyek pada terminal:

```powershell
.\.venv\Scripts\python.exe eksperimen/skrip/execute_project_notebook.py
```

Runner memakai kernel `.venv`, menyimpan output per sel dan berhenti pada error. Jangan menjalankan dua proses Run All bersamaan. Agent tidak boleh membuka, memaksimalkan, atau mengambil fokus VS Code/browser.

Bukti terkini:

- `reports/reproducibility/notebook_v3_progress.json` ? kemajuan per sel.
- `reports/reproducibility/notebook_v3_execution.json` ? status, durasi, kernel dan hash notebook.
- `reports/reproducibility/notebook_v3_delivery_validation.json` ? hash sumber/crop, koordinat piksel, fold, 480 test dan 990 referensi galeri.
- `reports/eda/stage3_visual_20261106/summary.json` ? cakupan metode dan pembatasannya.

Pemeriksaan sesudah Run All dan pembaruan pratinjau:

```powershell
.\.venv\Scripts\python.exe eksperimen/skrip/verify_notebook_delivery.py
.\.venv\Scripts\python.exe eksperimen/skrip/preview_executed_notebook.py
```

`verify_notebook_delivery.py` memilih validator sesuai versi notebook. Untuk membangun ulang dokumen v3 dari sumber builder, gunakan `build_visual_notebook.py`; ini menghapus output di file hasil builder sehingga harus diikuti Run All. `build_stage1_notebook.py` adalah builder v2 historis.

## Setelah clone

Repo berisi kode, notebook beroutput, konfigurasi dan dokumen kecil. Dataset, gambar terpisah, crop, embedding/checkpoint, backup dan file besar tetap lokal. Preview tertanam di notebook dipertahankan.

Siapkan data resmi pada `data/Dataset DSC GelarRasa 2026/train/`, `test/`, serta `data/SampleSubmission_DSC.csv`.

```powershell
git clone https://github.com/Ridho-Dwi-Syahputra/gelarrasa.git
Set-Location gelarrasa
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name gelar-rasa-venv --display-name "Python (GELAR-RASA .venv)"
```

Di VS Code, pengguna dapat memilih kernel tersebut pada notebook. `start_notebook.ps1` menyediakan opsi JupyterLab jika pengguna sendiri menghendakinya; agent tidak meluncurkan jendela.

## Keluaran lokal

- EDA/preview/tabel: `reports/eda/stage3_visual_20261106/`.
- Crop train: `data/processed/stage3_crop_candidates_v1/train_candidates/<ID>/`.
- Crop test untuk preview: `data/processed/stage3_crop_candidates_v1/test_previews/<ID>/`.
- Koordinat/hash/parent/fold: `data/processed/stage3_crop_candidates_v1/crop_manifest.csv`.
- Native RGB, baseline checkpoint, embedding dan hasil lama tetap tersedia pada path versi sebelumnya.

Versi v2 memiliki 35 sel terjalankan dan bukti historis `notebook_v2_*`. Snapshot sebelum perluasan v3 berada pada `reports/reproducibility/backups/20261106_before_visual_v3/`. Bukti v2 tidak digunakan untuk mengklaim hash notebook v3.
