"""Build and execute a compact review notebook for the frozen named model.

Reads model predictions, image metadata, and local images only. Private archive
comparison is deliberately outside this notebook and its measurements.
"""
from pathlib import Path
import asyncio,hashlib,json,sys,time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'notebook/02_Material_Resolution_Review.ipynb'

def main():
    if sys.platform=='win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    cells=[nbformat.v4.new_markdown_cell('''# Tinjauan model kategori material

Notebook pendamping EDA utama. Semua nama kelas di sini merupakan **prediksi algoritmis**, belum label benar terverifikasi. Delapan folder memastikan setiap citra mempunyai keluaran; hal itu tidak membuktikan kemurnian kelas atau Accuracy sempurna.

Notebook ini meninjau hasil checkpoint yang sudah dilatih dan dibekukan, bukan mengulang training. Artefak lokal, dataset resmi, serta lingkungan pada requirements diperlukan. Tidak ada panggilan LLM/VLM atau layanan prediksi eksternal. Model awal non-VLM, adaptasi lokal, dan sumber bobot dijelaskan pada laporan eksperimen. Kontrak label aktif belum dipromosikan.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display

candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next(p for p in candidates if (p / 'configs/data_contract.json').exists())
common = ROOT / 'reports/errors/stage6_material_resolution_20261106'
selection = json.loads((common / 'selected_material_run.json').read_text())
source = ROOT / selection['output']
report = ROOT / selection['report']
frozen = json.loads((report / 'selection_frozen.json').read_text())
assignment = source / 'named_assignments.csv'
assert hashlib.sha256(assignment.read_bytes()).hexdigest() == frozen['assignment_sha256']
predictions = pd.read_csv(assignment, dtype={'image_id': str})
classes = ['glass', 'paper', 'electronic', 'metal', 'plastic', 'cardboard', 'textile', 'foodwaste']
assert len(predictions) == 2400 and predictions.key.is_unique
assert predictions.predicted_class.isin(classes).all()
assert not predictions.ground_truth_verified.any()
print('Model:', selection['output'])
print('Train:', (predictions.split == 'train').sum(), '| Test:', (predictions.split == 'test').sum())
display(pd.Series(json.loads((report / 'summary.json').read_text())['selected'], name='pemilihan model tanpa label resmi'))'''),
    nbformat.v4.new_markdown_cell('''## Distribusi prediksi dan tanda tinjau

Jumlah per kelas adalah keluaran model, bukan kuota yang dipaksakan. Probabilitas softmax belum dikalibrasi. Tanda `needs_review` mencakup keyakinan rendah, margin kecil, atau konflik dengan salah satu sumber CNN; konflik juga dapat berasal dari kekeliruan sumber tersebut.'''),
    nbformat.v4.new_code_cell('''counts = pd.crosstab(predictions.predicted_class, predictions.split).reindex(classes, fill_value=0)
display(counts)
axes = counts.plot.bar(figsize=(12, 4), rot=25, title='Distribusi prediksi kategori material')
axes.set_ylabel('Jumlah gambar'); axes.set_xlabel('Nama kategori'); plt.tight_layout(); plt.show()
predictions['keyakinan_atau_margin_rendah'] = predictions.uncalibrated_probability.lt(.65) | predictions.margin.lt(.25)
predictions['konflik_sumber_saja'] = predictions.cnn_prior_disagrees & ~predictions.keyakinan_atau_margin_rendah
review = predictions.groupby('split').agg(gambar=('key', 'size'), perlu_tinjau=('needs_review', 'sum'), keyakinan_atau_margin_rendah=('keyakinan_atau_margin_rendah', 'sum'), konflik_sumber_saja=('konflik_sumber_saja', 'sum'))
display(review)
print('Tanda tinjau bukan ukuran kelompok campuran atau jumlah salah. Konflik sumber saja dapat berasal dari CNN pembanding yang keliru.')
print('Accuracy semantik: belum dapat diukur tanpa label evaluasi independen.')'''),
    nbformat.v4.new_markdown_cell('''## Preview seluruh kategori: train dan test

Contoh dipilih secara deterministik dari prediksi bermargin tinggi pada setiap kategori. Pemilihan contoh ini membantu membaca keluaran, bukan mengonfirmasi semua anggota kelas benar.'''),
    nbformat.v4.new_code_cell('''fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for c, name in enumerate(classes):
    for offset, split in enumerate(['train', 'test']):
        ax = axes.flat[2*c+offset]
        rows = predictions[(predictions.split == split) & (predictions.predicted_class == name)].sort_values(['margin', 'key'], ascending=[False, True])
        if len(rows):
            row = rows.iloc[0]
            with Image.open(ROOT / row.processed_path) as image:
                image = image.convert('RGB'); image.thumbnail((260, 260)); ax.imshow(image)
            ax.set_title(f'{name} | {split} {row.image_id}\\np={row.uncalibrated_probability:.3f}')
        else: ax.set_title(f'{name} | {split}: tidak ada prediksi')
        ax.axis('off')
plt.tight_layout(); plt.show()'''),
    nbformat.v4.new_markdown_cell('''## Contoh yang masih memerlukan tinjau

Prioritaskan test yang ditandai dan margin terkecil. Gambar tetap dipertahankan; tidak dihapus atau dipindah secara manual. Objek yang berdekatan, bahan transparan dan tekstur kemasan dapat menyebabkan konflik. Kandidat crop dan kemiripan piksel lebih rinci tersedia pada notebook EDA utama.'''),
    nbformat.v4.new_code_cell('''queue = predictions[predictions.needs_review].sort_values(['split', 'margin', 'key'], ascending=[False, True, True]).head(12)
display(queue[['key', 'predicted_class', 'uncalibrated_probability', 'margin', 'cnn_prior_disagrees']])
fig, axes = plt.subplots(3, 4, figsize=(14, 11))
for ax in axes.flat: ax.axis('off')
for ax, row in zip(axes.flat, queue.itertuples()):
    with Image.open(ROOT / row.processed_path) as image:
        image = image.convert('RGB'); image.thumbnail((260, 260)); ax.imshow(image)
    ax.set_title(f'{row.predicted_class} | {row.key}\\nmargin={row.margin:.3f}')
plt.tight_layout(); plt.show()
print('Data asli dipertahankan. Prediksi dan tanda tinjau tersimpan pada manifest terpisah.')''')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'gelar-rasa-venv','display_name':'Python (GELAR-RASA .venv)','language':'python'},'language_info':{'name':'python','version':'3.10.11'}})
    started=time.time();NotebookClient(nb,timeout=180,kernel_name='gelar-rasa-venv',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,TARGET)
    codes=[c for c in nb.cells if c.cell_type=='code'];errors=[o for c in codes for o in c.outputs if o.output_type=='error'];assert not errors and all(c.execution_count for c in codes)
    images=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs)
    assert images>=3
    proof=dict(status='passed',notebook=str(TARGET.relative_to(ROOT)),sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),code_cells=len(codes),executed=len(codes),errors=len(errors),embedded_figures=images,seconds=time.time()-started,bytes=TARGET.stat().st_size,background_execution=True,fresh_model_training=False)
    out=ROOT/'reports/reproducibility/notebook_material_resolution_execution.json';out.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
