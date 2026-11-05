"""Export actual saved notebook outputs as local HTML, without executing code."""
import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[2]
notebook = ROOT / 'notebook/01_EDA_Preprocessing_Clustering.ipynb'
nb = nbformat.read(notebook, as_version=4)
cells = [cell for cell in nb.cells if cell.cell_type == 'code']
assert [cell.execution_count for cell in cells] == list(range(1, len(cells)+1))
assert not [out for cell in cells for out in cell.outputs if out.output_type == 'error']
prefix=nb.metadata.get('execution_report_prefix','notebook_v2')
run = json.loads((ROOT / f'reports/reproducibility/{prefix}_execution.json').read_text())
assert run['status'] == 'passed'
exporter = HTMLExporter(template_name='lab')
html, resources = exporter.from_notebook_node(nb)
banner = ('<div style="padding:18px;background:#e8f5e9;border:1px solid #438c52;">'
          f'<strong>Output Run All tersimpan: {len(cells)}/{len(cells)} sel kode, tanpa error.</strong><br>'
          f"Selesai: {run['finished_wib']}. Pratinjau ini membaca output notebook; "
          'tidak menjalankan ulang training.</div>')
html = html.replace('<main>', '<main>' + banner, 1)
folder = ROOT / 'reports/reproducibility/notebook_preview'
folder.mkdir(parents=True, exist_ok=True)
target = folder / '01_EDA_Preprocessing_Clustering.html'
target.write_text(html, encoding='utf-8')
print('Saved:', target)
print('Saved code cells:', len(cells), '; embedded images:', html.count('data:image/'))
