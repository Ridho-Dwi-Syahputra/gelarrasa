"""Run the complete notebook with the project venv; save actual outputs per cell."""
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[2]
path=root/'notebook/01_EDA_Preprocessing_Clustering.ipynb'
expected=root/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
assert Path(sys.executable).resolve()==expected.resolve(), 'Use the project .venv interpreter.'
nb=nbformat.read(path,as_version=4)
report_prefix=nb.metadata.get('execution_report_prefix','notebook_v2')
execution_scope=nb.metadata.get('execution_scope','Full stage1+stage2 Run All; RECOMPUTE_DINO=True; checkpoints reused, features and fitting recomputed')
report=root/'reports/reproducibility'
report.mkdir(parents=True,exist_ok=True)
started=time.time()
start_wib=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat()
source_digest=hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest()
code_total=sum(c.cell_type=='code' for c in nb.cells)
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
nb.metadata.kernelspec=dict(name='gelar-rasa-venv',display_name='Python (GELAR-RASA .venv)',language='python')


def save_notebook():
    temporary=path.with_suffix('.ipynb.tmp')
    nbformat.write(nb,temporary)
    # Windows file watchers/antivirus can briefly lock the destination.
    # Retry the atomic replacement without restarting a successful kernel cell.
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.2 * (attempt + 1), 2.0))


def before(cell,cell_index,**kwargs):
    progress=dict(status='running',active_cell_index=cell_index,
                  completed_code_cells=sum(c.get('execution_count') is not None for c in nb.cells if c.cell_type=='code'),
                  total_code_cells=code_total,seconds=time.time()-started,started_wib=start_wib)
    (report/(report_prefix+'_progress.json')).write_text(json.dumps(progress,indent=2),encoding='utf-8')
    print(f'Start cell {cell_index}: {cell.source.splitlines()[0][:100]}',flush=True)


def after(cell,cell_index,execute_reply=None,**kwargs):
    save_notebook()
    print(f'Cell {cell_index} complete ({time.time()-started:.0f}s)',flush=True)


save_notebook()
client=NotebookClient(nb,timeout=7200,kernel_name='gelar-rasa-venv',resources={'metadata':{'path':str(root)}},
                      allow_errors=False,on_cell_start=before,on_cell_executed=after)
status='failed'
try:
    client.execute()
    status='passed'
finally:
    save_notebook()
    cells=[c for c in nb.cells if c.cell_type=='code']
    errors=[o.get('evalue') for c in cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
    result=dict(status=status,seconds=time.time()-started,code_cells=len(cells),
                executed_cells=sum(c.execution_count is not None for c in cells),errors=errors,
                execution=execution_scope,
                started_wib=start_wib,finished_wib=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
                kernel='gelar-rasa-venv',interpreter=sys.executable,
                notebook_code_sha256=source_digest,notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (report/(report_prefix+'_execution.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    (report/(report_prefix+'_progress.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
