"""Execute the whole notebook and save outputs after each completed cell."""
import json
import os
import sys
import time
from pathlib import Path

dependency_path = Path(os.environ.get('TEMP','')) / 'gelar_stage1_deps'
if dependency_path.exists():
    sys.path.insert(0,str(dependency_path))
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[2]
path=root/'notebook/01_EDA_Preprocessing_Clustering.ipynb'
nb=nbformat.read(path,as_version=4)
report=root/'reports/eda/stage1_20261105'
report.mkdir(parents=True,exist_ok=True)
started=time.time()
def after(cell,cell_index,execute_reply=None,**kwargs):
    nbformat.write(nb,path)
    print(f'Cell {cell_index} complete ({time.time()-started:.0f}s)',flush=True)
client=NotebookClient(nb,timeout=7200,kernel_name='python3',resources={'metadata':{'path':str(root)}},
                      allow_errors=False,on_cell_executed=after)
status='failed'
try:
    client.execute()
    status='passed'
finally:
    nbformat.write(nb,path)
    cells=[c for c in nb.cells if c.cell_type=='code']
    errors=[o.get('evalue') for c in cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
    result=dict(status=status,seconds=time.time()-started,code_cells=len(cells),
                executed_cells=sum(c.execution_count is not None for c in cells),errors=errors,
                execution='Run All with computation from source; no cache skip')
    (report/'notebook_execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
