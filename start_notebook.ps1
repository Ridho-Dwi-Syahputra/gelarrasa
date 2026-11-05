# Launch the project notebook with the isolated environment and named kernel.
$ErrorActionPreference = 'Stop'
$gelarRoot = $PSScriptRoot
$gelarPython = Join-Path $gelarRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gelarPython)) {
    throw 'Create .venv and install requirements-lock.txt first; see docs/MENJALANKAN_NOTEBOOK.md.'
}
Set-Location -LiteralPath $gelarRoot
$env:JUPYTER_RUNTIME_DIR = Join-Path $gelarRoot '.jupyter_runtime'
& $gelarPython -m jupyterlab 'notebook/01_EDA_Preprocessing_Clustering.ipynb' '--ServerApp.ip=127.0.0.1' '--ServerApp.root_dir=.'
