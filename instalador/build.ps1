# Gera o instalador: venv de build -> PyInstaller (dist\Ditado) -> Inno Setup (Output\Ditado-Setup.exe).
# Uso (PowerShell, nesta pasta):  powershell -ExecutionPolicy Bypass -File build.ps1
# Precisa de: Python 3.10+, Inno Setup 6 (winget install JRSoftware.InnoSetup).
# O venv de build NAO leva nvidia-*: as bibliotecas CUDA sao baixadas pelo proprio programa, so em quem tem NVIDIA.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PIP_NO_CACHE_DIR = "1"
if (-not (Test-Path .\bvenv\Scripts\python.exe)) { py -3.10 -m venv bvenv }
$Py = ".\bvenv\Scripts\python.exe"
& $Py -m pip install --upgrade pip -q
& $Py -m pip install -r requirements-build.txt -q
if ($LASTEXITCODE -ne 0) { throw "pip falhou" }
if (-not (Test-Path ditado.ico)) { & $Py fazer_icone.py }
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
& $Py -m PyInstaller --noconfirm --clean ditado.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou" }
$Iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) { throw "Inno Setup 6 nao encontrado" }
& $Iscc Ditado.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup falhou" }
Get-Item Output\Ditado-Setup.exe | Select-Object FullName, Length
