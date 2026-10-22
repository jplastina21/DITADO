# Instala o Ditado do zero: venv, pacotes, modelo do Whisper e atalho na Area de Trabalho.
# Uso (PowerShell, na pasta do programa):
#   powershell -ExecutionPolicy Bypass -File install.ps1            # tudo
#   powershell -ExecutionPolicy Bypass -File install.ps1 -SemAtalho # sem atalho na Area de Trabalho
param([switch]$SemAtalho, [string]$Python = "")
$ErrorActionPreference = "Stop"
$Pasta = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Pasta
$env:HF_HOME = Join-Path $Pasta "modelos"      # modelos ficam aqui, nao no C:
$env:PIP_NO_CACHE_DIR = "1"                      # sem cache do pip no C:

if (-not $Python) {
    $Python = (& py -3.10 -c "import sys; print(sys.executable)" 2>$null)
    if (-not $Python) { $Python = "python" }
}
Write-Host "Python: $Python"

if (-not (Test-Path "$Pasta\venv\Scripts\python.exe")) {
    Write-Host "Criando venv..."
    & $Python -m venv "$Pasta\venv"
}
$Py = "$Pasta\venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip --quiet
Write-Host "Instalando pacotes (faster-whisper, CUDA, audio, bandeja)..."
& $Py -m pip install -r "$Pasta\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip falhou" }

Write-Host "Baixando/carregando o modelo (a primeira vez demora: ~1,6 GB)..."
& $Py -c "import sys; sys.argv=['x']; import ditado; c=ditado.carregar_config(); ditado.configurar_log(True); ditado.preparar_ambiente(c); m=ditado.Motor(c); m.carregar(); print('OK:', m.modelo_nome, m.dispositivo)"
if ($LASTEXITCODE -ne 0) { throw "modelo nao carregou (veja ditado.log)" }

& $Py "$Pasta\fazer_icone.py"

if (-not $SemAtalho) {
    $Desk = [Environment]::GetFolderPath("Desktop")
    $Ws = New-Object -ComObject WScript.Shell
    $L = $Ws.CreateShortcut((Join-Path $Desk "Ditado.lnk"))
    $L.TargetPath = "$Pasta\venv\Scripts\pythonw.exe"
    $L.Arguments = "`"$Pasta\ditado.py`""
    $L.WorkingDirectory = $Pasta
    $L.IconLocation = "$Pasta\ditado.ico"
    $L.Description = "Ditado por voz (segure Ctrl direito e fale)"
    $L.Save()
    Write-Host "Atalho criado em $Desk\Ditado.lnk"
}
Write-Host "Pronto. Abra pelo atalho 'Ditado' ou por DITADO.bat."
