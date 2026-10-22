# Remove o Ditado: fecha o programa, apaga o atalho da Area de Trabalho e (com -Tudo) a pasta inteira.
param([switch]$Tudo)
$Pasta = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' OR Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*ditado.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
$Lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Ditado.lnk"
if (Test-Path $Lnk) { Remove-Item $Lnk -Force; Write-Host "Atalho removido." }
$Ini = Join-Path ([Environment]::GetFolderPath("Startup")) "Ditado.lnk"
if (Test-Path $Ini) { Remove-Item $Ini -Force; Write-Host "Removido da inicializacao." }
if ($Tudo) {
    Set-Location ..
    Remove-Item $Pasta -Recurse -Force
    Write-Host "Pasta $Pasta apagada (venv + modelos)."
} else {
    Write-Host "Programa fechado e atalho removido. Para apagar a pasta (venv + modelos ~3 GB): desinstalar.ps1 -Tudo"
}
