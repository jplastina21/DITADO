@echo off
rem Abre o Ditado sem janela de console. O icone aparece na bandeja (perto do relogio).
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0ditado.py"
