# -*- mode: python ; coding: utf-8 -*-
# Gera dist\Ditado\ com dois executaveis na mesma pasta:
#   Ditado.exe      (sem console; o programa de verdade, icone na bandeja)
#   Ditado-cli.exe  (com console; so para --arquivo x.wav, testes e diagnostico)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "av", "onnxruntime", "tokenizers", "sounddevice"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += ["pystray._win32", "pynput.keyboard._win32", "pynput.mouse._win32",
                  "win32clipboard", "tomli", "huggingface_hub", "tqdm.auto"]
datas += [("config.padrao.toml", "."), ("ditado.ico", ".")]

a = Analysis(["ditado.py"], pathex=[], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest", "torch"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Ditado", icon="ditado.ico",
          console=False, upx=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Ditado-cli", icon="ditado.ico",
          console=True, upx=False)
coll = COLLECT(exe, cli, a.binaries, a.datas, strip=False, upx=False, name="Ditado")
