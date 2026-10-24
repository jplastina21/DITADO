# Instalador do Ditado (Windows)

Gera `Ditado-Setup.exe` (~86 MB): programa empacotado com PyInstaller + instalador Inno Setup.

**Gerar:** `powershell -ExecutionPolicy Bypass -File build.ps1` (precisa de Python 3.10 e Inno Setup 6).
Resultado em `Output\Ditado-Setup.exe`. Sem assinatura digital: o Windows mostra "editor desconhecido" (SmartScreen).

**O que vai dentro:** programa, Python, faster-whisper/ctranslate2 (CPU), `config.padrao.toml`.
**O que NAO vai:** o modelo Whisper (baixado na 1a execucao para `%LOCALAPPDATA%\Ditado\modelos`) e as
bibliotecas CUDA (~1,4 GB, baixadas do PyPI so se houver placa NVIDIA, para `%LOCALAPPDATA%\Ditado\cuda`).
Config e log: `%APPDATA%\Ditado\`.

**Dispositivo:** `auto` = NVIDIA com `large-v3-turbo` (float16) se der; senao CPU com `small` (int8).
Se nao tem as bibliotecas CUDA ainda, comeca na CPU e troca para a GPU sozinho quando terminar de baixar.

**Testes sem mexer em teclado/bandeja:** `Ditado-cli.exe --verificar` e `Ditado-cli.exe --arquivo x.wav --dispositivo cpu`.
`DITADO_DADOS=<pasta>` redireciona config, log, modelos e CUDA para outra pasta.

`ditado.py` aqui e uma variante do `ditado.py` da raiz (que e a do PC do Juliano): mesma logica de teclado/colagem,
caminhos e escolha de dispositivo diferentes.
