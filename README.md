# ditado

Ditado por voz **local** para Windows: segure uma tecla, fale, solte — o texto cai onde o cursor
estiver. A transcrição roda na placa de vídeo (Whisper `large-v3-turbo` via faster-whisper);
nenhum áudio sai do computador e não há mensalidade. Feito como substituto do Wispr Flow, só
para ditar.

## Como usar

1. Abra pelo atalho **Ditado** na Área de Trabalho (ou `DITADO.bat`). Aparece um ícone redondo
   com microfone na bandeja, perto do relógio. Cinza = carregando (uns 5 s), **verde = pronto**.
2. Clique onde quer o texto, **segure o Ctrl direito**, fale, solte. Bipe agudo = gravando,
   bipe grave = parou. O ícone fica laranja enquanto transcreve e o texto é colado.
3. Ctrl direito com outra tecla (Ctrl+C, Ctrl+Win...) não dita: o programa entende que era atalho.
   Toque rápido (menos de 0,4 s) também é ignorado.
4. Para fechar: clique com o botão direito no ícone > **Sair**. **Abrir log** mostra o que aconteceu.

O que estava no seu Ctrl+C antes de ditar volta sozinho depois da colagem.

## Instalar do zero

Precisa de Windows, Python 3.10+, placa NVIDIA com driver recente. Na pasta do programa
(ex. `D:\Claudinho\ditado`), num PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

O script cria o `venv`, instala os pacotes (inclusive `nvidia-cublas-cu12` e `nvidia-cudnn-cu12`,
que o faster-whisper precisa no Windows), baixa o modelo (~1,6 GB) para `modelos\` dentro da pasta
(não enche o C:) e cria o atalho na Área de Trabalho. `-SemAtalho` pula o atalho.

Teste sem microfone: `venv\Scripts\python ditado.py --arquivo audio.wav` transcreve o arquivo,
mostra o tempo e a VRAM. Teste do clipboard: `venv\Scripts\python tests\test_colagem.py`.

## Configurar (`config.toml`)

Depois de mudar, feche (Sair) e abra de novo.

- **Trocar a tecla:** `[atalho] tecla = "ctrl_r"`. Exemplos: `"alt_r"`, `"f13"`, `"pause"`,
  `"scroll_lock"`. Evite Ctrl+Win / Fn enquanto o Wispr estiver instalado.
- **Trocar o modelo:** `[transcricao] modelo = "large-v3-turbo"` (`"large-v3"` é mais lento e um
  pouco mais preciso; `"medium"`/`"small"` mais leves). `modelo_fallback` entra se o principal falhar.
  `dispositivo = "cpu"` força CPU. Se a placa falhar, ele cai sozinho para CPU e escreve
  **ATENCAO** no log (e o ícone avisa "CPU!").
- **Vocabulário:** `vocabulario = [...]` — palavras que o Whisper deve conhecer.
- **Erro que se repete:** `[substituicoes]` troca uma frase errada pela certa.
- **Bipe:** `bipe = false` desliga; `volume_bipe` ajusta.
- **Espaço no fim do texto:** `[colagem] espaco_no_fim`.
- **Microfone:** `dispositivo = ""` usa o padrão do Windows. Com `microfone_sempre_aberto = true`
  o microfone fica aberto (o Windows mostra o ícone de microfone em uso) para não comer a primeira
  sílaba; `false` abre só quando você aperta.

## Abrir junto com o Windows

Ainda não está ligado de propósito. Quando quiser: `Win+R`, `shell:startup`, e copie o atalho
**Ditado** da Área de Trabalho para essa pasta.

## Desinstalar

```powershell
powershell -ExecutionPolicy Bypass -File desinstalar.ps1        # fecha e tira o atalho
powershell -ExecutionPolicy Bypass -File desinstalar.ps1 -Tudo  # apaga também a pasta (venv + modelo)
```

Nada é instalado fora da pasta do programa, além do atalho.
