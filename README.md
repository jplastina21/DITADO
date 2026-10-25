# ditado

Ditado por voz **local** para Windows: segure uma tecla, fale, solte — o texto cai onde o cursor
estiver. Para fala longa (10 min ou mais) tem um segundo atalho, **Win+J**, que liga e desliga. A transcrição roda na placa de vídeo (Whisper `large-v3-turbo` via faster-whisper);
nenhum áudio sai do computador e não há mensalidade. Feito como substituto do Wispr Flow, só
para ditar.

## Como usar

1. Abra pelo atalho **Ditado** na Área de Trabalho (ou `DITADO.bat`). Aparece um ícone redondo
   com microfone na bandeja, perto do relógio. Cinza = carregando (uns 5 s), **verde = pronto**.
2. Clique onde quer o texto, **segure o Ctrl** (o lado está no `config.toml`; no PC do Juliano é o
   esquerdo), fale, solte. Bipe agudo = gravando,
   bipe grave = parou. O ícone fica laranja enquanto transcreve e o texto é colado.
3. Ctrl com outra tecla (Ctrl+C, Ctrl+Win...) não dita: o programa entende que era atalho.
   Toque rápido (menos de 0,4 s) também é ignorado.
4. Para fechar: clique com o botão direito no ícone > **Sair**. **Abrir log** mostra o que aconteceu.

O que estava no seu Ctrl+C antes de ditar volta sozinho depois da colagem.

### Fala longa: Win+J (aperta para começar, aperta de novo para parar)

1. Aperte **Win+J** uma vez: bipe agudo, ícone **rosa** ("gravando (alternar)", com o tempo
   correndo). Pode soltar as teclas e falar à vontade.
2. Enquanto fala, pode digitar, clicar, usar o PC: **outras teclas não cancelam nada**.
3. Aperte **Win+J** de novo: bipe grave, o ícone fica laranja enquanto transcreve e o texto é
   colado onde o cursor estiver **nessa hora**. 10 minutos de fala levam uns 17 s para transcrever.
4. Teto de segurança: com 30 minutos ele para sozinho (bipe longo) e transcreve o que gravou.

O áudio vai sendo gravado em disco **enquanto você fala**, em `gravacoes\` (wav 16 kHz mono,
nome com data e hora). Se o programa ou o PC travar no meio, o que já foi falado está lá. O texto
de cada gravação também fica salvo num `.txt` com o mesmo nome; se a colagem falhar, o texto fica
no Ctrl+V para colar à mão. Guarda as últimas 20 gravações e apaga as mais velhas.

No menu do ícone: **Transcrever última gravação de novo** (transcreve o último wav de novo, salva
o `.txt`, põe o texto no Ctrl+V e abre o `.txt`) e **Abrir pasta de gravações**.

Por que Win+J: o programa registra o atalho no próprio Windows (`RegisterHotKey`). Se a
combinação já for do Windows ou de outro programa, o registro falha e o log diz isso — não tem
colisão silenciosa. Em 24/10/2026, no Windows 11 (build 26300) do Juliano, Win+J estava livre
(Win+H, Win+F, Win+D, Win+L, Win+O, Win+F1, Win+F11 e Win+1/0 estavam ocupados). Também não
encosta nos atalhos do Wispr (todos com Ctrl+Win, Ctrl+Alt+Win, Shift+Alt+Z/X, Alt+M+Win).
Enquanto o Ditado está aberto, Win+J é só dele (o Windows não repassa para outro app). Antes de
colar, ele espera você soltar o Win (Win + Ctrl+V viraria o atalho do Wispr).

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
mostra o tempo e a VRAM. Teste do clipboard: `venv\Scripts\python tests\test_colagem.py`. Teste do modo fala longa
(sem microfone, sem Whisper, sem apertar tecla): `venv\Scripts\python tests\test_alternar.py`
(o teste do atalho de verdade só roda na sessão do usuário; por SSH ele é pulado).

## Configurar (`config.toml`)

Depois de mudar, feche (Sair) e abra de novo.

- **Fala longa (`[alternar]`):** `atalho = "win+j"` (ex. `"win+f12"`, `"win+shift+j"`; `""`
  desliga), `teto_min = 30`, `guardar_ultimas = 20`, `pasta_gravacoes = ""` (vazio = `gravacoes\`
  na pasta do programa). Se o atalho não registrar, o menu do ícone mostra "NAO registrou" e o
  log diz por quê.
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
