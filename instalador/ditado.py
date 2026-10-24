"""Ditado (versao do INSTALADOR): segure a tecla, fale, solte. O texto cai onde o cursor estiver.

Transcricao local (faster-whisper). Nada de voz sai do PC. Usa a placa NVIDIA se houver
(baixa as bibliotecas CUDA sob demanda); senao roda na CPU.

Diferencas da versao da raiz do repo (que e a do PC do Juliano):
  - config, log e modelos moram na pasta do USUARIO (%APPDATA%\\Ditado, %LOCALAPPDATA%\\Ditado);
    a variavel DITADO_DADOS aponta tudo para outra pasta (usada nos testes);
  - dispositivo "auto": GPU NVIDIA se der, CPU (int8) se nao; nunca fica mudo;
  - o modelo e baixado na primeira execucao, com aviso na bandeja.

Uso:
    Ditado.exe                          # modo normal (icone na bandeja)
    Ditado-cli.exe --arquivo x.wav      # so transcreve um arquivo e mede o tempo (mostra texto no console)
"""
from __future__ import annotations

import argparse
import io
import logging
import logging.handlers
import math
import os
import re
import shutil
import struct
import sys
import threading
import time
import wave
from pathlib import Path

if getattr(sys, "frozen", False):
    PASTA = Path(sys.executable).resolve().parent  # pasta do programa instalado
else:
    PASTA = Path(__file__).resolve().parent
_DADOS_ENV = os.environ.get("DITADO_DADOS")
if _DADOS_ENV:  # testes: tudo numa pasta so
    DADOS = Path(_DADOS_ENV)
    LOCAL = Path(_DADOS_ENV)
else:
    DADOS = Path(os.environ.get("APPDATA") or Path.home()) / "Ditado"              # config + log
    LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Ditado"         # modelos + CUDA (pesado)
CONFIG_PADRAO = DADOS / "config.toml"
CONFIG_MODELO = Path(getattr(sys, "_MEIPASS", PASTA)) / "config.padrao.toml"  # PyInstaller 6: dados ficam em _internal   # copiado para CONFIG_PADRAO na primeira execucao
LOG_PATH = DADOS / "ditado.log"
MODELOS = LOCAL / "modelos"
CUDA_DIR = LOCAL / "cuda"
MUTEX = "DitadoInstaladoMutex"                 # o mesmo nome esta em Ditado.iss (AppMutex)

# exe sem console: stdout/stderr sao None e bibliotecas (tqdm, huggingface) quebram ao escrever
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

log = logging.getLogger("ditado")


# --------------------------------------------------------------------------- config

def garantir_config(caminho: Path) -> None:
    """Primeira execucao: copia a config padrao (com comentarios) para a pasta do usuario."""
    if caminho.exists():
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_MODELO.exists():
        caminho.write_bytes(CONFIG_MODELO.read_bytes())
    else:
        caminho.write_text('[atalho]\ntecla = "ctrl_r"\n', encoding="utf-8")


def salvar_tecla(valor: str, caminho: Path = None) -> None:
    """Troca so a linha `tecla = ...` da config, preservando comentarios e o resto."""
    caminho = caminho or CONFIG_PADRAO
    garantir_config(caminho)
    linhas = caminho.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    for i, l in enumerate(linhas):
        if re.match(r"\s*tecla\s*=", l):
            linhas[i] = f'tecla = "{valor}"\n'
            break
    else:
        linhas.append(f'\n[atalho]\ntecla = "{valor}"\n')
    caminho.write_text("".join(linhas), encoding="utf-8")


def carregar_config(caminho: Path = CONFIG_PADRAO) -> dict:
    garantir_config(caminho)
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    # utf-8-sig: aceita config salva com BOM (Bloco de Notas, instalador)
    cfg = tomllib.loads(Path(caminho).read_text(encoding="utf-8-sig"))
    cfg.setdefault("atalho", {}).setdefault("tecla", "ctrl_r")
    cfg.setdefault("audio", {})
    cfg.setdefault("transcricao", {})
    cfg.setdefault("colagem", {})
    return cfg


def configurar_log(console: bool = False) -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    h.setFormatter(fmt)
    log.addHandler(h)
    if console and sys.stdout is not None:
        c = logging.StreamHandler(sys.stdout)
        c.setFormatter(fmt)
        log.addHandler(c)


def _pasta_modelos(cfg: dict) -> Path:
    return Path(cfg["transcricao"].get("pasta_modelos") or MODELOS)


def preparar_ambiente(cfg: dict) -> None:
    """Pasta dos modelos e (se ja baixadas) DLLs da CUDA visiveis antes de importar o ctranslate2."""
    pm = _pasta_modelos(cfg)
    pm.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(pm)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    expor_cuda()


# --------------------------------------------------------------------------- CUDA sob demanda

# Wheels oficiais da NVIDIA no PyPI (so Windows x64). Versoes fixas = as que o ctranslate2 4.8 foi testado.
CUDA_PACOTES = (
    ("nvidia-cublas-cu12", "12.9.2.10", "cublas"),
    ("nvidia-cudnn-cu12", "9.27.0.42", "cudnn"),
    ("nvidia-cuda-nvrtc-cu12", "12.9.86", "cuda_nvrtc"),
)


def expor_cuda() -> bool:
    """Poe as DLLs baixadas em CUDA_DIR no PATH. Devolve True se as tres bibliotecas estao la."""
    if os.name != "nt":
        return False
    ok = True
    for _pkg, _ver, sub in CUDA_PACOTES:
        d = CUDA_DIR / "nvidia" / sub / "bin"
        if d.is_dir() and any(d.glob("*.dll")):
            if str(d) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(d))
                except (OSError, AttributeError):
                    pass
        else:
            ok = False
    return ok


def tem_nvidia() -> bool:
    """True se ha driver NVIDIA instalado (nvcuda.dll carrega e enxerga ao menos uma placa)."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        cuda = ctypes.WinDLL("nvcuda.dll")
        if cuda.cuInit(0) != 0:
            return False
        n = ctypes.c_int(0)
        return cuda.cuDeviceGetCount(ctypes.byref(n)) == 0 and n.value > 0
    except OSError:
        return False


def _baixar(url: str, destino: Path, sha256: str, progresso) -> None:
    import hashlib
    import urllib.request
    parcial = destino.with_suffix(destino.suffix + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "Ditado-instalador"})
    with urllib.request.urlopen(req, timeout=60) as r, open(parcial, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        feito = 0
        while True:
            bloco = r.read(1 << 20)
            if not bloco:
                break
            f.write(bloco)
            h.update(bloco)
            feito += len(bloco)
            if total:
                progresso(feito / total)
    if h.hexdigest() != sha256:
        parcial.unlink(missing_ok=True)
        raise RuntimeError(f"download corrompido (sha256 diferente): {destino.name}")
    parcial.replace(destino)


def baixar_cuda(aviso=lambda *_: None) -> bool:
    """Baixa cublas/cudnn/nvrtc do PyPI e extrai so as DLLs em CUDA_DIR. Devolve True se deu certo."""
    import json
    import urllib.request
    import zipfile
    if expor_cuda():
        return True
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CUDA_DIR / "_download"
    tmp.mkdir(exist_ok=True)
    n = len(CUDA_PACOTES)
    try:
        for i, (pkg, ver, sub) in enumerate(CUDA_PACOTES):
            destino_dll = CUDA_DIR / "nvidia" / sub / "bin"
            if destino_dll.is_dir() and any(destino_dll.glob("*.dll")):
                continue
            log.info("baixando %s %s", pkg, ver)
            with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60) as r:
                meta = json.load(r)
            arq = next(u for u in meta["urls"] if u["filename"].endswith("win_amd64.whl"))
            whl = tmp / arq["filename"]
            _baixar(arq["url"], whl, arq["digests"]["sha256"],
                    lambda f, i=i: aviso("baixando", f"bibliotecas da placa de vídeo {int((i + f) / n * 100)}%"))
            with zipfile.ZipFile(whl) as z:
                for nome in z.namelist():
                    if nome.startswith(f"nvidia/{sub}/bin/") and nome.endswith(".dll"):
                        alvo = CUDA_DIR / nome
                        alvo.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(nome) as src, open(alvo, "wb") as dst:
                            shutil.copyfileobj(src, dst)
            whl.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.error("nao consegui baixar as bibliotecas CUDA: %s", e)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return expor_cuda()


# --------------------------------------------------------------------------- transcricao

ALUCINACOES = (
    "legendas pela comunidade amara.org",
    "obrigado por assistir",
    "inscreva-se no canal",
    "legenda adriana zanotto",
)

REPOS = {  # nome -> repositorio no Hugging Face (o mesmo mapa do faster-whisper)
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
PADROES_ARQUIVOS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def caminho_modelo(nome: str, pasta: Path, aviso=lambda *_: None) -> str:
    """Pasta local do modelo; baixa na primeira vez (com aviso de progresso). Aceita caminho de pasta."""
    if os.path.isdir(nome):
        return nome
    from huggingface_hub import snapshot_download
    repo = REPOS.get(nome) or (nome if "/" in nome else f"Systran/faster-whisper-{nome}")
    cache = str(pasta / "whisper")
    try:
        return snapshot_download(repo, allow_patterns=PADROES_ARQUIVOS, cache_dir=cache, local_files_only=True)
    except Exception:  # noqa: BLE001  (nao esta em disco ainda)
        pass
    from tqdm.auto import tqdm

    class Barra(tqdm):
        _ultimo = -1

        def update(self, n=1):
            r = super().update(n)
            if self.unit == "B" and self.total and self.total > 5_000_000:
                pct = int(self.n * 100 / self.total)
                if pct != Barra._ultimo:
                    Barra._ultimo = pct
                    aviso("baixando", f"modelo {nome} {pct}%")
            return r

    log.info("baixando o modelo %s (%s)", nome, repo)
    aviso("baixando", f"modelo {nome}")
    caminho = snapshot_download(repo, allow_patterns=PADROES_ARQUIVOS, cache_dir=cache, tqdm_class=Barra)
    log.info("modelo %s baixado", nome)
    return caminho


class Motor:
    """Carrega o Whisper UMA vez e transcreve arrays de audio 16 kHz mono float32."""

    def __init__(self, cfg: dict, aviso=None):
        t = cfg["transcricao"]
        self.modelo_gpu = t.get("modelo_gpu", "large-v3-turbo")
        self.modelo_cpu = t.get("modelo_cpu", "small")
        self.idioma = t.get("idioma", "pt")
        self.vad = bool(t.get("vad", True))
        self.beam = int(t.get("beam_size", 5))
        self.cpu_threads = int(t.get("cpu_threads", 0))  # 0 = automatico (nucleos fisicos)
        vocab = t.get("vocabulario", [])
        self.prompt = (t.get("prompt_inicial") or "") + (", ".join(vocab) + "." if vocab else "")
        self.dispositivo_pedido = str(t.get("dispositivo", "auto")).lower()  # auto | cuda | cpu
        self.baixar_cuda = bool(t.get("baixar_cuda", True))
        self.pasta_modelos = _pasta_modelos(cfg)
        self.trocas = dict(cfg.get("substituicoes", {}))
        self.aviso = aviso or (lambda *_: None)
        self.dispositivo = None
        self.modelo_nome = None
        self.modelo = None
        self.quer_gpu = False  # True = tem NVIDIA mas ainda nao esta rodando nela

    def _threads_cpu(self) -> int:
        if self.cpu_threads > 0:
            return self.cpu_threads
        n = os.cpu_count() or 4
        return max(4, n // 2)  # nucleos fisicos (a CPU anuncia 2 threads por nucleo)

    def _tentar(self, nome: str, disp: str, tipo: str) -> None:
        from faster_whisper import WhisperModel
        import numpy as np
        t0 = time.perf_counter()
        caminho = caminho_modelo(nome, self.pasta_modelos, self.aviso)
        kw = {"cpu_threads": self._threads_cpu()} if disp == "cpu" else {}
        m = WhisperModel(caminho, device=disp, compute_type=tipo, **kw)
        # aquecimento: a primeira chamada e lenta (aloca memoria / kernels). Tambem prova que o dispositivo funciona.
        list(m.transcribe(np.zeros(16000, dtype="float32"), language=self.idioma, vad_filter=False)[0])
        self.modelo, self.dispositivo, self.modelo_nome = m, disp, nome
        log.info("modelo %s carregado em %s/%s em %.1fs", nome, disp, tipo, time.perf_counter() - t0)

    def _tentar_gpu(self) -> bool:
        if not expor_cuda():
            return False
        try:
            self._tentar(self.modelo_gpu, "cuda", "float16")
            return True
        except Exception as e:  # noqa: BLE001
            log.error("GPU falhou (%s); seguindo na CPU", e)
            return False

    def carregar(self) -> None:
        """Caminho rapido: GPU se ja estiver tudo em disco, senao CPU (nunca fica sem motor)."""
        if self.dispositivo_pedido != "cpu":
            if tem_nvidia():
                if self._tentar_gpu():
                    return
                self.quer_gpu = self.baixar_cuda and not expor_cuda()
            else:
                log.info("nenhuma placa NVIDIA detectada: usando a CPU")
        self._tentar(self.modelo_cpu, "cpu", "int8")

    def melhorar_para_gpu(self) -> bool:
        """Se ha NVIDIA mas faltam as bibliotecas: baixa e troca para a GPU sem parar de funcionar."""
        if not self.quer_gpu:
            return False
        self.quer_gpu = False
        if not baixar_cuda(self.aviso):
            self.aviso("aviso", "Não consegui baixar as bibliotecas da placa de vídeo; seguindo na CPU.")
            return False
        anterior = (self.modelo, self.dispositivo, self.modelo_nome)
        if self._tentar_gpu():
            return True
        self.modelo, self.dispositivo, self.modelo_nome = anterior
        self.aviso("aviso", "A placa de vídeo não funcionou com o Ditado; seguindo na CPU.")
        return False

    def transcrever(self, audio) -> str:
        segs, _info = self.modelo.transcribe(
            audio,
            language=self.idioma,
            beam_size=self.beam,
            vad_filter=self.vad,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=self.prompt or None,
            condition_on_previous_text=False,
        )
        texto = " ".join(s.text.strip() for s in segs).strip()
        texto = " ".join(texto.split())
        for errado, certo in self.trocas.items():
            texto = re.sub(re.escape(errado), certo, texto, flags=re.IGNORECASE)
        if texto.lower().strip(" .!") in ALUCINACOES:
            log.info("descartado (alucinacao conhecida): %r", texto)
            return ""
        return texto


def ler_wav_16k(caminho: str):
    import numpy as np
    try:
        from faster_whisper import decode_audio
        return decode_audio(caminho, sampling_rate=16000)
    except Exception:  # noqa: BLE001
        with wave.open(caminho, "rb") as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768.0


def vram_usada_mib() -> str:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def modo_arquivo(cfg: dict, arquivos: list[str], repeticoes: int, esperar_gpu: bool = False) -> int:
    print("pasta de modelos:", _pasta_modelos(cfg))
    print("NVIDIA detectada:", tem_nvidia(), "| VRAM:", vram_usada_mib())
    motor = Motor(cfg, aviso=lambda e, d="": print(f"[{e}] {d}", flush=True) if not d.endswith("%") else None)
    t0 = time.perf_counter()
    motor.carregar()
    print(f"modelo={motor.modelo_nome} dispositivo={motor.dispositivo} carga={time.perf_counter() - t0:.1f}s")
    if esperar_gpu and motor.quer_gpu:
        t0 = time.perf_counter()
        ok = motor.melhorar_para_gpu()
        print(f"melhorar_para_gpu={ok} em {time.perf_counter() - t0:.1f}s -> modelo={motor.modelo_nome} dispositivo={motor.dispositivo}")
    for arq in arquivos:
        audio = ler_wav_16k(arq)
        dur = len(audio) / 16000
        tempos = []
        texto = ""
        for _ in range(repeticoes):
            t1 = time.perf_counter()
            texto = motor.transcrever(audio)
            tempos.append(time.perf_counter() - t1)
        print(f"\n[{arq}] audio={dur:.1f}s tempos={', '.join(f'{x:.2f}s' for x in tempos)} melhor={min(tempos):.2f}s")
        print("TEXTO:", texto)
    print("\nVRAM no fim:", vram_usada_mib())
    return 0


# --------------------------------------------------------------------------- clipboard / colagem

class ClipboardWin:
    """Clipboard do Windows guardando TODOS os formatos que dao pra copiar como bytes/texto."""

    # formatos que sao handles GDI (nao bytes): o Windows recria a partir do CF_DIB / texto
    _PULAR = {2, 3, 9, 14}  # CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE

    def __init__(self):
        import win32clipboard
        self.wc = win32clipboard

    def _abrir(self):
        for _ in range(20):
            try:
                self.wc.OpenClipboard()
                return
            except Exception:  # noqa: BLE001  (outro programa segurando)
                time.sleep(0.025)
        self.wc.OpenClipboard()

    def salvar(self) -> list:
        guardado = []
        self._abrir()
        try:
            fmt = self.wc.EnumClipboardFormats(0)
            while fmt:
                if fmt not in self._PULAR:
                    try:
                        guardado.append((fmt, self.wc.GetClipboardData(fmt)))
                    except Exception:  # noqa: BLE001
                        pass
                fmt = self.wc.EnumClipboardFormats(fmt)
        finally:
            self.wc.CloseClipboard()
        return guardado

    def por_texto(self, texto: str) -> None:
        self._abrir()
        try:
            self.wc.EmptyClipboard()
            self.wc.SetClipboardData(self.wc.CF_UNICODETEXT, texto)
        finally:
            self.wc.CloseClipboard()

    def ler_texto(self) -> str | None:
        self._abrir()
        try:
            if self.wc.IsClipboardFormatAvailable(self.wc.CF_UNICODETEXT):
                return self.wc.GetClipboardData(self.wc.CF_UNICODETEXT)
            return None
        finally:
            self.wc.CloseClipboard()

    def restaurar(self, guardado: list) -> None:
        self._abrir()
        try:
            self.wc.EmptyClipboard()
            for fmt, dado in guardado:
                try:
                    self.wc.SetClipboardData(fmt, dado)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self.wc.CloseClipboard()


def enviar_ctrl_v() -> None:
    from pynput.keyboard import Controller, Key
    k = Controller()
    with k.pressed(Key.ctrl_l):
        k.press("v")
        k.release("v")


def colar(texto: str, clipboard, enviar=enviar_ctrl_v, espera_restaurar: float = 0.4) -> None:
    """Salva o clipboard, poe o texto, manda Ctrl+V e devolve o clipboard como estava."""
    guardado = None
    try:
        guardado = clipboard.salvar()
    except Exception as e:  # noqa: BLE001
        log.warning("nao consegui salvar o clipboard (%s); vai colar mesmo assim", e)
    clipboard.por_texto(texto)
    time.sleep(0.03)
    enviar()
    if guardado is not None:
        time.sleep(espera_restaurar)  # o app de destino le o clipboard depois do Ctrl+V
        clipboard.restaurar(guardado)


# --------------------------------------------------------------------------- som / gravacao

def _wav_bipe(freq: float, dur: float, volume: float) -> bytes:
    sr = 22050
    n = int(sr * dur)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        amostras = []
        for i in range(n):
            env = min(1.0, i / (sr * 0.005), (n - i) / (sr * 0.01))  # ataque/queda curtos (sem estalo)
            amostras.append(int(32767 * volume * env * math.sin(2 * math.pi * freq * i / sr)))
        w.writeframes(struct.pack(f"<{n}h", *amostras))
    return buf.getvalue()


class Bipe:
    def __init__(self, cfg: dict):
        a = cfg["audio"]
        self.ligado = bool(a.get("bipe", True))
        vol = float(a.get("volume_bipe", 0.25))
        self.ini = _wav_bipe(880, 0.07, vol)
        self.fim = _wav_bipe(587, 0.07, vol)

    def tocar(self, qual: str) -> None:
        if not self.ligado or os.name != "nt":
            return
        import winsound
        dados = self.ini if qual == "ini" else self.fim
        # SND_MEMORY nao aceita SND_ASYNC: toca numa thread pra nao travar a gravacao
        threading.Thread(target=winsound.PlaySound, args=(dados, winsound.SND_MEMORY), daemon=True).start()


class Gravador:
    """Microfone padrao, 16 kHz mono. O stream fica aberto e so guarda o audio enquanto grava."""

    SR = 16000

    def __init__(self, cfg: dict):
        a = cfg["audio"]
        self.sempre_aberto = bool(a.get("microfone_sempre_aberto", True))
        self.pre_roll = float(a.get("pre_roll_s", 0.2))
        self.dispositivo = a.get("dispositivo") or None
        self._lock = threading.Lock()
        self._gravando = False
        self._blocos: list = []
        self._anel: list = []
        self._stream = None
        if self.sempre_aberto:
            self._abrir()

    def _callback(self, indata, frames, tempo, status):  # noqa: ARG002
        bloco = indata[:, 0].copy()
        with self._lock:
            if self._gravando:
                self._blocos.append(bloco)
            else:
                self._anel.append(bloco)
                total = sum(len(b) for b in self._anel)
                while self._anel and total - len(self._anel[0]) >= self.pre_roll * self.SR:
                    total -= len(self._anel.pop(0))

    def _abrir(self):
        import sounddevice as sd
        if self._stream is not None:
            return
        self._stream = sd.InputStream(samplerate=self.SR, channels=1, dtype="float32",
                                      blocksize=1600, device=self.dispositivo, callback=self._callback)
        self._stream.start()
        log.info("microfone aberto: %s", sd.query_devices(self._stream.device)["name"])

    def _fechar(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def comecar(self):
        try:
            if self._stream is None or not self._stream.active:
                self._stream = None
                self._abrir()
        except Exception as e:  # noqa: BLE001
            log.error("microfone falhou: %s", e)
            raise
        with self._lock:
            self._blocos = list(self._anel)
            self._anel = []
            self._gravando = True

    def parar(self):
        import numpy as np
        with self._lock:
            self._gravando = False
            blocos, self._blocos = self._blocos, []
        if not self.sempre_aberto:
            self._fechar()
        if not blocos:
            return np.zeros(0, dtype="float32")
        return np.concatenate(blocos)


# --------------------------------------------------------------------------- app

class Ditado:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.motor = Motor(cfg, aviso=self.avisar)
        self.bipe = Bipe(cfg)
        self.gravador = None
        self.clip = ClipboardWin()
        self.tecla_nome = cfg["atalho"]["tecla"]
        self.min_s = float(cfg["audio"].get("minimo_s", 0.4))
        self.espaco_fim = bool(cfg["colagem"].get("espaco_no_fim", True))
        self.espera_restaurar = float(cfg["colagem"].get("espera_restaurar_s", 0.4))
        self.estado = "carregando"
        self.gravando = False
        self.cancelado = False
        self.colando = False
        self.t_inicio = 0.0
        self.icone = None
        self._fila = threading.Semaphore(1)
        self._ja_avisou: set = set()

    # --- bandeja
    def _imagem(self, cor):
        from PIL import Image, ImageDraw
        im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse((4, 4, 60, 60), fill=cor)
        d.rounded_rectangle((25, 14, 39, 38), radius=7, fill="white")  # microfone
        d.arc((19, 24, 45, 46), 0, 180, fill="white", width=3)
        d.line((32, 46, 32, 52), fill="white", width=3)
        return im

    CORES = {"carregando": "#888888", "baixando": "#2060d0", "pronto": "#1f8f3a", "gravando": "#d02020",
             "transcrevendo": "#d08000", "erro": "#5a005a"}

    def avisar(self, tipo: str, detalhe: str = "") -> None:
        """Chamado pelo Motor: 'baixando' (estado azul + titulo com o progresso) ou 'aviso' (notificacao)."""
        if tipo == "baixando":
            if self.estado in ("carregando", "baixando"):
                self.mudar_estado("baixando", detalhe)
            elif self.icone is not None:  # ja esta pronto (troca CPU -> GPU em segundo plano): so o titulo
                self.icone.title = f"Ditado: {self.estado} - baixando {detalhe}"[:127]
            alvo = detalhe.split(" ", 1)[0]  # "modelo" ou "bibliotecas": uma notificacao so por alvo
            if alvo not in self._ja_avisou:
                self._ja_avisou.add(alvo)
                if alvo == "modelo":
                    self.notificar("Baixando o modelo de voz (só na primeira vez; pode levar alguns minutos). "
                                   "O ícone fica azul até terminar.")
        else:
            self.notificar(detalhe)

    def notificar(self, msg: str) -> None:
        log.info("aviso: %s", msg)
        try:
            if self.icone is not None:
                self.icone.notify(msg, "Ditado")
        except Exception:  # noqa: BLE001
            pass

    def mudar_estado(self, estado: str, detalhe: str = ""):
        self.estado = estado
        if self.icone is not None:
            self.icone.icon = self._imagem(self.CORES.get(estado, "#888888"))
            self.icone.title = f"Ditado: {estado}{(' - ' + detalhe) if detalhe else ''}"[:127]
            try:
                self.icone.update_menu()
            except Exception:  # noqa: BLE001
                pass

    # --- teclado
    def _eh_tecla(self, key) -> bool:
        from pynput.keyboard import Key, KeyCode
        alvo = getattr(Key, self.tecla_nome, None)
        if alvo is not None:
            return key == alvo
        return isinstance(key, KeyCode) and key.char == self.tecla_nome

    def ao_apertar(self, key):
        if self.colando:
            return
        if self._eh_tecla(key):
            if not self.gravando and self.estado == "pronto":
                try:
                    self.gravador.comecar()
                except Exception as e:  # noqa: BLE001
                    self.mudar_estado("erro", f"microfone: {e}")
                    return
                self.gravando, self.cancelado = True, False
                self.t_inicio = time.perf_counter()
                self.bipe.tocar("ini")
                self.mudar_estado("gravando")
        elif self.gravando and not self.cancelado:
            # outra tecla junto (Ctrl+C, Ctrl+Win do Wispr...): era atalho, nao ditado
            self.cancelado = True
            log.info("cancelado: outra tecla apertada junto (%s)", key)

    def ao_soltar(self, key):
        if self.colando or not self._eh_tecla(key) or not self.gravando:
            return
        self.gravando = False
        audio = self.gravador.parar()
        dur = time.perf_counter() - self.t_inicio
        if self.cancelado:
            self.mudar_estado("pronto")
            return
        self.bipe.tocar("fim")
        if dur < self.min_s:
            log.info("descartado: %.2fs (< %.2fs)", dur, self.min_s)
            self.mudar_estado("pronto")
            return
        threading.Thread(target=self._processar, args=(audio,), daemon=True).start()

    def _processar(self, audio):
        with self._fila:
            self.mudar_estado("transcrevendo")
            try:
                t0 = time.perf_counter()
                texto = self.motor.transcrever(audio)
                dt = time.perf_counter() - t0
                log.info("transcrito %.1fs de audio em %.2fs (%d caracteres)", len(audio) / 16000, dt, len(texto))
                if texto:
                    if self.espaco_fim:
                        texto += " "
                    self.colando = True
                    try:
                        colar(texto, self.clip, espera_restaurar=self.espera_restaurar)
                    finally:
                        self.colando = False
            except Exception as e:  # noqa: BLE001
                log.exception("erro ao transcrever/colar: %s", e)
            finally:
                if self.estado != "erro":
                    self.mudar_estado("pronto")

    # --- ciclo de vida
    def _carregar(self):
        try:
            self.gravador = Gravador(self.cfg)
            self.motor.carregar()
            self._pronto()
            if self.motor.quer_gpu:
                self.notificar("Placa NVIDIA encontrada. Vou baixar as bibliotecas dela em segundo plano "
                               "(cerca de 1 GB, só desta vez). Enquanto isso você já pode ditar.")
                if self.motor.melhorar_para_gpu():
                    self._pronto()
                    self.notificar("Pronto: agora o Ditado usa a placa de vídeo (bem mais rápido).")
                else:
                    self.mudar_estado("pronto", self._detalhe())
        except Exception as e:  # noqa: BLE001
            log.exception("falha ao iniciar: %s", e)
            self.mudar_estado("erro", str(e))
            self.notificar(f"O Ditado não conseguiu iniciar: {e}. Veja o log (menu do ícone).")

    def _detalhe(self) -> str:
        return f"{self.motor.modelo_nome} ({'placa de vídeo' if self.motor.dispositivo == 'cuda' else 'CPU'})"

    def _pronto(self):
        self.mudar_estado("pronto", self._detalhe())
        log.info("pronto. Segure %s para ditar. (%s)", self.tecla_nome, self._detalhe())

    def rodar(self):
        import pystray
        from pynput import keyboard

        listener = keyboard.Listener(on_press=self.ao_apertar, on_release=self.ao_soltar)
        listener.start()
        log.info("atalho registrado: segurar '%s'", self.tecla_nome)

        def abrir(caminho):
            import subprocess
            subprocess.Popen(["notepad.exe", str(caminho)])  # noqa: S603

        def abrir_log(_icon, _item):
            abrir(LOG_PATH)

        def abrir_config(_icon, _item):
            abrir(CONFIG_PADRAO)

        def escolher_tecla(valor):
            def f(_icon, _item):
                self.tecla_nome = valor  # vale na hora: _eh_tecla le este campo
                try:
                    salvar_tecla(valor)
                    log.info("tecla trocada pelo menu: %s", valor)
                except Exception as e:  # noqa: BLE001
                    log.error("nao consegui salvar a tecla na config: %s", e)
                    self.notificar("A tecla mudou agora, mas não consegui salvar na configuração.")
                try:
                    self.icone.update_menu()
                except Exception:  # noqa: BLE001
                    pass
            return f

        def e_tecla(valor):
            return lambda _item: self.tecla_nome == valor

        def tentar_de_novo(_icon, _item):
            if self.estado == "erro":
                self.mudar_estado("carregando")
                threading.Thread(target=self._carregar, daemon=True).start()

        def sair(icon, _item):
            log.info("saindo pelo menu")
            listener.stop()
            icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem(lambda _i: f"Status: {self.estado}", None, enabled=False),
            pystray.MenuItem("Tecla: Ctrl esquerdo", escolher_tecla("ctrl_l"), checked=e_tecla("ctrl_l"), radio=True),
            pystray.MenuItem("Tecla: Ctrl direito", escolher_tecla("ctrl_r"), checked=e_tecla("ctrl_r"), radio=True),
            pystray.MenuItem("Tentar de novo (se deu erro)", tentar_de_novo, visible=lambda _i: self.estado == "erro"),
            pystray.MenuItem("Abrir configuração (tecla, modelo...)", abrir_config),
            pystray.MenuItem("Abrir log", abrir_log),
            pystray.MenuItem("Sair", sair),
        )
        self.icone = pystray.Icon("ditado", self._imagem(self.CORES["carregando"]), "Ditado: carregando", menu)

        def ao_aparecer(icone):
            icone.visible = True
            threading.Thread(target=self._carregar, daemon=True).start()

        self.icone.run(setup=ao_aparecer)
        os._exit(0)


def verificar(cfg: dict) -> int:
    """Autoteste do pacote: importa tudo que o modo normal usa, sem registrar teclado nem abrir a bandeja."""
    import importlib
    for mod in ("pystray", "pynput.keyboard", "sounddevice", "win32clipboard", "PIL.Image",
                "faster_whisper", "ctranslate2", "onnxruntime", "av", "huggingface_hub"):
        importlib.import_module(mod)
        print("ok", mod)
    d = Ditado(cfg)
    d._imagem(d.CORES["pronto"])
    print("config:", CONFIG_PADRAO, "| tecla:", d.tecla_nome, "| NVIDIA:", tem_nvidia())
    return 0


def trava_unica():
    """Mutex nomeado do Windows: se ja existe, outra instancia esta rodando."""
    if os.name != "nt":
        return object()
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    h = k32.CreateMutexW(None, False, MUTEX)
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ditado local com Whisper")
    ap.add_argument("--arquivo", nargs="+", help="transcreve arquivo(s) de audio e sai (teste)")
    ap.add_argument("--repetir", type=int, default=3, help="repeticoes no modo --arquivo")
    ap.add_argument("--esperar-gpu", action="store_true", help="no modo --arquivo: baixa as bibliotecas CUDA e troca para a GPU")
    ap.add_argument("--verificar", action="store_true", help="confere se todas as bibliotecas do programa carregam (sem abrir atalho nem bandeja)")
    ap.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], help="forca o dispositivo (so neste uso)")
    ap.add_argument("--config", default=str(CONFIG_PADRAO))
    args = ap.parse_args(argv)

    cfg = carregar_config(Path(args.config))
    if args.dispositivo:
        cfg["transcricao"]["dispositivo"] = args.dispositivo
    configurar_log(console=bool(args.arquivo))
    preparar_ambiente(cfg)

    if args.arquivo:
        return modo_arquivo(cfg, args.arquivo, args.repetir, args.esperar_gpu)
    if args.verificar:
        return verificar(cfg)

    trava = trava_unica()
    if trava is None:
        log.warning("ja tem um Ditado aberto; esta instancia vai sair")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "O Ditado já está aberto (veja o ícone na bandeja, perto do relógio).", "Ditado", 0x40)
        except Exception:  # noqa: BLE001
            pass
        return 1
    log.info("=== iniciando Ditado (pid %d) ===", os.getpid())
    Ditado(cfg).rodar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
