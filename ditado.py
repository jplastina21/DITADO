"""Ditado: segure a tecla, fale, solte. O texto cai onde o cursor estiver.

Dois jeitos de ditar:
  - SEGURAR (padrao Ctrl esquerdo): segura, fala, solta. Outra tecla junto cancela (era atalho, tipo Ctrl+C).
  - ALTERNAR (padrao Win+J): aperta uma vez pra comecar, aperta de novo pra parar. Pra fala longa.
    Outras teclas nao cancelam; o audio vai sendo gravado na pasta gravacoes (wav) enquanto voce fala.

Transcricao local na GPU (faster-whisper). Nada de voz sai do PC.

Uso:
    pythonw ditado.py                 # modo normal (icone na bandeja)
    python  ditado.py --arquivo x.wav # so transcreve um arquivo e mede o tempo
"""
from __future__ import annotations

import argparse
import io
import logging
import logging.handlers
import math
import os
import queue
import re
import struct
import sys
import threading
import time
import wave
from pathlib import Path

PASTA = Path(__file__).resolve().parent
CONFIG_PADRAO = PASTA / "config.toml"
LOG_PATH = PASTA / "ditado.log"
GRAVACOES_PADRAO = PASTA / "gravacoes"

log = logging.getLogger("ditado")


# --------------------------------------------------------------------------- config

def carregar_config(caminho: Path = CONFIG_PADRAO) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(caminho, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("atalho", {}).setdefault("tecla", "ctrl_r")
    cfg.setdefault("audio", {})
    cfg.setdefault("transcricao", {})
    cfg.setdefault("colagem", {})
    cfg.setdefault("alternar", {})
    return cfg


def configurar_log(console: bool = False) -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    h = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    h.setFormatter(fmt)
    log.addHandler(h)
    if console and sys.stdout is not None:
        c = logging.StreamHandler(sys.stdout)
        c.setFormatter(fmt)
        log.addHandler(c)


def preparar_ambiente(cfg: dict) -> None:
    """Modelos no D: e DLLs da CUDA (pacotes nvidia-*) visiveis antes de importar o ctranslate2."""
    modelos = cfg["transcricao"].get("pasta_modelos") or str(PASTA / "modelos")
    os.environ.setdefault("HF_HOME", modelos)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if os.name != "nt":
        return
    try:
        import nvidia  # pacote namespace dos nvidia-*-cu12
        bases = list(getattr(nvidia, "__path__", []))
    except ImportError:
        bases = []
    for base in bases:
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = Path(base) / sub / "bin"
            if d.is_dir():
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(d))
                except (OSError, AttributeError):
                    pass


# --------------------------------------------------------------------------- transcricao

ALUCINACOES = (
    "legendas pela comunidade amara.org",
    "obrigado por assistir",
    "inscreva-se no canal",
    "legenda adriana zanotto",
)


class Motor:
    """Carrega o Whisper UMA vez e transcreve arrays de audio 16 kHz mono float32."""

    def __init__(self, cfg: dict):
        t = cfg["transcricao"]
        self.modelo_nome = t.get("modelo", "large-v3-turbo")
        self.fallback = t.get("modelo_fallback", "medium")
        self.idioma = t.get("idioma", "pt")
        self.vad = bool(t.get("vad", True))
        self.beam = int(t.get("beam_size", 5))
        vocab = t.get("vocabulario", [])
        self.prompt = (t.get("prompt_inicial") or "") + (", ".join(vocab) + "." if vocab else "")
        self.dispositivo_pedido = t.get("dispositivo", "cuda")
        self.dispositivo = None
        self.modelo = None
        self.pasta_modelos = os.environ.get("HF_HOME")
        self.trocas = dict(cfg.get("substituicoes", {}))

    def carregar(self) -> None:
        from faster_whisper import WhisperModel

        tentativas = []
        if self.dispositivo_pedido == "cuda":
            tentativas += [(self.modelo_nome, "cuda", "float16"), (self.fallback, "cuda", "float16")]
        tentativas += [(self.modelo_nome, "cpu", "int8"), (self.fallback, "cpu", "int8")]
        ultimo_erro = None
        for nome, disp, tipo in tentativas:
            try:
                t0 = time.perf_counter()
                m = WhisperModel(nome, device=disp, compute_type=tipo,
                                 download_root=os.path.join(self.pasta_modelos or str(PASTA / "modelos"), "whisper"))
                # aquecimento: a primeira chamada na GPU e lenta (compila kernels / aloca)
                import numpy as np
                list(m.transcribe(np.zeros(16000, dtype="float32"), language=self.idioma, vad_filter=False)[0])
                self.modelo, self.dispositivo, self.modelo_nome = m, disp, nome
                log.info("modelo %s carregado em %s/%s em %.1fs", nome, disp, tipo, time.perf_counter() - t0)
                if disp == "cpu" and self.dispositivo_pedido == "cuda":
                    log.warning("ATENCAO: CUDA falhou, rodando na CPU (vai ficar lento). Ultimo erro: %s", ultimo_erro)
                return
            except Exception as e:  # noqa: BLE001
                ultimo_erro = e
                log.error("falhou carregar %s em %s: %s", nome, disp, e)
        raise RuntimeError(f"nenhum modelo carregou: {ultimo_erro}")

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


def modo_arquivo(cfg: dict, arquivos: list[str], repeticoes: int) -> int:
    print("VRAM antes de carregar:", vram_usada_mib())
    motor = Motor(cfg)
    t0 = time.perf_counter()
    motor.carregar()
    print(f"modelo={motor.modelo_nome} dispositivo={motor.dispositivo} carga={time.perf_counter() - t0:.1f}s")
    print("VRAM com modelo carregado:", vram_usada_mib())
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


# --------------------------------------------------------------------------- atalho global (modo alternar)

_MODS = {"win": 0x0008, "ctrl": 0x0002, "alt": 0x0001, "shift": 0x0004}
_MOD_NOREPEAT = 0x4000  # segurar o atalho nao dispara de novo
_VK_NOMES = {
    "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "pause": 0x13, "scroll_lock": 0x91, "space": 0x20, "`": 0xC0, "'": 0xDE, ";": 0xBA,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC,
}
_VK_MODIFICADORES = {"win": (0x5B, 0x5C), "ctrl": (0x11,), "alt": (0x12,), "shift": (0x10,)}


def ler_atalho(texto: str) -> tuple[int, int, list[str]]:
    """'win+j' -> (modificadores do RegisterHotKey, codigo virtual da tecla, nomes dos modificadores)."""
    partes = [p.strip().lower() for p in texto.replace(" ", "").split("+") if p.strip()]
    if len(partes) < 2:
        raise ValueError(f"atalho {texto!r}: use modificador + tecla, ex. 'win+j'")
    *mods, tecla = partes
    m = 0
    for x in mods:
        if x not in _MODS:
            raise ValueError(f"atalho {texto!r}: modificador {x!r} desconhecido (win, ctrl, alt, shift)")
        m |= _MODS[x]
    if len(tecla) == 1 and (tecla.isalpha() or tecla.isdigit()):
        vk = ord(tecla.upper())
    elif re.fullmatch(r"f([1-9]|1\d|2[0-4])", tecla):
        vk = 0x70 + int(tecla[1:]) - 1
    elif tecla in _VK_NOMES:
        vk = _VK_NOMES[tecla]
    else:
        raise ValueError(f"atalho {texto!r}: tecla {tecla!r} desconhecida")
    return m, vk, mods


def nome_bonito(texto: str) -> str:
    return "+".join(p.strip().capitalize() if len(p.strip()) > 1 else p.strip().upper()
                    for p in texto.split("+") if p.strip())


class AtalhoGlobal:
    """Atalho do Windows (RegisterHotKey). Se outro programa ou o proprio Windows ja usa a combinacao,
    o registro FALHA (erro 1409) - e assim que se sabe que nao vai colidir. Enquanto registrado, o
    Windows entrega a combinacao so pra nos (nao chega no app em foco)."""

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012

    def __init__(self, texto: str, ao_disparar):
        self.texto = texto
        self.mods, self.vk, self.nomes_mods = ler_atalho(texto)
        self.ao_disparar = ao_disparar
        self.ok = False
        self.erro = ""
        self._tid = None
        self._pronto = threading.Event()
        self._t = None

    def iniciar(self, espera: float = 5.0) -> bool:
        self._t = threading.Thread(target=self._loop, name="atalho-alternar", daemon=True)
        self._t.start()
        self._pronto.wait(espera)
        return self.ok

    def _loop(self):
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        k = ctypes.WinDLL("kernel32")
        self._tid = k.GetCurrentThreadId()
        if not u.RegisterHotKey(None, 1, self.mods | _MOD_NOREPEAT, self.vk):
            err = ctypes.get_last_error()
            self.erro = (f"{nome_bonito(self.texto)} ja e usado por outro programa/Windows (erro 1409)"
                         if err == 1409 else f"RegisterHotKey falhou (erro {err})")
            self._pronto.set()
            return
        self.ok = True
        self._pronto.set()
        msg = wintypes.MSG()
        try:
            while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == self.WM_HOTKEY and msg.wParam == 1:
                    threading.Thread(target=self._disparar, daemon=True).start()
        finally:
            u.UnregisterHotKey(None, 1)

    def _disparar(self):
        try:
            self.ao_disparar()
        except Exception as e:  # noqa: BLE001
            log.exception("erro no atalho alternar: %s", e)

    def parar(self):
        if self._tid:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self._tid, self.WM_QUIT, 0, 0)

    def esperar_soltar(self, limite: float = 5.0) -> bool:
        """Espera o usuario soltar Win/Ctrl/... do atalho antes do Ctrl+V. Win ainda apertado + Ctrl+V
        viraria Ctrl+Win (o atalho do Wispr) ou Win+Ctrl+V (som do Windows)."""
        if os.name != "nt":
            return True
        import ctypes
        gks = ctypes.windll.user32.GetAsyncKeyState
        vks = [v for n in self.nomes_mods for v in _VK_MODIFICADORES[n]] + [self.vk]
        fim = time.monotonic() + limite
        while time.monotonic() < fim:
            if not any(gks(v) & 0x8000 for v in vks):
                return True
            time.sleep(0.03)
        return False


def limpar_gravacoes(pasta: Path, guardar: int) -> list[Path]:
    """Deixa so as `guardar` gravacoes mais novas (wav + txt com o mesmo nome). Devolve o que apagou."""
    wavs = sorted(Path(pasta).glob("*.wav"), key=lambda p: p.name, reverse=True)
    apagados = []
    for w in wavs[max(guardar, 1):]:
        for p in (w, w.with_suffix(".txt")):
            try:
                if p.exists():
                    p.unlink()
                    apagados.append(p)
            except OSError as e:
                log.warning("nao consegui apagar %s: %s", p, e)
    return apagados


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
        self.nao = _wav_bipe(330, 0.15, vol)       # atalho ignorado (ocupado)
        self.teto = _wav_bipe(440, 0.35, vol)      # parou sozinho no teto de tempo

    def tocar(self, qual: str) -> None:
        if not self.ligado or os.name != "nt":
            return
        import winsound
        dados = {"ini": self.ini, "fim": self.fim, "nao": self.nao, "teto": self.teto}.get(qual, self.fim)
        # SND_MEMORY nao aceita SND_ASYNC: toca numa thread pra nao travar a gravacao
        threading.Thread(target=winsound.PlaySound, args=(dados, winsound.SND_MEMORY), daemon=True).start()


class EscritorWav:
    """Grava blocos float32 num wav 16 kHz mono int16 DURANTE a gravacao (thread propria).

    O cabecalho e corrigido e o arquivo descarregado no disco a cada bloco: se o programa travar
    ou o PC desligar no meio, o que ja foi falado fica num wav valido.
    """

    def __init__(self, caminho: Path, sr: int = 16000):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.caminho, "wb")  # noqa: SIM115
        self._w = wave.open(self._f, "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(sr)
        self._fila: queue.Queue = queue.Queue()
        self.erro = None
        self._t = threading.Thread(target=self._loop, name="escritor-wav", daemon=True)
        self._t.start()

    def escrever(self, bloco) -> None:  # chamado do callback do microfone: so enfileira
        self._fila.put(bloco)

    def _loop(self):
        import numpy as np
        while True:
            b = self._fila.get()
            if b is None:
                break
            if self.erro is not None:
                continue
            try:
                self._w.writeframes((np.clip(b, -1.0, 1.0) * 32767).astype("<i2").tobytes())
                self._f.flush()
            except Exception as e:  # noqa: BLE001
                self.erro = e
                log.error("falhou gravar %s: %s", self.caminho, e)

    def fechar(self) -> None:
        self._fila.put(None)
        self._t.join(timeout=10)
        try:
            self._w.close()
        finally:
            self._f.close()


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
        self._escritor: EscritorWav | None = None
        if self.sempre_aberto:
            self._abrir()

    def _callback(self, indata, frames, tempo, status):  # noqa: ARG002
        bloco = indata[:, 0].copy()
        with self._lock:
            if self._gravando:
                self._blocos.append(bloco)
                if self._escritor is not None:
                    self._escritor.escrever(bloco)
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

    def comecar(self, arquivo: Path | None = None):
        """arquivo: se vier, o audio tambem vai sendo escrito nesse wav enquanto grava (modo alternar)."""
        try:
            if self._stream is None or not self._stream.active:
                self._stream = None
                self._abrir()
        except Exception as e:  # noqa: BLE001
            log.error("microfone falhou: %s", e)
            raise
        escritor = EscritorWav(arquivo, self.SR) if arquivo is not None else None
        with self._lock:
            self._blocos = list(self._anel)
            self._anel = []
            self._escritor = escritor
            if escritor is not None:
                for b in self._blocos:  # o pre-roll tambem vai pro arquivo
                    escritor.escrever(b)
            self._gravando = True

    def parar(self):
        import numpy as np
        with self._lock:
            self._gravando = False
            blocos, self._blocos = self._blocos, []
            escritor, self._escritor = self._escritor, None
        if escritor is not None:
            escritor.fechar()
        if not self.sempre_aberto:
            self._fechar()
        if not blocos:
            return np.zeros(0, dtype="float32")
        return np.concatenate(blocos)


# --------------------------------------------------------------------------- app

class Ditado:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.motor = Motor(cfg)
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
        # modo alternar (fala longa)
        alt = cfg["alternar"]
        self.atalho_alt_txt = str(alt.get("atalho", "win+j")).strip()
        self.teto_s = float(alt.get("teto_min", 30)) * 60
        self.guardar = int(alt.get("guardar_ultimas", 20))
        self.pasta_grav = Path(alt.get("pasta_gravacoes") or GRAVACOES_PADRAO)
        self.atalho_alt: AtalhoGlobal | None = None
        self.atalho_alt_erro = ""
        self.alternando = False
        self.arquivo_alt: Path | None = None
        self.t_inicio_alt = 0.0
        self._sessao_alt = 0
        self._lock_alt = threading.Lock()
        self.intervalo_vigia = 1.0

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

    CORES = {"carregando": "#888888", "pronto": "#1f8f3a", "gravando": "#d02020",
             "transcrevendo": "#d08000", "erro": "#5a005a", "gravando (alternar)": "#c0007a"}

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

    # --- modo alternar (Win+J): aperta = comeca, aperta de novo = para. Outras teclas nao cancelam.
    def alternar(self):
        with self._lock_alt:
            if self.alternando:
                self._parar_alternar("atalho")
            elif self.estado == "pronto" and not self.gravando:
                self._comecar_alternar()
            else:
                log.info("atalho alternar ignorado (estado: %s)", self.estado)
                self.bipe.tocar("nao")

    def _comecar_alternar(self):
        self.pasta_grav.mkdir(parents=True, exist_ok=True)
        arq = self.pasta_grav / (time.strftime("%Y-%m-%d_%H%M%S") + ".wav")
        try:
            self.gravador.comecar(arquivo=arq)
        except Exception as e:  # noqa: BLE001
            self.mudar_estado("erro", f"microfone: {e}")
            return
        self.alternando, self.arquivo_alt = True, arq
        self.t_inicio_alt = time.perf_counter()
        self._sessao_alt += 1
        self.bipe.tocar("ini")
        self.mudar_estado("gravando (alternar)", "0:00")
        log.info("alternar: comecou a gravar em %s (teto %.0f min)", arq.name, self.teto_s / 60)
        threading.Thread(target=self._vigia, args=(self._sessao_alt,), daemon=True).start()

    def _vigia(self, sessao: int):
        """Teto de tempo + relogio no icone."""
        ultimo = -1
        while True:
            time.sleep(self.intervalo_vigia)
            with self._lock_alt:
                if not self.alternando or sessao != self._sessao_alt:
                    return
                dt = time.perf_counter() - self.t_inicio_alt
                if dt >= self.teto_s:
                    log.warning("alternar: chegou no teto de %.0f min, parando sozinho", self.teto_s / 60)
                    self._parar_alternar("teto")
                    return
                seg = int(dt) // 5 * 5
                if seg != ultimo:
                    ultimo = seg
                    self.mudar_estado("gravando (alternar)",
                                      f"{seg // 60}:{seg % 60:02d} ({nome_bonito(self.atalho_alt_txt)} para)")

    def _parar_alternar(self, motivo: str):
        """Chamar com _lock_alt na mao."""
        self.alternando = False
        audio = self.gravador.parar()
        arq, self.arquivo_alt = self.arquivo_alt, None
        dur = len(audio) / Gravador.SR
        self.bipe.tocar("teto" if motivo == "teto" else "fim")
        log.info("alternar: parou (%s), %.1fs gravados em %s", motivo, dur, arq.name if arq else "?")
        if dur < self.min_s:
            log.info("alternar: descartado, curto demais (%.2fs)", dur)
            try:
                if arq:
                    arq.unlink()
            except OSError:
                pass
            self.mudar_estado("pronto")
            return
        self.mudar_estado("transcrevendo", f"{dur / 60:.1f} min de audio")
        threading.Thread(target=self._processar_alternar, args=(audio, arq), daemon=True).start()

    def _colar(self, texto: str) -> None:
        colar(texto, self.clip, espera_restaurar=self.espera_restaurar)

    def _processar_alternar(self, audio, arq: Path | None):
        with self._fila:
            self.mudar_estado("transcrevendo", f"{len(audio) / 16000 / 60:.1f} min de audio")
            texto = ""
            try:
                t0 = time.perf_counter()
                texto = self.motor.transcrever(audio)
                log.info("alternar: transcrito %.1fs de audio em %.1fs (%d caracteres)",
                         len(audio) / 16000, time.perf_counter() - t0, len(texto))
                if arq is not None:
                    arq.with_suffix(".txt").write_text(texto, encoding="utf-8")
                if texto:
                    if self.espaco_fim:
                        texto += " "
                    if self.atalho_alt is not None and not self.atalho_alt.esperar_soltar():
                        log.warning("alternar: Win ainda apertado depois de 5s; colando mesmo assim")
                    self.colando = True
                    try:
                        self._colar(texto)
                    finally:
                        self.colando = False
            except Exception as e:  # noqa: BLE001
                log.exception("alternar: erro ao transcrever/colar: %s (audio salvo em %s)", e, arq)
                if texto:  # transcreveu mas nao colou: deixa no clipboard pra colar a mao
                    try:
                        self.clip.por_texto(texto)
                        log.info("alternar: texto deixado no clipboard (Ctrl+V a mao)")
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                try:
                    limpar_gravacoes(self.pasta_grav, self.guardar)
                except Exception as e:  # noqa: BLE001
                    log.warning("limpeza das gravacoes falhou: %s", e)
                if self.estado != "erro":
                    self.mudar_estado("pronto")

    def ultima_gravacao(self) -> Path | None:
        wavs = sorted(self.pasta_grav.glob("*.wav"), key=lambda p: p.name)
        return wavs[-1] if wavs else None

    def retranscrever_ultima(self):
        """Menu: transcreve de novo a ultima gravacao, salva o .txt, poe no clipboard e abre o .txt."""
        arq = self.ultima_gravacao()
        if arq is None or self.alternando:
            log.info("retranscrever: %s", "gravando agora" if self.alternando else "nenhuma gravacao")
            self.bipe.tocar("nao")
            return None
        with self._fila:
            self.mudar_estado("transcrevendo", arq.name)
            try:
                audio = ler_wav_16k(str(arq))
                t0 = time.perf_counter()
                texto = self.motor.transcrever(audio)
                txt = arq.with_suffix(".txt")
                txt.write_text(texto, encoding="utf-8")
                log.info("retranscrever: %s (%.1fs de audio) em %.1fs, %d caracteres",
                         arq.name, len(audio) / 16000, time.perf_counter() - t0, len(texto))
                if texto:
                    self.clip.por_texto(texto)
                if os.name == "nt" and getattr(self, "abrir_txt", True):
                    os.startfile(str(txt))  # noqa: S606
                return texto
            except Exception as e:  # noqa: BLE001
                log.exception("retranscrever falhou: %s", e)
                return None
            finally:
                if self.estado != "erro":
                    self.mudar_estado("pronto")

    def registrar_alternar(self) -> bool:
        if not self.atalho_alt_txt:
            log.info("modo alternar desligado no config")
            return False
        try:
            self.atalho_alt = AtalhoGlobal(self.atalho_alt_txt, self.alternar)
        except ValueError as e:
            self.atalho_alt_erro = str(e)
            log.error("atalho alternar invalido: %s", e)
            return False
        if self.atalho_alt.iniciar():
            log.info("atalho registrado: alternar '%s' (aperta = comeca, aperta de novo = para; teto %.0f min)",
                     nome_bonito(self.atalho_alt_txt), self.teto_s / 60)
            return True
        self.atalho_alt_erro = self.atalho_alt.erro
        log.error("atalho alternar NAO registrado: %s. Troque [alternar] atalho no config.toml", self.atalho_alt.erro)
        self.atalho_alt = None
        return False

    # --- ciclo de vida
    def _carregar(self):
        try:
            self.gravador = Gravador(self.cfg)
            self.motor.carregar()
            extra = "" if self.motor.dispositivo == "cuda" else " (CPU! CUDA falhou, veja o log)"
            self.mudar_estado("pronto", f"{self.motor.modelo_nome}{extra}")
            log.info("pronto. Segure %s para ditar.", self.tecla_nome)
        except Exception as e:  # noqa: BLE001
            log.exception("falha ao iniciar: %s", e)
            self.mudar_estado("erro", str(e))

    def rodar(self):
        import pystray
        from pynput import keyboard

        listener = keyboard.Listener(on_press=self.ao_apertar, on_release=self.ao_soltar)
        listener.start()
        log.info("atalho registrado: segurar '%s'", self.tecla_nome)
        self.registrar_alternar()

        def abrir_log(_icon, _item):
            os.startfile(str(LOG_PATH))  # noqa: S606

        def abrir_gravacoes(_icon, _item):
            self.pasta_grav.mkdir(parents=True, exist_ok=True)
            os.startfile(str(self.pasta_grav))  # noqa: S606

        def retranscrever(_icon, _item):
            threading.Thread(target=self.retranscrever_ultima, daemon=True).start()

        def sair(icon, _item):
            log.info("saindo pelo menu")
            with self._lock_alt:
                if self.alternando:  # nao perde a fala: fecha o wav direito antes de sair
                    self.alternando = False
                    self.gravador.parar()
                    log.info("alternar: gravacao interrompida pelo Sair; o wav ficou em %s", self.pasta_grav)
            if self.atalho_alt is not None:
                self.atalho_alt.parar()
            listener.stop()
            icon.stop()

        def texto_alternar(_i):
            if self.atalho_alt is not None:
                return f"Fala longa: {nome_bonito(self.atalho_alt_txt)} (aperta e aperta de novo)"
            if self.atalho_alt_txt:
                return f"Fala longa: {nome_bonito(self.atalho_alt_txt)} NAO registrou (veja o log)"
            return "Fala longa: desligada"

        menu = pystray.Menu(
            pystray.MenuItem(lambda _i: f"Status: {self.estado}", None, enabled=False),
            pystray.MenuItem(lambda _i: f"Segure: {self.tecla_nome}", None, enabled=False),
            pystray.MenuItem(texto_alternar, None, enabled=False),
            pystray.MenuItem("Transcrever ultima gravacao de novo", retranscrever),
            pystray.MenuItem("Abrir pasta de gravacoes", abrir_gravacoes),
            pystray.MenuItem("Abrir log", abrir_log),
            pystray.MenuItem("Sair", sair),
        )
        self.icone = pystray.Icon("ditado", self._imagem(self.CORES["carregando"]), "Ditado: carregando", menu)
        threading.Thread(target=self._carregar, daemon=True).start()
        self.icone.run()
        os._exit(0)


def trava_unica():
    """Mutex nomeado do Windows: se ja existe, outra instancia esta rodando."""
    if os.name != "nt":
        return object()
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    h = k32.CreateMutexW(None, False, "Local\\DitadoClaudinhoMutex")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ditado local com Whisper")
    ap.add_argument("--arquivo", nargs="+", help="transcreve arquivo(s) de audio e sai (teste)")
    ap.add_argument("--repetir", type=int, default=3, help="repeticoes no modo --arquivo")
    ap.add_argument("--config", default=str(CONFIG_PADRAO))
    args = ap.parse_args(argv)

    cfg = carregar_config(Path(args.config))
    configurar_log(console=bool(args.arquivo))
    preparar_ambiente(cfg)

    if args.arquivo:
        return modo_arquivo(cfg, args.arquivo, args.repetir)

    trava = trava_unica()
    if trava is None:
        log.warning("ja tem um Ditado aberto; esta instancia vai sair")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "O Ditado ja esta aberto (veja o icone na bandeja).", "Ditado", 0x40)
        except Exception:  # noqa: BLE001
            pass
        return 1
    log.info("=== iniciando Ditado (pid %d) ===", os.getpid())
    Ditado(cfg).rodar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
