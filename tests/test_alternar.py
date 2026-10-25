"""Testa o modo ALTERNAR (fala longa) sem microfone, sem Whisper e sem mandar tecla de verdade.

    venv\\Scripts\\python -m pytest tests   (ou)   venv\\Scripts\\python tests\\test_alternar.py
"""
import ctypes
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ditado  # noqa: E402


class GravadorFalso:
    SR = 16000

    def __init__(self, segundos=3.0):
        self.segundos = segundos
        self.arquivo = None
        self.gravando = False

    def comecar(self, arquivo=None):
        self.arquivo, self.gravando = arquivo, True
        if arquivo is not None:  # como o real: o wav nasce no disco ao comecar
            e = ditado.EscritorWav(arquivo)
            e.escrever(np.zeros(int(self.segundos * self.SR), dtype="float32"))
            e.fechar()

    def parar(self):
        self.gravando = False
        return np.zeros(int(self.segundos * self.SR), dtype="float32")


class MotorFalso:
    def __init__(self, texto="fala longa transcrita", erro=False):
        self.texto, self.erro, self.chamadas = texto, erro, 0

    def transcrever(self, audio):
        self.chamadas += 1
        return self.texto


class BipeFalso:
    def __init__(self):
        self.tocados = []

    def tocar(self, qual):
        self.tocados.append(qual)


class ClipFalso:
    def __init__(self):
        self.texto = None

    def por_texto(self, t):
        self.texto = t


def montar(pasta, **alt):
    cfg = ditado.carregar_config()
    cfg["alternar"].update({"pasta_gravacoes": str(pasta), **alt})
    d = ditado.Ditado(cfg)
    d.gravador, d.motor, d.bipe, d.clip = GravadorFalso(), MotorFalso(), BipeFalso(), ClipFalso()
    d.colados = []
    d._colar = d.colados.append
    d.abrir_txt = False
    d.estado = "pronto"
    return d


def esperar(cond, limite=5.0):
    fim = time.monotonic() + limite
    while time.monotonic() < fim:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_ler_atalho():
    assert ditado.ler_atalho("win+j")[:2] == (0x8, 0x4A)
    assert ditado.ler_atalho("Win + Shift + F12")[:2] == (0x8 | 0x4, 0x7B)
    for ruim in ("j", "win+", "hyper+j", "win+tecla_que_nao_existe"):
        try:
            ditado.ler_atalho(ruim)
        except ValueError:
            continue
        raise AssertionError(f"devia recusar {ruim!r}")
    assert ditado.nome_bonito("win+j") == "Win+J"


def test_wav_escrito_durante_a_gravacao():
    with tempfile.TemporaryDirectory() as tmp:
        arq = Path(tmp) / "x.wav"
        e = ditado.EscritorWav(arq)
        for _ in range(10):
            e.escrever(np.full(1600, 0.1, dtype="float32"))
        assert esperar(lambda: arq.stat().st_size >= 44 + 10 * 1600 * 2)
        with wave.open(str(arq), "rb") as w:  # ANTES de fechar: o wav ja e valido e tem o audio
            assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 16000)
        e.fechar()
        assert len(ditado.ler_wav_16k(str(arq))) == 16000


def test_comeca_para_e_outra_tecla_nao_cancela():
    from pynput.keyboard import Key, KeyCode
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp)
        d.alternar()
        assert d.alternando and d.estado == "gravando (alternar)" and d.bipe.tocados == ["ini"]
        arq = d.arquivo_alt
        assert arq is not None and arq.exists()           # o wav ja esta no disco durante a fala
        for k in (KeyCode.from_char("\x03"), KeyCode.from_char("a"), Key.space, Key.ctrl_l, Key.ctrl_r, Key.cmd):
            d.ao_apertar(k)                                # digitar / Ctrl+C / Ctrl esquerdo
            d.ao_soltar(k)
        assert d.alternando and not d.gravando and not d.cancelado
        d.alternar()                                       # mesmo atalho de novo = para
        assert not d.alternando and d.bipe.tocados == ["ini", "fim"]
        assert esperar(lambda: d.colados and d.estado == "pronto")
        assert d.colados == ["fala longa transcrita "]
        assert arq.with_suffix(".txt").read_text(encoding="utf-8") == "fala longa transcrita"


def test_segurar_continua_igual_quando_nao_esta_alternando():
    from pynput.keyboard import Key, KeyCode
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp)
        tecla = getattr(Key, d.tecla_nome)                 # a tecla de segurar do config (ctrl_l / ctrl_r)
        d.ao_apertar(tecla)
        assert d.gravando and d.bipe.tocados == ["ini"] and d.gravador.arquivo is None  # sem wav no segurar
        d.ao_apertar(KeyCode.from_char("\x03"))            # Ctrl+C: cancela, como sempre
        assert d.cancelado
        d.ao_soltar(tecla)
        assert not d.gravando and d.estado == "pronto"


def test_atalho_ignorado_enquanto_segura_ou_transcreve():
    from pynput.keyboard import Key
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp)
        tecla = getattr(Key, d.tecla_nome)
        d.ao_apertar(tecla)
        assert d.gravando
        d.alternar()
        assert not d.alternando and d.bipe.tocados[-1] == "nao"
        d.ao_soltar(tecla)


def test_teto_para_sozinho():
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp, teto_min=0.6 / 60)
        d.intervalo_vigia = 0.05
        d.alternar()
        assert esperar(lambda: not d.alternando, 3)
        assert d.bipe.tocados == ["ini", "teto"]
        assert esperar(lambda: d.colados == ["fala longa transcrita "])


def test_colagem_falhou_texto_fica_no_clipboard_e_no_txt():
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp)

        def quebra(_t):
            raise RuntimeError("colagem falhou")
        d._colar = quebra
        d.alternar()
        arq = d.arquivo_alt
        d.alternar()
        assert esperar(lambda: d.estado == "pronto" and d.clip.texto)
        assert d.clip.texto == "fala longa transcrita "
        assert arq.with_suffix(".txt").read_text(encoding="utf-8") == "fala longa transcrita"


def test_guarda_so_as_ultimas():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        for i in range(25):
            (p / f"2026-10-{i + 1:02d}_120000.wav").write_bytes(b"x")
            (p / f"2026-10-{i + 1:02d}_120000.txt").write_text("t")
        ditado.limpar_gravacoes(p, 20)
        wavs = sorted(x.name for x in p.glob("*.wav"))
        assert len(wavs) == 20 and wavs[0] == "2026-10-06_120000.wav" and len(list(p.glob("*.txt"))) == 20


def test_retranscrever_ultima():
    with tempfile.TemporaryDirectory() as tmp:
        d = montar(tmp)
        assert d.retranscrever_ultima() is None              # sem gravacao: nao quebra
        d.alternar()
        d.alternar()
        assert esperar(lambda: d.estado == "pronto" and d.colados)
        d.motor.texto = "segunda tentativa"
        assert d.retranscrever_ultima() == "segunda tentativa"
        assert d.clip.texto == "segunda tentativa"
        assert d.ultima_gravacao().with_suffix(".txt").read_text(encoding="utf-8") == "segunda tentativa"


def test_registerhotkey_dispara_e_detecta_colisao():
    """Registra de verdade no Windows (na sessao em que o teste roda) e entrega um WM_HOTKEY pra
    thread - nao aperta tecla nenhuma. Um segundo registro da mesma combinacao tem que falhar."""
    if sys.platform != "win32":
        return
    chamou = threading.Event()
    a = ditado.AtalhoGlobal("win+j", chamou.set)
    if not a.iniciar() and "1459" in a.erro:  # SSH/servico: sem area de trabalho interativa
        print("pulado: RegisterHotKey precisa da sessao do usuario (erro 1459)")
        return
    assert a.ok, a.erro
    try:
        b = ditado.AtalhoGlobal("win+j", lambda: None)
        assert not b.iniciar() and "1409" in b.erro, b.erro   # colisao detectada
        ctypes.windll.user32.PostThreadMessageW(a._tid, a.WM_HOTKEY, 1, 0)
        assert chamou.wait(3)
    finally:
        a.parar()


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_") and callable(f):
            f()
            print("ok:", nome)
