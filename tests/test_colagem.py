"""Testa salvar/colar/restaurar o clipboard SEM mandar Ctrl+V de verdade.

    venv\Scripts\python -m pytest tests  (ou)  venv\Scripts\python tests\test_colagem.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ditado  # noqa: E402


class ClipFalso:
    def __init__(self, inicial):
        self.formatos = list(inicial)

    def salvar(self):
        return list(self.formatos)

    def por_texto(self, t):
        self.formatos = [(13, t)]

    def ler_texto(self):
        return dict(self.formatos).get(13)

    def restaurar(self, g):
        self.formatos = list(g)


def test_colar_com_clipboard_falso():
    clip = ClipFalso([(13, "antes"), (49999, b"\x00\x01")])
    visto = []
    ditado.colar("olá próstata", clip, enviar=lambda: visto.append(clip.ler_texto()), espera_restaurar=0)
    assert visto == ["olá próstata"], visto           # na hora do Ctrl+V o texto ditado estava la
    assert clip.formatos == [(13, "antes"), (49999, b"\x00\x01")]  # e depois voltou tudo


def test_colar_clipboard_real_windows():
    if sys.platform != "win32":
        return
    clip = ditado.ClipboardWin()
    clip.por_texto("conteudo original do Juliano")
    visto = []
    ditado.colar("texto ditado com acentuação: Maringá", clip,
                 enviar=lambda: visto.append(clip.ler_texto()), espera_restaurar=0.05)
    assert visto == ["texto ditado com acentuação: Maringá"], visto
    assert clip.ler_texto() == "conteudo original do Juliano", clip.ler_texto()


if __name__ == "__main__":
    test_colar_com_clipboard_falso()
    print("ok: clipboard falso")
    test_colar_clipboard_real_windows()
    print("ok: clipboard real do Windows")
