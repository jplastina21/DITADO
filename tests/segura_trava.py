"""Auxiliar do teste: segura a trava de instancia unica por alguns segundos."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ditado  # noqa: E402

h = ditado.trava_unica()
print("primeira instancia pegou a trava:", h is not None, flush=True)
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 8)
