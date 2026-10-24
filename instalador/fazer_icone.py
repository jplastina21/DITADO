"""Gera ditado.ico (usado pelo atalho da Area de Trabalho)."""
from pathlib import Path
from PIL import Image, ImageDraw

im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.ellipse((8, 8, 248, 248), fill="#1f8f3a")
d.rounded_rectangle((100, 52, 156, 156), radius=28, fill="white")
d.arc((76, 96, 180, 186), 0, 180, fill="white", width=12)
d.line((128, 186, 128, 212), fill="white", width=12)
im.save(Path(__file__).with_name("ditado.ico"), sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
