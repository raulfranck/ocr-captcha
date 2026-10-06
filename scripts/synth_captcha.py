"""Generate synthetic captchas in the style of captchas/rotulados.

Style: 200x68 grayscale, 4-6 lowercase letters/digits in a sans font, each
glyph slightly rotated, 2-4 wavy lines crossing the image, salt-and-pepper
dots and a 16-level palette. File names follow the evaluate.py convention
(text as the name), so the output folder can be evaluated or trained on.

    python scripts/synth_captcha.py saida/ --count 5000
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(size: int, path: str | None = None) -> ImageFont.FreeTypeFont:
    for candidate in [path] if path else FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise SystemExit("Nenhuma fonte sans encontrada; passe uma com --font caminho.ttf")


def _glyph(char: str, font: ImageFont.FreeTypeFont, angle: float) -> Image.Image:
    """Render one character on a tile whose height is the font's full line box.

    Every tile shares the same baseline, so descenders (g, p, q, y) still hang
    below it after the tiles are pasted at a common top. Only the sides are trimmed.
    """
    ascent, descent = font.getmetrics()
    left, _, right, _ = font.getbbox(char)
    tile = Image.new("L", (right - left + 12, ascent + descent + 12), 0)
    ImageDraw.Draw(tile).text((6 - left, 6), char, font=font, fill=255)
    tile = tile.rotate(angle, resample=Image.BICUBIC)
    x0, _, x1, _ = tile.getbbox() or (0, 0, tile.width, 0)
    return tile.crop((x0, 0, x1, tile.height))


def _wave(draw: ImageDraw.ImageDraw, width: int, y: float, rng: random.Random) -> None:
    x0 = rng.uniform(-20, width * 0.4)
    x1 = rng.uniform(x0 + 100, width + 40)
    amp = rng.uniform(3, 9)
    period = rng.uniform(35, 80)
    phase = rng.uniform(0, 2 * math.pi)
    slope = rng.uniform(-0.15, 0.15)
    thickness = rng.choice((2, 2, 3))
    shade = rng.randint(0, 50)
    points = [
        (x, y + slope * (x - x0) + amp * math.sin(2 * math.pi * x / period + phase))
        for x in range(int(x0), int(x1), 2)
    ]
    draw.line(points, fill=shade, width=thickness, joint="curve")
    if rng.random() < 0.3:  # the real captchas often show a doubled stroke
        offset = rng.choice((-4, -3, 3, 4))
        draw.line([(x, py + offset) for x, py in points], fill=shade, width=thickness)


def generate(text: str, rng: random.Random, font_path: str | None = None,
             width: int = 200, height: int = 68) -> Image.Image:
    image = Image.new("L", (width, height), 251)
    font = find_font(rng.randint(32, 38), font_path)
    glyphs = [_glyph(c, font, rng.uniform(-15, 15)) for c in text]
    spacing = rng.randint(-2, 1)
    total = sum(g.width for g in glyphs) + spacing * (len(glyphs) - 1)
    x = rng.randint(0, max(0, width - total))
    line_box = glyphs[0].height
    y = rng.randint(-8, max(-8, height - line_box + 4))
    mask = Image.new("L", (width, height), 0)
    for g in glyphs:
        mask.paste(g, (x, y + rng.randint(-3, 3)), g)
        x += g.width + spacing
    image.paste(rng.randint(0, 20), (0, 0), mask)

    # Most lines run above or below the text; in the real captchas only one or
    # two cross it, so the letters stay readable.
    _, text_top, _, text_bottom = mask.getbbox() or (0, 0, 0, height)
    free = [v for v in range(4, height - 2) if v < text_top - 6 or v > text_bottom + 6]
    draw = ImageDraw.Draw(image)
    crossing = rng.choices((0, 1, 2), weights=(3, 5, 2))[0]
    for i in range(rng.randint(max(2, crossing), 4)):
        if i < crossing or not free:
            line_y = rng.uniform(text_top + 4, max(text_top + 5, text_bottom - 4))
        else:
            line_y = rng.choice(free)
        _wave(draw, width, line_y, rng)

    pixels = image.load()
    density = rng.uniform(0.04, 0.10)
    for _ in range(int(width * height * density)):
        px, py = rng.randrange(width), rng.randrange(height)
        pixels[px, py] = rng.choice((4, 22, 72, 200, 234))
    # The real captchas are stored with 16 gray levels.
    return image.point(lambda v: (v // 16) * 17)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--font", help="Fonte .ttf (padrão: Arial/Liberation Sans/DejaVu Sans)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    for _ in range(args.count):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(args.min_len, args.max_len)))
        seen[text] = seen.get(text, 0) + 1
        name = text if seen[text] == 1 else f"{text}_{seen[text]}"
        generate(text, rng, args.font).save(args.out_dir / f"{name}.png")
    print(f"{args.count} captchas em {args.out_dir}")


if __name__ == "__main__":
    main()
