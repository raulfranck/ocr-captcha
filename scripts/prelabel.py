"""Pré-rotula captchas novos com o modelo, para você só corrigir os erros.

Renomeia cada imagem da pasta para o texto que o modelo leu (`m9v3e.png`; se o texto
repetir, `m9v3e_2.png`), o mesmo formato de captchas/rotulados. Imagens cujo nome já é
um rótulo ficam como estão, então rodar de novo não desfaz as suas correções.

Também gera uma folha de conferência (--sheet) com cada imagem ao lado do texto lido,
das leituras menos confiáveis para as mais confiáveis. Para revisar: abra a folha,
corrija o nome dos arquivos errados e mova a pasta para captchas/rotulados.

    python scripts/prelabel.py captchas/revisar
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ocr import ALPHABET, CaptchaOCR, load_image  # noqa: E402
from scripts.evaluate import IMAGE_SUFFIXES, label_of  # noqa: E402

LOW_CONFIDENCE = 0.8


def is_label(path: Path) -> bool:
    text = label_of(path)
    return bool(text) and set(text) <= set(ALPHABET)


def free_name(folder: Path, text: str, suffix: str) -> Path:
    path, n = folder / f"{text}{suffix}", 1
    while path.exists():
        n += 1
        path = folder / f"{text}_{n}{suffix}"
    return path


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "consola.ttf", "C:/Windows/Fonts/consola.ttf", "Menlo.ttc", "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_sheet(rows: list[tuple[Path, Image.Image, float]], out: Path) -> None:
    """rows: (new path, image, confidence), already sorted."""
    font, small = _font(22), _font(15)
    cell_w, cell_h, cols = 640, 120, 2
    sheet = Image.new("RGB", (cell_w * cols, cell_h * ((len(rows) + cols - 1) // cols)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (path, image, confidence) in enumerate(rows):
        x, y = (k % cols) * cell_w, (k // cols) * cell_h
        if confidence < LOW_CONFIDENCE:
            draw.rectangle([x, y, x + cell_w - 4, y + cell_h - 4], fill=(255, 238, 200))
        sheet.paste(image.resize((300, 102)), (x + 6, y + 8))
        draw.text((x + 320, y + 20), path.stem, font=font, fill=(0, 0, 0))
        color = (190, 0, 0) if confidence < LOW_CONFIDENCE else (0, 120, 0)
        draw.text((x + 320, y + 60), f"confiança {confidence:.0%}", font=small, fill=color)
    sheet.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", help="padrão: MODEL_PATH do .env")
    parser.add_argument("--sheet", type=Path, default=Path("captchas/conferir.png"), help="folha de conferência")
    args = parser.parse_args()

    settings = get_settings()
    ocr = CaptchaOCR(args.model or settings.model_path, device=settings.device)
    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    pending = [p for p in paths if not is_label(p)]
    print(f"{len(paths)} imagens em {args.folder}, {len(pending)} sem rótulo")

    rows = []
    for path in pending:
        image = load_image(path.read_bytes())
        text, confidence = ocr.predict_with_confidence([image])[0]
        target = free_name(args.folder, text or "vazio", path.suffix.lower())
        path.rename(target)
        rows.append((target, image, confidence))
        print(f"{path.name:45s} -> {target.name:14s} confiança {confidence:.0%}")

    if rows:
        rows.sort(key=lambda row: row[2])
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        save_sheet(rows, args.sheet)
        doubtful = sum(c < LOW_CONFIDENCE for _, _, c in rows)
        print(f"\nFolha de conferência: {args.sheet} ({doubtful} leituras com confiança abaixo de "
              f"{LOW_CONFIDENCE:.0%}, em destaque no topo)")


if __name__ == "__main__":
    main()
