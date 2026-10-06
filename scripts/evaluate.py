"""Mede a precisão do modelo em captchas rotulados e compara configurações.

O rótulo de cada imagem é o nome do arquivo: `u4ep.png` deve ser lido como "u4ep".
Para repetir um texto, use um sufixo: `u4ep_2.png`.

Uso:
    python scripts/evaluate.py captchas/rotulados
    python scripts/evaluate.py captchas/rotulados --preprocess none median --beams 1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ocr import PREPROCESS_MODES, CaptchaOCR, load_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def label_of(path: Path) -> str:
    stem = path.stem
    head, sep, tail = stem.rpartition("_")
    return head if sep and tail.isdigit() else stem


def normalize(text: str) -> str:
    return "".join(text.split()).lower()


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--preprocess", nargs="+", default=list(PREPROCESS_MODES), choices=PREPROCESS_MODES)
    parser.add_argument("--beams", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--batch", type=int, default=8, help="imagens por chamada ao modelo")
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        sys.exit(f"Nenhuma imagem em {args.folder}")
    labels = [label_of(p) for p in paths]
    images = [load_image(p.read_bytes()) for p in paths]
    print(f"{len(paths)} imagens rotuladas em {args.folder}")
    print("rótulos: " + " ".join(labels) + "\n")

    settings = get_settings()
    ocr = CaptchaOCR(settings.model_path, device=settings.device)

    results = []
    for mode in args.preprocess:
        for beams in args.beams:
            ocr.preprocess, ocr.num_beams = mode, beams
            start = time.perf_counter()
            preds = []
            for i in range(0, len(images), args.batch):
                preds += ocr.predict(images[i:i + args.batch])
            ms = (time.perf_counter() - start) * 1000 / len(images)
            exact = sum(normalize(p) == normalize(t) for p, t in zip(preds, labels))
            errors = sum(edit_distance(normalize(p), normalize(t)) for p, t in zip(preds, labels))
            cer = errors / max(1, sum(len(normalize(t)) for t in labels))
            results.append((mode, beams, exact, cer))
            print(f"PREPROCESS={mode:10s} NUM_BEAMS={beams}  acertos={exact}/{len(labels)} "
                  f"({exact / len(labels):.0%})  CER={cer:.1%}  {ms:.0f} ms/img")
            pairs = [f"{t}->{p or '(vazio)'}{'' if normalize(p) == normalize(t) else '*'}" for p, t in zip(preds, labels)]
            print("   " + "  ".join(pairs) + "\n", flush=True)

    mode, beams, exact, cer = max(results, key=lambda r: (r[2], -r[3]))
    print(f"Melhor: PREPROCESS={mode} NUM_BEAMS={beams} ({exact}/{len(labels)}, CER {cer:.1%})")
    print("* = errou (comparação ignora maiúsculas e espaços)")


if __name__ == "__main__":
    main()
