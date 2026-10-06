"""Mede a precisão de um ou mais modelos em captchas rotulados.

O rótulo de cada imagem é o nome do arquivo: `u4ep.png` deve ser lido como "u4ep".
Para repetir um texto, use um sufixo: `u4ep_2.png`.

Modelos aceitos em --models:
  - um arquivo .pt treinado por scripts/train_crnn.py (padrão: MODEL_PATH do .env)
  - easyocr (detector + leitura) ou easyocr:full (lê a imagem inteira); requer `pip install easyocr`
  - ddddocr; requer `pip install ddddocr`

Uso:
    python scripts/evaluate.py captchas/rotulados
    python scripts/evaluate.py captchas/rotulados --models models/crnn.pt easyocr ddddocr
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ocr import ALPHABET, CaptchaOCR, load_image  # noqa: E402

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


class EasyOCRReader:
    """EasyOCR's English model, limited to lowercase letters and digits.

    `easyocr` runs its text detector first and joins the boxes left to right;
    `easyocr:full` skips the detector and reads the whole image as one line.
    """

    def __init__(self, spec: str, device: str):
        try:
            import easyocr
        except ImportError:
            raise SystemExit("EasyOCR não está instalado: pip install easyocr")
        import torch

        self.full = spec == "easyocr:full"
        gpu = device in ("auto", "cuda") and torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def predict(self, images):
        texts = []
        for image in images:
            arr = np.asarray(image.convert("L"))
            if self.full:
                h, w = arr.shape
                found = self.reader.recognize(arr, horizontal_list=[[0, w, 0, h]], free_list=[],
                                              allowlist=ALPHABET, detail=1)
            else:
                found = self.reader.readtext(arr, allowlist=ALPHABET, detail=1)
            found = sorted(found, key=lambda item: min(x for x, _ in item[0]))
            texts.append("".join(text for _, text, _ in found))
        return texts


class DdddOcrReader:
    def __init__(self):
        try:
            import ddddocr
        except ImportError:
            raise SystemExit("ddddocr não está instalado: pip install ddddocr")
        self.reader = ddddocr.DdddOcr(show_ad=False)

    def predict(self, images):
        texts = []
        for image in images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            texts.append(self.reader.classification(buf.getvalue()))
        return texts


def load_backend(spec: str, device: str):
    if spec in ("easyocr", "easyocr:full"):
        return EasyOCRReader(spec, device)
    if spec == "ddddocr":
        return DdddOcrReader()
    return CaptchaOCR(spec, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models", nargs="+", help="padrão: MODEL_PATH do .env")
    parser.add_argument("--batch", type=int, default=32, help="imagens por chamada ao modelo")
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        sys.exit(f"Nenhuma imagem em {args.folder}")
    labels = [label_of(p) for p in paths]
    images = [load_image(p.read_bytes()) for p in paths]
    print(f"{len(paths)} imagens rotuladas em {args.folder}\n")

    settings = get_settings()
    summary = []
    for spec in args.models or [settings.model_path]:
        print(f"===== {spec}")
        try:
            backend = load_backend(spec, settings.device)
        except Exception as exc:  # one broken candidate must not stop the comparison
            print(f"   não carregou: {type(exc).__name__}: {exc}\n")
            summary.append((spec, None))
            continue

        start = time.perf_counter()
        preds = []
        for i in range(0, len(images), args.batch):
            preds += backend.predict(images[i:i + args.batch])
        ms = (time.perf_counter() - start) * 1000 / len(images)
        exact = sum(normalize(p) == normalize(t) for p, t in zip(preds, labels))
        errors = sum(edit_distance(normalize(p), normalize(t)) for p, t in zip(preds, labels))
        cer = errors / max(1, sum(len(normalize(t)) for t in labels))
        print(f"acertos={exact}/{len(labels)} ({exact / len(labels):.0%})  CER={cer:.1%}  {ms:.0f} ms/img")
        pairs = [f"{t}->{p or '(vazio)'}{'' if normalize(p) == normalize(t) else '*'}"
                 for p, t in zip(preds, labels)]
        print("   " + "  ".join(pairs) + "\n", flush=True)
        summary.append((spec, (exact, cer, ms)))

    print("===== Resumo")
    for spec, result in summary:
        if result is None:
            print(f"{spec:30s} não carregou")
            continue
        exact, cer, ms = result
        print(f"{spec:30s} {exact}/{len(labels)} ({exact / len(labels):.0%})  CER={cer:.1%}  {ms:.0f} ms/img")
    print("* = errou (comparação ignora maiúsculas e espaços)")


if __name__ == "__main__":
    main()
