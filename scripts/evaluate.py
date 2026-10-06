"""Mede a precisão de um ou mais modelos em captchas rotulados e compara configurações.

O rótulo de cada imagem é o nome do arquivo: `u4ep.png` deve ser lido como "u4ep".
Para repetir um texto, use um sufixo: `u4ep_2.png`.

Modelos aceitos em --models:
  - qualquer TrOCR do Hugging Face ou pasta local (ex.: anuashok/ocr-captcha-v3)
  - AndresDev/captCHAD ou um arquivo .onnx no mesmo formato (requer `pip install onnxruntime`)
  - modelos Qwen2-VL treinados para captcha, como ddanielsantos/qwen2-correios-captcha
    (requer `pip install torchvision`; ~4,4 GB e ~5 GB de RAM)
  - easyocr (detector + leitura) ou easyocr:full (lê a imagem inteira), requer `pip install easyocr`

Uso:
    python scripts/evaluate.py captchas/rotulados
    python scripts/evaluate.py captchas/rotulados --preprocess none median --beams 1
    python scripts/evaluate.py captchas/rotulados --beams 1 --models AndresDev/captCHAD DunnBC22/trocr-base-printed_captcha_ocr
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ocr import PREPROCESS_MODES, CaptchaOCR, load_image, preprocess

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
CAPTCHAD_REPO = "AndresDev/captCHAD"


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


class CaptchadOnnx:
    """CRNN + CTC model from AndresDev/captCHAD, run with onnxruntime."""

    CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    uses_beams = False

    def __init__(self, spec: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("captCHAD precisa do onnxruntime: pip install onnxruntime")
        path = spec
        if not spec.endswith(".onnx"):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=spec, filename="captchad.onnx")
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.preprocess = "none"

    def predict(self, images):
        texts = []
        for image in images:
            image = preprocess(image, self.preprocess).convert("RGB").resize((192, 64), Image.BILINEAR)
            arr = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) - 127.5) / 127.5
            logits = self.session.run(None, {self.input_name: arr[np.newaxis]})[0]
            ids = np.argmax(logits[:, 0, :], axis=-1)
            chars, prev = [], None
            for t in ids:
                if t != prev and t != 0:
                    chars.append(self.CHARSET[t - 1])
                prev = t
            texts.append("".join(chars))
        return texts


class QwenVL:
    """Qwen2-VL fine-tuned to answer a captcha's text, e.g. ddanielsantos/qwen2-correios-captcha."""

    PROMPT = "Decodifique este captcha."  # the instruction used in that model's training data
    FALLBACK_PROCESSOR = "Qwen/Qwen2-VL-2B-Instruct"
    uses_beams = False

    def __init__(self, spec: str, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(spec)
        except (OSError, ValueError, TypeError):
            self.processor = AutoProcessor.from_pretrained(self.FALLBACK_PROCESSOR)
        # bfloat16 halves the RAM on CPU (~5 GB instead of ~9 GB for 2B parameters).
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(spec, dtype=dtype).to(self.device).eval()
        self.preprocess = "none"

    def predict(self, images):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        texts = []
        for image in images:
            image = preprocess(image, self.preprocess).convert("RGB")
            inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=12, do_sample=False)
            answer = out[:, inputs["input_ids"].shape[1]:]
            texts.append(self.processor.batch_decode(answer, skip_special_tokens=True)[0].strip())
        return texts


class EasyOCRReader:
    """EasyOCR's English model, limited to lowercase letters and digits.

    `easyocr` runs its text detector first and joins the boxes left to right;
    `easyocr:full` skips the detector and reads the whole image as one line.
    Pass a custom model name after the colon (`easyocr:full:meu_modelo`) to use
    a recognizer trained with EasyOCR's trainer and saved in ~/.EasyOCR/.
    """

    ALLOWLIST = "abcdefghijklmnopqrstuvwxyz0123456789"
    uses_beams = False

    def __init__(self, spec: str, device: str):
        try:
            import easyocr
        except ImportError:
            raise SystemExit("EasyOCR não está instalado: pip install easyocr")
        _, *options = spec.split(":")
        self.full = bool(options) and options[0] == "full"
        network = options[1] if len(options) > 1 else "standard"
        gpu = device in ("auto", "cuda") and __import__("torch").cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=gpu, recog_network=network, verbose=False)
        self.preprocess = "none"

    def predict(self, images):
        texts = []
        for image in images:
            arr = np.asarray(preprocess(image, self.preprocess).convert("L"))
            if self.full:
                h, w = arr.shape
                found = self.reader.recognize(arr, horizontal_list=[[0, w, 0, h]], free_list=[],
                                              allowlist=self.ALLOWLIST, detail=1)
            else:
                found = self.reader.readtext(arr, allowlist=self.ALLOWLIST, detail=1)
            found = sorted(found, key=lambda item: min(x for x, _ in item[0]))
            texts.append("".join(text for _, text, _ in found))
        return texts


def _model_type(spec: str) -> str:
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(spec).model_type
    except (OSError, ValueError):
        return ""


def load_backend(spec: str, device: str):
    if spec == "easyocr" or spec.startswith("easyocr:"):
        return EasyOCRReader(spec, device)
    if spec.endswith(".onnx") or spec == CAPTCHAD_REPO:
        return CaptchadOnnx(spec)
    if _model_type(spec).startswith("qwen2"):
        return QwenVL(spec, device)
    backend = CaptchaOCR(spec, device=device)
    backend.uses_beams = True
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models", nargs="+", help="padrão: MODEL_PATH do .env")
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
    summary = []
    for spec in args.models or [settings.model_path]:
        print(f"===== {spec}")
        try:
            backend = load_backend(spec, settings.device)
        except Exception as exc:  # one broken candidate must not stop the comparison
            print(f"   não carregou: {type(exc).__name__}: {exc}\n")
            summary.append((spec, None))
            continue

        results = []
        for mode in args.preprocess:
            for beams in (args.beams if backend.uses_beams else [1]):
                backend.preprocess = mode
                if backend.uses_beams:
                    backend.num_beams = beams
                start = time.perf_counter()
                preds = []
                for i in range(0, len(images), args.batch):
                    preds += backend.predict(images[i:i + args.batch])
                ms = (time.perf_counter() - start) * 1000 / len(images)
                exact = sum(normalize(p) == normalize(t) for p, t in zip(preds, labels))
                errors = sum(edit_distance(normalize(p), normalize(t)) for p, t in zip(preds, labels))
                cer = errors / max(1, sum(len(normalize(t)) for t in labels))
                results.append((mode, beams, exact, cer, ms))
                print(f"PREPROCESS={mode:10s} NUM_BEAMS={beams}  acertos={exact}/{len(labels)} "
                      f"({exact / len(labels):.0%})  CER={cer:.1%}  {ms:.0f} ms/img")
                pairs = [f"{t}->{p or '(vazio)'}{'' if normalize(p) == normalize(t) else '*'}"
                         for p, t in zip(preds, labels)]
                print("   " + "  ".join(pairs) + "\n", flush=True)
        summary.append((spec, max(results, key=lambda r: (r[2], -r[3]))))
        del backend

    print("===== Resumo (melhor configuração de cada modelo)")
    for spec, best in summary:
        if best is None:
            print(f"{spec:50s} não carregou")
            continue
        mode, beams, exact, cer, ms = best
        print(f"{spec:50s} {exact}/{len(labels)} ({exact / len(labels):.0%})  CER={cer:.1%}  "
              f"PREPROCESS={mode} NUM_BEAMS={beams}  {ms:.0f} ms/img")
    print("* = errou (comparação ignora maiúsculas e espaços)")


if __name__ == "__main__":
    main()
