"""Diagnóstico: mostra como o modelo carregou e o que ele lê em cada variação da imagem.

Uso:
    python scripts/diagnose.py caminho/captcha1.png caminho/captcha2.png

Salva, ao lado de cada imagem, as versões que o modelo recebe (debug_*.png).
"""
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import transformers
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import get_settings
from app.ocr import load_image


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    settings = get_settings()
    print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"modelo: {settings.model_path}\n")

    processor = TrOCRProcessor.from_pretrained(settings.model_path)
    model, info = VisionEncoderDecoderModel.from_pretrained(settings.model_path, output_loading_info=True)
    model.eval()
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        values = list(info.get(key) or [])
        print(f"{key}: {len(values)} {values[:10]}")
    print()

    for path in map(Path, sys.argv[1:]):
        original = Image.open(path)
        original.load()
        alpha = original.convert("RGBA").getchannel("A").getextrema()
        print(f"== {path.name}: mode={original.mode} size={original.size} "
              f"alpha(min,max)={alpha} info={ {k: v for k, v in original.info.items() if k != 'icc_profile'} }")

        variants = {
            "api": load_image(path.read_bytes()),
            "rgb_direto": original.convert("RGB"),  # what a naive conversion produces
        }
        for name, image in variants.items():
            debug_path = path.with_name(f"debug_{path.stem}_{name}.png")
            image.save(debug_path)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.inference_mode():
                beams = model.generate(pixel_values, num_beams=2)
                greedy = model.generate(pixel_values, num_beams=1)
            text_beams = processor.batch_decode(beams, skip_special_tokens=True)[0]
            text_greedy = processor.batch_decode(greedy, skip_special_tokens=True)[0]
            print(f"  {name:13s} beams={text_beams!r:20s} greedy={text_greedy!r:20s} ids={beams[0].tolist()[:12]}")
        print()


if __name__ == "__main__":
    main()
