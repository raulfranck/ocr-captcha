import io
import logging
import threading

import numpy as np
import torch
from PIL import Image, ImageFilter, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    pass


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_8bit(image: Image.Image) -> Image.Image:
    """Scale high-bit-depth grayscale (16-bit PNGs load as I;16) down to 8-bit.

    Pillow's convert() clips these values at 255 instead of scaling them,
    which turns a 16-bit captcha into a blank white image.
    """
    if not (image.mode.startswith("I") or image.mode == "F"):
        return image
    arr = np.asarray(image, dtype=np.float64)
    if image.mode.startswith("I;16") or arr.max() > 255:
        arr = arr / 257.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")


def load_image(data: bytes) -> Image.Image:
    """Decode bytes and flatten any transparency onto white, as the model card does."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Arquivo não é uma imagem válida.") from exc
    rgba = _to_8bit(image).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


PREPROCESS_MODES = ("none", "median", "median_bin", "median_bold")
BASE_PROCESSOR = "microsoft/trocr-base-printed"


def preprocess(image: Image.Image, mode: str) -> Image.Image:
    """Optional cleanup before the model sees the image.

    none: as decoded. median: 3x3 median filter, removes salt-and-pepper dots.
    median_bin: median, then threshold to pure black and white.
    median_bold: median, then thicken the strokes (3x3 min filter), closer to bold fonts.
    """
    if mode == "none":
        return image
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"PREPROCESS inválido: {mode!r}. Use um de {PREPROCESS_MODES}.")
    gray = image.convert("L").filter(ImageFilter.MedianFilter(3))
    if mode == "median_bin":
        gray = gray.point(lambda v: 0 if v < 140 else 255)
    elif mode == "median_bold":
        gray = gray.filter(ImageFilter.MinFilter(3))
    return gray.convert("RGB")


class CaptchaOCR:
    def __init__(self, model_path: str, device: str = "auto", num_beams: int = 2, preprocess: str = "none"):
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"PREPROCESS inválido: {preprocess!r}. Use um de {PREPROCESS_MODES}.")
        self.device = _pick_device(device)
        self.num_beams = num_beams
        self.preprocess = preprocess
        logger.info("Carregando modelo %s em %s", model_path, self.device)
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path)
        except (OSError, ValueError, TypeError):
            # Some TrOCR fine-tunes ship only the weights; their tokenizer is the base model's.
            logger.info("%s não traz o processor; usando o de %s", model_path, BASE_PROCESSOR)
            self.processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        # One generation at a time: parallel generate calls only fight for the same cores/GPU.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, images: list[Image.Image]) -> list[str]:
        images = [preprocess(image, self.preprocess) for image in images]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with self._lock:
            generated_ids = self.model.generate(pixel_values, num_beams=self.num_beams)
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [text.strip() for text in texts]
