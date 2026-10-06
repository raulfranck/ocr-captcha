import io
import logging
import threading

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

logger = logging.getLogger(__name__)

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
HEIGHT, WIDTH = 64, 192


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
    """Decode bytes and flatten any transparency onto white."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Arquivo não é uma imagem válida.") from exc
    rgba = _to_8bit(image).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
    return torch.from_numpy(np.asarray(gray, dtype=np.float32) / 127.5 - 1.0)[None]


class CRNN(nn.Module):
    """CNN + 2-layer bidirectional LSTM + CTC head, ~1M parameters."""

    def __init__(self, classes: int = len(ALPHABET) + 1):
        super().__init__()

        def block(cin, cout, pool):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool)]

        self.cnn = nn.Sequential(
            *block(1, 32, 2), *block(32, 64, 2), *block(64, 128, (2, 1)), *block(128, 128, (2, 1)),
            nn.Conv2d(128, 128, (4, 1)), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # height 4 -> 1
        )
        self.rnn = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Linear(256, classes)

    def forward(self, x):  # (B, 1, 64, 192) -> (T=48, B, classes) log-probs
        features = self.cnn(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.rnn(features)
        return self.head(out).log_softmax(-1).permute(1, 0, 2)


def decode(log_probs: torch.Tensor, alphabet: str = ALPHABET) -> list[str]:
    """Greedy CTC decoding: collapse repeats, drop blanks (index 0)."""
    texts = []
    for seq in log_probs.argmax(-1).permute(1, 0).tolist():
        chars, prev = [], 0
        for t in seq:
            if t != prev and t != 0:
                chars.append(alphabet[t - 1])
            prev = t
        texts.append("".join(chars))
    return texts


def save_model(model: CRNN, path) -> None:
    torch.save({"state_dict": model.state_dict(), "alphabet": ALPHABET, "size": (HEIGHT, WIDTH)}, path)


class CaptchaOCR:
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = _pick_device(device)
        logger.info("Carregando modelo %s em %s", model_path, self.device)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.alphabet = checkpoint.get("alphabet", ALPHABET)
        self.model = CRNN(len(self.alphabet) + 1)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        # One inference at a time: parallel calls only fight for the same cores/GPU.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, images: list[Image.Image]) -> list[str]:
        batch = torch.stack([to_tensor(image) for image in images]).to(self.device)
        with self._lock:
            log_probs = self.model(batch)
        return decode(log_probs.cpu(), self.alphabet)
