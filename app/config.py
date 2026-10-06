import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    model_path: str
    device: str
    max_image_bytes: int
    api_key: str


def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        model_path=os.getenv("MODEL_PATH") or "./models/crnn.pt",
        device=os.getenv("DEVICE", "auto"),
        max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", "2000000")),
        api_key=os.getenv("API_KEY", ""),
    )
