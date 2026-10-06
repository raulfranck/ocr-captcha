import base64
import binascii
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.ocr import CaptchaOCR, InvalidImageError, load_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_BATCH = 16


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    if not hasattr(app.state, "ocr"):  # tests inject a fake before startup
        app.state.ocr = CaptchaOCR(
            settings.model_path,
            device=settings.device,
            num_beams=settings.num_beams,
            preprocess=settings.preprocess,
        )
    yield


app = FastAPI(title="OCR Captcha API", version="1.0.0", lifespan=lifespan)


class Base64Request(BaseModel):
    image: str  # base64 puro ou data URL (data:image/png;base64,...)


class OCRResponse(BaseModel):
    text: str
    elapsed_ms: float


class BatchOCRResponse(BaseModel):
    texts: list[str]
    elapsed_ms: float


def get_ocr(request: Request) -> CaptchaOCR:
    return request.app.state.ocr


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(
    settings: Settings = Depends(get_app_settings),
    x_api_key: str | None = Header(default=None),
) -> None:
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="X-API-Key ausente ou inválida.")


def _decode(data: bytes, settings: Settings):
    if not data:
        raise HTTPException(status_code=400, detail="Imagem vazia.")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail=f"Imagem maior que {settings.max_image_bytes} bytes.")
    try:
        return load_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health(request: Request):
    ocr = request.app.state.ocr
    return {"status": "ok", "device": getattr(ocr, "device", "unknown")}


# Endpoints are plain `def` so FastAPI runs the CPU-bound inference in its threadpool.
@app.post("/ocr", response_model=OCRResponse, dependencies=[Depends(require_api_key)])
def ocr_upload(
    file: UploadFile = File(..., description="Imagem do captcha (PNG, JPG, GIF...)"),
    ocr: CaptchaOCR = Depends(get_ocr),
    settings: Settings = Depends(get_app_settings),
):
    start = time.perf_counter()
    image = _decode(file.file.read(settings.max_image_bytes + 1), settings)
    text = ocr.predict([image])[0]
    return OCRResponse(text=text, elapsed_ms=round((time.perf_counter() - start) * 1000, 1))


@app.post("/ocr/base64", response_model=OCRResponse, dependencies=[Depends(require_api_key)])
def ocr_base64(
    body: Base64Request,
    ocr: CaptchaOCR = Depends(get_ocr),
    settings: Settings = Depends(get_app_settings),
):
    start = time.perf_counter()
    payload = body.image.split(",", 1)[1] if body.image.startswith("data:") else body.image
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Base64 inválido.") from exc
    image = _decode(data, settings)
    text = ocr.predict([image])[0]
    return OCRResponse(text=text, elapsed_ms=round((time.perf_counter() - start) * 1000, 1))


@app.post("/ocr/batch", response_model=BatchOCRResponse, dependencies=[Depends(require_api_key)])
def ocr_batch(
    files: list[UploadFile] = File(..., description=f"Até {MAX_BATCH} imagens"),
    ocr: CaptchaOCR = Depends(get_ocr),
    settings: Settings = Depends(get_app_settings),
):
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Envie no máximo {MAX_BATCH} imagens por chamada.")
    start = time.perf_counter()
    images = [_decode(f.file.read(settings.max_image_bytes + 1), settings) for f in files]
    texts = ocr.predict(images)
    return BatchOCRResponse(texts=texts, elapsed_ms=round((time.perf_counter() - start) * 1000, 1))
