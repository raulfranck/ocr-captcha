"""Baixa o modelo do Hugging Face para ./models/ocr-captcha-v3 (uma vez só, ~1,3 GB).

Baixa cada arquivo pelo nome, sem usar a API de listagem do repositório (que tem
limite baixo para acesso anônimo e responde 429). Em caso de 429, espera e tenta
de novo. Com HF_TOKEN definido (ou após `hf auth login`), o limite é bem maior.
"""
import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

REPO_ID = "anuashok/ocr-captcha-v3"
FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]
MAX_ATTEMPTS = 6


def _retry_after(exc: HfHubHTTPError, attempt: int) -> float:
    header = exc.response.headers.get("Retry-After") if exc.response is not None else None
    if header and header.isdigit():
        return float(header)
    return min(15 * 2**attempt, 300)


def download(repo: str, filename: str, out: Path, token: str | None) -> None:
    for attempt in range(MAX_ATTEMPTS):
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=out, token=token)
            return
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 429 or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = _retry_after(exc, attempt)
            print(f"  429 (limite de requisições). Tentando de novo em {wait:.0f}s...")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--out", default=str(Path("models") / "ocr-captcha-v3"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None  # None lets huggingface_hub use a saved `hf auth login` token

    for filename in FILES:
        target = out / filename
        if target.is_file() and target.stat().st_size > 0:
            print(f"[ok] {filename} (já existe)")
            continue
        print(f"[..] {filename}")
        try:
            download(args.repo, filename, out, token)
        except HfHubHTTPError as exc:
            sys.exit(
                f"\nFalha ao baixar {filename}: {exc}\n"
                "Espere alguns minutos e rode de novo (os arquivos já baixados são mantidos),\n"
                "ou crie um token em https://huggingface.co/settings/tokens e defina HF_TOKEN."
            )
        print(f"[ok] {filename}")

    print(f"\nModelo salvo em: {out.resolve()}")


if __name__ == "__main__":
    main()
