"""Fine-tuning do TrOCR (por padrão o ocr-captcha-v3) em captchas rotulados.

Cada pasta em --train tem imagens com o texto certo no nome (`u4ep.png`, `u4ep_2.png`),
como em scripts/evaluate.py. A pasta --val é avaliada no fim de cada época
(acertos e CER) e o melhor checkpoint vai para --out, pronto para MODEL_PATH.

    python scripts/synth_captcha.py dados/sinteticos --count 5000
    python scripts/train.py --train dados/sinteticos --val captchas/rotulados --out models/ocr-captcha-ft

Na CPU cada passo leva alguns segundos; --freeze-encoder treina só o decoder
(bem mais rápido) e costuma bastar quando a fonte do captcha não muda.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ocr import BASE_PROCESSOR, PREPROCESS_MODES, load_image, preprocess  # noqa: E402
from scripts.evaluate import IMAGE_SUFFIXES, edit_distance, label_of, normalize  # noqa: E402


def labeled_images(folders: list[Path]) -> list[tuple[Path, str]]:
    items = []
    for folder in folders:
        items += [(p, label_of(p)) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return items


def load_processor(spec: str) -> TrOCRProcessor:
    try:
        return TrOCRProcessor.from_pretrained(spec)
    except (OSError, ValueError, TypeError):
        return TrOCRProcessor.from_pretrained(BASE_PROCESSOR)


def pixel_values(processor, paths: list[Path], mode: str) -> torch.Tensor:
    images = [preprocess(load_image(p.read_bytes()), mode) for p in paths]
    return processor(images=images, return_tensors="pt").pixel_values


@torch.inference_mode()
def evaluate(model, processor, items, mode: str, device: str, batch: int) -> tuple[int, float, list[str]]:
    model.eval()
    preds = []
    for i in range(0, len(items), batch):
        values = pixel_values(processor, [p for p, _ in items[i:i + batch]], mode).to(device)
        ids = model.generate(values, num_beams=1, max_new_tokens=12)
        preds += [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
    labels = [t for _, t in items]
    exact = sum(normalize(p) == normalize(t) for p, t in zip(preds, labels))
    errors = sum(edit_distance(normalize(p), normalize(t)) for p, t in zip(preds, labels))
    cer = errors / max(1, sum(len(normalize(t)) for t in labels))
    pairs = [f"{t}->{p or '(vazio)'}{'' if normalize(p) == normalize(t) else '*'}" for p, t in zip(preds, labels)]
    return exact, cer, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, nargs="+", required=True, help="pastas com imagens rotuladas")
    parser.add_argument("--val", type=Path, required=True, help="pasta rotulada para medir cada época")
    parser.add_argument("--out", type=Path, required=True, help="onde salvar o melhor modelo")
    parser.add_argument("--model", default="./models/ocr-captcha-v3", help="modelo de partida")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--preprocess", default="none", choices=PREPROCESS_MODES)
    parser.add_argument("--freeze-encoder", action="store_true", help="treina só o decoder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_items = labeled_images(args.train)
    val_items = labeled_images([args.val])
    if not train_items or not val_items:
        sys.exit("Pasta de treino ou de validação sem imagens.")
    print(f"{len(train_items)} imagens de treino, {len(val_items)} de validação, device={args.device}")

    processor = load_processor(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(args.device)
    model.config.decoder_start_token_id = model.config.decoder_start_token_id or processor.tokenizer.cls_token_id
    model.config.pad_token_id = model.config.pad_token_id or processor.tokenizer.pad_token_id
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    exact, cer, pairs = evaluate(model, processor, val_items, args.preprocess, args.device, args.batch)
    print(f"antes do treino: acertos={exact}/{len(val_items)} CER={cer:.1%}\n   " + "  ".join(pairs), flush=True)
    best = (exact, -cer)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    pad_id = processor.tokenizer.pad_token_id
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_items)
        start, running = time.perf_counter(), 0.0
        steps = range(0, len(train_items), args.batch)
        for step, i in enumerate(steps, 1):
            chunk = train_items[i:i + args.batch]
            values = pixel_values(processor, [p for p, _ in chunk], args.preprocess).to(args.device)
            labels = processor.tokenizer([t for _, t in chunk], padding=True, return_tensors="pt").input_ids
            labels[labels == pad_id] = -100
            loss = model(pixel_values=values, labels=labels.to(args.device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 20 == 0 or step == len(steps):
                print(f"época {epoch} passo {step}/{len(steps)} loss={running / step:.3f} "
                      f"({(time.perf_counter() - start) / step:.1f} s/passo)", flush=True)

        exact, cer, pairs = evaluate(model, processor, val_items, args.preprocess, args.device, args.batch)
        print(f"época {epoch}: acertos={exact}/{len(val_items)} CER={cer:.1%}\n   " + "  ".join(pairs), flush=True)
        if (exact, -cer) > best:
            best = (exact, -cer)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"   melhor até agora, salvo em {args.out}")

    print(f"\nMelhor: acertos={best[0]}/{len(val_items)} CER={-best[1]:.1%}. Use MODEL_PATH={args.out}")


if __name__ == "__main__":
    main()
