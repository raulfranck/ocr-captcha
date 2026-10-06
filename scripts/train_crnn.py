"""Treina uma CRNN pequena (CNN + LSTM bidirecional + CTC) nos captchas sintéticos.

É a mesma família do leitor do EasyOCR, mas com ~1 milhão de parâmetros: treina
na CPU em minutos e lê um captcha em poucos milissegundos. As imagens de treino
são geradas na hora por scripts/synth_captcha.py (nunca se repetem); --real soma
captchas reais rotulados ao treino. --holdout separa algumas dessas imagens
(sempre as mesmas para o mesmo --seed) para medir cada época, e o melhor modelo
vai para --out (um arquivo .pt).

    python scripts/train_crnn.py --real captchas/rotulados --holdout 10 --init models/crnn.pt --out models/crnn.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ocr import ALPHABET, CRNN, decode, load_image, save_model, to_tensor  # noqa: E402
from scripts.evaluate import IMAGE_SUFFIXES, edit_distance, label_of  # noqa: E402
from scripts.synth_captcha import generate  # noqa: E402


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    """Small shift, rotation and extra dots, so each real captcha is seen slightly differently."""
    gray = image.convert("L")
    gray = gray.rotate(rng.uniform(-3, 3), resample=Image.BILINEAR,
                       translate=(rng.randint(-4, 4), rng.randint(-3, 3)), fillcolor=251)
    pixels = gray.load()
    for _ in range(rng.randint(0, 300)):
        pixels[rng.randrange(gray.width), rng.randrange(gray.height)] = rng.choice((4, 72, 200))
    return gray


class SynthStream(torch.utils.data.IterableDataset):
    def __init__(self, seed: int, real: list[tuple[Image.Image, str]], real_ratio: float):
        self.seed, self.real, self.real_ratio = seed, real, real_ratio

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 100_003)
        while True:
            if self.real and rng.random() < self.real_ratio:
                image, text = rng.choice(self.real)
                yield to_tensor(augment(image, rng)), text
                continue
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 6)))
            yield to_tensor(generate(text, rng, height=rng.choice((68, 70)))), text


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.tensor([ALPHABET.index(c) + 1 for _, t in batch for c in t])
    lengths = torch.tensor([len(t) for _, t in batch])
    return images, targets, lengths, [t for _, t in batch]


def labeled(paths: list[Path]) -> list[tuple[Image.Image, str]]:
    return [(load_image(p.read_bytes()), label_of(p).lower()) for p in paths]


def image_paths(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


@torch.inference_mode()
def evaluate(model, items):
    model.eval()
    preds = decode(model(torch.stack([to_tensor(image) for image, _ in items])))
    labels = [t for _, t in items]
    exact = sum(p == t for p, t in zip(preds, labels))
    cer = sum(edit_distance(p, t) for p, t in zip(preds, labels)) / sum(len(t) for t in labels)
    pairs = [f"{t}->{p or '(vazio)'}{'' if p == t else '*'}" for p, t in zip(preds, labels)]
    return exact, cer, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real", type=Path, help="captchas reais rotulados para somar ao treino")
    parser.add_argument("--holdout", type=int, default=0, help="imagens de --real separadas para validação")
    parser.add_argument("--val", type=Path, help="pasta de validação (alternativa a --holdout)")
    parser.add_argument("--init", type=Path, help="continua de um modelo salvo por este script")
    parser.add_argument("--real-ratio", type=float, default=0.3, help="fração de cada lote vinda de --real")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300, help="passos por época")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="use 1e-3 ao treinar do zero")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    real_paths = image_paths(args.real) if args.real else []
    if args.holdout:
        held = set(random.Random(args.seed).sample(range(len(real_paths)), args.holdout))
        val_paths = [p for i, p in enumerate(real_paths) if i in held]
        real_paths = [p for i, p in enumerate(real_paths) if i not in held]
    elif args.val:
        val_paths = image_paths(args.val)
    else:
        parser.error("informe --holdout (com --real) ou --val")
    val, real = labeled(val_paths), labeled(real_paths)
    model = CRNN()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"])
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parâmetros, {len(val)} imagens de validação, "
          f"{len(real)} reais no treino")
    loader = torch.utils.data.DataLoader(SynthStream(args.seed, real, args.real_ratio), batch_size=args.batch,
                                         num_workers=args.workers, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * args.steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best = (-1, 0.0)
    batches = iter(loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.perf_counter(), 0.0
        for _ in range(args.steps):
            images, targets, lengths, _ = next(batches)
            log_probs = model(images)
            steps_in = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, steps_in, lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        exact, cer, pairs = evaluate(model, val)
        print(f"época {epoch:2d} loss={running / args.steps:.3f} val acertos={exact}/{len(val)} CER={cer:.1%} "
              f"({time.perf_counter() - start:.0f} s)\n   " + "  ".join(pairs), flush=True)
        if (exact, -cer) > best:
            best = (exact, -cer)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, args.out)
    print(f"\nMelhor: acertos={best[0]}/{len(val)} CER={-best[1]:.1%}, salvo em {args.out}")


if __name__ == "__main__":
    main()
