"""Treina uma CRNN pequena (CNN + LSTM bidirecional + CTC) nos captchas sintéticos.

É a mesma família do leitor do EasyOCR, mas com ~1 milhão de parâmetros: treina
na CPU em minutos e lê um captcha em poucos milissegundos. As imagens de treino
são geradas na hora por scripts/synth_captcha.py (nunca se repetem); --real soma
captchas reais rotulados ao treino. A pasta --val é medida a cada época e o
melhor modelo vai para --out (um arquivo .pt).

    python scripts/train_crnn.py --val captchas/rotulados --out models/crnn.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ocr import load_image  # noqa: E402
from scripts.evaluate import IMAGE_SUFFIXES, edit_distance, label_of  # noqa: E402
from scripts.synth_captcha import ALPHABET, generate  # noqa: E402

HEIGHT, WIDTH = 64, 192


def to_tensor(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
    return torch.from_numpy(np.asarray(gray, dtype=np.float32) / 127.5 - 1.0)[None]


class CRNN(nn.Module):
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


def decode(log_probs: torch.Tensor) -> list[str]:
    texts = []
    for seq in log_probs.argmax(-1).permute(1, 0).tolist():
        chars, prev = [], 0
        for t in seq:
            if t != prev and t != 0:
                chars.append(ALPHABET[t - 1])
            prev = t
        texts.append("".join(chars))
    return texts


class SynthStream(torch.utils.data.IterableDataset):
    def __init__(self, seed: int, real: list[tuple[torch.Tensor, str]], real_ratio: float):
        self.seed, self.real, self.real_ratio = seed, real, real_ratio

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 100_003)
        while True:
            if self.real and rng.random() < self.real_ratio:
                yield rng.choice(self.real)
                continue
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 6)))
            yield to_tensor(generate(text, rng)), text


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.tensor([ALPHABET.index(c) + 1 for _, t in batch for c in t])
    lengths = torch.tensor([len(t) for _, t in batch])
    return images, targets, lengths, [t for _, t in batch]


def labeled(folder: Path) -> list[tuple[torch.Tensor, str]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [(to_tensor(load_image(p.read_bytes())), label_of(p).lower()) for p in paths]


@torch.inference_mode()
def evaluate(model, items):
    model.eval()
    preds = decode(model(torch.stack([x for x, _ in items])))
    labels = [t for _, t in items]
    exact = sum(p == t for p, t in zip(preds, labels))
    cer = sum(edit_distance(p, t) for p, t in zip(preds, labels)) / sum(len(t) for t in labels)
    pairs = [f"{t}->{p or '(vazio)'}{'' if p == t else '*'}" for p, t in zip(preds, labels)]
    return exact, cer, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--real", type=Path, help="captchas reais rotulados para somar ao treino")
    parser.add_argument("--real-ratio", type=float, default=0.2, help="fração de cada lote vinda de --real")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300, help="passos por época")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    val = labeled(args.val)
    real = labeled(args.real) if args.real else []
    model = CRNN()
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
            torch.save({"state_dict": model.state_dict(), "alphabet": ALPHABET, "size": (HEIGHT, WIDTH)}, args.out)
    print(f"\nMelhor: acertos={best[0]}/{len(val)} CER={-best[1]:.1%}, salvo em {args.out}")


if __name__ == "__main__":
    main()
