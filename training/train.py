"""Train a tissue-type classifier on CRC H&E tiles."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from data import CLASSES, class_weights, make_loader


def evaluate(model, loader, device, criterion=None):
    model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                if criterion is not None:
                    loss_sum += criterion(out, y).item() * y.size(0)
            logits_all.append(out.float().cpu())
            y_all.append(y.cpu())
            n += y.size(0)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    pred = logits.argmax(1).numpy()
    return {
        "loss": loss_sum / max(n, 1),
        "acc": float((pred == y).mean()),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }, logits, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--target", default="multiclass", choices=["multiclass", "binary"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="models/run1")
    ap.add_argument("--limit-steps", type=int, default=0, help="smoke-test only")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = Path(args.manifests)

    n_classes = 2 if args.target == "binary" else len(CLASSES)
    names = ["non-tumor", "tumor"] if args.target == "binary" else CLASSES

    train_loader = make_loader(man / "train.csv", True, args.batch_size,
                               args.size, args.workers, args.target)
    val_loader = make_loader(man / "val.csv", False, args.batch_size,
                             args.size, args.workers, args.target)

    model = timm.create_model(args.arch, pretrained=True, num_classes=n_classes)
    model = model.to(device, memory_format=torch.channels_last)

    w = class_weights(man / "train.csv", args.target).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = args.limit_steps or len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.25
    )

    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for i, (x, y) in enumerate(bar):
            if args.limit_steps and i >= args.limit_steps:
                break
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            bar.set_postfix(loss=f"{run_loss/seen:.4f}",
                            lr=f"{sched.get_last_lr()[0]:.2e}")

        metrics, _, _ = evaluate(model, val_loader, device, criterion)
        metrics.update(epoch=epoch + 1, train_loss=run_loss / max(seen, 1),
                       secs=round(time.time() - t0, 1))
        history.append(metrics)
        print(f"  epoch {epoch+1}: val_acc={metrics['acc']:.4f} "
              f"bal_acc={metrics['balanced_acc']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} ({metrics['secs']}s)")

        if metrics["balanced_acc"] > best:
            best = metrics["balanced_acc"]
            torch.save(
                {"model": model.state_dict(), "arch": args.arch,
                 "classes": names, "target": args.target,
                 "size": args.size, "epoch": epoch + 1, "val": metrics},
                out / "best.pt",
            )
            print(f"  -> saved new best (balanced_acc={best:.4f})")

    (out / "history.json").write_text(json.dumps(
        {"args": vars(args), "history": history}, indent=2))
    print(f"\nbest val balanced accuracy: {best:.4f}\nartifacts in {out}")


if __name__ == "__main__":
    main()
