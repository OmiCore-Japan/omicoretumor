"""Train a U-Net to segment tumour epithelium on EBHI-SEG.

IMPORTANT about the labels: EBHI-SEG masks delineate glandular/epithelial
structure for EVERY class -- measured mean foreground is 55-70% even for
`Normal`. The mask alone therefore does NOT mean "tumour"; the folder name
carries the diagnosis. We build a tumour target by keeping the mask for
malignant classes and using an empty mask for `Normal`.

Consequence: with only 76 Normal images against 795 Adenocarcinoma, negative
evidence is thin, so this model tends to segment epithelium in general. Use it
to refine boundaries inside regions the tile classifier already flagged, not as
a standalone tumour detector.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from seg_model import UNet, dice_bce_loss

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def norm_key(name: str) -> str:
    """EBHI-SEG has a naming bug: some labels carry a doubled 'GT' prefix."""
    s = Path(name).stem
    return re.sub(r"^(GT)+", "", s)


def index_ebhi(root: Path, positive, negative):
    rows = []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if cls not in positive and cls not in negative:
            continue
        imgs = {norm_key(p.name): p for p in (root / cls / "image").glob("*")}
        labs = {norm_key(p.name): p for p in (root / cls / "label").glob("*")}
        for k in sorted(set(imgs) & set(labs)):
            rows.append({"image": str(imgs[k]), "label": str(labs[k]),
                         "cls": cls, "is_pos": int(cls in positive)})
        missing = set(imgs) ^ set(labs)
        if missing:
            print(f"  {cls}: {len(missing)} unmatched file(s) skipped")
    return rows


class EBHIDataset(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB").resize((224, 224), Image.BILINEAR)
        m = Image.open(r["label"]).convert("L").resize((224, 224), Image.NEAREST)
        img = np.asarray(img).astype(np.float32) / 255.0
        # Masks are anti-aliased greyscale, so threshold rather than assume 0/1.
        msk = (np.asarray(m) > 127).astype(np.float32)
        if not r["is_pos"]:
            msk = np.zeros_like(msk)          # normal glands are not tumour
        if self.train:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
            img = np.clip(img * np.random.uniform(0.85, 1.15)
                          + np.random.uniform(-0.06, 0.06), 0, 1)
        img = (img - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
                torch.from_numpy(np.ascontiguousarray(msk))[None].float())


@torch.inference_mode()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    inter = union = tp = fp = fn = 0.0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
        pr = (p > thr).float()
        inter += (pr * y).sum().item()
        union += ((pr + y) > 0).float().sum().item()
        tp += (pr * y).sum().item()
        fp += (pr * (1 - y)).sum().item()
        fn += ((1 - pr) * y).sum().item()
    return {"iou": inter / max(union, 1), "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/processed/EBHI-SEG")
    ap.add_argument("--positive", default="Adenocarcinoma,High-grade IN")
    ap.add_argument("--negative", default="Normal")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="../models/seg_ebhi")
    args = ap.parse_args()

    pos = [c.strip() for c in args.positive.split(",") if c.strip()]
    neg = [c.strip() for c in args.negative.split(",") if c.strip()]
    rows = index_ebhi(Path(args.root), pos, neg)
    if not rows:
        raise SystemExit("no image/label pairs found")
    n_pos = sum(r["is_pos"] for r in rows)
    print(f"{len(rows)} pairs: {n_pos} positive ({pos}), {len(rows)-n_pos} negative ({neg})")

    strat = [r["cls"] for r in rows]
    tr, va = train_test_split(rows, test_size=0.25, random_state=42, stratify=strat)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    dl_tr = DataLoader(EBHIDataset(tr, True), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(EBHIDataset(va, False), batch_size=args.batch_size,
                       num_workers=args.workers, pin_memory=True)

    model = UNet(args.arch, n_classes=1, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr), pct_start=0.3)

    best, hist = -1.0, []
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y in tqdm(dl_tr, desc=f"epoch {ep+1}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = dice_bce_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        m = evaluate(model, dl_va, device)
        m.update(epoch=ep + 1, train_loss=tot / n)
        hist.append(m)
        print(f"  epoch {ep+1}: loss={m['train_loss']:.4f} IoU={m['iou']:.4f} "
              f"Dice={m['dice']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}")
        if m["iou"] > best:
            best = m["iou"]
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "positive": pos, "negative": neg, "val": m}, out / "best.pt")
            print(f"  -> saved best (IoU={best:.4f})")
    (out / "history.json").write_text(json.dumps({"args": vars(args), "history": hist}, indent=2))
    print(f"\nbest val IoU {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
