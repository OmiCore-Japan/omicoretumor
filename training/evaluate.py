"""Evaluate a trained checkpoint on a manifest (default: external test set)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from data import make_loader
from train import evaluate as run_eval


def plot_confusion(cm, names, path, title):
    cmn = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=names, yticklabels=names, ax=ax,
                cbar_kws={"label": "fraction of true class"})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true_bin, score, path, title):
    fpr, tpr, _ = roc_curve(y_true_bin, score)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/run1/best.pt")
    ap.add_argument("--manifest", default="data/manifests/test.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="external_test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    names = ck["classes"]
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=len(names))
    model.load_state_dict(ck["model"])
    model = model.to(device, memory_format=torch.channels_last)

    loader = make_loader(args.manifest, False, args.batch_size, ck["size"],
                         args.workers, ck["target"])
    metrics, logits, y = run_eval(model, loader, device)
    prob = torch.softmax(logits, 1).numpy()
    pred = prob.argmax(1)

    print(f"\n=== {args.tag} ({len(y):,} tiles) ===")
    print(f"accuracy           {metrics['acc']:.4f}")
    print(f"balanced accuracy  {metrics['balanced_acc']:.4f}")
    print(f"macro F1           {metrics['macro_f1']:.4f}")
    print("\n" + classification_report(y, pred, target_names=names, digits=4))

    cm = confusion_matrix(y, pred, labels=range(len(names)))
    plot_confusion(cm, names, out / f"confusion_{args.tag}.png",
                   f"Confusion matrix -- {args.tag}")

    # Tumor detection framed as a binary screen, whatever the training head was.
    report = {"tag": args.tag, "n": int(len(y)), **metrics}
    if "TUM" in names:
        ti = names.index("TUM")
        y_bin, score = (y == ti).astype(int), prob[:, ti]
    elif len(names) == 2:
        y_bin, score = (y == 1).astype(int), prob[:, 1]
    else:
        y_bin = None
    if y_bin is not None:
        report["tumor_auroc"] = float(roc_auc_score(y_bin, score))
        # sklearn's multi_class="ovr" requires every class to appear in y.
        # Kather-2016 has no MUC/MUS tiles, so compute the macro one-vs-rest
        # AUROC over the classes actually present instead of failing.
        if len(names) > 2:
            present = sorted(set(int(v) for v in y))
            aucs = {}
            for c in present:
                yc = (y == c).astype(int)
                if 0 < yc.sum() < len(yc):
                    aucs[names[c]] = float(roc_auc_score(yc, prob[:, c]))
            if aucs:
                report["tumor_ovr_macro_auroc"] = float(np.mean(list(aucs.values())))
                report["per_class_auroc"] = aucs
                report["classes_present"] = [names[c] for c in present]
                if len(present) < len(names):
                    absent = [n for i, n in enumerate(names) if i not in present]
                    report["classes_absent"] = absent
                    print(f"note: {absent} absent from this set; macro OvR AUROC "
                          f"is over the {len(aucs)} present classes only")
        else:
            report["tumor_ovr_macro_auroc"] = report["tumor_auroc"]
        plot_roc(y_bin, score, out / f"roc_tumor_{args.tag}.png",
                 f"Tumor vs rest -- {args.tag}")
        print(f"tumor AUROC        {report['tumor_auroc']:.4f}")
        if "tumor_ovr_macro_auroc" in report:
            print(f"macro OvR AUROC    {report['tumor_ovr_macro_auroc']:.4f}")

    np.savez_compressed(out / f"preds_{args.tag}.npz",
                        prob=prob, y=y, classes=np.array(names))
    (out / f"metrics_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote reports to {out}/")


if __name__ == "__main__":
    main()
