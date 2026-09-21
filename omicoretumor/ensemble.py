"""Average several tile classifiers into one drop-in nn.Module.

Members must share a class list; probabilities (not logits) are averaged so
models trained with different losses/temperatures combine sensibly.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbEnsemble(nn.Module):
    def __init__(self, checkpoints, device="cuda", weights=None):
        super().__init__()
        self.models = nn.ModuleList()
        self.classes = None
        self.size = None
        for ck_path in checkpoints:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            if self.classes is None:
                self.classes, self.size = ck["classes"], ck["size"]
            elif list(ck["classes"]) != list(self.classes):
                raise ValueError(f"class mismatch in {ck_path}: "
                                 f"{ck['classes']} vs {self.classes}")
            m = timm.create_model(ck["arch"], pretrained=False,
                                  num_classes=len(ck["classes"]))
            m.load_state_dict(ck["model"])
            self.models.append(m.to(device, memory_format=torch.channels_last).eval())
        n = len(self.models)
        w = torch.ones(n) if weights is None else torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("w", (w / w.sum()).to(device))

    def forward(self, x):
        p = sum(wi * F.softmax(m(x).float(), 1)
                for wi, m in zip(self.w, self.models))
        # Downstream code applies softmax to our output, so return log-probs:
        # softmax(log p) == p, keeping the ensemble a drop-in replacement.
        return torch.log(p.clamp_min(1e-8))
