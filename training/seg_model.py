"""Compact U-Net with a timm encoder (no external segmentation library)."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Encoder from timm `features_only`, symmetric decoder with skips."""

    def __init__(self, arch="resnet34", n_classes=1, pretrained=True,
                 decoder_ch=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = timm.create_model(arch, pretrained=pretrained,
                                         features_only=True)
        chs = self.encoder.feature_info.channels()          # low -> high res order
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = chs[-1]
        for i, ch in enumerate(decoder_ch):
            skip = chs[-(i + 2)] if i + 2 <= len(chs) else 0
            self.up.append(nn.ConvTranspose2d(cin, ch, 2, stride=2))
            self.dec.append(ConvBlock(ch + skip, ch))
            cin = ch
        self.head = nn.Conv2d(cin, n_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        y = feats[-1]
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            y = up(y)
            j = len(feats) - (i + 2)
            if j >= 0:
                skip = feats[j]
                if skip.shape[-2:] != y.shape[-2:]:
                    y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
                y = torch.cat([y, skip], 1)
            y = dec(y)
        y = self.head(y)
        if y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return y


def dice_bce_loss(logit, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return bce + (1 - num / den).mean()
