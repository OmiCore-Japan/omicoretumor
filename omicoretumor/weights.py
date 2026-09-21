"""Resolve model weights, downloading release assets on first use.

Checkpoints are far larger than GitHub's 100 MB per-file git limit, so they
ship as release assets rather than in the repository.
"""
import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm

BASE = os.environ.get(
    "OMICORETUMOR_WEIGHTS_URL",
    "https://github.com/OmiCore-Japan/omicoretumor/releases/download/v0.1.0",
)

#: name -> (filename, sha256). sha256 None disables verification.
CHECKPOINTS = {
    "omicoretumor-crc-he-convnext-norm-v0.1":   ("omicoretumor-crc-he-convnext-norm-v0.1.pt", "e04e8a1404d1a15b2d9bf0130d63c64ba48d01c343221382772f0a4ef7e99f41"),
    "omicoretumor-crc-he-convnext-nonorm-v0.1": ("omicoretumor-crc-he-convnext-nonorm-v0.1.pt", "0e05236c00b662eb807f87548ebecae029a511dbf65fe148046ad87c487e8da5"),
    "omicoretumor-crc-he-effnetv2-nonorm-v0.1": ("omicoretumor-crc-he-effnetv2-nonorm-v0.1.pt", "3fb935faa7b519995cec8a343abfa220f470c2d41e42e84246bbf185964be3b3"),
    "omicoretumor-crc-he-seg-unet-v0.1":        ("omicoretumor-crc-he-seg-unet-v0.1.pt", "330333c59d35b980df1d47a65f12dbd310feed853e4039322eae80f378e9049b"),
}

#: Recommended configuration (see MODEL_CARD.md).
ENSEMBLES = {
    # The released default: 3-model ensemble + Macenko stain normalisation.
    "omicoretumor-crc-he-v0.1": [
        "omicoretumor-crc-he-convnext-norm-v0.1",
        "omicoretumor-crc-he-convnext-nonorm-v0.1",
        "omicoretumor-crc-he-effnetv2-nonorm-v0.1",
    ],
}

#: Default model used when none is given.
DEFAULT_MODEL = "omicoretumor-crc-he-v0.1"


def cache_dir() -> Path:
    d = Path(os.environ.get("OMICORETUMOR_HOME",
                            Path.home() / ".cache" / "omicoretumor"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, progress: bool = True) -> Path:
    """Return a local path to `name`, downloading it if not cached."""
    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; "
                       f"available: {sorted(CHECKPOINTS)}")
    fname, want = CHECKPOINTS[name]
    dest = cache_dir() / fname
    if dest.exists():
        if want and _sha256(dest) != want:
            dest.unlink()
        else:
            return dest

    url = f"{BASE}/{fname}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code == 404:
            raise FileNotFoundError(
                f"{url} returned 404. Publish the weights as release assets, or "
                f"point OMICORETUMOR_WEIGHTS_URL at where they are hosted, or "
                f"pass explicit checkpoint paths to TumorDetector().")
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=fname,
                   disable=not progress)
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    if want and _sha256(tmp) != want:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {fname}")
    tmp.rename(dest)
    return dest


def resolve(name_or_path) -> list:
    """Accept an ensemble name, a checkpoint name, or filesystem path(s)."""
    if isinstance(name_or_path, (list, tuple)):
        return [p for n in name_or_path for p in resolve(n)]
    s = str(name_or_path)
    if s in ENSEMBLES:
        return [fetch(n) for n in ENSEMBLES[s]]
    if s in CHECKPOINTS:
        return [fetch(s)]
    p = Path(s)
    if p.exists():
        return [p]
    raise FileNotFoundError(
        f"{s!r} is not a known model name or an existing file. "
        f"Models: {sorted(ENSEMBLES) + sorted(CHECKPOINTS)}")


def install_local(paths: dict, move: bool = False) -> list:
    """Stage locally trained checkpoints into the cache under their released
    names, so the package works before any release has been published.

    `paths` maps checkpoint name -> local .pt file.
    """
    import shutil

    done = []
    for name, src in paths.items():
        if name not in CHECKPOINTS:
            raise KeyError(f"unknown checkpoint name {name!r}")
        dest = cache_dir() / CHECKPOINTS[name][0]
        (shutil.move if move else shutil.copy2)(str(src), dest)
        done.append(dest)
    return done


def clear_cache() -> int:
    """Delete cached weights; returns bytes freed."""
    freed = 0
    for f in cache_dir().glob("*.pt"):
        freed += f.stat().st_size
        f.unlink()
    return freed
