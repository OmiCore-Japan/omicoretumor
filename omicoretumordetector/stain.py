"""Macenko stain normalisation -- maps a tile onto a fixed H&E stain profile.

Reference: Macenko et al., ISBI 2009. The default target vectors are the
standard reference H&E values used by staintools -- NOT verified to be the
exact values Kather used for NCT-CRC-HE-100K. Treat them as a fixed common
target: what matters is that train and inference tiles are mapped the same
way, which `fit_target_from_tiles` below lets you enforce directly.
"""
import numpy as np

# Target H&E stain matrix (3x2, columns = haematoxylin, eosin in OD space)
TARGET_STAINS = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
TARGET_CONC = np.array([1.9705, 1.0308])


def _od(img, io=240):
    """RGB uint8 -> optical density, flattened to (N,3)."""
    x = img.reshape(-1, 3).astype(np.float32)
    return np.maximum(-np.log10((x + 1.0) / io), 1e-6)


def stain_matrix(img, io=240, beta=0.15, alpha=1.0):
    """Estimate the 3x2 stain matrix of one image."""
    od = _od(img, io)
    od_thin = od[~np.any(od < beta, axis=1)]
    if od_thin.shape[0] < 50:
        return None
    # The stain directions are a population property; a capped random sample
    # estimates them just as well and bounds the cov/percentile cost.
    if od_thin.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(od_thin.shape[0], 20000, replace=False)
        od_thin = od_thin[idx]
    cov = np.cov(od_thin.T)
    _, vecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; columns 1:3 are the two largest, in
    # the (2nd-largest, largest) order the angle convention below assumes.
    # Using (largest, 2nd-largest) instead rotates the plane and makes the
    # alpha/100-alpha percentiles collapse across the arctan2 branch cut,
    # yielding a near-singular stain matrix.
    v = vecs[:, 1:3]
    proj = od_thin @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = v @ np.array([np.cos(lo), np.sin(lo)])
    v2 = v @ np.array([np.cos(hi), np.sin(hi)])
    # Haematoxylin is the stain with the larger red-channel OD.
    hem_first = v1[0] > v2[0]
    m = np.stack([v1, v2] if hem_first else [v2, v1], axis=1)
    return m / np.linalg.norm(m, axis=0, keepdims=True)


def concentrations(img, m, io=240):
    # pinv(m) is 2x3 and m is fixed per tile, so one small matmul beats
    # lstsq solving for ~50k right-hand sides. Same least-squares answer.
    od = _od(img, io)
    return np.linalg.pinv(m) @ od.T           # (2, N)


def macenko_normalize(img, target_m=TARGET_STAINS, target_c=TARGET_CONC,
              io=240, beta=0.15, alpha=1.0):
    """Map one RGB uint8 tile onto the target stain profile.

    Returns the input unchanged when the stain matrix cannot be estimated
    (near-empty or single-stain tiles) rather than emitting garbage.
    """
    m = stain_matrix(img, io, beta, alpha)
    if m is None:
        return img
    c = concentrations(img, m, io)
    max_c = np.percentile(c, 99, axis=1)
    if np.any(max_c <= 0):
        return img
    c = c * (target_c / max_c)[:, None]
    # Deconvolution can return small negative concentrations on noisy tiles;
    # they reconstruct to negative OD, i.e. "brighter than the illuminant",
    # which overflows 10**(-od) to inf and saturates the tile to white.
    # OD is non-negative by definition, so clamp before exponentiating.
    od = np.clip(target_m @ c, 0.0, 10.0)
    out = io * np.power(10.0, -od)
    return np.clip(out.T.reshape(img.shape), 0, 255).astype(np.uint8)


def fit_target_from_tiles(paths, io=240, beta=0.15, alpha=1.0):
    """Estimate a target stain profile from real tiles (e.g. NCT training data).

    Safer than trusting literature constants: it measures the domain we are
    actually mapping onto. Returns (stain_matrix, concentration_p99).
    """
    from PIL import Image

    mats, concs = [], []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        m = stain_matrix(img, io, beta, alpha)
        if m is None:
            continue
        mats.append(m)
        concs.append(np.percentile(concentrations(img, m, io), 99, axis=1))
    if not mats:
        raise ValueError("no tile yielded a stain matrix")
    m = np.median(np.stack(mats), axis=0)
    m /= np.linalg.norm(m, axis=0, keepdims=True)
    return m, np.median(np.stack(concs), axis=0)


# Backwards-compatible alias.
normalize = macenko_normalize
