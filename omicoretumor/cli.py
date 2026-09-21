"""Command line interface: omicoretumor <command>."""
import argparse
import json
import sys
import pathlib
from pathlib import Path


def _unique_stems(paths):
    """Map each input path to a filesystem-safe, collision-free output stem.

    Distinct sections frequently share a basename -- Visium HD writes every
    sample to <sample>/segmentation/he_mpp0.5.tiff -- so when stems collide we
    prepend parent directories to ALL members of the colliding group until they
    differ, which keeps the naming symmetric and informative.
    """
    stems = {str(p): pathlib.PurePath(p).stem for p in paths}
    parents = {str(p): list(pathlib.Path(p).resolve().parts[:-1]) for p in paths}
    depth = {str(p): 0 for p in paths}

    for _ in range(12):
        groups = {}
        for k, v in stems.items():
            groups.setdefault(v, []).append(k)
        clashing = [g for g in groups.values() if len(g) > 1]
        if not clashing:
            break
        for group in clashing:
            if not any(len(parents[k]) > depth[k] for k in group):
                break  # identical paths; nothing left to disambiguate with
            for k in group:
                d = depth[k]
                if len(parents[k]) > d:
                    depth[k] = d + 1
                    stems[k] = f"{parents[k][-(d + 1)]}_{stems[k]}"

    # de-duplicate anything still identical, and keep names filesystem-safe
    seen, out = set(), {}
    for k, v in stems.items():
        v = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in v)
        base, i = v, 2
        while v in seen:
            v = f"{base}_{i}"
            i += 1
        seen.add(v)
        out[k] = v
    return out


def _predict(a):
    from .api import TumorDetector

    det = TumorDetector.from_pretrained(
        a.model, stain_norm=not a.no_stain_norm, tta=a.tta,
        allow_auto_threshold=(a.threshold == "auto"))
    thr = a.threshold if a.threshold == "auto" else float(a.threshold)
    out = Path(a.out or ".")
    out.mkdir(parents=True, exist_ok=True)

    # Distinct inputs often share a basename -- Visium HD writes every section
    # to <sample>/segmentation/he_mpp0.5.tiff -- so disambiguate with parent
    # directories rather than silently overwriting earlier results.
    stems = _unique_stems(a.images)

    rows = []
    for img in a.images:
        stem = stems[str(img)]
        res = det.predict(img, mpp=a.mpp, stride=a.stride, batch_size=a.batch_size)
        n = det.to_geojson(res, out / f"{stem}_tumor.geojson", threshold=thr,
                           min_area_px=a.min_area)
        t = res["threshold_used"]
        area = float(((res["p_tumor"] >= t) & res["tissue"]).sum()
                     / max(res["tissue"].sum(), 1))
        rec = {"image": str(img), "threshold": t, "tumor_area": area,
               "tissue_tiles": res["n_tissue_tiles"], "regions": n,
               "composition": det.tissue_composition(res)}
        rows.append(rec)
        print(f"{stem}: tumour area {area:.1%}  regions {n}  "
              f"tiles {res['n_tissue_tiles']:,}  threshold {t:.3f}")
        if a.save_heatmap:
            import numpy as np
            np.savez_compressed(out / f"{stem}_heatmap.npz",
                                prob=res["prob"].astype("float16"),
                                tissue=res["tissue"].astype("uint8"),
                                classes=np.array(res["classes"]),
                                xs=res["xs"], ys=res["ys"],
                                tile=res["tile"], scale=res["scale"])
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}/")


def _models(a):
    from .weights import CHECKPOINTS, ENSEMBLES, cache_dir
    print("ensembles:")
    for k, v in ENSEMBLES.items():
        print(f"  {k}  ->  {len(v)} models")
    print("checkpoints:")
    for k, (f, _) in CHECKPOINTS.items():
        cached = (cache_dir() / f).exists()
        print(f"  {k:32s} {'[cached]' if cached else ''}")
    print(f"\ncache: {cache_dir()}")


def _clear(a):
    from .weights import clear_cache
    print(f"freed {clear_cache()/1e6:.0f} MB")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="omicoretumor",
        description="Tumour-region mapping for colorectal H&E. RESEARCH USE ONLY.")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("predict", help="map tumour regions on one or more images")
    r.add_argument("images", nargs="+")
    r.add_argument("--mpp", type=float, required=True,
                   help="microns per pixel of the input (models expect 0.5; "
                        "other values are rescaled)")
    r.add_argument("--model", default=None,
                   help="model name (default: omicoretumor-crc-he-v0.1)")
    r.add_argument("--threshold", default="0.4",
                   help="float, or 'auto' (only for slides known to be tumour)")
    r.add_argument("--stride", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=256)
    r.add_argument("--min-area", type=int, default=50_000)
    r.add_argument("--no-stain-norm", action="store_true")
    r.add_argument("--tta", action="store_true")
    r.add_argument("--save-heatmap", action="store_true")
    r.add_argument("-o", "--out", default="omicoretumor_out")
    r.set_defaults(func=_predict)

    sub.add_parser("models", help="list available and cached weights").set_defaults(func=_models)
    sub.add_parser("clear-cache", help="delete downloaded weights").set_defaults(func=_clear)

    a = p.parse_args(argv)
    if a.version:
        from . import __version__
        print(__version__)
        return 0
    if not getattr(a, "func", None):
        p.print_help()
        return 1
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
