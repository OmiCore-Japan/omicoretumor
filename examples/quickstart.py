"""Map tumour regions on one H&E section and export QuPath annotations."""
import sys

from omicoretumor import TumorDetector

image = sys.argv[1] if len(sys.argv) > 1 else "colon_section.tiff"
mpp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

det = TumorDetector.from_pretrained("omicoretumor-crc-he-v0.1")
res = det.predict(image, mpp=mpp)

print(f"tissue tiles : {res['n_tissue_tiles']:,}")
print(f"tumour area  : {res['tumor_fraction']:.1%}")
print("composition  : " + ", ".join(
    f"{k}={v:.1%}" for k, v in
    sorted(det.tissue_composition(res).items(), key=lambda x: -x[1])[:4]))

n = det.to_geojson(res, "tumor_regions.geojson", threshold=0.4)
print(f"wrote tumor_regions.geojson ({n} region(s)) -- open it in QuPath")
