# Installing on Windows and macOS

## Windows

1. Install Python 3.10 or 3.11 from python.org (tick "Add Python to PATH").
2. Open PowerShell:

```powershell
python -m venv omicore-env
omicore-env\Scripts\activate
pip install "git+https://github.com/OmiCore-Japan/omicoretumor.git"
```

3. If you have an NVIDIA GPU:

```powershell
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

## macOS

Apple Silicon runs on CPU here; MPS is not used because the model relies on
bfloat16 autocast paths that are CUDA-specific. Expect several minutes per
whole section.

```bash
python3 -m venv omicore-env
source omicore-env/bin/activate
pip install "git+https://github.com/OmiCore-Japan/omicoretumor.git"
```

## Checking the install

```bash
omicoretumor --version
omicoretumor models
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

## Reading common slide formats

`.svs`, `.ndpi`, `.mrxs` and other whole-slide formats are not read directly.
Export a region as TIFF/PNG first, or convert with OpenSlide:

```python
import openslide, numpy as np
from PIL import Image
from omicoretumor import TumorDetector

slide = openslide.OpenSlide("slide.svs")
mpp = float(slide.properties["openslide.mpp-x"])     # the value to pass
region = slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB")

det = TumorDetector.from_pretrained()
res = det.predict(region, mpp=mpp)
```

`predict()` accepts a PIL image directly, so no intermediate file is needed.
