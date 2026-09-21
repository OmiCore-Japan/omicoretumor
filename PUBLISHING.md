# Publishing this repository to GitHub

The code is ready to push as-is. The only extra step is the model weights,
which are too large for git.

## 1. Create the repository

```bash
cd omicoretumor-package
git init
git add .
git commit -m "OmiCoreTumor v0.1.0 - CRC tumour-region mapping from H&E"
git branch -M main
git remote add origin https://github.com/<YOUR-ORG>/omicoretumordetector.git
git push -u origin main
```

`.gitignore` already excludes `*.pt`, so the 384 MB in `release_assets/` is not
committed. Verify before pushing:

```bash
git status --porcelain | grep '\.pt$' && echo "STOP: weights staged" || echo "ok: no weights staged"
```

## 2. Attach the weights as a release

GitHub allows 2 GB per release asset, so all four files fit comfortably.

**Using the `gh` CLI:**

```bash
gh release create v0.1.0 \
  release_assets/omicoretumor-crc-he-convnext-norm-v0.1.pt \
  release_assets/omicoretumor-crc-he-convnext-nonorm-v0.1.pt \
  release_assets/omicoretumor-crc-he-effnetv2-nonorm-v0.1.pt \
  release_assets/omicoretumor-crc-he-seg-unet-v0.1.pt \
  release_assets/SHA256SUMS.txt \
  --title "omicoretumor-crc-he-v0.1" \
  --notes "Model weights for omicoretumor-crc-he-v0.1. Research use only."
```

**Or in the browser:** Releases → Draft a new release → tag `v0.1.0` → drag the
five files from `release_assets/` into the attachments box → Publish.

The asset filenames must stay exactly as they are — `weights.py` builds its
download URLs from them, and verifies each file against the SHA-256 recorded in
the package.

## 3. Point the package at your repository

If your GitHub org is not `OmiCore`, update the default URL in
`omicoretumor/weights.py`:

```python
BASE = os.environ.get(
    "OMICORETUMOR_WEIGHTS_URL",
    "https://github.com/<YOUR-ORG>/omicoretumordetector/releases/download/v0.1.0",
)
```

and the repository links in `pyproject.toml`, `README.md` and `CITATION.cff`.

## 4. Verify it works for a stranger

From a clean machine or fresh virtual environment:

```bash
python -m venv /tmp/t && source /tmp/t/bin/activate
pip install "git+https://github.com/<YOUR-ORG>/omicoretumordetector.git"
python -c "
from omicoretumor import TumorDetector
det = TumorDetector.from_pretrained()   # must download and verify
print('OK', det.classes)
"
```

If that succeeds, anyone can use the package.

## Before you publish — a checklist

- [ ] Weights attached to the release and downloadable
- [ ] `<YOUR-ORG>` replaced everywhere
- [ ] `pytest tests/` passes
- [ ] README limitations section left intact (it is the honest part)
- [ ] `NOTICE` kept — CC-BY-4.0 attribution is a licence obligation, not a courtesy

---

# Publishing to PyPI (optional)

`pip install omicoretumor` instead of a git URL. The package metadata already
passes `twine check`, and the name `omicoretumor` was free as of this writing.

## Do the GitHub release FIRST

**Order matters.** The package downloads weights from your GitHub release. If you
publish to PyPI before that release exists, every `pip install omicoretumor` user
gets a package that 404s the moment they load a model.

PyPI versions are **immutable** — you cannot re-upload `0.1.0` after fixing it.
You can only yank it and publish `0.1.1`. So get the release right first.

PyPI does **not** remove the need for GitHub releases: it caps projects at
100 MB per file too, so the ~400 MB of weights live on GitHub either way.

## Steps

```bash
# 1. Accounts (once): register on pypi.org AND test.pypi.org, enable 2FA,
#    create an API token on each. Tokens look like pypi-AgEIcHlwaS5vcmc...

pip install --upgrade build twine

# 2. Build clean
rm -rf dist build *.egg-info
python -m build --wheel --sdist
python -m twine check dist/*          # must PASS

# 3. Rehearse on TestPyPI first -- this is the step people skip and regret
python -m twine upload --repository testpypi dist/*

python -m venv /tmp/tp && source /tmp/tp/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ omicoretumor
python -c "from omicoretumor import TumorDetector; print('ok')"
deactivate

# 4. Publish for real
python -m twine upload dist/*
```

Use `__token__` as the username and the API token as the password, or put it in
`~/.pypirc`.

## Releasing a new version later

Bump `version` in `pyproject.toml` **and** `omicoretumor/__init__.py`, rebuild,
re-upload. If the weights change, publish a new GitHub release, update `BASE` and
the SHA-256 values in `weights.py`, and bump the model name (e.g.
`omicoretumor-crc-he-v0.2`) so cached older weights are never silently mixed with
newer code.

## Checklist

- [ ] GitHub release published and weights download successfully
- [ ] `<YOUR-ORG>` replaced in `weights.py`, `pyproject.toml`, `README.md`, `CITATION.cff`
- [ ] `twine check dist/*` passes
- [ ] Installed from TestPyPI in a clean venv and loaded a model
- [ ] Research-use-only wording intact in the README (it becomes the PyPI page)
