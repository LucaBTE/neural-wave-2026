# Submission, format, timing, checklist

## 1. The layout (it is fixed)

```
submissions/<team>/<garment_id>/front.<png|jpg|jpeg|webp>
```

Example:

```
submissions/identity/d001/front.png
submissions/identity/d002/front.png
submissions/identity/g01/front.png
```

* `<team>` is your team name, one folder per team. Keep it filesystem-safe: no spaces, use `-` or
  `_`.
* `<garment_id>` is the id from the manifest: `d001...` for dev, `g01...` for the on-site
  garments. It must match exactly. An id that is not in the manifest is a validation error: the
  validator rejects it, so fix the id before you hand in. A typo is not a missing slot; it is a
  validation failure. A garment you never submit scores 0.
* `<view>` is the view name. The required view is `front`; the references we score against do not
  publish any other view. Only `front` is scored unless the organisers announce another view at
  kickoff.
* One image per garment. Extensions `.png`, `.jpg`, `.jpeg`, `.webp`. If you leave two files with
  the same stem and different extensions, the scorer reads the first by extension order
  `png < jpg < jpeg < webp`, so do not leave a stale `.jpg` next to a new `.png`.

dev and test use the same layout, and their ids never collide, so one team folder can hold both.

## 2. Image requirements

| property | requirement | if violated |
|---|---|---|
| readable | a real decodable image | `invalid`, score 0 |
| short side | >= 256 px | `invalid`, score 0 |
| long side | <= 12000 px | `invalid`, score 0 |
| file size | <= 100 MB | `invalid`, score 0 |
| frames | a single still image (no animated GIF / WebP / APNG) | `invalid`, score 0 |
| file type | a real file, not a symlink | `invalid`, score 0 |
| folder type | real team and garment folders, not symlinks | `invalid`, score 0 |
| flat field | a real photo, not a near-uniform image (whole-image luma standard deviation >= 4/255) | `invalid`, score 0 |
| aspect ratio | <= 3:1 | `invalid`, score 0 |
| size | <= 150 MPix | `invalid` (decompression bomb guard) |
| format | PNG / JPEG / WebP (HEIC is decoded if `pillow-heif` is installed, but do not rely on it) | not read |

Notes that matter in practice:

* PNG. A heavy JPEG re-encode is measurable, pointless loss. Submit a lossless PNG unless your
  pipeline only emits JPEG.
* Orientation. The scorer applies EXIF orientation and ICC to sRGB, but you should bake both in
  yourself, so what you inspect is what is scored.
* Alpha. Do not submit transparency. It is composited onto pure white and anything under
  `alpha = 0` is discarded.
* Do not upscale a small output. The canvas is rescaled anyway, and you only add blur that LPIPS
  will find.

## 3. Missing files

* A missing view scores 0 for that view and stays in the garment average.
* A missing garment scores 0 for that garment and stays in the team average, equal weight per
  garment.
* An unknown id is not a missing garment. It is a validation error: `make_submission.py` exits
  non-zero until the id matches the manifest.

An imperfect guess is almost always better than an empty slot. Submit something for every garment
you photographed.

## 4. Validate before you hand in

```bash
.venv/bin/python scripts/make_submission.py --submissions submissions --team identity
```

This checks the layout, matches your ids against the dev and test manifests, and sanity-checks
every image. It exits non-zero on errors, so you can gate your pipeline on it. Point it at a
different dataset root with `--dataset <root>` if yours is not at the default
`../sartiq_neural_wave_2026`.

To see why a garment scored what it did, run the scorer's `debug` on a pair you can score. Make a
holdout and a prediction first (the quickstart in the top-level `README.md` does the same):

```bash
.venv/bin/python scripts/make_holdout.py \
    --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout/
.venv/bin/python baselines/baseline_identity.py \
    --manifest holdout/val.jsonl --out predictions --team identity \
    --dataset ../sartiq_neural_wave_2026
```

Then pick any pair id from `holdout/val.jsonl` and render its contact sheet:

```bash
PAIR=$(.venv/bin/python -c "import json,pathlib; print(json.loads(pathlib.Path('holdout/val.jsonl').read_text().splitlines()[0])['pair_id'])")
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli debug \
    --ref ../sartiq_neural_wave_2026/train/$PAIR/target.jpg \
    --sub predictions/identity/$PAIR/front.jpg --out sheet_$PAIR.jpg
```

## 5. Timing and hand-in

* The organisers score dev at regular intervals between Friday 16 October 19:30 and Sunday 18 October 10:00 CEST and post team totals. The official on-site
  ranking runs after Sunday 18 October 2026, 11:00 CEST.
* Hand in the submissions folder (or a single archive of it) through the channel the organisers announce at the event,
  together with your two written deliverables: the pipeline and the capture-protocol write-up.
* Anything under `predictions/`, `val_out/` or `holdout/` is your business, not part of the
  deliverable. Do not hand in the dataset.

## 6. Final checklist

- [ ] one folder per team, one folder per garment, spelled exactly like the manifest
- [ ] every id is in the dev or test manifest (no unknown ids), no symlinked folders or files
- [ ] every required view present for every garment you captured
- [ ] PNG/JPG/WebP, single frame, real file, <= 100 MB, >= 256 px short side,
      <= 12000 px long side, aspect <= 3:1, not a flat field (luma standard deviation >= 4/255)
- [ ] no duplicate stems across extensions, no leftover files from earlier runs
- [ ] `make_submission.py` reports 0 errors
- [ ] you have looked at a `debug` contact sheet at least once
- [ ] the capture-protocol write-up is reproducible from a cold start by someone else
