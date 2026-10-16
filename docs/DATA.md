# Data, layout, splits, manifests

The dataset is not in this repository. Download it from https://drive.google.com/drive/folders/1INEK1AxsB9Tt79mabm8ElxNAZLjWwXL3?usp=sharing as the archive file named
`sartiq_neural_wave_2026.zip` and unpack it next to the repo, so the dataset root sits as a sibling.
Download that named file: Google Drive also offers a zip of the surrounding folder, which unpacks to
the wrong layout.

```
<parent>/
├── sartiq_neural_wave_2026/     the data (this is the dataset root)
└── neural-wave-2026/            this repository (you may have cloned it under any name)
```

Put the downloaded `sartiq_neural_wave_2026.zip` in `<parent>` (next to the clone). From the
repository root, unpack it there:

Bash / zsh / macOS / WSL2:

```bash
unzip -q ../sartiq_neural_wave_2026.zip -d ..
```

PowerShell (native Windows) — Windows has no `unzip`, so use the interpreter you created in the
quickstart:

```powershell
.venv\Scripts\python.exe -m zipfile -e "..\sartiq_neural_wave_2026.zip" ".."
```

Either way the data lands at `../sartiq_neural_wave_2026/`. Every path inside a manifest is relative
to the dataset root.

---

## 1. Layout

```
sartiq_neural_wave_2026/
├── README.md
├── train/
│   ├── pairs.jsonl                          one line per pair
│   └── t0001/input_1.jpg [input_2.jpg ...] target.jpg
├── dev/
│   ├── inputs.jsonl                         one line per garment, no target
│   └── d001/input_1.jpg ...
└── test/
    ├── inputs.jsonl
    └── g01/input_1.jpg ...
```

Counts: 382 train pairs, 27 dev garments, 15 test garments.

All images are JPEG with metadata stripped.

---

## 2. Manifest schema

`train/pairs.jsonl`, one JSON object per line. The first line is:

```jsonl
{"pair_id": "t0001", "inputs": ["train/t0001/input_1.jpg"], "target": "train/t0001/target.jpg"}
```

`inputs` is a list of one or more amateur photos of the same garment. The number of photos varies
from row to row: some pairs have a single input, others have up to five. `target` is the studio
flat-lay packshot you produce.

`dev/inputs.jsonl` and `test/inputs.jsonl`, one line per garment. The first line of `dev` is:

```jsonl
{"garment_id": "d001", "view": "front", "inputs": ["dev/d001/input_1.jpg"]}
```

Ids are opaque and assigned after a seeded shuffle: train `t0001...`, dev `d001...`, test
`g01...g15`. Nothing else is in a row. There is no field that tells you where an image came
from or how it was made.

---

## 3. The example pairs (train/)

The example pairs (folder `train/`) show the input to target transformation. They mix studio
pairs, real amateur photos matched to brand packshots, and generated amateur photos. Every target
is a flat-lay packshot on a white background. Use them to design prompts, pick and tune models,
build a holdout, or supply few-shot references, and to fine-tune a model if you want to. Training
a model is one option, not a requirement.

The pool is noisy on purpose. It contains some bad pairs and it repeats garments: several pairs
can share a garment, or even the same target image. Nothing flags which pairs are which.

Exploring the pool is part of the job. Read the manifests, look at the images, and choose the
examples your pipeline uses.

---

## 4. dev, the live leaderboard

dev contains inputs only: 27 garments, no targets. The organisers score dev submissions
at regular intervals between Friday 16 October 19:30 and Sunday 18 October 10:00 CEST and post team totals. Individual garment scores stay hidden, and the dev
references stay private before and after the event.

---

## 5. test, the on-site set

The official ranking is on 15 physical garments we bring to the venue. They are labelled g01 to
g15, and each physical garment carries a tag with its id. `test/inputs.jsonl` lists phone photos
of each garment, so you can see what one looks like before you stand in front of it.

You may use those supplied phone photos as inputs, and you can also photograph the garments
yourself on site with your own protocol (that is encouraged). The test references (the studio
flat-lay we score against) stay private before and after the event.

---

## 6. Self-evaluation: measure your pipeline

There is no public validation split. Carve a holdout out of the example pairs and score pairs you
did not tune on:

```bash
.venv/bin/python scripts/make_holdout.py --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout/ --group-by-target
```

Pass `--group-by-target` so pairs that share the same target file stay on one side of the split.
That writes `holdout/train.jsonl` and `holdout/val.jsonl` (same schema, paths still relative to
the dataset root), then score predictions on `val.jsonl` with `scripts/eval_val.py`. Details and
an example in [docs/SCORING.md](SCORING.md).

One warning that a random split will not save you from: pairs can share a garment. If the same
garment lands in both your holdout and your training slice, your holdout score is optimistic.
Split by target (`--group-by-target`) or by garment where you can.

On Windows, run `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

This is a method hint, not a competition rule. Using the dev or the test inputs for anything
during the event, including unsupervised adaptation, is allowed.

---

## 7. Usage rules

1. Internal hackathon use only. Do not redistribute the dataset publicly or publish subsets or
   derivatives of it. Sending dataset images to a hosted image model or API during the event is
   allowed, on any provider and not only the fal.ai key handed out at kickoff; that is processing,
   and it is what the hosted baseline does.
2. Do not republish the amateur photos. They can contain faces and third-party content.
3. Do not use the data to release a public model without Sartiq's written agreement. The code
   you write is yours (see the IP section of
   [the challenge document](Sartiq_Neural_Wave_2026_Challenge.pdf)); the data is not.
4. You may use the dev and the test inputs for anything during the event, including unsupervised
   adaptation. There is no rule against it.
5. Delete your local copy when the event is over, unless Sartiq says otherwise. Outputs you
   generated are fine to keep.
