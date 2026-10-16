# Sartiq Neural Wave 2026, participant starter repo

Build a pipeline that turns an amateur phone photo of a garment into a clean studio flat-lay
packshot. This repo has the official scorer, two baselines, and the scripts to carve a holdout,
self-score and validate a submission.

## At a glance

* **Build:** a pipeline that turns a smartphone photo of a garment into a studio flat-lay packshot
  on a white background. Custom training is optional.
* **Hand in:**
  * one output image per garment, in the layout `submissions/<team>/<garment_id>/front.png`: all
    27 dev ids (for the leaderboard rounds) and all 16 on-site garments `g01`–`g16`;
  * the runnable pipeline source, with the instructions to run it;
  * the capture-protocol write-up.
* **Deadline:** submit by Sunday 18 October 2026, 11:00 CEST.
* **Channel:** hand in through the channel the organisers announce at the event.
* **Resources:** a fal.ai key with a $100 budget and an OpenCode Go key with a $10 budget, per team.
* **Rule:** outputs must come from your own pipeline, using the garment's own photos. Copying,
  recolouring or compositing the example targets or any retrieved product image disqualifies that
  garment. Finalists' outputs are checked against their pipeline.

The challenge runs at Neural Wave, 16 to 18 October 2026, USI Lugano. You own the whole loop:
how the garment is photographed is part of your solution, not a fixed input.

The example pairs in `train/` show the input to target transformation. Use them to design prompts,
pick and tune models, build a holdout, or supply few-shot references, and to fine-tune a model if you
want to. You do not need to train anything: a pipeline built from existing models (a hosted
generation or editing API, a vision model for checks, crops, compositing) is enough, and that is how
the organisers solved it.

You are scored two ways:

* dev, a live leaderboard. dev ships inputs without targets. The organisers score your dev
  submissions at regular intervals between Friday 16 October 19:30 and Sunday 18 October 10:00 CEST and post team totals only.
* test, the official ranking. 16 physical garments we bring on site, labelled g01 to g16. You run
  your pipeline on the phone photos we supply or on your own on-site shots. The scorer ranks the
  teams; the finalists (number announced at the event) go to a Sartiq jury, and teams tied at that cutoff all advance.

The full problem statement is in the challenge document,
[`docs/Sartiq_Neural_Wave_2026_Challenge.pdf`](docs/Sartiq_Neural_Wave_2026_Challenge.pdf).

---

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and `git`. Run the
commands below from the repository root.

```bash
git clone https://github.com/shootify-io/neural-wave-2026
cd neural-wave-2026

# 1. environment (CPU-only; the first score downloads ~1.2 GB of model weights)
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e 'scorer[test]'

# 2. dataset: download https://drive.google.com/drive/folders/1INEK1AxsB9Tt79mabm8ElxNAZLjWwXL3?usp=sharing and save sartiq_neural_wave_2026.zip next to the clone
#    (in its parent directory). From the repository root, unzip it there so the data sits at
#    ../sartiq_neural_wave_2026 (see docs/DATA.md)
unzip ../sartiq_neural_wave_2026.zip -d ..

# 3. carve a holdout out of the example pairs
.venv/bin/python scripts/make_holdout.py \
    --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl \
    --frac 0.1 --seed 0 --out holdout/

# 4. run the identity baseline on the holdout, then score it
.venv/bin/python baselines/baseline_identity.py \
    --manifest holdout/val.jsonl --out predictions --team identity \
    --dataset ../sartiq_neural_wave_2026
env -u PYTHONPATH .venv/bin/python scripts/eval_val.py \
    --pairs holdout/val.jsonl --preds predictions --team identity --out val_out

# 5. run the same baseline on dev and check the submission layout
.venv/bin/python baselines/baseline_identity.py \
    --manifest ../sartiq_neural_wave_2026/dev/inputs.jsonl --out submissions --team identity
.venv/bin/python scripts/make_submission.py --submissions submissions --team identity
```

Step 4 writes `val_out/leaderboard.csv`, the scorer's breakdown for your holdout. A pair costs
roughly 3 seconds on a laptop CPU, so use `--jobs 4` on anything large.

To score one image pair directly:

```bash
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli pair \
    --ref ../sartiq_neural_wave_2026/train/t0001/target.jpg \
    --sub ../sartiq_neural_wave_2026/train/t0001/input_1.jpg
```

---

## What you deliver

1. Your pipeline source, runnable on the 16 on-site garments, with the instructions to run it.
2. A short capture-protocol write-up: how a garment must be shot for your system to work, so
   someone else can reproduce it from a cold start.
3. Your output images, one per garment, in the fixed layout
   `submissions/<team>/<garment_id>/front.<png|jpg|jpeg|webp>`
   ([docs/SUBMISSION.md](docs/SUBMISSION.md)). Submit them for every dev id during the event and
   for all 16 on-site garments `g01`–`g16` at judging. A garment you do not submit scores 0.

Ranking rules are in [docs/SCORING.md](docs/SCORING.md). Shooting hints for the on-site set are in
[docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md). They are hints; the protocol is yours to design.

---

## What you get

At kickoff each team is handed two API keys, both for challenge work only:

* a fal.ai key with a $100 budget, for hosted image generation and editing. Set it in the
  environment as `FAL_KEY`.
* an OpenCode Go key with a $10 budget, for the OpenCode coding agent.

The budgets are per team, not per person, and they are not topped up unless staff say so. Ask a
Sartiq staff member first if you are unsure what you have left. Keys go in environment variables
only: never in the repo, never in a submission, never in your git history. `.env` is listed in
`.gitignore`; keep it out of your commits.

The organisers suggest [OpenCode](https://opencode.ai) as the coding harness. It is an agent that
runs in your terminal and edits code in your repo, and it is not required. To install it and
connect the key:

```bash
curl -fsSL https://opencode.ai/install | bash   # installs to ~/.opencode/bin
opencode auth login                             # connect the OpenCode Go key when prompted
opencode                                        # start the agent in this folder
```

`opencode auth list` shows what is connected. You can use any other models, harnesses or tools you
like, including your own keys and subscriptions, at your own cost. Scoring does not depend on which
provider or harness you pick.

---

## Repo map

* [README.md](README.md) — this file
* [docs/DATA.md](docs/DATA.md) — dataset layout, the example pairs (train/), dev, test, usage rules
* [docs/SCORING.md](docs/SCORING.md) — the exact scoring math, status rules, how to self-score
* [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md) — practical hints for the on-site capture
* [docs/SUBMISSION.md](docs/SUBMISSION.md) — submission format, timing, what happens if a file is missing
* [docs/FAQ.md](docs/FAQ.md) — short answers to the questions everyone asks
* [docs/Sartiq_Neural_Wave_2026_Challenge.pdf](docs/Sartiq_Neural_Wave_2026_Challenge.pdf) — the
  challenge document (full problem statement)
* [baselines/](baselines/) — identity + hosted image model, as reference points
* [scripts/make_holdout.py](scripts/make_holdout.py) — split a holdout out of the example pairs
* [scripts/eval_val.py](scripts/eval_val.py) — score predictions against a holdout with the official scorer
* [scripts/make_submission.py](scripts/make_submission.py) — validate a submission layout (+ image sanity)
* [scorer/](scorer/) — the official scorer (frozen); CLI + library

---

## The score in one paragraph

Your packshot and the Sartiq reference are both segmented (rembg `u2net`), cropped to the
garment and uniformly scaled onto the same 512x384 canvas, which reduces the effect of framing.
Backgrounds and shadows still reach the segmenter, so a background it handles badly changes the
mask. What matters is colour, material appearance, silhouette and semantics. The score is a weighted
similarity: colour 40%, LPIPS 25%, CLIP ViT-B-32 15%, silhouette IoU 15%, SSIM 5%, reported
0 to 100 and called reference similarity. It is a gate that seeds the ranking and picks
finalists. The Sartiq jury then judges the finalists on commercial usability and has the final
word. The weights are frozen before submissions open and are never fitted to submitted work.

---

## Rules of engagement

* The dataset is for internal hackathon use only. Do not redistribute it publicly, publish subsets or
  derivatives of it, or release a public model trained on it without Sartiq's written agreement.
  Sending dataset images to a hosted image model or API during the event is allowed, on any provider;
  it is processing, not distribution, and it is what the hosted baseline does.
* Submit only work your own pipeline produced from the provided inputs or your own on-site photos.
  Retrieving product images of a garment (web search, brand sites, reverse image search) and
  submitting or compositing them disqualifies that garment. Outputs must be generated by your
  pipeline from the garment's own photos; copying, recolouring or compositing the supplied example
  targets or any retrieved product image is not allowed and disqualifies that garment, and
  finalists' outputs are checked against their pipeline.
* Bring your own smartphone. The on-site set (backdrop and one controllable light) is shared.
* The budget is the two keys handed out at kickoff (see "What you get"): a fal.ai key for hosted
  image models and an OpenCode Go key for the coding agent. Both are per team and for challenge work
  only.
* The scorer, its weights and its anchors are frozen. Tune against the real garment, not against
  the metric. The dev and test references are private before and after the event, and probing for
  them is not allowed.

---

## Contacts

* Felipe Cardoso (primary): felipe.cardoso@sartiq.com
* Luca Pasini (secondary): luca.pasini@sartiq.com
* Sartiq: https://sartiq.com

## Licence

The code in this repository and the dataset are provided under the hackathon terms of service
(available from the Neural Wave organisers). Read them before reusing either outside the event.
