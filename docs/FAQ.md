# FAQ

## Where is the dataset?
https://drive.google.com/drive/folders/1INEK1AxsB9Tt79mabm8ElxNAZLjWwXL3?usp=sharing. It is not in this repo. Unzip it next to the repo so it sits at
`../sartiq_neural_wave_2026`. Details in `docs/DATA.md`.

## What exactly am I scored on?
Two things. dev drives a live leaderboard during the event, and the 16 on-site garments drive the
official ranking. Not the example pairs.

## How does the leaderboard work?
You submit predictions for the dev ids (`d001...`). The organisers score them at
regular intervals between Friday 16 October 19:30 and Sunday 18 October 10:00 CEST and post team totals. You do not see per-garment dev scores or dev targets.

## Is the automated score the final ranking?
No. It seeds the ranking and picks the finalists (number announced at the event). The Sartiq jury then scores
those on commercial usability, and the bonus criteria (how well the protocol generalises,
efficiency, automation) ride on top. Details in `docs/SCORING.md` section 7.

## What is in the example pairs (train/)?
The example pairs show the input to target transformation. The pool mixes studio pairs, real
amateur photos matched to brand packshots, and generated amateur photos, and every target is a
flat-lay packshot on white. It is noisy: it has some bad pairs and it repeats garments. Nothing in
the data labels which is which, and we do not say how many of each. Looking at it and choosing your
examples is part of the work.

## Do I have to train a model?
No. A strong solution is a pipeline built on existing models: a hosted generation or editing API, a
vision model for checks, crops and compositing. That is how the organisers solved it, with no
custom training. Training is one option among many, and the example pairs work just as well for
prompt design, model choice, few-shot references and a holdout.

## Can I train on my holdout, or on dev?
Your holdout is a method, not a rule: it measures your pipeline on pairs you did not tune on.
Tuning on it makes your own number optimistic, so it is a bad idea, but nothing forbids it. Using
the dev or the test inputs for anything during the event, including unsupervised adaptation, is
allowed. dev has no targets.

## How do I measure my pipeline?
Carve a holdout out of the example pairs with `scripts/make_holdout.py` and score the pairs you did
not tune on with `scripts/eval_val.py`. An example is in `docs/SCORING.md` section 6.

## My holdout score feels too high. Why?
Pairs can share a garment. If the same garment lands in your tuning slice and your holdout, the
holdout score is optimistic. Split by garment where you can.

## Does framing, cropping or background matter?
Less than you would expect, but not nothing. Both images are segmented, cropped to the garment's
mask and uniformly rescaled onto the same canvas before any metric runs, so framing and crop have
little effect. What matters is colour, material appearance, silhouette and semantics. The
background and any shadow are inputs to segmentation, so a background that confuses the segmenter
changes the mask and costs points. A clipped garment is still an error, because the mask bbox is
what gets fitted.

## Can I use a ghost mannequin style?
You can, and it will cost you: every reference is flat-lay, and silhouette IoU (15%), LPIPS (25%)
and SSIM (5%) all compare shape like for like. If you photograph in a way that is not flat-lay
(hanger, dress form), that is a legitimate capture choice, but your model still has to produce a
flat-lay.

## Can I just call a hosted image model?
Yes, that is one of the suggested approaches, and a hosted image-edit call with a good prompt is a
single-call baseline (see `baselines/`). Sending dataset images to a hosted image model or API
during the event is allowed. The budget is the fal.ai key handed out at kickoff; check with Sartiq
staff before you spend beyond it.

## How much budget do I get?
Each team gets a fal.ai key with a $100 budget and an OpenCode Go key with a $10 budget, both for
challenge work only, handed out at kickoff. A hosted image edit costs about $0.06. The budgets are
per team and are not topped up unless staff say so; ask a staff member if you are unsure.

## Do I have to use OpenCode?
No. The organisers suggest OpenCode (https://opencode.ai) as the coding harness and hand out an
OpenCode Go key for it, but any editor, agent or toolchain works. The README's "What you get"
section has the install lines. If you would rather not use an agent, ignore the key.

## Can I use other models or tools?
Yes. Call any image or vision model you like, bring your own API keys and subscriptions, and run
local models. Keys you bring are at your own cost. Nothing in the scoring depends on which provider
or harness you use.

## Can I fine-tune or train a LoRA on the provided garments?
Yes, the code you write is yours, and it is optional. Remember that a holdout you tune on is no
longer an honest number, and that a model tuned to a few garments is not a capture protocol.

## Why did my score come back review?
Something in the flags: `mask_empty`, `mask_suspect`, `truncated`, `multi_component` on your
submission, or a `clip` sub-score below 0.30. It is not a penalty and not a rejection. The score is
kept and a human checks it. The usual cause is a garment that silhouettes badly (white, cream or
lace on a light background) or a clipped edge.

## My output looks great but scores badly. Is the scorer broken?
Probably not. The scorer measures similarity to a specific reference photograph. A cleaner shot
with a different pose, drape or colour grade is a different image, which is why the jury exists.
Run the scorer's `debug` sub-command on the pair and inspect the contact sheet. The first example
pair is always available and scores without any setup:

```bash
env -u PYTHONPATH .venv/bin/python -m sartiq_score.cli debug \
    --ref ../sartiq_neural_wave_2026/train/t0001/target.jpg \
    --sub ../sartiq_neural_wave_2026/train/t0001/input_1.jpg --out sheet_t0001.jpg
```

## How long does scoring take?
About 3.3 s per pair warm on CPU, with a one-off ~10 s startup and a ~1.2 GB model download on
first use. Use `--jobs 4`.

## What if my pipeline fails on one garment?
Submit your best output anyway. A missing view or garment scores 0 and stays in the average; a
wrong guess usually scores better than nothing.

## Can I see the 16 references?
No. The test references stay private before and after the event. `test/inputs.jsonl` points at
phone photos we took of each garment, so you know what it looks like before you are standing in
front of it. You may use those phone photos as inputs; they are not the reference.

## Do I need a GUI?
No. Phone app, phone to desktop hand-off, or a form are all fine. What is judged is the procedure
and the resulting image.

## Is my code mine?
Yes, per the challenge document's IP section. Sartiq reserves the right to use and incorporate it
commercially. The dataset is not yours: keep it internal and delete it afterwards.

## Who do I ask?
Felipe Cardoso (felipe.cardoso@sartiq.com) and Luca Pasini (luca.pasini@sartiq.com), or
corner a Sartiq person at the venue.
