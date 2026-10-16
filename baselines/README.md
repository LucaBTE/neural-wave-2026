# Baselines

Two reference points, in increasing order of sophistication. Both read a manifest and write the
same submission layout, so you can swap them for your own pipeline without touching anything else.

| baseline | what it does | why it exists |
|---|---|---|
| `baseline_identity.py` | copies the input image through as the prediction | the floor. It proves your harness works and gives the number a real pipeline has to beat. |
| `baseline_nanobanana.py` | calls a hosted image-edit model with a flat-lay prompt | the "just prompt a big model" approach the challenge suggests. |

The identity baseline copies the input through unchanged, so it scores whatever a bare phone capture
scores. The hosted-model baseline sends the input to an existing hosted edit model with a flat-lay
prompt, with no capture protocol and no per-garment tuning, so it scores whatever a single prompt
produces. Custom training is optional. The organisers solved the challenge with existing models and
no custom training; improving on the hosted baseline means better prompts, model choice and
post-processing.

Both take the same arguments: `--manifest <jsonl> --out <dir> --team <name> [--dataset <root>]`.
They work on `train/pairs.jsonl` (using `pair_id` and the first input) and on the dev and test
manifests (using `garment_id` and the first input), and write `<out>/<team>/<id>/front.<ext>`.
`--dataset` defaults to the manifest's dataset root, the folder that contains `train/`, `dev/`
and `test/`.

## Identity baseline, the floor

```bash
.venv/bin/python baselines/baseline_identity.py --manifest ../sartiq_neural_wave_2026/train/pairs.jsonl --out predictions --team identity
```

Bytes are copied unchanged, so there is no re-encode loss. Point it at `dev/inputs.jsonl` with
`--out submissions` to build a dev submission, or at a holdout manifest to get a number you can
score. Carve the holdout first (the quickstart in the top-level [README.md](../README.md), or
[docs/DATA.md](../docs/DATA.md)):

```bash
.venv/bin/python scripts/make_holdout.py --pairs ../sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout/ --group-by-target
.venv/bin/python scripts/eval_val.py --pairs holdout/val.jsonl --preds predictions --team identity --out val_out --jobs 1
```

On native Windows, run `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

Some of its slots come back `status=review`, usually `truncated` (the raw capture cuts the garment
at the frame edge) or `clip_low`. That is the scorer working as intended on unprocessed captures.

## Hosted image-edit baseline

Provider-agnostic. The API key comes from the environment, never from this repo, and a `.env` file
is not read automatically. The default provider is fal.

Check the inputs first: `--dry-run` writes the request payloads to `submissions/_requests/` and
makes no network call, so it needs no key.

```bash
.venv/bin/python baselines/baseline_nanobanana.py --manifest ../sartiq_neural_wave_2026/dev/inputs.jsonl --out submissions --team hosted-edit --dry-run
```

To spend, set the key and drop `--dry-run`. Bash / zsh / macOS / WSL2:

```bash
export FAL_KEY=...                 # the key you were given at kickoff
.venv/bin/python baselines/baseline_nanobanana.py --manifest ../sartiq_neural_wave_2026/dev/inputs.jsonl --out submissions --team hosted-edit
```

PowerShell (native Windows):

```powershell
$env:FAL_KEY = "..."
.venv\Scripts\python.exe baselines/baseline_nanobanana.py --manifest ../sartiq_neural_wave_2026/dev/inputs.jsonl --out submissions --team hosted-edit
```

On fal the default model is `google/nano-banana-2.1/edit`, so no `--model` is needed. An edit costs
about $0.06, so a run over every dev garment is roughly $2. The key and its budget are for challenge
work only; the top-level [README.md](../README.md) has the details.

Options that matter:

| flag | meaning |
|---|---|
| `--provider fal` | default; fal queue REST API (submit, poll until the edit completes, download) |
| `--provider gemini` | Google `generateContent` with the image as inline data; set `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `--provider openai` | OpenAI-compatible image edits endpoint, base URL from `OPENAI_BASE_URL`; set `OPENAI_API_KEY` |
| `--model ID` | provider model id; defaults to `google/nano-banana-2.1/edit` for fal, required for gemini and openai |
| `--prompt` | the edit instruction; the default asks for a clean flat-lay packshot on a light background |
| `--limit N` | generate at most N images (one image per row; a retried row still counts as one image, but each retry is a separate request, so total API calls can exceed N). Do this the first time, you are on a shared budget. |
| `--dry-run` | write the request payloads to `<out>/_requests/` and make no network call. Use it to check inputs and see the prompt before spending anything. |
| `--timeout`, `--retries` | request timeout and retry count for transient 429/5xx; for fal the timeout also bounds the wait on a queued request |

Output is saved as PNG under `<out>/<team>/<garment_id>/front.png`, with a sibling
`<out>/_manifest.jsonl` recording model, prompt, latency and any error per call. Failures are
written to the manifest and skipped, so one bad API response does not kill a batch.

## Reference scores on dev

The organisers scored both baselines on the hidden dev set. They publish the number only.

| baseline | dev score |
|---|---|
| identity, the raw input unchanged | 48.37 |
| one-shot hosted image edit, single prompt, no pre- or post-processing | 68.71 |

The table reports both baselines on the same dev set. Beating the hosted number means beating a
single prompt. The identity number shows how much the raw capture already leaks through the metric.
