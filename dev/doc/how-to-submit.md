# Generate the dev submission

Run from the repository root, with the backend installed and its `.env` configured.
These commands generate all 27 dev products, four at a time, overwriting their
previous final PNGs. No local reference score exists for this split.

```bash
screen -S submission-dev
```

Inside screen:

```bash
mkdir -p dev/logs dev/submissions/btc

dev/btc-sartiq/backend/.venv/bin/python -c "import json,shlex; d='/home/luca/Downloads/sartiq_neural_wave_2026/'; [print(shlex.join([r['garment_id'],*[d+p for p in r['inputs']]])) for r in map(json.loads,open(d+'dev/inputs.jsonl'))]" | xargs -r -P 4 -L 1 bash -c '
  id=$0; first=$1
  dev/btc-sartiq/backend/.venv/bin/python -B dev/btc-sartiq/scripts/validate_studio.py --views "$@" > "dev/logs/$id.log" 2>&1 || { echo "FAIL $id"; exit 0; }
  mkdir -p "dev/submissions/btc/$id"
  cp "dev/btc-sartiq/validation/${id}_$(basename "${first%.*}")_studio.png" "dev/submissions/btc/$id/front.png" && echo "OK $id"'
```

- Final images: `dev/submissions/btc/<garment_id>/front.png`.
- Logs: `dev/logs/<garment_id>.log`; details: `dev/btc-sartiq/validation/`.
- Detach: **Ctrl+A**, then **D**. Reattach: `screen -r submission-dev`.

## Package

After generating every split required by the organisers, use their packager:

```bash
dev/.score-venv/bin/python scripts/make_submission.py \
  --submissions dev/submissions --team btc \
  --dataset /home/luca/Downloads/sartiq_neural_wave_2026 --zip
```

Check that validation reports no missing or invalid images. Generating only `dev`
does not supply any separately required `test` products.
