
## Root venv setup

```bash
uv venv --python 3.12
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e 'scorer[test]'
```


## Backend venv setup

```bash
cd dev/btc-sartiq/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Remember to add fal key to `.env`


## Scoring

Run from `neural-wave-2026/`

1. Make the holdout, one time:

```bash
.venv/bin/python scripts/make_holdout.py --pairs sartiq_neural_wave_2026/train/pairs.jsonl --frac 0.1 --seed 0 --out holdout/ --group-by-target
```

2. Run the pipeline on each val pair and copy the result into the scorer layout. Start small: `head -3` does 3 pairs

```bash
for id in $(.venv/bin/python -c "import json;[print(json.loads(l)['pair_id']) for l in open('holdout/val.jsonl')]" | head -3); do
    dev/btc-sartiq/backend/.venv/bin/python dev/btc-sartiq/scripts/validate_studio.py sartiq_neural_wave_2026/train/$id/input_1.jpg || {
  echo "FAIL $id"; continue; }
    mkdir -p predictions/btc/$id
    cp dev/btc-sartiq/validation/${id}_input_1_studio.png predictions/btc/$id/front.png
  done
```

Generated pics are moved from `dev/btc-sartiq/validation/${id}_input_1_studio.png` to `predictions/btc/$id/front.png`

3. Score:

```bash
.venv/bin/python scripts/eval_val.py --pairs holdout/val.jsonl --preds predictions --team btc --out val_out --jobs 1 --limit 3
```

Read val_out/leaderboard.csv. Each score is out of 100, and higher is better. Drop --limit 3 once you run all the pairs
