
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


## Layout

```
submissions/btc/<garment_id>/front.png
```

- dev: `d001`–`d027`, test: `g01`–`g15`. Both in the same team folder
- A missing garment scores 0, so submit all 42


## Submitting

Run from `neural-wave-2026/` after setup

1. Run the pipeline on every dev & test garment.

```bash
for row in $(.venv/bin/python -c "import json;[print(r['garment_id']+','+r['inputs'][0]) for r in map(json.loads, open('sartiq_neural_wave_2026/dev/inputs.jsonl'))]"); do
    id=${row%%,*}; img=sartiq_neural_wave_2026/${row#*,}
    dev/btc-sartiq/backend/.venv/bin/python dev/btc-sartiq/scripts/validate_studio.py "$img" || {
  echo "FAIL $id"; continue; }
    mkdir -p submissions/btc/$id
    cp dev/btc-sartiq/validation/${id}_$(basename "${img%.*}")_studio.png submissions/btc/$id/front.png
  done

for row in $(.venv/bin/python -c "import json;[print(r['garment_id']+','+r['inputs'][0]) for r in map(json.loads, open('sartiq_neural_wave_2026/test/inputs.jsonl'))]"); do
    id=${row%%,*}; img=sartiq_neural_wave_2026/${row#*,}
    dev/btc-sartiq/backend/.venv/bin/python dev/btc-sartiq/scripts/validate_studio.py "$img" || {
  echo "FAIL $id"; continue; }
    mkdir -p submissions/btc/$id
    cp dev/btc-sartiq/validation/${id}_$(basename "${img%.*}")_studio.png submissions/btc/$id/front.png
  done
```

5 fal calls per garment, about 210 for all 42

2. Validate and zip:

```bash
.venv/bin/python scripts/make_submission.py --submissions submissions --team btc --dataset sartiq_neural_wave_2026 --zip
```

Must report 0 errors. Writes `btc.zip`

3. Hand in, through the channel the organisers announce:
    1. `btc.zip`
    2. the pipeline source, with dependencies and run instructions
    3. the capture-protocol write-up
