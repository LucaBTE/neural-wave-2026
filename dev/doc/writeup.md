


## Pipeline

### Setup

Download the `sartiq_neural_wave_2026` dataset folder and place it in the root of the project. Then install backend dependencies with:

```bash
cd dev/btc-sartiq/backend
echo 'FAL_KEY="<key>"' >> .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### CLI (run manually)

The CLI entry point is the `validate_studio.py` script. You give it a list of images:

```bash
dev/btc-sartiq/backend/.venv/bin/python dev/btc-sartiq/scripts/validate_studio.py sartiq_neural_wave_2026/train/t0059/input_1.jpg
```

The result then lands in `dev/btc-sartiq/validation/t0059_input_1_studio.png`

### Frontend

TODO
