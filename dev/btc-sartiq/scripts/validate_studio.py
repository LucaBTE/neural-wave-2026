"""Live fal validation: four paid inference requests per input photo."""
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from pipeline import build_pipeline, normalize_photo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos', nargs='+', type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / 'backend' / '.env')
    output = ROOT / 'validation'
    output.mkdir(exist_ok=True)
    for photo in args.photos:
        name = photo.parent.name + '_' + photo.stem
        print(name, 'starting', flush=True)
        state = build_pipeline(lambda **event: print(name, event['status'], flush=True)).invoke(
            {'image': normalize_photo(photo.read_bytes()), 'garment': 'clothing'})
        state['result'].save(output / (name + '_studio.png'))
        (output / (name + '_quality.json')).write_text(json.dumps(state['quality'], indent=2))
        print(name, state['quality']['status'], flush=True)


if __name__ == '__main__':
    main()
