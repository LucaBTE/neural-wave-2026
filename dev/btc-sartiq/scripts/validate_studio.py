"""Live fal validation: six default inferences; up to eight with duplicate repair."""
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
    parser.add_argument('--views', action='store_true',
                        help='treat all photos as views of one garment (first is primary, max 5)')
    args = parser.parse_args()
    load_dotenv(ROOT / 'backend' / '.env')
    output = ROOT / 'validation'
    output.mkdir(exist_ok=True)
    runs = [(args.photos[0], args.photos[1:])] if args.views else [(photo, []) for photo in args.photos]
    for photo, extras in runs:
        name = photo.parent.name + '_' + photo.stem
        print(name, 'starting', flush=True)
        state = build_pipeline(lambda **event: print(name, event['status'], flush=True)).invoke(
            {'image': normalize_photo(photo.read_bytes()), 'garment': 'clothing',
             'additional_images': [normalize_photo(extra.read_bytes()) for extra in extras]})
        state['result'].save(output / (name + '_studio.png'))
        (output / (name + '_prompt.txt')).write_text(state['generation_prompt'], encoding='utf-8')
        (output / (name + '_quality.json')).write_text(json.dumps(state['quality'], indent=2))
        (output / (name + '_timings.json')).write_text(json.dumps(state['metrics'], indent=2))
        (output / (name + '_trace.json')).write_text(json.dumps(state['execution_trace'], indent=2))
        (output / (name + '_classification.json')).write_text(json.dumps(state['classification'], indent=2))
        print(name, state['quality']['status'], flush=True)


if __name__ == '__main__':
    main()
