"""Studio generation worker for dev/compare_pipelines.py; never consumes targets."""
import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from pipeline import build_pipeline, normalize_photo, generation_settings
from garment_analysis import analysis_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--preds', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / 'backend/.env')
    generation_settings()
    analysis_settings()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    failures = 0
    for row in rows:
        identity, view = row['pair_id'], row['view']
        destination = args.preds / 'studio' / identity / (view + '.png')
        if destination.exists():
            raise ValueError(f'Refusing an existing prediction: {destination}')
        folder = args.diagnostics / identity
        folder.mkdir(parents=True, exist_ok=True)
        try:
            source = Path(row['inputs'][0])
            print(f'\n{identity}: Studio starting', flush=True)
            def save_color_input(key, label, image):
                if key == 'composed':
                    image.save(folder / (view + '_before_color.png'), format='PNG')
            state = build_pipeline(lambda **event:print(identity, event['status'], flush=True),
                                   save_color_input).invoke(
                {'image':normalize_photo(source.read_bytes()), 'additional_images':[]})
            destination.parent.mkdir(parents=True, exist_ok=True)
            state['result'].save(destination, format='PNG')
            if 'generated_image' in state:
                state['generated_image'].save(folder / (view + '_generated.png'), format='PNG')
            # Save real production mattes for exact local color replays; never
            # reconstruct a target mask or send scoring references to the model.
            for name, key in [('source_mask', 'mask'), ('result_mask', 'result_mask')]:
                if state.get(key) is not None:
                    state[key].save(folder / (view + '_' + name + '.png'), format='PNG')
            diagnostics = {key:state[key] for key in ('classification','target_selection','quality',
                'metrics','execution_trace','analysis','generation_prompt','pressing_prompt')}
            diagnostics['generation_reference_policy'] = state['reference_policy']
            (folder / (view + '.json')).write_text(json.dumps(diagnostics, indent=2))
        except Exception as exc:
            failures += 1
            # Never leave an incomplete or stale image to receive a valid score.
            destination.unlink(missing_ok=True)
            error = {'garment':identity,'view':view,'status':'failed','error_type':type(exc).__name__}
            (folder / (view + '_error.json')).write_text(json.dumps(error, indent=2))
            print(f'{identity}: Studio failed ({type(exc).__name__}); official missing-slot policy applies.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
