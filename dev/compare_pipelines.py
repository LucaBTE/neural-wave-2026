#!/usr/bin/env python3
"""Compare the hosted baseline and Studio on identical inputs with the official scorer."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
DEV = Path(__file__).resolve().parent
REPO = DEV.parent
SCORE_PYTHON = DEV / '.score-venv/bin/python'
STUDIO_PYTHON = DEV / 'btc-sartiq/backend/.venv/bin/python'
METRICS = ('score', 'colour', 'lpips', 'clip', 'geometry', 'ssim')


def select_inputs(manifest, dataset, count, generate_only=False):
    if count < 1:
        raise ValueError('--count must be positive')
    helpers = runpy.run_path(str(REPO / 'baselines/baseline_identity.py'))
    rows = helpers['load_rows'](manifest, dataset)
    selected, seen = [], set()
    for row in rows:
        if row['id'] in seen:
            continue
        seen.add(row['id'])
        for value in (row['id'], row['view']):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
                raise ValueError('Unsafe garment ID or view in manifest')
        source = row['inputs'][0].resolve()
        if not source.is_file():
            raise ValueError(f'Input not found: {source}')
        target = row['target'].resolve() if row['target'] else None
        if not generate_only and (target is None or not target.is_file()):
            raise ValueError(f"{row['id']}: official scoring needs its target image. "
                             'The supplied test split has no targets. Use --split train, '
                             'a --manifest containing official targets, or --generate-only for images without scores.')
        item = {'pair_id':row['id'], 'view':row['view'], 'inputs':[str(source)]}
        if target is not None:
            item['target'] = str(target)
        selected.append(item)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} distinct products available; requested {count}')
    return selected


def run_command(command, allow_partial=False):
    result = subprocess.run([str(part) for part in command], cwd=REPO, check=False,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    if result.returncode != 0 and not (allow_partial and result.returncode == 1):
        raise RuntimeError(f'Command failed (exit {result.returncode}): {command[1]}')
    return result.returncode


def prior_baseline_files(folder, selected):
    """Reuse real predictions only when the recorded source paths/views match.

    Older runs have no source hashes, so this checks manifest identity, not that
    source files have remained byte-identical since the previous run.
    """
    previous = [json.loads(line) for line in (folder / 'selected.jsonl').read_text().splitlines() if line.strip()]
    by_key = {(row['pair_id'], row['view']):row for row in previous}
    if len(by_key) != len(previous):
        raise ValueError('Previous baseline manifest contains duplicate products')
    files = []
    for row in selected:
        key = (row['pair_id'], row['view'])
        old = by_key.get(key)
        if (old is None or [str(Path(p).resolve()) for p in old['inputs']] != row['inputs'] or
                (str(Path(old['target']).resolve()) if old.get('target') else None) != row.get('target')):
            raise ValueError(f'Previous baseline has different inputs/target for {key[0]}/{key[1]}')
        source = folder / 'predictions/baseline' / key[0] / (key[1] + '.png')
        if not source.is_file() or source.is_symlink() or not source.stat().st_size:
            raise ValueError(f'Previous baseline prediction missing or invalid: {source}')
        files.append((source, Path(key[0]) / (key[1] + '.png')))
    return files


def preflight(manifest, dataset, generate_only):
    for interpreter in (SCORE_PYTHON, STUDIO_PYTHON):
        if not interpreter.is_file():
            raise ValueError(f'Missing environment: {interpreter}. See dev/doc/how-to-score.md.')
    # Import model clients/configuration, but never instantiate scorer models or call fal.
    run_command([STUDIO_PYTHON, '-B', '-c',
        "import sys; from dotenv import load_dotenv; "
        f"sys.path.insert(0,{str(DEV / 'btc-sartiq/backend')!r}); "
        f"load_dotenv({str(DEV / 'btc-sartiq/backend/.env')!r}); "
        "from pipeline import generation_settings, analysis_settings; "
        "generation_settings(); analysis_settings()"])
    run_command([SCORE_PYTHON, '-B', '-c',
        "import os; from dotenv import load_dotenv; "
        f"load_dotenv({str(DEV / 'btc-sartiq/backend/.env')!r}); "
        "assert any(os.getenv(k) for k in ('IMAGE_API_KEY','FAL_KEY','FAL_API_KEY')), 'Missing fal key in backend/.env'"])
    if not generate_only:
        run_command([SCORE_PYTHON, '-B', '-c',
            f"import sys; sys.path.insert(0,{str(DEV)!r}); import eval_val; "
            f"eval_val.prepare_scoring({str(manifest)!r},{str(dataset)!r})"])


def generation_jobs(output, selected, dataset, reuse_baseline=False):
    """Isolate worker manifests/output, including the baseline's append-only log."""
    tasks = []
    for row in selected:
        identity, view = row['pair_id'], row['view']
        for team in ('baseline', 'studio'):
            if team == 'baseline' and reuse_baseline:
                continue
            work = output / 'work' / team / identity / view
            work.mkdir(parents=True)
            manifest = work / 'input.jsonl'
            # Workers only need inputs. The official scoring target remains in
            # selected.jsonl and never enters a generation worker's manifest.
            manifest.write_text(json.dumps({key: row[key] for key in ('pair_id', 'view', 'inputs')}) + '\n')
            predictions = work / 'predictions'
            if team == 'baseline':
                command = [SCORE_PYTHON, '-B', DEV / 'baseline_nanobanana.py',
                           '--manifest', manifest, '--dataset', dataset,
                           '--out', predictions, '--team', team]
            else:
                command = [STUDIO_PYTHON, '-B', DEV / 'btc-sartiq/scripts/benchmark_studio.py',
                           '--manifest', manifest, '--preds', predictions,
                           '--diagnostics', output / 'diagnostics']
            log = output / 'logs' / team / identity / (view + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            tasks.append({'team': team, 'garment': identity, 'view': view,
                          'command': command, 'predictions': predictions, 'log': log})
    return tasks


def run_generation(tasks, output, jobs):
    """Bound the total live worker count across BOTH pipelines, in this process.

    Separate processes isolate Studio's per-run metrics and the baseline's file
    appends. Only this parent publishes predictions and combines manifest records.
    Pending workers are never launched after Ctrl+C.
    """
    pending = iter(tasks)
    active, completed = [], []

    def write_status(status=None):
        temporary = output / 'generation_status.json.tmp'
        temporary.write_text(json.dumps({
            'status': status or ('complete' if len(completed) == len(tasks) else 'running'),
            'jobs': jobs, 'completed': len(completed), 'total': len(tasks), 'runs': completed,
            **{team + '_exit_code': int(any(r['exit_code'] for r in completed if r['team'] == team))
               for team in ('baseline', 'studio')},
        }, indent=2))
        temporary.replace(output / 'generation_status.json')

    def record(task, code, started, error_type=None):
        entry = {key: task[key] for key in ('team', 'garment', 'view')}
        entry.update(exit_code=code, seconds=round(time.monotonic() - started, 3),
                     log=str(task['log'].relative_to(output)))
        if error_type:
            entry['error_type'] = error_type
        completed.append(entry)
        write_status()
        outcome = 'complete' if code == 0 else 'failed'
        print(f"[{len(completed)}/{len(tasks)}] {task['team']} {task['garment']}/{task['view']}: "
              f"{outcome} ({entry['seconds']:.1f}s) · {entry['log']}", flush=True)

    def publish(task, code):
        source = task['predictions'] / task['team'] / task['garment'] / (task['view'] + '.png')
        # A successful process without its expected output is still a missing
        # prediction. Failed workers must never publish a half-written PNG.
        if code == 0:
            if not source.is_file() or source.is_symlink() or not source.stat().st_size:
                code = 1
            else:
                target = output / 'predictions' / task['team'] / task['garment'] / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
        manifest = task['predictions'] / '_manifest.jsonl'
        if task['team'] == 'baseline' and manifest.is_file():
            with (output / 'predictions/_manifest.jsonl').open('a') as handle:
                for line in manifest.read_text().splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if 'output' in entry:
                        if code == 0:
                            entry['output'] = str(target)
                        else:
                            entry.pop('output')
                    handle.write(json.dumps(entry) + '\n')
        return code

    write_status()
    try:
        while True:
            while len(active) < jobs:
                task = next(pending, None)
                if task is None:
                    break
                started = time.monotonic()
                log = task['log'].open('w')
                try:
                    process = subprocess.Popen([str(p) for p in task['command']], cwd=REPO,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'},
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    log.close()
                    record(task, 1, started, type(exc).__name__)
                    continue
                active.append((process, task, log, started))
                print(f"[running] {task['team']} {task['garment']}/{task['view']} "
                      f"· {task['log'].relative_to(output)}", flush=True)
            if not active:
                break
            for item in active[:]:
                process, task, log, started = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                record(task, publish(task, code), started)
            if active:
                time.sleep(.1)
    except KeyboardInterrupt:
        write_status('interrupted')
        raise
    except Exception:
        write_status('error')
        raise
    finally:
        # Cancel local polling/uploads and descendants as well as the worker.
        # Already submitted remote fal jobs may still finish on the provider.
        for process, _, _, _ in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, _, log, _ in active:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            finally:
                log.close()
    return {team: int(any(r['exit_code'] for r in completed if r['team'] == team))
            for team in ('baseline', 'studio')}


def write_comparison(output, selected):
    aggregates, records = {}, {}
    expected = {(row['pair_id'], row['view']) for row in selected}
    for team in ('baseline', 'studio'):
        folder = output / 'scores' / team
        with (folder / 'leaderboard.csv').open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError('Official evaluator returned an unexpected aggregate')
        aggregates[team] = rows[0]
        records[team] = {}
        for line in (folder / 'scores.jsonl').read_text().splitlines():
            row = json.loads(line)
            records[team][(row['garment'], row['view'])] = row
        if set(records[team]) != expected:
            raise ValueError('Official score records do not match the selected products')
    with (output / 'comparison.csv').open('w', newline='') as handle:
        fields = ['pipeline', *METRICS, 'pairs_scored', 'pairs_missing', 'n_review', 'n_invalid']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for team, row in aggregates.items():
            writer.writerow({'pipeline':team, **row})
    with (output / 'per_garment.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['garment','view','baseline_score','studio_score','studio_minus_baseline',
                         'baseline_status','studio_status','baseline_missing','studio_missing'])
        for row in selected:
            key = (row['pair_id'], row['view'])
            a, b = records['baseline'][key], records['studio'][key]
            writer.writerow([*key,a['score'],b['score'],round(b['score']-a['score'],2),
                             a['status'],b['status'],a.get('missing',False),b.get('missing',False)])
    lines = ['| Metric (0–100) | Baseline | Studio |', '|---|---:|---:|']
    for metric in METRICS:
        scale = 1 if metric == 'score' else 100
        values = [float(aggregates[team][metric]) * scale for team in ('baseline','studio')]
        lines.append(f'| {metric} | {values[0]:.2f} | {values[1]:.2f} |')
    lines += ['', 'Official garment-weighted aggregates; missing/invalid predictions retain the official zero-score policy.',
              'comparison.csv preserves original metric units (total 0–100, components 0–1).', '']
    report = '\n'.join(lines)
    (output / 'comparison.md').write_text(report, encoding='utf-8')
    print('\n' + report, flush=True)
    return aggregates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train','dev','test'), default='train')
    parser.add_argument('--manifest', type=Path, help='Optional manifest with official targets for the selected split')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=2,
                        help='Total concurrent generation workers across both pipelines (default: 2; 1 for serial)')
    parser.add_argument('--out', type=Path, help='New output directory, inside dev; existing directories are refused')
    parser.add_argument('--generate-only', action='store_true', help='Generate both sets without reference scores')
    parser.add_argument('--dry-run', action='store_true', help='Print the selected inputs; make no API/model calls')
    parser.add_argument('--baseline-from', type=Path,
                        help='Reuse baseline images from a previous comparison with matching input/target paths')
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        raise ValueError('--jobs must be between 1 and 8')
    dataset = args.dataset.resolve()
    manifest = (args.manifest or dataset / args.split /
                ('pairs.jsonl' if args.split == 'train' else 'inputs.jsonl')).resolve()
    selected = select_inputs(manifest, dataset, args.count, args.generate_only)
    baseline_from = args.baseline_from.resolve() if args.baseline_from else None
    reused = prior_baseline_files(baseline_from, selected) if baseline_from else None
    print(f'Split: {args.split}; products: {len(selected)}; same FIRST input for both pipelines.', flush=True)
    print(f'Generation concurrency: {args.jobs} total workers; official scoring remains sequential.', flush=True)
    for row in selected:
        print(f"  {row['pair_id']}/{row['view']}: {row['inputs'][0]}", flush=True)
    if baseline_from:
        print(f'Reusing existing baseline predictions: {baseline_from}', flush=True)
    if args.dry_run:
        print('Dry run: no generation or scoring was performed.')
        return 0
    output = (args.out or DEV / 'comparisons' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if not output.is_relative_to(DEV.resolve()):
        raise ValueError('--out must be inside dev')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Output: {output}', flush=True)
    subset = output / 'selected.jsonl'
    subset.write_text(''.join(json.dumps(row)+'\n' for row in selected))
    preflight(subset, dataset, args.generate_only)
    predictions = output / 'predictions'
    for team in ('baseline','studio'):
        (predictions / team).mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps({
        'dataset':str(dataset), 'source_manifest':str(manifest), 'split':args.split,
        'selected_manifest':str(subset), 'count':len(selected), 'input_policy':'first_input_only',
        'generate_only':args.generate_only, 'baseline_from':str(baseline_from) if baseline_from else None,
        'generation_jobs':args.jobs,
    }, indent=2))
    if reused is not None:
        for source, relative in reused:
            destination = predictions / 'baseline' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    tasks = generation_jobs(output, selected, dataset, reuse_baseline=reused is not None)
    statuses = run_generation(tasks, output, args.jobs)
    baseline_status, studio_status = statuses['baseline'], statuses['studio']
    if args.generate_only:
        print(f'\nImages: {predictions}\nNo reference targets: official scores were NOT calculated.')
        return 1 if baseline_status or studio_status else 0
    for team in ('baseline','studio'):
        print(f'\nOfficial local scoring: {team}', flush=True)
        run_command([SCORE_PYTHON, '-B', DEV / 'eval_val.py', '--pairs',subset,
            '--dataset',dataset,'--preds',predictions,'--team',team,
            '--out',output / 'scores' / team,'--jobs','1'])
    write_comparison(output, selected)
    print(f'Reports: {output}', flush=True)
    return 1 if baseline_status or studio_status else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        print('\nComparison interrupted; no further generation was started.', file=sys.stderr)
        raise SystemExit(130) from None
