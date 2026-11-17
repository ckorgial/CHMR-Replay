"""Reproduce CHMR / RG-CHMR experiments from source and user-supplied inputs.

Run ``python run.py --help`` for stages, dataset selection and data-cache options.
Outputs and downloaded inputs are never part of the source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STAGES = {
    'benchmark': [['experiment.py']],
    'diagnostics': [['diagnostics.py']],
    'analysis': [['extensions.py'], ['fixed_weights.py']],
    'figures': [['colour_figures.py'], ['supplement_plots.py']],
    'verify': [['verify.py']],
    'rg-main': [['rg_evaluate.py', 'main']],
    'rg-stress': [['rg_evaluate.py', 'stress']],
    'rg-controls': [['rg_controls.py']],
    'rg-kitti': [['rg_evaluate.py', 'kitti']],
    'rg-figures': [['rg_plot.py']],
    'gaps': [['gaps.py']],
    'tests': [['-m', 'unittest', 'discover', '-s', 'tests', '-v']],
}
BASELINE = ['benchmark', 'diagnostics', 'analysis', 'figures', 'verify']
ALL_SYNTHETIC = [*BASELINE, 'rg-main', 'rg-stress', 'rg-controls', 'rg-figures', 'tests']

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['all', 'baseline', *STAGES],
                    help='all: baseline, RG evaluation, controls and plots; baseline: original CHMR suite')
    ap.add_argument('--datasets', choices=['synthetic','all'], default='synthetic',
                    help='synthetic = MR/PA; all = MR/PA/KITTI with external raw data')
    ap.add_argument('--cache', type=Path, default=Path('cache'),
                    help='location of external KITTI labels and detection files')
    io=ap.add_mutually_exclusive_group()
    io.add_argument('--offline', action='store_true',help='disable all raw KITTI downloads (default)')
    io.add_argument('--allow-download', action='store_true',
                    help='explicitly permit downloading official KITTI inputs into cache')
    ap.add_argument('--keep-results', action='store_true',
                    help='preserve prior output directories when rebuilding benchmarks')
    args=ap.parse_args()

    if args.stage in ('gaps','rg-kitti') and args.datasets!='all':
        ap.error(f'{args.stage} requires --datasets all and the KITTI data cache')
    if args.stage=='all':
        stages=list(ALL_SYNTHETIC)
        if args.datasets=='all':
            stages.insert(stages.index('rg-figures'),'rg-kitti')
            stages.insert(stages.index('rg-kitti'),'gaps')
    elif args.stage=='baseline':stages=list(BASELINE)
    else:stages=[args.stage]

    cache=args.cache.expanduser().resolve()
    env=os.environ.copy()
    env.update(CHMR_SUITES='MR,PA' if args.datasets=='synthetic' else 'MR,PA,KITTI',
               CHMR_CACHE=str(cache), MPLBACKEND='Agg',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Data downloading must be explicit, even when a stale shell environment is present.
    env['CHMR_OFFLINE']='0' if args.allow_download else '1'
    env['CHMR_ALLOW_DOWNLOAD']='1' if args.allow_download else '0'
    if not args.keep_results and 'benchmark' in stages:
        for name in ('results', 'supplement', 'rg_results'):
            shutil.rmtree(ROOT/name, ignore_errors=True)
    for name in ('results','supplement','rg_results'):
        (ROOT/name).mkdir(exist_ok=True)

    prerequisites = {
        'diagnostics':['results/MR_predictions.npz'],
        'analysis':['results/settings.csv'],
        'figures':['supplement/scores.csv'],
        'rg-main':['results/settings.csv'],
        'rg-stress':['results/settings.csv'],
        'rg-controls':['results/settings.csv'],
        'rg-kitti':['results/settings.csv'],
        'rg-figures':['rg_results/main_metrics.csv'],
    }
    for stage in stages:
        for required in prerequisites.get(stage, []):
            if not (ROOT/required).exists():
                ap.error(f'{stage} requires {required}; first run the prerequisite benchmark/analysis stage')
        for script in STAGES[stage]:
            command=[sys.executable, '-u', *script]
            if script[0] in ('experiment.py', 'diagnostics.py', 'rg_controls.py','rg_evaluate.py'):
                command.extend(['--cache',str(cache)])
            if stage=='rg-figures' and args.datasets=='all':
                command.append('--require-kitti')
            print('\n==>',stage, ':',' '.join(command),flush=True)
            subprocess.run(command,cwd=ROOT,env=env,check=True)

    import numpy, pandas, scipy, matplotlib
    manifest={
        'finished_utc':datetime.now(timezone.utc).isoformat(),
        'datasets':args.datasets, 'stages':stages, 'offline':not args.allow_download,
        'python':sys.version.split()[0],
        'dependencies':{'numpy':numpy.__version__,'pandas':pandas.__version__,
                        'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(ROOT.glob('*.py'))},
    }
    (ROOT/'results'/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nCompleted requested stages. Generated files are under results/, supplement/, and rg_results/.',flush=True)

if __name__=='__main__':main()
