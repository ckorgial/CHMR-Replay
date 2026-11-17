# CHMR-Replay

Reproducible Python source for **Causal Horizon-Matched Replay (CHMR)** and the reliability-gated extension **RG-CHMR**, applied to lead-vehicle range forecasting during detector track loss. The repository includes the synthetic trajectory generators, FIR replay forecasts, Kalman and interacting-multiple-model (IMM) baselines, reliability-gated experiments, independent mathematical checks, KITTI tracking-data extraction, and plotting scripts.

This README describes **only how to install and run the code**. No archived scores, generated plots, predictions, or copied KITTI datasets are distributed in the repository.

## 1. Requirements

- Python **3.11–3.13** (64-bit recommended).
- `pip` and an environment capable of installing NumPy, SciPy, pandas, and Matplotlib.
- CPU execution; a GPU is not required.
- Disk space for generated CSV, compressed NumPy arrays, figures, and optionally the KITTI cache.
- Network access is **not required** for MR and PA synthetic experiments.

Run from the **repository root**, where `run.py` and `protocol.json` are located. For the complete workflow, use a fresh checkout to avoid mixing outputs from different source revisions.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the virtual environment's interpreter directly, e.g. `.\.venv\Scripts\python.exe run.py --help`, or use a temporary process-scope execution policy.

## 2. Check the installation

```bash
python -m unittest discover -s tests -v
python run.py verify --datasets synthetic --offline
python run.py --help
```

The unit tests exercise the RG forecast, replay causality, covariance-moment calculation, and constrained horizon smoother. `verify` independently checks FIR coefficients, numerical guards, covariance identities, and other mathematical properties; it does **not** need any previously exported results. GitHub Actions runs these checks on supported Python versions.

## 3. Run synthetic experiments without external data

The default workload includes **MR (multi-regime)** and **PA (piecewise acceleration)**. All random seeds and protocol parameters are preserved in `protocol.json` and the source scripts.

To run the entire synthetic pipeline, including original CHMR, RG-CHMR evaluations, robustness experiments, controls, figures, and tests:

```bash
python run.py all --datasets synthetic --offline
```

For a smaller sequence of runs, start with the benchmark and then execute individual stages:

```bash
python run.py benchmark --datasets synthetic --offline
python run.py diagnostics --datasets synthetic --offline
python run.py analysis --datasets synthetic --offline
python run.py figures --datasets synthetic --offline
python run.py rg-main --datasets synthetic --offline
python run.py rg-stress --datasets synthetic --offline
python run.py rg-controls --datasets synthetic --offline
python run.py rg-figures --datasets synthetic --offline
python run.py verify --datasets synthetic --offline
```

The order matters: the benchmark selects development-set parameters and writes `results/settings.csv`, which is consumed by the RG stages. `analysis` writes the supplementary CSV inputs consumed by `figures`. `rg-figures` requires a completed `rg-main` stage. The `benchmark` and `all` stages start by clearing previously generated experiment directories unless `--keep-results` is supplied.

For only the original CHMR benchmark, mathematical analyses, plots and verification:

```bash
python run.py baseline --datasets synthetic --offline
```

## 4. Run KITTI experiments

KITTI annotations and the PointRCNN detector files are external, copyrighted data, and are **not bundled** here. Follow the KITTI license and the upstream detector project's terms. The loader checks the official annotation archive SHA-256 and verifies pinned detector Git blob hashes using `detector_blobs.json`.

The following files must be present in `cache/` (or another folder supplied via `--cache`):

```text
cache/
├── labels.zip             # KITTI data_tracking_label_2.zip, renamed
├── 0000.txt
├── 0001.txt
├── ...
└── 0020.txt             # 21 pinned PointRCNN detector text files
```

**External download sources:**

- KITTI tracking labels: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
- PointRCNN detector files from AB3DMOT, pinned at commit `61f3bd72574093e367916c757b4747ca445f978c`: https://github.com/xinshuoweng/AB3DMOT/tree/61f3bd72574093e367916c757b4747ca445f978c/data/KITTI/detection/pointrcnn_Car_val
- KITTI benchmark and usage conditions: https://www.cvlibs.net/datasets/kitti/eval_tracking.php

You may download and stage these files manually. Alternatively, after reviewing their terms, explicitly opt into fetching the required public files:

```bash
python run.py all --datasets all --cache cache --allow-download
```

To keep the entire run offline after populating the cache:

```bash
python run.py all --datasets all --cache cache --offline
```

The `--datasets all` option runs the synthetic and KITTI benchmark, the first-missing-frame extraction (`gaps`), and the KITTI RG evaluation (`rg-kitti`). The script stops with an actionable error if an external dataset is missing or its checksum differs. The KITTI split uses the documented `sequence % 3` assignment.

**Data scope:** The KITTI procedure works from tracking annotations and externally published detector outputs. It is an encounter-level forecasting evaluation, not a deployable detector/tracker. Future annotation values are used to evaluate forecasts, never passed to the prediction function.

## 5. Available commands

Run `python run.py --help` for the authoritative option list. Main stages:

| Stage | Purpose | Required preceding stage |
|---|---|---|
| `benchmark` | Build CHMR baseline predictions and selected settings | none |
| `diagnostics` | Causal replay and matched-budget checks | `benchmark` |
| `analysis` | Supplementary weighting, stress and screening computations | `benchmark` |
| `figures` | Generate CHMR and supplemental plots | `analysis` |
| `verify` | Mathematical and consistency checks | none |
| `rg-main` | Evaluate RG-CHMR synthetic cohorts | `benchmark` |
| `rg-stress` | Run RG-CHMR stress protocols | `benchmark` |
| `rg-controls` | Generate frozen-protocol comparator controls | `benchmark` |
| `rg-figures` | Plot paired RG comparisons and architecture | `rg-main` |
| `rg-kitti` | Evaluate RG-CHMR on KITTI (external data) | KITTI `benchmark` |
| `gaps` | Test first-unmatched KITTI detector frames | KITTI data |
| `tests` | Run unit tests | none |
| `baseline` | Run all original CHMR stages | none |
| `all` | Run the complete available pipeline | none |

Other options: `--offline` (default), `--allow-download` (explicit KITTI input download), `--cache PATH` (raw-data folder), and `--keep-results` (do not discard prior outputs).

## 6. Generated files and reproducibility

All outputs are written locally and ignored by Git:

```text
results/        # baseline CSV files, per-encounter NPZ, figures and run_manifest.json
supplement/     # diagnostic and supplemental CSV files
rg_results/     # gated evaluation CSV files, paired predictions and figures
cache/          # external KITTI inputs, if downloaded/staged
```

Input configuration is in `protocol.json`. The master runner records Python/dependency versions and SHA-256 hashes of all Python source files in `results/run_manifest.json`. Keep that file with any externally shared evaluation package to identify the exact implementation that produced its outputs.

`rg_evaluate.py` writes paired synthetic predictions for the main cohorts. For formal comparisons, regenerate inputs from the same run and use encounter-level pairing for synthetic examples and sequence-level grouping for KITTI. Do not treat ten horizons from one encounter as ten independent observations. All proposed forecasts operate on pre-loss histories.

### Troubleshooting

- **`ModuleNotFoundError`**: activate the correct environment and reinstall `requirements.txt`.
- **Missing `results/settings.csv`**: run `python run.py benchmark --datasets synthetic --offline` before an RG stage, or run the complete `all` pipeline.
- **Missing supplement files when plotting**: run `analysis` before `figures`.
- **Missing `rg_results/main_metrics.csv`**: run `rg-main` before `rg-figures`.
- **KITTI checksum or missing-file error**: verify the official labels archive and every pinned detector file. The loader deliberately rejects different detector revisions.
- **Unattended or headless environment**: the runner sets `MPLBACKEND=Agg` and caps common BLAS thread counts.
- **Stale outputs**: use a new checkout or rerun `benchmark` / `all` without `--keep-results`. Never mix output folders from different runs.

## 7. File map

```text
run.py                  # recommended command-line runner
models.py               # synthetic data, FIR replay, Kalman/IMM filters
rg.py                   # reliability-gated causal forecasting extension
rg_evaluate.py          # RG evaluation and paired comparisons
rg_controls.py          # frozen controls and ablations
rg_plot.py              # RG charts
experiment.py           # CHMR benchmark
diagnostics.py          # causal and analytical checks
extensions.py           # supplementary experiments
fixed_weights.py        # weight diagnostics
gaps.py                 # KITTI missing-detector-match protocol
kitti.py                # dataset extraction, hash checking
colour_figures.py       # main CHMR figures
supplement_plots.py     # supplementary figures
verify.py               # source-based mathematical verification
tests/test_rg.py        # unit tests
protocol.json           # seeds and protocol
detector_blobs.json     # pinned input hashes
```

## 8. Publishing the repository

Choose and add the code license appropriate to your ownership and desired reuse permissions before making the repository public. No open-source license is asserted by this package. To create the repository, run:

```bash
git init
git add .
git commit -m "Release reproducible CHMR and RG-CHMR code"
git branch -M main
git remote add origin https://github.com/ckorgial/CHMR-Replay.git
git push -u origin main
```

The remote URL is the **intended** GitHub destination and is not evidence that a repository was created or pushed. If the remote already exists, clone it first or reconcile its history before pushing.
