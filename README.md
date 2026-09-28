# Learning diverse dynamics with compact quantum neural operators

Code for the QuanONet Reusability Report, covering reproduction, robustness and generalizability experiments on six original benchmarks and three further equations: Burgers, Allen–Cahn and the wave equation. Model implementations are from [QuanONet](https://github.com/Wang-Ruocheng/QuanONet); quantum models use the original MindQuantum `mqvector` simulator.

## Installation

Tested with Python 3.9 on Linux. All experiments support CPU execution; allow 16–32 GB of RAM for full PDE datasets.

```bash
conda create -n quanonet-rr python=3.9
conda activate quanonet-rr
pip install -r requirements.txt
python scripts/setup_upstream.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The setup script downloads QuanONet at commit `d01934380a148c96526261ed99218b22c5097353`. Run the commands below from the repository root.

## Quick start

Generate one dataset and evaluate the included TF-QuanONet checkpoint:

```bash
bash data/prep.sh --suite original --operators Antideriv --seeds 0 --num_workers 4

python scripts/evaluate.py \
  --checkpoint checkpoints/reproduction/Antideriv/TFQuanONet_scale0.1_n1000_budget1_seed0 \
  --data data/rr_original/main/Antideriv/seed0/dataset.npz \
  --out outputs/antiderivative --num_workers 1 --mq-workers 4
```

Predictions and metrics are saved under `outputs/antiderivative/`. The expected test relative L2 error is approximately **2.262%**.

Checkpoints are included under `checkpoints/<experiment>/<equation>/<run>/`. Each directory contains `config.json`, `metrics.json` and `best.ckpt` (MindSpore) or `best.pt` (PyTorch).

## Data

Prepare all training, validation and test data, including the numerical references used in Table 1 and Figure 3:

```bash
bash data/prep.sh --num_workers 8
```

This covers all nine equations and repetitions 0–4. Completed datasets are skipped. Use `--suite original` or `--suite extension`, `--operators` and `--seeds` to select a subset, as in the quick start. Generated data are stored under `data/rr_original/`, `data/rr_extension/` and `data/references/`.

## Training

`configs/experiments.jsonl` specifies the data, model settings, seeds and training commands for each experiment. For example:

```bash
python scripts/run_experiments.py --suite core \
  --operator Antideriv --variant TFQuanONet --seeds 0 \
  --num_workers 1 --mq-workers 4
```

Add `--dry-run` to preview commands. Omit the operator, variant and seed filters to run a complete suite. Training outputs are saved under `outputs/`.

| Suite | Experiment |
|---|---|
| `core` | Main comparison on the six original benchmarks |
| `table1_rest` | Additional models and encoding scales |
| `data_size` | Training-set size |
| `noise_augmentation` | Noise-augmented training and clean controls |
| `scaling` | Qubit number and latent dimension |
| `extension`, `extension_small`, `extension_large` | Three new equations at three model budgets |
| `training_curves` | Validation trajectories |

## Additional evaluation

The evaluation scripts accept `--help` for their full arguments.

- **Observation noise:** add `--noise-levels 0 0.01 0.05 0.1` to `scripts/evaluate.py`. Noise is scaled by the clean training-sensor standard deviation.
- **Scaling MSE:** add `--splits train test` to `scripts/evaluate.py`.
- **Local solution errors:** use `scripts/evaluate_local.py`, for example:

```bash
python scripts/evaluate_local.py \
  --checkpoint checkpoints/generalizability/Burgers/TFQuanONet_scale0.1_n1000_budget1_seed0 \
  --out outputs/burgers_local --num_workers 1 --mq-workers 4
```

For diffusion–reaction and advection, `scripts/rescore_references.py` evaluates saved predictions against the released labels and both numerical references. Supply `--reference data/references/<equation>/seed<seed>/reference_labels.npz`, `--predictions <evaluation_output>/predictions_test.npz` and `--output <scores.json>`. For noisy predictions, select the corresponding `predictions_test_noise*.npz` file; references always correspond to the clean latent input functions. Reference conventions are recorded in each dataset's `settings.json` and described in Methods.

For diffusion–reaction inference at physical time, `src.evaluation.coordinates.diffusion_model_coordinates` maps `(x, physical_time)` to model coordinates over the supported interval `0 <= physical_time <= 50/99`.

## Figures and results

```bash
python scripts/plot_figures.py --num_workers 1
python scripts/summarize.py --num_workers 4
```

These commands regenerate the five figures from the included source data and summarize the checkpoint test errors, writing to `results/generated/`. Per-run results and summaries are in `results/paper/source_data/`; Table 1 is provided in `results/paper/table1.tex`.

Please acknowledge [QuanONet and its authors](https://github.com/Wang-Ruocheng/QuanONet) when using their implementation. The upstream code retains its MIT license.
