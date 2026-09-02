# Brownian-model test scripts

Experiments for the model in `planning/markdown/PLAN_simple.md` Eq. (13):

    sigma^2 = a^2/3 + 2 D_z H

| Script | What it does | GPU? |
|---|---|---|
| `test_brownian_model.py` | Height sweep, linear fit of sigma^2 vs H, PASS/FAIL assessment. Writes `brownian_test/brownian_model_results.csv` + `brownian_model_summary.json`. | Yes |
| `figure1_variance_vs_funnel.py` | sigma^2 vs funnel width W = 2a, full height sweep repeated per width. | Yes |
| `figure2_variance_ratio.py` | sigma^2_out / sigma^2_in ratio at max height vs W; needs D_z from the test's JSON summary (or `--dz`). | Yes |
| `figure3_parameter_influence.py` | Pure-theory panels (influence of H and D_z); needs H_ref and D_z from the test's JSON summary (or `--h-ref`/`--dz`). | No (but runs in the same container) |

All scripts share the `DEFAULTS` dictionary in `test_brownian_model.py`; every
entry has a matching CLI flag (see `--help` on each script).

Run them **in order**: the test first (it writes the JSON summary the figure
scripts read), then the figures.

## Running through CUDA (Docker)

The GPU stack is Windows driver → WSL2 → Docker → `galton-cuda` image
(CUDA 12.4 toolkit + Python venv with numba). Nothing is installed on Windows
besides Docker Desktop and the NVIDIA driver. Details: `planning/markdown/CUDA_GUIDE.md`.

Build the image once (repo root, only needed after Dockerfile/requirements changes):

```bash
docker build -t galton-cuda .
docker run --rm --gpus all galton-cuda nvidia-smi   # verify GPU passthrough
```

### Git Bash / MSYS terminals

`MSYS_NO_PATHCONV=1` stops Git Bash from mangling the `-v` path:

```bash
# 1. Main test (writes brownian_test/brownian_model_summary.json)
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    galton-cuda python3 testing/test_brownian_model.py \
    --drop-range 1.0 --wall-width 40 --balls 5000 --min-rows 5 --max-rows 50 --row-points 10

# 2. Figure 1 (variance vs funnel width)
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    galton-cuda python3 testing/figure1_variance_vs_funnel.py \
    --min-funnel 1.0 --max-funnel 6.0 --funnel-points 6 --balls 5000

# 3. Figure 2 (variance ratio; reads the summary from step 1)
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    galton-cuda python3 testing/figure2_variance_ratio.py \
    --min-funnel 1.0 --max-funnel 6.0 --funnel-points 6 --balls 5000

# 4. Figure 3 (theory panels; reads the summary from step 1)
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    galton-cuda python3 testing/figure3_parameter_influence.py
```

### PowerShell

Same commands without the `MSYS_NO_PATHCONV=1` prefix and with backtick
line continuations:

```powershell
docker run --rm --gpus all `
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work `
    galton-cuda python3 testing/test_brownian_model.py --balls 5000
```

### Notes

- The repo is bind-mounted at `/work`, so code changes apply without a
  rebuild, and all output (CSV/JSON/PNG/PDF) is written straight back to your
  Windows folders. `--rm` deletes the container when it exits.
- `--gpus all` is required for the three simulation scripts; without it you
  get `RuntimeError: No CUDA device visible`.
- The first kernel launch of each run is a few seconds slower (Numba JIT).
- Quick smoke test of the model pipeline: add `--rows 5 10 15 --balls 500`.
