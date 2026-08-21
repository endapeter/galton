# Running the CUDA Galton Board simulation

This guide explains how to run `galton_cuda.py` (the GPU version of
`galton_ultra.py`) and how the Docker + GPU setup underneath it works.

---

## 1. What you need

| Requirement | Notes |
|---|---|
| Windows 10/11 with WSL2 | Docker Desktop uses the WSL2 backend on Windows |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Linux containers mode (the default), WSL2 backend enabled in Settings → General |
| NVIDIA GPU + up-to-date Windows driver | e.g. the GeForce driver from nvidia.com - nothing else to install on Windows |
| ~15 GB free disk | the CUDA devel image is large (~12 GB) |

You do **not** need to install the CUDA toolkit, Python, or any Python
packages on Windows. Everything lives inside the Docker image.

Quick check that the basics are in place (any terminal):

```bash
nvidia-smi          # Windows driver sees the GPU
docker --version    # Docker Desktop is installed
```

## 2. How the Docker setup works

Three layers sit between Python and your GPU:

```
Windows host
 └─ NVIDIA Windows driver (the only GPU software installed on Windows)
     └─ WSL2 (lightweight Linux VM used by Docker Desktop)
         └─ Docker engine ("desktop-linux" context) runs the container
             └─ galton-cuda image: Ubuntu 22.04 + CUDA 12.4 toolkit + Python venv
```

Two different "CUDA" pieces are involved, and they come from different places:

- **The driver** (`nvidia-smi`) comes from Windows, passed through WSL2 into
  the container when you run with `--gpus all`. This is what lets the
  container *see* the GPU.
- **The CUDA toolkit** (compiler/NVVM libraries) is baked into the image
  (`nvidia/cuda:12.4.1-devel-ubuntu22.04`). Numba's CUDA target needs it at
  runtime to JIT-compile the kernels - that's why the image is the `-devel`
  flavour and so large.

The image also contains an isolated Python venv with exactly
`requirements.txt` (numpy, numba, matplotlib, pandas, scipy).

The run command bind-mounts this repository into the container at `/work`, so
the script reads the source from your disk and writes its output (`figures/`
PNGs and CSVs) straight back to your Windows folder - no copying in or out.

## 3. Build the image (once)

From the repository root (PowerShell or Git Bash):

```bash
docker build -t galton-cuda .
```

The image then shows up in Docker Desktop under the **Images** tab (search
"galton"). Note the **Containers** tab will look empty: every run command in
this guide uses `--rm`, so the container deletes itself the moment it exits -
you'll only see it listed while a run is in progress.

To verify GPU passthrough works:

```bash
docker run --rm --gpus all galton-cuda nvidia-smi
```

You should see your GPU listed inside the container.

## 4. Run the simulation

```bash
# Git Bash / MSYS terminals need MSYS_NO_PATHCONV=1 so the -v path is not mangled:
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    galton-cuda python3 galton_cuda.py

# PowerShell (no MSYS prefix):
docker run --rm --gpus all `
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work `
    galton-cuda python3 galton_cuda.py
```

What it does: a parameter sweep over the drop-funnel width (10 frames by
default), 2000 balls per frame. Each frame simulates every ball's full
trajectory on the GPU in a single kernel launch (one thread per ball), then
renders the histogram, Q-Q plot, and a single-ball verification walk on the
CPU with matplotlib. Output lands in `figures/` in the repository.

### Quick smoke test

Trim the workload with environment variables instead of editing the code:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
    -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
    -e GALTON_BALLS=200 -e GALTON_INTERVALS=2 \
    galton-cuda python3 galton_cuda.py
```

- `GALTON_BALLS` - balls per sweep frame (default 2000)
- `GALTON_INTERVALS` - number of sweep frames (default 10)

## 5. What to expect performance-wise

On an RTX 4050 Laptop GPU (24 SMs, and note: consumer GPUs run float64 at
1/64 of the float32 rate, which is why the kernel uses a closed-form quartic
solver instead of an iterative one):

| | per 2000-ball frame | full 10-frame sweep |
|---|---|---|
| `galton_ultra.py` (CPU, Numba) | ~8 minutes of simulation | ~1.5 hours |
| `galton_cuda.py` (GPU) | ~2 seconds of simulation | ~20 seconds of simulation |

End-to-end the CUDA run takes ~3.5 minutes: after the fix, ~90% of the wall
clock is matplotlib rendering (the verification-walk plot draws all ~2500 pegs
at 300 dpi), not simulation.

The GPU results match the CPU engine: identical landing statuses and a
statistically identical distribution. Individual ball positions on the full
50-row board differ, because the board is chaotic - a 1e-12 difference in
quartic-root precision amplifies over ~335 collisions per ball - but on short
boards CPU and GPU agree per-ball to ~1e-7.

## 6. Troubleshooting

**`docker: Error response from daemon: could not select device driver "nvidia"...`**
GPU passthrough isn't working. Update the NVIDIA driver, make sure Docker
Desktop uses the WSL2 backend (Settings → General → "Use the WSL 2 based
engine"), and check `nvidia-smi` works inside WSL (`wsl nvidia-smi`).

**`RuntimeError: No CUDA device visible` (from the script itself)**
You forgot `--gpus all` on the `docker run` command.

**Can't find the image / container in Docker Desktop**
Images live in the **Images** tab. Containers started with `--rm` vanish from
the **Containers** tab as soon as they exit - run a longer job (or drop
`--rm`) to watch it there. `docker images` and `docker ps` from a terminal
always show the truth.

**First frame is slower than the rest**
The first kernel launch includes Numba's JIT compilation of the kernel (a few
seconds, one time per process). Subsequent frames re-use the compiled kernel.

**Changed the code? Nothing changed in the output?**
The container mounts your source at runtime, so code changes apply
immediately - no rebuild needed. Only `requirements.txt` / `Dockerfile`
changes require `docker build` again.
