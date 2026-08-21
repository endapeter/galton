# CUDA build for the galton_cuda package.
#
# The NVIDIA *driver* comes from the Windows host through WSL2 (verified with
# nvidia-smi inside WSL); the CUDA *toolkit* (nvcc / NVVM that Numba's CUDA
# target links against at JIT time) is provided by this image.
#
# Build:
#   docker build -t galton-cuda .
#
# Run (mounts this folder into the container so figures/CSVs land on Windows):
#   MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
#       -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
#       galton-cuda python3 -m galton_cuda
#
# Quick smoke test (200 balls, 2 sweep frames):
#   MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
#       -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
#       -e GALTON_BALLS=200 -e GALTON_INTERVALS=2 \
#       galton-cuda python3 -m galton_cuda

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv to avoid any PEP 668 externally-managed-environment issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source and output are bind-mounted at runtime; nothing else to bake in.
CMD ["python3", "-m", "galton_cuda"]
