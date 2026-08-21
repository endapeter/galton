# galton_cuda: GPU Galton board simulation as an importable package.
#
# The heavy lifting (quartic solver, trajectory kernel, batch launcher) lives
# in .kernels; geometry generation and the spatial hash in .geometry; CPU
# rendering in .visualization; the parameter-sweep controller in .sweep.
#
# Typical use (inside the CUDA Docker container, see CUDA_GUIDE.md):
#
#     from galton_cuda import run_sweep
#
#     results = run_sweep(
#         sweep_param='X_DROP_RANGE',   # parameter to sweep
#         low=0.05, high=47.0,          # first / last swept value
#         steps=10,                     # number of iterative steps
#         fixed={'N_ROWS': 50},         # parameters held constant
#         n_balls=2000,
#     )
#
# Or from the command line:
#
#     python3 -m galton_cuda --sweep COEFF_E --low 0.1 --high 0.9 --steps 5 \
#         --set N_ROWS=30

from .geometry import generate_circle_geometry, build_spatial_grid
from .kernels import simulate_batch_cuda, require_cuda
from .sweep import DEFAULT_PARAMS, PARAM_NAMES, run_frame, run_sweep

__all__ = [
    "DEFAULT_PARAMS",
    "PARAM_NAMES",
    "run_sweep",
    "run_frame",
    "simulate_batch_cuda",
    "require_cuda",
    "generate_circle_geometry",
    "build_spatial_grid",
]
