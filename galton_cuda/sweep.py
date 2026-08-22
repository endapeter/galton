# Parameter-sweep controller for the CUDA Galton board.
#
# The public entry point is run_sweep(): it takes a request describing which
# parameter to sweep, which parameters to hold constant (overriding the
# defaults), and how many iterative steps (sweep frames) to take, then runs
# one GPU batch per step and writes the histograms / Q-Q plots /
# verification walks / CSVs to the output directory.

import csv
import os
import random

import numpy as np

from .geometry import generate_circle_geometry, build_spatial_grid
from .kernels import simulate_batch_cuda, require_cuda
from .visualization import (simulate_and_visualize_trajectory_optimized,
                            generate_qq_plot)
import matplotlib.pyplot as plt  # after .visualization: Agg backend is set there

# Base parameters (identical to galton_ultra.py). Any of these can be held
# constant via `fixed=` or swept via `sweep_param=` in run_sweep().
DEFAULT_PARAMS = {
    'N_FIRST_ROW': 51,      # pegs in the first (widest) row
    'N_ROWS': 50,           # number of peg rows
    'DISTANCE': 2.0,        # peg spacing
    'H_INIT': 3.0,          # drop height above the first row
    'H_FINAL': 1.0,         # detection line offset below the last row
    'RADIUS': 0.7,          # peg radius
    'X_DROP_RANGE': (-0.05, 0.05),  # (min, max) initial x of the balls
    'WALL_PADDING': 0.5,    # gap between the outermost peg edge and the wall
    'COEFF_E': 0.1,         # coefficient of restitution
    'CELL_SIZE': 2.0,       # spatial hash cell size
    'G': 9.81,              # gravitational acceleration
}

PARAM_NAMES = frozenset(DEFAULT_PARAMS)

# Parameters that must stay integers (peg counts); whole floats passed for
# these (e.g. N_ROWS=10 parsed from the CLI) are coerced back to int.
_INT_PARAMS = frozenset(name for name, value in DEFAULT_PARAMS.items()
                        if isinstance(value, int))


def _check_param_names(names, context):
    unknown = [name for name in names if name not in PARAM_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} in {context}. "
            f"Valid parameters: {sorted(PARAM_NAMES)}")


# --- SINGLE FRAME: GPU batch + histogram + verification walk + Q-Q plot ---
def run_frame(params, generated_circles, circles_centers, grid_data, grid_counts,
              min_cx, min_cy, wall_distance, n_balls, n_bins, frame_index,
              param_name, param_val, csv_filename, output_dir="figures",
              x_sampler=None):
    """
    Executes one sweep frame: n_balls in a single GPU kernel launch, then the
    histogram, the single-ball CPU verification walk, and the Q-Q plot.
    Returns a dict summarizing the frame (counts, bin edges, success rate...).

    x_sampler: optional callable (n_balls, drop_min, drop_max) -> ndarray of
        initial x positions. None (default) keeps the Stage-3 uniform
        sampling x ~ U(drop_min, drop_max); e.g. the step-8 experiment passes
        a Gaussian sampler x ~ N(0, sigma_in) instead.
    """
    os.makedirs(output_dir, exist_ok=True)

    drop_min, drop_max = params['X_DROP_RANGE']
    if x_sampler is None:
        x_init_values = np.random.uniform(drop_min, drop_max, size=n_balls)
    else:
        x_init_values = np.asarray(x_sampler(n_balls, drop_min, drop_max),
                                   dtype=np.float64)

    final_x_positions = []
    final_positions_data = []

    print(f"\n--- Starting simulation for Frame {frame_index:03d} | {param_name} = {param_val:.4f} ---")
    print(f"Executing {n_balls} balls in parallel on the GPU...")

    final_x_arr, status_arr = simulate_batch_cuda(
        circles_centers, grid_data, grid_counts, min_cx, min_cy,
        params['CELL_SIZE'], params['RADIUS'], x_init_values, params['COEFF_E'],
        params['H_INIT'], params['N_ROWS'], params['DISTANCE'],
        params['H_FINAL'], wall_distance, g=params['G'])

    for ball_index in range(n_balls):
        if status_arr[ball_index] == 1:
            final_x_positions.append(final_x_arr[ball_index])
            final_positions_data.append({
                "ball_id": ball_index,
                "initial_x": x_init_values[ball_index],
                "final_x": final_x_arr[ball_index],
                "status": "success"
            })
        else:
            final_positions_data.append({
                "ball_id": ball_index,
                "initial_x": x_init_values[ball_index],
                "final_x": np.nan,
                "status": "failed_or_out_of_bounds"
            })

    # Write landing positions to CSV specific to this frame
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['ball_id', 'initial_x', 'final_x', 'status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_positions_data:
            writer.writerow(row)

    # Create Distribution Histogram
    counts, bin_edges = np.histogram(final_x_positions, bins=n_bins, range=(-wall_distance, wall_distance))

    if len(final_x_positions) > 0:
        plt.figure(figsize=(10, 6))
        bin_width = np.diff(bin_edges)
        plt.bar(bin_edges[:-1], counts, width=bin_width, color='#3498db', edgecolor='#2980b9', alpha=0.8, align='edge',
                label=f'Balls Reached Line ({len(final_x_positions)}/{n_balls})')
        plt.xlim(-wall_distance, wall_distance)
        plt.axvline(x=0, color='#e74c3c', linestyle=':', alpha=0.7, label='Centerline')

        # Add parameter information directly on the plot title
        plt.title(f'Galton Board Final Ball Distribution ({n_bins} Bins) [CUDA]\n{param_name} = {param_val:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate at Detection Line', fontsize=12)
        plt.ylabel('Ball Count', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save sequentially for animation
        hist_save_path = os.path.join(output_dir, f"galton_distribution_{frame_index:03d}.png")
        plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Histogram for frame {frame_index:03d} saved to {hist_save_path}")

    # High-fidelity physics check for one random ball in this batch (CPU)
    random_verify_idx = random.randint(0, n_balls - 1)
    random_x_init = x_init_values[random_verify_idx]

    simulate_and_visualize_trajectory_optimized(
        circles_data=generated_circles,
        circles_centers=circles_centers,
        grid_data=grid_data,
        grid_counts=grid_counts,
        min_cx=min_cx,
        min_cy=min_cy,
        cell_size=params['CELL_SIZE'],
        radius=params['RADIUS'],
        x_init=random_x_init,
        y_init=0.0,
        vx_init=0.0,
        vy_init=0.0,
        e=params['COEFF_E'],
        h_init=params['H_INIT'],
        rows=params['N_ROWS'],
        spacing=params['DISTANCE'],
        h_final=params['H_FINAL'],
        WALL_DISTANCE=wall_distance,
        frame_index=frame_index,
        g=params['G'],
        output_dir=output_dir
    )

    # Generate the Q-Q Plot for this batch using the saved CSV
    generate_qq_plot(csv_filename, frame_index, param_name, param_val, output_dir=output_dir)

    return {
        "frame_index": frame_index,
        "param_name": param_name,
        "param_val": param_val,
        "csv_filename": csv_filename,
        "n_balls": n_balls,
        "n_success": len(final_x_positions),
        "counts": counts,
        "bin_edges": bin_edges,
    }


# --- SWEEP CONTROLLER ---
def run_sweep(sweep_param, low, high, steps, fixed=None, n_balls=2000,
              n_bins=200, output_dir="figures", x_sampler=None):
    """
    Run a parameter sweep on the GPU.

    Args:
        sweep_param: name of the parameter to sweep (see DEFAULT_PARAMS, e.g.
            'X_DROP_RANGE', 'COEFF_E', 'RADIUS', ...). Sweeping X_DROP_RANGE
            varies the drop-funnel half-width: step value v -> range (-v, v).
        low, high: first and last values of the swept parameter
            (np.linspace(low, high, steps)).
        steps: number of iterative steps (sweep frames) to take.
        fixed: dict of parameters to hold constant for every frame, e.g.
            {'N_ROWS': 30, 'COEFF_E': 0.3}. Overrides DEFAULT_PARAMS; must
            not contain sweep_param. Parameters not mentioned stay at their
            DEFAULT_PARAMS values (also constant).
        n_balls: balls per frame (one GPU kernel launch per frame).
        n_bins: histogram bins.
        output_dir: directory for the PNGs and per-frame CSVs.
        x_sampler: optional callable (n_balls, drop_min, drop_max) -> ndarray
            overriding the Stage-3 uniform input sampling (see run_frame).

    Returns:
        List of per-frame result dicts (frame_index, param_val, n_success,
        counts, bin_edges, csv_filename, ...), one entry per step.

    Example:
        from galton_cuda import run_sweep
        results = run_sweep(
            sweep_param='COEFF_E', low=0.1, high=0.9, steps=5,
            fixed={'N_ROWS': 30, 'X_DROP_RANGE': (-0.05, 0.05)},
            n_balls=2000)
    """
    _check_param_names([sweep_param], "sweep_param")
    fixed = dict(fixed) if fixed else {}
    _check_param_names(fixed, "fixed")
    for name in _INT_PARAMS:
        value = fixed.get(name)
        if isinstance(value, float) and value.is_integer():
            fixed[name] = int(value)
    if sweep_param in fixed:
        raise ValueError(
            f"{sweep_param!r} is both swept and fixed - remove it from `fixed`.")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")

    require_cuda()

    base_params = dict(DEFAULT_PARAMS)
    base_params.update(fixed)

    sweep_values = np.linspace(low, high, steps)

    results = []
    for frame_idx, param_val in enumerate(sweep_values):
        frame_params = dict(base_params)

        # Sweeping X_DROP_RANGE means varying the funnel half-width: the step
        # value v becomes the symmetric range (-v, v).
        if sweep_param == 'X_DROP_RANGE':
            frame_params[sweep_param] = (-param_val, param_val)
        else:
            frame_params[sweep_param] = param_val

        # Regenerate geometry
        generated_circles, wall_distance = generate_circle_geometry(
            n_first_row=frame_params['N_FIRST_ROW'],
            n_rows=frame_params['N_ROWS'],
            d=frame_params['DISTANCE'],
            h_init=frame_params['H_INIT'],
            r_custom=frame_params['RADIUS'],
            WALL_PADDING=frame_params['WALL_PADDING']
        )

        # Rebuild format for device arrays
        circles_centers = np.array([c["center"] for c in generated_circles], dtype=np.float64)

        # Initialize the dense spatial grid
        grid_data, grid_counts, min_cx, min_cy = build_spatial_grid(
            circles_centers,
            radius=frame_params['RADIUS'],
            cell_size=frame_params['CELL_SIZE']
        )

        # Execute single batch frame on the GPU
        csv_filename = os.path.join(output_dir, f"galton_ball_positions_{frame_idx:03d}.csv")
        results.append(run_frame(
            params=frame_params,
            generated_circles=generated_circles,
            circles_centers=circles_centers,
            grid_data=grid_data,
            grid_counts=grid_counts,
            min_cx=min_cx,
            min_cy=min_cy,
            wall_distance=wall_distance,
            n_balls=n_balls,
            n_bins=n_bins,
            frame_index=frame_idx,
            param_name=sweep_param,
            param_val=param_val,
            csv_filename=csv_filename,
            output_dir=output_dir,
            x_sampler=x_sampler
        ))

    print("\nSweep Complete! Images are saved sequentially in the "
          f"'{output_dir}' directory.")
    return results
