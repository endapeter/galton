#!/usr/bin/env python3
"""
test_brownian_model.py

Test the Brownian-motion Galton-board model from
planning/markdown/PLAN_simple.md using the galton_cuda simulation package.

All default parameters are collected in the DEFAULTS dictionary just below
the imports: edit a value there for a permanent change, or override any of
them from the command line with the matching flag
(e.g. DEFAULTS["drop_range"] <-> --drop-range).

The paper predicts, for fixed release half-width a and negligible wall effects,

    sigma^2(H) = a^2 / 3 + 2 D_z H        (Equation 13)

Important wall/peg-width behavior:

- The user customizes the wall width with --wall-width.
- Peg rows are automatically widened to fill that wall width as densely as
  possible without moving pegs outside the walls.
- The user no longer separately customizes first-row peg count, wall padding,
  edge margin, or minimum wall distance.
- The simulation physics, CUDA kernel calls, restitution behavior, gravity,
  geometry spacing, peg radius, and Brownian-model fitting logic are otherwise
  unchanged.

Typical use, inside the CUDA container:

    docker run --rm --gpus all -v "C:/Users/Enda/Data/Code/galton:/work" -w /work galton-cuda python3 test_brownian_model.py --drop-range 1.0 --wall-width 40 --balls 5000 --min-rows 5 --max-rows 50 --row-points 10

The effective height H is, by default,

    H = N_rows * Delta_z

where Delta_z = spacing * sqrt(3) / 2 is the vertical row spacing of the
triangular peg lattice. This excludes constant non-peg fall distances such as
h_init, which helps the intercept correspond to the initial release variance
a^2 / 3.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


# =====================================================================
# DEFAULT PARAMETERS - the one place to change the experiment.
#
# Every entry can be overridden from the command line with the matching
# flag (drop_range <-> --drop-range, row_points <-> --row-points, ...).
# The model being tested is PLAN_simple.md Eq. (13):
#
#     sigma^2 = a^2/3 + 2 D_z H
#
# =====================================================================
DEFAULTS = {
    # --- release funnel --------------------------------------------------
    "drop_range": 1.0,        # a: release half-width; initial x ~ U[-a, a].
                              # Sets the predicted intercept a^2/3.
    "balls": 5000,            # Balls simulated per board height.

    # --- height sweep ----------------------------------------------------
    "min_rows": 5,            # Smallest peg-row count tested.
    "max_rows": 50,           # Largest peg-row count tested.
    "row_points": 8,          # Number of heights between min and max rows.
    "height_mode": "rows",    # 'rows'      H = N_rows * Delta_z  (recommended,
                              #             matches PLAN_simple.md §4)
                              # 'total'     H = h_init + N*Delta_z + h_final
                              # 'peg-span'  H = (N_rows - 1) * Delta_z

    # --- board geometry --------------------------------------------------
    "spacing": 2.0,           # d: peg center-to-center spacing.
    "radius": 0.5,            # Peg radius.
    "h_init": 1.0,            # Free-fall distance from release line to row 1.
    "h_final": 0.0,           # Free-fall distance from last row to detection.

    # --- walls -----------------------------------------------------------
    # wall_width: None        # Full wall-to-wall distance; None = automatic.
                              # Peg rows autofill this width as densely as
                              # spacing/radius allow. The model needs
                              # k*sigma << b (PLAN_simple.md §6), checked via
                              # wall_check_sigma below.

    # --- simulation ------------------------------------------------------
    "restitution": 0.5,       # e: coefficient of restitution (pegs & walls).
    "g": 9.81,                # Gravitational acceleration.
    "cell_size": 0.0,         # Spatial-grid cell size; <= 0 = automatic
                              # (max(spacing, 2*radius)).
    "sampling": "random",     # 'random': uniform RNG; 'grid': deterministic
                              # midpoints (lower initial-sampling noise).
    "seed": 12345,            # RNG seed for 'random' sampling.

    # --- output ----------------------------------------------------------
    "output": "brownian_test",  # Output directory for CSV/JSON/plots.
    # no_plot: False            # Set True to skip matplotlib plots.

    # --- acceptance thresholds ---------------------------------------------
    "r2_threshold": 0.95,         # Minimum R^2 of the free sigma^2-vs-H fit.
    "intercept_tolerance": 0.25,  # Max |intercept - a^2/3| / (a^2/3), i.e.
                                  # 0.25 = 25%.
    "wall_check_sigma": 3.0,      # Wall criterion k*sigma < b (§6 uses k = 3).
}


def _add_package_path():
    """Add a likely location of the galton_cuda package to sys.path."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "galton_cuda").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return True
    return False


try:
    from galton_cuda import (
        require_cuda,
        generate_circle_geometry,
        build_spatial_grid,
        simulate_batch_cuda,
    )
except ImportError:
    _add_package_path()
    try:
        from galton_cuda import (
            require_cuda,
            generate_circle_geometry,
            build_spatial_grid,
            simulate_batch_cuda,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import galton_cuda.\n"
            "Run this script from the directory containing the galton_cuda folder,\n"
            "or set PYTHONPATH to the parent directory of galton_cuda."
        ) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Test the Brownian Galton-board model "
            "sigma^2 = a^2/3 + 2 D_z H using the galton_cuda GPU simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--drop-range",
        type=float,
        default=DEFAULTS["drop_range"],
        help="Release half-width a. Initial x positions are uniform in [-a, a].",
    )
    parser.add_argument(
        "--balls",
        type=int,
        default=DEFAULTS["balls"],
        help="Number of balls to simulate for each height.",
    )

    height_group = parser.add_argument_group("height sweep")
    height_group.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit list of peg-row counts to test, e.g. --rows 5 10 15 20. "
            "If omitted, a linear range from --min-rows to --max-rows is used."
        ),
    )
    height_group.add_argument("--min-rows", type=int, default=DEFAULTS["min_rows"])
    height_group.add_argument("--max-rows", type=int, default=DEFAULTS["max_rows"])
    height_group.add_argument(
        "--row-points",
        type=int,
        default=DEFAULTS["row_points"],
        help="Number of distinct row counts to use when --rows is not given.",
    )
    height_group.add_argument(
        "--height-mode",
        choices=["rows", "total", "peg-span"],
        default=DEFAULTS["height_mode"],
        help=(
            "How to define H. "
            "'rows': H = N_rows * Delta_z, recommended for comparison with "
            "the discrete row model. "
            "'total': H = h_init + N_rows * Delta_z + h_final, the full fall distance. "
            "'peg-span': H = (N_rows - 1) * Delta_z, distance between first and last peg row."
        ),
    )

    geom_group = parser.add_argument_group("board geometry")
    geom_group.add_argument(
        "--spacing",
        type=float,
        default=DEFAULTS["spacing"],
        help="Center-to-center spacing d of neighboring pegs.",
    )
    geom_group.add_argument(
        "--radius",
        type=float,
        default=DEFAULTS["radius"],
        help="Peg radius used by the collision kernel.",
    )
    geom_group.add_argument(
        "--h-init",
        type=float,
        default=DEFAULTS["h_init"],
        help="Vertical distance from the release line y=0 to the first peg row.",
    )
    geom_group.add_argument(
        "--h-final",
        type=float,
        default=DEFAULTS["h_final"],
        help="Extra vertical distance after the last peg row before the detection line.",
    )

    wall_group = parser.add_argument_group("walls")
    wall_group.add_argument(
        "--wall-width",
        type=float,
        default=None,
        help=(
            "Full horizontal distance between the left and right walls. "
            "If omitted, an automatic width is chosen from the drop range and spacing. "
            "Peg counts are computed automatically from this width."
        ),
    )
    wall_group.add_argument(
        "--wall-check-sigma",
        type=float,
        default=DEFAULTS["wall_check_sigma"],
        help="Warn about wall effects if this many final sigmas approach the wall.",
    )

    sim_group = parser.add_argument_group("simulation")
    sim_group.add_argument(
        "--restitution",
        type=float,
        default=DEFAULTS["restitution"],
        help="Coefficient of restitution e used in peg and wall reflections.",
    )
    sim_group.add_argument("--g", type=float, default=DEFAULTS["g"], help="Gravity.")
    sim_group.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULTS["cell_size"],
        help="Spatial-grid cell size. If <= 0, automatically set to max(spacing, 2*radius).",
    )
    sim_group.add_argument(
        "--sampling",
        choices=["random", "grid"],
        default=DEFAULTS["sampling"],
        help=(
            "How to choose initial x positions. "
            "'random': uniform random sampling. "
            "'grid': evenly spaced midpoint sampling, which reduces initial sampling noise."
        ),
    )
    sim_group.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output",
        default=DEFAULTS["output"],
        help="Output directory for CSV, JSON, and PNG/PDF results.",
    )
    out_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create matplotlib plots.",
    )

    check_group = parser.add_argument_group("acceptance thresholds")
    check_group.add_argument(
        "--r2-threshold",
        type=float,
        default=DEFAULTS["r2_threshold"],
        help="Minimum R^2 of the free sigma^2 vs H linear fit for a PASS.",
    )
    check_group.add_argument(
        "--intercept-tolerance",
        type=float,
        default=DEFAULTS["intercept_tolerance"],
        help=(
            "Allowed relative difference between the free-fit intercept and a^2/3. "
            "Example: 0.25 means 25%%."
        ),
    )

    args = parser.parse_args(argv)

    if args.balls < 2:
        parser.error("--balls must be at least 2.")
    if args.drop_range < 0.0:
        parser.error("--drop-range must be non-negative.")
    if args.spacing <= 0.0:
        parser.error("--spacing must be positive.")
    if args.radius <= 0.0:
        parser.error("--radius must be positive.")
    if args.h_init < 0.0:
        parser.error("--h-init must be non-negative.")
    if args.h_final < 0.0:
        parser.error("--h-final must be non-negative.")
    if args.g <= 0.0:
        parser.error("--g must be positive.")
    if args.restitution < 0.0:
        parser.error("--restitution must be non-negative.")
    if args.wall_check_sigma <= 0.0:
        parser.error("--wall-check-sigma must be positive.")
    if args.row_points < 1:
        parser.error("--row-points must be at least 1.")
    if args.r2_threshold < 0.0:
        parser.error("--r2-threshold must be non-negative.")
    if args.intercept_tolerance < 0.0:
        parser.error("--intercept-tolerance must be non-negative.")
    if args.cell_size < 0.0:
        parser.error("--cell-size must be non-negative.")

    if args.rows:
        if any(r < 1 for r in args.rows):
            parser.error("All --rows values must be positive integers.")
    else:
        if args.min_rows < 1 or args.max_rows < 1:
            parser.error("--min-rows and --max-rows must be positive.")

    if args.wall_width is not None and args.wall_width <= 0.0:
        parser.error("--wall-width must be positive.")

    return args


def make_rows(args):
    """Build the list of peg-row counts to test."""
    if args.rows:
        rows = sorted({int(r) for r in args.rows if r > 0})
    else:
        lo = int(args.min_rows)
        hi = int(args.max_rows)
        if lo > hi:
            lo, hi = hi, lo

        if args.row_points == 1:
            values = [lo]
        else:
            values = np.linspace(lo, hi, args.row_points)

        rows = sorted({max(1, int(round(float(v)))) for v in values})

    if len(rows) < 2:
        print(
            "[warn] Fewer than two distinct row counts selected. "
            "A linear fit will not be possible."
        )

    return rows


def make_x_inits(n_balls, drop_range, sampling, seed):
    """Initial horizontal positions, uniformly distributed in [-a, a]."""
    if drop_range == 0.0:
        return np.zeros(n_balls, dtype=np.float64)

    if sampling == "grid":
        step = 2.0 * drop_range / float(n_balls)
        return (-drop_range + (np.arange(n_balls) + 0.5) * step).astype(np.float64)

    rng = np.random.default_rng(seed)
    return rng.uniform(-drop_range, drop_range, size=n_balls).astype(np.float64)


def default_wall_width(drop_range, spacing):
    """
    Pick a conservative automatic full wall width when --wall-width is omitted.

    The user may override this with --wall-width. Peg count is always derived
    from the resulting wall width, so the peg bed autofills the requested walls
    as densely as the requested spacing and radius allow.
    """
    margin = max(4.0 * spacing, 2.0 * drop_range)
    return 2.0 * (drop_range + 2.0 * margin)


def peg_edge_distance(first_row, spacing, radius):
    """Half-width occupied by the outer edge of the widest peg row."""
    return ((first_row - 1) * spacing / 2.0) + radius


def pegs_for_wall_distance(wall_distance, spacing, radius):
    """
    Compute the widest first-row peg count that fits inside +/- wall_distance.

    For a row with n pegs, the outermost center is at

        +/- ((n - 1) * spacing / 2)

    and the outermost peg edge is therefore at

        ((n - 1) * spacing / 2) + radius.

    We choose the largest n such that this edge does not exceed the wall.
    This makes the peg bed auto-fill the requested wall width as densely as the
    requested spacing and radius allow, without changing the physics constants.
    """
    if wall_distance <= radius:
        raise ValueError(
            f"Wall half-distance {wall_distance:g} must exceed peg radius {radius:g}."
        )

    n_float = 2.0 * (wall_distance - radius) / spacing + 1.0
    n = int(math.floor(n_float + 1e-9))

    if n < 2:
        raise ValueError(
            "Requested wall width is too small to fit at least two first-row pegs "
            f"with spacing {spacing:g} and radius {radius:g}."
        )

    occupied = peg_edge_distance(n, spacing, radius)
    if occupied > wall_distance + 1e-9:
        raise ValueError(
            "Internal error: computed peg count does not fit inside the requested walls."
        )

    return n


def effective_height(n_rows, args, delta_z):
    """Effective board height H used for the Brownian-model fit."""
    if args.height_mode == "total":
        return float(args.h_init + n_rows * delta_z + args.h_final)

    if args.height_mode == "peg-span":
        return float(max(0.0, (n_rows - 1) * delta_z))

    return float(n_rows * delta_z)


def simulate_one_height(
    n_rows,
    args,
    x_inits,
    first_row,
    wall_distance,
    cell_size,
):
    """Run one GPU simulation for a given number of peg rows."""
    circles, geometry_wall_distance = generate_circle_geometry(
        first_row,
        n_rows,
        args.spacing,
        args.h_init,
        args.radius,
        WALL_PADDING=0.0,
    )

    centers = np.array([c["center"] for c in circles], dtype=np.float64)

    occupied_half_width = float(geometry_wall_distance)
    if occupied_half_width > wall_distance + 1e-9:
        raise RuntimeError(
            "Internal wall/peg-width inconsistency: generated pegs extend "
            f"to {occupied_half_width:g}, beyond requested wall half-distance "
            f"{wall_distance:g}."
        )

    if args.drop_range >= wall_distance:
        raise ValueError(
            f"Drop range a={args.drop_range:g} is not inside the walls at "
            f"+/-{wall_distance:g}. Increase --wall-width or reduce --drop-range."
        )

    grid_data, grid_counts, min_cx, min_cy = build_spatial_grid(
        centers,
        args.radius,
        cell_size,
    )

    if np.max(grid_counts) >= 20:
        print(
            f"[warn] rows={n_rows}: a spatial-grid cell reached the hard-coded "
            "maximum of 20 pegs. Consider increasing --cell-size."
        )

    final_x, status = simulate_batch_cuda(
        centers,
        grid_data,
        grid_counts,
        min_cx,
        min_cy,
        cell_size,
        args.radius,
        x_inits,
        args.restitution,
        args.h_init,
        n_rows,
        args.spacing,
        args.h_final,
        wall_distance,
        args.g,
    )

    return final_x, status


def fit_model(H, sigma2, theory_intercept, sample_initial_variance):
    """
    Fit the paper model.

    Free fit:
        sigma^2 = intercept + slope * H

    Fixed-intercept fits:
        sigma^2 = a^2/3 + slope * H
        sigma^2 = sample_initial_variance + slope * H
    """
    fit = {"n_points": int(len(H))}

    if len(H) < 2:
        return fit
    if not np.any(H != H[0]):
        return fit

    slope_free, intercept_free = map(float, np.polyfit(H, sigma2, 1))
    pred_free = intercept_free + slope_free * H

    ss_res = float(np.sum((sigma2 - pred_free) ** 2))
    ss_tot = float(np.sum((sigma2 - np.mean(sigma2)) ** 2))
    r2_free = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    fit.update(
        slope_free=slope_free,
        intercept_free=intercept_free,
        Dz_free=0.5 * slope_free,
        r2_free=r2_free,
        rmse_sigma2_free=float(np.sqrt(np.mean((sigma2 - pred_free) ** 2))),
    )

    if theory_intercept != 0.0 and math.isfinite(intercept_free):
        fit["intercept_relative_error_free_vs_theory"] = (
            intercept_free - theory_intercept
        ) / theory_intercept
    else:
        fit["intercept_relative_error_free_vs_theory"] = float("nan")

    def fixed_slope(c):
        denom = float(np.dot(H, H))
        if denom == 0.0 or not math.isfinite(c):
            return float("nan")
        return float(np.dot(H, sigma2 - c) / denom)

    def add_fixed_intercept_metrics(tag, intercept_value):
        slope = fixed_slope(intercept_value)
        pred = intercept_value + slope * H
        finite = np.isfinite(pred)

        if np.any(finite):
            rmse_var = float(np.sqrt(np.mean((sigma2[finite] - pred[finite]) ** 2)))
            mean_var = float(np.mean(sigma2[finite]))
            rel_rmse_var = rmse_var / mean_var if mean_var > 0.0 else float("nan")

            sigma_obs = np.sqrt(np.maximum(sigma2[finite], 0.0))
            sigma_pred = np.sqrt(np.maximum(pred[finite], 0.0))
            rmse_sigma = float(np.sqrt(np.mean((sigma_obs - sigma_pred) ** 2)))
        else:
            rmse_var = float("nan")
            rel_rmse_var = float("nan")
            rmse_sigma = float("nan")

        fit[f"slope_fixed_{tag}"] = slope
        fit[f"Dz_fixed_{tag}"] = 0.5 * slope
        fit[f"rmse_sigma2_fixed_{tag}"] = rmse_var
        fit[f"rel_rmse_sigma2_fixed_{tag}"] = rel_rmse_var
        fit[f"rmse_sigma_fixed_{tag}"] = rmse_sigma

    add_fixed_intercept_metrics("theory", theory_intercept)
    add_fixed_intercept_metrics("sample", sample_initial_variance)

    return fit


def write_csv(path, records):
    fieldnames = [
        "rows",
        "H",
        "mean_x",
        "sigma",
        "sigma_stderr",
        "sigma2",
        "n_valid",
        "n_total",
        "valid_fraction",
        "wall_distance",
        "wall_width",
        "first_row",
        "peg_edge_distance",
        "peg_wall_gap",
        "wall_ok",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def make_multi_panel_plot(out_dir, records, fit, theory_intercept, args):
    """
    Creates a multi-panel diagnostic plot showing:

    1. Prediction line overlay: data vs theoretical Brownian model.
    2. Variance In vs Variance Out as a function of funnel width.
    3a. Influence of H on variance as a function of funnel width.
    3b. Influence of D_z on variance as a function of funnel width.

    Saves both PNG and PDF formats.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped because matplotlib is unavailable: {exc}")
        return None, None

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    H_data = np.array([r["H"] for r in records], dtype=np.float64)
    sigma2_data = np.array([r["sigma2"] for r in records], dtype=np.float64)

    a = float(args.drop_range)

    Dz = fit.get("Dz_free")
    if Dz is None or not math.isfinite(Dz) or Dz <= 0.0:
        Dz = fit.get("Dz_fixed_theory")
    if Dz is None or not math.isfinite(Dz) or Dz <= 0.0:
        Dz = 0.1  # last-resort placeholder so the model curves still render

    H_ref = float(np.mean(H_data)) if H_data.size > 0 else 10.0
    if H_ref <= 0.0:
        H_ref = 1.0

    theory_var = float(theory_intercept)

    # Plot 1: sigma^2 vs H
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        H_data,
        sigma2_data,
        color="tab:blue",
        label="Data-informed actual variance",
        zorder=3,
        s=60,
        edgecolor="black",
    )

    h_min = float(H_data.min()) if H_data.size > 0 else 0.1
    h_max = float(H_data.max()) if H_data.size > 0 else 10.0
    h_lo = max(0.1, h_min)
    h_hi = max(h_lo * 1.1, h_max * 1.1)
    H_fit = np.linspace(h_lo, h_hi, 100)

    ax1.plot(
        H_fit,
        theory_var + 2.0 * Dz * H_fit,
        color="tab:red",
        label=r"Model line: $a^2/3 + 2\,D_z H$ (fitted $D_z$)",
        lw=2,
    )
    ax1.axhline(
        theory_var,
        color="tab:green",
        linestyle="--",
        label="Uniform reference input variance ($a^2/3$)",
        lw=2,
    )
    ax1.set_xlabel("Effective Height ($H$)", fontsize=12)
    ax1.set_ylabel(r"Variance ($\sigma^2$)", fontsize=12)
    ax1.set_title("1. Variance vs Height (Prediction vs Data)", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Variance In vs Variance Out vs funnel width
    ax2 = fig.add_subplot(gs[0, 1])
    W_max = max(4.0 * a, 4.0)
    W = np.linspace(0.1, W_max, 100)
    var_in = (W**2) / 12.0
    var_out_theory = var_in + 2.0 * Dz * H_ref

    ax2.plot(
        W,
        var_in,
        color="tab:green",
        linestyle="--",
        label=r"Variance In ($W^2/12$)",
        lw=2,
    )
    ax2.plot(
        W,
        var_out_theory,
        color="tab:red",
        label="Variance Out (Theoretical)",
        lw=2,
    )

    if H_data.size > 0:
        ax2.scatter(
            [2.0 * a],
            [theory_var],
            color="tab:green",
            edgecolor="black",
            zorder=4,
            s=80,
            label="Simulated Variance In",
        )
        ax2.scatter(
            [2.0 * a],
            [theory_var + 2.0 * Dz * H_ref],
            color="tab:red",
            edgecolor="black",
            zorder=4,
            s=80,
            label="Variance Out (model, at mean H)",
        )

    ax2.set_xlabel(r"Funnel Width ($W = 2a$)", fontsize=12)
    ax2.set_ylabel("Variance", fontsize=12)
    ax2.set_title("2. Variance In vs Out as Function of Funnel Width", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    baseline_var = (W**2) / 12.0

    # Plot 3a: Influence of H on variance vs funnel width
    ax3 = fig.add_subplot(gs[1, 0])
    H_levels = [H_ref * 0.5, H_ref, H_ref * 1.5, H_ref * 2.0]
    colors_H = plt.cm.viridis(np.linspace(0.2, 0.8, len(H_levels)))

    for i, h_val in enumerate(H_levels):
        var_vs_W = (W**2) / 12.0 + 2.0 * Dz * h_val
        ax3.plot(W, var_vs_W, color=colors_H[i], lw=2, label=f"$H = {h_val:.1f}$")

    ax3.plot(
        W,
        baseline_var,
        "k--",
        alpha=0.6,
        lw=2,
        label=r"Influence of $a$ only ($W^2/12$)",
    )
    ax3.set_xlabel(r"Funnel Width ($W = 2a$)", fontsize=12)
    ax3.set_ylabel(r"Variance ($\sigma^2$)", fontsize=12)
    ax3.set_title(r"3a. Influence of $H$ on Variance vs Funnel Width", fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 3b: Influence of D_z on variance vs funnel width
    ax4 = fig.add_subplot(gs[1, 1])
    D_levels = [Dz * 0.5, Dz, Dz * 1.5, Dz * 2.0]
    colors_D = plt.cm.plasma(np.linspace(0.2, 0.8, len(D_levels)))

    for i, d_val in enumerate(D_levels):
        var_vs_W = (W**2) / 12.0 + 2.0 * d_val * H_ref
        ax4.plot(W, var_vs_W, color=colors_D[i], lw=2, label=f"$D_z = {d_val:.2f}$")

    ax4.plot(
        W,
        baseline_var,
        "k--",
        alpha=0.6,
        lw=2,
        label=r"Influence of $a$ only ($W^2/12$)",
    )
    ax4.set_xlabel(r"Funnel Width ($W = 2a$)", fontsize=12)
    ax4.set_ylabel(r"Variance ($\sigma^2$)", fontsize=12)
    ax4.set_title(r"3b. Influence of $D_z$ on Variance vs Funnel Width", fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    png_path = out_dir / "brownian_model_multi_panel.png"
    pdf_path = out_dir / "brownian_model_multi_panel.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def fmt(x, spec=".6g"):
    try:
        if x is None or not math.isfinite(x):
            return "nan"
        return format(x, spec)
    except Exception:
        return str(x)


def print_report(args, records, fit, theory_var, sample_var, delta_z):
    first = records[0]

    print("\n=== Brownian Galton-board model test ===")
    print(f"Release half-width a                    : {fmt(args.drop_range)}")
    print(f"Theoretical initial variance a^2/3      : {fmt(theory_var)}")
    print(f"Sample initial variance                 : {fmt(sample_var)}")
    print(f"Vertical row spacing Delta_z            : {fmt(delta_z)}")
    print(f"Height mode                             : {args.height_mode}")
    print(f"Restitution e                           : {fmt(args.restitution)}")
    print(f"Balls per height                        : {args.balls}")
    print(f"Wall width                              : {fmt(first['wall_width'])}")
    print(f"Wall half-distance                      : {fmt(first['wall_distance'])}")
    print(f"Auto first-row peg count                : {first['first_row']}")
    print(f"Peg outer-edge half-width               : {fmt(first['peg_edge_distance'])}")
    print(f"Gap from outer peg edge to wall         : {fmt(first['peg_wall_gap'])}")

    print("\nMeasured results:")
    header = (
        f"{'rows':>6} {'H':>12} {'sigma':>12} {'± se':>10} {'sigma^2':>12} "
        f"{'mean x':>12} {'valid':>14} {'wall ok':>8}"
    )
    print(header)
    print("-" * len(header))

    for rec in records:
        valid_str = f"{rec['n_valid']}/{rec['n_total']}"
        print(
            f"{rec['rows']:6d} "
            f"{rec['H']:12.6g} "
            f"{rec['sigma']:12.6g} "
            f"{fmt(rec.get('sigma_stderr')):>10} "
            f"{rec['sigma2']:12.6g} "
            f"{rec['mean_x']:12.6g} "
            f"{valid_str:>14} "
            f"{str(rec['wall_ok']):>8}"
        )

    if fit.get("n_points", 0) < 2 or "slope_free" not in fit:
        print("\nNot enough distinct height points for a linear fit.")
        return

    print("\nFree linear fit: sigma^2 = intercept + slope * H")
    print(f"  intercept                  : {fmt(fit.get('intercept_free'))}")
    print(f"  slope                      : {fmt(fit.get('slope_free'))}")
    print(f"  D_z = slope / 2            : {fmt(fit.get('Dz_free'))}")
    print(f"  R^2                        : {fmt(fit.get('r2_free'))}")
    print(f"  RMSE in sigma^2            : {fmt(fit.get('rmse_sigma2_free'))}")
    print(
        "  intercept relative error   : "
        f"{fmt(fit.get('intercept_relative_error_free_vs_theory'))}"
    )

    print("\nFixed-intercept fit using theoretical a^2/3:")
    print(f"  slope                      : {fmt(fit.get('slope_fixed_theory'))}")
    print(f"  D_z = slope / 2            : {fmt(fit.get('Dz_fixed_theory'))}")
    print(f"  RMSE in sigma^2            : {fmt(fit.get('rmse_sigma2_fixed_theory'))}")
    print(
        "  relative RMSE in sigma^2   : "
        f"{fmt(fit.get('rel_rmse_sigma2_fixed_theory'))}"
    )
    print(f"  RMSE in sigma              : {fmt(fit.get('rmse_sigma_fixed_theory'))}")

    print("\nFixed-intercept fit using empirical sample initial variance:")
    print(f"  slope                      : {fmt(fit.get('slope_fixed_sample'))}")
    print(f"  D_z = slope / 2            : {fmt(fit.get('Dz_fixed_sample'))}")
    print(f"  RMSE in sigma^2            : {fmt(fit.get('rmse_sigma2_fixed_sample'))}")
    print(
        "  relative RMSE in sigma^2   : "
        f"{fmt(fit.get('rel_rmse_sigma2_fixed_sample'))}"
    )
    print(f"  RMSE in sigma              : {fmt(fit.get('rmse_sigma_fixed_sample'))}")


def assess_model(fit, records, args, theory_var):
    issues = []

    if fit.get("n_points", 0) < 2 or "slope_free" not in fit:
        issues.append("not enough height points for a linear fit")
    else:
        r2 = fit.get("r2_free")
        if r2 is None or not math.isfinite(r2):
            issues.append("could not compute R^2")
        elif r2 < args.r2_threshold:
            issues.append(
                f"free-fit R^2 = {fmt(r2)} is below threshold {fmt(args.r2_threshold)}"
            )

        slope_theory = fit.get("slope_fixed_theory")
        if slope_theory is None or not math.isfinite(slope_theory):
            issues.append("could not compute fixed-intercept diffusion slope")
        elif slope_theory <= 0.0:
            issues.append(
                f"fixed-intercept slope = {fmt(slope_theory)} is non-positive"
            )

        if theory_var > 0.0:
            rel_err = fit.get("intercept_relative_error_free_vs_theory")
            if rel_err is not None and math.isfinite(rel_err):
                if abs(rel_err) > args.intercept_tolerance:
                    issues.append(
                        "free-fit intercept differs from a^2/3 by "
                        f"{fmt(abs(rel_err) * 100.0)}%, exceeding tolerance "
                        f"{fmt(args.intercept_tolerance * 100.0)}%"
                    )

    bad_wall = sum(1 for rec in records if not rec["wall_ok"])
    if bad_wall:
        issues.append(
            f"{bad_wall} height(s) failed the wall criterion "
            f"{fmt(args.wall_check_sigma)} * sigma < wall_distance"
        )

    low_valid = sum(1 for rec in records if rec["valid_fraction"] < 0.9)
    if low_valid:
        issues.append(
            f"{low_valid} height(s) had fewer than 90% of balls reaching the detection line"
        )

    print("\nModel consistency assessment:")
    if issues:
        print("  CONCERNS: the simulation data do not clearly satisfy the model.")
        for issue in issues:
            print(f"   - {issue}")
        print(
            "\nPossible remedies: increase --wall-width, reduce heights, "
            "increase --balls, adjust restitution, or check for systematic asymmetry."
        )
    else:
        print(
            "  PASS: within the selected thresholds, the data are consistent with "
            "sigma^2 = a^2/3 + 2 D_z H."
        )


def main(argv=None):
    args = parse_args(argv)

    require_cuda()

    rows_list = make_rows(args)

    a = float(args.drop_range)
    n_balls = int(args.balls)

    if args.restitution > 1.0:
        print("[warn] restitution > 1 injects energy and may cause unstable trajectories.")

    wall_width = (
        float(args.wall_width)
        if args.wall_width is not None
        else default_wall_width(a, args.spacing)
    )
    wall_distance = wall_width / 2.0

    if a >= wall_distance:
        raise SystemExit(
            f"Drop range a={a:g} is not inside the walls at +/-{wall_distance:g}. "
            "Increase --wall-width or reduce --drop-range."
        )

    try:
        first_row = pegs_for_wall_distance(
            wall_distance=wall_distance,
            spacing=args.spacing,
            radius=args.radius,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    occupied_half_width = peg_edge_distance(first_row, args.spacing, args.radius)
    peg_wall_gap = wall_distance - occupied_half_width

    if first_row > 5000:
        raise SystemExit(
            f"Auto-selected first row count is very large ({first_row}). "
            "Increase --spacing, reduce --wall-width, or use a larger peg radius."
        )

    cell_size = (
        float(args.cell_size)
        if args.cell_size > 0.0
        else max(args.spacing, 2.0 * args.radius, 1e-6)
    )

    delta_z = args.spacing * math.sqrt(3.0) / 2.0

    x_inits = make_x_inits(n_balls, a, args.sampling, args.seed)
    x_inits = np.ascontiguousarray(x_inits, dtype=np.float64)

    theory_initial_variance = a * a / 3.0
    sample_initial_variance = (
        float(np.var(x_inits, ddof=1)) if x_inits.size > 1 else 0.0
    )

    print("Brownian-model test configuration:")
    print(f"  wall width                       : {fmt(wall_width)}")
    print(f"  wall half-distance               : {fmt(wall_distance)}")
    print(f"  auto first-row peg count         : {first_row}")
    print(f"  peg outer-edge half-width        : {fmt(occupied_half_width)}")
    print(f"  gap from outer peg edge to wall  : {fmt(peg_wall_gap)}")
    print(f"  row spacing Delta_z              : {fmt(delta_z)}")
    print(f"  spatial-grid cell size           : {fmt(cell_size)}")
    print(f"  initial sample variance          : {fmt(sample_initial_variance)}")
    print(f"  theoretical a^2/3                : {fmt(theory_initial_variance)}")
    print()

    records = []

    for n_rows in rows_list:
        final_x, status = simulate_one_height(
            n_rows=n_rows,
            args=args,
            x_inits=x_inits,
            first_row=first_row,
            wall_distance=wall_distance,
            cell_size=cell_size,
        )

        valid = (status == 1) & np.isfinite(final_x)
        n_valid = int(np.count_nonzero(valid))

        if n_valid < 2:
            print(f"[warn] rows={n_rows}: fewer than two valid trajectories; skipping.")
            continue

        x_final = final_x[valid]

        mean_x = float(np.mean(x_final))
        sigma = float(np.std(x_final, ddof=1))
        sigma2 = sigma * sigma
        # Standard error of the sample std-dev, se(s) ~ s / sqrt(2(n - 1))
        # (leading-order Gaussian approximation). Gives the statistical
        # noise floor against which the fit residuals should be judged.
        sigma_stderr = sigma / math.sqrt(2.0 * (n_valid - 1))
        H = effective_height(n_rows, args, delta_z)

        wall_ok = bool(args.wall_check_sigma * sigma < wall_distance)

        rec = {
            "rows": int(n_rows),
            "H": float(H),
            "mean_x": mean_x,
            "sigma": float(sigma),
            "sigma_stderr": float(sigma_stderr),
            "sigma2": float(sigma2),
            "n_valid": n_valid,
            "n_total": int(n_balls),
            "valid_fraction": float(n_valid) / float(n_balls),
            "wall_distance": float(wall_distance),
            "wall_width": float(wall_width),
            "first_row": int(first_row),
            "peg_edge_distance": float(occupied_half_width),
            "peg_wall_gap": float(peg_wall_gap),
            "wall_ok": wall_ok,
        }
        records.append(rec)

        print(
            f"rows={n_rows:5d}  H={H:12.6g}  sigma={sigma:12.6g}  "
            f"sigma^2={sigma2:12.6g}  valid={n_valid}/{n_balls}  "
            f"wall_ok={wall_ok}"
        )

    if not records:
        raise SystemExit("No valid simulation results were produced.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "brownian_model_results.csv"
    write_csv(csv_path, records)

    H = np.array([rec["H"] for rec in records], dtype=np.float64)
    sigma2 = np.array([rec["sigma2"] for rec in records], dtype=np.float64)

    fit = fit_model(
        H=H,
        sigma2=sigma2,
        theory_intercept=theory_initial_variance,
        sample_initial_variance=sample_initial_variance,
    )

    payload = {
        "args": vars(args),
        "derived": {
            "a": a,
            "delta_z": delta_z,
            "wall_width": wall_width,
            "wall_distance": wall_distance,
            "first_row": first_row,
            "peg_edge_distance": occupied_half_width,
            "peg_wall_gap": peg_wall_gap,
            "cell_size": cell_size,
            "theory_initial_variance": theory_initial_variance,
            "sample_initial_variance": sample_initial_variance,
        },
        "records": records,
        "fit": fit,
    }

    json_path = out_dir / "brownian_model_summary.json"
    write_json(json_path, payload)

    png_path, pdf_path = (None, None)
    if not args.no_plot:
        png_path, pdf_path = make_multi_panel_plot(
            out_dir,
            records,
            fit,
            theory_initial_variance,
            args,
        )

    print_report(
        args=args,
        records=records,
        fit=fit,
        theory_var=theory_initial_variance,
        sample_var=sample_initial_variance,
        delta_z=delta_z,
    )

    assess_model(fit, records, args, theory_initial_variance)

    print("\nOutput files:")
    print(f"  CSV   : {csv_path}")
    print(f"  JSON  : {json_path}")
    if png_path is not None:
        print(f"  Plot (PNG) : {png_path}")
    if pdf_path is not None:
        print(f"  Plot (PDF) : {pdf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())