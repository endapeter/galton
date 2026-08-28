#!/usr/bin/env python3
"""
test_brownian_model.py

Test the Brownian-motion Galton-board model from galton_board_brownian_note.pdf
using the galton_cuda simulation package.

The paper predicts, for fixed release half-width a and negligible wall effects,

    sigma^2(H) = a^2 / 3 + 2 D_z H        (Equation 13)

This script:
  1. Generates Galton boards with different numbers of peg rows.
  2. Uses galton_cuda to simulate many balls for each height.
  3. Estimates sigma^2 from the final x positions.
  4. Fits sigma^2 versus H.
  5. Checks whether the intercept is close to a^2 / 3 and whether the
     relationship is approximately linear.
  6. Writes CSV, JSON, and optional PNG diagnostic output.

Typical use, inside the CUDA container:

    python3 test_brownian_model.py \
        --drop-range 1.0 \
        --balls 5000 \
        --min-rows 5 \
        --max-rows 50 \
        --row-points 10

The effective height H is, by default, H = N_rows * Delta_z, where
Delta_z = spacing * sqrt(3) / 2 is the vertical row spacing of the triangular
peg lattice. This excludes constant non-peg fall distances such as h_init,
which helps the intercept correspond to the initial release variance a^2 / 3.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


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
            "Test the Brownian Galton-board model sigma^2 = a^2/3 + 2 D_z H "
            "using the galton_cuda GPU simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--drop-range",
        type=float,
        default=1.0,
        help="Release half-width a. Initial x positions are uniform in [-a, a].",
    )
    parser.add_argument(
        "--balls",
        type=int,
        default=5000,
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
    height_group.add_argument("--min-rows", type=int, default=5)
    height_group.add_argument("--max-rows", type=int, default=50)
    height_group.add_argument(
        "--row-points",
        type=int,
        default=8,
        help="Number of distinct row counts to use when --rows is not given.",
    )
    height_group.add_argument(
        "--height-mode",
        choices=["rows", "total", "peg-span"],
        default="rows",
        help=(
            "How to define H. "
            "'rows': H = N_rows * Delta_z, recommended for comparison with the discrete row model. "
            "'total': H = h_init + N_rows * Delta_z + h_final, the full fall distance. "
            "'peg-span': H = (N_rows - 1) * Delta_z, distance between first and last peg row."
        ),
    )

    geom_group = parser.add_argument_group("board geometry")
    geom_group.add_argument(
        "--spacing",
        type=float,
        default=2.0,
        help="Center-to-center spacing d of neighboring pegs.",
    )
    geom_group.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Peg radius used by the collision kernel.",
    )
    geom_group.add_argument(
        "--first-row",
        type=int,
        default=0,
        help=(
            "Number of pegs in the first row. If 0, it is chosen automatically "
            "so the board is wider than the drop opening plus --edge-margin."
        ),
    )
    geom_group.add_argument(
        "--edge-margin",
        type=float,
        default=None,
        help=(
            "Extra horizontal peg-board margin beyond the drop range when auto-selecting "
            "--first-row. Default: max(4 * spacing, 2 * drop-range)."
        ),
    )
    geom_group.add_argument(
        "--h-init",
        type=float,
        default=1.0,
        help="Vertical distance from the release line y=0 to the first peg row.",
    )
    geom_group.add_argument(
        "--h-final",
        type=float,
        default=0.0,
        help="Extra vertical distance after the last peg row before the detection line.",
    )

    wall_group = parser.add_argument_group("walls")
    wall_group.add_argument(
        "--wall-padding",
        type=float,
        default=None,
        help=(
            "Extra padding added to the wall half-distance. "
            "Default: max(4 * spacing, 2 * drop-range)."
        ),
    )
    wall_group.add_argument(
        "--min-wall-distance",
        type=float,
        default=None,
        help="If set, force the wall half-distance to be at least this value.",
    )
    wall_group.add_argument(
        "--wall-check-sigma",
        type=float,
        default=3.0,
        help="Warn about wall effects if this many final sigmas approach the wall.",
    )

    sim_group = parser.add_argument_group("simulation")
    sim_group.add_argument(
        "--restitution",
        type=float,
        default=0.5,
        help="Coefficient of restitution e used in peg and wall reflections.",
    )
    sim_group.add_argument("--g", type=float, default=9.81, help="Gravity.")
    sim_group.add_argument(
        "--cell-size",
        type=float,
        default=0.0,
        help="Spatial-grid cell size. If <= 0, automatically set to max(spacing, 2*radius).",
    )
    sim_group.add_argument(
        "--sampling",
        choices=["random", "grid"],
        default="random",
        help=(
            "How to choose initial x positions. "
            "'random': uniform random sampling. "
            "'grid': evenly spaced midpoint sampling, which reduces initial sampling noise."
        ),
    )
    sim_group.add_argument("--seed", type=int, default=12345)

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output",
        default="brownian_test",
        help="Output directory for CSV, JSON, and PNG results.",
    )
    out_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create a matplotlib plot.",
    )

    check_group = parser.add_argument_group("acceptance thresholds")
    check_group.add_argument(
        "--r2-threshold",
        type=float,
        default=0.95,
        help="Minimum R^2 of the free sigma^2 vs H linear fit for a PASS.",
    )
    check_group.add_argument(
        "--intercept-tolerance",
        type=float,
        default=0.25,
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

    if args.rows:
        if any(r < 1 for r in args.rows):
            parser.error("All --rows values must be positive integers.")
    else:
        if args.min_rows < 1 or args.max_rows < 1:
            parser.error("--min-rows and --max-rows must be positive.")

    if args.wall_padding is not None and args.wall_padding < 0.0:
        parser.error("--wall-padding must be non-negative.")
    if args.edge_margin is not None and args.edge_margin < 0.0:
        parser.error("--edge-margin must be non-negative.")
    if args.min_wall_distance is not None and args.min_wall_distance < 0.0:
        parser.error("--min-wall-distance must be non-negative.")
    if args.cell_size < 0.0:
        parser.error("--cell-size must be non-negative.")

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


def auto_first_row(drop_range, spacing, radius, edge_margin):
    """
    Choose a first-row peg count large enough that the peg board extends beyond
    the release opening by edge_margin.
    """
    target_half_width = drop_range + edge_margin

    # For a row with n pegs, the outermost center is at +/- ((n - 1) * spacing / 2),
    # so the outermost peg edge is at +/- (((n - 1) * spacing / 2) + radius).
    needed = 2.0 * max(0.0, target_half_width - radius) / spacing + 1.0
    n = int(math.ceil(needed))

    return max(3, n)


def effective_height(n_rows, args, delta_z):
    """Effective board height H used for the Brownian-model fit."""
    if args.height_mode == "total":
        return float(args.h_init + n_rows * delta_z + args.h_final)

    if args.height_mode == "peg-span":
        return float(max(0.0, (n_rows - 1) * delta_z))

    # Default: one vertical row interval per row, matching the discrete N-row model.
    return float(n_rows * delta_z)


def simulate_one_height(
    n_rows,
    args,
    x_inits,
    first_row,
    wall_padding,
    cell_size,
):
    """Run one GPU simulation for a given number of peg rows."""
    circles, geometry_wall_distance = generate_circle_geometry(
        first_row,
        n_rows,
        args.spacing,
        args.h_init,
        args.radius,
        WALL_PADDING=wall_padding,
    )

    centers = np.array([c["center"] for c in circles], dtype=np.float64)

    wall_distance = float(geometry_wall_distance)
    if args.min_wall_distance is not None:
        wall_distance = max(wall_distance, float(args.min_wall_distance))

    if args.drop_range >= wall_distance:
        raise ValueError(
            f"Drop range a={args.drop_range} is not inside the walls at "
            f"+/-{wall_distance}. Increase --wall-padding, --min-wall-distance, "
            "or reduce --drop-range."
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

    return final_x, status, wall_distance


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
        "sigma2",
        "n_valid",
        "n_total",
        "valid_fraction",
        "wall_distance",
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


def make_plot(out_dir, records, fit, theory_intercept):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped because matplotlib is unavailable: {exc}")
        return None

    H = np.array([r["H"] for r in records], dtype=np.float64)
    sigma = np.array([r["sigma"] for r in records], dtype=np.float64)
    sigma2 = sigma**2

    if H.size == 0:
        return None

    if H.size > 1 and H.max() > H.min():
        H_fine = np.linspace(H.min(), H.max(), 200)
    else:
        H_fine = H

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    ax0.scatter(H, sigma2, color="tab:blue", label="simulation")

    slope_free = fit.get("slope_free")
    intercept_free = fit.get("intercept_free")
    if slope_free is not None and math.isfinite(slope_free):
        ax0.plot(
            H_fine,
            intercept_free + slope_free * H_fine,
            color="tab:orange",
            label="free linear fit",
        )

    slope_theory = fit.get("slope_fixed_theory")
    if slope_theory is not None and math.isfinite(slope_theory):
        ax0.plot(
            H_fine,
            theory_intercept + slope_theory * H_fine,
            "--",
            color="tab:green",
            label="fixed intercept $a^2/3$",
        )

    ax0.axhline(
        theory_intercept,
        color="gray",
        linewidth=0.8,
        label="theoretical intercept $a^2/3$",
    )
    ax0.set_xlabel("effective height H")
    ax0.set_ylabel(r"$\sigma^2$")
    ax0.set_title("Paper test: linear growth of variance")
    ax0.grid(alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    ax1.scatter(H, sigma, color="tab:blue", label="simulation")

    if slope_free is not None and math.isfinite(slope_free):
        ax1.plot(
            H_fine,
            np.sqrt(np.maximum(intercept_free + slope_free * H_fine, 0.0)),
            color="tab:orange",
            label="free fit",
        )

    if slope_theory is not None and math.isfinite(slope_theory):
        ax1.plot(
            H_fine,
            np.sqrt(np.maximum(theory_intercept + slope_theory * H_fine, 0.0)),
            "--",
            color="tab:green",
            label="fixed intercept model",
        )

    ax1.set_xlabel("effective height H")
    ax1.set_ylabel(r"$\sigma$")
    ax1.set_title("Standard deviation")
    ax1.grid(alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    plot_path = out_dir / "brownian_model_test.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def fmt(x, spec=".6g"):
    try:
        if x is None or not math.isfinite(x):
            return "nan"
        return format(x, spec)
    except Exception:
        return str(x)


def print_report(args, records, fit, theory_var, sample_var, delta_z):
    print("\n=== Brownian Galton-board model test ===")
    print(f"Release half-width a                    : {fmt(args.drop_range)}")
    print(f"Theoretical initial variance a^2/3      : {fmt(theory_var)}")
    print(f"Sample initial variance                 : {fmt(sample_var)}")
    print(f"Vertical row spacing Delta_z            : {fmt(delta_z)}")
    print(f"Height mode                             : {args.height_mode}")
    print(f"Restitution e                           : {fmt(args.restitution)}")
    print(f"Balls per height                        : {args.balls}")

    print("\nMeasured results:")
    header = (
        f"{'rows':>6} {'H':>12} {'sigma':>12} {'sigma^2':>12} "
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
            "\nPossible remedies: increase --wall-padding/--edge-margin, reduce heights, "
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

    edge_margin = (
        args.edge_margin
        if args.edge_margin is not None
        else max(4.0 * args.spacing, 2.0 * a)
    )
    wall_padding = (
        args.wall_padding
        if args.wall_padding is not None
        else max(4.0 * args.spacing, 2.0 * a)
    )

    if args.min_wall_distance is None:
        args.min_wall_distance = 0.0

    if args.first_row and args.first_row > 0:
        first_row = max(2, int(args.first_row))
    else:
        first_row = auto_first_row(a, args.spacing, args.radius, edge_margin)
        first_row = max(2, first_row)

    if first_row > 5000:
        raise SystemExit(
            f"Auto-selected first row count is very large ({first_row}). "
            "Use --first-row to set it explicitly, or reduce --edge-margin."
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
    print(f"  galton_cuda first-row peg count : {first_row}")
    print(f"  row spacing Delta_z             : {fmt(delta_z)}")
    print(f"  wall padding                    : {fmt(wall_padding)}")
    print(f"  edge margin                     : {fmt(edge_margin)}")
    print(f"  spatial-grid cell size          : {fmt(cell_size)}")
    print(f"  initial sample variance         : {fmt(sample_initial_variance)}")
    print(f"  theoretical a^2/3               : {fmt(theory_initial_variance)}")
    print()

    records = []

    for n_rows in rows_list:
        final_x, status, wall_distance = simulate_one_height(
            n_rows=n_rows,
            args=args,
            x_inits=x_inits,
            first_row=first_row,
            wall_padding=wall_padding,
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
        H = effective_height(n_rows, args, delta_z)

        wall_ok = bool(args.wall_check_sigma * sigma < wall_distance)

        rec = {
            "rows": int(n_rows),
            "H": float(H),
            "mean_x": mean_x,
            "sigma": float(sigma),
            "sigma2": float(sigma2),
            "n_valid": n_valid,
            "n_total": int(n_balls),
            "valid_fraction": float(n_valid) / float(n_balls),
            "wall_distance": float(wall_distance),
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
            "first_row": first_row,
            "wall_padding": wall_padding,
            "edge_margin": edge_margin,
            "cell_size": cell_size,
            "theory_initial_variance": theory_initial_variance,
            "sample_initial_variance": sample_initial_variance,
        },
        "records": records,
        "fit": fit,
    }

    json_path = out_dir / "brownian_model_summary.json"
    write_json(json_path, payload)

    plot_path = None
    if not args.no_plot:
        plot_path = make_plot(out_dir, records, fit, theory_initial_variance)

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
    if plot_path is not None:
        print(f"  Plot  : {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())