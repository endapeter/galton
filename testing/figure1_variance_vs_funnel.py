#!/usr/bin/env python3
"""
figure1_variance_vs_funnel.py

Figure 1 of the Brownian Galton-board study.

Independent variable : funnel width W = 2a
Dependent variable   : measured variance sigma^2 of the final x positions
Sweep structure      : for EVERY funnel width, the full effective-height
                       sweep from test_brownian_model.py is repeated.

The Brownian model (PLAN_simple.md Eq. 13, with a = W/2) is

    sigma^2(W, H) = W^2 / 12 + 2 D_z H.

At each funnel width a free linear fit sigma^2 = intercept + slope * H is
performed over the height sweep; the fitted values are drawn as lines
connecting the swept heights, while markers are the measured variances.

Outputs (written to --output):
    figure1_variance_vs_funnel.png / .pdf
    figure1_results.csv
    figure1_summary.json

Typical use, inside the CUDA container (see testing/README.md):
    python3 testing/figure1_variance_vs_funnel.py --min-funnel 1.0 \
        --max-funnel 6.0 --funnel-points 6 --balls 5000 \
        --min-rows 5 --max-rows 50 --row-points 8
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from test_brownian_model import (
    DEFAULTS,
    require_cuda,
    make_rows,
    make_x_inits,
    simulate_one_height,
    fit_model,
    default_wall_width,
    pegs_for_wall_distance,
    peg_edge_distance,
    effective_height,
    write_json,
    fmt,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Figure 1: measured variance sigma^2 vs funnel width W = 2a. "
            "A full effective-height sweep is repeated at every funnel width."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    funnel_group = parser.add_argument_group(
        "funnel-width sweep (independent variable)"
    )
    funnel_group.add_argument(
        "--funnels",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit list of funnel widths W = 2a, e.g. --funnels 1 2 3 4. "
            "Overrides --min-funnel/--max-funnel/--funnel-points."
        ),
    )
    funnel_group.add_argument(
        "--min-funnel", type=float, default=1.0,
        help="Smallest funnel width W = 2a.",
    )
    funnel_group.add_argument(
        "--max-funnel", type=float, default=6.0,
        help="Largest funnel width W = 2a.",
    )
    funnel_group.add_argument(
        "--funnel-points", type=int, default=6,
        help="Number of funnel widths in the sweep.",
    )
    parser.add_argument(
        "--balls", type=int, default=DEFAULTS["balls"],
        help="Balls simulated per (funnel width, height) point.",
    )

    height_group = parser.add_argument_group(
        "height sweep (repeated at every funnel width)"
    )
    height_group.add_argument(
        "--rows", type=int, nargs="+", default=None,
        help="Explicit list of peg-row counts, e.g. --rows 5 10 20.",
    )
    height_group.add_argument("--min-rows", type=int, default=DEFAULTS["min_rows"])
    height_group.add_argument("--max-rows", type=int, default=DEFAULTS["max_rows"])
    height_group.add_argument(
        "--row-points", type=int, default=DEFAULTS["row_points"]
    )
    height_group.add_argument(
        "--height-mode",
        choices=["rows", "total", "peg-span"],
        default=DEFAULTS["height_mode"],
    )

    geom_group = parser.add_argument_group("board geometry")
    geom_group.add_argument("--spacing", type=float, default=DEFAULTS["spacing"])
    geom_group.add_argument("--radius", type=float, default=DEFAULTS["radius"])
    geom_group.add_argument("--h-init", type=float, default=DEFAULTS["h_init"])
    geom_group.add_argument("--h-final", type=float, default=DEFAULTS["h_final"])

    wall_group = parser.add_argument_group("walls")
    wall_group.add_argument(
        "--wall-width",
        type=float,
        default=None,
        help=(
            "Fixed full wall width used for every funnel width. If omitted, "
            "the wall width is chosen automatically for each funnel width."
        ),
    )
    wall_group.add_argument(
        "--wall-check-sigma",
        type=float,
        default=DEFAULTS["wall_check_sigma"],
    )

    sim_group = parser.add_argument_group("simulation")
    sim_group.add_argument(
        "--restitution", type=float, default=DEFAULTS["restitution"]
    )
    sim_group.add_argument("--g", type=float, default=DEFAULTS["g"])
    sim_group.add_argument("--cell-size", type=float, default=DEFAULTS["cell_size"])
    sim_group.add_argument(
        "--sampling", choices=["random", "grid"], default=DEFAULTS["sampling"]
    )
    sim_group.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    parser.add_argument(
        "--output", default=DEFAULTS["output"],
        help="Output directory for CSV/JSON/PNG/PDF results.",
    )

    args = parser.parse_args(argv)
    if args.balls < 2:
        parser.error("--balls must be at least 2.")
    if args.funnel_points < 1:
        parser.error("--funnel-points must be at least 1.")
    if args.funnels:
        if any(w <= 0.0 for w in args.funnels):
            parser.error("All --funnels values must be positive.")
    elif args.min_funnel <= 0.0 or args.max_funnel <= 0.0:
        parser.error("--min-funnel and --max-funnel must be positive.")
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
    if args.cell_size < 0.0:
        parser.error("--cell-size must be non-negative.")
    if args.wall_width is not None and args.wall_width <= 0.0:
        parser.error("--wall-width must be positive.")
    if args.rows and any(r < 1 for r in args.rows):
        parser.error("All --rows values must be positive integers.")
    return args


def make_funnels(args):
    """Build the list of funnel widths W = 2a to test."""
    if args.funnels:
        widths = sorted({float(w) for w in args.funnels if w > 0.0})
    else:
        lo, hi = float(args.min_funnel), float(args.max_funnel)
        if lo > hi:
            lo, hi = hi, lo
        if args.funnel_points == 1:
            widths = [lo]
        else:
            widths = [float(w) for w in np.linspace(lo, hi, args.funnel_points)]
    if not widths:
        raise SystemExit("No positive funnel widths selected.")
    return widths


def run_funnel_sweep(args, funnel_widths, rows_list, delta_z):
    """For each funnel width, run the complete effective-height sweep."""
    sweeps = []
    for W in funnel_widths:
        a = 0.5 * W
        wall_width = (
            float(args.wall_width)
            if args.wall_width is not None
            else default_wall_width(a, args.spacing)
        )
        wall_distance = wall_width / 2.0
        if a >= wall_distance:
            raise SystemExit(
                f"Funnel width W={W:g} (a={a:g}) does not fit inside the walls "
                f"at +/-{wall_distance:g}. Increase --wall-width or reduce "
                "the funnel sweep range."
            )
        try:
            first_row = pegs_for_wall_distance(
                wall_distance, args.spacing, args.radius
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        occupied = peg_edge_distance(first_row, args.spacing, args.radius)
        cell_size = (
            float(args.cell_size)
            if args.cell_size > 0.0
            else max(args.spacing, 2.0 * args.radius, 1e-6)
        )
        sim_args = argparse.Namespace(
            spacing=args.spacing,
            radius=args.radius,
            h_init=args.h_init,
            h_final=args.h_final,
            restitution=args.restitution,
            g=args.g,
            drop_range=a,
        )
        x_inits = np.ascontiguousarray(
            make_x_inits(args.balls, a, args.sampling, args.seed),
            dtype=np.float64,
        )
        sample_var = float(np.var(x_inits, ddof=1)) if x_inits.size > 1 else 0.0

        records = []
        for n_rows in rows_list:
            final_x, status = simulate_one_height(
                n_rows, sim_args, x_inits, first_row, wall_distance, cell_size,
            )
            valid = (status == 1) & np.isfinite(final_x)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 2:
                print(
                    f"[warn] W={W:g}, rows={n_rows}: fewer than two valid "
                    "trajectories; skipping."
                )
                continue
            x_final = final_x[valid]
            mean_x = float(np.mean(x_final))
            sigma = float(np.std(x_final, ddof=1))
            sigma2 = sigma * sigma
            H = effective_height(n_rows, args, delta_z)
            wall_ok = bool(args.wall_check_sigma * sigma < wall_distance)
            records.append(
                {
                    "funnel_width": float(W),
                    "a": float(a),
                    "rows": int(n_rows),
                    "H": float(H),
                    "mean_x": mean_x,
                    "sigma": sigma,
                    "sigma2": sigma2,
                    "n_valid": n_valid,
                    "n_total": int(args.balls),
                    "valid_fraction": float(n_valid) / float(args.balls),
                    "wall_distance": float(wall_distance),
                    "wall_width": float(wall_width),
                    "first_row": int(first_row),
                    "wall_ok": wall_ok,
                }
            )
            print(
                f"W={W:8.4g}  rows={n_rows:5d}  H={H:10.6g}  "
                f"sigma^2={sigma2:12.6g}  valid={n_valid}/{args.balls}  "
                f"wall_ok={wall_ok}"
            )

        if not records:
            print(f"[warn] W={W:g}: no valid records; funnel width skipped.")
            continue

        H_arr = np.array([r["H"] for r in records], dtype=np.float64)
        s2_arr = np.array([r["sigma2"] for r in records], dtype=np.float64)
        fit = fit_model(H_arr, s2_arr, a * a / 3.0, sample_var)
        sweeps.append(
            {
                "funnel_width": float(W),
                "a": float(a),
                "wall_width": float(wall_width),
                "wall_distance": float(wall_distance),
                "first_row": int(first_row),
                "peg_edge_distance": float(occupied),
                "theory_initial_variance": a * a / 3.0,
                "sample_initial_variance": sample_var,
                "records": records,
                "fit": fit,
            }
        )
    return sweeps


def write_results_csv(path, sweeps):
    fieldnames = [
        "funnel_width", "a", "rows", "H", "mean_x", "sigma", "sigma2",
        "n_valid", "n_total", "valid_fraction",
        "wall_distance", "wall_width", "first_row", "wall_ok",
    ]
    rows = [rec for s in sweeps for rec in s["records"]]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(out_dir, sweeps, rows_list, args, delta_z):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped because matplotlib is unavailable: {exc}")
        return None, None

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    W_values = np.array([s["funnel_width"] for s in sweeps], dtype=np.float64)
    W_grid = np.linspace(0.9 * W_values.min(), 1.05 * W_values.max(), 200)
    ax.plot(
        W_grid,
        W_grid ** 2 / 12.0,
        "k--",
        lw=2.0,
        label=r"Input variance $W^2/12$",
    )

    cmap = plt.get_cmap("viridis")
    colors = [cmap(t) for t in np.linspace(0.15, 0.85, max(len(rows_list), 1))]
    for j, n_rows in enumerate(rows_list):
        H_this = effective_height(n_rows, args, delta_z)
        W_pts, s2_pts, fit_pts = [], [], []
        for s in sweeps:
            rec = next((r for r in s["records"] if r["rows"] == n_rows), None)
            if rec is not None:
                W_pts.append(s["funnel_width"])
                s2_pts.append(rec["sigma2"])
            fit = s["fit"]
            slope = fit.get("slope_free")
            intercept = fit.get("intercept_free")
            if (
                slope is not None
                and intercept is not None
                and math.isfinite(slope)
                and math.isfinite(intercept)
            ):
                fit_pts.append((s["funnel_width"], intercept + slope * H_this))
        ax.scatter(
            W_pts,
            s2_pts,
            color=colors[j],
            s=55,
            edgecolor="black",
            zorder=3,
            label=f"$H = {H_this:.2f}$ ({n_rows} rows)",
        )
        if fit_pts:
            fit_pts.sort()
            ax.plot(
                [p[0] for p in fit_pts],
                [p[1] for p in fit_pts],
                color=colors[j],
                lw=1.7,
                alpha=0.85,
                zorder=2,
            )

    ax.set_xlabel(r"Funnel width $W = 2a$", fontsize=12)
    ax.set_ylabel(r"Variance $\sigma^2$", fontsize=12)
    ax.set_title(
        "Figure 1: Variance vs funnel width\n"
        "(effective-height sweep repeated at every funnel width)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    png_path = out_dir / "figure1_variance_vs_funnel.png"
    pdf_path = out_dir / "figure1_variance_vs_funnel.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return png_path, pdf_path


def print_report(sweeps):
    print("\n=== Figure 1: free fits sigma^2 = intercept + slope*H per width ===")
    header = (
        f"{'W':>10} {'a^2/3':>12} {'intercept':>12} "
        f"{'slope':>12} {'D_z':>12} {'R^2':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in sweeps:
        fit = s["fit"]
        print(
            f"{s['funnel_width']:10.6g} "
            f"{s['theory_initial_variance']:12.6g} "
            f"{fmt(fit.get('intercept_free')):>12} "
            f"{fmt(fit.get('slope_free')):>12} "
            f"{fmt(fit.get('Dz_free')):>12} "
            f"{fmt(fit.get('r2_free')):>10}"
        )


def main(argv=None):
    args = parse_args(argv)
    require_cuda()
    rows_list = make_rows(args)
    funnel_widths = make_funnels(args)
    delta_z = args.spacing * math.sqrt(3.0) / 2.0

    print("Figure 1 configuration:")
    print(f"  funnel widths : {[f'{w:g}' for w in funnel_widths]}")
    print(f"  rows          : {rows_list}")
    print(f"  balls/point   : {args.balls}")
    print(f"  Delta_z       : {fmt(delta_z)}")
    print()

    sweeps = run_funnel_sweep(args, funnel_widths, rows_list, delta_z)
    if not sweeps:
        raise SystemExit("No valid simulation results were produced.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "figure1_results.csv"
    write_results_csv(csv_path, sweeps)
    json_path = out_dir / "figure1_summary.json"
    write_json(
        json_path,
        {
            "args": vars(args),
            "delta_z": delta_z,
            "rows": rows_list,
            "sweeps": sweeps,
        },
    )
    png_path, pdf_path = make_plot(out_dir, sweeps, rows_list, args, delta_z)
    print_report(sweeps)

    print("\nOutput files:")
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    if png_path is not None:
        print(f"  Plot (PNG) : {png_path}")
    if pdf_path is not None:
        print(f"  Plot (PDF) : {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())