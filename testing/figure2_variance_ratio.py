#!/usr/bin/env python3
"""
figure2_variance_ratio.py

Figure 2 of the Brownian Galton-board study.

For every funnel width W = 2a the board is simulated ONCE, at the maximum
height of the height specification (largest row count), and the result is
plotted as a single ratio

    sigma^2_out / sigma^2_in ,      sigma^2_in = W^2 / 12 (= a^2 / 3),

i.e. input and output variance are combined into one ratio instead of two
separate lines. No mean-height approximation is used: every plotted point
is an actual simulation at the full (maximum) board height.

If a diffusion coefficient is available (from --dz or from the JSON summary
written by test_brownian_model.py), the theoretical curve

    ratio(W) = 1 + 2 D_z H_max / (W^2 / 12)

is overlaid.

Outputs (written to --output):
    figure2_variance_ratio.png / .pdf
    figure2_results.csv
    figure2_summary.json

Typical use, inside the CUDA container (see testing/README.md):
    python3 testing/figure2_variance_ratio.py --min-funnel 1.0 \
        --max-funnel 6.0 --funnel-points 6 --balls 5000 --max-rows 50
"""
import argparse
import csv
import json
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
            "Figure 2: output/input variance ratio at the maximum board "
            "height, plotted against funnel width W = 2a."
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
        help="Explicit list of funnel widths W = 2a, e.g. --funnels 1 2 3 4.",
    )
    funnel_group.add_argument("--min-funnel", type=float, default=1.0)
    funnel_group.add_argument("--max-funnel", type=float, default=6.0)
    funnel_group.add_argument("--funnel-points", type=int, default=6)
    parser.add_argument("--balls", type=int, default=DEFAULTS["balls"])

    height_group = parser.add_argument_group(
        "height (only the LARGEST row count of this specification is simulated)"
    )
    height_group.add_argument("--rows", type=int, nargs="+", default=None)
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

    model_group = parser.add_argument_group("model curve")
    model_group.add_argument(
        "--dz",
        type=float,
        default=None,
        help="Diffusion coefficient D_z for the theory curve. If omitted it "
             "is read from --summary-json.",
    )
    model_group.add_argument(
        "--summary-json",
        default=f"{DEFAULTS['output']}/brownian_model_summary.json",
        help="JSON summary written by test_brownian_model.py.",
    )

    parser.add_argument("--output", default=DEFAULTS["output"])

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
    if args.dz is not None and args.dz <= 0.0:
        parser.error("--dz must be positive.")
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


def resolve_dz(args):
    """Obtain D_z from --dz or from the JSON summary of the main test."""
    if args.dz is not None:
        return float(args.dz), "command line --dz"
    path = Path(args.summary_json)
    if not path.is_file():
        print(
            f"[warn] {path} not found; run testing/test_brownian_model.py "
            "first or pass --dz. The theoretical model curve will be skipped."
        )
        return None, None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}; model curve skipped.")
        return None, None
    fit = payload.get("fit", {})
    for key in ("Dz_free", "Dz_fixed_theory", "Dz_fixed_sample"):
        try:
            value = float(fit.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value, f"{path} -> fit.{key}"
    print(f"[warn] no usable D_z found in {path}; model curve skipped.")
    return None, None


def run_max_height_sweep(args, funnel_widths, n_rows, delta_z):
    """One simulation per funnel width, at the maximum height."""
    H_max = effective_height(n_rows, args, delta_z)
    records = []
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

        final_x, status = simulate_one_height(
            n_rows, sim_args, x_inits, first_row, wall_distance, cell_size,
        )
        valid = (status == 1) & np.isfinite(final_x)
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 2:
            print(f"[warn] W={W:g}: fewer than two valid trajectories; skipping.")
            continue
        x_final = final_x[valid]
        mean_x = float(np.mean(x_final))
        sigma = float(np.std(x_final, ddof=1))
        sigma2 = sigma * sigma
        var_in_theory = a * a / 3.0
        wall_ok = bool(args.wall_check_sigma * sigma < wall_distance)
        records.append(
            {
                "funnel_width": float(W),
                "a": float(a),
                "rows": int(n_rows),
                "H": float(H_max),
                "mean_x": mean_x,
                "sigma": sigma,
                "sigma2_out": sigma2,
                "var_in_theory": var_in_theory,
                "var_in_sample": sample_var,
                "ratio_out_over_in": sigma2 / var_in_theory,
                "ratio_out_over_in_sample": (
                    sigma2 / sample_var if sample_var > 0.0 else float("nan")
                ),
                "n_valid": n_valid,
                "n_total": int(args.balls),
                "valid_fraction": float(n_valid) / float(args.balls),
                "wall_distance": float(wall_distance),
                "wall_width": float(wall_width),
                "first_row": int(first_row),
                "peg_edge_distance": float(
                    peg_edge_distance(first_row, args.spacing, args.radius)
                ),
                "wall_ok": wall_ok,
            }
        )
        print(
            f"W={W:8.4g}  rows={n_rows:5d}  H={H_max:10.6g}  "
            f"sigma^2_out={sigma2:12.6g}  var_in={var_in_theory:12.6g}  "
            f"ratio={sigma2 / var_in_theory:10.6g}  valid={n_valid}/{args.balls}"
        )
    return records


def write_results_csv(path, records):
    fieldnames = [
        "funnel_width", "a", "rows", "H", "mean_x", "sigma",
        "sigma2_out", "var_in_theory", "var_in_sample",
        "ratio_out_over_in", "ratio_out_over_in_sample",
        "n_valid", "n_total", "valid_fraction",
        "wall_distance", "wall_width", "first_row", "wall_ok",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def make_plot(out_dir, records, dz, dz_source, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped because matplotlib is unavailable: {exc}")
        return None, None

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    W_values = [r["funnel_width"] for r in records]
    ratios = [r["ratio_out_over_in"] for r in records]
    ax.scatter(
        W_values,
        ratios,
        color="tab:red",
        s=70,
        edgecolor="black",
        zorder=3,
        label=(
            f"Simulated ratio at $H_{{max}}$ "
            f"({records[0]['rows']} rows, $H = {records[0]['H']:.2f}$)"
        ),
    )

    if dz is not None:
        H_max = records[0]["H"]
        W_grid = np.linspace(0.9 * min(W_values), 1.05 * max(W_values), 200)
        curve = 1.0 + 2.0 * dz * H_max / (W_grid ** 2 / 12.0)
        ax.plot(
            W_grid,
            curve,
            color="tab:blue",
            lw=2.0,
            label=(
                r"Model $1 + 2 D_z H_{max}/(W^2/12)$, "
                rf"$D_z = {dz:.4g}$ ({dz_source})"
            ),
        )

    ax.axhline(
        1.0, color="tab:green", linestyle="--", lw=2.0,
        label="No broadening (ratio = 1)",
    )
    ax.set_xlabel(r"Funnel width $W = 2a$", fontsize=12)
    ax.set_ylabel(
        r"Variance ratio $\sigma^2_\mathrm{out}/\sigma^2_\mathrm{in}$",
        fontsize=12,
    )
    ax.set_title(
        "Figure 2: Output/input variance ratio at maximum height",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    png_path = out_dir / "figure2_variance_ratio.png"
    pdf_path = out_dir / "figure2_variance_ratio.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return png_path, pdf_path


def print_report(records, dz, dz_source):
    print("\n=== Figure 2: variance ratio at maximum height ===")
    if dz is not None:
        print(f"D_z used for the model curve: {fmt(dz)}  ({dz_source})")
    else:
        print("No D_z available; model curve skipped.")
    header = (
        f"{'W':>10} {'H':>10} {'sigma^2_out':>14} {'W^2/12':>12} "
        f"{'ratio':>10} {'valid':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['funnel_width']:10.6g} {r['H']:10.6g} "
            f"{r['sigma2_out']:14.6g} {r['var_in_theory']:12.6g} "
            f"{r['ratio_out_over_in']:10.6g} "
            f"{r['n_valid']}/{r['n_total']:>10}"
        )


def main(argv=None):
    args = parse_args(argv)
    require_cuda()
    rows_list = make_rows(args)
    n_rows = max(rows_list)
    funnel_widths = make_funnels(args)
    delta_z = args.spacing * math.sqrt(3.0) / 2.0

    print("Figure 2 configuration:")
    print(f"  funnel widths : {[f'{w:g}' for w in funnel_widths]}")
    print(f"  max rows      : {n_rows}")
    print(f"  balls/point   : {args.balls}")
    print(f"  Delta_z       : {fmt(delta_z)}")
    print()

    records = run_max_height_sweep(args, funnel_widths, n_rows, delta_z)
    if not records:
        raise SystemExit("No valid simulation results were produced.")

    dz, dz_source = resolve_dz(args)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "figure2_results.csv"
    write_results_csv(csv_path, records)
    json_path = out_dir / "figure2_summary.json"
    write_json(
        json_path,
        {
            "args": vars(args),
            "delta_z": delta_z,
            "rows_simulated": n_rows,
            "dz": dz,
            "dz_source": dz_source,
            "records": records,
        },
    )
    png_path, pdf_path = make_plot(out_dir, records, dz, dz_source, args)
    print_report(records, dz, dz_source)

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