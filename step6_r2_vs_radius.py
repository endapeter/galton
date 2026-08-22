# Step 6 of the science experiment queue (PLAN.md, section 7):
#
#   R² of normality vs. peg radius, iterated over peg distance.
#
# Nested two-parameter sweep built on the galton_cuda package: the outer loop
# fixes DISTANCE (peg spacing) and reuses run_sweep() to sweep RADIUS, so the
# existing per-frame machinery (GPU batch, histogram, Q-Q plot, verification
# walk, CSV) runs unchanged - one output subdirectory per distance value.
# The aggregation stage below then re-reads each frame's CSV, recomputes the
# Q-Q normality R² (scipy.stats.probplot, the same recipe as
# galton_cuda.visualization.generate_qq_plot) and collects every
# (distance, radius, R²) point into one summary figure - one R²-vs-radius
# curve per distance value on shared axes - plus a summary CSV.
#
# Question answered: how peg size and gap size *jointly* shape the normality
# of the output distribution.
#
# Run inside the CUDA container (see CUDA_GUIDE.md):
#
#       docker run --rm --gpus all -v "C:/Users/Enda/Data/Code/galton:/work" -w /work galton-cuda python3 step6_r2_vs_radius.py
#
# Smoke test (fewer balls / radius steps, same env knobs as -m galton_cuda):
#
#   ... -e GALTON_BALLS=200 -e GALTON_INTERVALS=3 ... \
#       python3 step6_r2_vs_radius.py

import argparse
import csv
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt  # Agg backend set by galton_cuda.visualization

from galton_cuda import run_sweep

# RADIUS must stay below half the peg pitch, otherwise neighbouring pegs
# overlap and the board closes up entirely.
_MAX_RADIUS_MARGIN = 1e-6


def qq_r_squared(csv_filename):
    """R² of the Q-Q linearity of the successful final_x values in a frame CSV.

    Same computation as galton_cuda.visualization.generate_qq_plot, minus the
    figure: probplot against the standard normal, R² = r² of the linear fit.
    Returns (r_squared, n_valid); (nan, n) when fewer than 30 valid points
    remain - the same threshold the Q-Q plot stage uses.
    """
    df = pd.read_csv(csv_filename)
    df['final_x'] = pd.to_numeric(df['final_x'], errors='coerce')
    data = df.dropna(subset=['final_x'])['final_x']
    if len(data) < 30:
        return float('nan'), len(data)
    (osm, osr), (slope, intercept, r_value) = stats.probplot(data, dist="norm")
    return r_value ** 2, len(data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="step6_r2_vs_radius",
        description="Nested (DISTANCE x RADIUS) sweep: Q-Q normality R² vs. "
                    "peg radius, one curve per peg distance "
                    "(run inside the CUDA container - see CUDA_GUIDE.md).")
    parser.add_argument("--distances", type=float, nargs='+',
                        default=[2.0, 2.5, 3.0, 3.5, 4.0],
                        help="peg DISTANCE values, one R²-vs-radius curve "
                             "each (default: 2.0 2.5 3.0 3.5 4.0)")
    parser.add_argument("--radius-low", type=float, default=0.1,
                        help="first swept peg radius (default: 0.1)")
    parser.add_argument("--radius-high", type=float, default=0.7,
                        help="last swept peg radius (default: 0.7)")
    parser.add_argument("--radius-steps", type=int,
                        default=int(os.environ.get("GALTON_INTERVALS", 7)),
                        help="number of radius steps per distance curve "
                             "(default: 7, or $GALTON_INTERVALS)")
    parser.add_argument("--balls", type=int,
                        default=int(os.environ.get("GALTON_BALLS", 2000)),
                        help="balls per frame (default: 2000, or $GALTON_BALLS)")
    parser.add_argument("--bins", type=int, default=200,
                        help="histogram bins per frame (default: 200)")
    parser.add_argument("--output", default=os.path.join("figures", "step6_r2_vs_radius"),
                        help="output directory for the per-distance sweeps and "
                             "the summary figure/CSV "
                             "(default: figures/step6_r2_vs_radius)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # The ball is a point mass; RADIUS is the collision distance from the peg
    # center. Above half the smallest pitch the pegs would touch, leaving no
    # gap to funnel through.
    min_distance = min(args.distances)
    if args.radius_high >= min_distance / 2 - _MAX_RADIUS_MARGIN:
        raise SystemExit(
            f"radius-high ({args.radius_high}) must stay below half the "
            f"smallest distance ({min_distance / 2:.2f}), otherwise "
            f"neighbouring pegs overlap")

    records = []
    for dist in args.distances:
        dist_dir = os.path.join(args.output, f"dist_{dist:.2f}")
        print(f"\n=== DISTANCE = {dist:.2f} : sweeping RADIUS "
              f"{args.radius_low} -> {args.radius_high} "
              f"in {args.radius_steps} steps ===")

        results = run_sweep(
            sweep_param='RADIUS',
            low=args.radius_low,
            high=args.radius_high,
            steps=args.radius_steps,
            fixed={'DISTANCE': dist},
            n_balls=args.balls,
            n_bins=args.bins,
            output_dir=dist_dir,
        )

        for res in results:
            r_squared, n_valid = qq_r_squared(res['csv_filename'])
            records.append({
                "distance": dist,
                "radius": res["param_val"],
                "n_success": res["n_success"],
                "n_valid": n_valid,
                "r_squared": r_squared,
            })

    # --- Summary CSV: one row per (distance, radius) pair ---
    os.makedirs(args.output, exist_ok=True)
    summary_csv = os.path.join(args.output, "r2_vs_radius_summary.csv")
    with open(summary_csv, mode='w', newline='') as csv_file:
        fieldnames = ["distance", "radius", "n_success", "n_valid", "r_squared"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            record = dict(record)
            record["r_squared"] = ("" if np.isnan(record["r_squared"])
                                   else f"{record['r_squared']:.6f}")
            writer.writerow(record)
    print(f"\n-> Summary CSV saved to {summary_csv}")

    # --- Summary figure: one R²-vs-radius curve per distance value ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colormap = plt.get_cmap('viridis')
    distances = sorted(set(r["distance"] for r in records))
    for i, dist in enumerate(distances):
        pts = [(r["radius"], r["r_squared"]) for r in records
               if r["distance"] == dist and not np.isnan(r["r_squared"])]
        if not pts:
            print(f"Warning: no valid R² points for DISTANCE = {dist:.2f} "
                  "- curve skipped")
            continue
        radii, r2s = zip(*pts)
        color = colormap(i / max(len(distances) - 1, 1))
        ax.plot(radii, r2s, marker='o', color=color, linewidth=2,
                label=f'DISTANCE = {dist:.2f}')

    ax.set_xlabel('Peg Radius (collision distance)', fontsize=12)
    ax.set_ylabel('Q-Q Normality R² (linearity)', fontsize=12)
    ax.set_title('Galton Board Normality: R² vs. Peg Radius, per Peg Distance\n'
                 f'{args.balls} balls per frame, narrow funnel '
                 '(-0.05, 0.05), e = 0.1', fontsize=13, fontweight='bold')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Peg distance', loc='best')
    plt.tight_layout()

    summary_png = os.path.join(args.output, "r2_vs_radius_by_distance.png")
    plt.savefig(summary_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Summary figure saved to {summary_png}")

    # Console table for a quick read without opening the figure
    print("\n distance | radius |  n_success |      R²")
    print("-" * 42)
    for r in records:
        r2 = "n/a" if np.isnan(r["r_squared"]) else f"{r['r_squared']:.4f}"
        print(f" {r['distance']:>8.2f} | {r['radius']:>6.2f} | "
              f"{r['n_success']:>10d} | {r2:>7s}")


if __name__ == "__main__":
    main()
