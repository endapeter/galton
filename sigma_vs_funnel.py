# Step 7 of the experiment queue in PLAN.md (section 7):
# sigma(output) vs. funnel width - one static figure.
#
# The X_DROP_RANGE sweep already produces the data: this script runs that
# sweep once through the galton_cuda package (one GPU batch per funnel
# half-width v, drop window (-v, +v)), then reads each frame's CSV, computes
# the standard deviation of the successful final_x values, and aggregates
# every (funnel half-width, sigma) pair into a single summary figure plus a
# small CSV. No animation, no per-frame changes - the per-frame machinery
# (histogram, Q-Q plot, verification walk, CSV) is untouched.
#
# Run inside the CUDA container from the repository root (see CUDA_GUIDE.md):
#
#     MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
#         -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
#         galton-cuda python3 sigma_vs_funnel.py
#
# Smoke test (same env knobs as python3 -m galton_cuda):
#
#     ... -e GALTON_BALLS=200 -e GALTON_INTERVALS=2 ... python3 sigma_vs_funnel.py

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, matches galton_cuda.visualization
import matplotlib.pyplot as plt

from galton_cuda import run_sweep


def sigma_from_csv(csv_filename):
    """Sample std of the successful final_x values in one frame's CSV.

    Returns (sigma, n_success); sigma is NaN if no ball landed.
    """
    final_x = []
    with open(csv_filename, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["status"] == "success":
                final_x.append(float(row["final_x"]))
    arr = np.asarray(final_x, dtype=np.float64)
    sigma = arr.std(ddof=1) if arr.size >= 2 else np.nan
    return sigma, arr.size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Step 7: sigma(output) vs. funnel width "
                    "(run inside the CUDA container - see CUDA_GUIDE.md).")
    parser.add_argument("--low", type=float, default=0.05,
                        help="first funnel half-width (default: 0.05)")
    parser.add_argument("--high", type=float, default=47.0,
                        help="last funnel half-width (default: 47.0)")
    parser.add_argument("--steps", type=int,
                        default=int(os.environ.get("GALTON_INTERVALS", 10)),
                        help="number of sweep frames (default: 10, or "
                             "$GALTON_INTERVALS)")
    parser.add_argument("--balls", type=int,
                        default=int(os.environ.get("GALTON_BALLS", 2000)),
                        help="balls per frame (default: 2000, or $GALTON_BALLS)")
    parser.add_argument("--output", default="figures",
                        help="output directory (default: figures)")
    args = parser.parse_args(argv)

    # --- Stage 1: run the existing funnel-width sweep on the GPU ---
    results = run_sweep(
        sweep_param="X_DROP_RANGE",
        low=args.low,
        high=args.high,
        steps=args.steps,
        n_balls=args.balls,
        output_dir=args.output,
    )

    # --- Stage 2: aggregate sigma per frame from the CSVs ---
    # sigma_in = v / sqrt(3) is the std of the uniform input U(-v, +v),
    # plotted as the reference the board transforms.
    rows = []
    for r in results:
        v = float(r["param_val"])
        sigma_out, n_success = sigma_from_csv(r["csv_filename"])
        if np.isnan(sigma_out):
            print(f"WARNING: frame {r['frame_index']:03d} (v = {v:.4f}) has "
                  f"{n_success} successful balls - sigma undefined, skipped")
            continue
        rows.append({
            "frame_index": r["frame_index"],
            "funnel_half_width": v,
            "funnel_full_width": 2.0 * v,
            "sigma_input": v / np.sqrt(3.0),
            "sigma_output": sigma_out,
            "n_success": n_success,
            "n_balls": r["n_balls"],
        })

    if not rows:
        raise RuntimeError("no frame produced enough successful balls to "
                           "compute sigma")

    half_widths = np.array([row["funnel_half_width"] for row in rows])
    sigma_in = np.array([row["sigma_input"] for row in rows])
    sigma_out = np.array([row["sigma_output"] for row in rows])

    summary_csv = os.path.join(args.output, "sigma_vs_funnel_width.csv")
    with open(summary_csv, "w", newline="") as csv_file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV written to {summary_csv}")

    print(f"\n{'v (half-width)':>15} {'sigma_in':>10} {'sigma_out':>10} "
          f"{'success':>8}")
    for row in rows:
        print(f"{row['funnel_half_width']:>15.4f} "
              f"{row['sigma_input']:>10.4f} {row['sigma_output']:>10.4f} "
              f"{row['n_success']:>8}")

    # --- Stage 3: the one static figure ---
    plt.figure(figsize=(10, 6))
    plt.plot(half_widths, sigma_out, "o-", color="#3498db",
             label=r"$\sigma$(output): std of final $x$")
    plt.plot(half_widths, sigma_in, "--", color="#e74c3c", alpha=0.8,
             label=r"$\sigma$(input) $= v/\sqrt{3}$ (uniform reference)")
    plt.xlabel("Funnel half-width $v$ (drop window $(-v, +v)$)", fontsize=12)
    plt.ylabel(r"$\sigma$", fontsize=12)
    plt.title(f"Output spread vs. funnel width "
              f"({args.balls} balls per frame, {len(rows)} frames)",
              fontsize=14, fontweight="bold")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    figure_path = os.path.join(args.output, "sigma_vs_funnel_width.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary figure written to {figure_path}")


if __name__ == "__main__":
    main()
