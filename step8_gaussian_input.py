# Step 8 of the science experiment queue (PLAN.md, section 7):
#
#   Gaussian input through the widest funnel - side-by-side sigma comparison.
#
# One full GPU batch through the galton_cuda package, but with the Stage-3
# input sampling changed from x ~ U(-v, +v) to x ~ N(0, sigma_in) via the
# x_sampler hook in run_sweep()/run_frame(). The funnel stays at its widest
# (the X_DROP_RANGE sweep maximum, half-width 47.0), realized as a degenerate
# single-frame sweep so the per-frame machinery (GPU batch, histogram, Q-Q
# plot, verification walk, CSV) runs unchanged.
#
# The aggregation stage then reads the frame CSV and emits ONE figure with
# two panels side by side:
#   left  = the input distribution (histogram of the sampled Gaussian drop
#           positions with the Gaussian curve overlaid),
#   right = the output distribution (histogram of final_x),
# both with a shaded +/-sigma band about the mean and the numerical sigma
# value printed on the image, so the input-sigma -> output-sigma relationship
# can be read directly off the figure. First concrete data point for goal 5b.
#
# Run inside the CUDA container from the repository root (see CUDA_GUIDE.md):
#
#     docker run --rm --gpus all -v "C:/Users/Enda/Data/Code/galton:/work" -w /work galton-cuda python3 step8_gaussian_input.py
#
# Smoke test (same env knobs as python3 -m galton_cuda):
#
#     ... -e GALTON_BALLS=200 ... python3 step8_gaussian_input.py

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, matches galton_cuda.visualization
import matplotlib.pyplot as plt

from galton_cuda import run_sweep

# Widest funnel half-width: the maximum of the X_DROP_RANGE sweep (step 7).
WIDEST_FUNNEL_HALF_WIDTH = 47.0

# Default input sigma: 47/3, so that +/-3 sigma of the Gaussian input spans
# the widest funnel half-width and the tails stay on the board.
DEFAULT_SIGMA_IN = WIDEST_FUNNEL_HALF_WIDTH / 3.0


def read_frame_csv(csv_filename):
    """Read one frame's CSV: the sampled input x and the successful final_x.

    Returns (input_x, output_x) as float arrays; output_x holds only the
    balls whose status is 'success'.
    """
    input_x, output_x = [], []
    with open(csv_filename, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            input_x.append(float(row["initial_x"]))
            if row["status"] == "success":
                output_x.append(float(row["final_x"]))
    return np.asarray(input_x), np.asarray(output_x)


def sigma_panel(ax, data, color, label, curve_sigma=None):
    """One histogram panel with a shaded +/-sigma band and the sigma printed.

    If curve_sigma is given, the corresponding Gaussian pdf N(mean, sigma) is
    overlaid (used for the input panel, whose shape should be Gaussian).
    """
    mean = data.mean()
    sigma = data.std(ddof=1)

    ax.hist(data, bins=100, density=True, color=color, edgecolor='none',
            alpha=0.8, label=label)
    ax.axvline(mean, color='#7f8c8d', linestyle=':', alpha=0.7)

    # Shaded +/-sigma band about the mean, and the numerical sigma on the image
    ax.axvspan(mean - sigma, mean + sigma, color='#e74c3c', alpha=0.25,
               label=r'$\pm\sigma$ band')
    ax.text(0.02, 0.95, rf'$\sigma = {sigma:.4f}$',
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if curve_sigma is not None:
        xs = np.linspace(data.min(), data.max(), 500)
        ax.plot(xs, (1.0 / (curve_sigma * np.sqrt(2.0 * np.pi)))
                * np.exp(-0.5 * ((xs - mean) / curve_sigma) ** 2),
                color='#2c3e50', linewidth=2,
                label=r'$\mathcal{N}(\bar{x},\ \sigma)$ curve')

    ax.set_ylabel('Density', fontsize=12)
    ax.grid(linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    return sigma


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="step8_gaussian_input",
        description="Step 8: Gaussian input x ~ N(0, sigma_in) through the "
                    "widest funnel (half-width 47.0), side-by-side "
                    "input/output sigma figure "
                    "(run inside the CUDA container - see CUDA_GUIDE.md).")
    parser.add_argument("--sigma-in", type=float, default=DEFAULT_SIGMA_IN,
                        help=f"std of the Gaussian input drop positions "
                             f"(default: {DEFAULT_SIGMA_IN:.4f} = 47/3, so "
                             f"3*sigma spans the widest funnel half-width)")
    parser.add_argument("--balls", type=int,
                        default=int(os.environ.get("GALTON_BALLS", 2000)),
                        help="balls in the batch (default: 2000, or "
                             "$GALTON_BALLS)")
    parser.add_argument("--bins", type=int, default=200,
                        help="bins for the engine's per-frame histogram "
                             "(default: 200)")
    parser.add_argument("--output", default=os.path.join("figures", "step8_gaussian_input"),
                        help="output directory (default: figures/step8_gaussian_input)")
    args = parser.parse_args(argv)

    # --- Stage 1: one GPU batch, Gaussian input through the widest funnel ---
    # Degenerate single-frame X_DROP_RANGE sweep (low = high = 47.0) so the
    # existing per-frame machinery runs unchanged; the sampler replaces the
    # uniform drop-window sampling with x ~ N(0, sigma_in).
    def gaussian_sampler(n_balls, drop_min, drop_max):
        return np.random.normal(0.0, args.sigma_in, size=n_balls)

    print(f"Gaussian input x ~ N(0, {args.sigma_in:.4f}) through the widest "
          f"funnel (half-width {WIDEST_FUNNEL_HALF_WIDTH})")
    results = run_sweep(
        sweep_param="X_DROP_RANGE",
        low=WIDEST_FUNNEL_HALF_WIDTH,
        high=WIDEST_FUNNEL_HALF_WIDTH,
        steps=1,
        n_balls=args.balls,
        n_bins=args.bins,
        output_dir=args.output,
        x_sampler=gaussian_sampler,
    )
    result = results[0]

    # --- Stage 2: read the frame CSV and compute both sigmas ---
    input_x, output_x = read_frame_csv(result["csv_filename"])
    if input_x.size != args.balls:
        print(f"WARNING: CSV holds {input_x.size} input positions, "
              f"expected {args.balls}")
    if output_x.size < 2:
        raise RuntimeError(
            f"only {output_x.size} successful balls - output sigma undefined")

    sigma_in = input_x.std(ddof=1)   # std of the sampled Gaussian input
    sigma_out = output_x.std(ddof=1)  # std of the final_x the board produced

    summary_csv = os.path.join(args.output, "gaussian_input_sigma_summary.csv")
    with open(summary_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "sigma_in_requested", "sigma_in_sampled", "sigma_output",
            "ratio_sigma_out_over_sigma_in", "n_success", "n_balls",
            "funnel_half_width"])
        writer.writeheader()
        writer.writerow({
            "sigma_in_requested": f"{args.sigma_in:.6f}",
            "sigma_in_sampled": f"{sigma_in:.6f}",
            "sigma_output": f"{sigma_out:.6f}",
            "ratio_sigma_out_over_sigma_in": f"{sigma_out / sigma_in:.6f}",
            "n_success": output_x.size,
            "n_balls": args.balls,
            "funnel_half_width": WIDEST_FUNNEL_HALF_WIDTH,
        })
    print(f"\nSummary CSV written to {summary_csv}")

    print(f"\ninput : sigma = {sigma_in:.4f} (requested {args.sigma_in:.4f}), "
          f"n = {input_x.size}")
    print(f"output: sigma = {sigma_out:.4f}, n = {output_x.size} "
          f"({100.0 * output_x.size / args.balls:.1f}% success)")
    print(f"sigma(output) / sigma(input) = {sigma_out / sigma_in:.4f}")

    # --- Stage 3: the one side-by-side figure ---
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(14, 6))

    measured_in = sigma_panel(
        ax_in, input_x, color='#3498db',
        label=f'Input drop positions (n = {input_x.size})',
        curve_sigma=sigma_in)
    measured_out = sigma_panel(
        ax_out, output_x, color='#27ae60',
        label=f'Output final_x (n = {output_x.size})')

    ax_in.set_xlabel('Input x (drop position)', fontsize=12)
    ax_in.set_title(f'Input: Gaussian drops '
                    rf'$x \sim \mathcal{{N}}(0,\ {args.sigma_in:.2f})$',
                    fontsize=13, fontweight='bold')
    ax_out.set_xlabel('Output x (detection line)', fontsize=12)
    ax_out.set_title('Output: final positions after the board',
                     fontsize=13, fontweight='bold')

    fig.suptitle(
        f'Galton board: Gaussian input through the widest funnel '
        f'(half-width {WIDEST_FUNNEL_HALF_WIDTH})\n'
        rf'$\sigma_{{in}} = {measured_in:.4f}$'
        rf'$\ \rightarrow\ $'
        rf'$\sigma_{{out}} = {measured_out:.4f}$'
        rf'$\ \ (\sigma_{{out}}/\sigma_{{in}} = {measured_out / measured_in:.3f})$',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    figure_path = os.path.join(args.output, "gaussian_input_vs_output_sigma.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary figure written to {figure_path}")


if __name__ == "__main__":
    main()
