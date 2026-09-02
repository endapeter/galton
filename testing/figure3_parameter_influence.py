#!/usr/bin/env python3
"""
figure3_parameter_influence.py

Figure 3 of the Brownian Galton-board study (both panels in ONE image).

Panel a: influence of the effective height H on the variance.
Panel b: influence of the diffusion coefficient D_z on the variance.

Both panels plot the theoretical model

    sigma^2(W; H, D_z) = W^2/12 + 2 D_z H

against the funnel width W = 2a. Compared with the old multi-panel version,
H and D_z are sampled with many more levels (smaller spacing between
neighbouring curves), rendered as colormaps with colorbars.

H_ref and D_z are taken from the JSON summary written by
test_brownian_model.py unless overridden with --h-ref / --dz.

Outputs (written to --output):
    figure3_parameter_influence.png / .pdf

Typical use, inside the CUDA container (see testing/README.md):
    python3 testing/figure3_parameter_influence.py
    python3 testing/figure3_parameter_influence.py --h-points 16 --dz-points 16
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from test_brownian_model import DEFAULTS, fmt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Figure 3: theoretical influence of H (panel a) and D_z (panel b) "
            "on the variance as a function of funnel width W = 2a."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-json",
        default=f"{DEFAULTS['output']}/brownian_model_summary.json",
        help="JSON summary from test_brownian_model.py used for H_ref and D_z.",
    )
    parser.add_argument(
        "--h-ref", type=float, default=None,
        help="Reference height H_ref. Overrides the JSON summary.",
    )
    parser.add_argument(
        "--dz", type=float, default=None,
        help="Diffusion coefficient D_z. Overrides the JSON summary.",
    )

    level_group = parser.add_argument_group("parameter levels")
    level_group.add_argument(
        "--h-points", type=int, default=12,
        help="Number of H levels in panel a (more points = finer spacing).",
    )
    level_group.add_argument(
        "--dz-points", type=int, default=12,
        help="Number of D_z levels in panel b (more points = finer spacing).",
    )
    level_group.add_argument(
        "--min-factor", type=float, default=0.5,
        help="Smallest level as a multiple of the reference value.",
    )
    level_group.add_argument(
        "--max-factor", type=float, default=2.0,
        help="Largest level as a multiple of the reference value.",
    )

    funnel_group = parser.add_argument_group("funnel-width axis")
    funnel_group.add_argument("--funnel-min", type=float, default=0.1)
    funnel_group.add_argument("--funnel-max", type=float, default=8.0)
    funnel_group.add_argument("--funnel-points", type=int, default=200)

    parser.add_argument("--output", default=DEFAULTS["output"])

    args = parser.parse_args(argv)
    if args.h_points < 2:
        parser.error("--h-points must be at least 2.")
    if args.dz_points < 2:
        parser.error("--dz-points must be at least 2.")
    if args.min_factor < 0.0:
        parser.error("--min-factor must be non-negative.")
    if args.max_factor <= args.min_factor:
        parser.error("--max-factor must exceed --min-factor.")
    if args.funnel_min <= 0.0:
        parser.error("--funnel-min must be positive.")
    if args.funnel_max <= args.funnel_min:
        parser.error("--funnel-max must exceed --funnel-min.")
    if args.funnel_points < 2:
        parser.error("--funnel-points must be at least 2.")
    if args.h_ref is not None and args.h_ref <= 0.0:
        parser.error("--h-ref must be positive.")
    if args.dz is not None and args.dz <= 0.0:
        parser.error("--dz must be positive.")
    return args


def resolve_reference(args):
    """Determine H_ref and D_z from CLI overrides or the JSON summary."""
    h_ref = args.h_ref
    dz = args.dz
    sources = {"h_ref": "--h-ref", "dz": "--dz"}
    if h_ref is not None and dz is not None:
        return h_ref, dz, sources

    payload = {}
    path = Path(args.summary_json)
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[warn] could not parse {path}: {exc}")
    else:
        print(f"[warn] summary file {path} not found.")

    if dz is None:
        fit = payload.get("fit", {})
        for key in ("Dz_free", "Dz_fixed_theory", "Dz_fixed_sample"):
            try:
                value = float(fit.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                dz = value
                sources["dz"] = f"{path} -> fit.{key}"
                break

    if h_ref is None:
        H_values = [
            float(r["H"]) for r in payload.get("records", []) if "H" in r
        ]
        if H_values:
            h_ref = float(np.mean(H_values))
            sources["h_ref"] = f"{path} -> mean H of records"

    if dz is None or h_ref is None:
        missing = []
        if dz is None:
            missing.append("--dz")
        if h_ref is None:
            missing.append("--h-ref")
        raise SystemExit(
            "Could not determine " + " and ".join(missing) + ". "
            "Run testing/test_brownian_model.py first (it writes "
            f"{args.summary_json}) or pass the missing values explicitly."
        )
    return h_ref, dz, sources


def make_plot(out_dir, h_ref, dz, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except Exception as exc:
        print(f"[plot] skipped because matplotlib is unavailable: {exc}")
        return None, None

    W = np.linspace(args.funnel_min, args.funnel_max, args.funnel_points)
    baseline = W ** 2 / 12.0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # Panel a: H levels at fixed D_z.
    h_levels = np.linspace(
        args.min_factor * h_ref, args.max_factor * h_ref, args.h_points
    )
    cmap_h = plt.get_cmap("viridis")
    norm_h = Normalize(float(h_levels[0]), float(h_levels[-1]))
    for h in h_levels:
        ax_a.plot(W, baseline + 2.0 * dz * h, color=cmap_h(norm_h(h)), lw=1.8)
    ax_a.plot(
        W, baseline, "k--", lw=2.0, label=r"$W^2/12$ baseline ($H = 0$)"
    )
    ax_a.set_xlabel(r"Funnel width $W = 2a$", fontsize=12)
    ax_a.set_ylabel(r"Variance $\sigma^2$", fontsize=12)
    ax_a.set_title(r"3a. Influence of $H$ ($D_z$ fixed)", fontsize=13)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=10)
    sm_h = ScalarMappable(cmap=cmap_h, norm=norm_h)
    sm_h.set_array([])
    fig.colorbar(sm_h, ax=ax_a, label="Effective height $H$")

    # Panel b: D_z levels at fixed H_ref.
    d_levels = np.linspace(
        args.min_factor * dz, args.max_factor * dz, args.dz_points
    )
    cmap_d = plt.get_cmap("plasma")
    norm_d = Normalize(float(d_levels[0]), float(d_levels[-1]))
    for d in d_levels:
        ax_b.plot(W, baseline + 2.0 * d * h_ref, color=cmap_d(norm_d(d)), lw=1.8)
    ax_b.plot(
        W, baseline, "k--", lw=2.0, label=r"$W^2/12$ baseline ($D_z = 0$)"
    )
    ax_b.set_xlabel(r"Funnel width $W = 2a$", fontsize=12)
    ax_b.set_ylabel(r"Variance $\sigma^2$", fontsize=12)
    ax_b.set_title(r"3b. Influence of $D_z$ ($H$ fixed)", fontsize=13)
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=10)
    sm_d = ScalarMappable(cmap=cmap_d, norm=norm_d)
    sm_d.set_array([])
    fig.colorbar(sm_d, ax=ax_b, label=r"Diffusion coefficient $D_z$")

    fig.tight_layout()
    png_path = out_dir / "figure3_parameter_influence.png"
    pdf_path = out_dir / "figure3_parameter_influence.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return png_path, pdf_path


def main(argv=None):
    args = parse_args(argv)
    h_ref, dz, sources = resolve_reference(args)

    print("Figure 3 configuration:")
    print(f"  H_ref      : {fmt(h_ref)}  ({sources['h_ref']})")
    print(f"  D_z        : {fmt(dz)}  ({sources['dz']})")
    print(
        f"  H levels   : {args.h_points} between "
        f"{fmt(args.min_factor * h_ref)} and {fmt(args.max_factor * h_ref)}"
    )
    print(
        f"  D_z levels : {args.dz_points} between "
        f"{fmt(args.min_factor * dz)} and {fmt(args.max_factor * dz)}"
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = make_plot(out_dir, h_ref, dz, args)

    if png_path is not None:
        print("\nOutput files:")
        print(f"  Plot (PNG) : {png_path}")
        print(f"  Plot (PDF) : {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())