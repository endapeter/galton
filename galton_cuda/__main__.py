# Command-line entry point: python3 -m galton_cuda ...
#
# Batch size / sweep length can still be trimmed for quick tests with the
# same environment variables the old script used:
#   docker run ... -e GALTON_BALLS=200 -e GALTON_INTERVALS=2 ... \
#       galton-cuda python3 -m galton_cuda

import argparse
import json
import os

from .sweep import DEFAULT_PARAMS, PARAM_NAMES, run_sweep


def _parse_value(raw):
    """Parse a --set value: a number, or JSON (for tuples like [-1, 1])."""
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"cannot parse {raw!r} as a number or a JSON value")
    if isinstance(val, list):
        return tuple(val)
    return val


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="galton_cuda",
        description="GPU Galton board parameter sweep "
                    "(run inside the CUDA container - see CUDA_GUIDE.md).")
    parser.add_argument("--sweep", default="X_DROP_RANGE", choices=sorted(PARAM_NAMES),
                        help="parameter to sweep (default: X_DROP_RANGE, the "
                             "drop-funnel half-width)")
    parser.add_argument("--low", type=float, default=None,
                        help="first swept value (default: 0.05 for "
                             "X_DROP_RANGE, otherwise required)")
    parser.add_argument("--high", type=float, default=None,
                        help="last swept value (default: 47.0 for "
                             "X_DROP_RANGE, otherwise required)")
    parser.add_argument("--steps", type=int,
                        default=int(os.environ.get("GALTON_INTERVALS", 10)),
                        help="number of iterative steps / sweep frames "
                             "(default: 10, or $GALTON_INTERVALS)")
    parser.add_argument("--balls", type=int,
                        default=int(os.environ.get("GALTON_BALLS", 2000)),
                        help="balls per frame (default: 2000, or $GALTON_BALLS)")
    parser.add_argument("--bins", type=int, default=200,
                        help="histogram bins per frame (default: 200)")
    parser.add_argument("--output", default="figures",
                        help="output directory for PNGs and CSVs (default: figures)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        dest="overrides",
                        help="hold a parameter constant, overriding its default "
                             "(repeatable), e.g. --set N_ROWS=30 or "
                             "--set X_DROP_RANGE=[-0.05,0.05]")
    parser.add_argument("--list-params", action="store_true",
                        help="print all parameter names with their defaults and exit")
    args = parser.parse_args(argv)

    if args.list_params:
        for name, value in sorted(DEFAULT_PARAMS.items()):
            print(f"{name} = {value!r}")
        return

    low, high = args.low, args.high
    if low is None or high is None:
        if args.sweep == "X_DROP_RANGE":
            low = 0.05 if low is None else low
            high = 47.0 if high is None else high
        else:
            parser.error(f"--low and --high are required when sweeping "
                         f"{args.sweep} (they only have defaults for "
                         f"X_DROP_RANGE)")

    fixed = {}
    for item in args.overrides:
        name, sep, raw = item.partition("=")
        if not sep or not name:
            parser.error(f"--set expects NAME=VALUE, got {item!r}")
        if name not in PARAM_NAMES:
            parser.error(f"unknown parameter {name!r} in --set "
                         f"(valid: {sorted(PARAM_NAMES)}, or use --list-params)")
        if name == args.sweep:
            parser.error(f"{name} is being swept - remove it from --set")
        fixed[name] = _parse_value(raw)

    run_sweep(
        sweep_param=args.sweep,
        low=low,
        high=high,
        steps=args.steps,
        fixed=fixed,
        n_balls=args.balls,
        n_bins=args.bins,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
