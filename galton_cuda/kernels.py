# CUDA device code: closed-form quartic solver, trajectory kernel, batch launcher.
#
# Architecture (carried over from the original galton_cuda.py):
#   - One GPU thread per ball. The entire trajectory loop (time-to-line solver,
#     wall solver, peg collision search, reflections) runs on the device, so a
#     whole frame of n_balls is a single kernel launch instead of a sequential
#     CPU loop.
#   - np.roots (used on the CPU for the quartic time-to-peg-impact polynomial)
#     is not available inside CUDA kernels. It is replaced by an in-kernel
#     closed-form (Ferrari) quartic solver (_quartic_smallest_root). That is
#     ~200 double-precision flops with no data-dependent iteration, versus
#     ~13k flops for an iterative complex-arithmetic root finder - which
#     matters a lot because consumer GeForce GPUs execute float64 at 1/64th
#     of the float32 rate, so an iterative float64 solver dominates the
#     kernel time otherwise. Candidate roots get two Newton polish steps on
#     the original polynomial to recover full double accuracy.
#   - The spatial grid stores each peg in exactly ONE cell (the cell
#     containing its center). Queries expand their swept bounding box by the
#     peg radius, so every reachable peg is still found - but, unlike
#     multi-cell registration, no peg is ever tested twice.

import math
import time

import numpy as np
from numba import cuda


def require_cuda():
    """Raise a helpful error unless a CUDA device is visible."""
    if not cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. Run this inside the CUDA container with "
            "--gpus all (see Dockerfile), or from WSL with the NVIDIA driver set up.")
    cuda.detect()  # prints device name / compute capability


# --- DEVICE-SIDE QUARTIC SOLVER (replaces np.roots inside the kernel) ---
@cuda.jit(device=True, inline=True)
def _cbrt(x):
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


@cuda.jit(device=True, inline=True)
def _newton_polish(c4, c3, c2, c1, c0, t):
    """Refine an approximate root with two Newton steps on the quartic."""
    for _ in range(2):
        f = (((c4 * t + c3) * t + c2) * t + c1) * t + c0
        fp = ((4.0 * c4 * t + 3.0 * c3) * t + 2.0 * c2) * t + c1
        if fp == 0.0:
            break
        step = f / fp
        t = t - step
        if abs(step) < 1e-14 * (1.0 + abs(t)):
            break
    return t


@cuda.jit(device=True, inline=True)
def _try_root(c4, c3, c2, c1, c0, t, t_lower, t_upper, best):
    """Keep t if it is a real root in (t_lower, t_upper) smaller than best."""
    # NaN candidates (degenerate coefficients) fail every comparison and are
    # dropped here, so numerical garbage can never become a collision time.
    if not (t_lower < t < t_upper):
        return best
    t = _newton_polish(c4, c3, c2, c1, c0, t)
    if not (t_lower < t < t_upper):
        return best
    if best < 0.0 or t < best:
        return t
    return best


@cuda.jit(device=True, inline=True)
def _quartic_smallest_root(c4, c3, c2, c1, c0, t_lower, t_upper):
    """
    Smallest real root of c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0 lying in
    the open interval (t_lower, t_upper). Returns the root, or -1.0 if none.

    Closed-form solution: normalize to monic, depress the quartic
    (t = u - a/4 -> u^4 + p*u^2 + q*u + r), solve the resolvent cubic
    (Cardano / trigonometric branch on its discriminant) for its largest real
    root z >= 0, and factor into two real quadratics via alpha = sqrt(z).
    Each candidate is Newton-polished on the original polynomial, so the
    closed-form round-off (worst near double roots) is recovered.

    All arithmetic is real float64: no iteration whose trip count depends on
    the data (an earlier Durand-Kerner port cost ~13k flops per call and, at
    the 1/64 float64 rate of consumer GPUs, dominated the whole kernel).
    """
    a = c3 / c4
    b = c2 / c4
    c = c1 / c4
    d = c0 / c4

    # Depressed quartic: t = u - a/4  ->  u^4 + p*u^2 + q*u + r
    a2 = a * a
    p = b - 0.375 * a2                       # b - 3*a^2/8
    q = 0.125 * a * a2 - 0.5 * a * b + c     # a^3/8 - a*b/2 + c
    r = -0.01171875 * a2 * a2 + 0.0625 * a2 * b - 0.25 * a * c + d

    # Resolvent cubic: z^3 + 2p*z^2 + (p^2 - 4r)*z - q^2 = 0.
    # Its largest real root is always >= 0 for a real-coefficient quartic.
    A = 2.0 * p
    B = p * p - 4.0 * r
    C = -q * q
    P = B - A * A / 3.0                      # depressed cubic w^3 + P*w + Q
    Q = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C
    disc = 0.25 * Q * Q + (P * P * P) / 27.0

    if disc >= 0.0:
        # One real root (Cardano); real cube roots handle negative arguments.
        sq = math.sqrt(disc)
        z = _cbrt(-0.5 * Q + sq) + _cbrt(-0.5 * Q - sq) - A / 3.0
    else:
        # Three real roots; the largest is the k = 0 trigonometric branch.
        m = 2.0 * math.sqrt(-P / 3.0)
        arg = (3.0 * Q / (2.0 * P)) * math.sqrt(-3.0 / P)
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        z = m * math.cos(math.acos(arg) / 3.0) - A / 3.0

    best = -1.0

    if z > 0.0:
        # Factor: (u^2 + alpha*u + beta) * (u^2 - alpha*u + gamma)
        alpha = math.sqrt(z)
        if alpha > 0.0:
            qa = q / alpha
        else:
            qa = 0.0
        s = p + z
        beta = 0.5 * (s - qa)
        gamma = 0.5 * (s + qa)

        # u^2 + alpha*u + beta = 0
        d1 = alpha * alpha - 4.0 * beta
        if d1 >= 0.0:
            sq1 = math.sqrt(d1)
            best = _try_root(c4, c3, c2, c1, c0, -0.5 * (alpha - sq1) - 0.25 * a,
                             t_lower, t_upper, best)
            best = _try_root(c4, c3, c2, c1, c0, -0.5 * (alpha + sq1) - 0.25 * a,
                             t_lower, t_upper, best)

        # u^2 - alpha*u + gamma = 0
        d2 = alpha * alpha - 4.0 * gamma
        if d2 >= 0.0:
            sq2 = math.sqrt(d2)
            best = _try_root(c4, c3, c2, c1, c0, 0.5 * (alpha - sq2) - 0.25 * a,
                             t_lower, t_upper, best)
            best = _try_root(c4, c3, c2, c1, c0, 0.5 * (alpha + sq2) - 0.25 * a,
                             t_lower, t_upper, best)
    else:
        # z <= 0 only through round-off when q ~ 0: the quartic is
        # effectively biquadratic, u^4 + p*u^2 + r = 0.
        dw = p * p - 4.0 * r
        if dw >= 0.0:
            sqw = math.sqrt(dw)
            w1 = -0.5 * (p + sqw)           # u^2 = w
            w2 = -0.5 * (p - sqw)
            if w1 >= 0.0:
                u = math.sqrt(w1)
                best = _try_root(c4, c3, c2, c1, c0, u - 0.25 * a,
                                 t_lower, t_upper, best)
                best = _try_root(c4, c3, c2, c1, c0, -u - 0.25 * a,
                                 t_lower, t_upper, best)
            if w2 >= 0.0:
                u = math.sqrt(w2)
                best = _try_root(c4, c3, c2, c1, c0, u - 0.25 * a,
                                 t_lower, t_upper, best)
                best = _try_root(c4, c3, c2, c1, c0, -u - 0.25 * a,
                                 t_lower, t_upper, best)

    return best


# --- CORE CUDA SIMULATION KERNEL: one thread per ball ---
@cuda.jit
def simulate_trajectory_cuda(circles_centers, grid_data, grid_counts,
                             min_cx, min_cy, cell_size, radius,
                             x_inits, final_x, status,
                             e, y_target, wall_distance, g):
    ball = cuda.grid(1)
    if ball >= x_inits.shape[0]:
        return

    x_imp = x_inits[ball]
    y_imp = 0.0
    vx = 0.0
    vy = 0.0

    grid_w = grid_counts.shape[0]
    grid_h = grid_counts.shape[1]

    step = 0
    while step <= 1000:
        step += 1

        # 1. Time-to-line solver
        a_line = 0.5 * g
        b_line = -vy
        c_line = y_target - y_imp
        discriminant = b_line * b_line - 4.0 * a_line * c_line

        t_line = math.inf
        if discriminant >= 0.0:
            sq = math.sqrt(discriminant)
            t1 = (-b_line + sq) / (2.0 * a_line)
            t2 = (-b_line - sq) / (2.0 * a_line)
            if 1e-5 < t1 < t_line:
                t_line = t1
            if 1e-5 < t2 < t_line:
                t_line = t2

        # 2. Time-to-wall solver
        t_wall = math.inf
        hit_wall_side = 0  # 1: left, 2: right
        if vx < 0.0:
            t_left = (-wall_distance - x_imp) / vx
            if t_left > 1e-5:
                t_wall = t_left
                hit_wall_side = 1
        elif vx > 0.0:
            t_right = (wall_distance - x_imp) / vx
            if t_right > 1e-5:
                t_wall = t_right
                hit_wall_side = 2

        # 3. Dynamic bounding box + dense spatial grid query
        t_max = t_line if t_line < t_wall else t_wall
        if t_max == math.inf:
            break

        x0 = x_imp
        x1 = x_imp + vx * t_max
        x_min = x0 if x0 < x1 else x1
        x_max = x0 if x0 > x1 else x1

        y0 = y_imp
        y1 = y_imp + vy * t_max - 0.5 * g * t_max * t_max
        y_min = y0 if y0 < y1 else y1
        y_max = y0 if y0 > y1 else y1

        t_vertex = vy / g
        if 0.0 < t_vertex < t_max:
            y_vertex = y_imp + (vy * vy) / (2.0 * g)
            if y_vertex > y_max:
                y_max = y_vertex

        cx_start = int(math.floor((x_min - radius) / cell_size))
        cx_end = int(math.floor((x_max + radius) / cell_size))
        cy_start = int(math.floor((y_min - radius) / cell_size))
        cy_end = int(math.floor((y_max + radius) / cell_size))

        min_t_circle = math.inf
        next_peg_idx = -1

        # 4. Local analytical scanning loop. The grid stores each peg in
        #    exactly one cell (see build_spatial_grid), so every peg in the
        #    expanded swept box is tested exactly once - no dedup needed.
        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                idx_x = cx - min_cx
                idx_y = cy - min_cy
                if 0 <= idx_x < grid_w and 0 <= idx_y < grid_h:
                    count = grid_counts[idx_x, idx_y]
                    for k in range(count):
                        circle_idx = grid_data[idx_x, idx_y, k]
                        xc = circles_centers[circle_idx, 0]
                        yc = circles_centers[circle_idx, 1]
                        dx = x_imp - xc
                        dy = y_imp - yc

                        c3 = -vy * g
                        c2 = (vx * vx) + (vy * vy) - (dy * g)
                        c1 = 2.0 * (dx * vx + dy * vy)
                        c0 = (dx * dx) + (dy * dy) - (radius * radius)

                        t_val = _quartic_smallest_root(
                            0.25 * g * g, c3, c2, c1, c0, 1e-5, min_t_circle)
                        if t_val > 0.0:
                            min_t_circle = t_val
                            next_peg_idx = circle_idx

        # 5. Intersect evaluation mapping
        t_end = math.inf
        next_event = -1  # 0: line, 1: peg, 2: wall

        if t_line < t_end:
            t_end = t_line
            next_event = 0
        if min_t_circle < t_end:
            t_end = min_t_circle
            next_event = 1
        if t_wall < t_end:
            t_end = t_wall
            next_event = 2

        if next_event == -1:
            break

        x_next = x_imp + vx * t_end
        y_next = y_imp + vy * t_end - 0.5 * g * t_end * t_end

        if next_event == 0:
            final_x[ball] = x_next
            status[ball] = 1
            return

        vx_before = vx
        vy_before = vy - g * t_end

        if next_event == 1:
            xc = circles_centers[next_peg_idx, 0]
            yc = circles_centers[next_peg_idx, 1]
            nx = (x_next - xc) / radius
            ny = (y_next - yc) / radius
            v_dot_n = vx_before * nx + vy_before * ny
            vx_after = vx_before - (1 + e) * v_dot_n * nx
            vy_after = vy_before - (1 + e) * v_dot_n * ny
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next + (ny * 1e-4)

        elif next_event == 2:
            vx_after = -e * vx_before
            vy_after = vy_before
            nx = 1.0 if hit_wall_side == 1 else -1.0
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next

        vx, vy = vx_after, vy_after

    # Fell out of the loop: exceeded step budget or no further event.
    final_x[ball] = np.nan
    status[ball] = 0


# --- HOST-SIDE BATCH LAUNCHER ---
def simulate_batch_cuda(circles_centers, grid_data, grid_counts, min_cx, min_cy,
                        cell_size, radius, x_inits, e, h_init, rows, spacing,
                        h_final, wall_distance, g=9.81):
    """
    Runs every ball's trajectory in parallel on the GPU (one thread per ball).
    Returns (final_x, status) host arrays: status 1 = reached the detection line.
    """
    n_balls = x_inits.shape[0]
    y_target = -h_init - (rows * spacing * math.cos(math.radians(30.0))) - h_final

    d_centers = cuda.to_device(circles_centers)
    d_grid_data = cuda.to_device(grid_data)
    d_grid_counts = cuda.to_device(grid_counts)
    d_x = cuda.to_device(x_inits)
    d_final = cuda.device_array(n_balls, dtype=np.float64)
    d_status = cuda.device_array(n_balls, dtype=np.int32)

    # Small blocks: a 2000-ball frame is only ~32 blocks of 64 threads, which
    # already under-fills the GPU's SMs - 256-thread blocks would leave even
    # more of the device idle at these problem sizes.
    threads_per_block = 64
    blocks = (n_balls + threads_per_block - 1) // threads_per_block

    start = time.perf_counter()
    simulate_trajectory_cuda[blocks, threads_per_block](
        d_centers, d_grid_data, d_grid_counts,
        int(min_cx), int(min_cy), cell_size, radius,
        d_x, d_final, d_status,
        e, y_target, wall_distance, g)
    cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"[GPU] {n_balls} trajectories in {elapsed:.3f} s "
          f"({blocks} blocks x {threads_per_block} threads)")

    return d_final.copy_to_host(), d_status.copy_to_host()
