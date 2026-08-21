# CUDA port of galton_ultra.py.
#
# Architecture:
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
#     multi-cell registration, no peg is ever tested twice. This replaces the
#     per-thread "seen pegs" dedup array of the CPU version, which would need
#     per-thread local memory on the device.
#   - Rendering (histogram, Q-Q plot, verification walk) stays on the CPU with
#     matplotlib.
#
# Run inside the CUDA Docker container (see Dockerfile / requirements.txt):
#   docker run --rm --gpus all -v "C:/Users/Enda/Data/Code/galton:/work" -w /work \
#       galton-cuda python3 galton_cuda.py

import os
import csv
import math
import time
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # headless rendering inside the container
import matplotlib.pyplot as plt
from numba import cuda

# Batch size / sweep length can be trimmed for quick tests without editing:
#   docker run ... -e GALTON_BALLS=200 -e GALTON_INTERVALS=2 ... galton-cuda python3 galton_cuda.py
N_BALLS_DEFAULT = int(os.environ.get("GALTON_BALLS", 2000))
INTERVALS_DEFAULT = int(os.environ.get("GALTON_INTERVALS", 10))


# --- GEOMETRY GENERATION ---
def generate_circle_geometry(n_first_row, n_rows, d, h_init, r_custom, WALL_PADDING=0):
    circles_data = []
    max_edge_distance = 0
    row_height_step = d * (3**0.5) / 2

    for row_idx in range(n_rows):
        is_even_row = row_idx % 2 == 0
        row_count = n_first_row if is_even_row else n_first_row - 1
        y_center = -h_init - (row_idx * row_height_step)
        start_x = -((row_count - 1) * d) / 2

        for col_idx in range(row_count):
            x_center = start_x + (col_idx * d)
            furthest_edge = abs(x_center) + r_custom
            if furthest_edge > max_edge_distance:
                max_edge_distance = furthest_edge

            circles_data.append({
                "center": (x_center, y_center),
                "radius": r_custom,
            })

    WALL_DISTANCE = max_edge_distance + WALL_PADDING
    return circles_data, WALL_DISTANCE


# --- SPATIAL HASHING (host side, plain numpy port of the Numba version) ---
def build_spatial_grid(circles_centers, radius, cell_size):
    """
    Creates dense 2D index arrays for O(1) local lookups:
    grid_data[x_cell, y_cell, k] = circle index (or -1), grid_counts[x_cell, y_cell] = n.

    Each peg is registered in exactly ONE cell: the cell containing its
    center. Consumers query the grid over a swept bounding box expanded by
    `radius`, so any peg whose disk can touch the sweep is still found, and
    no peg is ever returned twice (this is the device-side replacement for
    the CPU version's "seen pegs" dedup array).
    """
    if len(circles_centers) == 0:
        return np.zeros((1, 1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.int32), 0, 0

    min_x = np.min(circles_centers[:, 0]) - radius
    max_x = np.max(circles_centers[:, 0]) + radius
    min_y = np.min(circles_centers[:, 1]) - radius
    max_y = np.max(circles_centers[:, 1]) + radius

    min_cx = int(np.floor(min_x / cell_size))
    max_cx = int(np.floor(max_x / cell_size))
    min_cy = int(np.floor(min_y / cell_size))
    max_cy = int(np.floor(max_y / cell_size))

    grid_width = max_cx - min_cx + 1
    grid_height = max_cy - min_cy + 1

    # Pre-allocate 3D grid data: [x_cell, y_cell, max_pegs_per_cell]
    max_pegs = 20
    grid_data = np.full((grid_width, grid_height, max_pegs), -1, dtype=np.int32)
    grid_counts = np.zeros((grid_width, grid_height), dtype=np.int32)

    for i in range(len(circles_centers)):
        xc, yc = circles_centers[i]
        idx_x = int(np.floor(xc / cell_size)) - min_cx
        idx_y = int(np.floor(yc / cell_size)) - min_cy
        count = grid_counts[idx_x, idx_y]
        if count < max_pegs:
            grid_data[idx_x, idx_y, count] = i
            grid_counts[idx_x, idx_y] = count + 1

    return grid_data, grid_counts, min_cx, min_cy


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


# --- VISUALIZATION FUNCTION FOR A SINGLE RANDOM BALL (CPU, pure Python) ---
def simulate_and_visualize_trajectory_optimized(circles_data, circles_centers, grid_data, grid_counts, min_cx, min_cy,
                                                cell_size, radius, x_init, y_init, vx_init, vy_init, e, h_init, rows, spacing, h_final, WALL_DISTANCE, frame_index, g=9.81):
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    y_target = -h_init - (rows * spacing * np.cos(np.radians(30))) - h_final

    x_imp, y_imp = x_init, y_init
    vx, vy = vx_init, vy_init

    ax.scatter([x_imp], [y_imp], color='#2ecc71', marker='o', s=100, label='Release Point', edgecolor='black', zorder=5)

    c4 = 0.25 * (g**2)
    step = 0
    all_x, all_y = [x_imp], [y_imp]
    traj_lbl, collision_lbl = False, False

    n_circles = len(circles_centers)
    seen_centers = np.zeros(n_circles, dtype=np.bool_)

    while True:
        step += 1

        a_line = 0.5 * g
        b_line = -vy
        c_line = y_target - y_imp
        discriminant = b_line**2 - 4 * a_line * c_line

        t_line = np.inf
        if discriminant >= 0:
            t1 = (-b_line + np.sqrt(discriminant)) / (2 * a_line)
            t2 = (-b_line - np.sqrt(discriminant)) / (2 * a_line)
            if t1 > 1e-5 and t1 < t_line: t_line = t1
            if t2 > 1e-5 and t2 < t_line: t_line = t2

        t_wall = np.inf
        hit_wall_side = 0
        if vx < 0:
            t_left = (-WALL_DISTANCE - x_imp) / vx
            if t_left > 1e-5:
                t_wall = t_left
                hit_wall_side = 1
        elif vx > 0:
            t_right = (WALL_DISTANCE - x_imp) / vx
            if t_right > 1e-5:
                t_wall = t_right
                hit_wall_side = 2

        t_max = t_line
        if t_wall < t_max:
            t_max = t_wall

        if t_max == np.inf:
            break

        x0, x1 = x_imp, x_imp + vx * t_max
        x_min, x_max = min(x0, x1), max(x0, x1)
        y0 = y_imp
        y1 = y_imp + vy * t_max - 0.5 * g * (t_max**2)
        y_min, y_max = min(y0, y1), max(y0, y1)

        t_vertex = vy / g
        if 0 < t_vertex < t_max:
            y_vertex = y_imp + (vy**2) / (2 * g)
            if y_vertex > y_max: y_max = y_vertex

        cx_start = int(np.floor((x_min - radius) / cell_size))
        cx_end = int(np.floor((x_max + radius) / cell_size))
        cy_start = int(np.floor((y_min - radius) / cell_size))
        cy_end = int(np.floor((y_max + radius) / cell_size))

        seen_centers[:] = False
        min_t_circle = np.inf
        next_peg_idx = -1

        grid_w = grid_counts.shape[0]
        grid_h = grid_counts.shape[1]

        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                idx_x = cx - min_cx
                idx_y = cy - min_cy
                if 0 <= idx_x < grid_w and 0 <= idx_y < grid_h:
                    count = grid_counts[idx_x, idx_y]
                    for i in range(count):
                        circle_idx = grid_data[idx_x, idx_y, i]
                        if not seen_centers[circle_idx]:
                            seen_centers[circle_idx] = True
                            xc = circles_centers[circle_idx, 0]
                            yc = circles_centers[circle_idx, 1]
                            dx = x_imp - xc
                            dy = y_imp - yc

                            c3 = -vy * g
                            c2 = (vx**2) + (vy**2) - (dy * g)
                            c1 = 2.0 * (dx * vx + dy * vy)
                            c0 = (dx**2) + (dy**2) - (radius**2)

                            coeffs = np.array([c4, c3, c2, c1, c0], dtype=np.complex128)
                            roots = np.roots(coeffs)
                            for r_idx in range(len(roots)):
                                r = roots[r_idx]
                                if abs(r.imag) < 1e-6:
                                    t_val = r.real
                                    if 1e-5 < t_val < min_t_circle:
                                        min_t_circle = t_val
                                        next_peg_idx = circle_idx

        t_end = np.inf
        next_event = -1
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

        t_segment = np.linspace(0, t_end, num=50)
        x_segment = x_imp + vx * t_segment
        y_segment = y_imp + vy * t_segment - 0.5 * g * t_segment**2

        all_x.extend(x_segment)
        all_y.extend(y_segment)

        lbl_traj = 'Trajectory' if not traj_lbl else ""
        traj_lbl = True
        ax.plot(x_segment, y_segment, color='#2980b9', linewidth=2, label=lbl_traj, zorder=4)

        if next_event == 0:
            ax.scatter([x_segment[-1]], [y_segment[-1]], color='#2c3e50', marker='X', s=120,
                       label='Detection Intersection', edgecolor='black', zorder=5)
            break

        x_next, y_next = x_segment[-1], y_segment[-1]
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

            lbl_coll = 'Collisions' if not collision_lbl else ""
            collision_lbl = True
            ax.scatter([x_next], [y_next], color='#e74c3c', marker='o', s=30, edgecolor='black', label=lbl_coll, zorder=5)

            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next + (ny * 1e-4)

        elif next_event == 2:
            vx_after = -e * vx_before
            vy_after = vy_before

            lbl_coll = 'Collisions' if not collision_lbl else ""
            collision_lbl = True
            ax.scatter([x_next], [y_next], color='#f39c12', marker='s', s=30, edgecolor='black', label=lbl_coll, zorder=5)

            nx = 1.0 if hit_wall_side == 1 else -1.0
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next

        vx, vy = vx_after, vy_after
        if step > 500:
            break

    # Build surrounding geometry
    for circle in circles_data:
        xc, yc = circle["center"]
        R = circle["radius"]
        peg_patch = plt.Circle((xc, yc), R, facecolor='#7f8c8d', edgecolor='#34495e', alpha=0.7, zorder=2)
        ax.add_patch(peg_patch)

    ax.axhline(y=y_target, color='#8e44ad', linestyle='--', linewidth=2, label='Detection Line', zorder=3)
    ax.axvline(x=-WALL_DISTANCE, color='#d35400', linestyle='-', linewidth=2, label='Boundary Wall', zorder=3)
    ax.axvline(x=WALL_DISTANCE, color='#d35400', linestyle='-', linewidth=2, label='Boundary Wall', zorder=3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Randomly Selected Verification Walk (Frame {frame_index:03d})')
    ax.grid(True, linestyle=':', alpha=0.5)

    if circles_data:
        cy = [c["center"][1] for c in circles_data]
        ax.set_xlim(-WALL_DISTANCE - spacing, WALL_DISTANCE + spacing)
        ax.set_ylim(y_target - spacing * 2, max(max(cy) + spacing * 2, y_init + 1))

    handles, labels = ax.get_legend_handles_labels()
    unique_legend_map = dict(zip(labels, handles))
    ax.legend(unique_legend_map.values(), unique_legend_map.keys(), loc='upper right')

    plt.tight_layout()

    # Save sequentially for animation
    save_path = os.path.join("figures", f"random_verification_path_{frame_index:03d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Q-Q PLOT GENERATION FUNCTION ---
def generate_qq_plot(csv_filename, frame_index, param_name, param_val):
    # 1. Load the data
    df = pd.read_csv(csv_filename)

    # 2. Robust Filtering:
    df['final_x'] = pd.to_numeric(df['final_x'], errors='coerce')
    df_clean = df.dropna(subset=['final_x'])

    if len(df_clean) < 30:
        print(f"Error: Only {len(df_clean)} valid points remaining after filtering for Q-Q plot.")
        return

    data = df_clean['final_x']

    # 3. Quantify Normality
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # 4. Create the Q-Q Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    (osm, osr), (slope, intercept, r_value) = stats.probplot(data, dist="norm", plot=ax)

    r_squared = r_value ** 2

    # Visual settings
    ax.get_lines()[0].set_alpha(0.3)
    ax.get_lines()[0].set_markersize(3)

    ax.set_title(f"Q-Q Plot (N = {len(data)}) | {param_name} = {param_val:.4f}\nFrame {frame_index:03d}")
    ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
    ax.set_ylabel("Ordered Values (final_x)")
    ax.grid(True, linestyle='--', alpha=0.7)

    # 5. Add metrics
    textstr = '\n'.join((
        f'Valid Points: {len(data)}',
        f'R² (Linearity): {r_squared:.4f}',
        f'Skewness: {skewness:.3f}',
        f'Kurtosis: {kurtosis:.3f}'
    ))

    props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)

    # Save sequentially for animation
    qq_save_path = os.path.join("figures", f"galton_qq_plot_{frame_index:03d}.png")
    plt.savefig(qq_save_path, dpi=300)
    plt.close()
    print(f"-> Q-Q Plot for frame {frame_index:03d} saved to {qq_save_path}")


# --- GPU BATCH EXECUTION MANAGER ---
def run_and_plot_galton_batch_cuda(n_balls, n_bins, generated_circles, circles_centers, grid_data, grid_counts, min_cx, min_cy,
                                   cell_size, H_INIT, N_ROWS, DISTANCE, H_FINAL, WALL_DISTANCE, RADIUS, COEFF_E,
                                   x_drop_range, frame_index, param_name, param_val, csv_filename):

    os.makedirs("figures", exist_ok=True)

    drop_min, drop_max = x_drop_range
    x_init_values = np.random.uniform(drop_min, drop_max, size=n_balls)

    final_x_positions = []
    final_positions_data = []

    print(f"\n--- Starting simulation for Frame {frame_index:03d} | {param_name} = {param_val:.4f} ---")
    print(f"Executing {n_balls} balls in parallel on the GPU...")

    final_x_arr, status_arr = simulate_batch_cuda(
        circles_centers, grid_data, grid_counts, min_cx, min_cy,
        cell_size, RADIUS, x_init_values, COEFF_E, H_INIT, N_ROWS,
        DISTANCE, H_FINAL, WALL_DISTANCE)

    for ball_index in range(n_balls):
        if status_arr[ball_index] == 1:
            final_x_positions.append(final_x_arr[ball_index])
            final_positions_data.append({
                "ball_id": ball_index,
                "initial_x": x_init_values[ball_index],
                "final_x": final_x_arr[ball_index],
                "status": "success"
            })
        else:
            final_positions_data.append({
                "ball_id": ball_index,
                "initial_x": x_init_values[ball_index],
                "final_x": np.nan,
                "status": "failed_or_out_of_bounds"
            })

    # Write landing positions to CSV specific to this frame
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['ball_id', 'initial_x', 'final_x', 'status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_positions_data:
            writer.writerow(row)

    # Create Distribution Histogram
    counts, bin_edges = np.histogram(final_x_positions, bins=n_bins, range=(-WALL_DISTANCE, WALL_DISTANCE))

    if len(final_x_positions) > 0:
        plt.figure(figsize=(10, 6))
        bin_width = np.diff(bin_edges)
        plt.bar(bin_edges[:-1], counts, width=bin_width, color='#3498db', edgecolor='#2980b9', alpha=0.8, align='edge',
                label=f'Balls Reached Line ({len(final_x_positions)}/{n_balls})')
        plt.xlim(-WALL_DISTANCE, WALL_DISTANCE)
        plt.axvline(x=0, color='#e74c3c', linestyle=':', alpha=0.7, label='Centerline')

        # Add parameter information directly on the plot title
        plt.title(f'Galton Board Final Ball Distribution ({n_bins} Bins) [CUDA]\n{param_name} = {param_val:.4f}', fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate at Detection Line', fontsize=12)
        plt.ylabel('Ball Count', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save sequentially for animation
        hist_save_path = os.path.join("figures", f"galton_distribution_{frame_index:03d}.png")
        plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Histogram for frame {frame_index:03d} saved to {hist_save_path}")

    # High-fidelity physics check for one random ball in this batch (CPU)
    random_verify_idx = random.randint(0, n_balls - 1)
    random_x_init = x_init_values[random_verify_idx]

    simulate_and_visualize_trajectory_optimized(
        circles_data=generated_circles,
        circles_centers=circles_centers,
        grid_data=grid_data,
        grid_counts=grid_counts,
        min_cx=min_cx,
        min_cy=min_cy,
        cell_size=cell_size,
        radius=RADIUS,
        x_init=random_x_init,
        y_init=0.0,
        vx_init=0.0,
        vy_init=0.0,
        e=COEFF_E,
        h_init=H_INIT,
        rows=N_ROWS,
        spacing=DISTANCE,
        h_final=H_FINAL,
        WALL_DISTANCE=WALL_DISTANCE,
        frame_index=frame_index
    )

    # Generate the Q-Q Plot for this batch using the saved CSV
    generate_qq_plot(csv_filename, frame_index, param_name, param_val)

    return counts, bin_edges


# --- CONTROLLER INITIALIZATION & PARAMETER SWEEP ---
if __name__ == '__main__':

    if not cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. Run this inside the CUDA container with "
            "--gpus all (see Dockerfile), or from WSL with the NVIDIA driver set up.")

    cuda.detect()  # prints device name / compute capability

    # 1. Base Parameters (identical to galton_ultra.py)
    params = {
        'N_FIRST_ROW': 51,
        'N_ROWS': 50,
        'DISTANCE': 2.0,
        'H_INIT': 3.0,
        'H_FINAL': 1.0,
        'RADIUS': 0.7,
        'X_DROP_RANGE': (-0.05, 0.05),
        'WALL_PADDING': 0.5,
        'COEFF_E': 0.1,
        'CELL_SIZE': 2.0
    }

    # 2. Setup the Parameter Sweep
    SWEEP_PARAM = 'X_DROP_RANGE'
    PARAM_LOW = 0.05
    PARAM_HIGH = 47.0
    INTERVALS = INTERVALS_DEFAULT

    # Generate the range of values to sweep
    sweep_values = np.linspace(PARAM_LOW, PARAM_HIGH, INTERVALS)

    # 3. Main Loop: Iterate over the parameter values
    for frame_idx, param_val in enumerate(sweep_values):

        # Check if we are sweeping the drop range.
        # If so, convert the float to a symmetric tuple representing the funnel width.
        if SWEEP_PARAM == 'X_DROP_RANGE':
            params[SWEEP_PARAM] = (-param_val, param_val)
        else:
            params[SWEEP_PARAM] = param_val

        # Regenerate geometry
        generated_circles, WALL_DISTANCE = generate_circle_geometry(
            n_first_row=params['N_FIRST_ROW'],
            n_rows=params['N_ROWS'],
            d=params['DISTANCE'],
            h_init=params['H_INIT'],
            r_custom=params['RADIUS'],
            WALL_PADDING=params['WALL_PADDING']
        )

        # Rebuild format for device arrays
        circles_centers = np.array([c["center"] for c in generated_circles], dtype=np.float64)

        # Initialize the dense spatial grid
        grid_data, grid_counts, min_cx, min_cy = build_spatial_grid(
            circles_centers,
            radius=params['RADIUS'],
            cell_size=params['CELL_SIZE']
        )

        # Execute single batch frame on the GPU
        csv_filename = f"figures/galton_ball_positions_{frame_idx:03d}.csv"
        run_and_plot_galton_batch_cuda(
            n_balls=N_BALLS_DEFAULT,
            n_bins=200,
            generated_circles=generated_circles,
            circles_centers=circles_centers,
            grid_data=grid_data,
            grid_counts=grid_counts,
            min_cx=min_cx,
            min_cy=min_cy,
            cell_size=params['CELL_SIZE'],
            H_INIT=params['H_INIT'],
            N_ROWS=params['N_ROWS'],
            DISTANCE=params['DISTANCE'],
            H_FINAL=params['H_FINAL'],
            WALL_DISTANCE=WALL_DISTANCE,
            RADIUS=params['RADIUS'],
            COEFF_E=params['COEFF_E'],
            x_drop_range=params['X_DROP_RANGE'],
            frame_index=frame_idx,
            param_name=SWEEP_PARAM,
            param_val=param_val,
            csv_filename=csv_filename
        )

    print("\nSweep Complete! Images are saved sequentially in the 'figures' directory.")
