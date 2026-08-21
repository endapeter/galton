# CPU-side rendering with matplotlib: verification walk and Q-Q plot.
# Rendering stays on the CPU (headless "Agg" backend) - only the physics runs
# on the GPU.

import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # headless rendering inside the container
import matplotlib.pyplot as plt


# --- VISUALIZATION FUNCTION FOR A SINGLE RANDOM BALL (CPU, pure Python) ---
def simulate_and_visualize_trajectory_optimized(circles_data, circles_centers, grid_data, grid_counts, min_cx, min_cy,
                                                cell_size, radius, x_init, y_init, vx_init, vy_init, e, h_init, rows, spacing, h_final, WALL_DISTANCE, frame_index, g=9.81,
                                                output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
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
    save_path = os.path.join(output_dir, f"random_verification_path_{frame_index:03d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Q-Q PLOT GENERATION FUNCTION ---
def generate_qq_plot(csv_filename, frame_index, param_name, param_val, output_dir="figures"):
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
    os.makedirs(output_dir, exist_ok=True)

    # Save sequentially for animation
    qq_save_path = os.path.join(output_dir, f"galton_qq_plot_{frame_index:03d}.png")
    plt.savefig(qq_save_path, dpi=300)
    plt.close()
    print(f"-> Q-Q Plot for frame {frame_index:03d} saved to {qq_save_path}")
