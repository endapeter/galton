
# This is an optimized Galton Board simulation using Numba for JIT compilation and multithreading for batch execution.

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import random
import statistics
from numba import njit

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


# --- NUMBA JIT-COMPATIBLE SPATIAL HASHING ---
@njit
def build_spatial_grid_numba(circles_centers, radius, cell_size):
    """
    Creates dense 2D index arrays mappings for O(1) lookups.
    Numba requires contiguous typed memory over Python dictionaries.
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
        c_min_x = int(np.floor((xc - radius) / cell_size))
        c_max_x = int(np.floor((xc + radius) / cell_size))
        c_min_y = int(np.floor((yc - radius) / cell_size))
        c_max_y = int(np.floor((yc + radius) / cell_size))

        for cx in range(c_min_x, c_max_x + 1):
            for cy in range(c_min_y, c_max_y + 1):
                idx_x = cx - min_cx
                idx_y = cy - min_cy
                count = grid_counts[idx_x, idx_y]
                if count < max_pegs:
                    grid_data[idx_x, idx_y, count] = i
                    grid_counts[idx_x, idx_y] += 1

    return grid_data, grid_counts, min_cx, min_cy


# --- CORE NUMBA COMPILED SIMULATION ENGINE ---
@njit(nogil=True)
def simulate_trajectory_numba(circles_centers, grid_data, grid_counts, min_cx, min_cy,
                              cell_size, radius, x_init, y_init, vx_init, vy_init,
                              e, h_init, rows, spacing, h_final, WALL_DISTANCE, g=9.81):
    y_target = -h_init - (rows * spacing * np.cos(30.0 * np.pi / 180.0)) - h_final
    x_imp, y_imp = x_init, y_init
    vx, vy = vx_init, vy_init
    
    c4 = 0.25 * (g**2)
    step = 0
    
    n_circles = len(circles_centers)
    seen_centers = np.zeros(n_circles, dtype=np.bool_)
    
    while True:
        step += 1
        
        # 1. Time-to-line solver
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

        # 2. Time-to-wall solver
        t_wall = np.inf
        hit_wall_side = 0  # 1: left, 2: right
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

        # 3. Dynamic Bounding Box + Dense Spatial Grid Query
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

        # 4. Local analytical scanning loop mapped entirely via dense NumPy arrays
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

        # 5. Intersect evaluation mapping
        t_end = np.inf
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
        y_next = y_imp + vy * t_end - 0.5 * g * t_end**2
        
        if next_event == 0:
            return True, x_next

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
        if step > 1000:
            break

    return False, 0.0


# --- TOP LEVEL MP SERIALIZATION WORKER ---
def _simulation_worker(args):
    return simulate_trajectory_numba(*args)


# --- BATCH EXECUTION MANAGER ---
def run_and_plot_galton_batch(n_balls, n_bins, generated_circles, circles_centers, grid_data, grid_counts, min_cx, min_cy,
                              cell_size, H_INIT, N_ROWS, DISTANCE, H_FINAL, WALL_DISTANCE, RADIUS, COEFF_E, x_drop_range=None, csv_filename="galton_ball_positions.csv"):
    
    os.makedirs("figures", exist_ok=True)
    
    if x_drop_range is None:
        drop_min, drop_max = -DISTANCE, DISTANCE
    else:
        drop_min, drop_max = x_drop_range

    x_init_values = np.random.uniform(drop_min, drop_max, size=n_balls)
    
    # Assemble execution tasks
    tasks = [
        (circles_centers, grid_data, grid_counts, min_cx, min_cy, cell_size, 
         RADIUS, x_val, 0.0, 0.0, 0.0, COEFF_E, H_INIT, N_ROWS, DISTANCE, H_FINAL, WALL_DISTANCE)
        for x_val in x_init_values
    ]
    
    final_x_positions = []
    final_positions_data = [] # List to accumulate records structured for CSV export
    
    print(f"Launching batch execution of {n_balls} balls using ThreadPoolExecutor + Numba JIT...")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_ball = {executor.submit(_simulation_worker, task): i for i, task in enumerate(tasks)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_ball):
            completed_count += 1
            ball_index = future_to_ball[future]
            success, final_x = future.result()
            
            if success:
                final_x_positions.append(final_x)
                final_positions_data.append({
                    "ball_id": ball_index,
                    "initial_x": x_init_values[ball_index],
                    "final_x": final_x,
                    "status": "success"
                })
                print(f"[Walk Update] Ball {completed_count}/{n_balls} finished. Spawn X: {x_init_values[ball_index]:.4f} -> Final X: {final_x:.4f}")
            else:
                final_positions_data.append({
                    "ball_id": ball_index,
                    "initial_x": x_init_values[ball_index],
                    "final_x": np.nan,
                    "status": "failed_or_out_of_bounds"
                })
                print(f"[Walk Update] Ball {completed_count}/{n_balls} finished. Failed or fell out of bounds.")
                
    # --- WRITE LANDING POSITIONS TO CSV ---
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['ball_id', 'initial_x', 'final_x', 'status']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in final_positions_data:
            writer.writerow(row)

# --- CONTROLLER INITIALIZATION ---
if __name__ == '__main__':
    N_FIRST_ROW = 51   
    N_ROWS = 50          
    DISTANCE = 2.0
    H_INIT = 3.0        
    H_FINAL = 1.0       
    RADIUS = 0.7       
    X_DROP_RANGE = (-47, 47)  
    WALL_PADDING = 0.5  
    COEFF_E = 0.1
    CELL_SIZE = DISTANCE 

    generated_circles, WALL_DISTANCE = generate_circle_geometry(
        n_first_row=N_FIRST_ROW, n_rows=N_ROWS, d=DISTANCE, h_init=H_INIT, r_custom=RADIUS, WALL_PADDING=WALL_PADDING
    )
    
    # Rebuild format specifically for Numba JIT arrays
    circles_centers = np.array([c["center"] for c in generated_circles], dtype=np.float64)

    # Initialize Numba-compatible dense spatial grids
    grid_data, grid_counts, min_cx, min_cy = build_spatial_grid_numba(circles_centers, radius=RADIUS, cell_size=CELL_SIZE)

    variance = []
    variance_idx = np.arange(500, 1025, 25)

    for i in range(500, 1025, 25):

        # Execute and verify
        run_and_plot_galton_batch(
            n_balls=i, 
            n_bins=200, 
            generated_circles=generated_circles,
            circles_centers=circles_centers,
            grid_data=grid_data,
            grid_counts=grid_counts,
            min_cx=min_cx,
            min_cy=min_cy,
            cell_size=CELL_SIZE,
            H_INIT=H_INIT,
            N_ROWS=N_ROWS,
            DISTANCE=DISTANCE,
            H_FINAL=H_FINAL,
            WALL_DISTANCE=WALL_DISTANCE,
            RADIUS=RADIUS,
            COEFF_E=COEFF_E,
            x_drop_range=X_DROP_RANGE,
            csv_filename="galton_ball_positions.csv" # CSV Parameter passed here
        )

        final_x = []

        with open("galton_ball_positions.csv", "r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["status"] == "success":
                    final_x.append(float(row["final_x"]))

        print(f"Successful trials: {len(final_x)}")
        print(f"Sample variance: {statistics.variance(final_x):.6f}")
        print(f"Population variance: {statistics.pvariance(final_x):.6f}")
        variance.append(statistics.pvariance(final_x))

    plt.plot(variance_idx, variance, color='#e67e22', linewidth=2)
    plt.title("Variance of Final X Positions vs Number of Balls", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Balls", fontsize=12)
    plt.ylabel("Population Variance of Final X Positions", fontsize=12)
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "variance_vs_balls.png"), dpi=300, bbox_inches='tight')
    plt.close()