# optimized galton_basic.ipynb with multiprocessing and spatial hashing

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import random

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


# --- SPATIAL HASHING IMPLEMENTATION ---
def build_spatial_grid(circles_data, cell_size):
    """Maps static circles into discrete grid cells to achieve O(1) local lookups."""
    spatial_grid = {}
    for circle in circles_data:
        xc, yc = circle["center"]
        r = circle["radius"]
        
        x_min_cell = int(np.floor((xc - r) / cell_size))
        x_max_cell = int(np.floor((xc + r) / cell_size))
        y_min_cell = int(np.floor((yc - r) / cell_size))
        y_max_cell = int(np.floor((yc + r) / cell_size))
        
        for cx in range(x_min_cell, x_max_cell + 1):
            for cy in range(y_min_cell, y_max_cell + 1):
                if (cx, cy) not in spatial_grid:
                    spatial_grid[(cx, cy)] = []
                spatial_grid[(cx, cy)].append(circle)
    return spatial_grid


# --- CORE OPTIMIZED SIMULATION ENGINE (FOR MULTIPROCESSING BATCH) ---
def simulate_trajectory(spatial_grid, cell_size, radius, x_init, y_init, vx_init, vy_init, 
                        e, h_init, rows, spacing, h_final, WALL_DISTANCE, g=9.81):
    y_target = -h_init - (rows * spacing * np.cos(np.radians(30))) - h_final
    x_imp, y_imp = x_init, y_init
    vx, vy = vx_init, vy_init
    
    c4 = 0.25 * (g**2)
    step = 0
    
    while True:
        step += 1
        
        # 1. Time-to-line solver
        a_line = 0.5 * g
        b_line = -vy
        c_line = y_target - y_imp
        discriminant = b_line**2 - 4 * a_line * c_line
        
        t_line = float('inf')
        if discriminant >= 0:
            t_candidates = [
                (-b_line + np.sqrt(discriminant)) / (2 * a_line),
                (-b_line - np.sqrt(discriminant)) / (2 * a_line)
            ]
            valid_line_times = [t for t in t_candidates if t > 1e-5]
            if valid_line_times:
                t_line = min(valid_line_times)

        # 2. Time-to-wall solver
        t_wall = float('inf')
        hit_wall_side = None
        if vx < 0:
            t_left = (-WALL_DISTANCE - x_imp) / vx
            if t_left > 1e-5:
                t_wall = t_left
                hit_wall_side = 'left'
        elif vx > 0:
            t_right = (WALL_DISTANCE - x_imp) / vx
            if t_right > 1e-5:
                t_wall = t_right
                hit_wall_side = 'right'

        # 3. Dynamic Bounding Box + Spatial Grid Query
        t_max = min(t_line, t_wall)
        if t_max == float('inf'):
            break

        x0, x1 = x_imp, x_imp + vx * t_max
        x_min, x_max = min(x0, x1), max(x0, x1)

        y0 = y_imp
        y1 = y_imp + vy * t_max - 0.5 * g * (t_max**2)
        y_min, y_max = min(y0, y1), max(y0, y1)

        t_vertex = vy / g
        if 0 < t_vertex < t_max:
            y_vertex = y_imp + (vy**2) / (2 * g)
            y_max = max(y_max, y_vertex)

        cx_start = int(np.floor((x_min - radius) / cell_size))
        cx_end = int(np.floor((x_max + radius) / cell_size))
        cy_start = int(np.floor((y_min - radius) / cell_size))
        cy_end = int(np.floor((y_max + radius) / cell_size))

        candidate_circles = []
        seen_centers = set()
        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                if (cx, cy) in spatial_grid:
                    for circle in spatial_grid[(cx, cy)]:
                        center = circle["center"]
                        if center not in seen_centers:
                            seen_centers.add(center)
                            candidate_circles.append(circle)

        # 4. Local analytical scanning loop
        min_t_circle = float('inf')
        next_peg = None
        
        for circle in candidate_circles:
            xc, yc = circle["center"]
            R = radius
            dx = x_imp - xc
            dy = y_imp - yc
            
            c3 = -vy * g
            c2 = (vx**2) + (vy**2) - (dy * g)
            c1 = 2.0 * (dx * vx + dy * vy)
            c0 = (dx**2) + (dy**2) - (R**2)
            
            roots = np.roots([c4, c3, c2, c1, c0])
            for r in roots:
                if abs(r.imag) < 1e-6:
                    t_val = r.real
                    if 1e-5 < t_val < min_t_circle:
                        min_t_circle = t_val
                        next_peg = (xc, yc, R)

        # 5. Intersect evaluation
        event_times = {'line': t_line, 'peg': min_t_circle, 'wall': t_wall}
        valid_events = {k: v for k, v in event_times.items() if v != float('inf')}
        
        if not valid_events:
            break
            
        next_event = min(valid_events, key=valid_events.get)
        t_end = valid_events[next_event]
        is_final_step = (next_event == 'line')

        x_next = x_imp + vx * t_end
        y_next = y_imp + vy * t_end - 0.5 * g * t_end**2
        
        if is_final_step:
            return True, x_next

        vx_before = vx
        vy_before = vy - g * t_end

        if next_event == 'peg':
            xc, yc, R = next_peg
            nx = (x_next - xc) / R
            ny = (y_next - yc) / R
            v_dot_n = vx_before * nx + vy_before * ny
            vx_after = vx_before - (1 + e) * v_dot_n * nx
            vy_after = vy_before - (1 + e) * v_dot_n * ny
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next + (ny * 1e-4)
            
        elif next_event == 'wall':
            vx_after = -e * vx_before
            vy_after = vy_before
            nx = 1.0 if hit_wall_side == 'left' else -1.0
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next 
            
        vx, vy = vx_after, vy_after
        if step > 1000:
            break

    return False, None


# --- VISUALIZATION FUNCTION FOR A SINGLE RANDOM BALL ---
def simulate_and_visualize_trajectory_optimized(circles_data, spatial_grid, cell_size, radius, x_init, y_init, 
                                                vx_init, vy_init, e, h_init, rows, spacing, h_final, WALL_DISTANCE, g=9.81):
    """
    Simulates a single validation path using the spatial grid engine 
    and plots it using your continuous trajectory visualization style.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    y_target = -h_init - (rows * spacing * np.cos(np.radians(30))) - h_final
    
    x_imp, y_imp = x_init, y_init
    vx, vy = vx_init, vy_init
    
    ax.scatter([x_imp], [y_imp], color='#2ecc71', marker='o', s=100, label='Release Point', edgecolor='black', zorder=5)
    
    c4 = 0.25 * (g**2)
    step = 0
    all_x, all_y = [x_imp], [y_imp]
    traj_lbl, collision_lbl = False, False
    
    while True:
        step += 1
        
        # 1. Line Time
        a_line = 0.5 * g
        b_line = -vy
        c_line = y_target - y_imp
        discriminant = b_line**2 - 4 * a_line * c_line
        t_line = float('inf')
        if discriminant >= 0:
            t_candidates = [(-b_line + np.sqrt(discriminant)) / (2 * a_line), (-b_line - np.sqrt(discriminant)) / (2 * a_line)]
            valid_line_times = [t for t in t_candidates if t > 1e-5]
            if valid_line_times:
                t_line = min(valid_line_times)

        # 2. Wall Time
        t_wall = float('inf')
        hit_wall_side = None
        if vx < 0:
            t_left = (-WALL_DISTANCE - x_imp) / vx
            if t_left > 1e-5:
                t_wall = t_left
                hit_wall_side = 'left'
        elif vx > 0:
            t_right = (WALL_DISTANCE - x_imp) / vx
            if t_right > 1e-5:
                t_wall = t_right
                hit_wall_side = 'right'

        # 3. Spatial Hash Culling
        t_max = min(t_line, t_wall)
        if t_max == float('inf'):
            break

        x0, x1 = x_imp, x_imp + vx * t_max
        x_min, x_max = min(x0, x1), max(x0, x1)
        y0 = y_imp
        y1 = y_imp + vy * t_max - 0.5 * g * (t_max**2)
        y_min, y_max = min(y0, y1), max(y0, y1)

        t_vertex = vy / g
        if 0 < t_vertex < t_max:
            y_vertex = y_imp + (vy**2) / (2 * g)
            y_max = max(y_max, y_vertex)

        cx_start = int(np.floor((x_min - radius) / cell_size))
        cx_end = int(np.floor((x_max + radius) / cell_size))
        cy_start = int(np.floor((y_min - radius) / cell_size))
        cy_end = int(np.floor((y_max + radius) / cell_size))

        candidate_circles = []
        seen_centers = set()
        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                if (cx, cy) in spatial_grid:
                    for circle in spatial_grid[(cx, cy)]:
                        center = circle["center"]
                        if center not in seen_centers:
                            seen_centers.add(center)
                            candidate_circles.append(circle)

        # 4. Quartic Scan
        min_t_circle = float('inf')
        next_peg = None
        for circle in candidate_circles:
            xc, yc = circle["center"]
            R = radius
            dx = x_imp - xc
            dy = y_imp - yc
            c3 = -vy * g
            c2 = (vx**2) + (vy**2) - (dy * g)
            c1 = 2.0 * (dx * vx + dy * vy)
            c0 = (dx**2) + (dy**2) - (R**2)
            
            roots = np.roots([c4, c3, c2, c1, c0])
            for r in roots:
                if abs(r.imag) < 1e-6:
                    t_val = r.real
                    if 1e-5 < t_val < min_t_circle:
                        min_t_circle = t_val
                        next_peg = (xc, yc, R)

        # 5. Resolve Event
        event_times = {'line': t_line, 'peg': min_t_circle, 'wall': t_wall}
        valid_events = {k: v for k, v in event_times.items() if v != float('inf')}
        if not valid_events:
            break
            
        next_event = min(valid_events, key=valid_events.get)
        t_end = valid_events[next_event]
        is_final_step = (next_event == 'line')

        # Generate flight path segment for high-fidelity plot mapping
        t_segment = np.linspace(0, t_end, num=50)
        x_segment = x_imp + vx * t_segment
        y_segment = y_imp + vy * t_segment - 0.5 * g * t_segment**2
        
        all_x.extend(x_segment)
        all_y.extend(y_segment)
        
        lbl_traj = 'Trajectory' if not traj_lbl else ""
        traj_lbl = True
        ax.plot(x_segment, y_segment, color='#2980b9', linewidth=2, label=lbl_traj, zorder=4)
        
        if is_final_step:
            ax.scatter([x_segment[-1]], [y_segment[-1]], color='#2c3e50', marker='X', s=120, 
                       label='Detection Intersection', edgecolor='black', zorder=5)
            print(f"\n[Verification Walk Summary] Success! Detection line hit at X = {x_segment[-1]:.6f}")
            break

        x_next, y_next = x_segment[-1], y_segment[-1]
        vx_before = vx
        vy_before = vy - g * t_end

        if next_event == 'peg':
            xc, yc, R = next_peg
            nx = (x_next - xc) / R
            ny = (y_next - yc) / R
            v_dot_n = vx_before * nx + vy_before * ny
            vx_after = vx_before - (1 + e) * v_dot_n * nx
            vy_after = vy_before - (1 + e) * v_dot_n * ny
            
            lbl_coll = 'Collisions' if not collision_lbl else ""
            collision_lbl = True
            ax.scatter([x_next], [y_next], color='#e74c3c', marker='o', s=30, edgecolor='black', label=lbl_coll, zorder=5)
            
            x_imp = x_next + (nx * 1e-4)
            y_imp = y_next + (ny * 1e-4)
            
        elif next_event == 'wall':
            vx_after = -e * vx_before
            vy_after = vy_before
            
            lbl_coll = 'Collisions' if not collision_lbl else ""
            collision_lbl = True
            ax.scatter([x_next], [y_next], color='#f39c12', marker='s', s=30, edgecolor='black', label=lbl_coll, zorder=5)
            
            nx = 1.0 if hit_wall_side == 'left' else -1.0
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
    ax.axvline(x=WALL_DISTANCE, color='#d35400', linestyle='-', linewidth=2, zorder=3)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Randomly Selected Verification Walk (Initial X: {x_init:.4f})')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if circles_data:
        cy = [c["center"][1] for c in circles_data]
        ax.set_xlim(-WALL_DISTANCE - spacing, WALL_DISTANCE + spacing)
        ax.set_ylim(y_target - spacing * 2, max(max(cy) + spacing * 2, y_init + 1))
    
    handles, labels = ax.get_legend_handles_labels()
    unique_legend_map = dict(zip(labels, handles))
    ax.legend(unique_legend_map.values(), unique_legend_map.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()


# --- TOP LEVEL MP SERIALIZATION WORKER ---
def _simulation_worker(args):
    return simulate_trajectory(*args)


# --- BATCH EXECUTION MANAGER ---
def run_and_plot_galton_batch(n_balls, n_bins, generated_circles, spatial_grid, cell_size,
                              H_INIT, N_ROWS, DISTANCE, H_FINAL, WALL_DISTANCE, RADIUS, x_drop_range=None):
    
    if x_drop_range is None:
        drop_min, drop_max = -DISTANCE, DISTANCE
    else:
        drop_min, drop_max = x_drop_range

    x_init_values = np.random.uniform(drop_min, drop_max, size=n_balls)
    
    # 1. Assemble execution tasks
    tasks = [
        (spatial_grid, cell_size, RADIUS, x_val, 0.0, 1.0, 0.0, 0.9, 
         H_INIT, N_ROWS, DISTANCE, H_FINAL, WALL_DISTANCE)
        for x_val in x_init_values
    ]
    
    final_x_positions = []
    
    print(f"Launching batch execution of {n_balls} balls using ProcessPoolExecutor...")
    
    # 2. Process tasks asynchronously using futures to print immediate progress updates
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the process pool
        future_to_ball = {executor.submit(_simulation_worker, task): i for i, task in enumerate(tasks)}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_ball):
            completed_count += 1
            ball_index = future_to_ball[future]
            success, final_x = future.result()
            
            if success:
                final_x_positions.append(final_x)
                # Output real-time terminal progress for every completed walk
                print(f"[Walk Update] Ball {completed_count}/{n_balls} finished. Spawn X: {x_init_values[ball_index]:.4f} -> Final X: {final_x:.4f}")
            else:
                print(f"[Walk Update] Ball {completed_count}/{n_balls} finished. Failed or fell out of bounds.")
                
    # 3. Create Distribution Histogram
    counts, bin_edges = np.histogram(final_x_positions, bins=n_bins, range=(-WALL_DISTANCE, WALL_DISTANCE))
    
    if len(final_x_positions) > 0:
        plt.figure(figsize=(10, 6))
        bin_width = np.diff(bin_edges)
        plt.bar(bin_edges[:-1], counts, width=bin_width, color='#3498db', edgecolor='#2980b9', alpha=0.8, align='edge',
                label=f'Balls Reached Line ({len(final_x_positions)}/{n_balls})')
        plt.xlim(-WALL_DISTANCE, WALL_DISTANCE)
        plt.axvline(x=0, color='#e74c3c', linestyle=':', alpha=0.7, label='Centerline')
        plt.title(f'Galton Board Final Ball Distribution ({n_bins} Bins)', fontsize=14, fontweight='bold')
        plt.xlabel('X Coordinate at Detection Line', fontsize=12)
        plt.ylabel('Ball Count', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
    # 4. Feature: Randomly select ONE configuration from the completed batch to visualize high-fidelity physics
    random_verify_idx = random.randint(0, n_balls - 1)
    random_x_init = x_init_values[random_verify_idx]
    
    print(f"\nSelecting Random Ball Walk for Physics Quality Check...")
    print(f"Visualizing Path for Ball index {random_verify_idx} (Spawn X: {random_x_init:.6f})...")
    
    simulate_and_visualize_trajectory_optimized(
        circles_data=generated_circles,
        spatial_grid=spatial_grid,
        cell_size=cell_size,
        radius=RADIUS,
        x_init=random_x_init,
        y_init=0.0,
        vx_init=0.0,
        vy_init=0.0,
        e=0.8,
        h_init=H_INIT,
        rows=N_ROWS,
        spacing=DISTANCE,
        h_final=H_FINAL,
        WALL_DISTANCE=WALL_DISTANCE
    )
    
    return counts, bin_edges


# --- CONTROLLER INITIALIZATION ---
if __name__ == '__main__':
    N_FIRST_ROW = 50     
    N_ROWS = 50          
    DISTANCE = 2.0      
    H_INIT = 3.0        
    H_FINAL = 1.0       
    RADIUS = 0.7       
    X_DROP_RANGE = (-0.05, 0.05)  # Range for random initial x positions of balls
    WALL_PADDING = 0.5  
    CELL_SIZE = DISTANCE # Hash grid bounds link uniformly to peg structural distribution interval

    generated_circles, WALL_DISTANCE = generate_circle_geometry(
        n_first_row=N_FIRST_ROW, n_rows=N_ROWS, d=DISTANCE, h_init=H_INIT, r_custom=RADIUS, WALL_PADDING=WALL_PADDING
    )

    # Initialize optimized static spatial grid mapping
    spatial_grid = build_spatial_grid(generated_circles, cell_size=CELL_SIZE)

    # Execute and verify
    run_and_plot_galton_batch(
        n_balls=500, 
        n_bins=100, 
        generated_circles=generated_circles,
        spatial_grid=spatial_grid,
        cell_size=CELL_SIZE,
        H_INIT=H_INIT,
        N_ROWS=N_ROWS,
        DISTANCE=DISTANCE,
        H_FINAL=H_FINAL,
        WALL_DISTANCE=WALL_DISTANCE,
        RADIUS=RADIUS,
        x_drop_range=X_DROP_RANGE
    )