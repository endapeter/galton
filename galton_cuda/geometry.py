# Geometry generation and host-side spatial hashing for the CUDA Galton board.

import numpy as np


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
