import numpy as np
import config

def depth_to_grid(depth_array, grid_cols=10):
    """
    depth_array: numpy (480, 640) uint16 from Orbbec
    Returns list of 'FREE' or 'BLOCKED' strings, length = grid_cols
    """
    # Take horizontal band at floor level
    band = depth_array[200:280, :]
    col_w = band.shape[1] // grid_cols
    grid = []
    for c in range(grid_cols):
        col = band[:, c * col_w:(c + 1) * col_w]
        # Count pixels closer than STOP threshold
        obstacles = np.sum((col > 0) & (col < config.STOP_MM))
        grid.append('BLOCKED' if obstacles > 50 else 'FREE')
    return grid
