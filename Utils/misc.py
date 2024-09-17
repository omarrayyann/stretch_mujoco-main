import numpy as np
import os
import shutil
import mujoco
import time

def create_or_empty_dir(directory):
        if os.path.exists(directory):
            # Remove all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(directory)

def print_debug(x,debug_mode,module_id=None):
    if debug_mode:
        if module_id is not None:
            print(f"[{module_id} MODULE]: {x}")
        else:
            print(x)

def sleep(seconds,sleep=True):
    if sleep:
        time.sleep(seconds)

def make_fake_pcd(x_range=(-1, 1), y_range=(-1, 1), resolution=0.01):
    """
    Creates a fake point cloud with z=0 in the center and z=1 at the borders of the given x and y range.
    Also returns a labels array where 1 indicates z=1 (border points) and 0 otherwise.

    Args:
        x_range (tuple): Range of x values (min_x, max_x).
        y_range (tuple): Range of y values (min_y, max_y).
        resolution (float): Step size for creating grid points.

    Returns:
        points (np.ndarray): Array of point cloud coordinates [N, 3].
        labels (np.ndarray): Array of labels (1 for z=1 points, 0 otherwise) [N,].
        grid_shape (tuple): The shape of the grid (height, width) for neighbor calculations.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Create a grid of x and y points
    x_values = np.arange(x_min, x_max + resolution, resolution)
    y_values = np.arange(y_min, y_max + resolution, resolution)
    
    # Generate a 2D meshgrid from x and y values
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Flatten the grid arrays
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # Initialize z values to 0 and labels to 0
    z_flat = np.zeros_like(x_flat)
    labels = np.zeros_like(x_flat)

    # Get grid dimensions
    grid_height, grid_width = x_grid.shape

    # Set the border points with z=1 (the boundary conditions)
    border_condition = (
        (x_flat == x_min) | (x_flat == x_max) |
        (y_flat == y_min) | (y_flat == y_max)
    )

    z_flat[border_condition] = 1  # Set z to 1 on the borders
    labels[border_condition] = 1  # Set label to 1 for border points

    # Stack x, y, z into a point cloud format (N, 3)
    points = np.vstack((x_flat, y_flat, z_flat)).T

    # Return the grid shape for neighbor indexing
    grid_shape = (grid_height, grid_width)

    print(f"Generated {len(points)} points with labels.")
    
    # Return the points, labels, and grid shape
    return points, labels, points