import numpy as np
import os
import shutil
import mujoco
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def make_fake_pcd(main_rectangle, border_rectangles, resolution=0.01):
    """
    Creates a fake point cloud with z=0 inside the main rectangle and z=1 inside the border rectangles.
    Also plots the points in 2D (x, y), where points with z=1 (border) are in a different color than points with z=0.

    Args:
        main_rectangle (tuple): Main rectangle ((x_min, x_max), (y_min, y_max)) for points with z=0.
        border_rectangles (list of tuples): List of border rectangles, where each tuple contains ((x_min, x_max), (y_min, y_max)).
        resolution (float): Step size for creating grid points.

    Returns:
        points (np.ndarray): Array of point cloud coordinates [N, 3].
        labels (np.ndarray): Array of labels (1 for z=1 points, 0 for z=0 points) [N,].
    """
    # Unpack the main rectangle
    x_min_main, x_max_main = main_rectangle[0]
    y_min_main, y_max_main = main_rectangle[1]

    # Create a grid of x and y points for the main rectangle
    x_values_main = np.arange(x_min_main, x_max_main + resolution, resolution)
    y_values_main = np.arange(y_min_main, y_max_main + resolution, resolution)

    # Generate a 2D meshgrid from x and y values for the main rectangle
    x_grid_main, y_grid_main = np.meshgrid(x_values_main, y_values_main)

    # Flatten the grid arrays
    x_flat_main = x_grid_main.flatten()
    y_flat_main = y_grid_main.flatten()

    # Initialize z values to 0 (interior) for the main rectangle
    z_flat_main = np.zeros_like(x_flat_main)

    # Store the points and labels for the main rectangle
    points_main = np.vstack((x_flat_main, y_flat_main, z_flat_main)).T
    labels_main = np.zeros_like(x_flat_main)

    all_points = [points_main]
    all_labels = [labels_main]

    # Now handle the border rectangles
    for rect in border_rectangles:
        x_min, x_max = rect[0]
        y_min, y_max = rect[1]

        # Create a grid of x and y points for each border rectangle
        x_values = np.arange(x_min, x_max + resolution, resolution)
        y_values = np.arange(y_min, y_max + resolution, resolution)

        x_grid, y_grid = np.meshgrid(x_values, y_values)

        # Flatten the grid arrays
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        # Set z values to 1 (border) for the border rectangles
        z_flat = np.ones_like(x_flat)

        # Stack x, y, z into a point cloud format (N, 3)
        points_border = np.vstack((x_flat, y_flat, z_flat)).T
        labels_border = np.ones_like(x_flat)

        # Append the border points and labels
        all_points.append(points_border)
        all_labels.append(labels_border)

    # Concatenate all points and labels
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)

    # Plot the results
    plt.figure(figsize=(8, 8))
    
    # Points with z=0 (interior)
    mask_z0 = all_labels == 0
    plt.scatter(all_points[mask_z0, 0], all_points[mask_z0, 1], c='blue', label='z=0 (interior)', s=5)

    # Points with z=1 (border)
    mask_z1 = all_labels == 1
    plt.scatter(all_points[mask_z1, 0], all_points[mask_z1, 1], c='red', label='z=1 (border)', s=5)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud with Borders')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    print(f"Generated {len(all_points)} points with labels.")

    return all_points, all_labels

# def make_fake_pcd(x_range=(-1, 1), y_range=(-1, 1), resolution=0.01):
#     """
#     Creates a fake point cloud with z=0 in the center and z=1 at the borders of the given x and y range.
#     Also returns a labels array where 1 indicates z=1 (border points) and 0 otherwise.

#     Args:
#         x_range (tuple): Range of x values (min_x, max_x).
#         y_range (tuple): Range of y values (min_y, max_y).
#         resolution (float): Step size for creating grid points.

#     Returns:
#         points (np.ndarray): Array of point cloud coordinates [N, 3].
#         labels (np.ndarray): Array of labels (1 for z=1 points, 0 otherwise) [N,].
#         grid_shape (tuple): The shape of the grid (height, width) for neighbor calculations.
#     """
#     x_min, x_max = x_range
#     y_min, y_max = y_range

#     # Create a grid of x and y points
#     x_values = np.arange(x_min, x_max + resolution, resolution)
#     y_values = np.arange(y_min, y_max + resolution, resolution)
    
#     # Generate a 2D meshgrid from x and y values
#     x_grid, y_grid = np.meshgrid(x_values, y_values)

#     # Flatten the grid arrays
#     x_flat = x_grid.flatten()
#     y_flat = y_grid.flatten()

#     # Initialize z values to 0 and labels to 0
#     z_flat = np.zeros_like(x_flat)
#     labels = np.zeros_like(x_flat)

#     # Get grid dimensions
#     grid_height, grid_width = x_grid.shape

#     # Set the border points with z=1 (the boundary conditions)
#     border_condition = (
#         (x_flat == x_min) | (x_flat == x_max) |
#         (y_flat == y_min) | (y_flat == y_max)
#     )

#     z_flat[border_condition] = 1  # Set z to 1 on the borders
#     labels[border_condition] = 1  # Set label to 1 for border points

#     # Stack x, y, z into a point cloud format (N, 3)
#     points = np.vstack((x_flat, y_flat, z_flat)).T

#     # Return the grid shape for neighbor indexing
#     grid_shape = (grid_height, grid_width)

#     print(f"Generated {len(points)} points with labels.")
    
#     # Return the points, labels, and grid shape
#     return points, labels, points

def plot_detections(rgb_image, detections):
    """
    Plots the RGB image with bounding boxes and labels.

    Parameters:
        rgb_image (numpy.ndarray): The RGB image array.
        detections (list): A list of detection dictionaries with keys 'label' and 'box'.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(rgb_image)

    # Iterate over detections and add boxes and labels
    for det in detections:
        label = det['label']
        score = det['score']
        box = det['box']  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        

        # Add label and score
        plt.text(xmin, ymin - 10, f"{label}: {score:.2f}", color='yellow', fontsize=12, backgroundcolor='black')

    # Show the plot
    plt.show()