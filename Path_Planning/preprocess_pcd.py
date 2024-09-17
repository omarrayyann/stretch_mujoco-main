import argparse
import pickle
import numpy as np
from collections import deque

def expand_obstacles_with_radius(points, labels, radius, grid_size, min_x, max_x, min_y, max_y, x_range, y_range):
    expanded_labels = labels.copy()
    for i, label in enumerate(labels):
        if label == 1:
            x, y, _ = points[i]
            for dx in np.linspace(-radius, radius, num=int(radius*2*grid_size/(max_x - min_x))):
                for dy in np.linspace(-radius, radius, num=int(radius*2*grid_size/(max_y - min_y))):
                    if dx**2 + dy**2 <= radius**2:
                        new_x, new_y = x + dx, y + dy
                        if min_x <= new_x <= max_x and min_y <= new_y <= max_y:
                            grid_x = np.searchsorted(x_range, new_x) - 1
                            grid_y = np.searchsorted(y_range, new_y) - 1
                            if expanded_labels[grid_y * grid_size + grid_x] != 1:
                                expanded_labels[grid_y * grid_size + grid_x] = 2
    return expanded_labels

def get_neighbors(index, grid_size):
    neighbors = []
    if index % grid_size != 0:
        neighbors.append(index - 1)
    if index % grid_size != grid_size - 1:
        neighbors.append(index + 1)
    if index >= grid_size:
        neighbors.append(index - grid_size)
    if index < grid_size * (grid_size - 1):
        neighbors.append(index + grid_size)
    return neighbors

def flood_fill(pcd_labels, start_index, grid_size):
    queue = deque([start_index])
    lump = []
    while queue:
        idx = queue.popleft()
        if pcd_labels[idx] != -1:
            continue
        pcd_labels[idx] = -2  # Temporary mark to avoid reprocessing
        lump.append(idx)
        for neighbor in get_neighbors(idx, grid_size):
            if pcd_labels[neighbor] == -1:
                queue.append(neighbor)
    return lump

def preprocess(file_path, grid_size, z_threshold, robot_radius, save_processed=None):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    all_pcd_points = []

    for obj in data["objects"]:
        pcd_points = obj["pcd_np"]
        all_pcd_points.append(pcd_points)

    all_pcd_points = np.vstack(all_pcd_points)

    min_x, max_x = all_pcd_points[:, 0].min(), all_pcd_points[:, 0].max()
    min_y, max_y = all_pcd_points[:, 1].min(), all_pcd_points[:, 1].max()

    x_range = np.linspace(min_x, max_x, num=grid_size)
    y_range = np.linspace(min_y, max_y, num=grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    new_pcd_points = []
    new_pcd_labels = []

    for point in grid_points:
        x, y = point
        mask = (np.abs(all_pcd_points[:, 0] - x) < (max_x - min_x) / grid_size) & \
               (np.abs(all_pcd_points[:, 1] - y) < (max_y - min_y) / grid_size)
        if np.any(mask):
            if np.any(all_pcd_points[mask][:, 2] > z_threshold):
                new_pcd_labels.append(1)
            else:
                new_pcd_labels.append(0)
        else:
            new_pcd_labels.append(-1)
        new_pcd_points.append([x, y, 0])

    new_pcd_points = np.array(new_pcd_points)
    new_pcd_labels = np.array(new_pcd_labels)

    new_pcd_labels = expand_obstacles_with_radius(new_pcd_points, new_pcd_labels, robot_radius, grid_size, min_x, max_x, min_y, max_y, x_range, y_range)

    for i, label in enumerate(new_pcd_labels):
        if label == -1:
            lump = flood_fill(new_pcd_labels, i, grid_size)
            is_free = True
            for idx in lump:
                for neighbor in get_neighbors(idx, grid_size):
                    if new_pcd_labels[neighbor] == 1:
                        is_free = False
                        break
                if not is_free:
                    break
            if is_free:
                for idx in lump:
                    new_pcd_labels[idx] = 0
            else:
                for idx in lump:
                    new_pcd_labels[idx] = 1  # Mark as obstacle if the lump is not free

    new_pcd_labels[new_pcd_labels == -2] = -1  # Restore the undetermined label for those not processed

    if save_processed:
        with open(save_processed, 'wb') as file:
            pickle.dump({'points': new_pcd_points, 'labels': new_pcd_labels, 'all_pcd_points': all_pcd_points}, file)
    
    return new_pcd_points, new_pcd_labels, all_pcd_points

def load_processed(file_path):
    with open(file_path, 'rb') as file:
        processed_data = pickle.load(file)
        new_pcd_points = processed_data['points']
        new_pcd_labels = processed_data['labels']
        all_pcd_points = processed_data['all_pcd_points']
        return new_pcd_points, new_pcd_labels, all_pcd_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess point cloud data.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the .pkl file")
    parser.add_argument('--grid_size', type=int, default=200, help="Number of points in the grid along each dimension")
    parser.add_argument('--z_threshold', type=float, default=0.03, help="Threshold for z value to determine obstacle points")
    parser.add_argument('--robot_radius', type=float, default=0.25, help="Radius of the robot")
    parser.add_argument('--save_processed', type=str, help="Path to save processed grid and labels")

    args = parser.parse_args()
    
    preprocess(args.file_path, args.grid_size, args.z_threshold, args.robot_radius, args.save_processed)
