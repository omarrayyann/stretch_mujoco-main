import argparse
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

# Set up argument parser
parser = argparse.ArgumentParser(description="Process point cloud data.")
parser.add_argument('--file_path', type=str, required=True, help="Path to the .pkl file")
parser.add_argument('--grid_size', type=int, default=100, help="Number of points in the grid along each dimension")
parser.add_argument('--z_threshold', type=float, default=0.03, help="Threshold for z value to determine obstacle points")
parser.add_argument('--undetermined_to_free', type=bool, default=True, help="Flag to determine if undetermined points surrounded by free points should be considered free")
parser.add_argument('--use_matplotlib', action='store_true', help="Flag to visualize using matplotlib")

args = parser.parse_args()

with open(args.file_path, 'rb') as file:
    data = pickle.load(file)

all_pcd_points = []

for obj in data["objects"]:
    pcd_points = obj["pcd_np"]
    all_pcd_points.append(pcd_points)

all_pcd_points = np.vstack(all_pcd_points)

min_x, max_x = all_pcd_points[:, 0].min(), all_pcd_points[:, 0].max()
min_y, max_y = all_pcd_points[:, 1].min(), all_pcd_points[:, 1].max()

x_range = np.linspace(min_x, max_x, num=args.grid_size)
y_range = np.linspace(min_y, max_y, num=args.grid_size)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]

new_pcd_points = []
new_pcd_labels = []

for point in grid_points:
    x, y = point
    mask = (np.abs(all_pcd_points[:, 0] - x) < (max_x - min_x) / args.grid_size) & \
           (np.abs(all_pcd_points[:, 1] - y) < (max_y - min_y) / args.grid_size)
    if np.any(mask):
        if np.any(all_pcd_points[mask][:, 2] > args.z_threshold):
            new_pcd_labels.append(1)
        else:
            new_pcd_labels.append(0)
    else:
        new_pcd_labels.append(-1)
    new_pcd_points.append([x, y, 0])

new_pcd_points = np.array(new_pcd_points)
new_pcd_labels = np.array(new_pcd_labels)

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

if args.undetermined_to_free:
    for i, label in enumerate(new_pcd_labels):
        if label == -1:
            lump = flood_fill(new_pcd_labels, i, args.grid_size)
            is_free = True
            for idx in lump:
                for neighbor in get_neighbors(idx, args.grid_size):
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

new_pcd_colors = np.zeros((new_pcd_labels.size, 3))
new_pcd_colors[new_pcd_labels == 1] = [0.8, 0, 0]
new_pcd_colors[new_pcd_labels == 0] = [0, 0.8, 0]
new_pcd_colors[new_pcd_labels == -1] = [1, 1, 1]

if args.use_matplotlib:
    grid_labels = new_pcd_labels.reshape((args.grid_size, args.grid_size))
    cmap = ListedColormap(['white', 'green', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.Normalize(-1, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_labels, cmap=cmap, norm=norm, origin='lower', extent=[min_x, max_x, min_y, max_y])
    plt.colorbar(ticks=[-1, 0, 1], format=plt.FuncFormatter(lambda x, _: ['Undetermined', 'Free', 'Obstacle'][int(x)+1]))
    plt.title('Point Cloud Grid Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
else:
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_pcd_points)
    new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
    o3d.visualization.draw_geometries([new_pcd])
