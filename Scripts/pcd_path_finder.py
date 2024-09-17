import argparse
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import heapq

# Set up argument parser
parser = argparse.ArgumentParser(description="Process point cloud data.")
parser.add_argument('--file_path', type=str, required=True, help="Path to the .pkl file")
parser.add_argument('--grid_size', type=int, default=200, help="Number of points in the grid along each dimension")
parser.add_argument('--z_threshold', type=float, default=0.03, help="Threshold for z value to determine obstacle points")
parser.add_argument('--undetermined_to_free', type=bool, default=True, help="Flag to determine if undetermined points surrounded by free points should be considered free")
parser.add_argument('--use_matplotlib', action='store_true', help="Flag to visualize using matplotlib")
parser.add_argument('--start_point', type=float, nargs=2, required=True, help="Starting point for pathfinding")
parser.add_argument('--end_point', type=float, nargs=2, required=True, help="Ending point for pathfinding")
parser.add_argument('--robot_radius', type=float, default=0.4, help="Radius of the robot")
parser.add_argument('--save_processed', type=str, help="Path to save processed grid and labels")
parser.add_argument('--load_processed', type=str, help="Path to load processed grid and labels")
parser.add_argument('--visualize', type=bool, default=True, help="Visualize the path found")
parser.add_argument('--min_distance', type=float, default=0.0, help="Minimum Distance from end point")

args = parser.parse_args()

def find_nearest_free_point(point, grid_points, grid_labels, min_distance=0.4):
    
    point = np.array(point)
    free_points = grid_points[grid_labels == 0]
    
    if free_points.size == 0:
        raise ValueError("No free points available in the grid.")
    
    distances = np.linalg.norm(free_points[:, :2] - point[:2], axis=1)
    
    valid_points = free_points[distances >= min_distance]
    if valid_points.size == 0:
        nearest_free_point = free_points[np.argmin(distances)]
    else:
        nearest_free_point = valid_points[np.argmin(distances[distances >= min_distance])]
    
    return nearest_free_point

def expand_obstacles_with_radius(points, labels, radius, grid_size):
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

if args.load_processed:
    with open(args.load_processed, 'rb') as file:
        processed_data = pickle.load(file)
    new_pcd_points = processed_data['points']
    new_pcd_labels = processed_data['labels']
    min_x, max_x = new_pcd_points[:, 0].min(), new_pcd_points[:, 0].max()
    min_y, max_y = new_pcd_points[:, 1].min(), new_pcd_points[:, 1].max()
else:
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

    original_obstacles = new_pcd_labels.copy()

    new_pcd_labels = expand_obstacles_with_radius(new_pcd_points, new_pcd_labels, args.robot_radius, args.grid_size)

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

    if args.save_processed:
        with open(args.save_processed, 'wb') as file:
            pickle.dump({'points': new_pcd_points, 'labels': new_pcd_labels}, file)

start = np.array(args.start_point)
end = np.array(args.end_point)
if args.min_distance > 0.0:
    end = find_nearest_free_point(end,new_pcd_points,new_pcd_labels,args.min_distance)[:2]

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, end, grid, grid_labels, grid_size):
    start_idx = np.argmin(np.linalg.norm(grid[:, :2] - start, axis=1))
    end_idx = np.argmin(np.linalg.norm(grid[:, :2] - end, axis=1))
    
    open_set = []
    heapq.heappush(open_set, (0, start_idx, None))  # Add None to keep track of the previous node
    
    came_from = {}
    g_score = {i: float('inf') for i in range(len(grid))}
    g_score[start_idx] = 0
    
    f_score = {i: float('inf') for i in range(len(grid))}
    f_score[start_idx] = heuristic(start, end)
    
    while open_set:
        _, current, prev = heapq.heappop(open_set)
        
        if current == end_idx:
            path = []
            while current in came_from:
                path.append(grid[current][:2])
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor in get_neighbors(current, grid_size):
            if grid_labels[neighbor] in [1, 2]:  # Treat expanded obstacles as actual obstacles
                continue
            
            tentative_g_score = g_score[current] + heuristic(grid[current][:2], grid[neighbor][:2])
            
            # Add a penalty for turns
            if prev is not None and (grid[neighbor][:2] - grid[current][:2]).dot(grid[current][:2] - grid[prev][:2]) == 0:
                tentative_g_score += 0.1  # Adjust the penalty value as needed
            
            # Add a cost for proximity to obstacles
            if grid_labels[neighbor] == 1:
                tentative_g_score += 1.0  # Adjust the proximity cost as needed
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(grid[neighbor][:2], end)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor, current))
    
    return None

def is_line_obstacle_free(start, end, grid, grid_labels):
    x0, y0 = start
    x1, y1 = end
    num_points = int(max(abs(x1 - x0), abs(y1 - y0)) * 1000)
    x_values = np.linspace(x0, x1, num_points)
    y_values = np.linspace(y0, y1, num_points)
    for x, y in zip(x_values, y_values):
        idx = np.argmin(np.linalg.norm(grid[:, :2] - np.array([x, y]), axis=1))
        if grid_labels[idx] in [1, 2]:
            return False
    return True

def optimize_waypoints(waypoints, grid, grid_labels):
    optimized_path = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        for j in range(len(waypoints) - 1, i, -1):
            if is_line_obstacle_free(waypoints[i], waypoints[j], grid, grid_labels):
                optimized_path.append(waypoints[j])
                i = j
                break
    return optimized_path

optimized_waypoints = a_star(start, end, new_pcd_points, new_pcd_labels, args.grid_size)
if optimized_waypoints:
    optimized_waypoints = optimize_waypoints(optimized_waypoints, new_pcd_points, new_pcd_labels)

if args.visualize:
    new_pcd_colors = np.zeros((new_pcd_labels.size, 3))
    new_pcd_colors[new_pcd_labels == 1] = [0.8, 0, 0]
    new_pcd_colors[new_pcd_labels == 0] = [0, 0.8, 0]
    new_pcd_colors[new_pcd_labels == -1] = [1, 1, 1]
    new_pcd_colors[new_pcd_labels == 2] = [1, 0.65, 0]  # Orange for expanded obstacles

    if args.use_matplotlib:
        if optimized_waypoints:
            optimized_waypoints = np.array(optimized_waypoints)
            plt.figure(figsize=(10, 10))
            grid_labels = new_pcd_labels.reshape((args.grid_size, args.grid_size))
            cmap = ListedColormap(['white', 'green', 'red', 'orange'])
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
            norm = plt.Normalize(-1, 2)
            plt.imshow(grid_labels, cmap=cmap, norm=norm, origin='lower', extent=[min_x, max_x, min_y, max_y])
            plt.plot(optimized_waypoints[:, 0], optimized_waypoints[:, 1], 'b', label='Optimized Waypoints')
            plt.scatter(optimized_waypoints[:, 0], optimized_waypoints[:, 1], c='blue')
            plt.colorbar(ticks=[-1, 0, 1, 2], format=plt.FuncFormatter(lambda x, _: ['Undetermined', 'Free', 'Obstacle', 'Expanded Obstacle'][int(x)+1]))
            plt.title('Point Cloud Grid Map with Optimized Waypoints')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()
        else:
            plt.figure(figsize=(10, 10))
            grid_labels = new_pcd_labels.reshape((args.grid_size, args.grid_size))
            cmap = ListedColormap(['white', 'green', 'red', 'orange'])
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
            norm = plt.Normalize(-1, 2)
            plt.imshow(grid_labels, cmap=cmap, norm=norm, origin='lower', extent=[min_x, max_x, min_y, max_y])
            plt.colorbar(ticks=[-1, 0, 1, 2], format=plt.FuncFormatter(lambda x, _: ['Undetermined', 'Free', 'Obstacle', 'Expanded Obstacle'][int(x)+1]))
            plt.title('Point Cloud Grid Map - No Path Found')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
    else:
        if optimized_waypoints:
            optimized_waypoints = np.array(optimized_waypoints)
            waypoints_3d = np.hstack([optimized_waypoints, np.zeros((optimized_waypoints.shape[0], 1))])
            waypoints_pcd = o3d.geometry.PointCloud()
            waypoints_pcd.points = o3d.utility.Vector3dVector(waypoints_3d)
            waypoints_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(optimized_waypoints), 1)))
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(new_pcd_points)
            new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
            o3d.visualization.draw_geometries([new_pcd, waypoints_pcd])
        else:
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(new_pcd_points)
            new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
            o3d.visualization.draw_geometries([new_pcd])