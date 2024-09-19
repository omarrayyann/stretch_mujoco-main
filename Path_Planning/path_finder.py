import argparse
import pickle
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import open3d as o3d
import io
from PIL import Image

module_id = "PATH_FINDER"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Path_Planning
import Locomotion
import Utils
import Skills
import GPT

def find_nearest_free_point(self,point, grid_points, grid_labels, min_distance=0.35):

    Utils.print_debug("Finding optimal base end position from object",self.args.debug,module_id)

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

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(self, start, end, grid, grid_labels, grid_size):

    Utils.print_debug("Running A* algorithm to find obstacles-free path",self.args.debug,module_id)

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
            if neighbor>=len(grid_labels) or grid_labels[neighbor] in [1, 2]:  # Treat expanded obstacles as actual obstacles
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

def optimize_waypoints(self, waypoints, grid, grid_labels):

    Utils.print_debug("Optimizing (reducing) number of waypoints",self.args.debug,module_id)
    Utils.print_debug(f"Number of waypoints before: {len(waypoints)}",self.args.debug,module_id)

    optimized_path = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        for j in range(len(waypoints) - 1, i, -1):
            if is_line_obstacle_free(waypoints[i], waypoints[j], grid, grid_labels):
                optimized_path.append(waypoints[j])
                i = j
                break

    Utils.print_debug(f"Reduced number of waypoints from {len(waypoints)} to {len(optimized_path)}",self.args.debug,module_id)

    return optimized_path

def find_path(self,new_pcd_points, new_pcd_labels, start_point, end_point, min_distance, return_visualization):
    
    Utils.print_debug(f"Finding obstacles-free path from position: {start_point}",self.args.debug,module_id)

    start_original = np.array(start_point)
    start = find_nearest_free_point(self, np.array(start_point), new_pcd_points, new_pcd_labels, 0.0)[:2]
    end = np.array(end_point)
    end = find_nearest_free_point(self, end, new_pcd_points, new_pcd_labels, min_distance)[:2]
    
    optimized_waypoints = a_star(self, start, end, new_pcd_points, new_pcd_labels, len(np.unique(new_pcd_points[:, 0])))
    if optimized_waypoints:
        optimized_waypoints = optimize_waypoints(self, optimized_waypoints, new_pcd_points, new_pcd_labels)

    # Visualization logic
    min_x, max_x = new_pcd_points[:, 0].min(), new_pcd_points[:, 0].max()
    min_y, max_y = new_pcd_points[:, 1].min(), new_pcd_points[:, 1].max()

    new_pcd_colors = np.zeros((new_pcd_labels.size, 3))
    new_pcd_colors[new_pcd_labels == 1] = [0.8, 0, 0]
    new_pcd_colors[new_pcd_labels == 0] = [0, 0.8, 0]
    new_pcd_colors[new_pcd_labels == -1] = [1, 1, 1]
    new_pcd_colors[new_pcd_labels == 2] = [1, 0.65, 0]  # Orange for expanded obstacles

    grid_labels = new_pcd_labels.reshape((len(np.unique(new_pcd_points[:, 0])), len(np.unique(new_pcd_points[:, 1]))))
    cmap = ListedColormap(['white', 'green', 'red', 'orange'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.Normalize(-1, 2)

    if return_visualization == False:
        return np.array(optimized_waypoints)
    
    plt.imshow(grid_labels, cmap=cmap, norm=norm, origin='lower', extent=[min_x, max_x, min_y, max_y])
    if optimized_waypoints is not None:
        optimized_waypoints = np.array(optimized_waypoints)
        plt.plot(optimized_waypoints[:, 0], optimized_waypoints[:, 1], 'b', label='Optimized Waypoints')
        plt.scatter(optimized_waypoints[:, 0], optimized_waypoints[:, 1], c='blue')
    plt.colorbar(ticks=[-1, 0, 1, 2], format=plt.FuncFormatter(lambda x, _: ['Undetermined', 'Free', 'Obstacle', 'Expanded Obstacle'][int(x)+1]))
    plt.title('Point Cloud Grid Map with Optimized Waypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Convert the plot to a numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    plt.close()

    optimized_waypoints = np.block([start_original,optimized_waypoints])

    return optimized_waypoints, img_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pathfinding on preprocessed data.")
    parser.add_argument('--start_point', type=float, nargs=2, required=True, help="Starting point for pathfinding")
    parser.add_argument('--end_point', type=float, nargs=2, required=True, help="Ending point for pathfinding")
    parser.add_argument('--min_distance', type=float, default=0.0, help="Minimum distance from the endpoint")
    parser.add_argument('--load_processed', type=str, required=True, help="Path to load processed grid and labels")
    parser.add_argument('--visualize', type=bool, default=True, help="Visualize the path found")
    parser.add_argument('--use_matplotlib', action='store_true', help="Flag to visualize using matplotlib")

    args = parser.parse_args()

    with open(args.load_processed, 'rb') as file:
        processed_data = pickle.load(file)
    new_pcd_points = processed_data['points']
    new_pcd_labels = processed_data['labels']

    optimized_waypoints = find_path(new_pcd_points, new_pcd_labels, args.start_point, args.end_point, 0.4,False)

    if args.visualize:
        min_x, max_x = new_pcd_points[:, 0].min(), new_pcd_points[:, 0].max()
        min_y, max_y = new_pcd_points[:, 1].min(), new_pcd_points[:, 1].max()
        
        new_pcd_colors = np.zeros((new_pcd_labels.size, 3))
        new_pcd_colors[new_pcd_labels == 1] = [0.8, 0, 0]
        new_pcd_colors[new_pcd_labels == 0] = [0, 0.8, 0]
        new_pcd_colors[new_pcd_labels == -1] = [1, 1, 1]
        new_pcd_colors[new_pcd_labels == 2] = [1, 0.65, 0]  # Orange for expanded obstacles

        if args.use_matplotlib:
            if optimized_waypoints:
                optimized_waypoints = np.array(optimized_waypoints)
                plt.figure(figsize=(10, 10))
                grid_labels = new_pcd_labels.reshape((len(np.unique(new_pcd_points[:, 0])), len(np.unique(new_pcd_points[:, 1]))))
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
                grid_labels = new_pcd_labels.reshape((len(np.unique(new_pcd_points[:, 0])), len(np.unique(new_pcd_points[:, 1]))))
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
                print("No path found")
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(new_pcd_points)
                new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors)
                o3d.visualization.draw_geometries([new_pcd])
