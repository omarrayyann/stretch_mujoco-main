import open3d as o3d
import numpy as np

# Load the PLY file
pcd = o3d.io.read_point_cloud("output/cloud.ply")

# Check the point cloud is loaded
print(pcd)
print(np.asarray(pcd.points))

# Optionally, apply some transformation to see it better (e.g., rescaling)
pcd.scale(0.001, center=pcd.get_center())

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="PLY Point Cloud", width=800, height=600)

# Add the point cloud
vis.add_geometry(pcd)

# Set point size
opt = vis.get_render_option()
opt.point_size = 5  # Increase point size for visibility

# Run the visualizer
vis.run()

# Close the visualizer
vis.destroy_window()
