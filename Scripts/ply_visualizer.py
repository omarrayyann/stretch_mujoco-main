import open3d as o3d
import numpy as np
import glob

# Load the point cloud
pcd = o3d.io.read_point_cloud("tests/cloud.ply")

# Optionally, apply some transformation to scale it for better visualization
pcd.scale(1, center=pcd.get_center())

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud with Grippers", width=800, height=600)

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Set the render options
opt = vis.get_render_option()
opt.point_size = 5  # Increase point size for better visibility

# Load and add all gripper meshes
gripper_files = glob.glob("tests/gripper_*.ply")
for gripper_file in gripper_files:
    gripper_mesh = o3d.io.read_triangle_mesh(gripper_file)
    vis.add_geometry(gripper_mesh)

# Run the visualizer
vis.run()

# Close the visualizer
vis.destroy_window()
