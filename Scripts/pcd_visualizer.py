import pickle
import numpy as np
import open3d as o3d
import argparse
import json

def visualize_pcd(pcd_path, json_path):
    with open(pcd_path, 'rb') as file:
        data = pickle.load(file)

    with open(json_path, 'r') as file:
        json_data = json.load(file)

    all_pcd_points = []
    all_pcd_colors = []

    for obj in data["objects"]:
        pcd_points = obj["pcd_np"]
        pcd_colors = obj["pcd_color_np"]
        all_pcd_points.append(pcd_points)
        all_pcd_colors.append(pcd_colors)

    all_pcd_points = np.vstack(all_pcd_points)
    all_pcd_colors = np.vstack(all_pcd_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(all_pcd_colors)

    geometries = [pcd]

    for key, value in json_data.items():
        bbox_center = np.array(value["bbox_center"])
        bbox_extent = np.array(value["bbox_extent"])

        # Create axis-aligned bounding box
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_center - bbox_extent / 2, max_bound=bbox_center + bbox_extent / 2)
        
        # Convert axis-aligned bounding box to oriented bounding box
        obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
        obb.color = (1, 0, 0)  # Red color for bounding box
        geometries.append(obb)

        # Create text geometry for labeling
        # text_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # text_mesh.translate(bbox_center + np.array([0, 0, bbox_extent[2] / 2 + 0.1]))
        # geometries.append(text_mesh)

        # Add a label to the text_mesh
        text_label = o3d.geometry.PointCloud()
        text_label.points = o3d.utility.Vector3dVector([bbox_center + np.array([0, 0, bbox_extent[2] / 2 + 0.1])])
        text_label.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        text_label_text = value["object_caption"]
        geometries.append(text_label)

        print(f"Label: {text_label_text}, Position: {bbox_center + np.array([0, 0, bbox_extent[2] / 2 + 0.1])}")

    o3d.visualization.draw_geometries(geometries)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PCD data from a pickle file and label objects based on a JSON file.")
    parser.add_argument('--pcd_path', type=str, help='Path to the pickle file containing the PCD data.')
    parser.add_argument('--objects_path', type=str, help='Path to the JSON file containing object information.')

    args = parser.parse_args()
    visualize_pcd(args.pcd_path, args.objects_path)

