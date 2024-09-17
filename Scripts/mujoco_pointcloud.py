import numpy as np
import imageio
import os
import open3d as o3d

def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

def load_and_transform(depth_file, color_file, pose_file):
    # Load the data
    depth = np.load(depth_file)
    color = np.asarray(imageio.imread(color_file))
    pose = np.load(pose_file)
    
    # Camera intrinsic parameters
    fx, fy = 318.49075719, 318.49075719
    cx, cy = 320.0, 240.0  # Optical center
    scaling_factor = 1.0  # Depth scaling

    # b2w_r = quat2Mat([0, 1, 0, 0])
    # pose[0:3,0:3] = np.matmul(pose[0:3,0:3], b2w_r)
    
    # Create point cloud from depth
    points = []
    colors = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u] / scaling_factor
            if z == 0: continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point = [x, y, z, 1]  # Homogeneous coordinates for transformation
            point = np.dot(pose, point)[:3]  # Transforming the point
            points.append(point)
            colors.append(color[v, u] / 255.0)  # Normalize colors

    return np.array(points), np.array(colors)

def process_files(base_dir='recordings', num_files=30):
    combined_points = []
    combined_colors = []
    i = 0
    while i<120:
        i += 2
        pose_file = f'{base_dir}/poses/{i:06d}.npy'
        depth_file = f'{base_dir}/depth/{i:06d}.npy'
        color_file = f'{base_dir}/rgb/{i:06d}.jpg'
        
        if os.path.exists(pose_file) and os.path.exists(depth_file) and os.path.exists(color_file):
            points, colors = load_and_transform(depth_file, color_file, pose_file)
            combined_points.append(points)
            combined_colors.append(colors)
        else:
            break  # Stop if files are missing

    # Combine all points and colors
    combined_points = np.vstack(combined_points)
    combined_colors = np.vstack(combined_colors)
    return combined_points, combined_colors

# Load and transform the point clouds
points, colors = process_files()

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
