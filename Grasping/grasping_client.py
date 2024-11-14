import socket
import numpy as np
# import open3d as o3d
import time
from Grasping.grasp_detection.anygrasp_caller import *
module_id = "GRASPING"
import Utils

def load_camera_intrinsics():
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=480,
        height=640,
        fx=432.97146127,
        fy=432.97146127,
        cx=240,
        cy=320,
    )
    return intrinsic

def get_3d_points(depth, mask):
    # Get the shape of the depth image
    height, width = depth.shape

    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the pixel coordinates and depth values
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    mask = mask.flatten()

    # Filter out points within the mask
    valid_points = mask == 0
    x = x[valid_points]
    y = y[valid_points]
    depth = depth[valid_points]

    # Get intrinsic parameters
    fx, fy = 432.97146127, 432.97146127
    cx, cy = 240, 320

    # Compute 3D coordinates
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points_3d = np.vstack((x, y, z)).transpose()

    return points_3d

def filter_obastacle_free_trajectories(indices, grasp_poses, point_cloud, traj_length, traj_radius, segmented_object_mask, tolerance=0.02):
    filtered_indices = []
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    
    for i in indices:
        pos = grasp_poses[i][0:3, 3].copy()
        rot = grasp_poses[i][0:3, 0:3].copy()
        x_vector = rot[0:3, 0] * traj_length
        pos -= x_vector
        
        trajectory_valid = True
        for t in np.linspace(0, traj_length, num=15):
            traj_point = pos + t * x_vector / traj_length
            [k, idx, _] = pcd_tree.search_radius_vector_3d(traj_point, traj_radius)
            
            if k > 0:
                for j in idx:
                    pt = point_cloud.points[j]
                    u, v = project_points_to_image([pt], load_camera_intrinsics())[0]
                    if 0 <= u < segmented_object_mask.shape[1] and 0 <= v < segmented_object_mask.shape[0]:
                        if segmented_object_mask[v, u]:
                            continue
                        else:
                            # Check neighbors within the tolerance
                            neighbors_within_tolerance = False
                            for du in range(-int(tolerance * traj_radius), int(tolerance * traj_radius) + 1):
                                for dv in range(-int(tolerance * traj_radius), int(tolerance * traj_radius) + 1):
                                    nu, nv = u + du, v + dv
                                    if 0 <= nu < segmented_object_mask.shape[1] and 0 <= nv < segmented_object_mask.shape[0]:
                                        if segmented_object_mask[nv, nu]:
                                            neighbors_within_tolerance = True
                                            break
                                if neighbors_within_tolerance:
                                    break
                            if neighbors_within_tolerance:
                                continue
                    trajectory_valid = False
                    break

            if not trajectory_valid:
                break

        if trajectory_valid:
            filtered_indices.append(i)

    return filtered_indices


def get_grasps(self, colors, depths, prompt, traj_length=0.2, traj_radius=0.05):
   
    colors = colors.astype(np.float32)
    depths = depths.astype(np.float32)
    print("depths: ",depths)

    grasp_poses, grasp_scores, grasp_widths, langsam_masks = anygrasp_detection(colors, depths, prompt)

    grasp_poses = grasp_poses.reshape(-1, 4, 4)
    langsam_masks = langsam_masks.reshape((-1,640,480))

    Utils.print_debug(f"Grasp Poses: {grasp_poses.shape}",self.args.debug)
    Utils.print_debug(f"Grasp Scores: {grasp_scores.shape}",self.args.debug)
    Utils.print_debug(f"Grasp Widths: {grasp_widths.shape}",self.args.debug)
    Utils.print_debug(f"LangSAM Mask: {langsam_masks.shape}",self.args.debug)

    grasp_centers = grasp_poses[:,0:3,3].copy()
    intrinsic = load_camera_intrinsics()
    image_points_all = project_points_to_image(grasp_centers, intrinsic)
    image_points = []
    for image_point in image_points_all:
        if image_point[0] >= 480 or image_point[1] >= 640:
            continue
        image_points.append(image_point)
    filtered_indicies = []
    

    mask = np.full((640,480), False)
    for langsam_mask in langsam_masks:
        mask = np.logical_or(mask,langsam_mask)

    for i, point in enumerate(image_points):
        if mask[point[1],point[0]]:
            filtered_indicies.append(i)


    free_indicies = filter_collision(depths.copy()/1000.0,mask,grasp_poses,radius=traj_radius,length=traj_length,exclusion_length=0.01)

    Utils.print_debug(f"Received {len(grasp_poses)} grasp poses and {len(grasp_scores)} grasp scores and {len(grasp_widths)} grasp widths and {langsam_masks.shape[0]} masks and {len(filtered_indicies)} langsam filtered grasps and {len(free_indicies)} collision free grasps",self.args.debug,module_id)

    return grasp_poses, grasp_scores, grasp_widths, langsam_masks, filtered_indicies, free_indicies

def get_all_grasps(self, colors, depths):
   
    colors = colors.astype(np.float32)
    depths = depths.astype(np.float32)

    grasp_poses, grasp_scores, grasp_widths, langsam_masks = anygrasp_detection(colors, depths, "all")

    grasp_poses = grasp_poses.reshape(-1, 4, 4)
    langsam_masks = langsam_masks.reshape((-1,640,480))

    Utils.print_debug(f"Grasp Poses: {grasp_poses.shape}",self.args.debug)
    Utils.print_debug(f"Grasp Scores: {grasp_scores.shape}",self.args.debug)
    Utils.print_debug(f"Grasp Widths: {grasp_widths.shape}",self.args.debug)
    Utils.print_debug(f"LangSAM Mask: {langsam_masks.shape}",self.args.debug)

    grasp_centers = grasp_poses[:,0:3,3].copy()
    intrinsic = load_camera_intrinsics()
    image_points_all = project_points_to_image(grasp_centers, intrinsic)
    image_points = []
    for image_point in image_points_all:
        if image_point[0] >= 480 or image_point[1] >= 640:
            continue
        image_points.append(image_point)
    filtered_indicies = []
    

    mask = np.full((640,480), False)
    for langsam_mask in langsam_masks:
        mask = np.logical_or(mask,langsam_mask)

    for i, point in enumerate(image_points):
        if mask[point[1],point[0]]:
            filtered_indicies.append(i)


    free_indicies = filter_collision(depths.copy()/1000.0,mask,grasp_poses,radius=traj_radius,length=traj_length,exclusion_length=0.01)

    Utils.print_debug(f"Received {len(grasp_poses)} grasp poses and {len(grasp_scores)} grasp scores and {len(grasp_widths)} grasp widths and {langsam_masks.shape[0]} masks and {len(filtered_indicies)} langsam filtered grasps and {len(free_indicies)} collision free grasps",self.args.debug,module_id)

    return grasp_poses, grasp_scores, grasp_widths, langsam_masks, filtered_indicies, free_indicies

def filter_collision(depth, mask, grasp_poses, radius=0.03,length=0.2,exclusion_length=0.03):

    points_3d = get_3d_points(depth, mask)
    free_poses_indices = []
    for i, pose in enumerate(grasp_poses):
        position = pose[:3, 3]
        direction = -pose[:3, 1]  
        has_collision, collision_indices = check_collision(position, direction, points_3d, radius=radius, length=length, exclusion_length=exclusion_length)
        if has_collision:
            continue
        free_poses_indices.append(i)
        
    return free_poses_indices

def check_collision(position, direction, points_3d, radius=0.03, length=0.2, exclusion_length=0.03):

    direction = direction / np.linalg.norm(direction)
    segment_end = position + direction * length

    points_vector = points_3d - position
    proj_lengths = np.dot(points_vector, direction)
    proj_points = position + np.outer(proj_lengths, direction)
    distances = np.linalg.norm(points_3d - proj_points,axis=1)

    within_radius = distances < radius
    within_length = (proj_lengths > exclusion_length) & (proj_lengths < length)

    collisions = within_radius & within_length

    return np.sum(collisions) > 3, collisions

def project_points_to_image(points, intrinsic):
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    image_points = []
    for point in points:
        x, y, z = point
        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)
        image_points.append((u, v))
    
    return np.array(image_points)
