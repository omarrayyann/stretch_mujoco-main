import numpy as np
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import rerun as rr
from pyntcloud import PyntCloud
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm  # For displaying a progress bar
from multiprocessing import Process, Queue
import cv2
from pycpd import RigidRegistration

module_id = "FRAMES_COMPARE"
import Utils

# Constants
image_width = 640
image_height = 480



def find_delta(self, reference_frame, first_frame, second_frame):

    predictor = self.predictor

    path = "video_frames"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    unblocked_image_pil = Image.fromarray(reference_frame.rgb)
    unblocked_image_pil.save('video_frames/00000.jpg')
    base_image_pil = Image.fromarray(first_frame.rgb)
    base_image_pil.save('video_frames/00001.jpg')
    new_image_pil = Image.fromarray(second_frame.rgb)
    new_image_pil.save('video_frames/00002.jpg')
    frame_names = [
        p for p in os.listdir(path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=path)

    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[image_width/2, image_height/2]], dtype=np.float32)
    labels = np.array([1], np.int32)

    Utils.print_debug("Predicting Segment of Reference Frame",self.args.debug, module_id)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    Utils.print_debug("Propagating Segment into the Other frames",self.args.debug, module_id)
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    vis_frame_stride = 1
    
    masks = []
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            masks.append(get_mask(out_mask, out_obj_id))

    masks = np.array(masks)

    reference_frame.mask = masks[0][:,:,0]
    first_frame.mask = masks[1][:,:,0]
    second_frame.mask = masks[2][:,:,0]

    intrinsic = {'fx': reference_frame.fx, 'fy': reference_frame.fy , 'cx': reference_frame.cx, 'cy': reference_frame.cy}
    
    print("Computing point clouds")
    first_pcd = depth_image_to_point_cloud(first_frame.depth*first_frame.mask, intrinsic)
    second_pcd = depth_image_to_point_cloud(second_frame.depth*second_frame.mask, intrinsic)

    cloud1 = pd.DataFrame(first_pcd, columns=["x", "y", "z"])
    cloud1 = cloud1[cloud1["z"] != 0]
    cloud1 = PyntCloud(cloud1)

    cloud2 = pd.DataFrame(second_pcd, columns=["x", "y", "z"])
    cloud2 = cloud2[cloud2["z"] != 0]
    cloud2 = PyntCloud(cloud2)


    first_transform = np.eye(4)
    first_transform[:3, :3] = first_frame.pose[0:3,0:3]
    first_transform[:3, 3] = first_frame.pose[0:3,3]

    second_transform = np.eye(4)
    second_transform[:3, :3] = second_frame.pose[0:3,0:3]
    second_transform[:3, 3] = second_frame.pose[0:3,3]

    cloud1.points = pd.DataFrame(np.dot(cloud1.points.values, first_transform[:3, :3].T) + first_transform[:3, 3], columns=["x", "y", "z"])
    cloud2.points = pd.DataFrame(np.dot(cloud2.points.values, second_transform[:3, :3].T) + second_transform[:3, 3], columns=["x", "y", "z"])

    threshold = 0.01
    distances = compute_distances(cloud1.points.values, cloud2.points.values, threshold)
    far_points_indices = np.where(distances > threshold)[0]

    far_points_pcd = cloud2.points.iloc[far_points_indices].values

    reward = len(far_points_indices)

    # print("Reward: ", reward)
    
     # Visualize far points using the custom plot 

    # custom_plot(far_points_pcd)

    # # Filter by occlusion
    filtered_points_array, _ = filter_by_occlusion(far_points_pcd, first_frame.depth, intrinsic, first_frame.pose)

    mask_change = np.zeros(second_frame.depth.shape)
   
    depth_normalized = cv2.normalize(second_frame.depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
    
    if len(filtered_points_array)==0:
        return filtered_points_array, depth_colored

    for point in filtered_points_array:
        u, v, z = project_to_image(point, intrinsic, second_frame.pose)

        if 0 <= u < mask_change.shape[1] and 0 <= v < mask_change.shape[0]:
            mask_change[v, u] = 1
    
    mask_scaled = (mask_change * 255).astype(np.uint8)
    occlusion_colored = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_JET)
    alpha = 0.5
    combined_image = cv2.addWeighted(occlusion_colored, alpha, depth_colored, 1 - alpha, 0)

    return reward, combined_image

def find_delta_move(self, reference_frame, first_frame, second_frame):

    predictor = self.predictor

    path = "video_frames"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    unblocked_image_pil = Image.fromarray(reference_frame.rgb)
    unblocked_image_pil.save('video_frames/00000.jpg')
    base_image_pil = Image.fromarray(first_frame.rgb)
    base_image_pil.save('video_frames/00001.jpg')
    new_image_pil = Image.fromarray(second_frame.rgb)
    new_image_pil.save('video_frames/00002.jpg')
    frame_names = [
        p for p in os.listdir(path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=path)

    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[image_width/2, image_height/2]], dtype=np.float32)
    labels = np.array([1], np.int32)

    Utils.print_debug("Predicting Segment of Reference Frame",self.args.debug, module_id)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    Utils.print_debug("Propagating Segment into the Other frames",self.args.debug, module_id)
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    vis_frame_stride = 1
    
    masks = []
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            masks.append(get_mask(out_mask, out_obj_id))

    masks = np.array(masks)

    reference_frame.mask = masks[0][:,:,0]
    first_frame.mask = masks[1][:,:,0]
    second_frame.mask = masks[2][:,:,0]

    intrinsic = {'fx': reference_frame.fx, 'fy': reference_frame.fy , 'cx': reference_frame.cx, 'cy': reference_frame.cy}
    
    print("Computing point clouds")
    first_pcd = depth_image_to_point_cloud(first_frame.depth*first_frame.mask, intrinsic)
    second_pcd = depth_image_to_point_cloud(second_frame.depth*second_frame.mask, intrinsic)

    cloud1 = pd.DataFrame(first_pcd, columns=["x", "y", "z"])
    cloud1 = cloud1[cloud1["z"] != 0]
    cloud1 = PyntCloud(cloud1)

    cloud2 = pd.DataFrame(second_pcd, columns=["x", "y", "z"])
    cloud2 = cloud2[cloud2["z"] != 0]
    cloud2 = PyntCloud(cloud2)

    first_transform = np.eye(4)
    first_transform[:3, :3] = first_frame.pose[0:3,0:3]
    first_transform[:3, 3] = first_frame.pose[0:3,3]

    second_transform = np.eye(4)
    second_transform[:3, :3] = second_frame.pose[0:3,0:3]
    second_transform[:3, 3] = second_frame.pose[0:3,3]

    cloud1.points = pd.DataFrame(np.dot(cloud1.points.values, first_transform[:3, :3].T) + first_transform[:3, 3], columns=["x", "y", "z"])
    cloud2.points = pd.DataFrame(np.dot(cloud2.points.values, second_transform[:3, :3].T) + second_transform[:3, 3], columns=["x", "y", "z"])

    # Set up the Rigid ICP registration
    first_avg = np.mean(cloud1.points.values,axis=0)
    second_avg = np.mean(cloud2.points.values,axis=0)

    return second_avg-first_avg, masks
   


def get_mask(mask, obj_id):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def project_to_image(point, intrinsics, transform):
    point_h = np.append(point, 1)

    point_cam = np.dot(np.linalg.inv(transform), point_h)
    x, y, z = point_cam[:3]
    
    u = (intrinsics['fx'] * x) / z + intrinsics['cx']
    v = (intrinsics['fy'] * y) / z + intrinsics['cy']

    return int(u), int(v), z


def compute_distances(cloud1_points, cloud2_points, threshold):

    print("Computing")
    threshold = 0.01

    tree = cKDTree(cloud1_points)
    distances, indices = tree.query(cloud2_points, distance_upper_bound=threshold)

    return distances




def filter_by_occlusion(far_points_pcd, original_depth, intrinsics, transform):
    filtered_indicies = []
    occlusion_mask = np.zeros(original_depth.shape, dtype=bool)

    for i, point in enumerate(far_points_pcd):
        u, v, z_new = project_to_image(point, intrinsics, transform)
        
        if 0 <= u < original_depth.shape[1] and 0 <= v < original_depth.shape[0]:
            z_old = original_depth[v, u]
            
            if z_old == 0 or z_new < z_old:
                filtered_indicies.append(i)
            else:
                occlusion_mask[v, u] = True
    
    filtered_points = far_points_pcd[filtered_indicies]
    
    return np.array(filtered_points), occlusion_mask

def depth_image_to_point_cloud(depth_image, intrinsics):
    h, w = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # Create meshgrid for image coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate normalized image coordinates
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into point cloud (N x 3)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def custom_plot(points, point_size=1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, s=point_size)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal scaling for all axes
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.show()