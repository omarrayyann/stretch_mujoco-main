import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from PIL import Image
# from lang_sam import LangSAM
import gc

torch.set_grad_enabled(False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.07, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.05, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Manually specifying the arguments
    args_list = [
        '--checkpoint_path', 'Grasping/grasp_detection/log/checkpoint_detection.tar',  # Required argument
        '--max_gripper_width', '0.08',
        '--gripper_height', '0.05',
        '--top_down_grasp',
        '--debug'
    ]

    cfgs = parser.parse_args(args_list)
    return cfgs


def anygrasp_detection(colors, depths):
    
    
    torch.cuda.empty_cache()
    cfgs = get_args()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()


    colors = np.frombuffer(colors, dtype=np.float32).reshape((640, 480, 3))
    depths = np.frombuffer(depths, dtype=np.float32).reshape((640, 480))

    
    print("Colors: ", colors)
    # get camera intrinsics
    fx, fy = 432.97146127, 577.2952
    cx, cy = 240, 320
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    zmin, zmax = -1, 1
    
    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    print(points_z)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z >= zmin) & (points_z <= zmax) & (points_y >= ymin) & (points_y <= ymax) & (points_x >= xmin) & (points_x <= xmax)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)


    colors = colors/255.0
    gg, cloud = anygrasp.get_grasp(points, colors, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
    else:
        gg = gg.nms().sort_by_score()
        gg_pick = gg[:]
        print(gg_pick.scores)
        print('grasp score:', gg_pick[0].score)

    grasp_poses = []
    grasp_scores = []
    grasp_widths = []

    for g in gg_pick:
        grasp_center = g.translation
        grasp_rotation = g.rotation_matrix
        grasp_pose = np.identity(4)
        grasp_pose[0:3,0:3] = grasp_rotation
        grasp_pose[0:3,3] = grasp_center
        grasp_score = g.score
        grasp_width = g.width
        grasp_poses.append(grasp_pose)
        grasp_scores.append(grasp_score)
        grasp_widths.append(grasp_width)


    trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud.transform(trans_mat)
    grippers = gg.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    
    grasp_poses = np.array(grasp_poses, dtype=np.float32)
    grasp_scores = np.array(grasp_scores, dtype=np.float32)
    grasp_widths = np.array(grasp_widths, dtype=np.float32)
    # langsam_mask = np.array(masks, dtype=bool)

    del anygrasp, colors, depths
    torch.cuda.empty_cache()
    gc.collect()

    return grasp_poses, grasp_scores, grasp_widths