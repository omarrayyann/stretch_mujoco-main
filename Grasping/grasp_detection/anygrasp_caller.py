import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from PIL import Image
from lang_sam import LangSAM


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


def anygrasp_detection(colors, depths, prompt_string):
    
    torch.cuda.empty_cache()
    model = LangSAM(sam_type='vit_b')
    colors_int = (colors * 255).astype(np.uint8).reshape((640, 480, 3))
    print(colors_int.shape)
    image_pil =  Image.fromarray(colors_int)
    masks, boxes, phrases, logits = model.predict(image_pil, prompt_string)
    print(masks.shape)
    np.save("masks.npy",masks)

    
    torch.cuda.empty_cache()
    cfgs = get_args()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()


    colors = np.frombuffer(colors, dtype=np.float32).reshape((640, 480, 3))
    depths = np.frombuffer(depths, dtype=np.float32).reshape((640, 480))

    
    print("Colors: ", colors)
    # get camera intrinsics
    fx, fy = 432.97146127, 432.97146127
    cx, cy = 240, 320
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.2
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    

    min_x = 100000
    max_x = -100000
    min_y = 100000
    max_y = -1000000
    min_z = 100000
    max_z = -1000000

   
    for r in range(len(colors)):
        for c in range(len(colors[r])):
            for mask_current in masks:
                if mask_current[r,c] == 1:
                    point_z = depths[r,c] / scale
                    point_x = (c - cx) / fx * point_z
                    point_y = (r - cy) / fy * point_z
                
                    min_x = min(min_x,point_x)
                    max_x = max(max_x,point_x)

                    min_y = min(min_y,point_y)
                    max_y = max(max_y,point_y)

                    min_z = min(min_z,point_z)
                    max_z = max(max_z,point_z)
                
    # Find min and max for x and y separately
    tol = 0.1

    if min_x>max_x:
        min_x = -10000
        max_x = 100000

    if min_y>max_y:
        min_y = -10000
        max_y = 10000

    if min_z>max_z:
        min_z = -10000
        max_z = 10000

    xmin, xmax = min_x-tol, max_x+tol
    ymin, ymax = min_y-tol, max_y+tol
    zmin, zmax = min_z-tol, max_z+tol

    lims = [-1000000, 1000000, -1000000, 1000000, -1000000, 1000000 ]
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    print("lims: ", lims)

    np.save("depth.npy",depths)
    np.save("rgb.npy",colors)

    
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
    print(points.min(axis=0), points.max(axis=0))

    print("cikirs: ", colors)

    colors = colors/255.0
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

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
    
    # Save point cloud and grippers
    # o3d.io.write_point_cloud("output/cloud.ply", cloud)
    # for i, gripper in enumerate(grippers):
    #     o3d.io.write_triangle_mesh(f"output/gripper_{i}.ply", gripper)


    # o3d.visualization.draw_geometries([*grippers, cloud])
    # o3d.visualization.draw_geometries([grippers[0], cloud])

    # Serialize the grasp poses and scores
    grasp_poses = np.array(grasp_poses, dtype=np.float32)
    grasp_scores = np.array(grasp_scores, dtype=np.float32)
    grasp_widths = np.array(grasp_widths, dtype=np.float32)
    langsam_mask = np.array(masks, dtype=bool)

    return grasp_poses, grasp_scores, grasp_widths, masks


