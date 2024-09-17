mport os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import socket
from PIL import Image
from lang_sam import LangSAM

server_host = 'localhost'
server_port = 9875

# Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_host, server_port))
server_socket.listen(1)

print(f"Server listening on {server_host}:{server_port}...")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.07, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.05, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Manually specifying the arguments
    args_list = [
        '--checkpoint_path', 'log/checkpoint_detection.tar',  # Required argument
        '--max_gripper_width', '0.08',
        '--gripper_height', '0.05',
        '--top_down_grasp',
        '--debug'
    ]

    cfgs = parser.parse_args(args_list)
    return cfgs

def receive_data(client_socket):
    # Receive size of data
    size_bytes = client_socket.recv(4)
    data_size = int.from_bytes(size_bytes, byteorder='big')

    # Receive data
    received_data = b""
    while len(received_data) < data_size:
        chunk = client_socket.recv(data_size - len(received_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        received_data += chunk
    return received_data

def receive_string(client_socket):
    # Receive size of string
    size_bytes = client_socket.recv(4)
    string_size = int.from_bytes(size_bytes, byteorder='big')

    # Receive string data
    received_string_data = b""
    while len(received_string_data) < string_size:
        chunk = client_socket.recv(string_size - len(received_string_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        received_string_data += chunk
    return received_string_data.decode('utf-8')

def handle_client(client_socket, anygrasp):
    print("Client connected.")

    # Receive colors data
    colors_data = receive_data(client_socket)
    print(f"Received colors data ({len(colors_data)} bytes)")

    # Receive depths data
    depths_data = receive_data(client_socket)
    print(f"Received depths data ({len(depths_data)} bytes)")

    # Receive Prompt
    prompt_string = receive_string(client_socket)
    print(f"Received prompt string: {prompt_string}")

    # Convert received data to numpy arrays
    colors = np.frombuffer(colors_data, dtype=np.float32).reshape((640, 480, 3))/255.0
    depths = np.frombuffer(depths_data, dtype=np.float32).reshape((640, 480))

    # get camera intrinsics
    fx, fy = 432.97146127, 432.97146127
    cx, cy = 240, 320
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.2
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    
    model = LangSAM()
    colors_int = (colors * 255).astype(np.uint8).reshape((640, 480, 3))
    print(colors_int.shape)
    image_pil =  Image.fromarray(colors_int)
    masks, boxes, phrases, logits = model.predict(image_pil, prompt_string)
    print(masks.shape)
    np.save("masks.npy",masks)

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
    
    print("lims: ", lims)


    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z >= zmin) & (points_z <= zmax) & (points_y >= ymin) & (points_y <= ymax) & (points_x >= xmin) & (points_x <= xmax)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

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

    # Lang SAM
    # model = LangSAM()
    # colors_int = (colors * 255).astype(np.uint8).reshape((640, 480, 3))
    # print(colors_int.shape)
    # image_pil =  Image.fromarray(colors_int)
    # masks, boxes, phrases, logits = model.predict(image_pil, prompt_string)
    # print(masks.shape)
    # np.save("masks.npy",masks)
    
    # Serialize the grasp poses and scores
    grasp_poses_data = np.array(grasp_poses, dtype=np.float32).tobytes()
    grasp_scores_data = np.array(grasp_scores, dtype=np.float32).tobytes()
    grasp_widths_data = np.array(grasp_widths, dtype=np.float32).tobytes()
    langsam_mask_data = np.array(masks, dtype=bool).tobytes()

    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        
        # Save point cloud and grippers
        o3d.io.write_point_cloud("output/cloud.ply", cloud)
        for i, gripper in enumerate(grippers):
            o3d.io.write_triangle_mesh(f"output/gripper_{i}.ply", gripper)


    # Send the size of the data and the data
    client_socket.sendall(len(grasp_poses_data).to_bytes(4, byteorder='big'))
    client_socket.sendall(grasp_poses_data)
    client_socket.sendall(len(grasp_scores_data).to_bytes(4, byteorder='big'))
    client_socket.sendall(grasp_scores_data)
    client_socket.sendall(len(grasp_widths_data).to_bytes(4, byteorder='big'))
    client_socket.sendall(grasp_widths_data)
    client_socket.sendall(len(langsam_mask_data).to_bytes(4, byteorder='big'))
    client_socket.sendall(langsam_mask_data)
    client_socket.close()
    print("Client disconnected.")

while True:
    cfgs = get_args()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()
    client_socket, addr = server_socket.accept()
    handle_client(client_socket, anygrasp)

