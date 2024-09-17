import numpy as np
import os
import shutil
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import matplotlib.pyplot as plt
import rerun as rr
from pyntcloud import PyntCloud
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
from multiprocessing import Process, Queue
import cv2
from pycpd import RigidRegistration


module_id = "FRAMES_COMPARE"
import Utils
# Constants
image_width = 640
image_height = 480




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

def compute_distances(cloud1_points, cloud2_points, threshold):

    threshold = 0.01

    tree = cKDTree(cloud1_points)
    distances, indices = tree.query(cloud2_points, distance_upper_bound=threshold)

    return distances