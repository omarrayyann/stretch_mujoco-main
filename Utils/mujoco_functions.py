import mujoco
import numpy as np
from Utils.robotics_functions import *
from scipy.spatial.transform import Rotation as R

def set_geom_pose(model, name, pos=None, rot=None, debug_mode=True):
    if debug_mode:
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if pos is not None:
            pos = np.array(pos)
            model.geom_pos[target_geom_id] = pos.tolist()
        if rot is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat,rot.flatten())
            model.geom_quat[target_geom_id] = quat
        model.geom_rgba[target_geom_id][3] = 0.5

def set_geom_size(model, name, size):
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_size[target_geom_id] = size

def set_geom_cylinder_radius(model, name, size):
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_size[target_geom_id][0] = size

def set_body_pose(model, name, pos=None, rot=None, debug_mode=True):
    if debug_mode:
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if pos is not None:
            model.body_pos[target_body_id] = np.array(pos).tolist()
        if rot is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.flatten())
            model.body_quat[target_body_id] = quat

def hide_geom(model, name):
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    model.geom_rgba[target_geom_id][3] = 0.0

def camera_to_world_frame(mjdata, camera_name, T_camera_grasp):

    camera_position = mjdata.camera(camera_name).xpos[:3]
    camera_orientation = mjdata.camera(camera_name).xmat.reshape((3,3))
    T_world_camera = np.eye(4)
    T_world_camera[0:3,0:3] = camera_orientation
    T_world_camera[0:3,3] = camera_position

    trans_mat = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,-1,0],[0,0,0,1]])
    T_world_camera = T_world_camera @ trans_mat
    
    grasp = T_world_camera@T_camera_grasp
    grasp = grasp@homogeneous_matrix(rotation_matrix_z(np.pi/2)@rotation_matrix_x(np.pi/2),None)
 
    return grasp

def rotation_matrix_to_align_with_vector(normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    initial_vector = np.array([0, 0, 1])

    axis = np.cross(initial_vector, normal_vector)
    angle = np.arccos(np.dot(initial_vector, normal_vector))
    
    if np.linalg.norm(axis) == 0:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis)
    
    rotation = R.from_rotvec(axis * angle)
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix