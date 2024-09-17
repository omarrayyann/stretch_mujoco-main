import numpy as np

module_id = "SKILLS"
# Untidy-Bot Modules
import Grasping
import Classes
import Skills
import Manipulation
import Locomotion
import Path_Planning
import Utils
import GPT
import time
import matplotlib.pyplot as plt
import rerun as rr

from sam2.build_sam import build_sam2_video_predictor

import cv2

class Vector3DSystem:
    def __init__(self):
        self.vectors = []
        self.colors = []

    def add_vector(self, vector):
        print("VECTOR: ", vector)
        self.vectors.append(np.array(vector))
        self.normalize_vectors()

    def normalize_vectors(self):
        if not self.vectors:
            return
        
        magnitudes = [np.linalg.norm(v) for v in self.vectors]
        max_magnitude = max(magnitudes)

        normed_vectors = [v / max_magnitude for v in self.vectors]

        self.colors = [ [1.0, 0.984, 0.0, 1.0]  for _ in self.vectors]
        longest_vector_index = magnitudes.index(max_magnitude)
        self.colors[longest_vector_index] = [0.5, 0.5, 0.5, 1.0] 

        return normed_vectors

    def visualize_vectors(self):
        # Define the reference axes (X, Y, Z)

        normalized_vectors = self.normalize_vectors()

        reference_origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        reference_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        reference_colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]

        # Combine the reference axes with the user vectors
        all_origins = np.vstack((reference_origins, np.zeros((len(normalized_vectors), 3))))
        all_vectors = np.vstack((reference_vectors, np.array(normalized_vectors)))
        all_colors = reference_colors + self.colors

        # Log all the arrows (reference axes + vectors) in one call
        rr.log("arrows", rr.Arrows3D(
            origins=all_origins,
            vectors=all_vectors,
            colors=all_colors,
            
        ))


def compute_reward(self, vector, grasp_pose, reference_frame):

    # q = np.array([ 0.00000000, 0.00000000, 0.05335826, 0.00494873, -0.00087141, 0.99894166, 0.00007970, 0.00178841, -0.04596037, 0, 0, 0.50990128, 0.07738893, 0.07595047, 0.07723606, 0.07566633, -0.22209796, -0.43430982, -0.05816221, 0.01955896, 0.19552636, -0.00022914, -0.00010475, 0.19544104, 0.00000000, 0.00000000, 0.00000000, -0.00451851, 0.00009783])
    q = np.array([-0.00001880, 0.00000000, 0.05416444, 0.00050961, -0.00087150, 0.99842935, 0.00009639, 0.00178786, -0.05599665, 0, 0, 0.76072400, 0.09342424, 0.09249239, 0.09255225, 0.09356264, -0.36359575, -1.04146461, -0.49324822, 0.00809127, 0.08123380, 0.00256101, -0.04030883, 0.08120029, 0.00099282, -0.00234462, 0.00000000, -0.00451851, 0.00009783])
    # q = np.array([0.00000000, 0.10628421, 0.13789239, -0.00087139, 0.92922255, -0.00066401, 0.00166244, 0.36951622, 0, 0, 0.52022841, 0.08149477, 0.08006597, 0.08139197, 0.07993936, -0.78819279, -0.27537196, 0.36480256, 0.02001729, 0.19980054, -0.00022874, -0.00002526, 0.19975080, 0.00000000, 0.00000000, 0.00000000, -0.00451852, 0.00009783])
    joints_index = Manipulation.low_level.get_joints_indices(self)
    q0 = q[joints_index]
    q0 = np.block([q0[0:3], q0[3] + q0[4] + q0[5] + q0[6], q0[7:]])

    self.mjdata.ctrl = q0
    self.mjdata.qpos = q
    
    Manipulation.low_level.grasp(self)
    time.sleep(5)
    Manipulation.point_camera_to(self,grasp_pose[0:3,3])
    time.sleep(2)

    rgb_frames = []
    
    camera_position = self.mjdata.camera("nav_camera_rgb").xpos[:3]
    camera_orientation = self.mjdata.camera("nav_camera_rgb").xmat.reshape((3,3))
    b2w_r = Utils.robotics_functions.quat2Mat([0, 1, 0, 0])
    camera_orientation = np.matmul(camera_orientation, b2w_r)
    first_pose = np.block([[camera_orientation.reshape((3,3)),camera_position.reshape((3,1))],[0,0,0,1]]).copy()
    self.camera_data = self.pull_camera_data()
    first_rgb = cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB)
    first_depth = self.camera_data["cam_nav_depth"]
    first_frame = Classes.Frame(first_rgb, first_depth, None, 320, 240, 318.49075719, 318.49075719, first_pose)

    rgb_frames.append(cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB))

    first_position = self.mjdata.body("link_grasp_center").xpos.copy()

    length = 0.0
    max_length = 0.25
    incrementation = 0.025
    error = 0.0
    max_error = 0.05

    while Manipulation.low_level.is_grasping(self) == True and error<max_error and length <= (max_length-incrementation):

        length += incrementation
        second_position = first_position + vector*length
        Utils.set_body_pose(self.mjmodel,"grasping_head",second_position,grasp_pose[0:3,0:3])
        self.args.debug = False
        Manipulation.move_grasp(self,second_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = True
        Utils.sleep(5.0)
        current_position = self.mjdata.body("link_grasp_center").xpos.copy()
        Manipulation.point_camera_to(self,current_position)
        Utils.sleep(2)
        error = np.linalg.norm(second_position-current_position)
        print("error: ", error)  
        self.camera_data = self.pull_camera_data()
        rgb_frames.append(cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB))

    Manipulation.point_camera_to(self,current_position)
    time.sleep(2)

    camera_position = self.mjdata.camera("nav_camera_rgb").xpos[:3]
    camera_orientation = self.mjdata.camera("nav_camera_rgb").xmat.reshape((3,3))
    b2w_r = Utils.robotics_functions.quat2Mat([0, 1, 0, 0])
    camera_orientation = np.matmul(camera_orientation, b2w_r)
    second_pose = np.block([[camera_orientation.reshape((3,3)),camera_position.reshape((3,1))],[0,0,0,1]]).copy()
    self.camera_data = self.pull_camera_data()
    second_rgb = cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB)
    second_depth = self.camera_data["cam_nav_depth"]
    second_frame = Classes.Frame(second_rgb, second_depth, None, 320, 240, 318.49075719, 318.49075719, second_pose)

    rgb_frames.append(cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB))

    # reward, masked_image = Skills.find_delta(self, reference_frame, first_frame, second_frame)

    move_vector, masks = Skills.find_delta_move(self, reference_frame, first_frame, second_frame)

    first_frame_mask = masks[1,:,:,1:]
    second_frame_mask = masks[2,:,:,:1:]

    np.save("masks.npy",masks)
    np.save("rgb_frames.npy",rgb_frames)


    first_image = first_frame_mask*0.4*255 + first_frame.rgb*0.6
    second_image = second_frame_mask*0.4*255 + second_frame.rgb*0.6

    rr.log("First-Image", rr.Image(first_image))
    rr.log("Second-Image", rr.Image(second_image))
    
    mag = np.linalg.norm(move_vector)
    move_vector_norm = move_vector/mag


    return mag, move_vector_norm

def vector_projection_scalar(a, b):

    dot_product = np.dot(a, b)

    b_dot_b = np.dot(b, b)

    projection_scalar = dot_product / b_dot_b
    
    return projection_scalar

    
def add_reward(vector, reward, total_rewards):

    if vector[0]>0:
        total_rewards[0] += reward*abs(vector[0])
    else:
        total_rewards[1] += reward*abs(vector[0])

    if vector[1]>0:
        total_rewards[2] += reward*abs(vector[1])
    else:
        total_rewards[3] += reward*abs(vector[1])

    if vector[2]>0:
        total_rewards[4] += reward*abs(vector[2])
    else:
        total_rewards[5] += reward*abs(vector[2])

    print("total rewards: ", total_rewards)
    
def get_vectors(grasp_pose):

    rot = grasp_pose[0:3, 0:3]
    trans = grasp_pose[0:3, 3]

    back = -rot[0:3, 0]
    front = rot[0:3, 0]

    outside = -rot[0:3, 2]
    inside = rot[0:3, 2]
    down = -rot[0:3, 1]
    up = rot[0:3, 1]

    return [down,up,back,front,inside,outside]

def affordance(self, grasp_pose):

    vector_system = Vector3DSystem()

    rr.init("Affordance", spawn=True)

    # vectors = get_vectors(grasp_pose)

    vectors = [  np.array([-1,0,0]), np.array([0,1,0]),  np.array([1,0,0]),  np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1]) ]

    total_rewards = np.zeros(6)

    movementNode = MovementNode(
        start_point=grasp_pose[0:3,3],
        end_point=grasp_pose[0:3,3],
        previous_direction=None,
        test_grasp_pose=grasp_pose,
    )

    Manipulation.move_joint_to(self,"arm",0.0)
    time.sleep(2.0)
    Manipulation.point_camera_to(self,grasp_pose[0:3,3])
    time.sleep(2)

    camera_position = self.mjdata.camera("nav_camera_rgb").xpos[:3]
    camera_orientation = self.mjdata.camera("nav_camera_rgb").xmat.reshape((3,3))
    b2w_r = Utils.robotics_functions.quat2Mat([0, 1, 0, 0])
    camera_orientation = np.matmul(camera_orientation, b2w_r)
    reference_pose = np.block([[camera_orientation.reshape((3,3)),camera_position.reshape((3,1))],[0,0,0,1]]).copy()
    self.camera_data = self.pull_camera_data()
    reference_rgb = cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB)
    reference_depth = self.camera_data["cam_nav_depth"]
    reference_frame = Classes.Frame(reference_rgb, reference_depth, None, 320, 240, 318.49075719, 318.49075719, reference_pose)

    for direction in vectors:

        reward, vector = compute_reward(self, direction, grasp_pose, reference_frame)

        rr.log("Reward", rr.Scalar(reward))

        # rr.log("Last-Attempt", rr.Clear(recursive=True))
        # rr.log("Last-Attempt", rr.Image(masked_image))

        if reward != 0:
            vector_system.add_vector(np.array(vector)*reward)

        vector_system.visualize_vectors()

        if reward is not None:
            movementNode.update_reward(reward,vector)
        else:
            print("Ungrasped")


    

    while True:

        direction = movementNode.sample_movement()

        print(f"Testing direction: {direction}")

        reward, vector = compute_reward(self, direction, grasp_pose, reference_frame)

        rr.set_time_sequence("step", 100)
        rr.log("Reward", rr.Scalar(reward))

        # rr.log("Last-Attempt", rr.Clear(recursive=True))
        # rr.log("Last-Attempt", rr.Image(masked_image))
        if reward != 0:
            vector_system.add_vector(np.array(vector)*reward)

        vector_system.visualize_vectors()

        if reward is not None:
            movementNode.update_reward(reward,vector)
        else:
            print("Ungrasped")

        second_direction = movementNode.sample_movement()

        print("Direction DIfference: ", np.linalg.norm(second_direction-direction))
    
    return movementNode
        

class MovementNode:

    def __init__(self, start_point, end_point, previous_direction, test_grasp_pose):
        
        self.test_grasp_pose = test_grasp_pose

        self.previous_direction = previous_direction
        self.start_point = start_point
        self.end_point = end_point

        self.total_rewards = np.ones(6)

    def update_reward(self, reward, vector):

        vector = vector/np.linalg.norm(vector)

        if vector[0]>0:
            self.total_rewards[0] += reward*abs(vector[0])
        else:
            self.total_rewards[1] += reward*abs(vector[0])

        if vector[1]>0:
            self.total_rewards[2] += reward*abs(vector[1])
        else:
            self.total_rewards[3] += reward*abs(vector[1])

        if vector[2]>0:
            self.total_rewards[4] += reward*abs(vector[2])
        else:
            self.total_rewards[5] += reward*abs(vector[2])
    
    def sample_movement(self):

        vector = np.array([ self.total_rewards[0]-self.total_rewards[1] , self.total_rewards[2]-self.total_rewards[3] , self.total_rewards[4]-self.total_rewards[5] ])
        vector_magnitude = np.linalg.norm(vector)
        vector = vector/vector_magnitude

        return vector

    
    