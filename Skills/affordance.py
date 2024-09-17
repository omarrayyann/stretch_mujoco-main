import numpy as np
import rerun as rr
import time
from queue import Queue
import cv2

module_id = "SKILLS"
# Untidy-Bot Modules
import Manipulation
import Utils
import Classes
import SAM2_Tapnet

def create_numpy_image(frame, points):

    mask_image = (frame.mask).astype(np.uint8)
    rgb_image = (frame.rgb).astype(np.uint8)
    
    mask_image_3channel = np.stack((mask_image,mask_image*255,mask_image), axis=-1)
    
    # combined_image = np.maximum(mask_image_3channel, rgb_image)
    combined_image = rgb_image
    
    def draw_circle(image, center, radius, color):
        y_center, x_center = center
        for x in range(x_center - radius, x_center + radius + 1):
            for y in range(y_center - radius, y_center + radius + 1):
                if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        image[y, x] = color
    
    for point in points:
        x, y = int(point[1]), int(point[0])
        draw_circle(combined_image, (x, y), 2, [0, 0, 255])

    return combined_image

def roll_vector(self, vector, grasp_pose, original_frame, sam_tap, max_error=0.03, incrementation=0.01, max_length=0.15, method=1):
   
    original_position = self.mjdata.body("link_grasp_center").xpos.copy()
    initial_grasp = self.mjdata.actuator("gripper").length[0].copy()

    length = 0.0
    error = 0.0
    reward = 0.0

    eef_trajectory = [original_position]
    frames = [original_frame,get_frame(self,original_position)]

    while Manipulation.low_level.is_grasping(self) == True and error<max_error and length <= (max_length-incrementation) and (self.mjdata.actuator("gripper").length[0].copy()-initial_grasp)<0.005:
        length += incrementation
        new_position = original_position + vector*length

        Utils.set_body_pose(self.mjmodel,"grasping_head",new_position,grasp_pose[0:3,0:3],self.args.debug)

        debug_status = self.args.debug
        # self.args.debug = False
        Manipulation.move_grasp(self,new_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        current_position = self.mjdata.body("link_grasp_center").xpos.copy()

        eef_trajectory.append(current_position)
        frames.append(get_frame(self,current_position))

        error = np.linalg.norm(new_position-current_position)
        Utils.print_debug(f"Error: {error}",self.args.debug) 
         
    movement_vector = eef_trajectory[-1] - eef_trajectory[0]

    for position in eef_trajectory[::-1]:
        debug_status = self.args.debug
        # self.args.debug = False
        Manipulation.move_grasp(self,position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

    eef_trajectory = np.array(eef_trajectory)
    
    if sam_tap:
        centers, normals, rewards, _, tracks = SAM2_Tapnet.compute_sam2_tapnet_reward(frames, 300)
    else:
        centers = []
        normals = []
        rewards = []

    center, _, normal = Utils.fit_circle_to_points(eef_trajectory)
    # Visualiza Circle and Arc Actual
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"actual_circle",center,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"actual_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"actual_norm",center+normal*0.02,None,self.args.debug)
    reward = compute_movement_reward(eef_trajectory, frames, method=method)

    centers.append(center)
    normals.append(normal)
    rewards.append(reward)

    original_points = tracks[0][:,0]
    original_image = create_numpy_image(frames[0], original_points)
    first_points = tracks[0][:,1]
    first_image = create_numpy_image(frames[1], first_points)
    second_points = tracks[0][:,-1]
    second_image = create_numpy_image(frames[-1], second_points)

    rr.log("Orignal Frame", rr.Image(original_image))
    rr.log("First Frame", rr.Image(first_image))
    rr.log("Last Frame", rr.Image(second_image))

    return centers, normals, rewards, movement_vector

def roll_arc(self, center, normal, grasp_pose, original_frame, sam_tap, max_error=0.03, incrementation=0.01, max_length=0.15, method=1):

    original_position = self.mjdata.body("link_grasp_center").xpos.copy()
    initial_grasp = self.mjdata.actuator("gripper").length[0].copy()

    arc_trajectory, center = Utils.generate_cylinder_trajectory(original_position, normal, center, incrementation, max_length)

    # Visualiza Circle and Arc
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"test_circle",center,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"test_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"test_norm",center+normal*0.02,None,self.args.debug)
    for traj_index in range(len(arc_trajectory)):
        Utils.set_geom_pose(self.mjmodel,f"traj_{traj_index}",arc_trajectory[traj_index],None,self.args.debug)

    Utils.sleep(10.0, self.args.debug)
    Utils.set_geom_pose(self.mjmodel,"test_circle",np.array([100,100,100]),rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"test_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"test_norm",np.array([100,100,100]),None,self.args.debug)
    for traj_index in range(len(arc_trajectory)):
        Utils.set_geom_pose(self.mjmodel,f"traj_{traj_index}",np.array([100,100,100]),None,self.args.debug)



    length = 0.0
    error = 0.0

    eef_trajectory = [original_position]
    frames = [original_frame,get_frame(self,original_position)]

    arc_index = 0
    while arc_index<len(arc_trajectory) and Manipulation.low_level.is_grasping(self) == True and error<max_error and length <= (max_length-incrementation) and (self.mjdata.actuator("gripper").length[0].copy()-initial_grasp)<0.005:

        new_position = arc_trajectory[arc_index]
        arc_index += 1

        Utils.set_body_pose(self.mjmodel,"grasping_head",new_position,grasp_pose[0:3,0:3],self.args.debug)
        
        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,new_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        current_position = self.mjdata.body("link_grasp_center").xpos.copy()
        eef_trajectory.append(current_position)
        frames.append(get_frame(self,current_position))

        error = np.linalg.norm(new_position-current_position)
        Utils.print_debug(f"Error: {error}",self.args.debug) 


    final_position = self.mjdata.body("link_grasp_center").xpos.copy()

    for position in eef_trajectory[::-1]:
        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

    eef_trajectory = np.array(eef_trajectory)

    if sam_tap:
        centers, normals, rewards, _, tracks = SAM2_Tapnet.compute_sam2_tapnet_reward(frames, 300)
    else:
        centers = []
        normals = []
        rewards = []

    center, _, normal = Utils.fit_circle_to_points(eef_trajectory)
    # Visualiza Circle and Arc Actual
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"actual_circle",center,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"actual_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"actual_norm",center+normal*0.02,None,self.args.debug)
    reward = compute_movement_reward(eef_trajectory, frames, method=method)

    centers.append(center)
    normals.append(normal)
    rewards.append(reward)

    first_points = tracks[0][:,1]
    first_image = create_numpy_image(frames[1], first_points)
    second_points = tracks[0][:,-1]
    second_image = create_numpy_image(frames[-1], second_points)

    rr.log("First Frame", rr.Image(first_image))
    rr.log("Last Frame", rr.Image(second_image))

    return centers, normals, rewards

    

def affordance(self, grasp_pose):

    rr.init("Affordance", spawn=True)

    Manipulation.move_joint_to(self,"arm",0.0)
    time.sleep(2.0)
    Manipulation.point_camera_to(self,grasp_pose[0:3,3])
    time.sleep(2)
    original_frame = get_frame(self,grasp_pose[0:3,3])
    
    reset(self)

    vectors = [  np.array([0,1,0]), np.array([1,0,0]), np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1]) ]
    vectors = [ np.array([-1,1,0]) , np.array([-1,0,0])  ]

    pre_grasp_position = self.mjdata.body("link_grasp_center").xpos.copy()
    Manipulation.low_level.grasp(self)
    Utils.sleep(2.0)

    post_grasp_position = self.mjdata.body("link_grasp_center").xpos.copy()

    circle_sampler = Utils.CircleSampler()
    cylinder_sampler = Utils.Cylinder_Sampler()
    linear_sampler = Utils.LinearSampler()

    for vector in vectors:

        Manipulation.ungrasp(self)

        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        Manipulation.grasp(self)

        centers, normals, rewards, movement_vector = roll_vector(self, vector, grasp_pose, original_frame, sam_tap=True, max_error=0.05, incrementation=0.025, max_length=0.15)
        for (center, normal, reward) in zip(centers, normals, rewards):
            circle_sampler.update(center, normal, reward)
            cylinder_sampler.update(center, normal, reward)
        linear_sampler.update(rewards[-1], movement_vector)

    for _ in range(0,1):


        Manipulation.ungrasp(self)
        
        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status
        
        Manipulation.grasp(self)

        vector = linear_sampler.sample()
        centers, normals, rewards, movement_vector = roll_vector(self, vector, grasp_pose, original_frame, sam_tap=True, max_error=0.05, incrementation=0.025, max_length=0.15)

        for (center, normal, reward) in zip(centers, normals, rewards):
            circle_sampler.update(center, normal, reward)
            cylinder_sampler.update(center, normal, reward)
        linear_sampler.update(rewards[-1], movement_vector)


    while True:
        
        Manipulation.ungrasp(self)

        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        Manipulation.grasp(self)

        center, normal = circle_sampler.sample(post_grasp_position, delta_length=0.01, total_length=0.2)
        center, normal = cylinder_sampler.sample()
        centers, normals, rewards = roll_arc(self, center, normal, grasp_pose, original_frame, sam_tap=True, max_error=0.08, incrementation=0.02, max_length=0.3)
        
        for (center, normal, reward) in zip(centers, normals, rewards):
            circle_sampler.update(center, normal, reward)
            cylinder_sampler.update(center, normal, reward)
    

def reset(self):

    q = np.array([ 0.00000000, 0.00000000, 0.05335826, 0.00494873, -0.00087141, 0.99894166, 0.00007970, 0.00178841, -0.04596037, 0, 0, 0.50990128, 0.07738893, 0.07595047, 0.07723606, 0.07566633, -0.22209796, -0.43430982, -0.05816221, 0.01955896, 0.19552636, -0.00022914, -0.00010475, 0.19544104, 0.00000000, 0.00000000, 0.00000000, -0.00451851, 0.00009783])
    # q = np.array([-0.00001880, 0.00000000, 0.05416444, 0.00050961, -0.00087150, 0.99842935, 0.00009639, 0.00178786, -0.05599665, 0, 0, 0.76072400, 0.09342424, 0.09249239, 0.09255225, 0.09356264, -0.36359575, -1.04146461, -0.49324822, 0.00809127, 0.08123380, 0.00256101, -0.04030883, 0.08120029, 0.00099282, -0.00234462, 0.00000000, -0.00451851, 0.00009783])
    # q = np.array([0.00000000, 0.10628421, 0.13789239, -0.00087139, 0.92922255, -0.00066401, 0.00166244, 0.36951622, 0, 0, 0.52022841, 0.08149477, 0.08006597, 0.08139197, 0.07993936, -0.78819279, -0.27537196, 0.36480256, 0.02001729, 0.19980054, -0.00022874, -0.00002526, 0.19975080, 0.00000000, 0.00000000, 0.00000000, -0.00451852, 0.00009783])
    
    joints_index = Manipulation.low_level.get_joints_indices(self)
    q0 = q[joints_index]
    q0 = np.block([q0[0:3], q0[3] + q0[4] + q0[5] + q0[6], q0[7:]])
    self.mjdata.ctrl = q0
    self.mjdata.qpos = q
    Utils.sleep(1.0)

# Methods
# 1: EEF Movements
# 2: SAM-2 Object Movement
def compute_movement_reward(eef_trajectory, frames, method=1):

    np.save("eef_trajectory.npy",eef_trajectory)
    import pickle
    with open('frames.pkl', 'wb') as file:
        pickle.dump(frames, file)

    reward = 0.0
    if len(eef_trajectory) > 1:
        for i in range(len(eef_trajectory)-1):
            reward += np.linalg.norm(eef_trajectory[i+1]-eef_trajectory[i])

    return reward

def get_frame(self, position):

    Utils.print_debug("Getting Frame", self.args.debug)

    Manipulation.point_camera_to(self,position)
    Utils.sleep(1.0)

    self.camera_data = self.pull_camera_data()

    rgb = cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB)
    depth = self.camera_data["cam_nav_depth"]

    camera_position = self.mjdata.camera("nav_camera_rgb").xpos[:3]
    camera_orientation = self.mjdata.camera("nav_camera_rgb").xmat.reshape((3,3))
    b2w_r = Utils.robotics_functions.quat2Mat([0, 1, 0, 0])
    camera_orientation = np.matmul(camera_orientation, b2w_r)
    pose = np.block([[camera_orientation.reshape((3,3)),camera_position.reshape((3,1))],[0,0,0,1]]).copy()
    frame = Classes.Frame(rgb, depth, None, 320, 240, 318.49075719, 318.49075719, pose)

    Utils.print_debug("Done Getting Frame", self.args.debug)

    return frame

