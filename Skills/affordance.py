import numpy as np

module_id = "SKILLS"
# Untidy-Bot Modules
import Manipulation
import Utils
import Classes
import cv2

def compute_vector_reward(self, vector, grasp_pose, max_error=0.03, incrementation=0.01, max_length=0.15):
   
    original_position = self.mjdata.body("link_grasp_center").xpos.copy()
    initial_grasp = self.mjdata.actuator("gripper").length[0].copy()

    length = 0.0
    error = 0.0
    reward = 0.0
    trajectory = [original_position]

    while Manipulation.low_level.is_grasping(self) == True and error<max_error and length <= (max_length-incrementation) and (self.mjdata.actuator("gripper").length[0].copy()-initial_grasp)<0.005:

        length += incrementation
        new_position = original_position + vector*length

        Utils.set_body_pose(self.mjmodel,"grasping_head",new_position,grasp_pose[0:3,0:3],self.args.debug)

        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,new_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        current_position = self.mjdata.body("link_grasp_center").xpos.copy()
        trajectory.append(current_position)

        error = np.linalg.norm(new_position-current_position)
        Utils.print_debug(f"Error: {error}",self.args.debug) 
         

    movement_vector = trajectory[-1] - trajectory[0]

    for position in trajectory[::-1]:
        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

    trajectory = np.array(trajectory)


    for traj_index in range(len(trajectory)):
        Utils.set_geom_pose(self.mjmodel,f"test_{traj_index}",trajectory[traj_index],None)


    center, _, normal = Utils.fit_circle_to_points(trajectory)

    # Visualiza Circle and Arc Actual
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"actual_circle",center,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"actual_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"actual_norm",center+normal*0.02,None,self.args.debug)

    if center is None:
        return None, None, reward, movement_vector

    # reward = movement_vector[0]*movement_vector[0] + movement_vector[1]*movement_vector[1] + movement_vector[2]*movement_vector[2]
    reward = 0.0
    if len(trajectory) > 1:
        for i in range(1,len(trajectory)):
            reward += np.linalg.norm(trajectory[i]-trajectory[i-1])

    return center, normal, reward, movement_vector

def compute_arc_reward(self, center, normal, grasp_pose, max_error=0.03, incrementation=0.01, max_length=0.15, stay=False):

    original_position = self.mjdata.body("link_grasp_center").xpos.copy()
    initial_grasp = self.mjdata.actuator("gripper").length[0].copy()

    arc_trajectory = generate_trajectory(original_position, normal, center, incrementation, max_length)

    # Visualiza Circle and Arc
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"test_circle",center,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"test_circle",np.linalg.norm(original_position-center))
    Utils.set_geom_pose(self.mjmodel,"test_norm",center+normal*0.02,None,self.args.debug)
    for traj_index in range(len(arc_trajectory)):
        Utils.set_geom_pose(self.mjmodel,f"traj_{traj_index}",arc_trajectory[traj_index],None)

    length = 0.0
    error = 0.0
    trajectory = [original_position]

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
        trajectory.append(current_position)
        
        error = np.linalg.norm(new_position-current_position)
        Utils.print_debug(f"Error: {error}",self.args.debug) 

    final_position = self.mjdata.body("link_grasp_center").xpos.copy()
    movement_vector = (final_position-original_position)

    if stay==True:
        for position in trajectory[::-1]:
            debug_status = self.args.debug
            self.args.debug = False
            Manipulation.move_grasp(self,position,grasp_pose[0:3,0:3],2,False,False)
            self.args.debug = debug_status

    trajectory = np.array(trajectory)

    for traj_index in range(len(trajectory)):
        Utils.set_geom_pose(self.mjmodel,f"test_{traj_index}",trajectory[traj_index],None)

    c_fitted, _, normal = Utils.fit_circle_to_points(trajectory)
    if c_fitted is None:
        return None, None, None
    
    # Visualiza Circle and Arc Actual
    rot = Utils.rotation_matrix_to_align_with_vector(normal)
    Utils.set_geom_pose(self.mjmodel,"actual_circle",c_fitted,rot,self.args.debug)
    Utils.set_geom_cylinder_radius(self.mjmodel,"actual_circle",np.linalg.norm(original_position-c_fitted))
    Utils.set_geom_pose(self.mjmodel,"actual_norm",c_fitted+normal*0.02,None,self.args.debug)

    # reward = movement_vector[0]*movement_vector[0] + movement_vector[1]*movement_vector[1] + movement_vector[2]*movement_vector[2]
    reward = 0.0
    if len(trajectory) > 1:
        for i in range(1,len(trajectory)):
            reward += np.linalg.norm(trajectory[i]-trajectory[i-1])

    return c_fitted, normal, reward
    

def affordance(self, grasp_pose):
    
    # reset(self)

    vectors = [ np.array([1,0,0]), np.array([-1,0,0]), np.array([0,1,0]), np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1]) ]
    # vectors = [ np.array([-1,0,0]), np.array([0,1,0])]

    pre_grasp_position = self.mjdata.body("link_grasp_center").xpos.copy()
    Manipulation.low_level.grasp(self)
    Utils.sleep(2.0)

    post_grasp_position = self.mjdata.body("link_grasp_center").xpos.copy()

    circle_sampler = CircleSampler()
    linear_sampler = LinearSampler()

    for vector in vectors:


        Manipulation.ungrasp(self)
        Manipulation.point_camera_to(self,pre_grasp_position)

        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        Manipulation.grasp(self)

        center, normal, reward, movement_vector = compute_vector_reward(self, vector, grasp_pose, max_error=0.05, incrementation=0.03, max_length=0.1)
        if center is not None:
            circle_sampler.update(center, normal, reward)
        linear_sampler.update(reward, movement_vector)

    for _ in range(0,1):


        Manipulation.ungrasp(self)
        
        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status
        
        Manipulation.grasp(self)

        vector = linear_sampler.sample()
        center, normal, reward, movement_vector = compute_vector_reward(self, vector, grasp_pose, max_error=0.06, incrementation=0.02, max_length=0.1)

        circle_sampler.update(center, normal, reward)
        linear_sampler.update(reward, movement_vector)


    for i in range(0,2):
        
        Manipulation.ungrasp(self)

        debug_status = self.args.debug
        self.args.debug = False
        Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
        self.args.debug = debug_status

        Manipulation.grasp(self)

        center, normal = circle_sampler.sample(post_grasp_position, delta_length=0.01, total_length=0.2)
        center, normal, reward = compute_arc_reward(self, center, normal, grasp_pose, max_error=0.08, incrementation=0.01, max_length=0.3)
        if center is not None:
            circle_sampler.update(center, normal, reward)

    # Final:

    Manipulation.ungrasp(self)
    Manipulation.point_camera_to(self,pre_grasp_position)
    first_frame = get_frame(self)

    debug_status = self.args.debug
    self.args.debug = False
    Manipulation.move_grasp(self,pre_grasp_position,grasp_pose[0:3,0:3],2,False,False)
    self.args.debug = debug_status

    Manipulation.grasp(self)

    center, normal = circle_sampler.sample(post_grasp_position, delta_length=0.01, total_length=0.2)
    center, normal, reward = compute_arc_reward(self, center, normal, grasp_pose, max_error=0.08, incrementation=0.01, max_length=0.3, stay=True)

    Manipulation.point_camera_to(self,pre_grasp_position)
    last_frame = get_frame(self)

    return first_frame, last_frame

    

    
        

class LinearSampler:

    def __init__(self):
        
        self.centers = []
        self.direction = []
        self.rewards = []
        self.total_rewards = [0,0,0,0,0,0]

    def update(self, reward, vector):

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
    
    def sample(self):

        vector = np.array([ self.total_rewards[0]-self.total_rewards[1] , self.total_rewards[2]-self.total_rewards[3] , self.total_rewards[4]-self.total_rewards[5] ])
        vector_magnitude = np.linalg.norm(vector)
        vector = vector/vector_magnitude

        return vector

    
class CircleSampler:

    def __init__(self):
        self.centers = []
        self.normals = []
        self.rewards = []

    def update(self, center, normal, reward):
        self.centers.append(center)
        self.normals.append(normal)
        self.rewards.append(reward)
            
    def sample(self, original_position, delta_length=0.025, total_length=0.2):
        
        print(self.centers)
        min_radius = min(np.linalg.norm(np.array(original_position) - np.array(center)) for center in self.centers)

        if delta_length > min_radius:
            delta_length = min_radius / 2.0
        if total_length > 2 * np.pi * min_radius:
            total_length = 2 * np.pi * min_radius / 2.0

        num_steps = int(total_length / delta_length) + 1
        weighted_trajectory = np.zeros((num_steps, 3))
        total_reward = sum(self.rewards)
        
        for center, normal, reward in zip(self.centers, self.normals, self.rewards):
            trajectory = np.array(generate_trajectory(original_position, normal, center, delta_length, total_length))
            weighted_trajectory += trajectory * reward
        
        averaged_trajectory = weighted_trajectory / total_reward

        c_fitted, r_fitted, normal = Utils.fit_circle_to_points(averaged_trajectory)

        return c_fitted, normal
    
def generate_trajectory(start_point, normal_vector, circle_center, delta_length, total_length):
    x0, y0, z0 = start_point
    cx, cy, cz = circle_center
    
    # Calculate the normal vector and the vector from the circle center to the start point
    n_vec = np.array(normal_vector) / np.linalg.norm(normal_vector)
    r_vec = np.array([x0 - cx, y0 - cy, z0 - cz])
    
    # Project start_point onto the plane of the circle
    distance_from_plane = np.dot(r_vec, n_vec)  # Distance from start_point to the plane
    projection_on_plane = np.array(start_point) - distance_from_plane * n_vec

    # Calculate the radius vector in the plane from the circle center to the projection
    vec_from_center_to_projection = projection_on_plane - np.array(circle_center)
    radius = np.linalg.norm(vec_from_center_to_projection)  # Distance from center to the projection
    
    # Correct the projection point to lie exactly on the circle
    if not np.isclose(radius, 0):
        vec_from_center_to_projection_normalized = vec_from_center_to_projection / radius
        start_point_on_circle = np.array(circle_center) + vec_from_center_to_projection_normalized * radius
    else:
        raise ValueError("Starting point cannot coincide with the circle center.")

    # New radius vector using the corrected start point
    r_vec = start_point_on_circle - np.array(circle_center)
    radius = np.linalg.norm(r_vec)  # Re-calculate radius after correction

    # First orthogonal vector (in the plane of the circle)
    u1 = r_vec / radius

    # Second orthogonal vector (perpendicular to both normal and u1)
    u2 = np.cross(n_vec, u1)
    u2 = u2 / np.linalg.norm(u2)

    # Angle increment per step
    delta_theta = delta_length / radius

    # Trajectory generation
    trajectory = []
    num_steps = int(total_length / delta_length)

    for i in range(num_steps + 1):
        theta = i * delta_theta
        point = np.array(circle_center) + radius * (np.cos(theta) * u1 + np.sin(theta) * u2)
        trajectory.append(point)

    return trajectory

def reset(self):

    # q = np.array([ 0.00000000, 0.00000000, 0.05335826, 0.00494873, -0.00087141, 0.99894166, 0.00007970, 0.00178841, -0.04596037, 0, 0, 0.50990128, 0.07738893, 0.07595047, 0.07723606, 0.07566633, -0.22209796, -0.43430982, -0.05816221, 0.01955896, 0.19552636, -0.00022914, -0.00010475, 0.19544104, 0.00000000, 0.00000000, 0.00000000, -0.00451851, 0.00009783])
    q = np.array([-0.00001880, 0.00000000, 0.05416444, 0.00050961, -0.00087150, 0.99842935, 0.00009639, 0.00178786, -0.05599665, 0, 0, 0.76072400, 0.09342424, 0.09249239, 0.09255225, 0.09356264, -0.36359575, -1.04146461, -0.49324822, 0.00809127, 0.08123380, 0.00256101, -0.04030883, 0.08120029, 0.00099282, -0.00234462, 0.00000000, -0.00451851, 0.00009783])
    # q = np.array([0.00000000, 0.10628421, 0.13789239, -0.00087139, 0.92922255, -0.00066401, 0.00166244, 0.36951622, 0, 0, 0.52022841, 0.08149477, 0.08006597, 0.08139197, 0.07993936, -0.78819279, -0.27537196, 0.36480256, 0.02001729, 0.19980054, -0.00022874, -0.00002526, 0.19975080, 0.00000000, 0.00000000, 0.00000000, -0.00451852, 0.00009783])
    
    joints_index = Manipulation.low_level.get_joints_indices(self)
    q0 = q[joints_index]
    q0 = np.block([q0[0:3], q0[3] + q0[4] + q0[5] + q0[6], q0[7:]])
    self.mjdata.ctrl = q0
    self.mjdata.qpos = q
    Utils.sleep(1.0)


def get_frame(self):

    Utils.print_debug("Getting Frame", self.args.debug)

    self.camera_data = self.pull_camera_data()

    rgb = cv2.cvtColor(self.camera_data["cam_nav_rgb"], cv2.COLOR_BGR2RGB)
    
    return rgb