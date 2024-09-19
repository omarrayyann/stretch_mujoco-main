import Utils
import numpy as np
import scipy
import copy
import threading
import mujoco

module_id = "MANIPULATION"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Path_Planning
import Locomotion
import Utils
import Skills
import GPT

# eef_type
# 0: Freeze All Base
# 1: Freeze Base Position
# 2: Allow movement forward and backward within the current orientation
# 3: Do what has the least error
def move_grasp(self, position, rotation=None, eef_type=1 , lift_arm_first=False, move_first=False):

    Utils.print_debug(f"Computing inverse kinematics to the grasping pose with position: {position}",self.args.debug,module_id)
    if eef_type == 0:
        Utils.print_debug(f"Freezing base in inverse kinematics",self.args.debug,module_id)
    if eef_type == 1:
        Utils.print_debug(f"Freezing base position in inverse kinematics",self.args.debug,module_id)
    if eef_type == 2:
        Utils.print_debug(f"Freezing base direction in inverse kinematics",self.args.debug,module_id)
    if eef_type == 3:
        Utils.print_debug(f"Freezing what results in the least error",self.args.debug,module_id)

    # Redundancy Bounding
    if rotation is not None:
        if rotation[2,1] > 0.0:
            rotation= rotation.copy()@Utils.rotation_matrix_z(np.pi)
    
    current_q = self.mjdata.ctrl[:].copy()
    Utils.set_body_pose(self.mjmodel, "grasping_head", position, rotation, self.args.debug)

    base_orientation = Locomotion.get_base_orientation(self)
    cos_theta = np.cos(base_orientation)
    sin_theta = np.sin(base_orientation)

    bounds_freeze_all_base = [
        (None, None), # left_wheel_vel
        (None, None), # right_wheel_vel
        (0., 1.1), # lift
        (0.0, 0.51), # arm
        (-1.39, 4.42), # wrist_yaw
        (-1.57, 1.57), # wrist_pitch
        (-3.14, 3.14), # wrist_roll
        (None, None), # grippers
        (None, None), # head_pan
        (None, None), # head_tilt
        (self.mjdata.body("base_link").xpos.copy()[0], self.mjdata.body("base_link").xpos.copy()[0]), # base_pos_x
        (self.mjdata.body("base_link").xpos.copy()[1], self.mjdata.body("base_link").xpos.copy()[1]), # base_pos_y
        (Locomotion.get_base_orientation(self), Locomotion.get_base_orientation(self))  # base_orientation
    ]

    bounds_freeze_base_position = [
        (None, None), # left_wheel_vel
        (None, None), # right_wheel_vel
        (0., 1.1), # lift
        (0.0, 0.51), # arm
        (-1.39, 4.42), # wrist_yaw
        (-1.57, 1.57), # wrist_pitch
        (-3.14, 3.14), # wrist_roll
        (None, None), # grippers
        (None, None), # head_pan
        (None, None), # head_tilt
        (self.mjdata.body("base_link").xpos.copy()[0], self.mjdata.body("base_link").xpos.copy()[0]), # base_pos_x
        (self.mjdata.body("base_link").xpos.copy()[1], self.mjdata.body("base_link").xpos.copy()[1]), # base_pos_y
        (-np.pi, np.pi)  # base_orientation
    ]

    bounds_allow_forward_backward = [
        (None, None), # left_wheel_vel
        (None, None), # right_wheel_vel
        (0., 1.1), # lift
        (0.0, 0.51), # arm
        (-1.39, 4.42), # wrist_yaw
        (-1.57, 1.57), # wrist_pitch
        (-3.14, 3.14), # wrist_roll
        (None, None), # grippers
        (None, None), # head_pan
        (None, None), # head_tilt
        (None, None), # base_pos_x
        (None, None), # base_pos_y
        (Locomotion.get_base_orientation(self), Locomotion.get_base_orientation(self))  # base_orientation
    ]

    q0 = current_q[:10].tolist()
    base_position = self.mjdata.body("base_link").xpos[:2].copy()
    base_orientation = Locomotion.get_base_orientation(self).copy()

    self.initial_base_xmat = self.mjdata.body("base_link").xmat.copy()
    self.initial_base_xpos = self.mjdata.body("base_link").xpos.copy()
    q0.extend([*base_position, base_orientation])

    if eef_type == 2 or eef_type == 3:
        initial_x = self.mjdata.body("base_link").xpos.copy()[0]
        initial_y = self.mjdata.body("base_link").xpos.copy()[1]

        def get_error_constrained(q, position, rotation):
            delta = (q[10] - initial_x) * cos_theta + (q[11] - initial_y) * sin_theta
            q[10] = initial_x + delta * cos_theta
            q[11] = initial_y + delta * sin_theta
            return get_error(q, self, position, rotation)

    if eef_type == 3:

        results = []
        self.mjdata_fake = copy.deepcopy(self.mjdata)
        res_all_base = scipy.optimize.minimize(get_error, q0, args=(self, position, rotation), bounds=bounds_freeze_all_base, tol=0.0001, options={"maxiter": 10000000})
        results.append((res_all_base, 0))
        if res_all_base.fun >= 0.0002:
            self.mjdata_fake = copy.deepcopy(self.mjdata)
            res_base_position = scipy.optimize.minimize(get_error, q0, args=(self, position, rotation), bounds=bounds_freeze_base_position, tol=0.0001, options={"maxiter": 10000000})
            results.append((res_base_position, 1))
            if res_base_position.fun >= 0.0002:
                self.mjdata_fake = copy.deepcopy(self.mjdata)
                res_forward_backward = scipy.optimize.minimize(get_error_constrained, q0, args=(position, rotation), bounds=bounds_allow_forward_backward, tol=0.0001, options={"maxiter": 10000000})
                results.append((res_forward_backward, 2))
        
        # Treat errors less than 0.001 as effectively zero and use the priority order
        min_error = float('inf')
        best_result = None
        best_type = None
        for res, res_type in results:
            effective_error = 0 if res.fun < 0.0001 else res.fun
            if effective_error < min_error or (effective_error == min_error and best_type is not None and res_type < best_type):
                min_error = effective_error
                best_result = res
                best_type = res_type

        res = best_result
    else:
        if eef_type == 2:
            print(1)
            self.mjdata_fake = copy.deepcopy(self.mjdata)

            print(2)
            res = scipy.optimize.minimize(get_error_constrained, q0, args=(position, rotation), bounds=bounds_allow_forward_backward, tol=0.0001, options={"maxiter": 10000000})
        else:
            self.mjdata_fake = copy.deepcopy(self.mjdata)
            res = scipy.optimize.minimize(get_error, q0, args=(self, position, rotation), bounds=bounds_freeze_all_base if eef_type == 0 else bounds_freeze_base_position, tol=0.0001, options={"maxiter": 10000000})

    current_q = np.copy(q0)
    current_q[2:7] = res.x[2:7]
    current_q[10:] = res.x[10:]
    del self.mjdata_fake 
    Utils.set_geom_pose(self.mjmodel, "base_target", np.array([current_q[10], current_q[11], 0.000001]), None, debug_mode=self.args.debug)

    Utils.print_debug(f"Found solution. Starting joints movement to the grasping pose",self.args.debug,module_id)
    if eef_type == 0 or (eef_type == 3 and best_type == 0):
        Manipulation.move_q(current_q[:10], self, lift_arm_first=lift_arm_first)
    elif eef_type == 1 or (eef_type == 3 and best_type == 1):
        if move_first:
            Locomotion.orient_base_angle(self,current_q[12])
            move_q_thread = threading.Thread(target=Manipulation.move_q, args=(current_q[:10], self,), kwargs={'lift_arm_first': lift_arm_first})
            move_q_thread.start()
            move_q_thread.join()
        else:
            move_q_thread = threading.Thread(target=Manipulation.move_q, args=(current_q[:10], self,), kwargs={'lift_arm_first': lift_arm_first})
            move_q_thread.start()
            Locomotion.orient_base_angle(self,current_q[12])
            move_q_thread.join()
    elif eef_type == 2 or (eef_type == 3 and best_type == 2):
        if move_first:
            Locomotion.move_base_linear(self,np.array([current_q[10], current_q[11]]))
            move_q_thread = threading.Thread(target=Manipulation.move_q, args=(current_q[:10], self, ), kwargs={'lift_arm_first': lift_arm_first})
            move_q_thread.start()
            move_q_thread.join()
        else:
            move_q_thread = threading.Thread(target=Manipulation.move_q, args=(current_q[:10], self,), kwargs={'lift_arm_first': lift_arm_first})
            move_q_thread.start()
            Locomotion.move_base_linear(self,np.array([current_q[10], current_q[11]]))
            move_q_thread.join()

def get_error(q, self, position, rotation=None):

    current_transformation = forward_kinematics(self, q)[0:4, 0:4]
    desired_transformation = Utils.homogeneous_matrix(rotation, position.reshape((3, 1)))
    pose_error = Utils.transformation_to_twist(Utils.inverse_homogeneous_matrix(current_transformation) @ desired_transformation).flatten()        

    if rotation is None:
        error = pose_error[3:]
        return error.dot(error)

    return pose_error.dot(pose_error)

def forward_kinematics(self, q):

    assert len(q) >= 13, "Input q must have at least 13 elements"

    base_pos_x, base_pos_y = q[10], q[11]
    base_orientation = q[12]
    q = np.block([q[0:3], q[3]/4, q[3]/4, q[3]/4, q[3]/4, q[4:10]])

    joints_index = Manipulation.low_level.get_joints_indices(self)
    self.mjdata_fake.qpos[joints_index] = q
    mujoco.mj_forwardSkip(self.mjmodel, self.mjdata_fake, 0, 1)

    current_position = self.mjdata_fake.body("link_grasp_center").xpos.copy()
    current_orientation = self.mjdata_fake.body("link_grasp_center").xmat.copy().reshape((3, 3)) @ self.griper_rot_fix

    pose_matrix = np.eye(4)
    pose_matrix[0:3, 0:3] = current_orientation
    pose_matrix[0:3, 3] = current_position

    current_base_pose = np.eye(4)
    current_base_pose[0:3, 0:3] = self.initial_base_xmat.reshape((3,3))
    current_base_pose[0:3, 3] = self.initial_base_xpos.copy()

    new_base_pose = np.eye(4)
    new_base_pose[0:2, 0:2] = np.array([
        [np.cos(base_orientation), -np.sin(base_orientation)], 
        [np.sin(base_orientation), np.cos(base_orientation)]
    ])

    new_base_pose[0:2, 3] = [base_pos_x, base_pos_y]
    new_base_pose[2, 3] = self.initial_base_xpos[2]

    inv_current_base_pose = Utils.inverse_homogeneous_matrix(current_base_pose)
    pose_matrix_world = inv_current_base_pose @ pose_matrix
    pose_matrix_new_base = new_base_pose @ pose_matrix_world
    pose_matrix = pose_matrix_new_base

    return pose_matrix

def point_camera_to(self, target_position):

    Utils.print_debug(f"Pointing the camera to center position: {target_position}",self.args.debug,module_id)

    initial_angles = [0.0, 0.0]
    bounds = [(-4.04, 1.73), (-1.53, 0.79)]
    
    self.mjdata_fake = copy.deepcopy(self.mjdata)
    result = scipy.optimize.minimize(get_error_camera, initial_angles, args=(self,target_position,), bounds=bounds)
    self.mjdata_fake = None
    del self.mjdata_fake 

    pan_angle, tilt_angle = result.x
    Utils.sleep(2)
    Manipulation.low_level.move_joint_to(self,"head_pan",pan_angle)
    Manipulation.low_level.move_joint_to(self,"head_tilt",tilt_angle)
    
def get_error_camera(angles, self, target_position):
    pan_angle, tilt_angle = angles

    joints_index = Manipulation.low_level.get_joints_indices(self)
    self.mjdata_fake.qpos[joints_index[11]] = pan_angle
    self.mjdata_fake.qpos[joints_index[12]] = tilt_angle
    mujoco.mj_forwardSkip(self.mjmodel,self.mjdata_fake,0,1)

    camera_position = self.mjdata_fake.camera("d435i_camera_rgb").xpos[:3]
    camera_orientation = self.mjdata_fake.camera("d435i_camera_rgb").xmat.reshape((3,3))

    direction_vector = np.array(target_position) - camera_position
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize

    camera_view_direction = camera_orientation @ np.array([0, 0, -1])

    error = np.arccos(np.clip(np.dot(camera_view_direction, direction_vector), -1.0, 1.0))

    return error

