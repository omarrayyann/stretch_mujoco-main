import time
import threading
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData
import cv2
import os
import numpy as np
import pygame
from Utils.robotics_functions import *
from GPT.gpt import *
from Utils.constants import *
from Utils.mujoco_functions import *
from Utils.misc import *

class UntidyBotSimulator():

    def __init__(
            self,
            scene_xml_path="Environment/scene.xml",
            debug_mode=False,
            ):
        
        self.mjmodel = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.debug_mode = debug_mode
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()
        self.wheel_diameter = 0.1016
        self.wheel_seperation = 0.3153
        self.camera_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, 'nav_camera_rgb')
        self.lock = threading.Lock()
        self.rgb_dir = "./recordings/rgb"
        self.depth_dir = "./recordings/depth"
        self.pose_dir = "./recordings/poses"

        self.hide("base_link")
        self.hide("link_head")
        self.hide("link_head_tilt")
        self.hide("link_mast")
        self.hide("link_aruco_right_base")
        self.hide("link_aruco_left_base")
        self.hide("link_head_pan")
        self.hide("link_arm_l0")
        self.hide("link_arm_l1")
        self.hide("link_arm_l2")
        self.hide("link_arm_l3")
        self.hide("link_arm_l4")
        self.hide("link_wrist_yaw")
        self.hide("link_gripper_finger_left")
        self.hide("link_gripper_finger_right")
        self.hide("rubber_tip_right")
        self.hide("rubber_tip_left")
        self.hide("link_SG3_gripper_body")
        self.hide("link_DW3_wrist_pitch")
        self.hide("laser")
        self.hide("base_imu")
        self.hide("link_left_wheel")
        self.hide("link_right_wheel")

    def hide(self, body_name):
        body_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_id in range(self.mjmodel.ngeom):
            if self.mjmodel.geom_bodyid[geom_id] == body_id:
                self.mjmodel.geom_rgba[geom_id] = [1, 1, 1, 0]  # 
        
    def home(self)->None:
        self.mjdata.ctrl = self.mjmodel.keyframe('home').ctrl
    
    def set_base_velocity(self,v_linear:float,omega:float)->None:
        w_left, w_right = self.diff_drive_inv_kinematics(v_linear, omega)
        self.mjdata.actuator("left_wheel_vel").ctrl = w_left
        self.mjdata.actuator("right_wheel_vel").ctrl = w_right

    def get_base_orientation(self):
        rot_matrix = self.mjdata.body("base_link").xmat.reshape(3, 3)
        theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
        return theta

    def pull_camera_data(self)->dict:
        data = {}
        data['time'] = self.mjdata.time
        with self.lock:
            self.rgb_renderer.update_scene(self.mjdata, 'nav_camera_rgb')
            self.depth_renderer.update_scene(self.mjdata, 'nav_camera_rgb')
            data['cam_nav_rgb'] = cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR)
            data['cam_nav_depth'] = self.depth_renderer.render()
        return data

    def __ctrl_callback(self, model: MjModel, data: MjData)->None:
        self.mjdata = data
        self.mjmodel = model

    def diff_drive_inv_kinematics(self,
                                      V:float,
                                      omega:float)->tuple:
        R = self.wheel_diameter / 2
        L = self.wheel_seperation
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")
        
        w_left = (V - (omega * L / 2)) / R
        w_right = (V + (omega * L / 2)) / R
        
        return (w_left, w_right)

    def diff_drive_fwd_kinematics(self,
                              w_left:float,
                              w_right:float)->tuple:
        R = self.wheel_diameter / 2
        L = self.wheel_seperation
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")
        
        V = R * (w_left + w_right) / 2.0
        omega = R * (w_right - w_left) / L
        
        return (V, omega)

    def __run(self)->None:
        mujoco.set_mjcb_control(self.__ctrl_callback)
        mujoco.viewer.launch(self.mjmodel,show_left_ui=False,show_right_ui=False)


    def start(self)->None:
        threading.Thread(target=self.__run).start()
        time.sleep(0.5)
        self.home()
        time.sleep(0.5)
    
    def record_data(self):

        with self.lock:
            # Pull camera data from nav cam
            self.rgb_renderer.update_scene(self.mjdata, 'nav_camera_rgb')
            self.depth_renderer.update_scene(self.mjdata, 'nav_camera_rgb')
            rgb_image = cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR)
            depth_image = self.depth_renderer.render()
            time_stamp = self.mjdata.time
            
            # Pull XYZ position data of the camera
            camera_pos = self.mjdata.cam_xpos[self.camera_id]
            camera_rot = self.mjdata.cam_xmat[self.camera_id].reshape(3, 3)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = camera_rot
            pose_matrix[:3, 3] = camera_pos

            b2w_r = quat2Mat([0, 1, 0, 0])
            pose_matrix[0:3,0:3] = np.matmul(pose_matrix[0:3,0:3], b2w_r)

            # Save RGB image
            rgb_filename = os.path.join(self.rgb_dir, f"{self.frame_count:06d}.jpg")
            cv2.imwrite(rgb_filename, rgb_image)
            
            # Save depth image
            depth_filename = os.path.join(self.depth_dir, f"{self.frame_count:06d}.npy")
            np.save(depth_filename, depth_image)
            
            # Save pose as .npy file
            pose_filename = os.path.join(self.pose_dir, f"{self.frame_count:06d}.npy")
            np.save(pose_filename, pose_matrix)
            
            # Display RGB image
            cv2.imshow('cam_nav_rgb', rgb_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

            self.frame_count += 1
          
        return True
        
    def move_camera(self, vertical_change: float):
        self.mjmodel.cam_pos[self.camera_id][1] += vertical_change

    def rotate_camera(self, pitch_change: float):
        cam_quat = self.mjmodel.cam_quat[self.camera_id]
        cam_mat = np.zeros((9)) 
        mujoco.mju_quat2Mat(cam_mat,cam_quat)
        cam_mat = cam_mat.reshape((3,3))
        rot_axis = np.array([1, 0, 0])  # Pitch rotation around the x-axis
        rot_quat = np.zeros((4))
        mujoco.mju_axisAngle2Quat(rot_quat, rot_axis, np.deg2rad(pitch_change))
        res_quat = np.zeros((4))
        mujoco.mju_mulQuat(res_quat, rot_quat, cam_quat)
        self.mjmodel.cam_quat[self.camera_id] = res_quat

    def keyboard_control(self):

        create_or_empty_dir(self.depth_dir)
        create_or_empty_dir(self.rgb_dir)
        create_or_empty_dir(self.pose_dir)
        
        self.frame_count = 0

        v_linear = 0.0
        omega = 0.0
        linear_speed = 10.0
        angular_speed = 10.0
        camera_vertical_speed = 0.1
        rotation_speed = 10
        print("Control the robot using the PS4 controller's arrow buttons:")
        print("Up Arrow: move forward")
        print("Down Arrow: move backward")
        print("Left Arrow: rotate left")
        print("Right Arrow: rotate right")
        print("Triangle: move camera up")
        print("X: move camera down")
        print("Circle: rotate camera up")
        print("Square: rotate camera down")
        print("Escape: quit")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYHATMOTION:
                    hat = event.hat
                    value = event.value
                    if value == (0, 1):  # Up Arrow
                        v_linear = linear_speed
                        omega = 0.0
                    elif value == (0, -1):  # Down Arrow
                        v_linear = -linear_speed
                        omega = 0.0
                    elif value == (-1, 0):  # Left Arrow
                        v_linear = 0.0
                        omega = angular_speed
                    elif value == (1, 0):  # Right Arrow
                        v_linear = 0.0
                        omega = -angular_speed
                    elif value == (0, 0):  # No Arrow pressed
                        v_linear = 0.0
                        omega = 0.0
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 2:  # Triangle
                        self.move_camera(vertical_change=camera_vertical_speed)
                    elif event.button == 0:  # X
                        self.move_camera(vertical_change=-camera_vertical_speed)
                    elif event.button == 1:  # Circle
                        self.rotate_camera(pitch_change=rotation_speed)
                    elif event.button == 3:  # Square
                        self.rotate_camera(pitch_change=-rotation_speed)
                    elif event.button == 5:  # R1
                        print("R1 pressed. Stopping recording and exiting.")
                        running = False
                        break

            self.set_base_velocity(v_linear, omega)
            time.sleep(0.3)

            if not self.record_data():
                running = False

if __name__ == "__main__":
    
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    robot_sim = UntidyBotSimulator()
    robot_sim.start()
    time.sleep(5)
    robot_sim.keyboard_control()
    
    cv2.destroyAllWindows()