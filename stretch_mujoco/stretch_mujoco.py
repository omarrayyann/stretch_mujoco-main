import cv2
import time
import threading
import GPT.gpt


# import rerun as rr
import mujoco
import mujoco.viewer
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import Classes
import Grasping
import Manipulation
import Locomotion
import Path_Planning
import Skills.affordance
import Utils
import Skills
import GPT
import Perception

print("imported")

class UntidyBotSimulator():

    def __init__(self, args, model, objects_json=None):
        
        # Flags Variables
        self.args = args
        
        # Mujoco Related Variables
        self.mjmodel = model
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.rgb_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer = mujoco.Renderer(self.mjmodel, height=480, width=640)
        self.depth_renderer.enable_depth_rendering()
        self.wheel_diameter = 0.1016
        self.wheel_seperation = 0.3153
        self.period = 1.0/args.frequency
        self.camera_id = mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, 'nav_camera_rgb')
        self.griper_rot_fix = Utils.rotation_matrix_x(np.pi/2)@Utils.rotation_matrix_z(np.pi/2)  
        self.camera_data = None
        self.camera_intrinsics = [432.97146127, 432.97146127, 240, 320]
        self.objects_json = objects_json

        # Starting Rerun.io server
        if self.args.rerun:
            rr.init("Untidy-Bot", spawn=True)

        # Preparing Environment PointCloud
        # self.prepare_pcd()
        # self.fake_pcd()

        

    def prepare_pcd(self):
        if 0 and self.args.use_processed_pcd:
            self.pcd_points, self.pcd_labels, self.all_pcd_points = Path_Planning.load_processed(self.args.processed_pcd_path)
        else:
            self.pcd_points, self.pcd_labels, self.all_pcd_points = Path_Planning.preprocess(self.args.pcd_path,self.args.grid_size,0.03,0.35,self.args.processed_pcd_path)
    
    def fake_pcd(self):
        
        main_rectangle = ((-0.1,5.1), (-4,-0.5))
        
        borders = [
            ((-0.2,-0.1), (-4,-0.5)),
            ((-0.1,5), (-0.6,-0.5)),
            ((5,5.1), (-4,-0.5)),
            ((-0.1,5), (-4,-3.9)),
            
        ]

        self.pcd_points, self.pcd_labels = Utils.make_fake_pcd(main_rectangle, borders, resolution=0.03)
        self.all_pcd_points = self.pcd_points


    def home(self)->None:
        self.mjdata.ctrl = self.mjmodel.keyframe('home').ctrl
    
    def stow(self) -> None:
        self.mjdata.ctrl = self.mjmodel.keyframe("stow").ctrl

    def pull_camera_data(self) -> dict:
        
        data = {}
        data["time"] = self.mjdata.time
        self.rgb_renderer.update_scene(self.mjdata, "d405_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d405_rgb")
        data["cam_d405_rgb"] = cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR)
        data["cam_d405_depth"] = self.depth_renderer.render()
        self.rgb_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        self.depth_renderer.update_scene(self.mjdata, "d435i_camera_rgb")
        data["cam_d435i_rgb"] = cv2.rotate(
            cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR),
            cv2.ROTATE_90_CLOCKWISE,
        )
        data["cam_d435i_depth"] = cv2.rotate(
            self.depth_renderer.render(), cv2.ROTATE_90_CLOCKWISE
        )
        self.rgb_renderer.update_scene(self.mjdata, "nav_camera_rgb")
        self.depth_renderer.update_scene(self.mjdata, "nav_camera_rgb")
        data["cam_nav_rgb"] = cv2.cvtColor(self.rgb_renderer.render(), cv2.COLOR_RGB2BGR)
        data["cam_nav_depth"] =  self.depth_renderer.render()

        return data

    def __ctrl_callback(self, model: mujoco.MjModel, data: mujoco.MjData)->None:
        self.mjdata = data
        self.mjmodel = model

    def __run_ubuntu(self)->None:
        mujoco.set_mjcb_control(self.__ctrl_callback)
        mujoco.viewer.launch(self.mjmodel,show_left_ui=False,show_right_ui=False)
        # while True:
        #     print(self.mjdata.body("base_link").xquat)
        #     print(self.mjdata.body("base_link").xpos)

    def __run_mac(self)->None:
        with mujoco.viewer.launch_passive(self.mjmodel,self.mjdata,show_left_ui=False,show_right_ui=False) as viewer:
            while viewer.is_running():
                start_ts = time.perf_counter()
                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()
                elapsed = time.perf_counter() - start_ts
                if elapsed < self.mjmodel.opt.timestep:
                    time.sleep(self.mjmodel.opt.timestep - elapsed)

    def start(self) -> None:
        print("Starting")
        threading.Thread(target=self.__run_ubuntu).start()
        time.sleep(0.5)
        self.home()
        time.sleep(0.5)
        print("Started")
        self.bot_work()
        # threading.Thread(target=self.bot_work).start()


    def bot_work(self) -> None:

        print("Bot Work")
        
        # if 1 or self.args.instruction:   
        #     print("Instruction")
        #     self.instruct(self.args.instruction)
        
        camera_intrinsics = {
            'fx': 432.97146127,         # Focal length x
            'fy': 432.97146127,    # Focal length y
            'cx': 240,                  # Principal point x
            'cy': 320                   # Principal point y
        }

        end = (3.84091442 ,-0.60812156,  0.62934642)
        start = (1.0, -3, 0.0)
        min_distance = 0.6
        
        object_count = 1
        handle_count = 1

        Locomotion.orient_base_angle(self, 1.67, limit=0.01)

        self.camera_data = self.pull_camera_data()
        rgb = self.camera_data["cam_d435i_rgb"]
        depth = self.camera_data["cam_d435i_depth"]

        

        points = Perception.get_points(rgb, "Find me the location of points that I can grasp in the scene using a robot paralell gripper.", host='localhost', port=8080)
        global_positions = []
        
        for i, point in enumerate(points):

            u, v = point
            u_int = int(round(u))
            v_int = int(round(v))
            u_int = np.clip(u_int, 0, depth.shape[1] - 1)
            v_int = np.clip(v_int, 0, depth.shape[0] - 1)
            depth_value = depth[v_int, u_int]
            if depth_value == 0 or np.isnan(depth_value):
                print("Invalid depth value at detection center, skipping this detection.")
                continue
            Z_cam = depth_value
            X_cam = (u - camera_intrinsics['cx']) * Z_cam / camera_intrinsics['fx']
            Y_cam = (v - camera_intrinsics['cy']) * Z_cam / camera_intrinsics['fy']
            camera_position = self.mjdata.camera("d435i_camera_rgb").xpos[:3]
            camera_orientation = self.mjdata.camera("d435i_camera_rgb").xmat.reshape((3, 3))
            position = camera_position + Z_cam*(camera_orientation@np.array([0, 0, -1])) + -Y_cam*(camera_orientation@np.array([-1, 0, 0])) + X_cam*(camera_orientation@np.array([0, 1, 0]))
            
            if position[2] > 0.8 or position[2] < 0.4:
                continue
            
            waypoints = Path_Planning.find_path(self, start, position)
            for point in waypoints[1:]:
                Locomotion.move_base_to(self, point)
            Locomotion.orient_base_position(self, position)
            Manipulation.point_camera_to(self, position)
            time.sleep(5)
            self.camera_data = self.pull_camera_data()
            rgb = self.camera_data["cam_d435i_rgb"]
            depth = self.camera_data["cam_d435i_depth"]

            new_points = Perception.get_points(rgb.copy(), "Find me the location of points that I can grasp in the scene using a robot paralell gripper.", host='localhost', port=8080)
            closest_position = None

            for point in new_points:
                u, v = point
                u_int = int(round(u))
                v_int = int(round(v))
                u_int = np.clip(u_int, 0, depth.shape[1] - 1)
                v_int = np.clip(v_int, 0, depth.shape[0] - 1)
                depth_value = depth[v_int, u_int]
                if depth_value == 0 or np.isnan(depth_value):
                    print("Invalid depth value at detection center, skipping this detection.")
                    continue
                Z_cam = depth_value
                X_cam = (u - camera_intrinsics['cx']) * Z_cam / camera_intrinsics['fx']
                Y_cam = (v - camera_intrinsics['cy']) * Z_cam / camera_intrinsics['fy']
                camera_position = self.mjdata.camera("d435i_camera_rgb").xpos[:3]
                camera_orientation = self.mjdata.camera("d435i_camera_rgb").xmat.reshape((3, 3))
                new_position = camera_position + Z_cam*(camera_orientation@np.array([0, 0, -1])) + -Y_cam*(camera_orientation@np.array([-1, 0, 0])) + X_cam*(camera_orientation@np.array([0, 1, 0]))
                if closest_position is None or np.linalg.norm(new_position - position) < np.linalg.norm(closest_position - position):
                    closest_position = new_position

            position = closest_position

            Utils.set_body_pose(self.mjmodel, f"handle_0", position, None, self.args.debug)

            Manipulation.point_camera_to(self, position)

            print("Getting grasps")

            # Get the dimensions of the image
            height, width = depth.shape

            # Calculate the radius as 50% of the width
            radius = width // 2

            # Calculate the center coordinates of the image
            center_y, center_x = height // 2, width // 2

            # Create an index grid for the image
            y, x = np.ogrid[:height, :width]

            # Calculate the distance from the center for each point
            distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Apply the mask directly, setting values outside the circle to 0
            depth[distance_from_center > radius] = 0



            # grasp_poses, grasp_scores, grasp_widths = Grasping.get_grasps(rgb, depth)

            # print("Grasps: ", grasp_poses)

            # optimal_grasp_pose = None
            # Utils.sleep(5)

            # for i, grasp_pose in enumerate(grasp_poses):
            #     grasp_world_frame = Utils.camera_to_world_frame(self.mjdata, "d435i_camera_rgb", grasp_pose)
            #     if optimal_grasp_pose is None or np.linalg.norm(grasp_world_frame[0:3, 3] - position) < np.linalg.norm(optimal_grasp_pose[0:3, 3] - position):
            #         optimal_grasp_pose = grasp_world_frame

            #     Utils.set_body_pose(self.mjmodel, f"grasping_head_{i+1}", grasp_world_frame[0:3, 3], grasp_world_frame[0:3, 0:3], self.args.debug)
            # Utils.set_body_pose(self.mjmodel, f"grasping_head_1", optimal_grasp_pose[0:3, 3], optimal_grasp_pose[0:3, 0:3], self.args.debug)

            optimal_grasp_pose = np.array([
                [1, 0, 0, position[0]],
                [0, 0, 1, position[1]],
                [0, -1, 0, position[2]],
                [0, 0, 0, 1]

            ])


            Locomotion.orient_base_grasp(self,optimal_grasp_pose[0:3,3])

            
            # for i in range(len(grasp_poses)):
            #     Utils.set_body_pose(self.mjmodel, f"grasping_head_{i+1}", np.array([100,100,100]), None)
            

            # optimal_grasp_pose = np.identity(4)
            # optimal_grasp_pose[0:3,3] = position
            # optimal_grasp_pose[0:3,0:3] = camera_orientation
            Manipulation.move_joint_to_timed(self,"lift",0.5,5)
            Manipulation.ungrasp(self)


            trajectory_positions, extra_pos = Utils.get_trajectory_steps(optimal_grasp_pose,self.args.trajectory_steps,self.args.trajectory_length,deep=0.02)
            for i, trajectory_position in enumerate(trajectory_positions):
                Utils.set_body_pose(self.mjmodel,"grasping_head",trajectory_position,optimal_grasp_pose[0:3,0:3])
                Manipulation.move_grasp(self,trajectory_position,optimal_grasp_pose[0:3,0:3],2,lift_arm_first=i==0,move_first=i==0)
                Utils.sleep(0.1)
            Utils.sleep(1)


            Manipulation.grasp(self)

            first_frame, last_frame = Skills.affordance.affordance(self,optimal_grasp_pose)

            np.save("first_frame.npy",first_frame)
            np.save("last_frame.npy",last_frame)


            plt.figure()
            plt.imshow(first_frame)
            plt.show()
            plt.figure()
            plt.imshow(last_frame)
            plt.show()





            
            global_positions.append(position)

        # for i in range(20):
        #     Utils.set_body_pose(self.mjmodel, f"handle_{i+1}", [0,0,0], None, self.args.debug)
        
        for position in global_positions:
            if position[2] > 0.8 or position[2] < 0.4:
                continue
            waypoints = Path_Planning.find_path(self, start, position)
            for point in waypoints[1:]:
                Locomotion.move_base_to(self, point)
            Locomotion.orient_base_position(self, position)
            Manipulation.point_camera_to(self, position)
            time.sleep(5)
            self.camera_data = self.pull_camera_data()
            rgb = self.camera_data["cam_d435i_rgb"]
            depth = self.camera_data["cam_d435i_depth"]




            height, width = depth.shape

            # Calculate the center and radius of the circle
            center_x, center_y = width // 2, height // 2
            radius = 100  # 200 pixels in diameter means 100 pixels in radius

            # Create a mask of zeros with the same shape as depth
            mask = np.zeros_like(depth, dtype=np.uint8)

            # Draw a filled circle on the mask with value 1 inside the circular area
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)

            # Set all points outside the circle to 0 by multiplying the depth with the mask
            depth = depth * mask




            points = Perception.get_points(rgb.copy(), "Find me the location of points that I can hold to open a cabinet.", host='localhost', port=8080)
            
            for point in points:
                u, v = point
                u_int = int(round(u))
                v_int = int(round(v))
                u_int = np.clip(u_int, 0, depth.shape[1] - 1)
                v_int = np.clip(v_int, 0, depth.shape[0] - 1)
                depth_value = depth[v_int, u_int]
                if depth_value == 0 or np.isnan(depth_value):
                    print("Invalid depth value at detection center, skipping this detection.")
                    continue
                Z_cam = depth_value
                X_cam = (u - camera_intrinsics['cx']) * Z_cam / camera_intrinsics['fx']
                Y_cam = (v - camera_intrinsics['cy']) * Z_cam / camera_intrinsics['fy']
                camera_position = self.mjdata.camera("d435i_camera_rgb").xpos[:3]
                camera_orientation = self.mjdata.camera("d435i_camera_rgb").xmat.reshape((3, 3))
                position = camera_position + Z_cam*(camera_orientation@np.array([0, 0, -1])) + -Y_cam*(camera_orientation@np.array([-1, 0, 0])) + X_cam*(camera_orientation@np.array([0, 1, 0]))
                Utils.set_body_pose(self.mjmodel, f"handle_{1}", position, None, self.args.debug)
                Utils.sleep(2,True)
                Utils.set_body_pose(self.mjmodel, f"handle_{1}", [0,0,0], None, self.args.debug)
                print(rgb.shape)
                grasp_poses, grasp_scores, grasp_widths = Grasping.get_grasps(rgb, depth)
                print("poses: ", grasp_poses)
                grasp_poses = grasp_poses.copy()
                grasp_poses_test = grasp_poses.copy()

                if len(grasp_poses) > 0:
                    
                    print("FIRST")
                    for i in range(len(grasp_poses)):
                        # grasp_world_pose = Utils.camera_to_world_frame(self.mjdata,"d435i_camera_rgb",grasp_poses[i])

                          
                        camera_position = self.mjdata.camera("d435i_camera_rgb").xpos[:3]
                        camera_orientation = self.mjdata.camera("d435i_camera_rgb").xmat.reshape((3, 3))
                        position = camera_position + grasp_poses[i][2,3]*(camera_orientation@np.array([0, 0, -1])) + -grasp_poses[i][1,3]*(camera_orientation@np.array([-1, 0, 0])) + grasp_poses[i][0,3]*(camera_orientation@np.array([0, 1, 0]))

                        
                        grasp_poses_test[i][0:3,3] = position

                  
                    Utils.sleep(10,self.args.debug)


    def instruct(self, instruction):
        
        gpt_output = GPT.call_gpt_with_json(json_objects=self.objects_json,instruction=instruction,model=self.args.gpt_model)
        actions = GPT.parse_output(self,gpt_output)

        for action in actions:
            time.sleep(0.5)
            if action.type == "pick":
                Skills.pick(self, action.object, visualize_path=False)
            if action.type == "place":
                Skills.place(self, action.object, visualize_path=False)
