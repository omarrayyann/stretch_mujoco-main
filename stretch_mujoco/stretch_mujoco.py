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


import Classes
import Grasping
import Manipulation
import Locomotion
import Path_Planning
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
            'fx': 432.97146127,  # Focal length x
            'fy':  577.2952816868556,  # Focal length y
            'cx': 240,  # Principal point x
            'cy': 320   # Principal point y
        }

        end = (3.84091442 ,-0.60812156,  0.62934642)
        start = (1.0, -3, 0.0)
        min_distance = 0.6
        
        object_count = 1
        handle_count = 1

        global_positions = []
        global_labels = []

        do = False
        global_positions = np.load("global_positions.npy")
        global_labels = np.load("global_labels.npy")

        for i, angle in enumerate(range(10, 400, 15)):

            if not do:
                break

            Locomotion.orient_base_angle(self, angle/180*np.pi, limit=0.01)
            time.sleep(3.0)
            
            self.camera_data = self.pull_camera_data()
            rgb = self.camera_data["cam_d435i_rgb"]
            depth = self.camera_data["cam_d435i_depth"]
            response = Perception.send_detection_request("localhost", 4000, "drawer. cabinet. handle.", rgb)
   
            if response:

                if 'detections' in response:
                    Utils.plot_detections(rgb, response['detections'])
                    print("Detections:")
                    for det in response['detections']:
                        label = det['label']
                        score = det['score']
                        box = det['box']
                        
                        xmin, ymin, xmax, ymax = box
                        u = (xmin + xmax) / 2.0
                        v = (ymin + ymax) / 2.0

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
                        
                        
                        # if label == "handle":
                        #     Utils.set_body_pose(self.mjmodel, f"handle_{handle_count}", position, None, self.args.debug)
                        #     handle_count += 1
                        # else:
                        #     Utils.set_body_pose(self.mjmodel, f"object_{object_count}", position, None, self.args.debug)
                        #     object_count += 1

                        global_labels.append(label)
                        global_positions.append(position)

                else:
                    print(f"Error from server: {response.get('error', 'Unknown error')}")
            else:
                print("No response received.")
        if do:
            np.save("global_positions.npy",np.array(global_positions))
            np.save("global_labels.npy",np.array(global_labels))
        
        possible_positions = []

        for i in range(len(global_positions)):
            label = global_labels[i]
            position = global_positions[i]

            if position[2] > 0.8 or position[2] < 0.4:
                continue
            else:
                possible_positions.append(position)
                print(position)
                
            # if label == "handle":
            #     Utils.set_body_pose(self.mjmodel, f"handle_{handle_count}", position, None, self.args.debug)
            #     handle_count += 1
            # else:
            #     Utils.set_body_pose(self.mjmodel, f"object_{object_count}", position, None, self.args.debug)
            #     object_count += 1

            print(i)
        
        goal_position = possible_positions[4]

        start_position = self.mjdata.body("base_link").xpos[:2].copy()
        end_position = goal_position[0:2]

        waypoints = Path_Planning.find_path(self, start_position, end_position)
        for point in waypoints[1:]:
            Locomotion.move_base_to(self, point)
        
        Locomotion.orient_base_position(self, goal_position)
        Manipulation.point_camera_to(self, goal_position)
        print("pulling camera")
        self.camera_data = self.pull_camera_data()
        rgb = self.camera_data["cam_d435i_rgb"]
        depth = self.camera_data["cam_d435i_depth"]

        np.save("rgb.npy",rgb)
        np.save("depth.npy",depth)

        response = Perception.stretch_open.get_inference_results("localhost", 5000, rgb, depth)

        print(response)



    def instruct(self, instruction):
        
        gpt_output = GPT.call_gpt_with_json(json_objects=self.objects_json,instruction=instruction,model=self.args.gpt_model)
        actions = GPT.parse_output(self,gpt_output)

        for action in actions:
            time.sleep(0.5)
            if action.type == "pick":
                Skills.pick(self, action.object, visualize_path=False)
            if action.type == "place":
                Skills.place(self, action.object, visualize_path=False)
