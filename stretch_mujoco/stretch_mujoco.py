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
        self.fake_pcd()

        

    def prepare_pcd(self):
        if 0 and self.args.use_processed_pcd:
            self.pcd_points, self.pcd_labels, self.all_pcd_points = Path_Planning.load_processed(self.args.processed_pcd_path)
        else:
            self.pcd_points, self.pcd_labels, self.all_pcd_points = Path_Planning.preprocess(self.args.pcd_path,self.args.grid_size,0.03,0.35,self.args.processed_pcd_path)
    
    def fake_pcd(self):
        self.pcd_points, self.pcd_labels, self.all_pcd_points = Utils.make_fake_pcd(x_range=(0.45,4.8),y_range=(-1.6, -0.8))


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
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
        data["cam_d435i_depth"] = cv2.rotate(
            self.depth_renderer.render(), cv2.ROTATE_90_COUNTERCLOCKWISE
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
                mujoco.mj_step(self.mjmodel, self.mjdata)
                viewer.sync()

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
        if 1 or self.args.instruction:   
            print("Instruction")
            self.instruct(self.args.instruction)


    def instruct(self, instruction):
        
        # gpt_output = GPT.call_gpt_with_json(args.objects_path,args.relations_path,instruction,args.gpt_model)
        gpt_output = GPT.call_gpt_with_json(json_objects=self.objects_json,instruction=instruction,model=self.args.gpt_model)
        actions = GPT.parse_output(self,gpt_output)
        
        # actions = [Classes.Action("pick",Classes.Object("cup",np.array([1.39,-1.01,0.82]))),Classes.Action("place",Classes.Object("plate bin",np.array([-1.52,1.02,0.78])))]
        # actions = [Classes.Action("pick",Classes.Object("cup",np.array([-1.4,0.76,0.79]))),Classes.Action("place",Classes.Object("plate bin",np.array([0.04,-1.84,0.2])))]
        
        # actions = [Classes.Action("pick",Classes.Object("shoes",[1.39,-1.01,0.82])),Classes.Action("place",Classes.Object("plate bin",[-1.52,1.02,0.78]))]
        
        # actions = [Classes.Action("pick",Classes.Object("cup",[1.39,-1.01,0.82]))]

        for action in actions:
            time.sleep(0.5)
            if action.type == "pick":
                Skills.pick(self, action.object, visualize_path=False)
            if action.type == "place":
                Skills.place(self, action.object, visualize_path=False)

# if __name__ == "__main__":
#     self.args = Utils.parse_arguments()
#     robot_sim = UntidyBotSimulator(args)
#     robot_sim.start()

#     while True:
#         robot_sim.camera_data = robot_sim.pull_camera_data()

#         # cam_d405_rgb = cv2.cvtColor(robot_sim.camera_data["cam_d405_rgb"], cv2.COLOR_BGR2RGB)
#         # cam_d435i_rgb = cv2.cvtColor(robot_sim.camera_data["cam_d435i_rgb"], cv2.COLOR_BGR2RGB)
#         # cam_d435i_depth = cv2.cvtColor(robot_sim.camera_data["cam_d435i_depth"])

#         # np.save("depth.npy",robot_sim.camera_data["cam_d435i_depth"])
#         # np.save("rgb.npy",robot_sim.camera_data["cam_d435i_rgb"])

#         if not robot_sim.args.no_camera_visualization:

#             if robot_sim.args.rerun:
#                 rr.log("RealSense D405i", rr.Image(cam_d405_rgb))
#                 rr.log("RealSense D435i RGB", rr.Image(cam_d435i_rgb))
#             else:
#                 cv2.imshow("cam_d405_rgb", robot_sim.camera_data["cam_d405_rgb"])
#                 cv2.imshow("cam_d435i_rgb", robot_sim.camera_data["cam_d435i_rgb"])

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cv2.destroyAllWindows()