import numpy as np

module_id = "SKILLS-PICK"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Locomotion
import Path_Planning
import Utils
import GPT

def pick(self, object, visualize_path=False):

    Utils.print_debug(f"Started picking: {object.name}",self.args.debug,module_id)

    Utils.print_debug(f"Unextending arm",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"arm",0.0)

    Utils.print_debug(f"Lifting arm up",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"lift",1.1)

    starting_position = self.mjdata.body("base_link").xpos.copy()[:2]
    ending_position = object.position[:2]
    waypoints = Path_Planning.find_path(self,self.pcd_points,self.pcd_labels,starting_position,ending_position,self.args.min_distance,visualize_path)   

    for waypoint in waypoints[1:]:
        Locomotion.move_base_to(self,waypoint)
        
    successfully_picked = False
    while not successfully_picked:
        
        Locomotion.orient_base_position(self,object.position)
        Utils.sleep(1.0)
        
        Manipulation.point_camera_to(self,object.position)
        Utils.sleep(5.0)

        optimal_grasp_pose = None
        optimal_grasp_width = 0.3
                
        if not self.args.disable_any_grasp:

            Utils.print_debug(f"Getting RGB-D frames from navigation camera (cam_d435i)",self.args.debug,module_id)
            self.camera_data = self.pull_camera_data()
            camera_data_copy = dict(self.camera_data)
            rgb = camera_data_copy["cam_d435i_rgb"]
            depth = camera_data_copy["cam_d435i_depth"]*1000

            Utils.print_debug(f"Computing grasp pose using AnyGrasp",self.args.debug,module_id)

            grasp_poses, _, grasp_widths, _, langsam_indicies, trajectory_indicies = Grasping.get_grasps(self,rgb,depth,object.name,self.args.trajectory_length,self.args.trajectory_radius)
            grasp_poses = grasp_poses.copy()

            if len(grasp_poses) > 0:

                Utils.print_debug(f"Transforming grasps to world frame",self.args.debug,module_id)
                for i in range(len(grasp_poses)):
                    grasp_world_pose = Utils.camera_to_world_frame(self.mjdata,"d435i_camera_rgb",grasp_poses[i])
                    grasp_poses[i] = grasp_world_pose

                Utils.print_debug(f"Showing all grasp poses found",self.args.debug,module_id)
                for i in range(len(grasp_poses)):
                    Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",grasp_poses[i][0:3,3],grasp_poses[i][0:3,0:3],self.args.debug)
                Utils.sleep(5,self.args.debug)
                for i in range(len(grasp_poses)):
                    Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",np.array([100,100,100]),None)

                Utils.print_debug(f"Showing grasps filtered by the LangSAM prompt {object.name}",self.args.debug,module_id)
                for i, index in enumerate(langsam_indicies):
                    grasp_world_pose = grasp_poses[index]
                    Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",grasp_world_pose[0:3,3],grasp_world_pose[0:3,0:3],self.args.debug)
                Utils.sleep(5,self.args.debug)
                for i in range(len(grasp_poses)):
                    Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",np.array([100,100,100]),None)

                common_indicies = np.intersect1d(langsam_indicies, trajectory_indicies)
                if len(common_indicies)>0:
                    Utils.print_debug(f"Showing grasps filtered by the LangSAM prompt {object.name} and trajectory",self.args.debug,module_id)
                    grasp_indices = common_indicies
                    for i, index in enumerate(grasp_indices):
                        grasp_world_pose = grasp_poses[index]
                        Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",grasp_world_pose[0:3,3],grasp_world_pose[0:3,0:3],self.args.debug)
                for i in range(len(grasp_poses)):
                    Utils.set_body_pose(self.mjmodel,f"grasping_head_{i}",np.array([100,100,100]),None)

                if len(common_indicies) == 0:
                    Utils.print_debug(f"Failed to find optimal grasp.",self.args.debug,module_id)
                    if len(langsam_indicies)==0:
                        Utils.print_debug(f"Using grasp closest to object's {object.name} center.",self.args.debug,module_id)
                        min_distance = 100000
                        grasp_index = 0
                        for index, grasp in enumerate(grasp_poses):       
                            x = grasp[0,3]
                            y = grasp[1,3]
                            z = grasp[2,3]
                            distance = ((object.position[0]-x)**2 + (object.position[1]-y)**2 + (object.position[2]-z)**2)**0.5
                            if distance < min_distance:
                                min_distance = distance
                                grasp_index = index
                        optimal_grasp_pose = grasp_poses[grasp_index]
                        optimal_grasp_width = grasp_widths[grasp_index]
                    else:
                        Utils.print_debug(f"Using langsam grasp with highest score (trajectory-collision not filtered).",self.args.debug,module_id)
                        optimal_grasp_pose = grasp_poses[langsam_indicies[0]]
                        optimal_grasp_width = grasp_widths[langsam_indicies[0]]
                else:
                    optimal_grasp_pose = grasp_poses[common_indicies[0]]
                    optimal_grasp_width = grasp_widths[common_indicies[0]]

                # Adjusting grasp width
                optimal_grasp_width = optimal_grasp_width/2 - 0.01
                
        if optimal_grasp_pose is None:
            optimal_target_position = np.array(object.position)
            optimal_grasp_pose = Utils.homogeneous_matrix(Utils.rotation_matrix_x(np.pi),optimal_target_position.reshape((3,1)))
            optimal_grasp_width = 0.3

        Utils.print_debug(f"Showing choosen grasp.",self.args.debug,module_id)
        Utils.set_body_pose(self.mjmodel,"grasping_head_0",optimal_grasp_pose[0:3,3],optimal_grasp_pose[0:3,0:3],self.args.debug)
        Utils.sleep(5,self.args.debug)

        Locomotion.orient_base_grasp(self,optimal_grasp_pose[0:3,3])
        Utils.sleep(1.0)

        Manipulation.ungrasp(self,optimal_grasp_width)
        Utils.sleep(1.0)

        Utils.set_body_pose(self.mjmodel,"grasping_head_0",optimal_grasp_pose[0:3,3],optimal_grasp_pose[0:3,0:3],self.args.debug)
        trajectory_positions, extra_pos = Utils.get_trajectory_steps(optimal_grasp_pose,self.args.trajectory_steps,self.args.trajectory_length,deep=0.04)
        for i, trajectory_position in enumerate(trajectory_positions):
            Utils.set_body_pose(self.mjmodel,"grasping_head",trajectory_position,optimal_grasp_pose[0:3,0:3])
            Manipulation.move_grasp(self,trajectory_position,optimal_grasp_pose[0:3,0:3],2,lift_arm_first=i==0,move_first=i==0)
            Utils.sleep(0.1)
        Utils.sleep(1)

        Utils.set_body_pose(self.mjmodel,"grasping_head_0",np.array([100,100,100]),None)
        Utils.set_body_pose(self.mjmodel,"grasping_head",np.array([100,100,100]),None)

        Utils.print_debug(f"Closing gripper",self.args.debug,module_id)
        successfully_picked = Manipulation.grasp(self)
        Utils.sleep(3)

        Utils.print_debug(f"Lifting arm up",self.args.debug,module_id)
        Manipulation.move_joint_to_timed(self,"lift",1.1,5)

        Utils.print_debug(f"Unextending arm",self.args.debug,module_id)
        Manipulation.move_joint_to_timed(self,"arm",0.0,5)

        

        Utils.sleep(0.5)
    
    Utils.print_debug(f"Finished picking: {object.name}",self.args.debug,module_id)