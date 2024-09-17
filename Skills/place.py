import numpy as np

module_id = "SKILLS-PLACE"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Locomotion
import Path_Planning
import Utils
import GPT

def place(self, object, visualize_path=False):

    Utils.print_debug(f"Started placing at: {object.name}",self.args.debug,module_id)

    Utils.print_debug(f"Unextending arm",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"arm",0.0)

    Utils.print_debug(f"Lifting arm up",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"lift",1.1)

    starting_position = self.mjdata.body("base_link").xpos.copy()[:2]
    ending_position = object.position[:2]
    waypoints = Path_Planning.find_path(self,self.pcd_points,self.pcd_labels,starting_position,ending_position,self.args.min_distance,visualize_path)   
    
    print("from: ", starting_position)
    print("from: ", ending_position)

    for waypoint in waypoints[1:]:
        Locomotion.move_base_to(self,waypoint)
        
    Locomotion.orient_base_position(self,object.position)
    Utils.sleep(1.0)
    
    Manipulation.point_camera_to(self,object.position)
    Utils.sleep(1.0)

    # TO-DO: Better placing positioning

    Locomotion.orient_base_grasp(self,object.position)
    Utils.sleep(1.0)

    Utils.set_geom_pose(self.mjmodel,"eef_center",object.position,None,self.args.debug)
    Manipulation.move_grasp(self,object.position,None,5,True,True)
    Utils.sleep(1)

    Manipulation.ungrasp(self,0.04)
    Utils.sleep(1.0)

    Utils.print_debug(f"Unextending arm",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"arm",0.0)

    Utils.print_debug(f"Lifting arm up",self.args.debug,module_id)
    Manipulation.move_joint_to(self,"lift",1.1)

    Utils.sleep(0.5)

    Utils.print_debug(f"Finished placing object at {object.name}",self.args.debug,module_id)