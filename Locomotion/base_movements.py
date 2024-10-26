import numpy as np

module_id = "LOCOMOTION"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Path_Planning
import Utils

def move_base_linear(self, position):

    Utils.print_debug(f"Moving the base linearly to closest position to: {position}",self.args.debug,module_id)

    target_x = position[0]
    target_y = position[1]

    current_x, current_y = self.mjdata.body("base_link").xpos[:2]
    displacement_x = target_x - current_x
    displacement_y = target_y - current_y
    distance = np.sqrt(displacement_x**2 + displacement_y**2)
    last_distance = 10000.0
    increased_counter = 0
    sign = None

    while distance > 0.001 and increased_counter < 10:
        if distance > last_distance:
            increased_counter += 1

        angle_to_target = np.arctan2(displacement_y, displacement_x)
        base_orientation = get_base_orientation(self)
        angle_diff = angle_to_target - base_orientation
        
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        if np.abs(angle_diff) < np.pi / 2:
            v_linear = distance * 3
        else:
            v_linear = -distance * 3
        if sign == None:
            sign = (v_linear>0)
        else:
            if sign != (v_linear>0):
                break
        sign = v_linear>0
        set_base_velocity(self, v_linear, 0)
        # Utils.sleep(self.period)
        current_x, current_y = self.mjdata.body("base_link").xpos[:2]
        if self.args.debug:
            Utils.set_geom_pose(self.mjmodel, "base_center", np.block([self.mjdata.body("base_link").xpos[0:2], 0.001]),debug_mode=self.args.debug)
        displacement_x = target_x - current_x
        displacement_y = target_y - current_y
        last_distance = distance
        distance = np.sqrt(displacement_x**2 + displacement_y**2)
    
    set_base_velocity(self,0, 0)
    Utils.hide_geom(self.mjmodel, "base_target")

def orient_base_grasp(self, target_position):
        
        Utils.print_debug(f"Orienting base to have the grasp face position: {target_position}",self.args.debug,module_id)

        target_x = target_position[0]
        target_y = target_position[1]
        current_x, current_y = self.mjdata.body("base_link").xpos[:2]
        current_theta = get_base_orientation(self)

        displacement_x = target_x - current_x
        displacement_y = target_y - current_y
        target_theta = (np.pi + np.arctan2(displacement_y, displacement_x) + (np.pi / 2)) % (2 * np.pi) - np.pi

        while abs(Utils.normalize_angle(target_theta - current_theta)) > 0.005:
            angle_diff = Utils.normalize_angle(target_theta - current_theta)
            print(angle_diff)
            omega = angle_diff * 2.
            set_base_velocity(self, 0, omega)
            Utils.sleep(self.period)
            current_theta = get_base_orientation(self)

        set_base_velocity(self, 0, 0)

def orient_base_position(self, target_position):

    Utils.print_debug(f"Orienting base to face position: {target_position[:2]}",self.args.debug,module_id)

    target_x = target_position[0]
    target_y = target_position[1]
    current_x, current_y = self.mjdata.body("base_link").xpos[:2]
    current_theta = get_base_orientation(self)

    displacement_x = target_x - current_x
    displacement_y = target_y - current_y
    target_theta = np.arctan2(displacement_y, displacement_x)

    while abs(Utils.normalize_angle(target_theta - current_theta)) > 0.005:
        angle_diff = Utils.normalize_angle(target_theta - current_theta)
        omega = angle_diff * 1.0
        set_base_velocity(self, 0, omega)
        Utils.sleep(self.period)
        current_theta = get_base_orientation(self)

    set_base_velocity(self, 0, 0)

def orient_base_angle(self, target_theta, limit=0.001):
    
    Utils.print_debug(f"Orienting base angle to: {target_theta}",self.args.debug,module_id)

    current_theta = get_base_orientation(self)

    while abs(Utils.normalize_angle(target_theta - current_theta)) > limit:
        angle_diff = Utils.normalize_angle(target_theta - current_theta)
        omega = angle_diff * 1.0
        set_base_velocity(self,0, omega)
        Utils.sleep(self.period)
        current_theta = get_base_orientation(self)

    set_base_velocity(self,0, 0)

def get_base_orientation(self):
    rot_matrix = self.mjdata.body("base_link").xmat.reshape(3, 3)
    theta = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return theta

def set_base_velocity(self,v_linear:float,omega:float)->None:
        w_left, w_right =  diff_drive_inv_kinematics(self,v_linear, omega)
        self.mjdata.actuator("left_wheel_vel").ctrl = w_left
        self.mjdata.actuator("right_wheel_vel").ctrl = w_right
        

def diff_drive_inv_kinematics(self,V:float,omega:float)->tuple:
        R = self.wheel_diameter / 2
        L = self.wheel_seperation
        if R <= 0:
            raise ValueError("Radius must be greater than zero.")
        if L <= 0:
            raise ValueError("Distance between wheels must be greater than zero.")
        
        w_left = (V - (omega * L / 2)) / R
        w_right = (V + (omega * L / 2)) / R
        
        return (w_left, w_right)

def diff_drive_fwd_kinematics(self,w_left:float,w_right:float)->tuple:
    R = self.wheel_diameter / 2
    L = self.wheel_seperation
    if R <= 0:
        raise ValueError("Radius must be greater than zero.")
    if L <= 0:
        raise ValueError("Distance between wheels must be greater than zero.")
    
    V = R * (w_left + w_right) / 2.0
    omega = R * (w_right - w_left) / L
    
    return (V, omega)

def move_base_to(self, position, angle=None):
    
    Utils.print_debug(f"Moving base to position: {position}",self.args.debug,module_id)
    Utils.set_geom_pose(self.mjmodel, "base_target", np.array([position[0],position[1],0.001]),debug_mode=self.args.debug)
    print(1)
    orient_base_position(self,position)
    move_base_linear(self,position)
    if angle:
        orient_base_angle(angle)
    Utils.hide_geom(self.mjmodel,"base_target")