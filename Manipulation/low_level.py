import numpy as np

module_id = "MANIPULATION"
# Untidy-Bot Modules
import Grasping
import Manipulation
import Locomotion
import Path_Planning
import Utils
import Skills
import GPT

def move_q(q,self,seconds=2,lift_arm_first=True):
    
    self.done_move_q = False
    
    if lift_arm_first==False:
        steps = int(seconds / self.period)
        delta_q = (q-self.mjdata.ctrl)/steps
        for _ in range(steps+1):
            self.mjdata.ctrl[:] += delta_q[:]
            Utils.sleep(self.period)
    else:
        steps = int(seconds / self.period)
        delta_q = (q-self.mjdata.ctrl)/steps
        for _ in range(steps+1):
            self.mjdata.ctrl[0:2] += delta_q[0:2]
            self.mjdata.ctrl[4:] += delta_q[4:]
            Utils.sleep(self.period)
        for _ in range(steps+1):
            self.mjdata.ctrl[2] += delta_q[2]
            Utils.sleep(self.period)
        for _ in range(steps+1):
            self.mjdata.ctrl[3] += delta_q[3]
            Utils.sleep(self.period)

    self.done_move_q = True

def is_grasping(self):
    return self.mjdata.actuator("gripper").length[0] > -0.015

def grasp(self):
    
    Utils.print_debug(f"Closing gripper",self.args.debug,module_id)

    width = 0.02
    while width>=-0.02:
        move_joint_to(self,"gripper", width)
        width -= 0.005 # 0.001
        Utils.sleep(0.5)
        after = self.mjdata.actuator("gripper").length[0]
        if (abs(after-width)>0.02):
            break
        
    if not is_grasping(self):
        return False
    
    return True

def move_joint_to(self,actuator_name:str,pos:float)->None:
    self.mjdata.actuator(actuator_name).ctrl = pos

def move_joint_to_timed(self,actuator_name:str,pos:float, seconds:int)->None:
    current = self.mjdata.actuator(actuator_name).length[0] 
    per_second = (pos-current)/seconds
    if pos > current:
        while current < pos:
            Utils.print_debug(f"Current: {current}",self.args.debug,module_id)
            current += per_second
            self.mjdata.actuator(actuator_name).ctrl = current
            Utils.sleep(3)
    else:
        while current > pos:
            Utils.print_debug(f"Current: {current}",self.args.debug,module_id)
            current += per_second
            self.mjdata.actuator(actuator_name).ctrl = current
            Utils.sleep(3)


def ungrasp(self,width=0.03):
    Utils.print_debug(f"Opening gripper to width: {width}",self.args.debug,module_id)
    move_joint_to(self,"gripper",width)

def get_qpos_index_for_ctrl(self,ctrl_index):
    actuator_id = ctrl_index 
    joint_id = self.mjmodel.actuator_trnid[actuator_id, 0]
    qpos_index = self.mjmodel.jnt_qposadr[joint_id]
    return qpos_index

def get_joints_indices(self):
    wrong_qpos_indices = [get_qpos_index_for_ctrl(self,i) for i in range(10)]
    correct_qpos_indices = []
    for index in wrong_qpos_indices:
        if index != 0:
            correct_qpos_indices.append(index)
        else:
            correct_qpos_indices.append(correct_qpos_indices[-1]+1)
            correct_qpos_indices.append(correct_qpos_indices[-1]+1)
            correct_qpos_indices.append(correct_qpos_indices[-1]+1)
            correct_qpos_indices.append(correct_qpos_indices[-1]+1)

    return correct_qpos_indices
