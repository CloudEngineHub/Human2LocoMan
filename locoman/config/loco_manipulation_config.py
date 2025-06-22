from typing import Union
from params_proto import Meta
from config.config import Cfg
import numpy as np
import math

def loco_manipulation_config(Cnfg: Union[Cfg, Meta]):
    Cnfg.commander.reset_manipulator_when_switch = False

    Cnfg.locomotion.gait_params = [1.8, np.pi, np.pi, 0., 0.49]  # trotting
    Cnfg.locomotion.desired_pose = np.array([0., 0., 0.32, 0.0, 0., 0.])
    # Cnfg.locomotion.gait_params = [1.8, np.pi, np.pi / 2, np.pi * 3 / 2, 0.245]  # walking
    # Cnfg.locomotion.gait_params = [1.8, np.pi * 0.5, np.pi, np.pi * 1.5, 0.245]  # walking
    # Cnfg.locomotion.gait_params = [1.8, np.pi, 0., np.pi, 0.44]  # trotting
    Cnfg.locomotion.foot_height = 0.11
    Cnfg.locomotion.foot_landing_clearance_sim = 0.0
    Cnfg.locomotion.foot_landing_clearance_real = 0.0

    _ = Cnfg.loco_manipulation
    _.loco_manipulation_commander = True
    _.swing_foot_names = ['FL']
    _.swing_foot_names = ['FR']
    _.manipulate_leg_idx =  {'FR':0, 'FL':1}[_.swing_foot_names[0]] # 0:FR, 1:FL, 2:RR, 3:RL
    _.desired_eef_state = np.array([100, 100, 0.12, 0., 0., 100])
    _.desired_eef_rpy_w = np.array([-0.15, -2.65, 0.3]) if _.manipulate_leg_idx == 0 else np.array([0.3, -3.0, 0.1])
    _.desired_ending_rpy_w = np.array([-0.15, -2.65, 0.3]) if _.manipulate_leg_idx == 0 else np.array([0.2, -3.0, -0.2])



    _ = Cnfg.gripper
    # _.kP = np.array([1, 1, 1, 1, 1, 1, 1, 1]) * 6 * 128
    # _.kD = np.array([1, 1, 1, 1, 1, 1, 1, 1]) * 40* 16

    _.kP = np.array([14, 8, 6, 10, 8, 8, 6, 10]) * 128
    _.kD = np.array([180, 65, 40, 100, 65, 65, 40, 100]) * 16

    _ = Cnfg.motor_controller.real
    _.kp_swing_loco[0:3] = 20
    _.kd_swing_loco[0:3] = 0.8

    Cnfg.loco_manipulation.loco_manipulation_mode = True
    Cnfg.loco_manipulation.record_tracking = True
    # Cnfg.loco_manipulation.record_tracking = False
    Cnfg.loco_manipulation.save_tracking = True
    Cnfg.loco_manipulation.save_tracking = False
    Cnfg.loco_manipulation.locomotion_only = False
    # Cnfg.loco_manipulation.locomotion_only = True
    Cnfg.loco_manipulation.record_start_time = .0
    Cnfg.loco_manipulation.recording_time = 6.0
    Cnfg.loco_manipulation.x_vel = 0.18
    Cnfg.loco_manipulation.y_vel = -0.0 if Cnfg.loco_manipulation.manipulate_leg_idx == 0 else 0.05
    Cnfg.loco_manipulation.recording_file_path = 'experiments/loco_manipulation/rotation_tracking'





