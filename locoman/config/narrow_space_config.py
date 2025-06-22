from typing import Union
from params_proto import Meta
from config.config import Cfg
import numpy as np
import math

def narrow_space_config(Cnfg: Union[Cfg, Meta]):
    grp = Cnfg.gripper.reset_pos_sim

    _ = Cnfg.fsm_switcher.stance_and_manipulation
    _.base_movement = np.array([-0.04, 0.02, 0.01, .0, .0, .0])  # for right
    _.foot_movement = np.array([0., 0., 0.03])
    _.manipulator_angles = np.array([np.pi/20, 0., 0.])
    joint_2_delta = 0.3
    _.manipulation_to_stance_actions = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 0, 0, -1, grp[3], grp[7], 2.0, .1],
                                                [0., 0., 0., .0, .0, .0, -1, -1, -1, grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 1, 1, 0, -1, grp[3], grp[7], 1.5, .1],
                                                [-1, -1, 0., 0., 0., 0., 0., 0., 0., grp[0], grp[1], grp[2], grp[4], grp[5], grp[6], 0, 0, 1, -1, grp[3], grp[7], 1.0, .1],
                                                ])
    _.stance_to_manipulation_gripper_delta = np.array([0, -joint_2_delta, 0, 0, 0, joint_2_delta, 0, 0])


    # Cnfg.commander.body_pv_scale[[3, 4, 7]] = 0.  # vy, roll, pitch = 0
    Cnfg.commander.body_pv_scale[[3, 4]] = 0.  # vy, roll, pitch = 0
    Cnfg.commander.reset_manipulator_when_switch = False
    Cnfg.commander.real_limit.body_pv_limit[[7, -1]] = np.array([0.1, 0.1])  # limit vy, wz
    Cnfg.commander.gripper_angle_range[1] = 1.2
    Cnfg.commander.locomotion_height_range[0] = 0.16






