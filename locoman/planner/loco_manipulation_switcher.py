import numpy as np
import torch
from robot.base_robot import BaseRobot
from scipy.spatial.transform import Rotation as R
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy


class LocoManipulationSwitcher:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._device = self._robot._device
        self._env_ids = env_ids
        self._cfg = self._robot._cfg

        self._manipulate_leg_idx = self._cfg.fsm_switcher.stance_and_locomanipulation.manipulate_leg_idx
        self._no_manipulate_leg_idx = 0 if self._manipulate_leg_idx == 1 else 1
        self._desired_eef_rpy_w = self._cfg.loco_manipulation.desired_eef_rpy_w.copy()
        self._transition_frames = int(self._cfg.fsm_switcher.stance_and_locomanipulation.transition_time / self._robot._dt)
        self._stablize_frames = int(self._cfg.fsm_switcher.stance_and_locomanipulation.stablize_time / self._robot._dt)

        desired_eef_pva = np.zeros((2, 9))
        desired_eef_pva[0, 0:3] = self._cfg.gripper.reset_pos_sim[0:3]
        desired_eef_pva[1, 0:3] = self._cfg.gripper.reset_pos_sim[4:7]

        self._transition_action = {"action_mode": 0,
                "contact_state": np.ones(4, dtype=bool),
                "body_pva": np.zeros(18),
                "footeef_pva": np.zeros((3, 12)),
                "eef_pva": desired_eef_pva,
                "gripper_angles": np.zeros(2) * self._cfg.gripper.reset_pos_sim[3],
                }

        # to be updated
        # self._transition_action["eef_pva"][:, 0:3] = self._robot.eef_rpy_w_np[self._env_ids].copy()
        self._transition_action["eef_pva"][self._manipulate_leg_idx, 0:3] = self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        # self._current_eef_rot_w = self._robot.eef_rot_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        self._current_eef_rot_w = rpy_to_rot_mat(self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx])
        
        self._current_frame = 0
        delta_rotation = rpy_to_rot_mat(self._desired_eef_rpy_w) @ (self._current_eef_rot_w.T)
        r = R.from_matrix(delta_rotation)
        rotvec = r.as_rotvec()
        if np.linalg.norm(rotvec) > 1e-6:
            self._delta_eef_rotation_axis = rotvec / np.linalg.norm(rotvec)
            self._delta_eef_ratation_angle = np.linalg.norm(rotvec)
        else:
            self._delta_eef_rotation_axis = np.ones(3)
            self._delta_eef_ratation_angle = 0.


    def reset(self):
        # self._transition_action["eef_pva"][:, 0:3] = self._robot.eef_rpy_w_np[self._env_ids].copy()
        self._transition_action["eef_pva"][self._manipulate_leg_idx, 0:3] = self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        # self._current_eef_rot_w[:] = self._robot.eef_rot_w_np[self._env_ids, self._manipulate_leg_idx].copy()
        self._current_eef_rot_w[:] = rpy_to_rot_mat(self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx])
        self._current_frame = 0
        delta_rotation = rpy_to_rot_mat(self._desired_eef_rpy_w) @ (self._current_eef_rot_w.T)
        r = R.from_matrix(delta_rotation)
        rotvec = r.as_rotvec()
        if np.linalg.norm(rotvec) > 1e-6:
            self._delta_eef_rotation_axis[:] = rotvec / np.linalg.norm(rotvec)
            self._delta_eef_ratation_angle = np.linalg.norm(rotvec)
        else:
            self._delta_eef_rotation_axis[:] = np.ones(3)
            self._delta_eef_ratation_angle = 0.

    def compute_command_for_wbc(self):
        self._current_frame += 1
        tracking_ratio = max(0.0, min(1.0, self._current_frame / self._transition_frames))
        rot_interpolated = R.from_rotvec(self._delta_eef_ratation_angle*tracking_ratio*self._delta_eef_rotation_axis).as_matrix()
        self._transition_action["eef_pva"][self._manipulate_leg_idx, 0:3] = rot_mat_to_rpy(rot_interpolated @ self._current_eef_rot_w)
        return self._transition_action

    def feedback(self, command_executed):
        pass

    def check_finished(self):
        return self._current_frame >= self._transition_frames + self._stablize_frames



