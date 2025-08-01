import pickle
import numpy as np
import torch
from robot.base_robot import BaseRobot


class BimanipulationSwitcher:
    def __init__(self, robot: BaseRobot, env_ids=0, stand_up: bool = True):
        self._robot = robot
        self._device = self._robot._device
        self._env_ids = env_ids

        self._fps_ratio = int(self._robot._cfg.bimanual_trajectory.recording_fps_mul_motor_controller_fps)
        self._data_step_time = self._robot._cfg.motor_controller.dt / self._fps_ratio
        self._move_gripper_foot_frames = int(self._robot._cfg.bimanual_trajectory.move_gripper_foot_time / self._data_step_time)

        bimanual_config = self._robot._cfg.bimanual_trajectory.with_gripper if self._robot._use_gripper else self._robot._cfg.bimanual_trajectory.without_gripper
        traj_dir = bimanual_config.trajectory_path
        self._data = pickle.load(open(traj_dir, "rb"))

        self._stand_up_start_frame = int(bimanual_config.stand_up_start_time / self._data_step_time)
        self._stand_up_end_frame = int(bimanual_config.stand_up_end_time / self._data_step_time)
        self._stabilize_frames = int(bimanual_config.stabilize_time / self._data_step_time)

        self._stand_down_start_frame = int(bimanual_config.stand_down_start_time / self._data_step_time)
        self._stand_down_end_frame = int(bimanual_config.stand_down_end_time / self._data_step_time)

        self._stand_up = stand_up
        if self._stand_up:
            self._data = self._data[self._stand_up_start_frame:self._stand_up_end_frame]
        else:
            self._data = self._data[self._stand_down_start_frame:self._stand_down_end_frame]

        self._frame_idx = 0
        self._stand_up_total_frames = len(self._data) + self._stabilize_frames + self._move_gripper_foot_frames
        self._stand_down_total_frames = self._move_gripper_foot_frames + len(self._data)
        
        # buffers for command
        total_dof = 18 if self._robot._use_gripper else 12
        self._motor_pos = np.zeros(total_dof)
        self._motor_vel = np.zeros(total_dof)
        self._motor_torque = np.zeros(total_dof)
        self._dog_joint_idx = self._robot._cfg.asset.dog_joint_idx
        self._front_legs_joint_idx = self._dog_joint_idx[0:6]
        self._gripper_joint_idx = self._robot._cfg.asset.gripper_joint_idx

        # put out the manipulators
        self._front_legs_end_pos = np.zeros(6)
        if self._robot._use_gripper:
            self._dof_idx = self._robot._cfg.gripper.dof_idx
            self._gripper_init_pos = self._robot._cfg.gripper.reset_pos_sim[self._dof_idx]
            self._gripper_bimanual_pos = self._gripper_init_pos.copy()
            self._gripper_bimanual_pos[0] = 0.
            self._gripper_bimanual_pos[3] = 0.
            # need to be update
            self._gripper_end_pos = np.zeros_like(self._gripper_init_pos)

        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self._all_env_idx = torch.arange(self._robot._num_envs, device=self._device)
        self._frame_idx = 0
        self._front_legs_end_pos[:] = self._robot.joint_pos_np[self._env_ids, self._front_legs_joint_idx]
        if self._robot._use_gripper:
            self._motor_pos[self._gripper_joint_idx] = self._gripper_init_pos
            self._gripper_end_pos[:] = self._robot.joint_pos_np[self._env_ids, self._gripper_joint_idx]

    def compute_motor_command(self):
        # for standing up
        if self._stand_up:
            data_frame_idx = min(self._frame_idx, len(self._data) - 1)
            self._motor_pos[self._dog_joint_idx] = self._data[data_frame_idx]["motor_position"][:12].copy()
            self._motor_vel[self._dog_joint_idx] = self._data[data_frame_idx]["motor_vel"][:12].copy()
            self._motor_torque[self._dog_joint_idx] = self._data[data_frame_idx]["motor_torque"][:12].copy()
            if self._robot._use_gripper and self._frame_idx >= len(self._data) - 1 + self._stabilize_frames:
                    move_gripper_foot_ratio = min((self._frame_idx - (len(self._data) - 1)) / self._move_gripper_foot_frames, 1)
                    self._motor_pos[self._gripper_joint_idx] = (1-move_gripper_foot_ratio) * self._gripper_init_pos + move_gripper_foot_ratio * self._gripper_bimanual_pos
        # for standing down
        else:
            data_frame_idx = min(max(self._frame_idx - self._move_gripper_foot_frames, 0), len(self._data) - 1)
            self._motor_pos[self._dog_joint_idx] = self._data[data_frame_idx]["motor_position"][:12].copy()
            self._motor_vel[self._dog_joint_idx] = self._data[data_frame_idx]["motor_vel"][:12].copy()
            self._motor_torque[self._dog_joint_idx] = self._data[data_frame_idx]["motor_torque"][:12].copy()

            if self._frame_idx <= self._move_gripper_foot_frames:
                move_gripper_foot_ratio = min(self._frame_idx / self._move_gripper_foot_frames, 1)
                self._motor_pos[self._front_legs_joint_idx] = (1 - move_gripper_foot_ratio) * self._front_legs_end_pos + move_gripper_foot_ratio * self._motor_pos[self._front_legs_joint_idx]
                self._motor_vel[self._front_legs_joint_idx] = move_gripper_foot_ratio * self._motor_vel[self._front_legs_joint_idx]
                self._motor_torque[self._front_legs_joint_idx] = move_gripper_foot_ratio * self._motor_torque[self._front_legs_joint_idx]
                if self._robot._use_gripper:
                    self._motor_pos[self._gripper_joint_idx] = (1 - move_gripper_foot_ratio) * self._gripper_end_pos + move_gripper_foot_ratio * self._gripper_init_pos

        self._frame_idx += self._fps_ratio
        return self._motor_pos, self._motor_vel, self._motor_torque

    def feedback(self, command_executed):
        pass

    def check_finished(self):
        if self._stand_up:
            return self._frame_idx >= self._stand_up_total_frames
        else:
            return self._frame_idx >= self._stand_down_total_frames

















