import torch
import numpy as np
from robot.base_robot import BaseRobot
from robot.motors import MotorCommand, MotorControlMode

class StanceResetter:
    def __init__(self, robot: BaseRobot, env_ids=0):
        self._robot = robot
        self._dt = self._robot._dt
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._cfg = self._robot._cfg
        self._use_gripper = self._robot._use_gripper
        self._env_ids = env_ids
        self.reset()

    def reset(self, env_ids=None):
        self.initial_joint_pos = self._robot.joint_pos.clone()
        self.stable_joint_pos = self._robot._motors.init_positions
        self.reset_time = 0.6
        self.reset_total_steps = self.reset_time / self._dt
        self.reset_step = 0

    def compute_motor_command(self):
        blend_ratio = min(self.reset_step / self.reset_total_steps, 1)
        desired_joint_pos = blend_ratio * self.stable_joint_pos + (1 - blend_ratio) * self.initial_joint_pos
        stance_reset_action = MotorCommand(desired_position=desired_joint_pos,
                                kp=self._robot._motors.kps,
                                desired_velocity=torch.zeros((self._num_envs, self._robot._num_joints), device=self._device),
                                kd=self._robot._motors.kds,
                                desired_extra_torque=torch.zeros((self._num_envs, self._robot._num_joints), device=self._device))
        self.reset_step += 1
        return stance_reset_action

    def feedback(self, command_executed):
        pass

    def check_finished(self):
        finished = self.reset_step > self.reset_total_steps
        return finished
    