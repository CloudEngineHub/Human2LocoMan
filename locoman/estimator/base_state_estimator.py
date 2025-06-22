import torch
from utilities.rotation_utils import rot_mat_to_rpy, rot_mat_to_quaternion


class BaseStateEstimator:
    def __init__(self, robot):
        self._robot = robot
        self._num_envs = self._robot._num_envs
        self._device = self._robot._device
        self._dt = self._robot._dt

        # root state
        self._base_pos_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._base_rot_mat_w2b = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._base_lin_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._base_ang_vel_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._base_lin_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._base_ang_acc_w = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)


    def reset(self, env_ids: torch.Tensor):
        self._base_pos_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_rot_mat_w2b[env_ids] = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_lin_vel_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_ang_vel_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_lin_acc_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_ang_acc_w[env_ids] = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)

    def update(self):
        raise NotImplementedError

    def set_robot_base_state(self):
        self._robot._base_pos_w[:] = self._base_pos_w
        self._robot._base_rot_mat_w2b[:] = self._base_rot_mat_w2b
        self._robot._base_rot_mat_b2w[:] = self._base_rot_mat_w2b.transpose(-2, -1)
        self._robot._base_rpy_w2b[:] = rot_mat_to_rpy(self._base_rot_mat_w2b)
        self._robot._base_quat_w2b[:] = rot_mat_to_quaternion(self._base_rot_mat_w2b)

        self._robot._base_lin_vel_w[:] = self._base_lin_vel_w
        self._robot._base_lin_vel_b[:] = torch.matmul(self._robot._base_rot_mat_b2w, self._base_lin_vel_w.unsqueeze(-1)).squeeze(-1)
        self._robot._base_ang_vel_w[:] = self._base_ang_vel_w
        self._robot._base_ang_vel_b[:] = torch.matmul(self._robot._base_rot_mat_b2w, self._base_ang_vel_w.unsqueeze(-1)).squeeze(-1)

        self._robot._base_lin_acc_w[:] = self._base_lin_acc_w
        self._robot._base_ang_acc_w[:] = self._base_ang_acc_w


    def print_robot_base_state(self):
        print('-----------------set_robot_base_state-----------------')
        print('base_pos_w: \n', self._robot._base_pos_w)
        # print('base_rot_mat_w2b: \n', self._robot._base_rot_mat_w2b)
        print('base_rpy_w2b: \n', self._robot._base_rpy_w2b)
        # print('base_quat_w2b: \n', self._robot._base_quat_w2b)
        print('base_lin_vel_w: \n', self._robot._base_lin_vel_w)
        print('base_ang_vel_w: \n', self._robot._base_ang_vel_w)
        # print('base_lin_acc_w: \n', self._robot._base_lin_acc_w)
        # print('base_ang_acc_w: \n', self._robot._base_ang_acc_w)

    @property
    def base_pos_w(self):
        return self._base_pos_w

    @property
    def base_rot_mat_w2b(self):
        return self._base_rot_mat_w2b
    
    @property
    def base_lin_vel_w(self):
        return self._base_lin_vel_w

    @property
    def base_ang_vel_w(self):
        return self._base_ang_vel_w

    @property
    def base_rpy_w2b(self):
        return self._base_rpy_w2b




