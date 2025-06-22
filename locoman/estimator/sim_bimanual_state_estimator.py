import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy, quat_to_rot_mat, rpy_vel_to_skew_synmetric_mat, skew_symmetric_mat_to_rpy_vel
import numpy as np

class SimBiManualStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        # robot state in reset timing
        self._world_pos_sim = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_quat_sim2w = torch.zeros((self._num_envs, 4), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_quat_sim2w[:, 3] = 1.0
        self._world_rot_mat_w2sim = torch.eye(3, dtype=torch.float, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._reshaped_tensor = torch.zeros_like(self._robot._jacobian_w[:, :, 0:6, 0:6]).reshape(self._num_envs, -1, 4, 3, 3)
        self.log_one_time = True
        self._base_rot_mat_w2b[:, :, :] = 0.
        self._base_rot_mat_w2b[:, 2, 0] = 1.
        self._base_rot_mat_w2b[:, 1, 1] = 1.
        self._base_rot_mat_w2b[:, 0, 2] = -1.

    def reset(self, env_ids: torch.Tensor, horizontal_frame=True):
        super().reset(env_ids)
        self._world_pos_sim[env_ids] = self._robot._base_pos_sim[env_ids].clone()
        self._world_quat_sim2w[env_ids] = self._robot._base_quat_sim2b[env_ids].clone()
        if horizontal_frame:
            rpy = self.rot_mat_to_rpy_zxy(quat_to_rot_mat(self._world_quat_sim2w[env_ids])[0].cpu().numpy())
            self._base_rot_mat_w2b[env_ids, :, :] = rpy_to_rot_mat(torch.tensor([[0, rpy[1], 0]])).to(dtype=torch.float).to(device=self._base_rot_mat_w2b.device)
        else:
            self._base_rot_mat_w2b[env_ids, :, :] = 0.
            self._base_rot_mat_w2b[env_ids, 2, 0] = 1.
            self._base_rot_mat_w2b[env_ids, 1, 1] = 1.
            self._base_rot_mat_w2b[env_ids, 0, 2] = -1.
        self._world_rot_mat_w2sim[env_ids] = torch.matmul(self._base_rot_mat_w2b[env_ids], quat_to_rot_mat(self._world_quat_sim2w[env_ids]).transpose(-2, -1))
        # self._world_rot_mat_w2sim[env_ids] = quat_to_rot_mat(self._world_quat_sim2w[env_ids]).transpose(-2, -1)
        self._reshaped_tensor[env_ids] = self._robot._jacobian_w[:, :, 0:6, 0:6].reshape(self._num_envs, -1, 4, 3, 3)[env_ids].clone()

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """
        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])
        return np.array([x, y, z])

    def update(self):
        self._world_pos_sim[:] = self._robot._base_pos_sim.clone()
        self._world_quat_sim2w[:] = self._robot._base_quat_sim2b.clone()
        self._world_rot_mat_w2sim[:] = torch.matmul(self._base_rot_mat_w2b, quat_to_rot_mat(self._world_quat_sim2w).transpose(-2, -1))
        # self._world_rot_mat_w2sim[:] = quat_to_rot_mat(self._world_quat_sim2w).transpose(-2, -1)

    def compute_jacobian_w(self):
        # for the dofs from the floating base, we also need to transfer the space(inputs) from the body frame to the world frame:
        # v^{w} = R_w^s * v^{s} = R_w^s * J^{s} * dq^{s} = R_w^s * J^{s} * R_s^w * dq^{w} = R_w^s * J^{s} * R_s^w * R_w^b * dq^{b}
        self._reshaped_tensor[:, :, 0, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 0:3, 0:3]
        self._reshaped_tensor[:, :, 1, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 0:3, 3:6]
        self._reshaped_tensor[:, :, 2, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 3:6, 0:3]
        self._reshaped_tensor[:, :, 3, 0:3, 0:3] = self._robot._jacobian_sim[:, :, 3:6, 3:6]
        temp_jacobian_w = torch.matmul(
                                    torch.matmul(self._world_rot_mat_w2sim.reshape(-1, 1, 1, 3, 3),
                                         torch.matmul(self._reshaped_tensor,self._world_rot_mat_w2sim.reshape(-1, 1, 1, 3, 3).transpose(-2, -1))), 
                                                            self._base_rot_mat_w2b.reshape(-1, 1, 1, 3, 3))
        self._robot._jacobian_w[:, :, 0:3, 0:3] = temp_jacobian_w[:, :, 0, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 0:3, 3:6] = temp_jacobian_w[:, :, 1, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 3:6, 0:3] = temp_jacobian_w[:, :, 2, 0:3, 0:3]
        self._robot._jacobian_w[:, :, 3:6, 3:6] = temp_jacobian_w[:, :, 3, 0:3, 0:3]

        # for the dofs in the joint space, only need to transfer the outpus from the sim frame to the world frame:
        # v^{w} = R_w^s * v^{s} = R_w^s * J^{s} * dq^{j}
        self._robot._jacobian_w[:, :, 0:3, 6:] = torch.matmul(self._world_rot_mat_w2sim.unsqueeze(1), self._robot._jacobian_sim[:, :, 0:3, 6:])
        self._robot._jacobian_w[:, :, 3:6, 6:] = torch.matmul(self._world_rot_mat_w2sim.unsqueeze(1), self._robot._jacobian_sim[:, :, 3:6, 6:])


    def set_foot_global_state(self):
        self._robot._foot_pos_w[:] = torch.matmul(self._world_rot_mat_w2sim, (self._robot._foot_pos_sim - self._world_pos_sim.unsqueeze(-2)).transpose(-2, -1)).transpose(-2, -1)
        self._robot._foot_vel_w[:] = torch.matmul(self._world_rot_mat_w2sim, self._robot._foot_vel_sim.transpose(-2, -1)).transpose(-2, -1)
        self._robot._swing_foot_pos_w[:] = self._robot.foot_pos_w[:, self._robot._swing_foot_idx_of_four]
        self._robot._swing_foot_vel_w[:] = self._robot.foot_vel_w[:, self._robot._swing_foot_idx_of_four]
        self._robot._stance_foot_pos_w[:] = self._robot.foot_pos_w[:, self._robot._stance_foot_idx_of_four]
        self._robot._stance_foot_vel_w[:] = self._robot.foot_vel_w[:, self._robot._stance_foot_idx_of_four]

        if self._robot._use_gripper:
            self._robot._eef_pos_w[:] = torch.matmul(self._world_rot_mat_w2sim, (self._robot._eef_pos_sim - self._world_pos_sim.unsqueeze(-2)).transpose(-2, -1)).transpose(-2, -1)
            self._robot._eef_rot_w[:] = torch.matmul(self._world_rot_mat_w2sim, quat_to_rot_mat(self._robot._eef_quat_sim.reshape(-1, 4)).reshape(self._num_envs, -1, 3, 3))
            self._robot._eef_rpy_w[:] = rot_mat_to_rpy(self._robot._eef_rot_w.reshape(-1, 3, 3)).reshape(self._num_envs, -1, 3)


        # print('-----------------set_foot_global_state-----------------')
        # print('world_pos_sim: \n', self._world_pos_sim)
        # print('base_rot_mat_w2b: \n', self._base_rot_mat_w2b)
        # print('base_rot_mat_sim2b: \n', quat_to_rot_mat(self._robot._base_quat_sim2b))
        # print('world_rot_mat_w2sim: \n', self._world_rot_mat_w2sim)
        # print('foot_pos_w: \n', self._robot._foot_pos_w)
        # print('eef_pos_w: \n', self._robot._eef_pos_w)
        # print('eef_rot_w: \n', self._robot._eef_rot_w)
        # ans = input("Any Key...")
        # if ans in ['q', 'Q']:
        #     exit()



