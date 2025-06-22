import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rpy_to_rot_mat, rot_mat_to_rpy, quat_to_rot_mat, rpy_vel_to_skew_synmetric_mat, skew_symmetric_mat_to_rpy_vel


class SimLocoStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        
        # robot state in reset timing
        self._base_rpy_w2b = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_pos_sim = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_rpy_sim = torch.zeros((self._num_envs, 3), dtype=torch.float, device=self._device, requires_grad=False)
        self._world_rot_mat_w2sim = rpy_to_rot_mat(self._world_rpy_sim).transpose(-2, -1)

        self._reshaped_tensor = torch.zeros_like(self._robot._jacobian_w[:, :, 0:6, 0:6]).reshape(self._num_envs, -1, 4, 3, 3)
        self.log_one_time = True

    def reset(self, env_ids: torch.Tensor):
        self.update(env_ids)
        self._reshaped_tensor[env_ids] = self._robot._jacobian_w[:, :, 0:6, 0:6].reshape(self._num_envs, -1, 4, 3, 3)[env_ids].clone()
        # print('\n---------------state estimator reset------------------')
        # print('reset: _world_pos_sim: \n', self._world_pos_sim[env_ids])
        # print('reset: _world_rot_mat_sim2w: \n', self._world_rot_mat_w2sim[env_ids])

    def update(self, env_ids: torch.Tensor=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        # update the world(zero yaw) frame related to the sim frame
        self._world_pos_sim[env_ids, 0:2] = self._robot._base_pos_sim[env_ids, 0:2]
        base_rpy_sim = rot_mat_to_rpy(quat_to_rot_mat(self._robot._base_quat_sim2b))
        self._world_rpy_sim[env_ids, 2] = base_rpy_sim[env_ids, 2]
        self._world_rot_mat_w2sim[env_ids] = rpy_to_rot_mat(self._world_rpy_sim[env_ids]).transpose(-2, -1)

        # update the base state in the world frame
        self._base_pos_w[env_ids, 2] = self._robot._base_pos_sim[env_ids, 2]
        self._base_rpy_w2b[env_ids, 0:2] = base_rpy_sim[env_ids, 0:2]
        self._base_rot_mat_w2b[env_ids] = rpy_to_rot_mat(self._base_rpy_w2b[env_ids])
        self._base_lin_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim, self._robot._base_lin_vel_sim.unsqueeze(-1)).squeeze(-1)[env_ids]
        self._base_ang_vel_w[env_ids] = torch.matmul(self._world_rot_mat_w2sim, self._robot._base_ang_vel_sim.unsqueeze(-1)).squeeze(-1)[env_ids]

        # if self._robot._log_info_now:
        #     print('\n--------------- state ------------------')
        #     # print('base_pos_sim: \n', self._robot._base_pos_sim)
        #     # print('world_pos_sim: \n', self._world_pos_sim)
        #     # print('base_pos_w: \n', self._base_pos_w)
        #     # print('base_rpy_w2sim: \n', rot_mat_to_rpy(self._world_rot_mat_w2sim))
        #     # print('base_rpy_sim2b: \n', rot_mat_to_rpy(quat_to_rot_mat(self._robot._base_quat_sim2b)))
        #     # print('base_rpy_w2b: \n', rot_mat_to_rpy(self._base_rot_mat_w2b))
        #     # print('torques: \n', self._robot._torques)
        #     print('base_ang_vel_w: \n', self._base_ang_vel_w)
        #     print('base_ang_vel_w_temp: \n', base_ang_vel_w_temp)
        #     print('base_ang_vel_w_temp_direct: \n', torch.matmul(self._world_rot_mat_w2sim, self._robot._base_ang_vel_sim.unsqueeze(-1)).squeeze(-1))
    
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
        # if self.log_one_time and self._robot.finish_reset:
        #     print('\n---------------update foot global state------------------')
        #     print('base_pos_w: \n', self._base_pos_w)
        #     print('foot_pos_w: \n', self._robot._foot_pos_w)
        #     self.log_one_time = False
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





