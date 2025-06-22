import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rpy_to_rot_mat
import numpy as np

class BiManualStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        # the position is aligned with the body frame, the orientation is aligned with the default world frame
        self._base_rot_mat_w2b[:, :, :] = 0.
        self._base_rot_mat_w2b[:, 2, 0] = 1.
        self._base_rot_mat_w2b[:, 1, 1] = 1.
        self._base_rot_mat_w2b[:, 0, 2] = -1.

    def reset(self, env_ids: torch.Tensor, horizontal_frame=True):
        super().reset(env_ids)
        if horizontal_frame:
            # self._base_rot_mat_w2b[env_ids, :, :] = 0.
            # self._base_rot_mat_w2b[env_ids, 2, 0] = 1.
            # self._base_rot_mat_w2b[env_ids, 1, 1] = 1.
            # self._base_rot_mat_w2b[env_ids, 0, 2] = -1.
            
            # rpy = self.rot_mat_to_rpy_zxy(rpy_to_rot_mat(torch.tensor(self._robot._raw_state.imu.rpy).unsqueeze(0)).squeeze().cpu().numpy())
            # base_rot_mat_w2b1 = rpy_to_rot_mat(torch.tensor([[0, rpy[1], 0]])).to(dtype=torch.float).to(device=self._base_rot_mat_w2b.device)
            self._base_rot_mat_w2b[env_ids, :, :] = rpy_to_rot_mat(torch.tensor([[0, -1.525, 0]])).to(dtype=torch.float).to(device=self._base_rot_mat_w2b.device)

            # try:
            #     rpy = self.rot_mat_to_rpy_zxy(rpy_to_rot_mat(torch.tensor(self._robot._raw_state.imu.rpy).unsqueeze(0)).squeeze().cpu().numpy())
            #     self._base_rot_mat_w2b[env_ids, :, :] = rpy_to_rot_mat(torch.tensor([[0, rpy[1], 0]])).to(dtype=torch.float).to(device=self._base_rot_mat_w2b.device)
            # except Exception as e:
            #     print(f"An error occurred: {e}")
            #     self._base_rot_mat_w2b[env_ids, :, :] = 0.
            #     self._base_rot_mat_w2b[env_ids, 2, 0] = 1.
            #     self._base_rot_mat_w2b[env_ids, 1, 1] = 1.
            #     self._base_rot_mat_w2b[env_ids, 0, 2] = -1.
        else:
            self._base_rot_mat_w2b[env_ids, :, :] = 0.
            self._base_rot_mat_w2b[env_ids, 2, 0] = 1.
            self._base_rot_mat_w2b[env_ids, 1, 1] = 1.
            self._base_rot_mat_w2b[env_ids, 0, 2] = -1.

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
        # R_globalw2b = rpy_to_rot_mat(np.array(self._robot._raw_state.imu.rpy))
        # print('roll', self._robot._raw_state.imu.rpy[0])
        # print('pitch', self._robot._raw_state.imu.rpy[1])
        # print('yaw', self._robot._raw_state.imu.rpy[2])
        # print('R_globalw2b', R_globalw2b)
        # try:
        #     rpy = self.rot_mat_to_rpy_zxy(rpy_to_rot_mat(torch.tensor(self._robot._raw_state.imu.rpy).unsqueeze(0)).squeeze().cpu().numpy())
        #     print('rpy', rpy)
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        pass