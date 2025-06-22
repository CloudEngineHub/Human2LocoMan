import torch
from estimator.base_state_estimator import BaseStateEstimator
from utilities.rotation_utils import rot_mat_to_rpy
import time
import numpy as np
from utilities.orientation_utils_numpy import compute_transformation_matrix
# from pytorch3d.transforms import matrix_to_axis_angle

class FootStateEstimator(BaseStateEstimator):
    def __init__(self, robot):
        super().__init__(robot)
        self._origin_foot_pos_b = torch.zeros_like(self._robot.foot_pos_b, device=self._device, requires_grad=False)
        self._last_T = torch.eye(4, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._current_T = torch.eye(4, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._delta_R = torch.eye(3, device=self._device, requires_grad=False).repeat(self._num_envs, 1, 1)
        self._last_timestamp = time.time()

    def reset(self, env_ids: torch.Tensor):
        self._last_timestamp = time.time()
        super().reset(env_ids)
        self._origin_foot_pos_b[env_ids] = self._robot.foot_pos_b[env_ids].clone()
        self._last_T[env_ids] = torch.eye(4, device=self._device, requires_grad=False)
        self._current_T[env_ids] = torch.eye(4, device=self._device, requires_grad=False)
        self._delta_R[env_ids] = torch.eye(3, device=self._device, requires_grad=False)

    def update(self):
        self._dt = time.time() - self._last_timestamp
        self._last_timestamp = time.time()
        self._last_T = self._current_T.clone()
        # self._update_current_T_torch()
        self._update_current_T_numpy()

        self._base_pos_w[:] = self._current_T[:, :3, 3]
        last_base_lin_vel_w = self._base_lin_vel_w.clone()
        self._base_lin_vel_w[:] = (self._current_T[:, :3, 3] - self._last_T[:, :3, 3]) / self._dt
        self._base_lin_acc_w[:] = (self._base_lin_vel_w - last_base_lin_vel_w) / self._dt

        self._base_rot_mat_w2b[:] = self._current_T[:, :3, :3]
        last_base_ang_vel_w = self._base_ang_vel_w.clone()

        self._delta_R[:] = torch.matmul(self._current_T[:, :3, :3], self._last_T[:, :3, :3].transpose(-2, -1))
        delta_R_w = torch.matmul(torch.matmul(self._last_T[:, :3, :3], self._delta_R), self._last_T[:, :3, :3].transpose(-2, -1))
        self._base_ang_vel_w[:] = rot_mat_to_rpy(delta_R_w) / self._dt  # this is right !!!
        # self._base_ang_vel_w[:] = matrix_to_axis_angle(self._delta_R) / self._dt  # this is right !!!
        # self._base_ang_vel_w[:] = rot_mat_to_rpy(self._delta_R) / self._dt  # this is wrong !!!

        self._base_ang_acc_w[:] = (self._base_ang_vel_w - last_base_ang_vel_w) / self._dt

    def _update_current_T_torch(self):
        expanded_contact = self._robot._desired_foot_contact.unsqueeze(-1).expand(-1, -1, 3)
        centroid_A = torch.mean(self._robot.foot_pos_b[expanded_contact].view(self._num_envs, -1, 3), dim=1, keepdim=True)
        centroid_B = torch.mean(self._origin_foot_pos_b[expanded_contact].view(self._num_envs, -1, 3), dim=1, keepdim=True)
        centered_A = self._robot.foot_pos_b[expanded_contact].view(self._num_envs, -1, 3) - centroid_A
        centered_B = self._origin_foot_pos_b[expanded_contact].view(self._num_envs, -1, 3) - centroid_B

        H = torch.matmul(centered_A.transpose(-2, -1), centered_B)
        U, _, Vt = torch.linalg.svd(H)
        
        # Adjust Vt based on the determinant to ensure a right-handed coordinate system
        det = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
        if det < 0:
            Vt[..., -1] *= -1  # Negate the last column of Vt

        rotation_matrix = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
        translation_vector = centroid_B - torch.matmul(rotation_matrix, centroid_A.transpose(-2, -1)).transpose(-2, -1)
        self._current_T[:, :3, :3] = rotation_matrix
        self._current_T[:, :3, 3] = translation_vector.squeeze(-2)

    def _update_current_T_numpy(self):
        for i in range(self._num_envs):
            contact_state = self._robot.desired_foot_contact_np[i]
            T_np = compute_transformation_matrix(self._robot.foot_pos_b_np[i, contact_state, ::], self._robot.origin_foot_pos_b_np[i, contact_state, ::])
            self._current_T[i, :3, :3] = torch.tensor(T_np[:3, :3], dtype=torch.float, device=self._device, requires_grad=False)
            self._current_T[i, :3, 3] = torch.tensor(T_np[:3, 3], dtype=torch.float, device=self._device, requires_grad=False)
            # if self._robot._log_info_now:
            #     print('foot_pos_b_np: \n', self._robot.foot_pos_b_np[i, contact_state, ::])
            #     print('origin_foot_pos_b_np: \n', self._robot.origin_foot_pos_b_np[i, contact_state, ::])
            #     print('T_np: \n', T_np)
        




if __name__ == "__main__":
    import numpy as np
    from scipy.linalg import svd
    # from utilities.orientation_utils_numpy import rot_mat_to_rpy, rpy_to_rot_mat, compute_transformation_matrix
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200, nanstr='nan', precision=4, suppress=True, threshold=1000, formatter=None)

    def compute_transformation_matrix_numpy(points_A, points_B):
        # Ensure the points are numpy arrays
        points_A = np.array(points_A)
        points_B = np.array(points_B)
        
        # Step 1: Compute centroids
        centroid_A = np.mean(points_A, axis=0)
        centroid_B = np.mean(points_B, axis=0)
        
        # Step 2: Subtract centroids to center the points at the origin
        centered_A = points_A - centroid_A
        centered_B = points_B - centroid_B
        
        # Step 3: Compute the cross-covariance matrix
        H = centered_A.T @ centered_B
        
        # Step 4: Compute the rotation matrix using SVD
        U, _, Vt = svd(H)
        rotation_matrix = Vt.T @ U.T
        
        # Ensure a right-handed coordinate system
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = Vt.T @ U.T
        
        # Step 5: Compute the translation vector
        translation_vector = centroid_B - rotation_matrix @ centroid_A
        
        # Step 6: Assemble the transformation matrix
        T_AB = np.eye(4)
        T_AB[:3, :3] = rotation_matrix
        T_AB[:3, 3] = translation_vector.squeeze()
        
        return T_AB
    
    def compute_transformation_matrix_torch(points_A, points_B):
        centroid_A = torch.mean(points_A, dim=1, keepdim=True)
        centroid_B = torch.mean(points_B, dim=1, keepdim=True)
        centered_A = points_A - centroid_A
        centered_B = points_B - centroid_B

        H = torch.matmul(centered_A.transpose(-2, -1), centered_B)
        U, _, Vt = torch.linalg.svd(H)
        
        # Adjust Vt based on the determinant to ensure a right-handed coordinate system
        det = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
        if det < 0:
            Vt[..., -1] *= -1  # Negate the last column of Vt

        rotation_matrix = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

        translation_vector = centroid_B - torch.matmul(rotation_matrix, centroid_A.transpose(-2, -1)).transpose(-2, -1)

        T_AB = torch.eye(4).repeat(points_A.shape[0], 1, 1)
        T_AB[:, :3, :3] = rotation_matrix
        T_AB[:, :3, 3] = translation_vector.squeeze(-2)
        return T_AB

    # from utilities.orientation_utils_numpy import rot_mat_to_rpy
    # Origin_foot_pos_b = np.array([[0.194, -0.138, -0.255],
    #                     [0.193,  0.137, -0.255],
    #                     [-0.181, -0.136, -0.255],
    #                     [-0.183,  0.136, -0.255]])
    # Foot_pos_b = np.array([[ 0.195, -0.149, -0.211],
    #                     [ 0.194,  0.142, -0.202],
    #                     [-0.185, -0.133, -0.231],
    #                     [-0.178,  0.13,  -0.223]])
    # last_T = np.array([[ 0.999, -0.008,  0.052,  0.011],
    #                     [ 0.007,  1.,     0.03,   0.009],
    #                     [-0.053, -0.03,   0.998, -0.037],
    #                     [ 0.,     0.,     0.,     1.   ]])
    
    Origin_foot_pos_b = np.array([[0.1935, -0.138, -0.2548],
                        [0.1959,  0.1387, -0.2538],
                        [-0.1833, -0.1353, -0.2558],
                        [-0.1828,  0.1349, -0.2557]])
    Foot_pos_b = np.array([[ 0.2434, -0.1205, -0.2856],
                        [ 0.2476,  0.1614, -0.2826],
                        [-0.1331, -0.1221, -0.2833],
                        [-0.1343,  0.1586,  -0.2863]])


    contact_state = np.ones(4, dtype=bool)
    contact_state[1] = False
    
    current_T_numpy = compute_transformation_matrix_numpy(Foot_pos_b[contact_state, ::], Origin_foot_pos_b[contact_state, ::])

    Origin_foot_pos_b_torch = torch.tensor(Origin_foot_pos_b[contact_state, ::], dtype=torch.float).view(1, int(sum(contact_state)), 3)
    Foot_pos_b_torch = torch.tensor(Foot_pos_b[contact_state, ::], dtype=torch.float).view(1, int(sum(contact_state)), 3)
    current_T_torch = compute_transformation_matrix_torch(Foot_pos_b_torch, Origin_foot_pos_b_torch)

    
    print('current_T_numpy: \n', current_T_numpy)
    print('current_T_torch: \n', current_T_torch.cpu().numpy())

    # delta_p = current_T[:3, 3] - last_T[:3, 3]
    # delta_R = current_T[:3, :3].dot(last_T[:3, :3].T)
    # delta_t = 0.0050623416900634766

    # lin_vel = delta_p / delta_t
    # ang_vel = rot_mat_to_rpy(delta_R) / delta_t

    # print('current_T: \n', current_T)
    # print('delta_p: \n', delta_p)
    # print('delta_R: \n', delta_R)
    # print('lin_vel: \n', lin_vel)
    # print('ang_vel: \n', np.rad2deg(ang_vel))

