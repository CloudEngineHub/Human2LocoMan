from commander.base_commander import BaseCommander
import numpy as np
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy


class StanceCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)

    def reset(self, reset_estimator=True):
        self._is_updating_command = False
        self._first_receive_from_human = True
        self._body_pva_cmd[:] = self._desired_body_pva
        self._body_scaled_pva_cmd[:] = self._desired_body_pva
        self._last_executed_body_pva_cmd[:] = self._desired_body_pva
        self._body_pv_buffer[:] = 0

        if self._reset_manipulator_when_switch:
            self._eef_pva_cmd[:] = self._desired_eef_pva
            self._gripper_angles_cmd[:] = self._desired_gripper_angles
        else:
            manipulator_joint_pos = self._robot.joint_pos[self._env_ids, self._robot._gripper_joint_idx].cpu().numpy()
            self._eef_pva_cmd[0, 0:3] = manipulator_joint_pos[0:3]
            self._eef_pva_cmd[1, 0:3] = manipulator_joint_pos[3:6]
            self._gripper_angles_cmd[:] = self._robot.gripper_angles_np[self._env_ids]
        self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd[:]
        self._eef_joint_pos_buffer[:] = 0
        self._gripper_angles_buffer[:] = 0
        self._command_executed = True
        self._robot._update_state(reset_estimator=reset_estimator)

    def _update_joystick_command_callback(self, command_msg):
        if self._is_used:
            self._is_updating_command = True
            command_np = np.array(command_msg.data)
            self._body_pv_buffer[0:6] = command_np[0:6] * self._body_pose_scale
            # print('body_pv_buffer:', self._body_pv_buffer[0:6])
            self._eef_joint_pos_buffer[:] = command_np[6:12].reshape(2, 3) * self._eef_joint_pos_scale
            self._gripper_angles_buffer[:] = command_np[12:14] * self._gripper_angle_scale
            self._is_updating_command = False

    def _update_human_command_callback(self, command_msg):
        # if self._is_used:
        #     self._is_updating_command = True
        #     if self._first_receive_from_human:
        #         self._init_body_xyz = self._robot.base_pos_w_np[self._env_ids].copy()
        #         self._init_body_rpy = self._robot.base_rpy_w2b_np[self._env_ids].copy()
        #         self._curr_body_xyz = self._init_body_xyz.copy()
        #         self._curr_body_rpy = self._init_body_rpy.copy()
        #         self._first_receive_from_human = False
        #     else:
        #         command_np = np.array(command_msg.data)

        #         self._curr_body_xyz[:] = self._robot.base_pos_w_np[self._env_ids].copy()
        #         self._curr_body_rpy[:] = self._robot.base_rpy_w2b_np[self._env_ids].copy()
        #         curr_body_delta_xyz = self._curr_body_xyz - self._init_body_xyz
        #         curr_body_delta_rpy = rot_mat_to_rpy(rpy_to_rot_mat(self._curr_body_rpy) @ rpy_to_rot_mat(self._init_body_rpy).T)

        #         self._body_pv_buffer[0:3][(command_np[0:3] - curr_body_delta_xyz)>self.body_xyz_threshold] = 1.0
        #         self._body_pv_buffer[0:3][(command_np[0:3] - curr_body_delta_xyz)<-self.body_xyz_threshold] = -1.0
        #         self._body_pv_buffer[3:6][(command_np[3:6] - curr_body_delta_rpy)>self.body_rpy_threshold] = 1.0
        #         self._body_pv_buffer[3:6][(command_np[3:6] - curr_body_delta_rpy)<-self.body_rpy_threshold] = -1.0
        #     self._is_updating_command = False
        pass

    def compute_command_for_wbc(self):
        super().compute_command_for_wbc()
        if not self._is_updating_command:
            if self._body_p_command_delta:
                self._body_pva_cmd[0:6] += self._body_pv_buffer[0:6]
                self._body_pva_cmd[0:6] = np.clip(self._body_pva_cmd[0:6], -self._body_pv_limit[0:6], self._body_pv_limit[0:6])
                # self._body_pva_cmd[0:3] += self._body_pv_buffer[0:3] * self._body_pose_scale
                # self._body_pva_cmd[3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._body_pva_cmd[3:6]) @ rpy_to_rot_mat(self._body_pv_buffer[3:6] * self._body_pose_scale[3:6]))
                # self._body_pva_cmd[0:6] = np.clip(self._body_pva_cmd[0:6], -self._body_pv_limit[0:6], self._body_pv_limit[0:6])
            else:
                self._body_current_pv[:] = self.get_body_current_pv()
                self._body_pva_cmd[0:6] = np.clip(self._body_pv_buffer[0:6] * self._body_pv_limit[0:6], self._body_current_pv[0:6]-10*self._body_pose_scale, self._body_current_pv[0:6]+10*self._body_pose_scale)
                self._body_pva_cmd[0:6][self._body_pv_buffer[0:6]==0] = 0.
                if np.sum(np.abs(self._body_pv_buffer[0:6])) > 0:
                    print('_body_pv_buffer', self._body_pv_buffer[0:6])
                    print('current_scaled', self._body_pv_buffer[0:6] * self._body_pv_limit[0:6])
                    print('current_pv', self._body_current_pv[0:6])
                    print('_body_pose_scale - ', self._body_current_pv[0:6]-self._body_pose_scale)
                    print('_body_pose_scale + ', self._body_current_pv[0:6]+self._body_pose_scale)
                    print('commanded', self._body_pva_cmd[0:6])

            self._eef_pva_cmd[:, 0:3] += self._eef_joint_pos_buffer
            self._gripper_angles_cmd[:] += self._gripper_angles_buffer
            self._gripper_angles_cmd[:] = np.clip(self._gripper_angles_cmd, self._gripper_angle_range[0], self._gripper_angle_range[1])
            
            self._body_pv_buffer[:] = 0
            self._eef_joint_pos_buffer[:] = 0
            self._gripper_angles_buffer[:] = 0

            # if self._robot._log_info_now:
            #     print('------------------')
            #     print('eef_pv_cmd:', self._eef_pva_cmd)


        return {"action_mode": self._action_mode,
                "contact_state": self._contact_state,
                "body_pva": self._body_pva_cmd,
                "footeef_pva": self._desired_footeef_pva_w,
                "eef_pva": self._eef_pva_cmd,
                "gripper_angles": self._gripper_angles_cmd,
                }


    def check_finished(self):
        return True













