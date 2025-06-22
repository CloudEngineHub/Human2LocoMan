from commander.base_commander import BaseCommander
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
import time
from scipy.spatial.transform import Rotation as R
from utilities.rotation_interpolation import interpolate_rpy

class BiManipulationCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)

        self._action_mode = 5
        self._manipulate_eefs_idx = [0, 1]
        for i in self._manipulate_eefs_idx:
            self._desired_footeef_pva_w[0, 3*i:3*(i+1)] = self._robot.foot_pos_w_np[self._env_ids, i]
        self._footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._last_executed_footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._footeef_p_buffer = np.zeros((2, 3))
        self._footeef_p_scale = np.ones_like(self._footeef_p_buffer) * self._cfg.commander.footeef_p_scale
        self.reset_pose_command = np.zeros(20)
        self.reset_pose_command_msg = Float32MultiArray()
        
        self._desired_rear_legs_pos = np.array([0, 2.35, -2.17, 0, 2.35, -2.17])
        self._desired_rear_legs_vel = np.zeros(6)
        self._desired_rear_legs_torque = np.zeros(6)

        self.human_command_body_rpy_range = np.array([self._cfg.teleoperation.human_teleoperator.body_r_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_p_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_y_range,])
        self.auto_policy_command_body_rpy_range = np.array([self._cfg.teleoperation.human_teleoperator.body_r_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_p_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_y_range,])
        self.robot_reset_pose_publisher = rospy.Publisher(self._cfg.commander.robot_reset_pose_topic, Float32MultiArray, queue_size=1)

    def reset(self, reset_estimator=True):
        self._robot._update_state(reset_estimator=reset_estimator)

        self._contact_state[:] = True
        self._contact_state[self._manipulate_eefs_idx] = False

        # self._body_pva_cmd: [18]
        # [6d pose * 3 (pva)]
        self._body_pva_cmd[:] = self._desired_body_pva
        # self._body_pva_cmd[:3] = self._robot.base_pos_w_np[self._env_ids]
        self._body_pva_cmd[3:6] = self._robot.base_rpy_w2b_np[self._env_ids]
        # eef 3d position pose
        # self._footeef_pva_cmd: [3, 12]
        # [3 (pva), 3d position * 4 legs ]
        self._footeef_pva_cmd[:] = self._desired_footeef_pva_w
        for i in self._manipulate_eefs_idx:
            self._footeef_pva_cmd[0, 3 * i:3 * (i + 1)] = self._robot.eef_pos_w_np[self._env_ids, i] if self._robot._use_gripper else self._robot.foot_pos_w_np[self._env_ids, i]
        self._last_executed_footeef_pva_cmd[:] = self._footeef_pva_cmd
        self._footeef_p_buffer[:] = 0
        # eef 3d orientation pose
        # self._eef_pva_cmd: [2, 9]
        # [2 eef, 3d orientation * 3 (pva)]
        self._eef_pva_cmd[:] = self._desired_eef_pva
        if self._robot._use_gripper:
            self._eef_pva_cmd[:, 0:3] = self._robot.eef_rpy_w_np[self._env_ids, :]
        self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd
        self._eef_rpy_buffer[:] = 0
        # gripper angless
        # self._gripper_angles_cmd: [2]
        self._gripper_angles_cmd[:] = self._desired_gripper_angles
        self._gripper_angles_buffer[:] = 0

        # body xyzrpy at reset
        self.reset_pose_command[:6] = self._body_pva_cmd[:6]
        # right eef xyzrpy at reset
        self.reset_pose_command[6:9] = self._footeef_pva_cmd[0, :3]
        self.reset_pose_command[9:12] = self._eef_pva_cmd[0, :3]
        # left eef xyzrpy at reset
        self.reset_pose_command[12:15] = self._footeef_pva_cmd[0, 3:6]
        self.reset_pose_command[15:18] = self._eef_pva_cmd[1, :3]
        # gripper angle at reset
        self.reset_pose_command[18:20] = self._gripper_angles_cmd[:]
        # send out the reset pose
        self.reset_pose_command_msg.data = self.reset_pose_command.tolist()
        self.robot_reset_pose_publisher.publish(self.reset_pose_command_msg)

        self._is_updating_command = False
        self._is_computing_command = False

        # for command from human motions
        self._reset_from_human = True
        self._cmd_from_human = None
        self._first_receive_from_human = True
        self._human_time_at_first_stamp = time.time()
        self._human_time_from_first_stamp = 0.
        # for command from autonomous policy
        self._reset_from_auto_policy = True
        self._cmd_from_auto_policy = None
        self._first_receive_from_auto_policy = True
        self._auto_policy_time_at_first_stamp = time.time()
        self._auto_policy_time_from_first_stamp = 0
        # time stamp for human teloep. and auto. policy
        self._controller_time_at_first_stamp = time.time()
        self._controller_time_from_first_stamp = 0.

    def _update_joystick_command_callback(self, command_msg):
        if self._is_used:
            self._is_updating_command = True
            self._cmd_from_human = False
            self._cmd_from_auto_policy = False
            self._time_for_tracking_human = 0.0
            self._time_for_tracking_auto_policy = 0.0

            command_np = np.array(command_msg.data)
            self._footeef_p_buffer[0, :] = command_np[0:3] * self._footeef_p_scale[0, :]
            self._footeef_p_buffer[1, :] = command_np[6:9] * self._footeef_p_scale[1, :]
            self._eef_rpy_buffer[0, :] = command_np[3:6] * self._eef_rpy_scale[0, :]
            self._eef_rpy_buffer[1, :] = command_np[9:12] * self._eef_rpy_scale[1, :]
            self._gripper_angles_buffer[:] = command_np[12:14] * self._gripper_angle_scale
            self._is_updating_command = False

    def reset_last_command(self, command):
        self._last_eefs_xyz_cmd = command['footeef_pva'][0, 0:6].copy()
        self._last_eefs_rpy_cmd = command['eef_pva'][:, 0:3].copy()
        self._last_gripper_angles_cmd = command['gripper_angles'].copy()

    def _update_human_command_callback(self, command_msg):
        if self._is_used:
            self._is_updating_command = True
            self._cmd_from_human = True
            self._cmd_from_auto_policy = False
            # receiving the first command from the human motion after resetting the commander
            if self._reset_from_human:
                self._human_time_at_first_stamp = time.time()
                self._human_time_from_first_stamp = 0.
                self._controller_time_at_first_stamp = time.time()
                self._controller_time_from_first_stamp = 0.

                self._init_eefs_xyz = self._footeef_pva_cmd[0, 0:6].copy()
                self._init_eefs_rpy = self._eef_pva_cmd[:, 0:3].copy()
                self._init_gripper_angles = self._gripper_angles_cmd.copy()

                self._last_eefs_xyz_cmd = self._footeef_pva_cmd[0, 0:6].copy()
                self._last_eefs_rpy_cmd = self._eef_pva_cmd[:, 0:3].copy()
                self._last_gripper_angles_cmd = self._gripper_angles_cmd.copy()

                self._target_eefs_xyz = self._footeef_pva_cmd[0, 0:6].copy()
                self._target_eefs_rpy = self._eef_pva_cmd[:, 0:3].copy()
                # self._target_eefs_rotation_axis = np.ones_like(self._init_eefs_rpy)
                # self._target_eefs_ratation_angle = np.zeros(2)
                self._target_gripper_angles = self._gripper_angles_cmd.copy()

                self._reset_from_human = False
            else:
                self._human_time_from_first_stamp = time.time() - self._human_time_at_first_stamp
                if self._first_receive_from_human:
                    self._controller_time_at_first_stamp = time.time()
                    self._first_receive_from_human = False
                else:
                    self._controller_time_from_first_stamp = time.time() - self._controller_time_at_first_stamp
                # the time interval between two received commands from human teleop (given the frequency is stable)
                self._time_for_tracking_human = self._human_time_from_first_stamp - self._controller_time_from_first_stamp

                command_np = np.array(command_msg.data)

                if self._command_executed:
                    self._last_eefs_xyz_cmd = self._footeef_pva_cmd[0, 0:6].copy()
                    self._last_eefs_rpy_cmd = self._eef_pva_cmd[:, 0:3].copy()
                    self._last_gripper_angles_cmd = self._gripper_angles_cmd.copy()

                for manipulate_eef_idx in self._manipulate_eefs_idx:
                    self._target_eefs_xyz[3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] = command_np[6+6*manipulate_eef_idx:9+6*manipulate_eef_idx] * self.eef_xyz_scale + self._init_eefs_xyz[3*manipulate_eef_idx:3*(manipulate_eef_idx+1)]
                    self._target_eefs_rpy[manipulate_eef_idx] = rot_mat_to_rpy(rpy_to_rot_mat(self._init_eefs_rpy[manipulate_eef_idx]) @ rpy_to_rot_mat(command_np[9+6*manipulate_eef_idx:12+6*manipulate_eef_idx] * self.eef_rpy_scale))
                    self._target_gripper_angles[manipulate_eef_idx] = command_np[18+manipulate_eef_idx] * self.gripper_angle_scale + self._init_gripper_angles[manipulate_eef_idx]

            self._is_updating_command = False

    def _update_auto_policy_command_callback(self, command_msg):
        if self._is_used:
            self._is_updating_command = True
            self._cmd_from_auto_policy = True
            self._cmd_from_human = False
            # receiving the first command from the trained policy after resetting the commander
            if self._reset_from_auto_policy:
                self._auto_policy_time_at_first_stamp = time.time()
                self._auto_policy_time_from_first_stamp = 0.
                self._controller_time_at_first_stamp = time.time()
                self._controller_time_from_first_stamp = 0.

                self._init_eefs_xyz = self._footeef_pva_cmd[0, 0:6].copy()
                self._init_eefs_rpy = self._eef_pva_cmd[:, 0:3].copy()
                self._init_gripper_angles = self._gripper_angles_cmd.copy()

                self._last_eefs_xyz_cmd = self._footeef_pva_cmd[0, 0:6].copy()
                self._last_eefs_rpy_cmd = self._eef_pva_cmd[:, 0:3].copy()
                self._last_gripper_angles_cmd = self._gripper_angles_cmd.copy()

                self._target_eefs_xyz = self._footeef_pva_cmd[0, 0:6].copy()
                self._target_eefs_rpy = self._eef_pva_cmd[:, 0:3].copy()
                # self._target_eefs_rotation_axis = np.ones_like(self._init_eefs_rpy)
                # self._target_eefs_ratation_angle = np.zeros(2)
                self._target_gripper_angles = self._gripper_angles_cmd.copy()

                self._reset_from_auto_policy = False
            else:
                self._auto_policy_time_from_first_stamp = time.time() - self._auto_policy_time_at_first_stamp
                if self._first_receive_from_auto_policy:
                    self._controller_time_at_first_stamp = time.time()
                    self._first_receive_from_auto_policy = False
                else:
                    self._controller_time_from_first_stamp = time.time() - self._controller_time_at_first_stamp
                # the time interval between two received commands from autonomous policy (given the frequency is stable)
                # this part needs polishing
                self._time_for_tracking_auto_policy = self._auto_policy_time_from_first_stamp - self._controller_time_from_first_stamp

                command_np = np.array(command_msg.data)

                if self._command_executed:
                    self._last_eefs_xyz_cmd = self._footeef_pva_cmd[0, 0:6].copy()
                    self._last_eefs_rpy_cmd = self._eef_pva_cmd[:, 0:3].copy()
                    self._last_gripper_angles_cmd = self._gripper_angles_cmd.copy()

                for manipulate_eef_idx in self._manipulate_eefs_idx:
                    self._target_eefs_xyz[3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] = command_np[6+6*manipulate_eef_idx:9+6*manipulate_eef_idx]
                    self._target_eefs_rpy[manipulate_eef_idx] = command_np[9+6*manipulate_eef_idx:12+6*manipulate_eef_idx]
                    self._target_gripper_angles[manipulate_eef_idx] = command_np[18+manipulate_eef_idx]

            self._is_updating_command = False

    def compute_command_for_wbc(self):
        if not self._is_updating_command:
            self._is_computing_command = True
            if not (self._cmd_from_human or self._cmd_from_auto_policy):
                self._body_pva_cmd[0:3] += self._body_pv_buffer[0:3]
                self._body_pva_cmd[3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._body_pv_buffer[3:6]) @ rpy_to_rot_mat(self._body_pva_cmd[3:6]))
                for manipulate_eef_idx in self._manipulate_eefs_idx:
                    self._footeef_pva_cmd[0, 3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] += self._footeef_p_buffer[manipulate_eef_idx]
                    if self._eef_task_space_world:
                        self._eef_pva_cmd[manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._eef_rpy_buffer[manipulate_eef_idx]) @ rpy_to_rot_mat(self._eef_pva_cmd[manipulate_eef_idx, 0:3]))
                    else:
                        self._eef_pva_cmd[manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._eef_pva_cmd[manipulate_eef_idx, 0:3]) @ rpy_to_rot_mat(self._eef_rpy_buffer[manipulate_eef_idx]))
                self._gripper_angles_cmd[:] += self._gripper_angles_buffer
                self._gripper_angles_cmd[:] = np.clip(self._gripper_angles_cmd, self._gripper_angle_range[0], self._gripper_angle_range[1])

                self._body_pv_buffer[:] = 0
                self._footeef_p_buffer[:] = 0
                self._eef_rpy_buffer[:] = 0
                self._gripper_angles_buffer[:] = 0
            elif (self._cmd_from_human and not self._first_receive_from_human) or (self._cmd_from_auto_policy and not self._first_receive_from_auto_policy):
                if self._cmd_from_human:
                    tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_human))
                else:
                    tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_auto_policy))

                for manipulate_eef_idx in self._manipulate_eefs_idx:
                    self._footeef_pva_cmd[0, 3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] += np.clip(self._last_eefs_xyz_cmd[3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] * (1 - tracking_ratio) + self._target_eefs_xyz[3*manipulate_eef_idx:3*(manipulate_eef_idx+1)] * tracking_ratio - self._footeef_pva_cmd[0, 3*manipulate_eef_idx:3*(manipulate_eef_idx+1)], 
                                                                                                                        -self.eef_xyz_max_step, 
                                                                                                                        self.eef_xyz_max_step)
                    # is eef_rpy_max_step necessary here?
                    self._eef_pva_cmd[manipulate_eef_idx, 0:3] = interpolate_rpy(self._last_eefs_rpy_cmd[manipulate_eef_idx], self._target_eefs_rpy[manipulate_eef_idx], tracking_ratio)
                    # # using scipy
                    # rot_interpolated = R.from_rotvec(self._target_eef_rotation_angle*tracking_ratio*self._target_eef_rotation_axis).as_matrix()
                    # if self._eef_task_space_world:
                    #     self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rot_interpolated @ rpy_to_rot_mat(self._last_eef_rpy_cmd))

                    self._gripper_angles_cmd[manipulate_eef_idx] += max(-self.gripper_angle_max_step, min(self._last_gripper_angles_cmd[manipulate_eef_idx] * (1 - tracking_ratio) + self._target_gripper_angles[manipulate_eef_idx] * tracking_ratio - self._gripper_angles_cmd[manipulate_eef_idx], self.gripper_angle_max_step))

            self._is_computing_command = False

        return {"action_mode": self._action_mode,
                "contact_state": self._contact_state,
                "body_pva": self._body_pva_cmd,
                "footeef_pva": self._footeef_pva_cmd,
                "eef_pva": self._eef_pva_cmd,
                "gripper_angles": self._gripper_angles_cmd,
                "rear_legs_command": {"pos": self._desired_rear_legs_pos, "vel": self._desired_rear_legs_vel, "torque": self._desired_rear_legs_torque},
                }

    def compute_motor_command(self):
        freq = 1
        r = 0.04
        theta = 2 * np.pi * freq * (self._robot.time_since_reset_scalar) + np.pi
        fr = self.compute_joint_pos_from_foot_pos([0. + r * np.cos(theta), -0.08 + r * np.sin(theta), -0.2], l_hip_sign=-1)
        fl = self.compute_joint_pos_from_foot_pos([0. + r * np.cos(-theta), 0.08 + r * np.sin(-theta), -0.2], l_hip_sign=1)
        rr = [0, 2.35, -2.17]
        rl = [0, 2.35, -2.17]
        motor_pos = np.array(fr + fl + rr + rl)
        motor_vel = np.zeros(12)
        motor_torque = np.zeros(12)
        return motor_pos, motor_vel, motor_torque

    def prepare_to_stand(self):
        self._going_to_stand = True

    def compute_joint_pos_from_foot_pos(self, foot_position, l_hip_sign=1):
        l_up = self._robot._l_thi
        l_low = self._robot._l_cal
        l_hip = self._robot._l_hip * l_hip_sign
        x, y, z = foot_position[0], foot_position[1], foot_position[2]
        theta_knee = -np.arccos(
        np.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                (2 * l_low * l_up), -1, 1))
        l = np.sqrt(
        np.maximum(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee),
                    1e-7))
        theta_hip = np.arcsin(np.clip(-x / l, -1, 1)) - theta_knee / 2
        c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
        s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = np.arctan2(s1, c1)
        return [theta_ab, theta_hip, theta_knee]




