from commander.base_commander import BaseCommander
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
from fsm.finite_state_machine import Manipulate_Mode
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
import time
from utilities.rotation_interpolation import interpolate_rpy

class EEFManipulateCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)

        self._action_mode = 2
        self._manipulate_eef_idx = 0
        self._footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._last_executed_footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._footeef_p_buffer = np.zeros(3)
        self._footeef_p_scale = np.ones(3) * self._cfg.commander.footeef_p_scale
        self._manipute_eef_rpy_buffer = np.zeros(3)
        self._manipute_eef_rpy_scale = np.ones(3) * self._cfg.commander.eef_rpy_scale
        self.reset_pose_command = np.zeros(20)
        self.reset_pose_command_msg = Float32MultiArray()

        self.sub_time1 = 0
        self.sub_time2 = 0
        self.compute_cmd1 = 0
        self.compute_cmd2 = 0
        self.human_command_time = 0
        self.human_command_body_rpy_range = np.array([self._cfg.teleoperation.human_teleoperator.body_r_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_p_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_y_range,])
        self.auto_policy_command_body_rpy_range = np.array([self._cfg.teleoperation.human_teleoperator.body_r_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_p_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_y_range,])
        self.robot_reset_pose_publisher = rospy.Publisher(self._cfg.commander.robot_reset_pose_topic, Float32MultiArray, queue_size=1)

    def reset(self, reset_estimator=True):
        self._robot._update_state(reset_estimator=reset_estimator)

        self._manipulate_eef_idx = 0 if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_EEF else 1
        self._contact_state[:] = True
        self._contact_state[self._manipulate_eef_idx] = False

        # self._body_pva_cmd: [18]
        # [6d pose * 3 (pva)]
        self._body_pva_cmd = self._desired_body_pva.copy()
        self._body_pva_cmd[:3] = self._robot.base_pos_w_np[self._env_ids]
        self._body_pva_cmd[3:6] = self._robot.base_rpy_w2b_np[self._env_ids]
        self._body_pv_buffer[:] = 0
        # eef 3d position pose
        # self._footeef_pva_cmd: [3, 12]
        # [3 (pva), 3d position * 4 legs ]
        self._footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)] = self._robot.eef_pos_w_np[self._env_ids, self._manipulate_eef_idx]
        self._last_executed_footeef_pva_cmd[:] = self._footeef_pva_cmd
        self._footeef_p_buffer[:] = 0
        # eef 3d orientation pose
        # self._eef_pva_cmd: [2, 9]
        # [2 eef, 3d orientation * 3 (pva)]
        self._eef_pva_cmd = self._desired_eef_pva.copy()
        self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_eef_idx]
        self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd
        self._manipute_eef_rpy_buffer[:] = 0
        # gripper angless
        # self._gripper_angles_cmd: [2]
        self._gripper_angles_cmd = self._desired_gripper_angles.copy()
        self._gripper_angles_cmd[:] = self._robot.gripper_angles_np[self._env_ids]
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
        # joystick_time = time.time() - self.human_command_time
        # self.human_command_time = time.time()
        # print('time', joystick_time)
        # print('freq', 1 / joystick_time)
        if self._is_used:
            self._is_updating_command = True
            self._cmd_from_human = False
            self._cmd_from_auto_policy = False
            self._time_for_tracking_human = 0.0
            self._time_for_tracking_auto_policy = 0.0
            
            command_np = np.array(command_msg.data)
            self._body_pv_buffer[0:6] = command_np[0:6] * self._body_pose_scale
            self._footeef_p_buffer[:] = command_np[6:9] * self._footeef_p_scale
            self._manipute_eef_rpy_buffer[:] = command_np[9:12] * self._manipute_eef_rpy_scale
            self._gripper_angles_buffer[:] = command_np[12:14] * self._gripper_angle_scale
            self._is_updating_command = False

    def reset_last_command(self, command):
        self._last_body_xyz_cmd = command['body_pva'][0:3].copy()
        self._last_body_rpy_cmd = command['body_pva'][3:6].copy()
        self._last_eef_xyz_cmd = command['footeef_pva'][0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
        self._last_eef_rpy_cmd = command['eef_pva'][self._manipulate_eef_idx, 0:3].copy()
        self._last_gripper_angle_cmd = command['gripper_angles'][self._manipulate_eef_idx] 

    def _update_human_command_callback(self, command_msg):
    #     print('sub time1', time.time() - self.sub_time1)
    #     print('sub freq1', 1 / (time.time() - self.sub_time1))
    #     self.sub_time1 = time.time()

        if self._is_used:
            # print('sub time2', time.time() - self.sub_time2)
            # print('sub freq2', 1 / (time.time() - self.sub_time2))
            # self.sub_time2 = time.time()
            # while self._is_computing_command:
            #     pass
            self._is_updating_command = True
            self._cmd_from_human = True
            self._cmd_from_auto_policy = False
            # receiving the first command from the human motion after resetting the commander
            if self._reset_from_human:
                self._human_time_at_first_stamp = time.time()
                self._human_time_from_first_stamp = 0.
                self._controller_time_at_first_stamp = time.time()
                self._controller_time_from_first_stamp = 0.

                self._init_body_xyz = self._body_pva_cmd[0:3].copy()
                self._init_body_rpy = self._body_pva_cmd[3:6].copy()
                self._init_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._init_eef_rpy = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                self._init_gripper_angle = self._gripper_angles_cmd[self._manipulate_eef_idx]

                self._last_body_xyz_cmd = self._body_pva_cmd[0:3].copy()
                self._last_body_rpy_cmd = self._body_pva_cmd[3:6].copy()
                self._last_eef_xyz_cmd = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._last_eef_rpy_cmd = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_eef_idx]

                self._target_body_xyz = self._body_pva_cmd[0:3].copy()
                self._target_body_rpy = self._body_pva_cmd[3:6].copy()
                self._target_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._target_eef_rpy = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                # self._target_eef_rotation_axis = np.ones_like(self._init_eef_rpy)
                # self._target_eef_rotation_angle = 0.
                self._target_gripper_angle = self._gripper_angles_cmd[self._manipulate_eef_idx]

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
                    self._last_body_xyz_cmd[:] = self._body_pva_cmd[0:3].copy()
                    self._last_body_rpy_cmd[:] = self._body_pva_cmd[3:6].copy()
                    self._last_eef_xyz_cmd[:] = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                    self._last_eef_rpy_cmd[:] = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                    self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_eef_idx]
                
                # do not allow torso movements on xyz during manipulation
                command_np[0:3] = 0.
                #  self._target_body_xyz and  self._target_body_rpy are only for the position
                self._target_body_xyz[:] = command_np[0:3] * self.body_xyz_scale + self._init_body_xyz
                # R^{b}_{w} * R^{b'}_{b} = R^{b'}_{w}
                self._target_body_rpy[:] = rot_mat_to_rpy(rpy_to_rot_mat(self._init_body_rpy) @ rpy_to_rot_mat(command_np[3:6] * self.body_rpy_scale))
                self._target_body_rpy = np.clip(self._target_body_rpy, self.human_command_body_rpy_range[:, 0], self.human_command_body_rpy_range[:, 1])
                self._target_eef_xyz[:] = command_np[6+6*self._manipulate_eef_idx:9+6*self._manipulate_eef_idx] * self.eef_xyz_scale + self._init_eef_xyz
                # R^{e}_{w} * R^{e'}_{e} = R^{e'}_{w}
                self._target_eef_rpy[:] = rot_mat_to_rpy(rpy_to_rot_mat(self._init_eef_rpy) @ rpy_to_rot_mat(command_np[9+6*self._manipulate_eef_idx:12+6*self._manipulate_eef_idx] * self.eef_rpy_scale))

                # R^{current}_{init} * R^{init}_{world} * R^{world}_{last} = R^{current}_{last}
                # delta_rotation = rpy_to_rot_mat(command_np[9+6*self._manipulate_eef_idx:12+6*self._manipulate_eef_idx]) @ rpy_to_rot_mat(self._init_eef_rpy) @ (rpy_to_rot_mat(self._last_eef_rpy_cmd).T)
                # r = R.from_matrix(delta_rotation)
                # rotvec = r.as_rotvec()
                # if np.linalg.norm(rotvec) > 1e-6:
                #     self._target_eef_rotation_axis[:] = rotvec / np.linalg.norm(rotvec)
                #     self._target_eef_rotation_angle = np.linalg.norm(rotvec) * self.eef_rpy_scale
                # else:
                #     self._target_eef_rotation_axis[:] = np.ones(3)
                #     self._target_eef_rotation_angle = 0.

                self._target_gripper_angle = command_np[18+self._manipulate_eef_idx] * self._gripper_angle_scale + self._init_gripper_angle
                
            self._is_updating_command = False

            # print('\n-------------------')
            # print("_last_eef_rpy_cmd: ", self._last_eef_rpy_cmd)
            # print("_target_eef_delta_rpy: ", self._target_eef_delta_rpy)
            
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

                self._init_body_xyz = self._body_pva_cmd[0:3].copy()
                self._init_body_rpy = self._body_pva_cmd[3:6].copy()
                self._init_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._init_eef_rpy = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                self._init_gripper_angle = self._gripper_angles_cmd[self._manipulate_eef_idx]

                self._last_body_xyz_cmd = self._body_pva_cmd[0:3].copy()
                self._last_body_rpy_cmd = self._body_pva_cmd[3:6].copy()
                self._last_eef_xyz_cmd = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._last_eef_rpy_cmd = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_eef_idx]

                self._target_body_xyz = self._body_pva_cmd[0:3].copy()
                self._target_body_rpy = self._body_pva_cmd[3:6].copy()
                self._target_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                self._target_eef_rpy = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                # self._target_eef_rotation_axis = np.ones_like(self._init_eef_rpy)
                # self._target_eef_rotation_angle = 0.
                self._target_gripper_angle = self._gripper_angles_cmd[self._manipulate_eef_idx]

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
                    self._last_body_xyz_cmd[:] = self._body_pva_cmd[0:3].copy()
                    self._last_body_rpy_cmd[:] = self._body_pva_cmd[3:6].copy()
                    self._last_eef_xyz_cmd[:] = self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)].copy()
                    self._last_eef_rpy_cmd[:] = self._eef_pva_cmd[self._manipulate_eef_idx, 0:3].copy()
                    self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_eef_idx]
                
                # do not allow torso movements on xyz during manipulation
                #  self._target_body_xyz and  self._target_body_rpy are only for the position
                self._target_body_xyz[:] = self._init_body_xyz
                # R^{b}_{w} * R^{b'}_{b} = R^{b'}_{w}
                self._target_body_rpy[:] = command_np[3:6]
                self._target_body_rpy = np.clip(self._target_body_rpy, self.auto_policy_command_body_rpy_range[:, 0], self.auto_policy_command_body_rpy_range[:, 1])
                # # to align with training for human data, the eef_xyz should start from [0, 0, 0], so will need to consider init_eef_xyz
                # self._target_eef_xyz[:] = command_np[6+6*self._manipulate_eef_idx:9+6*self._manipulate_eef_idx] * self.eef_xyz_scale + self._init_eef_xyz
                self._target_eef_xyz[:] = command_np[6+6*self._manipulate_eef_idx:9+6*self._manipulate_eef_idx]
                # R^{e}_{w} * R^{e'}_{e} = R^{e'}_{w}
                self._target_eef_rpy[:] = command_np[9+6*self._manipulate_eef_idx:12+6*self._manipulate_eef_idx]
                self._target_gripper_angle = command_np[18+self._manipulate_eef_idx]
                
            self._is_updating_command = False
        
    def compute_command_for_wbc(self):
        # super().compute_command_for_wbc()
        # if self._command_executed:
        #     self._last_executed_footeef_pva_cmd[:] = self._footeef_pva_cmd
        # else:
        #     self._footeef_pva_cmd[:] = self._last_executed_footeef_pva_cmd
        
        # print('fps: ', 1/(time.time() - self._begin_time))
        # self._begin_time = time.time()
        # print('compute cmd1 time', time.time() - self.compute_cmd1)
        # print('compute cmd1 freq', 1 / (time.time() - self.compute_cmd1))
        # self.compute_cmd1 = time.time()

        ### in simulation around 70hz~75hz
        if not self._is_updating_command:
            # print('compute cmd2 time', time.time() - self.compute_cmd2)
            # print('compute cmd2 freq', 1 / (time.time() - self.compute_cmd2))
            # self.compute_cmd2 = time.time()
            self._is_computing_command = True
            if not (self._cmd_from_human or self._cmd_from_auto_policy):
                self._body_pva_cmd[0:3] += self._body_pv_buffer[0:3]
                self._body_pva_cmd[3:6] = rot_mat_to_rpy(rpy_to_rot_mat(self._body_pv_buffer[3:6]) @ rpy_to_rot_mat(self._body_pva_cmd[3:6]))
                self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)] += self._footeef_p_buffer
                if self._eef_task_space_world:
                    self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._manipute_eef_rpy_buffer) @ rpy_to_rot_mat(self._eef_pva_cmd[self._manipulate_eef_idx, 0:3]))
                else:
                    self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._eef_pva_cmd[self._manipulate_eef_idx, 0:3]) @ rpy_to_rot_mat(self._manipute_eef_rpy_buffer))
                self._gripper_angles_cmd[:] += self._gripper_angles_buffer
                self._gripper_angles_cmd[:] = np.clip(self._gripper_angles_cmd, self._gripper_angle_range[0], self._gripper_angle_range[1])

                self._body_pv_buffer[:] = 0
                self._footeef_p_buffer[:] = 0
                self._manipute_eef_rpy_buffer[:] = 0
                self._gripper_angles_buffer[:] = 0
            elif (self._cmd_from_human and not self._first_receive_from_human) or (self._cmd_from_auto_policy and not self._first_receive_from_auto_policy):
                if self._cmd_from_human:
                    tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_human))
                else:
                    tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_auto_policy))

                self._body_pva_cmd[0:3] += np.clip(self._last_body_xyz_cmd * (1 - tracking_ratio) + self._target_body_xyz * tracking_ratio - self._body_pva_cmd[0:3], -self.body_xyz_max_step, self.body_xyz_max_step)
                self._body_pva_cmd[3:6] += np.clip(interpolate_rpy(self._last_body_rpy_cmd, self._target_body_rpy, tracking_ratio) - self._body_pva_cmd[3:6], -self.body_rpy_max_step, self.body_rpy_max_step)
                # self._body_pva_cmd[3:6] += np.clip(self._last_body_rpy_cmd * (1 - tracking_ratio) + self._target_body_rpy * tracking_ratio - self._body_pva_cmd[3:6], -self.body_rpy_max_step, self.body_rpy_max_step)
                # do not allow torso movements on xyz during manipulation
                self._body_pva_cmd[0:3] = 0.
                self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)] += np.clip(self._last_eef_xyz_cmd * (1 - tracking_ratio) + self._target_eef_xyz * tracking_ratio - self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)], 
                                                                                                                     -self.eef_xyz_max_step, 
                                                                                                                     self.eef_xyz_max_step)
                # is eef_rpy_max_step necessary here?
                self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = interpolate_rpy(self._last_eef_rpy_cmd, self._target_eef_rpy, tracking_ratio)
                # # using scipy
                # rot_interpolated = R.from_rotvec(self._target_eef_rotation_angle*tracking_ratio*self._target_eef_rotation_axis).as_matrix()
                # if self._eef_task_space_world:
                #     self._eef_pva_cmd[self._manipulate_eef_idx, 0:3] = rot_mat_to_rpy(rot_interpolated @ rpy_to_rot_mat(self._last_eef_rpy_cmd))

                self._gripper_angles_cmd[self._manipulate_eef_idx] += max(-self.gripper_angle_max_step, min(self._last_gripper_angle_cmd * (1 - tracking_ratio) + self._target_gripper_angle * tracking_ratio - self._gripper_angles_cmd[self._manipulate_eef_idx], self.gripper_angle_max_step))

            self._is_computing_command = False
            
            # print("eef actual:", self._robot.eef_pos_w_np[0, self._manipulate_eef_idx])
            # print("eef desired:", self._footeef_pva_cmd[0, 3 * self._manipulate_eef_idx:3 * (self._manipulate_eef_idx + 1)])
            
        # print("gripper angles:", self._gripper_angles_cmd)    
        
        return {"action_mode": self._action_mode,
                "contact_state": self._contact_state,
                "body_pva": self._body_pva_cmd,
                "footeef_pva": self._footeef_pva_cmd,
                "eef_pva": self._eef_pva_cmd,
                "gripper_angles": self._gripper_angles_cmd,
                }
    
    def check_finished(self):
        return True
    
    def unuse_commander(self):
        print("deactivate the eef commander")
        # print(len(self.cmd_history))
        super().unuse_commander()
        
    