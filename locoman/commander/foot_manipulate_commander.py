from einops import rearrange
from matplotlib import pyplot as plt
from commander.base_commander import BaseCommander
import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
from fsm.finite_state_machine import Manipulate_Mode
from utilities.orientation_utils_numpy import rpy_to_rot_mat, rot_mat_to_rpy
import time
import h5py
from scipy.spatial.transform import Rotation as R
import torch, torchvision

class FootManipulateCommander(BaseCommander):
    def __init__(self, robot, env_ids=0):
        super().__init__(robot, env_ids=env_ids)

        self._action_mode = 1
        self._manipulate_leg_idx = 0
        self.device = robot._device

        # self._body_pose_scale *= 0.2  # smaller scale for body movement during manipulation

        self._desired_footeef_pva_w[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)] = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx]
        self._footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._last_executed_footeef_pva_cmd = self._desired_footeef_pva_w.copy()
        self._footeef_p_buffer = np.zeros(3)
        self._footeef_p_scale = np.ones(3) * self._cfg.commander.footeef_p_scale

        self._manipute_eef_rpy_buffer = np.zeros(3)
        self._manipute_eef_rpy_scale = np.ones(3) * self._cfg.commander.eef_rpy_scale
        
        self.qpos_history = []
        self.cmd_history = []
        self.camera_history = []
        
        self.record_episode_counter = 0
        
        self.raw_buffered_cmd = np.zeros(20)
        
        self._cmd_from_human = None
        
        self.use_real_robot = False

        self.bc_model = None
        
        self.dataset_stats = None
        
        self.command_msg = Float32MultiArray()
        
        self.robot_reset_msg = Int32()
        self.robot_reset_msg.data = 1
        self.robot_reset_publisher = rospy.Publisher(self._cfg.teleoperation.robot_reset_topic, Int32, queue_size=1)
        
        self.diffusion_model = None
        
        self.all_actions = None
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.anti_normalize = torchvision.transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        self.human_command_body_rpy_range = np.array([self._cfg.teleoperation.human_teleoperator.body_r_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_p_range,
                                                 self._cfg.teleoperation.human_teleoperator.body_y_range,])
        
    @property
    def increment_counter(self):
        self.record_episode_counter += 1
        return self.record_episode_counter
    
    def reset(self):
        self._robot._update_state(reset_estimator=True)

        self._manipulate_leg_idx = 0 if self._robot._cur_manipulate_mode == Manipulate_Mode.RIGHT_FOOT else 1
        self._contact_state[:] = True
        self._contact_state[self._manipulate_leg_idx] = False

        self._body_pva_cmd[:] = self._desired_body_pva
        self._body_pv_buffer[:] = 0

        # self._desired_footeef_pva_w[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)] = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx]
        self._footeef_pva_cmd[:] = self._desired_footeef_pva_w
        self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)] = self._robot.foot_pos_w_np[self._env_ids, self._manipulate_leg_idx]
        self._last_executed_footeef_pva_cmd[:] = self._footeef_pva_cmd
        self._footeef_p_buffer[:] = 0

        self._eef_pva_cmd[:] = self._desired_eef_pva
        # self._eef_pva_cmd[self._manipulate_leg_idx, 0:3] = self._robot.eef_rpy_w_np[self._env_ids, self._manipulate_leg_idx]
        # self._last_executed_eef_pva_cmd[:] = self._eef_pva_cmd
        self._manipute_eef_rpy_buffer[:] = 0

        self._gripper_angles_cmd[:] = self._desired_gripper_angles
        self._gripper_angles_buffer[:] = 0

        self._is_updating_command = False
        self._is_computing_command = False

        self._human_time_at_first_stamp = time.time()
        self._human_time_from_first_stamp = 0.
        self._controller_time_at_first_stamp = time.time()
        self._controller_time_from_first_stamp = 0.
        self._reset_from_human = True
        self._first_receive_from_human = True
        
        self._cmd_from_human = None
        
        self.use_real_robot = False
        

    def _update_joystick_command_callback(self, command_msg):
        if self._is_used:
            # while self._is_computing_command:
            #     pass
            self._is_updating_command = True
            if self._cmd_from_human is None:
                self._cmd_from_human = 0
            self._time_for_tracking_human = 0.0
            command_np = np.array(command_msg.data)
            self._body_pv_buffer[0:6] = command_np[0:6] * self._body_pose_scale
            self._footeef_p_buffer[:] = command_np[6:9] * self._footeef_p_scale
            self._manipute_eef_rpy_buffer[:] = command_np[9:12] * self._manipute_eef_rpy_scale
            self._gripper_angles_buffer[:] = command_np[12:14] * self._gripper_angle_scale
            # self.cmd_history.append(command_np)
            self.raw_buffered_cmd = command_np
            self._is_updating_command = False

    def reset_last_command(self, command):
        self._last_body_xyz_cmd = command['body_pva'][0:3].copy()
        self._last_body_rpy_cmd = command['body_pva'][3:6].copy()
        self._last_eef_xyz_cmd = command['footeef_pva'][0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)].copy()
        self._last_eef_rpy_cmd = command['eef_pva'][self._manipulate_leg_idx, 0:3].copy()
        self._last_gripper_angle_cmd = command['gripper_angles'][self._manipulate_leg_idx] 

    def _update_human_command_callback(self, command_msg):
        if self._is_used:
            # while self._is_computing_command:
            #     pass
            self._is_updating_command = True
            if self._cmd_from_human is None:
                self._cmd_from_human = 1
            if self._reset_from_human:
                self._human_time_at_first_stamp = time.time()
                self._human_time_from_first_stamp = 0.
                self._controller_time_at_first_stamp = time.time()
                self._controller_time_from_first_stamp = 0.

                self._init_body_xyz = self._body_pva_cmd[0:3].copy()
                self._init_body_rpy = self._body_pva_cmd[3:6].copy()
                self._init_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)].copy()
                self._init_eef_rpy = self._eef_pva_cmd[self._manipulate_leg_idx, 0:3].copy()
                self._init_gripper_angle = self._gripper_angles_cmd[self._manipulate_leg_idx]

                self._last_body_xyz_cmd = self._body_pva_cmd[0:3].copy()
                self._last_body_rpy_cmd = self._body_pva_cmd[3:6].copy()
                self._last_eef_xyz_cmd = self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)].copy()
                self._last_eef_rpy_cmd = self._eef_pva_cmd[self._manipulate_leg_idx, 0:3].copy()
                self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_leg_idx]

                self._target_body_xyz = self._body_pva_cmd[0:3].copy()
                self._target_body_rpy = self._body_pva_cmd[3:6].copy()
                self._target_eef_xyz = self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)].copy()
                self._target_eef_rotation_axis = np.ones_like(self._init_eef_rpy)
                self._target_eef_ratation_angle = 0.
                self._target_gripper_angle = self._gripper_angles_cmd[self._manipulate_leg_idx]

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
                    self._last_eef_xyz_cmd[:] = self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)].copy()
                    self._last_eef_rpy_cmd[:] = self._eef_pva_cmd[self._manipulate_leg_idx, 0:3].copy()
                    self._last_gripper_angle_cmd = self._gripper_angles_cmd[self._manipulate_leg_idx]

                # do not allow torso movements on xyz during manipulation
                command_np[0:3] = 0.
                self._target_body_xyz[:] = command_np[0:3] * self.body_xyz_scale + self._init_body_xyz
                self._target_body_rpy[:] = command_np[3:6] * self.body_rpy_scale + self._init_body_rpy  
                self._target_body_rpy = np.clip(self._target_body_rpy, self.human_command_body_rpy_range[:, 0], self.human_command_body_rpy_range[:, 1])
                self._target_eef_xyz[:] = command_np[6+6*self._manipulate_leg_idx:9+6*self._manipulate_leg_idx] * self.eef_xyz_scale + self._init_eef_xyz
                delta_rotation = rpy_to_rot_mat(command_np[9+6*self._manipulate_leg_idx:12+6*self._manipulate_leg_idx]) @ rpy_to_rot_mat(self._init_eef_rpy) @ (rpy_to_rot_mat(self._last_eef_rpy_cmd).T)
                r = R.from_matrix(delta_rotation)
                rotvec = r.as_rotvec()
                if np.linalg.norm(rotvec) > 1e-6:
                    self._target_eef_rotation_axis[:] = rotvec / np.linalg.norm(rotvec)
                    self._target_eef_ratation_angle = np.linalg.norm(rotvec) * self.eef_rpy_scale
                else:
                    self._target_eef_rotation_axis[:] = np.ones(3)
                    self._target_eef_ratation_angle = 0.

                # print('target_eef_rotation_axis: ', self._target_eef_rotation_axis)
                # print('target_eef_ratation_angle: ', self._target_eef_ratation_angle)

                self._target_gripper_angle = command_np[18+self._manipulate_leg_idx] * self.gripper_angle_scale + self._init_gripper_angle

                self.raw_buffered_cmd = command_np
                
            self._is_updating_command = False

            # print('\n-------------------')
            # print("_last_eef_rpy_cmd: ", self._last_eef_rpy_cmd)
            # print("_target_eef_delta_rpy: ", self._target_eef_delta_rpy)
        
    # for human teleoperation
    def compute_command_for_wbc(self):
        # super().compute_command_for_wbc()
        # if self._command_executed:
        #     self._last_executed_footeef_pva_cmd[:] = self._footeef_pva_cmd
        # else:
        #     self._footeef_pva_cmd[:] = self._last_executed_footeef_pva_cmd
        
        # print('fps: ', 1/(time.time() - self._begin_time))
        # self._begin_time = time.time()

        if not self._is_updating_command:
            self._is_computing_command = True
            if self._time_for_tracking_human == 0.0:
                self._body_pva_cmd[0:6] += self._body_pv_buffer[0:6]
                self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)] += self._footeef_p_buffer
                if self._eef_task_space_world:
                    self._eef_pva_cmd[self._manipulate_leg_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._manipute_eef_rpy_buffer) @ rpy_to_rot_mat(self._eef_pva_cmd[self._manipulate_leg_idx, 0:3]))
                else:
                    self._eef_pva_cmd[self._manipulate_leg_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(self._eef_pva_cmd[self._manipulate_leg_idx, 0:3]) @ rpy_to_rot_mat(self._manipute_eef_rpy_buffer))
                self._gripper_angles_cmd[:] += self._gripper_angles_buffer
                self._gripper_angles_cmd[:] = np.clip(self._gripper_angles_cmd, self._gripper_angle_range[0], self._gripper_angle_range[1])

                self._body_pv_buffer[:] = 0
                self._footeef_p_buffer[:] = 0
                self._manipute_eef_rpy_buffer[:] = 0
                self._gripper_angles_buffer[:] = 0
            elif not self._first_receive_from_human:
                tracking_ratio = max(0.0, min(1.0, (time.time() - self._controller_time_at_first_stamp - self._controller_time_from_first_stamp) / self._time_for_tracking_human))

                self._body_pva_cmd[0:3] += np.clip(self._last_body_xyz_cmd * (1 - tracking_ratio) + self._target_body_xyz * tracking_ratio - self._body_pva_cmd[0:3], -self.body_xyz_max_step, self.body_xyz_max_step)
                self._body_pva_cmd[3:6] += np.clip(self._last_body_rpy_cmd * (1 - tracking_ratio) + self._target_body_rpy * tracking_ratio - self._body_pva_cmd[3:6], -self.body_rpy_max_step, self.body_rpy_max_step)
                # do not allow torso movements on xyz during manipulation
                self._body_pva_cmd[0:3] = 0.
                self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)] += np.clip(self._last_eef_xyz_cmd * (1 - tracking_ratio) + self._target_eef_xyz * tracking_ratio - self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)], -self.eef_xyz_max_step, self.eef_xyz_max_step)
                
                # q_interpolated = tf.quaternion_about_axis(self._target_eef_ratation_angle*tracking_ratio, self._target_eef_rotation_axis)
                # rpy_interpolated = tf.euler_from_quaternion(q_interpolated)
                # if self._eef_task_space_world:
                #     self._eef_pva_cmd[self._manipulate_leg_idx, 0:3] = rot_mat_to_rpy(rpy_to_rot_mat(rpy_interpolated) @ rpy_to_rot_mat(self._last_eef_rpy_cmd))

                # using scipy
                rot_interpolated = R.from_rotvec(self._target_eef_ratation_angle*tracking_ratio*self._target_eef_rotation_axis).as_matrix()
                if self._eef_task_space_world:
                    self._eef_pva_cmd[self._manipulate_leg_idx, 0:3] = rot_mat_to_rpy(rot_interpolated @ rpy_to_rot_mat(self._last_eef_rpy_cmd))

                self._gripper_angles_cmd[self._manipulate_leg_idx] += max(-self.gripper_angle_max_step, min(self._last_gripper_angle_cmd * (1 - tracking_ratio) + self._target_gripper_angle * tracking_ratio - self._gripper_angles_cmd[self._manipulate_leg_idx], self.gripper_angle_max_step))

            self._is_computing_command = False
            
            print("eef desired:", self._footeef_pva_cmd[0, 3 * self._manipulate_leg_idx:3 * (self._manipulate_leg_idx + 1)])
            
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
        print("unusing eef!")
        print(len(self.cmd_history))
        super().unuse_commander()
        # flush obs/cmd history to file using h5py and re-init
        self.flush_trajectory()
